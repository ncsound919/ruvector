//! The public `RuLake` entry point — registers backends, routes searches.
//!
//! v1 fans out across registered backends, runs RaBitQ-cache searches,
//! and merges the results by score. v2 will push-down to backends that
//! support native vector ops.

use std::collections::HashMap;
use std::sync::Arc;

use crate::backend::{BackendAdapter, BackendId};
use crate::cache::{CacheKey, Consistency, VectorCache};
use crate::error::{Result, RuLakeError};

/// Result from a search — the external id and its estimated L2² score.
/// Includes the backend that produced the hit so callers can audit.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub backend: BackendId,
    pub collection: String,
    pub id: u64,
    pub score: f32,
}

/// Outcome of [`RuLake::refresh_from_bundle_dir`]. A cache sidecar
/// daemon decides what to log / alert on based on which variant it
/// sees — `Invalidated` is the normal "bundle just rotated" signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefreshResult {
    /// The cache pointer's witness matches the on-disk bundle.
    UpToDate,
    /// Witnesses differed; the cache entry for this key was dropped.
    Invalidated,
    /// No sidecar was found at the target directory.
    BundleMissing,
}

/// ruLake entry point. Cheap to clone (everything is behind `Arc`).
#[derive(Clone)]
pub struct RuLake {
    backends: Arc<std::sync::RwLock<HashMap<BackendId, Arc<dyn BackendAdapter>>>>,
    cache: Arc<VectorCache>,
    consistency: Consistency,
}

impl RuLake {
    /// Build a fresh ruLake. `rerank_factor` controls the RaBitQ cache
    /// precision (20 → 100% recall@10 on clustered D=128 at n ≥ 50k per
    /// `ruvector-rabitq::BENCHMARK.md`). `rotation_seed` is shared across
    /// all cached collections so the compression is deterministic
    /// (important for the reproducibility + witness story).
    pub fn new(rerank_factor: usize, rotation_seed: u64) -> Self {
        Self {
            backends: Arc::new(std::sync::RwLock::new(HashMap::new())),
            cache: Arc::new(VectorCache::new(rerank_factor, rotation_seed)),
            consistency: Consistency::default(),
        }
    }

    /// Set the cache consistency mode. Defaults to `Consistency::Fresh`.
    pub fn with_consistency(mut self, c: Consistency) -> Self {
        self.consistency = c;
        self
    }

    /// Cap the cache at `n` distinct compressed entries (LRU eviction
    /// over *unpinned* entries). Unbounded by default. Useful in
    /// serving processes where memory is finite.
    pub fn with_max_cache_entries(self, n: usize) -> Self {
        Self {
            cache: Arc::new(
                VectorCache::new(self.cache.rerank_factor(), self.cache.rotation_seed())
                    .with_max_entries(n),
            ),
            backends: self.backends,
            consistency: self.consistency,
        }
    }

    /// Register a backend under its `id()`. Returns an error if a backend
    /// with the same id already exists.
    pub fn register_backend(&self, backend: Arc<dyn BackendAdapter>) -> Result<()> {
        let mut map = self.backends.write().unwrap();
        let id = backend.id().to_string();
        if map.contains_key(&id) {
            return Err(RuLakeError::InvalidParameter(format!(
                "backend {id} already registered"
            )));
        }
        map.insert(id, backend);
        Ok(())
    }

    pub fn backend_ids(&self) -> Vec<BackendId> {
        self.backends.read().unwrap().keys().cloned().collect()
    }

    /// Access the cache stats for diagnostics / benchmarking.
    pub fn cache_stats(&self) -> crate::CacheStats {
        self.cache.stats()
    }

    /// What witness resolves from a `(backend, collection)` pair?
    /// Useful for diagnostics and cross-backend cache-sharing tests.
    pub fn cache_witness_of(&self, key: &CacheKey) -> Option<String> {
        self.cache.witness_of(key)
    }

    /// How many distinct compressed-index entries live in the cache?
    /// Smaller than the pointer count when multiple backends share a
    /// witness.
    pub fn cache_entry_count(&self) -> usize {
        self.cache.entry_count()
    }

    /// How many external pointers currently resolve to this witness?
    /// Returns 0 for unknown witnesses.
    pub fn cache_refcount_of(&self, witness: &str) -> u32 {
        self.cache.refcount_of(witness)
    }

    /// Drop the cache pointer for a given `(backend, collection)` pair.
    /// If this was the last pointer at the witness, the underlying
    /// compressed entry is garbage-collected.
    pub fn invalidate_cache(&self, key: &CacheKey) {
        self.cache.invalidate(key);
    }

    /// Publish the current bundle for `(backend, collection)` to `dir`
    /// as `table.rulake.json`. Pairs with `RuLakeBundle::read_from_dir`:
    /// a cache sidecar daemon can watch the target directory and hand
    /// the updated bundle to another ruLake instance, enabling
    /// warehouse-driven cache refreshes without the reader having to
    /// poll the backend directly.
    ///
    /// Returns the path of the written sidecar so callers can log /
    /// audit the publish step.
    pub fn publish_bundle(
        &self,
        key: &CacheKey,
        dir: impl AsRef<std::path::Path>,
    ) -> Result<std::path::PathBuf> {
        let backend = self.get_backend(&key.0)?;
        let bundle = backend.current_bundle(
            &key.1,
            self.cache.rotation_seed(),
            self.cache.rerank_factor(),
        )?;
        bundle.write_to_dir(dir)
    }

    /// Refresh the cache for `key` against a published bundle sidecar in
    /// `dir`. This is the reader side of the sidecar protocol: a cache
    /// daemon watches a publish directory (e.g. a GCS prefix mounted
    /// locally), and when a fresh `table.rulake.json` lands, calls this
    /// to invalidate stale entries so the next search primes against
    /// the new generation.
    ///
    /// Returns:
    /// - `RefreshResult::UpToDate` when the on-disk bundle's witness
    ///   matches what the cache currently points at — nothing to do.
    /// - `RefreshResult::Invalidated` when the witnesses differ; the
    ///   cache pointer for `key` has been dropped and the next search
    ///   will re-prime.
    /// - `RefreshResult::BundleMissing` when the sidecar isn't there —
    ///   caller decides whether that's expected (not yet published) or
    ///   an error.
    ///
    /// Rejects a tampered sidecar with `InvalidParameter` so a corrupt
    /// publish doesn't silently poison the cache.
    pub fn refresh_from_bundle_dir(
        &self,
        key: &CacheKey,
        dir: impl AsRef<std::path::Path>,
    ) -> Result<RefreshResult> {
        let dir = dir.as_ref();
        let path = dir.join(crate::RuLakeBundle::SIDECAR_FILENAME);
        if !path.exists() {
            return Ok(RefreshResult::BundleMissing);
        }
        let bundle = crate::RuLakeBundle::read_from_dir(dir)?;
        let current = self.cache.witness_of(key);
        if current.as_deref() == Some(bundle.rvf_witness.as_str()) {
            Ok(RefreshResult::UpToDate)
        } else {
            self.cache.invalidate(key);
            Ok(RefreshResult::Invalidated)
        }
    }

    /// Search a single (backend, collection) pair. Handles cache
    /// miss / staleness transparently.
    pub fn search_one(
        &self,
        backend: &str,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let key: CacheKey = (backend.to_string(), collection.to_string());
        self.ensure_fresh(&key)?;
        let hits = self.cache.search_cached(&key, query, k)?;
        Ok(hits
            .into_iter()
            .map(|(id, score)| SearchResult {
                backend: backend.to_string(),
                collection: collection.to_string(),
                id,
                score,
            })
            .collect())
    }

    /// Federated search: fan out to every `(backend, collection)` pair
    /// in `targets` in parallel, merge by score, return global top-k.
    ///
    /// Each shard's `search_one` runs on the shared rayon pool. The
    /// cache is behind a Mutex so the hot path (witness-match → pointer
    /// touch + search) does not hold it across the rabitq scan; only
    /// cache-miss prime paths briefly serialize. For a 2-shard all-hit
    /// workload this gets us close to the single-shard latency; miss
    /// paths serialize on prime but that's the rare case.
    pub fn search_federated(
        &self,
        targets: &[(&str, &str)],
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        use rayon::prelude::*;
        // Collect so we can return the first error; rayon's `collect`
        // into `Result<Vec<_>>` short-circuits on the first Err which is
        // exactly the semantics the sequential loop had.
        let shard_hits: Result<Vec<Vec<SearchResult>>> = targets
            .par_iter()
            .map(|(backend, collection)| self.search_one(backend, collection, query, k))
            .collect();
        let mut merged: Vec<SearchResult> = shard_hits?.into_iter().flatten().collect();
        // Ascending by score (L2²) — smaller = closer.
        merged.sort_by(|a, b| a.score.total_cmp(&b.score));
        merged.truncate(k);
        Ok(merged)
    }

    /// Coherence check: ask the backend for its current bundle and
    /// compare its witness with whatever the cache currently points at.
    ///
    /// Four cases:
    ///
    /// 1. Fast path, Eventual within TTL → skip check entirely.
    /// 2. Witness matches the cache pointer → hit, nothing to do.
    /// 3. Witness mismatch, but target witness is already in the
    ///    entry pool (cached under another pointer) → just move the
    ///    pointer, zero prime work. This is the cross-backend share.
    /// 4. Witness not in the pool → pull + prime.
    fn ensure_fresh(&self, key: &CacheKey) -> Result<()> {
        if self.cache.can_skip_check(key, self.consistency) {
            self.cache.mark_hit();
            return Ok(());
        }

        let backend = self.get_backend(&key.0)?;
        let bundle = backend.current_bundle(
            &key.1,
            self.cache.rotation_seed(),
            self.cache.rerank_factor(),
        )?;
        let target_witness = bundle.rvf_witness.clone();

        if self.cache.witness_of(key).as_deref() == Some(target_witness.as_str()) {
            // Case 2: pointer up-to-date.
            self.cache.mark_hit();
            self.cache.touch(key);
            return Ok(());
        }

        // Cases 3 + 4 are handled in `prime`: it reuses an existing
        // entry for the target witness if present, or builds a new one.
        self.cache.mark_miss();
        let batch = backend.pull_vectors(&key.1)?;
        self.cache.prime(key.clone(), target_witness, batch)?;
        Ok(())
    }

    fn get_backend(&self, id: &str) -> Result<Arc<dyn BackendAdapter>> {
        self.backends
            .read()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| RuLakeError::UnknownBackend(id.to_string()))
    }
}
