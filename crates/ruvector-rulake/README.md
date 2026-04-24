# ruLake — A Cache-Coherent Vector Execution Layer

> *A vector execution cache with deterministic compression and federated refill.*

`ruLake` is the RVF ecosystem's answer to "how do agents and apps query
1M+ vectors across heterogeneous backends without reimplementing
caching, coherence, governance, and the RaBitQ compressor every time?"

It's not a vector database. It's not a lakehouse adapter. It's the
substrate:

- A **RaBitQ 1-bit cache** that runs at ~1.02× the cost of raw
  [`ruvector-rabitq`](../ruvector-rabitq/) — the abstraction is
  effectively free.
- **Federated refill** across pluggable `BackendAdapter` implementations
  (Parquet, BigQuery, Iceberg, Delta, local). Federation is the *miss
  path*, not the product shape.
- **Witness-addressed** bundles (SHAKE-256 over the data reference)
  that let two processes, clouds, or customers share one compressed
  entry when they point at the same underlying bytes.
- **Three-mode consistency knob** — `Fresh` (compliance), `Eventual`
  (recall), `Frozen` (audit).
- **33,094 QPS** under 8 concurrent clients at n=100 k, D=128, rerank×20
  — 11.9× the serialized-mutex baseline.

```text
                ┌───────────────────────────────────┐
  caller ──▶    │           RuLake                   │   ──▶  SearchResult
                │   ┌──────────────────────────┐   │
                │   │  RaBitQ cache (Arc'd)    │   │
                │   │   witness → index        │   │
                │   └──────────────────────────┘   │
                │          ▲  miss                  │
                │   ┌──────┴──────┐                 │
                │   │ BackendAdapter │  ──▶  Parquet / BQ / Iceberg / RVF
                │   └───────────────┘                │
                └───────────────────────────────────┘
```

---

## Features

### 🗂 Cache-first vector execution

The hot path is one `Arc<RabitqPlusIndex>::search` lookup. Measured
intermediary tax: **1.00–1.02×** vs direct RaBitQ. You can afford the
abstraction — orchestration, governance, routing — because the cache
does the work.

### 🔐 Witness-authenticated bundles

Every cache entry is keyed on a `table.rulake.json` sidecar:

```json
{
  "format_version": 1,
  "data_ref": "gs://bucket/table.parquet",
  "dim": 768,
  "rotation_seed": 42,
  "rerank_factor": 20,
  "generation": 1729843200,
  "rvf_witness": "b3ac...0f7c",
  "pii_policy": "pii://policies/default",
  "lineage_id": "ol://jobs/ingest-42",
  "memory_class": "episodic"
}
```

Two processes anywhere — different clouds, different services, same
underlying bytes — produce the same witness and **share one compressed
cache entry**. No coordination required.

### 🔁 Federated search (rayon parallel fan-out)

Register N backends, search them all in parallel with one call:

```rust
let hits = lake.search_federated(
    &[("bigquery", "events"), ("snowflake", "profiles"), ("s3", "archive")],
    &query_vector,
    10,
)?;
```

Adaptive per-shard rerank keeps K-shard federation from paying K×
rerank cost. **4-shard concurrent throughput: 33,094 QPS** vs
single-shard 23,681.

### 🧩 Three-mode consistency

| Mode     | Use case                                          | Overhead            |
|----------|---------------------------------------------------|---------------------|
| `Fresh`  | compliance, finance, policy-enforced workloads    | 1 backend RTT/query |
| `Eventual{ttl_ms}` | search, AI retrieval, recommendation, RAG | 1 RTT per TTL       |
| `Frozen` | audit-tier historical snapshots, content-addressed | 0 backend RTTs      |

Set it per-`RuLake` with `with_consistency(...)`. The product knob
lets one deployment serve compliance collections and recall
collections from the same cache.

### 📦 Sidecar publish/refresh protocol

Symmetric writer + reader primitives for cross-process cache
coherence:

```rust
// Publisher side: warehouse job emits the bundle.
lake.publish_bundle(&("bq", "events"), "/mnt/gcs/events/")?;

// Reader side: daemon watches for updates.
match lake.refresh_from_bundle_dir(&("bq", "events"), "/mnt/gcs/events/")? {
    RefreshResult::UpToDate    => {}             // no-op
    RefreshResult::Invalidated => notify_ops(),  // cache was rotated
    RefreshResult::BundleMissing => warn!(),
}
```

Atomic temp+rename on write; witness verification on read; tampered
sidecars surface as `InvalidParameter`, not silent corruption.

### 📊 Cache-first KPIs

Operators get the numbers that matter for cache-first operation out
of the box:

```rust
let stats = lake.cache_stats();
println!("hit_rate = {:.3}",    stats.hit_rate().unwrap_or(0.0));
println!("avg_prime_ms = {:?}", stats.avg_prime_ms());

// Per-backend or per-collection attribution:
for (backend, s) in lake.cache_stats_by_backend() {
    println!("{backend}: hit_rate={:.3}", s.hit_rate().unwrap_or(0.0));
}
```

Acceptance gate for the cache-first reframe: **`hit_rate ≥ 0.95`**
on a realistic workload, measurable from the stats stream alone.

### 🔌 Pluggable VectorKernel (ADR-157)

Optional accelerator plane: a `VectorKernel` trait in
`ruvector-rabitq` lets GPU / SIMD / WASM kernels plug in alongside the
default CPU implementation. Determinism is gated by
`caps().deterministic` — non-deterministic kernels (GPU float reorder)
are filtered out of `Fresh` / `Frozen` paths automatically.

Kernels ship as separate crates (`ruvector-rabitq-cuda`, -metal, …) on
their own cadence. Laptop / WASM builds pay zero dep cost.

### 🛡 Security hardening

- **Path-traversal safe:** `FsBackend` rejects filenames with `..`,
  separators, control bytes, drive letters. 12-form attack surface
  covered.
- **JSON deserialization caps:** bundle input ≤ 64 KiB, individual
  fields ≤ 4 KiB. A compromised GCS object cannot DoS the reader.
- **Witness chain:** domain-separated + length-prefixed SHAKE-256.
  Tampered bundles fail `verify_witness()` and `read_from_dir()`.
- **Zero `unsafe`** in the ruLake crate or the new kernel module.

---

## Quick start

```toml
# Cargo.toml
[dependencies]
ruvector-rulake = "2.2"
```

```rust
use std::sync::Arc;
use ruvector_rulake::{cache::Consistency, LocalBackend, RuLake};

// 1. Spin up a backend with some vectors.
let backend = Arc::new(LocalBackend::new("my-backend"));
backend.put_collection(
    "memories",
    /* dim */ 128,
    /* ids */  vec![1, 2, 3],
    /* vecs */ vec![vec![0.0; 128]; 3],
)?;

// 2. Register it with ruLake.
let lake = RuLake::new(
    /* rerank_factor */ 20,
    /* rotation_seed */ 42,
).with_consistency(Consistency::Eventual { ttl_ms: 60_000 });
lake.register_backend(backend)?;

// 3. Query. First call primes the cache; subsequent calls serve from
//    the RaBitQ-compressed entry at ~1.02× direct-search cost.
let hits = lake.search_one("my-backend", "memories", &vec![0.0; 128], 10)?;

// 4. Observability.
println!("hit rate: {:.3}", lake.cache_stats().hit_rate().unwrap_or(0.0));
```

For a full publish/refresh loop see
[`examples/sidecar_daemon.rs`](examples/sidecar_daemon.rs).

---

## Usage patterns

### Cache-first RAG / retrieval

```rust
let lake = RuLake::new(20, 42)
    .with_consistency(Consistency::Eventual { ttl_ms: 60_000 })
    .with_max_cache_entries(100);  // LRU bound for serving processes
lake.register_backend(parquet_backend)?;

// Batch API amortizes ensure_fresh + lock acquisition across N queries.
let hits = lake.search_batch("parquet", "corpus", &query_batch, 10)?;
```

### Federated search across clouds

```rust
let hits = lake.search_federated(
    &[
        ("bigquery",  "events"),
        ("snowflake", "profiles"),
        ("iceberg",   "archive"),
    ],
    &query,
    10,
)?;
// Adaptive per-shard rerank = max(5, 20 / 3) = 6 per shard.
// Global top-10 merged across all three.
```

### Witness-sealed audit tier

```rust
let audit = RuLake::new(20, 42).with_consistency(Consistency::Frozen);
audit.register_backend(content_addressed_backend)?;

// First query primes; all subsequent reads stay on the frozen witness
// even if the backend "bumps" — caller asserts the data is immutable.
let hits = audit.search_one("ca", "snapshot-2026-q2", &q, 10)?;
```

### Cache sidecar daemon

```rust
// Reader process runs this loop alongside the serving process.
loop {
    match lake.refresh_from_bundle_dir(&key, publish_dir)? {
        RefreshResult::Invalidated => {
            metrics.bundle_rotations.inc();
        }
        _ => {}
    }
    std::thread::sleep(Duration::from_secs(5));
}
```

See [`examples/sidecar_daemon.rs`](examples/sidecar_daemon.rs) for a
runnable end-to-end demo.

### Memory substrate for agent brain systems (ADR-156)

```rust
let bundle = RuLakeBundle::new("mem://episodic/2026-04-23", 768, 42, 20, gen.into())
    .with_memory_class("episodic")
    .with_pii_policy("pii://policies/redact-pii")
    .with_lineage_id("ol://jobs/episodic-consolidation");
bundle.write_to_dir("/mnt/brain/bundles/")?;
```

`memory_class` is opaque to ruLake — the brain system owns semantics,
the substrate owns persistence.

---

## Benchmarks

All numbers from a single reproducible run of:

```bash
cargo run --release -p ruvector-rulake --bin rulake-demo
```

on a commodity Ryzen-class laptop, deterministic seeds.

### Intermediary tax (cache-hit path)

Clustered Gaussian, D=128, 100 clusters, rerank×20, 300 warm queries.

| n       | direct RaBitQ+ | ruLake Fresh | ruLake Eventual | tax     |
|--------:|---------------:|-------------:|----------------:|--------:|
|   5 000 |        17,567  |      17,431  |         17,567  | 1.01×   |
|  50 000 |         4,985  |       4,932  |          4,959  | 1.01×   |
| 100 000 |         2,975  |       3,020  |          2,963  | 1.00×   |

**The abstraction layer is not the bottleneck.** You can afford to
put governance, routing, and orchestration on top.

### Concurrent clients × shard count

n=100 k, 8 clients × 300 queries each, adaptive per-shard rerank.

| shards | wall (ms) |       QPS | vs pre-Arc 1-shard |
|-------:|----------:|----------:|-------------------:|
|      1 |     101.3 |    23,681 |              8.3×  |
|      2 |      82.8 |    28,971 |             10.1×  |
|      4 |      72.5 |    33,094 |             11.6×  |

The `Arc<RabitqPlusIndex>` refactor (drop cache lock before scan) is
the single biggest optimization on the M1 branch.

### Federated cold-path (miss) prime time

Parallel fan-out: a single federated query that misses every shard
primes them concurrently.

| n       | 1-shard prime | 2-shard prime | 4-shard prime | 2/4 speedup |
|--------:|--------------:|--------------:|--------------:|------------:|
|   5 000 |      22.3 ms  |      12.7 ms  |       6.6 ms  | 1.76× / 3.38× |
|  50 000 |     213.3 ms  |     109.5 ms  |      55.7 ms  | 1.95× / 3.83× |
| 100 000 |     424.8 ms  |     215.3 ms  |     110.1 ms  | 1.97× / 3.86× |

### Recall

`rulake_recall_at_10_above_90pct_vs_brute_force` gate test: recall@10
> 90% on clustered D=128 n=5 k at rerank×20.

`adaptive_per_shard_rerank_preserves_recall` gate test: recall@10 ≥
85% at K=2 and K=4 under adaptive per-shard rerank.

See [`BENCHMARK.md`](BENCHMARK.md) for full methodology and all
measurement runs.

---

## Comparison

| System           | Intermediary tax | Cross-cloud federation | Witness-authenticated | Cache sharing across processes | Deployable without GPU | `unsafe` count |
|------------------|-----------------:|-----------------------:|----------------------:|-------------------------------:|-----------------------:|---------------:|
| **ruLake**       | **1.02×**        | ✅ (rayon fan-out)     | ✅ (SHAKE-256)         | ✅ (content-addressed)         | ✅                     | **0**          |
| Pinecone         | n/a (hosted)     | ❌ single-region        | ❌                     | ❌                             | n/a                    | n/a            |
| Weaviate         | n/a (hosted)     | ❌                      | ❌                     | ❌                             | ✅                     | n/a            |
| Milvus           | ~1.5–2×          | partial                | ❌                     | ❌                             | ✅                     | many           |
| LanceDB          | ~1.1–1.3×        | ❌                      | ❌                     | ❌                             | ✅                     | some           |
| BQ Vector Search | n/a (hosted)     | ❌ (BQ-only)            | ❌                     | ❌                             | n/a                    | n/a            |

`ruLake` is explicitly **not** a vector database — it doesn't own
storage. It's the substrate that lets you query whichever vector DB
or lakehouse you already have, with a coherent compression +
governance story across all of them. If you want a standalone managed
vector DB, use Pinecone or Weaviate. If you want to use the vectors
that already live in your lake, use `ruLake`.

---

## Technical details

### Architecture

```text
┌────────────────── RuLake ────────────────────┐
│                                              │
│  Consistency knob: Fresh | Eventual | Frozen │
│                                              │
│  ┌──────── VectorCache (Arc<Mutex>) ────┐   │
│  │                                       │   │
│  │   entries: witness → Arc<RabitqIdx>  │   │
│  │   pointers: (backend, coll) → witness│   │
│  │   per_backend / per_collection stats │   │
│  │                                       │   │
│  └───────────────────────────────────────┘   │
│                  ▲                            │
│                  │ prime (on miss)            │
│  ┌───────────── BackendAdapter trait ─────┐  │
│  │                                         │  │
│  │  id() list_collections() pull_vectors() │  │
│  │  generation() current_bundle()          │  │
│  │                                         │  │
│  └─ LocalBackend ─ FsBackend ─ ParquetBackend (M2) ─ ... ─┘  │
└──────────────────────────────────────────────┘
```

### Bundle protocol

`table.rulake.json` is the portable unit. The witness is:

```
SHAKE-256(32)(
  "rulake-bundle-witness-v1|" ||
  len(data_ref) || data_ref ||
  "|" || dim || rotation_seed || rerank_factor ||
  "|" || len(generation) || generation
)
```

Length-prefixed + domain-separated so no field can collide with
another bundle's witness via concatenation games.

### Cache coherence

```text
search(backend, collection, query, k)
  │
  ▼
ensure_fresh(key) ─────────────── Consistency mode?
  │                                     │
  │    ┌────────────────────┬───────────┼─────────────┐
  │   Frozen               Eventual                  Fresh
  │   (skip check          (skip if within TTL)       (always check)
  │   after prime)                                    │
  │                                                   ▼
  │                            ask backend for current witness
  │                                     │
  │                 ┌───────────────────┼──────────────────┐
  │              match              mismatch           mismatch
  │             (hit)            & witness cached    & new witness
  │                              elsewhere           (share-cache)
  │                              (pointer move,
  │                              no prime)
  │                              │                       │
  │                              ▼                       ▼
  │                         just move pointer     pull + prime
  │                                                       │
  ▼                                                       ▼
Arc<RabitqPlusIndex>::search (no lock held) ◀────────────┘
```

### Adaptive per-shard rerank

Under federation, the RaBitQ `rerank_factor × k` rerank would run
once per shard — K× the work. ruLake's default divides the budget:

```rust
per_shard_rerank = max(MIN_PER_SHARD_RERANK, global_rerank / K)
// MIN_PER_SHARD_RERANK = 5 (floor below which rerank is meaningless)
```

Measured recall@10 stays above 85% at K=4. Callers who need byte-exact
single-shard parity use `search_federated_with_rerank(..., Some(global))`.

### Arc-based concurrency

`CacheEntry::index: Arc<RabitqPlusIndex>` is the key. Reader threads:

1. Lock the cache mutex
2. Clone the Arc (cheap — refcount bump)
3. Drop the lock
4. Scan without holding anything shared

The index is immutable after build, so concurrent scans are a pure
data race against nothing. 8-11× QPS improvement under concurrent
load vs the original `Mutex<CacheState>` scheme.

### Security model

- **No `unsafe`** in ruLake or rabitq kernel — every data-mutating
  path goes through checked borrows.
- **Path-traversal**: `FsBackend::register` + `write` validate
  filenames (no `..`, no separators, no control bytes, ≤ 255 bytes).
- **JSON size caps**: bundle ≤ 64 KiB, fields ≤ 4 KiB. A malicious
  sidecar cannot DoS the reader.
- **Witness verification**: every `read_from_dir` verifies the
  SHAKE-256 chain; tampered bundles error out as `InvalidParameter`.
- **Atomic writes**: `write_to_dir` uses `tmp + rename` so concurrent
  readers never observe a torn sidecar.
- **Mutex poisoning** on `.lock().unwrap()` is a deliberate
  fail-fast — a poisoned mutex means the invariants are compromised.

See ADR-155, ADR-156, ADR-157 for the full threat model + design
trade-offs.

---

## User guide

### Choosing a consistency mode

- **`Fresh`** — every search calls `generation()` on the backend.
  Appropriate for policy-enforced workloads where stale reads are a
  compliance failure. ~1 network RTT per query overhead on real
  backends.
- **`Eventual { ttl_ms }`** — cache the coherence decision for
  `ttl_ms`. Appropriate for search, RAG, recommendation. 60-second
  TTL is a good default.
- **`Frozen`** — never re-check the backend. Appropriate for
  content-addressed historical snapshots where the bundle is
  cryptographically pinned. Operators can still force-refresh via
  `refresh_from_bundle_dir`.

### Sizing the cache

```rust
// Unbounded cache (M1 default) — fine for small collections:
let lake = RuLake::new(20, 42);

// LRU-capped cache for serving processes with memory bounds:
let lake = RuLake::new(20, 42).with_max_cache_entries(100);
```

Only unpinned entries (refcount == 0, no live pointer) are evicted.
Active `(backend, collection)` pointers keep their entry alive.

### Operational observability

| Metric (from `cache_stats()`) | What to do when it moves |
|-------------------------------|--------------------------|
| `hit_rate`                    | Below 0.95 → increase cache size or warm more aggressively |
| `last_prime_ms`               | Spiking → backend RTT changed or data grew |
| `primes`                      | Growing unexpectedly → check for witness churn |
| `shared_hits`                 | Non-zero → cross-backend cache sharing is working |
| `invalidations`               | Steady-state → coherence protocol is firing |

Per-backend (`cache_stats_by_backend()`) and per-collection
(`cache_stats_by_collection()`) views let you identify which
specific data is hot.

### Writing a custom backend

Implement the `BackendAdapter` trait:

```rust
use ruvector_rulake::backend::{BackendAdapter, CollectionId, PulledBatch};

struct MyBackend { /* ... */ }

impl BackendAdapter for MyBackend {
    fn id(&self) -> &str { "my-backend" }

    fn list_collections(&self) -> Result<Vec<CollectionId>> { /* ... */ }

    fn pull_vectors(&self, collection: &str) -> Result<PulledBatch> {
        // Stream vectors from wherever — Parquet, HTTP, BigQuery, ...
    }

    fn generation(&self, collection: &str) -> Result<u64> {
        // Return a coherence token (mtime, snapshot id, version, ...).
    }

    // Optional: override current_bundle to use the canonical data_ref
    // for cross-backend cache sharing.
    fn current_bundle(&self, collection: &str, rotation_seed: u64, rerank_factor: usize)
        -> Result<RuLakeBundle> { /* ... */ }
}
```

See `FsBackend` ([`src/fs_backend.rs`](src/fs_backend.rs)) for a
~250-line reference implementation.

### Running the examples

```bash
# End-to-end sidecar daemon: publisher + reader + coherence loop
cargo run --release -p ruvector-rulake --example sidecar_daemon

# Benchmark harness (3 minutes on a laptop)
cargo run --release -p ruvector-rulake --bin rulake-demo

# Fast mode: just n=5k
cargo run --release -p ruvector-rulake --bin rulake-demo -- --fast
```

---

## Status

**M1: shipped and measured** (2026-04-23)

- Core abstraction (BackendAdapter, VectorCache, bundle protocol,
  Consistency modes, LRU eviction)
- Two backends: `LocalBackend` (reference), `FsBackend` (file-based)
- Optimization: adaptive per-shard rerank, Arc-based concurrency
  (11.9× concurrent QPS win)
- Observability: hit_rate, prime durations, per-backend, per-collection
- Substrate acceptance test (recall → verify → forget → rehydrate)
- Security hardening (path traversal + JSON caps)
- `VectorKernel` trait scaffolding in `ruvector-rabitq`
- 40 tests, clippy `-D warnings` clean, zero `unsafe`

**M2+ roadmap**: `ParquetBackend`, `BigQueryBackend`, HTTP wire layer,
governance MVP, `DeltaBackend` / `IcebergBackend`, GPU kernels in
separate crates.

See [`docs/adr/ADR-155`](../../docs/adr/ADR-155-rulake-datalake-layer.md)
(cache-first fabric), [`ADR-156`](../../docs/adr/ADR-156-rulake-as-memory-substrate.md)
(memory substrate for agent brains),
[`ADR-157`](../../docs/adr/ADR-157-optional-accelerator-plane.md)
(accelerator plane) for the full design record.

---

## License

Apache-2.0 OR MIT (workspace default)
