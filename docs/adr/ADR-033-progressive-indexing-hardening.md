# ADR-033: Progressive Indexing Hardening — Centroid Stability, Adversarial Resilience, Recall Framing, and Mandatory Signatures

**Status**: Accepted
**Date**: 2026-02-15
**Supersedes**: Partially amends ADR-029 (RVF Canonical Format), ADR-030 (Cognitive Container)
**Affects**: `rvf-types`, `rvf-runtime`, `rvf-manifest`, `rvf-crypto`, `rvf-wasm`

---

## Context

Analysis of the progressive indexing system (spec chapters 02-04) revealed four structural weaknesses that convert engineered guarantees into opportunistic behavior:

1. **Centroid stability** depends on physical layout, not logical identity
2. **Layer A recall** collapses silently under adversarial distributions
3. **Recall targets** are empirical, presented as if they were bounds
4. **Manifest integrity** is optional, leaving the hotset attack surface open

Each issue individually is tolerable. Together they form a compound vulnerability: an adversary who controls the data distribution AND the file tail can produce a structurally valid RVF file that returns confident, wrong answers with no detection mechanism.

This ADR converts all four from "known limitations" to "engineered defenses."

---

## Decision

### 1. Content-Addressed Centroid Stability

**Invariant**: Logical identity must not depend on physical layout.

#### 1.1 Content-Addressed Segment References

Hotset pointers in the Level 0 manifest currently store raw byte offsets:

```
0x058   8   centroid_seg_offset      Byte offset in file
```

Add a parallel content hash field for each hotset pointer:

```
Offset  Size  Field                    Description
------  ----  -----                    -----------
0x058   8     centroid_seg_offset      Byte offset (for fast seek)
0x0A0   16    centroid_content_hash    First 128 bits of SHAKE-256 of segment payload
```

The runtime validates:
1. Seek to `centroid_seg_offset`
2. Read segment header + payload
3. Compute SHAKE-256 of payload
4. Compare first 128 bits against `centroid_content_hash`
5. If mismatch: reject pointer, fall back to Level 1 directory scan

This makes compaction physically destructive but logically stable. The manifest re-points by offset for speed but verifies by hash for correctness.

#### 1.2 Centroid Epoch Monotonic Counter

Add to Level 0 root manifest:

```
Offset  Size  Field                    Description
------  ----  -----                    -----------
0x0B0   4     centroid_epoch           Monotonic counter, incremented on recomputation
0x0B4   4     max_epoch_drift          Maximum allowed drift before forced recompute
```

**Semantics**:
- `centroid_epoch` increments each time centroids are recomputed
- The manifest's global `epoch` counter tracks all mutations
- `epoch_drift = manifest.epoch - centroid_epoch`
- If `epoch_drift > max_epoch_drift`: runtime MUST either recompute centroids or widen `n_probe`

Default `max_epoch_drift`: 64 epochs.

#### 1.3 Automatic Quality Elasticity

When epoch drift is detected, the runtime applies controlled quality degradation instead of silent recall loss:

```rust
fn effective_n_probe(base_n_probe: u32, epoch_drift: u32, max_drift: u32) -> u32 {
    if epoch_drift <= max_drift / 2 {
        // Within comfort zone: no adjustment
        base_n_probe
    } else if epoch_drift <= max_drift {
        // Drift zone: linear widening up to 2x
        let scale = 1.0 + (epoch_drift - max_drift / 2) as f64 / max_drift as f64;
        (base_n_probe as f64 * scale).ceil() as u32
    } else {
        // Beyond max drift: double n_probe, schedule recomputation
        base_n_probe * 2
    }
}
```

This turns degradation into **controlled quality elasticity**: recall trades against latency in a predictable, bounded way.

#### 1.4 Wire Format Changes

Add content hash fields to Level 0 at reserved offsets (using the `0x094-0x0FF` reserved region before the signature):

```
Offset  Size  Field
------  ----  -----
0x0A0   16    entrypoint_content_hash
0x0B0   16    toplayer_content_hash
0x0C0   16    centroid_content_hash
0x0D0   16    quantdict_content_hash
0x0E0   16    hot_cache_content_hash
0x0F0   4     centroid_epoch
0x0F4   4     max_epoch_drift
0x0F8   8     reserved_hardening
```

Total: 96 bytes. Fits within the existing reserved region before the signature at `0x094`.

**Note**: The signature field at `0x094` must move to accommodate this. New signature offset: `0x100`. This is a breaking change to the Level 0 layout. Files written before ADR-033 are detected by `version < 2` in the root manifest and use the old layout.

---

### 2. Layer A Adversarial Resilience

**Invariant**: Silent catastrophic degradation must not be possible.

#### 2.1 Distance Entropy Detection

After computing distances to the top-K centroids, measure the discriminative power:

```rust
/// Detect adversarial or degenerate centroid distance distributions.
/// Returns true if the distribution is too uniform to trust centroid routing.
fn is_degenerate_distribution(distances: &[f32], k: usize) -> bool {
    if distances.len() < 2 * k {
        return true; // Not enough centroids
    }

    // Sort and take top-2k
    let mut sorted = distances.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let top = &sorted[..2 * k];

    // Compute coefficient of variation (CV = stddev / mean)
    let mean = top.iter().sum::<f32>() / top.len() as f32;
    if mean < f32::EPSILON {
        return true; // All distances near zero
    }

    let variance = top.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / top.len() as f32;
    let cv = variance.sqrt() / mean;

    // CV < 0.05 means top distances are within 5% of each other
    // This indicates centroids provide no discriminative power
    cv < DEGENERATE_CV_THRESHOLD
}

const DEGENERATE_CV_THRESHOLD: f32 = 0.05;
```

#### 2.2 Adaptive n_probe Widening

When degeneracy is detected, widen the search:

```rust
fn adaptive_n_probe(
    base_n_probe: u32,
    centroid_distances: &[f32],
    total_centroids: u32,
) -> u32 {
    if is_degenerate_distribution(centroid_distances, base_n_probe as usize) {
        // Degenerate: widen to sqrt(K) or 4x base, whichever is smaller
        let widened = (total_centroids as f64).sqrt().ceil() as u32;
        base_n_probe.max(widened).min(base_n_probe * 4)
    } else {
        base_n_probe
    }
}
```

#### 2.3 Multi-Centroid Fallback

When distance variance is below threshold AND Layer B is not yet loaded, fall back to a lightweight multi-probe strategy:

1. Compute distances to ALL centroids (not just top-K)
2. If all distances are within `mean +/- 2*stddev`: treat as uniform
3. For uniform distributions: scan the hot cache linearly (if available)
4. If no hot cache: return results with a `quality_flag = APPROXIMATE` in the response

This prevents silent wrong answers. The caller knows the result quality.

#### 2.4 Quality Flag in Search Results

Add a quality indicator to `SearchResult`:

```rust
/// Quality confidence for a search result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResultQuality {
    /// Full index available, high confidence.
    Full = 0x00,
    /// Partial index (Layer A+B), good confidence.
    Partial = 0x01,
    /// Layer A only, moderate confidence.
    LayerAOnly = 0x02,
    /// Degenerate distribution detected, low confidence.
    /// Caller should retry after Layer B/C loads.
    DegenerateDetected = 0x03,
    /// Brute-force fallback used, exact but slow.
    BruteForce = 0x04,
}
```

The caller can then decide: wait for better index, accept approximate, or force brute-force.

#### 2.5 Distribution Assumption Declaration

The spec MUST explicitly state:

> **Distribution Assumption**: Recall targets (0.70/0.85/0.95) assume sub-Gaussian embedding distributions typical of neural network outputs (sentence-transformers, OpenAI ada-002, Cohere embed-v3, etc.). For adversarial, synthetic, or uniform-random distributions, recall may be lower. When degenerate distributions are detected at query time, the runtime automatically widens its search and signals reduced confidence via `ResultQuality`.

This converts an implicit assumption into an explicit contract.

---

### 3. Recall Bound Framing

**Invariant**: Never claim theoretical guarantees without distribution assumptions.

#### 3.1 Monotonic Recall Improvement Property

Replace hard recall bounds with a provable structural property:

> **Monotonic Recall Property**: For any query Q and any two index states S1 and S2 where S2 includes all segments of S1 plus additional INDEX_SEGs:
>
> `recall(Q, S2) >= recall(Q, S1)`
>
> Proof: S2's candidate set is a superset of S1's (append-only segments, no removal). More candidates cannot reduce recall.

This is provable from the append-only invariant and requires no distribution assumption.

#### 3.2 Recall Target Classes

Replace the single recall table with benchmark-class-specific targets:

| State | Natural Embeddings | Synthetic Uniform | Adversarial Clustered |
|-------|-------------------|-------------------|----------------------|
| Layer A | >= 0.70 | >= 0.40 | >= 0.20 (with detection) |
| A + B | >= 0.85 | >= 0.70 | >= 0.60 |
| A + B + C | >= 0.95 | >= 0.90 | >= 0.85 |

"Natural Embeddings" = sentence-transformers, OpenAI, Cohere on standard corpora.

#### 3.3 Brute-Force Safety Net

When the candidate set from HNSW search is smaller than `2 * k`:

```rust
fn query_with_safety_net(
    query: &[f32],
    k: usize,
    hnsw_candidates: &[Candidate],
    store: &RvfStore,
) -> (Vec<SearchResult>, ResultQuality) {
    if hnsw_candidates.len() >= 2 * k {
        // Sufficient candidates, trust HNSW
        (top_k(hnsw_candidates, k), ResultQuality::Full)
    } else {
        // Insufficient candidates — HNSW may have missed neighbors
        // Fall back to brute-force on the hot cache
        let hot_candidates = brute_force_hot_cache(query, store);
        let merged = merge_candidates(hnsw_candidates, &hot_candidates);
        let quality = if merged.len() >= 2 * k {
            ResultQuality::Partial
        } else {
            ResultQuality::DegenerateDetected
        };
        (top_k(&merged, k), quality)
    }
}
```

This gives a **correctness safety net** at the cost of latency. Silent wrong answers become impossible — the system either returns high-quality results or signals that it cannot.

#### 3.4 Acceptance Test Update

Update `benchmarks/acceptance-tests.md` to:

1. Test against three distribution classes (natural, synthetic, adversarial)
2. Verify `ResultQuality` flag accuracy
3. Verify monotonic recall improvement across progressive load phases
4. Measure brute-force fallback frequency and latency impact

---

### 4. Mandatory Manifest Signatures

**Invariant**: No signature, no mount in secure mode.

#### 4.1 Security Mount Policy

Add a `SecurityPolicy` enum to `RvfOptions`:

```rust
/// Manifest signature verification policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SecurityPolicy {
    /// No signature verification. For development and testing only.
    /// Files open regardless of signature state.
    Permissive = 0x00,
    /// Warn on missing or invalid signatures, but allow open.
    /// Log events for auditing.
    WarnOnly = 0x01,
    /// Require valid signature on Level 0 manifest.
    /// Reject files with missing or invalid signatures.
    /// DEFAULT for production.
    Strict = 0x02,
    /// Require valid signatures on Level 0, Level 1, and all
    /// hotset-referenced segments. Full chain verification.
    Paranoid = 0x03,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self::Strict
    }
}
```

**Default is `Strict`**, not `Permissive`.

#### 4.2 Verification Chain

Under `Strict` policy, the open path becomes:

```
1. Read Level 0 (4096 bytes)
2. Validate CRC32C (corruption check)
3. Validate ML-DSA-65 signature (adversarial check)
4. If signature missing: REJECT with SecurityError::UnsignedManifest
5. If signature invalid: REJECT with SecurityError::InvalidSignature
6. Extract hotset pointers
7. For each hotset pointer: validate content hash (ADR-033 §1.1)
8. If any content hash fails: REJECT with SecurityError::ContentHashMismatch
9. System is now queryable with verified pointers
```

Under `Paranoid` policy, add:

```
10. Read Level 1 manifest
11. Validate Level 1 signature
12. For each segment in directory: verify content hash matches on first access
```

#### 4.3 Unsigned File Handling

Files without signatures can still be opened under `Permissive` or `WarnOnly` policies. This supports:

- Development and testing workflows
- Legacy files created before signature support
- Performance-critical paths where verification latency is unacceptable

But the default is `Strict`. If an enterprise deploys with defaults, they get signature enforcement. They must explicitly opt out.

#### 4.4 Signature Generation on Write

Every `write_manifest()` call MUST:

1. Compute SHAKE-256-256 content hashes for all hotset-referenced segments
2. Store hashes in Level 0 at the new offsets (§1.4)
3. If a signing key is available: sign Level 0 with ML-DSA-65
4. If no signing key: write `sig_algo = 0` (unsigned)

The `create()` and `open()` methods accept an optional signing key:

```rust
impl RvfStore {
    pub fn create_signed(
        path: &Path,
        options: RvfOptions,
        signing_key: &MlDsa65SigningKey,
    ) -> Result<Self, RvfError>;
}
```

#### 4.5 Runtime Policy Flag

The security policy is set at store open time and cannot be downgraded:

```rust
let store = RvfStore::open_with_policy(
    &path,
    RvfOptions::default(),
    SecurityPolicy::Strict,
)?;
```

A store opened with `Strict` policy will reject any hotset pointer that fails content hash verification, even if the CRC32C passes. This prevents the segment-swap attack identified in the analysis.

---

## Consequences

### Positive

- Centroid stability becomes a **logical invariant**, not a physical accident
- Adversarial distribution degradation becomes **detectable and bounded**
- Recall claims become **honest** — empirical targets with explicit assumptions
- Manifest integrity becomes **mandatory by default** — enterprises are secure without configuration
- Quality elasticity replaces silent degradation — the system tells you when it's uncertain

### Negative

- Level 0 layout change is **breaking** (version 1 -> version 2)
- Content hash computation adds ~50 microseconds per manifest write
- Strict signature policy adds ~200 microseconds per file open (ML-DSA-65 verify)
- Adaptive n_probe increases query latency by up to 4x under degenerate distributions

### Migration

- Level 0 version field (`0x004`) distinguishes v1 (pre-ADR-033) from v2
- v1 files are readable under `Permissive` policy (no content hashes, no signature)
- v1 files trigger a warning under `WarnOnly` policy
- v1 files are rejected under `Strict` policy unless explicitly migrated
- Migration tool: `rvf migrate --sign --key <path>` rewrites manifest with v2 layout

---

## Size Impact

| Component | Additional Bytes | Where |
|-----------|-----------------|-------|
| Content hashes (5 pointers * 16 bytes) | 80 B | Level 0 manifest |
| Centroid epoch + drift fields | 8 B | Level 0 manifest |
| ResultQuality field per result | 1 B | Query response |
| SecurityPolicy in options | 1 B | Runtime config |
| Total Level 0 overhead | 96 B | Within existing 4096 B page |

No additional segments. No file size increase beyond the 96 bytes in Level 0.

---

## Implementation Order

| Phase | Component | Estimated Effort |
|-------|-----------|-----------------|
| 1 | Content hash fields in `rvf-types` Level 0 layout | Small |
| 2 | `centroid_epoch` + `max_epoch_drift` in manifest | Small |
| 3 | `ResultQuality` enum in `rvf-runtime` | Small |
| 4 | `is_degenerate_distribution()` + adaptive n_probe | Medium |
| 5 | Content hash verification in read path | Medium |
| 6 | `SecurityPolicy` enum + enforcement in open path | Medium |
| 7 | ML-DSA-65 signing in write path | Large (depends on rvf-crypto) |
| 8 | Brute-force safety net in query path | Medium |
| 9 | Acceptance test updates (3 distribution classes) | Medium |
| 10 | Migration tool (`rvf migrate --sign`) | Medium |

---

## References

- RVF Spec 02: Manifest System (hotset pointers, Level 0 layout)
- RVF Spec 04: Progressive Indexing (Layer A/B/C recall targets)
- RVF Spec 03: Temperature Tiering (centroid refresh, sketch epochs)
- ADR-029: RVF Canonical Format (universal adoption across libraries)
- ADR-030: Cognitive Container (three-tier execution model)
- FIPS 204: ML-DSA (Module-Lattice Digital Signature Algorithm)
- Malkov & Yashunin (2018): HNSW search complexity analysis
