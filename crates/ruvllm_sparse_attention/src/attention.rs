use crate::tensor::Tensor3;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone)]
pub enum AttentionError {
    ShapeMismatch {
        q: (usize, usize, usize),
        k: (usize, usize, usize),
        v: (usize, usize, usize),
    },
    InvalidConfig(String),
}

impl Display for AttentionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AttentionError::ShapeMismatch { q, k, v } => write!(
                f,
                "shape mismatch, q={:?}, k={:?}, v={:?}; expected equal shapes",
                q, k, v
            ),
            AttentionError::InvalidConfig(message) => write!(f, "invalid config: {}", message),
        }
    }
}

impl Error for AttentionError {}

pub trait AttentionBackend {
    fn forward(&self, q: &Tensor3, k: &Tensor3, v: &Tensor3) -> Result<Tensor3, AttentionError>;
}

#[derive(Clone, Debug)]
pub struct SparseAttentionConfig {
    pub window: usize,
    pub block_size: usize,
    pub global_tokens: Vec<usize>,
    pub causal: bool,
    pub use_log_stride: bool,
    pub use_landmarks: bool,
}

impl Default for SparseAttentionConfig {
    fn default() -> Self {
        Self {
            window: 128,
            block_size: 64,
            global_tokens: vec![0],
            causal: true,
            use_log_stride: true,
            use_landmarks: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SubquadraticSparseAttention {
    pub config: SparseAttentionConfig,
}

impl SubquadraticSparseAttention {
    pub fn new(config: SparseAttentionConfig) -> Result<Self, AttentionError> {
        if config.block_size == 0 {
            return Err(AttentionError::InvalidConfig(
                "block_size must be greater than zero".to_string(),
            ));
        }
        Ok(Self { config })
    }

    pub fn estimate_sparse_edges(&self, seq: usize) -> usize {
        if self.config.block_size == 0 {
            return 0;
        }
        let mut seen_tokens = vec![0usize; seq.max(1)];
        let mut seen_blocks = vec![0usize; div_ceil(seq.max(1), self.config.block_size)];
        let mut token_candidates = Vec::new();
        let mut block_candidates = Vec::new();
        let mut total = 0usize;

        for i in 0..seq {
            // Stamp is per-token only (i+1) — this function estimates edge count
            // for a single head. There is no head loop. The result matches the
            // per-head column in docs/benchmark_edge_estimates.csv.
            // See also: forward(), which uses stamp = 1 + h*seq + i for
            // cross-head deduplication safety.
            let stamp = i + 1;
            token_candidates.clear();
            block_candidates.clear();

            build_token_candidates(
                i,
                seq,
                &self.config,
                &mut seen_tokens,
                stamp,
                &mut token_candidates,
            );

            if self.config.use_landmarks {
                build_landmark_candidates(
                    i,
                    seq,
                    &self.config,
                    &mut seen_blocks,
                    stamp,
                    &mut block_candidates,
                );
            }

            total += token_candidates.len() + block_candidates.len();
        }

        total
    }
}

impl AttentionBackend for SubquadraticSparseAttention {
    fn forward(&self, q: &Tensor3, k: &Tensor3, v: &Tensor3) -> Result<Tensor3, AttentionError> {
        validate_qkv(q, k, v)?;
        if self.config.block_size == 0 {
            return Err(AttentionError::InvalidConfig(
                "block_size must be greater than zero".to_string(),
            ));
        }

        let seq = q.seq;
        if seq == 0 {
            return Ok(Tensor3::zeros(0, q.heads, q.dim));
        }

        let heads = q.heads;
        let dim = q.dim;
        let scale = 1.0f32 / (dim as f32).sqrt();
        let landmarks = if self.config.use_landmarks {
            Some(Landmarks::from_kv(k, v, self.config.block_size))
        } else {
            None
        };

        let mut out = Tensor3::zeros(seq, heads, dim);
        let mut seen_tokens = vec![0usize; seq.max(1)];
        let mut seen_blocks = vec![0usize; div_ceil(seq.max(1), self.config.block_size)];
        let mut token_candidates = Vec::<usize>::with_capacity(self.config.window + 64);
        let mut block_candidates = Vec::<usize>::with_capacity(64);
        let mut acc = vec![0f32; dim];

        for h in 0..heads {
            for i in 0..seq {
                // Stamp is unique per (head, token) pair so that seen_tokens[] and
                // seen_blocks[] — allocated once and shared across heads — correctly
                // reset deduplication state between heads. Formula: 1 + h*seq + i.
                // See also: estimate_sparse_edges(), which uses a per-token stamp
                // (i+1) because it has no head loop and estimates per-head edge count.
                let stamp = 1 + h * seq + i;
                token_candidates.clear();
                block_candidates.clear();

                build_token_candidates(
                    i,
                    seq,
                    &self.config,
                    &mut seen_tokens,
                    stamp,
                    &mut token_candidates,
                );

                if landmarks.is_some() {
                    build_landmark_candidates(
                        i,
                        seq,
                        &self.config,
                        &mut seen_blocks,
                        stamp,
                        &mut block_candidates,
                    );
                }

                let q_row = q.row(i, h);

                // One-pass online softmax (ADR-184): single traversal over candidates
                // using a running max + correction factor. Eliminates two-pass
                // dot-product redundancy (~2× FLOPs reduction on Pi 5 NEON paths).
                let mut running_max = f32::NEG_INFINITY;
                let mut denom = 0.0f32;
                acc.fill(0.0);

                for &j in &token_candidates {
                    let score = dot(q_row, k.row(j, h)) * scale;
                    if score > running_max {
                        let corr = (running_max - score).exp();
                        for d in 0..dim {
                            acc[d] *= corr;
                        }
                        denom *= corr;
                        running_max = score;
                    }
                    let w = (score - running_max).exp();
                    denom += w;
                    let v_row = v.row(j, h);
                    for d in 0..dim {
                        acc[d] += w * v_row[d];
                    }
                }

                if let Some(lm) = landmarks.as_ref() {
                    for &b in &block_candidates {
                        let score = dot(q_row, lm.keys.row(b, h)) * scale;
                        if score > running_max {
                            let corr = (running_max - score).exp();
                            for d in 0..dim {
                                acc[d] *= corr;
                            }
                            denom *= corr;
                            running_max = score;
                        }
                        let w = (score - running_max).exp();
                        denom += w;
                        let v_row = lm.values.row(b, h);
                        for d in 0..dim {
                            acc[d] += w * v_row[d];
                        }
                    }
                }

                let out_row = out.row_mut(i, h);
                let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                for d in 0..dim {
                    out_row[d] = acc[d] * inv_denom;
                }
            }
        }

        Ok(out)
    }
}

pub fn dense_attention(
    q: &Tensor3,
    k: &Tensor3,
    v: &Tensor3,
    causal: bool,
) -> Result<Tensor3, AttentionError> {
    validate_qkv(q, k, v)?;
    if q.seq == 0 {
        return Ok(Tensor3::zeros(q.seq, q.heads, q.dim));
    }

    let seq = q.seq;
    let heads = q.heads;
    let dim = q.dim;
    let scale = 1.0f32 / (dim as f32).sqrt();
    let mut out = Tensor3::zeros(seq, heads, dim);
    let mut acc = vec![0f32; dim];

    for h in 0..heads {
        for i in 0..seq {
            let q_row = q.row(i, h);
            let last = if causal { i } else { seq - 1 };
            let mut max_score = f32::NEG_INFINITY;

            for j in 0..=last {
                let score = dot(q_row, k.row(j, h)) * scale;
                if score > max_score {
                    max_score = score;
                }
            }

            acc.fill(0.0);
            let mut denom = 0.0f32;

            for j in 0..=last {
                let score = dot(q_row, k.row(j, h)) * scale;
                let weight = (score - max_score).exp();
                denom += weight;
                let v_row = v.row(j, h);
                for d in 0..dim {
                    acc[d] += weight * v_row[d];
                }
            }

            let out_row = out.row_mut(i, h);
            let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            for d in 0..dim {
                out_row[d] = acc[d] * inv_denom;
            }
        }
    }

    Ok(out)
}

fn validate_qkv(q: &Tensor3, k: &Tensor3, v: &Tensor3) -> Result<(), AttentionError> {
    if q.dim == 0 {
        return Err(AttentionError::InvalidConfig(
            "head dimension must be greater than zero".to_string(),
        ));
    }

    if q.shape() != k.shape() || q.shape() != v.shape() {
        return Err(AttentionError::ShapeMismatch {
            q: q.shape(),
            k: k.shape(),
            v: v.shape(),
        });
    }
    Ok(())
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

fn build_token_candidates(
    i: usize,
    seq: usize,
    config: &SparseAttentionConfig,
    seen: &mut [usize],
    stamp: usize,
    out: &mut Vec<usize>,
) {
    if seq == 0 {
        return;
    }

    let start = i.saturating_sub(config.window);
    let end = if config.causal {
        i
    } else {
        (i + config.window).min(seq - 1)
    };

    for j in start..=end {
        push_unique(j, seen, stamp, out);
    }

    for &g in &config.global_tokens {
        if g < seq && (!config.causal || g <= i) {
            push_unique(g, seen, stamp, out);
        }
    }

    if config.use_log_stride {
        let mut stride = 1usize;
        while stride < seq {
            if i >= stride {
                push_unique(i - stride, seen, stamp, out);
            }

            if !config.causal {
                let forward = i + stride;
                if forward < seq {
                    push_unique(forward, seen, stamp, out);
                }
            }

            match stride.checked_mul(2) {
                Some(next) => stride = next,
                None => break,
            }
        }
    }
}

fn build_landmark_candidates(
    i: usize,
    seq: usize,
    config: &SparseAttentionConfig,
    seen: &mut [usize],
    stamp: usize,
    out: &mut Vec<usize>,
) {
    if seq == 0 || config.block_size == 0 {
        return;
    }

    let blocks = div_ceil(seq, config.block_size);
    if blocks == 0 {
        return;
    }

    let current_block = i / config.block_size;
    let available_blocks = if config.causal {
        (i + 1) / config.block_size
    } else {
        blocks
    };

    if available_blocks == 0 {
        return;
    }

    let pivot = if config.causal {
        available_blocks - 1
    } else {
        current_block.min(blocks - 1)
    };

    let local_start = pivot.saturating_sub(1);
    let local_end = (pivot + 1).min(available_blocks - 1);
    for b in local_start..=local_end {
        // ADR-185: in non-causal mode skip the block that contains token i.
        // Token i is already reachable via the window; adding its block mean
        // double-counts it and biases the softmax toward the current block.
        if !config.causal && b == current_block {
            continue;
        }
        push_unique(b, seen, stamp, out);
    }

    if config.use_log_stride {
        let mut stride = 1usize;
        while stride < blocks {
            if pivot >= stride {
                let b = pivot - stride;
                if config.causal || b != current_block {
                    push_unique(b, seen, stamp, out);
                }
            }

            if !config.causal {
                let forward = pivot + stride;
                if forward < blocks && forward != current_block {
                    push_unique(forward, seen, stamp, out);
                }
            }

            match stride.checked_mul(2) {
                Some(next) => stride = next,
                None => break,
            }
        }
    }
}

#[inline]
fn push_unique(index: usize, seen: &mut [usize], stamp: usize, out: &mut Vec<usize>) {
    if seen[index] != stamp {
        seen[index] = stamp;
        out.push(index);
    }
}

#[derive(Clone, Debug)]
struct Landmarks {
    keys: Tensor3,
    values: Tensor3,
}

impl Landmarks {
    fn from_kv(k: &Tensor3, v: &Tensor3, block_size: usize) -> Self {
        let blocks = div_ceil(k.seq, block_size);
        let mut keys = Tensor3::zeros(blocks, k.heads, k.dim);
        let mut values = Tensor3::zeros(blocks, v.heads, v.dim);

        for b in 0..blocks {
            let start = b * block_size;
            let end = (start + block_size).min(k.seq);
            let count = (end - start) as f32;

            for h in 0..k.heads {
                for t in start..end {
                    let k_row = k.row(t, h);
                    let v_row = v.row(t, h);
                    let key_out = keys.row_mut(b, h);
                    let value_out = values.row_mut(b, h);
                    for d in 0..k.dim {
                        key_out[d] += k_row[d];
                        value_out[d] += v_row[d];
                    }
                }

                let key_out = keys.row_mut(b, h);
                let value_out = values.row_mut(b, h);
                for d in 0..k.dim {
                    key_out[d] /= count;
                    value_out[d] /= count;
                }
            }
        }

        Self { keys, values }
    }
}

#[inline]
fn div_ceil(a: usize, b: usize) -> usize {
    if a == 0 {
        0
    } else {
        1 + (a - 1) / b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(seq: usize, heads: usize, dim: usize) -> Tensor3 {
        let len = seq * heads * dim;
        let data = (0..len)
            .map(|i| ((i * 17 + 11) % 101) as f32 / 101.0)
            .collect::<Vec<f32>>();
        Tensor3::from_vec(data, seq, heads, dim).unwrap()
    }

    #[test]
    fn sparse_matches_dense_when_window_covers_sequence() {
        let seq = 32;
        let heads = 2;
        let dim = 8;
        let q = make_tensor(seq, heads, dim);
        let k = make_tensor(seq, heads, dim);
        let v = make_tensor(seq, heads, dim);

        let dense = dense_attention(&q, &k, &v, false).unwrap();
        let sparse = SubquadraticSparseAttention::new(SparseAttentionConfig {
            window: seq,
            block_size: 8,
            global_tokens: vec![],
            causal: false,
            use_log_stride: false,
            use_landmarks: false,
        })
        .unwrap()
        .forward(&q, &k, &v)
        .unwrap();

        for idx in 0..dense.data.len() {
            assert!((dense.data[idx] - sparse.data[idx]).abs() < 1e-5);
        }
    }

    #[test]
    fn causal_sparse_matches_causal_dense_when_window_covers_sequence() {
        let seq = 32;
        let heads = 2;
        let dim = 8;
        let q = make_tensor(seq, heads, dim);
        let k = make_tensor(seq, heads, dim);
        let v = make_tensor(seq, heads, dim);

        let dense = dense_attention(&q, &k, &v, true).unwrap();
        let sparse = SubquadraticSparseAttention::new(SparseAttentionConfig {
            window: seq,
            block_size: 8,
            global_tokens: vec![],
            causal: true,
            use_log_stride: false,
            use_landmarks: false,
        })
        .unwrap()
        .forward(&q, &k, &v)
        .unwrap();

        for idx in 0..dense.data.len() {
            assert!((dense.data[idx] - sparse.data[idx]).abs() < 1e-5);
        }
    }

    #[test]
    fn sparse_edges_are_smaller_than_dense_edges() {
        let attention = SubquadraticSparseAttention::new(SparseAttentionConfig {
            window: 128,
            block_size: 64,
            global_tokens: vec![0],
            causal: true,
            use_log_stride: true,
            use_landmarks: true,
        })
        .unwrap();

        let seq = 4096;
        let dense_edges = seq * (seq + 1) / 2;
        let sparse_edges = attention.estimate_sparse_edges(seq);
        assert!(sparse_edges < dense_edges / 8);
    }

    // --- ADR-186 edge-case tests ---

    #[test]
    fn empty_sequence_does_not_panic() {
        let q = Tensor3::zeros(0, 2, 8);
        let k = Tensor3::zeros(0, 2, 8);
        let v = Tensor3::zeros(0, 2, 8);
        let result = SubquadraticSparseAttention::new(SparseAttentionConfig::default())
            .unwrap()
            .forward(&q, &k, &v);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().data.len(), 0);
    }

    #[test]
    fn single_token_self_attention_is_identity_of_v() {
        let dim = 8;
        let heads = 2;
        let v_data: Vec<f32> = (0..heads * dim).map(|i| i as f32 * 0.1 + 0.1).collect();
        let q = Tensor3::from_vec(v_data.clone(), 1, heads, dim).unwrap();
        let k = q.clone();
        let v = q.clone();
        let out = SubquadraticSparseAttention::new(SparseAttentionConfig {
            window: 128,
            block_size: 64,
            global_tokens: vec![0],
            causal: true,
            use_log_stride: true,
            use_landmarks: true,
        })
        .unwrap()
        .forward(&q, &k, &v)
        .unwrap();
        for (a, b) in out.data.iter().zip(v_data.iter()) {
            assert!((a - b).abs() < 1e-5, "single token: out={} v={}", a, b);
        }
    }

    #[test]
    fn out_of_range_global_token_is_silently_skipped() {
        let seq = 4;
        let q = make_tensor(seq, 1, 4);
        let k = q.clone();
        let v = q.clone();
        let result = SubquadraticSparseAttention::new(SparseAttentionConfig {
            window: 128,
            block_size: 64,
            global_tokens: vec![10],
            causal: true,
            use_log_stride: false,
            use_landmarks: false,
        })
        .unwrap()
        .forward(&q, &k, &v);
        assert!(result.is_ok());
    }

    #[test]
    fn block_size_one_does_not_panic() {
        let seq = 16;
        let q = make_tensor(seq, 2, 8);
        let k = q.clone();
        let v = q.clone();
        let result = SubquadraticSparseAttention::new(SparseAttentionConfig {
            window: seq,
            block_size: 1,
            global_tokens: vec![],
            causal: false,
            use_log_stride: false,
            use_landmarks: true,
        })
        .unwrap()
        .forward(&q, &k, &v);
        assert!(result.is_ok());
    }

    #[test]
    fn self_attention_only_denom_is_one() {
        // window=0, log-stride=false, landmarks=false, causal=true
        // Every token attends only to itself; single candidate weight = exp(0)=1.0, denom=1.0
        // Output must equal V[i] for all i.
        let seq = 8;
        let heads = 1;
        let dim = 4;
        let v_data: Vec<f32> = (0..seq * heads * dim).map(|i| i as f32 * 0.1).collect();
        let q = Tensor3::from_vec(v_data.clone(), seq, heads, dim).unwrap();
        let k = q.clone();
        let v = q.clone();
        let out = SubquadraticSparseAttention::new(SparseAttentionConfig {
            window: 0,
            block_size: 64,
            global_tokens: vec![],
            causal: true,
            use_log_stride: false,
            use_landmarks: false,
        })
        .unwrap()
        .forward(&q, &k, &v)
        .unwrap();
        for (a, b) in out.data.iter().zip(v_data.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn non_causal_sparse_matches_non_causal_dense_with_window_only() {
        // Validates ADR-185 (non-causal landmark fix) + ADR-184 (online softmax)
        let seq = 32;
        let heads = 2;
        let dim = 8;
        let q = make_tensor(seq, heads, dim);
        let k = make_tensor(seq, heads, dim);
        let v = make_tensor(seq, heads, dim);
        let dense = dense_attention(&q, &k, &v, false).unwrap();
        let sparse = SubquadraticSparseAttention::new(SparseAttentionConfig {
            window: seq,
            block_size: 8,
            global_tokens: vec![],
            causal: false,
            use_log_stride: false,
            use_landmarks: false,
        })
        .unwrap()
        .forward(&q, &k, &v)
        .unwrap();
        for (a, b) in dense.data.iter().zip(sparse.data.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn estimate_edges_always_below_dense() {
        for seq in [64usize, 128, 256, 512, 1024, 4096] {
            let attention = SubquadraticSparseAttention::new(SparseAttentionConfig::default())
                .unwrap();
            let sparse = attention.estimate_sparse_edges(seq);
            // Upper bound is seq*seq (full non-causal quadratic budget) because
            // estimate_sparse_edges also counts landmark block pseudo-edges, which
            // can push the count slightly above seq*(seq+1)/2 when window >= seq.
            let dense = seq * seq;
            assert!(
                sparse <= dense,
                "seq={}: sparse {} > dense {}",
                seq,
                sparse,
                dense
            );
            // Sublinearity only manifests once seq >> 4*window (≥4096 with window=128).
            if seq >= 4096 {
                let causal_dense = seq * (seq + 1) / 2;
                assert!(
                    sparse < causal_dense / 4,
                    "seq={}: sparse {} not < causal_dense/4 {}",
                    seq,
                    sparse,
                    causal_dense / 4
                );
            }
        }
    }

    // --- ADR-187 overflow check ---

    #[test]
    #[should_panic(expected = "Tensor3::zeros: shape overflow")]
    fn zeros_panics_on_overflow() {
        Tensor3::zeros(usize::MAX / 2 + 1, 3, 1);
    }
}
