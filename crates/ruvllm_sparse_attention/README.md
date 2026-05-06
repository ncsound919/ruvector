# ruvllm_sparse_attention

Subquadratic sparse attention kernel for the ruvllm inference stack, optimised for the Hailo-10H cluster (Pi 5 + Hailo NPU nodes). Implements ADR-183 through ADR-190.

## Features

| Feature | ADR | Status |
|---|---|---|
| Zero runtime dep footprint (`rand` dev-only) | ADR-183 | Accepted |
| One-pass online softmax (~2× FLOPs reduction) | ADR-184 | Accepted |
| Non-causal landmark block-exclusion fix | ADR-185 | Accepted |
| 17-test CI suite, validated on all 4 Pi 5 cluster nodes | ADR-186 | Accepted |
| Overflow-checked `Tensor3::zeros` | ADR-187 | Accepted |
| Stamp-scheme comments (cross-head dedup safety) | ADR-188 | Accepted |
| KV cache incremental decode (`decode_step`) | ADR-189 | Accepted |
| GQA/MQA support (`forward_gqa`, `forward_auto`) | ADR-190 | Accepted |

## Attention complexity

| Mode | Per-token cost | Total for N tokens |
|---|---|---|
| Dense MHA | O(T) | O(N²) |
| Sparse `forward` | O(log T) | O(N log N) |
| Sparse `decode_step` (KV cache) | O(log T) | O(N log N) |

## Supported model attention types

| Model | Type | Q heads | KV heads | `forward_auto` path |
|---|---|---|---|---|
| Phi-2 | MHA | 32 | 32 | `forward` |
| Mistral-7B | GQA | 32 | 8 | `forward_gqa` |
| Llama-3-8B | GQA | 32 | 8 | `forward_gqa` |
| Llama-3.2-1B | GQA | 32 | 8 | `forward_gqa` |
| TinyLlama-1.1B | MQA | 32 | 4 | `forward_gqa` |

## Quick start

```rust
use ruvllm_sparse_attention::{
    SubquadraticSparseAttention, SparseAttentionConfig, KvCache, Tensor3,
    AttentionBackend,
};

// MHA prefill
let attn = SubquadraticSparseAttention::new(SparseAttentionConfig::default()).unwrap();
let q = Tensor3::zeros(512, 32, 128);  // [seq, heads, dim]
let k = Tensor3::zeros(512, 32, 128);
let v = Tensor3::zeros(512, 32, 128);
let out = attn.forward(&q, &k, &v).unwrap();

// GQA prefill (Mistral-7B: 32 Q heads, 8 KV heads)
let q = Tensor3::zeros(512, 32, 128);
let k = Tensor3::zeros(512, 8, 128);   // 4x smaller KV cache
let v = Tensor3::zeros(512, 8, 128);
let out = attn.forward_auto(&q, &k, &v).unwrap();

// Incremental decode with KV cache (O(log T) per step)
let mut cache = KvCache::new(4096, 8, 128);
let new_k = Tensor3::zeros(1, 8, 128);
let new_v = Tensor3::zeros(1, 8, 128);
cache.append(&new_k, &new_v);
let q_new = Tensor3::zeros(1, 32, 128);
let out = attn.decode_step(&q_new, &cache).unwrap();
```

## Benchmarks

Config: 8 heads, dim=64, window=128, block_size=64, causal=true, log-stride+landmarks enabled.

| seq | x86-64 (AMD Ryzen) | Pi 5 Cortex-A76 | edge reduction vs dense |
|---|---|---|---|
| 512 | 13.1 ms | 85.8 ms | 2.2× |
| 1024 | 28.4 ms | 190.5 ms | 4.0× |
| 2048 | 60.1 ms | 401.0 ms | 7.7× |
| 4096 | 126.5 ms | 836.2 ms | 15.0× |
| 8192 | 262.6 ms | ~1,660 ms (est.) | 29.3× |

Pi 5 measurements collected on `cognitum-v0` (Raspberry Pi 5 Model B Rev 1.1,
aarch64, compiled with `-C target-cpu=cortex-a76 -C target-feature=+lse,+rcpc,+fp16,+crc`).

## Hailo-10H KV cache memory (Mistral-7B, seq=8192)

```
# MHA (naive expand): 8192 × 32 × 128 × 2 × 2 bytes = 8.6 GB  — does not fit
# GQA (this crate):   8192 ×  8 × 128 × 2 × 2 bytes = 2.1 GB  — fits in Hailo-10H DDR4
```

## Sparse edge estimates (default config)

```
seq       dense_causal   sparse   reduction
512          131,328      59,778     2.2×
1,024        524,800     129,858     4.0×
2,048      2,098,176     272,130     7.7×
4,096      8,390,656     560,834    15.0×
8,192     33,558,528   1,146,498    29.3×
16,384   134,225,920   2,334,274    57.5×
32,768   536,887,296   4,742,658   113.2×
```

## Cluster validation

All 17 tests pass on all 4 Hailo-10H cluster nodes (release build, aarch64):

| Node | Result |
|---|---|
| cognitum-v0 | 17/17 ✓ |
| cognitum-v1 | 17/17 ✓ |
| cognitum-cluster-2 | 17/17 ✓ |
| cognitum-cluster-3 | 17/17 ✓ |

## Build and test

```bash
# Local
cargo test -p ruvllm_sparse_attention --lib

# Cross-compile for Pi 5 (Cortex-A76)
RUSTFLAGS="-C target-cpu=cortex-a76 -C target-feature=+lse,+rcpc,+fp16,+crc" \
  CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
  cargo test -p ruvllm_sparse_attention --lib --target aarch64-unknown-linux-gnu

# Benchmark
cargo bench -p ruvllm_sparse_attention
```

## License

MIT
