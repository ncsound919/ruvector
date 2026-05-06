# ruvllm sparse attention

This prototype implements subquadratic sparse attention in Rust for a ruvllm style inference engine.

## Pattern

The backend keeps local detail and adds long range reachability through global tokens, logarithmic token jumps, and block landmarks.

## Run

```bash
cargo test
cargo run --example run_sparse_attention
cargo bench
```

## Files

```text
src/tensor.rs
src/attention.rs
src/model.rs
benches/attention_bench.rs
examples/run_sparse_attention.rs
docs/adr/ADR_0001_subquadratic_sparse_attention.md
```

## Default config

```rust
SparseAttentionConfig {
    window: 128,
    block_size: 64,
    global_tokens: vec![0],
    causal: true,
    use_log_stride: true,
    use_landmarks: true,
}
```

## Notes

The prototype is scalar CPU Rust. Production ruvllm should add KV cache support, SIMD, parallel execution, and GPU kernels.

## Analytical edge estimates

The table below uses the default config with 8 heads and 64 head dimension.

```text
seq,dense_causal_edges_per_head,sparse_edges_per_head,edge_reduction_x
512,131328,59778,2.20
1024,524800,129858,4.04
2048,2098176,272130,7.71
4096,8390656,560834,14.96
8192,33558528,1146498,29.27
16384,134225920,2334274,57.50
32768,536887296,4742658,113.20
65536,2147516416,9625026,223.12
```
