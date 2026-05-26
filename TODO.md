# TODO

## Kimi CUDA Driver

- ~~Investigate the remaining Moon-prompt repetition case.~~ (partially resolved)
  - **Fixed**: 4 bugs found and fixed:
    1. MLA KV cache uninitialized (`mla_cache.cpp`: `cudaMemset` after alloc)
    2. Missing `e_score_correction_bias` for noaux_tc MoE routing (`kimi.hpp/cpp`, `kimi_mla.cu/hpp`, `generic.rs`)
    3. YaRN RoPE not applied: `has_rope_scaling=true` missing from yarn branch, wrong `mscale_all_dim` handling (`hf_config.cpp`), added `launch_rope_yarn_original_bf16` call (`kimi_forward.cpp`)
    4. Routing weights used biased scores instead of original sigmoid (`kimi_mla.cu`)
  - **Result**: France smoke test correct ("Paris."). Model recognizes topics correctly. Raw completion produces correct content.
  - **Remaining**: Output still repeats for multi-sentence generation. Root cause: FlashInfer `BatchMLAPagedAttention` (absorbed Q, 512-dim dot products) accumulates more FP error per layer than vLLM's explicit-K/V prefill + TRT-LLM decode path (128-dim dot products). Fix: implement explicit K/V attention for prefill (compute K,V from `kv_b_proj @ kv_c`) matching vLLM's approach.
- Keep `PIE_CUDA_KIMI_FORCE_PREFILL_MOE` as a temporary diagnostic switch only.
  - Remove it once the decode path is validated.
- Implement explicit K/V prefill for Kimi MLA.
  - During prefill, compute `kv = kv_b_proj(kv_c)` → explicit K_nope, V.
  - Use standard FlashInfer prefill attention (not absorbed MLA).
  - Continue using absorbed MLA for decode (read from compressed KV cache).
  - This matches vLLM's architecture and should fix the remaining repetition.
- Add focused correctness tests for Kimi decode.
  - Single request, deterministic temperature 0.
  - Multi-token decode with EOS enabled and disabled.
  - Batch size > 1 with mixed prompt lengths.
  - Long context prefill followed by decode.

## Weight Loader Correctness

- Add regression coverage for async safetensors storage copies.
  - `copy_storage_bytes_to_device_async` must require mapped shard data before using `shard.data + file_offset`.
  - Cover mmap-backed and non-mmap fallback behavior if both paths remain supported.
- ~~Add tests for `SlabScatter` lowering and FFI layout.~~ (done)
  - ✓ Multiple placements from one source span (`slab_scatter_merges_nearby_bulk_extent_writes`).
  - ✓ Placement offsets within span bounds (`slab_scatter_placement_offsets_are_within_span`).
  - ✓ Rejection of excessive overread (`slab_scatter_rejects_excessive_overread`).
  - ✓ FFI layout (`slab_scatter_ffi_carries_placement_fields`).
- Add an executor-level smoke test for `SlabScatter`.
  - Copy a contiguous source slab to GPU staging.
  - Scatter into two or more destination ranges.
  - Read back and validate exact bytes.

## Weight Loader Performance

- Reduce the high copy-command count.
  - Current Kimi profile is roughly `161k` H2D copies per rank for about `48.9 GiB`.
  - This dominates load latency even with multiple copy streams.
  - Target is fewer, larger physical copy commands without excessive overread.
- Replace the current conservative slab heuristic with a file-layout aware batching plan.
  - Group by shard file and source byte order first.
  - Build bounded windows over nearby checkpoint ranges.
  - Scatter into runtime tensor layout from those windows.
  - Preserve layout algebra generality: the compiler should reason about source spans and destination placements, not model names.
- Make batching cost-based.
  - Inputs: payload bytes, overread bytes, placement count, staging memory, expected memcpy launch overhead, and available bandwidth.
  - Reject slabs when overread or staging pressure is worse than many direct copies.
  - Prefer slabs when command reduction wins at low memory cost.
- Improve per-rank load balance.
  - Current materialization varies from about `34s` to `46s` per rank.
  - Profile whether the gap is file locality, NUMA, GPU copy contention, or rank-specific tensor layout.
- Keep pinned staging and GDS as optional experiments.
  - Mmap direct host-to-device copies are currently the baseline to beat.
  - Pinned staging should only stay enabled if it improves both load time and peak memory.
  - GDS should be guarded by capability checks and should not complicate the default path.

## Weight Loader Compiler

- Revisit the compiled instruction shape for Kimi.
  - Current compiled program is still mostly many runtime-layout extent writes.
  - The better shape is file-window reads plus scatter placements when that wins by cost.
- Add a storage-plan optimizer pass after algebra lowering.
  - Input: exact source byte ranges and destination byte ranges.
  - Output: direct copies, slab scatters, or future DMA-friendly operations.
  - Keep this pass model-agnostic.
- Persist richer compile-cache metadata.
  - Program version and ABI version.
  - Checkpoint file paths, sizes, mtimes, and tensor metadata hash.
  - Compiler options that affect batching thresholds.
- Add a debug dump for compiled storage programs.
  - Number of instructions by kind.
  - H2D command count estimate.
  - Total payload bytes, overread bytes, and staging bytes.
  - Largest slabs and worst rejected candidates.
- Improve compile-time profiling.
  - Separate source metadata parsing, algebra lowering, extent coalescing, slab planning, ABI arena build, and cache serialization.
  - Report cache hit/miss timing clearly.

## Memory Footprint

- Measure true peak memory during model loading.
  - CPU RSS.
  - GPU allocated bytes per rank.
  - Persistent arena bytes.
  - Temporary staging bytes.
  - Page cache effects from mmap.
- Keep slab staging bounded.
  - Enforce a per-rank staging budget.
  - Reuse one staging allocation per executor.
  - Avoid plans that reduce copy count by consuming meaningful model capacity.
- Check fragmentation risk.
  - Persistent arena allocation should remain stable.
  - Temporary staging should be allocated once and reused.
  - Avoid many variable-size CUDA allocations during materialization.

## Benchmarks

- Add a repeatable Kimi loader benchmark under `benches/`.
  - Cold compile cache.
  - Warm compile cache.
  - Warm OS page cache.
  - Optional cold OS page cache if the environment allows it.
- Benchmark against vLLM with identical conditions.
  - Same checkpoint path.
  - Same GPU set.
  - Same prompt set.
  - Same max tokens and sampling settings.
- Track these metrics:
  - Time to parse metadata.
  - Time to compile/load cached storage plan.
  - Time to materialize weights.
  - Peak CPU RSS.
  - Peak GPU memory.
  - First-token latency.
  - Decode tokens/sec.
  - End-to-end throughput.
- Add correctness prompts to the benchmark suite.
  - Simple factual answer.
  - Short instruction requiring EOS.
  - Prompt that previously triggered Moon repetition.
  - Longer context prompt.

## Cleanup Before Merge

- Decide whether `SlabScatter` is the final name.
  - If it remains, document it in the weight-loader storage IR.
- Remove diagnostic-only logs or gate them behind env vars.
- Keep new loader optimizations general.
  - No Kimi-specific compiler branches.
  - Model-specific code should only describe model layout requirements, not loader execution strategy.
- Re-run:
  - `cargo test -p pie-weight-loader`
  - `cargo test -p pie --lib model::tokenizer::tests::test_tiktoken_loader_reads_hf_added_tokens_decoder`
  - `CMAKE_CUDA_ARCHITECTURES=90 cargo build --release -p pie-server --features driver-cuda`
- Fix or quarantine unrelated full-suite failure before claiming full test cleanliness.
  - Current known issue: `runtime/tests/rs_cache.rs` still uses the old `pie::context::spawn` signature.
