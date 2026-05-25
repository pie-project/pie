# TODO

## Kimi CUDA Driver

- Investigate the remaining Moon-prompt repetition case.
  - France smoke test on the normal CUDA path is sensible: `The capital of France is Paris.`
  - The Moon prompt still repeats meta text, and forcing the prefill MoE path did not change it.
  - Likely areas: MLA/KV cache state, decode context update, position handling, stop handling, or prompt/template edge cases.
- Keep `PIE_CUDA_KIMI_FORCE_PREFILL_MOE` as a temporary diagnostic switch only.
  - Remove it once the decode path is validated.
- Add focused correctness tests for Kimi decode.
  - Single request, deterministic temperature 0.
  - Multi-token decode with EOS enabled and disabled.
  - Batch size > 1 with mixed prompt lengths.
  - Long context prefill followed by decode.
- Compare intermediate tensors against a trusted implementation for a tiny request.
  - Start with logits after prefill.
  - Then compare first decode step.
  - If logits diverge, narrow by layer: embedding, MLA attention, MoE routing, expert output, residual/norm.

## Weight Loader Correctness

- Add regression coverage for async safetensors storage copies.
  - `copy_storage_bytes_to_device_async` must require mapped shard data before using `shard.data + file_offset`.
  - Cover mmap-backed and non-mmap fallback behavior if both paths remain supported.
- Add tests for `SlabScatter` lowering and FFI layout.
  - Multiple placements from one source span.
  - Destination offsets into persistent arena.
  - Rejection of source or destination ranges that exceed bounds.
  - Stable ABI layout for `PieLoaderStorageInstrKind::SlabScatter`.
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
