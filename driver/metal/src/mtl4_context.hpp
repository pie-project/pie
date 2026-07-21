#pragma once
// mtl4_context.hpp — alpha's Metal-4 wrapper scaffold for the raw-Metal decode path.
//
// Pure-C++ surface (no Obj-C types leak): every lane includes this from plain .cpp/.mm.
// The Metal-4 objects (MTL4CommandQueue / MTLHeap / MTLResidencySet / double-buffered
// MTL4CommandAllocator / MTL4ArgumentTable) live behind a PIMPL in mtl4_context.mm.
//
// Verified boilerplate reference: beta's files/icb-probes/mtl4probe.mm (runtime
// newLibraryWithSource + MTL4Compiler PSO + queue/allocator/argtable/residency/event).
//
// Contract keyed off delta's decode_abi.hpp (Region / IoSlot / bind:: / Kernel).
// Ownership split (manager): delta lays out the heap (region offsets) + ports kernels;
// beta encodes the per-step command buffer + replay; alpha owns these wrappers + harness.
//
// ── Toolchain reality (this box) ─────────────────────────────────────────────
//   No offline `metal`/`metallib` compiler is installed (CommandLineTools only, no Xcode).
//   So shaders are compiled at RUNTIME via [MTLDevice newLibraryWithSource:] + MTL4Compiler.
//   The CMake AOT `.metal`->`.metallib` path is gated on the toolchain being present
//   (PIE_RAW_METAL_AOT, default OFF). Runtime compile is the de-risked Phase-0 path.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "decode_abi.hpp"  // delta owns this (pure C++); Region / IoSlot / bind / Kernel

namespace pie::metal {

struct MetalStorageFacts {
    std::uint32_t alignment = 1;
    std::uint32_t page_size = 1;
};

MetalStorageFacts query_metal_storage_facts();

bool read_ptir_msl_source(
    const std::string& path,
    std::string& source,
    std::string* error);

// ── Opaque handles (borrowed; lifetime owned by RawMetalContext) ──────────────

// A sub-range of the single resident heap. `contents()` is the CPU-visible pointer
// (heap is Shared storage on UMA — valid for weight staging + IO-scalar writes).
struct SlotHandle {
    void*    buffer       = nullptr;  // id<MTLBuffer> (placed sub-buffer), borrowed
    void*    contents_ptr = nullptr;  // CPU pointer into the slot (Shared storage)
    uint64_t gpu_address  = 0;        // base GPU VA of the slot (for setAddress)
    size_t   offset       = 0;        // byte offset within the heap
    size_t   size         = 0;        // slot size in bytes
    bool     elastic      = false;    // placement-sparse VA backed by heap chunks

    bool  valid()    const { return buffer != nullptr; }
    void* contents() const { return contents_ptr; }
};

// A compiled compute pipeline state (built via MTL4Compiler). Borrowed.
struct Pso { void* obj = nullptr; bool valid() const { return obj != nullptr; } };

// 3D launch geometry.
struct Grid { uint32_t x = 1, y = 1, z = 1; };
struct Threadgroup { uint32_t x = 1, y = 1, z = 1; };

// Barrier cache-visibility for the intra-encoder compute->compute hazard. Pure-C++
// mirror of MTL4VisibilityOptions (no Obj-C in this header); mapped in mtl4_context.mm.
//   * Device        — flush caches to the GPU (device) coherence point. Correct for a
//                     real RAW where the consumer reads the producer's heap write.
//   * ExecutionOnly — order execution only, NO cache flush (MTL4VisibilityOptionNone).
//                     Cheaper; valid where ordering alone suffices / UMA L2-coherent.
// A `PIE_BARRIER_VIS=none|device` env var overrides ALL barriers at runtime (delta's
// global visibility sweep) regardless of the per-call argument; absent => per-call arg.
enum class BarrierVisibility : uint8_t { ExecutionOnly = 0, Device = 1 };

// Per-step timing split (manager wants BOTH reported separately).
struct StepTiming {
    double encode_ms   = 0.0;  // begin_step -> end_step (CPU command-buffer build)
    double gpu_exec_ms = 0.0;  // commit -> event wait (GPU execution)
    bool completed = false;    // event fence reached; command resources may be released
    bool timed_out = false;    // the initial bounded wait expired before the fence
    double total_ms()  const { return encode_ms + gpu_exec_ms; }
    bool succeeded() const { return completed; }
};

struct TransientBufferPoolStats {
    std::uint64_t allocations = 0;
    std::uint64_t reuse_hits = 0;
    std::uint64_t recycles = 0;
    std::uint64_t evictions = 0;
    std::uint64_t allocation_failures = 0;
    std::size_t resident_buffers = 0;
    std::size_t resident_bytes = 0;
    std::size_t cached_buffers = 0;
    std::size_t cached_bytes = 0;
    std::size_t in_use_buffers = 0;
    std::size_t in_use_bytes = 0;
    std::size_t peak_resident_bytes = 0;
    std::size_t capacity_bytes = 0;
};

// ── StepEncoder — the per-dispatch surface beta's executor drives ─────────────
// Mirrors beta's flow: setPSO / setArgumentTable(slot) / dispatchThreads / barrier.
// Obtained from RawMetalContext::begin_step(); finalized by RawMetalContext::end_step().
class StepEncoder {
  public:
    void set_pso(Pso pso);
    // Bind the prebuilt argument table for a dispatch instance, keyed by its FLAT
    // ORDINAL (beta's DAG walker: 0..321, unique + stable token-to-token since the CB
    // is byte-identical). `k` is a decorative tag (charlie's dump naming) — the ordinal
    // alone is the key, because within one layer Rms/Residual recur (so (kind,layer) is
    // NOT unique). Prefer set_argtable_ordinal; this overload forwards `layer` as ordinal.
    void set_argtable(Kernel k, int ordinal = -1);
    void set_argtable_ordinal(int ordinal);
    void dispatch(Grid grid, Threadgroup tg);
    // Intra-encoder compute->compute hazard. Default ExecutionOnly: proven correct (argmax
    // 264 holds) AND free (delta+beta sweeps: device-flush within noise) on M1 Max UMA —
    // the placement heap is L2-coherent intra-encoder without an explicit flush. Pass
    // Device for an explicit cache flush; PIE_BARRIER_VIS env overrides all calls.
    void barrier(BarrierVisibility vis = BarrierVisibility::ExecutionOnly);

    // Write a GPU timestamp into `heap` at `idx` (beta's per-dispatch attribution). `heap`
    // is an opaque MTL4CounterHeap from RawMetalContext::create_timestamp_heap. Relaxed
    // granularity (lowest overhead, no encoder split — preserves the single-CB model);
    // pass precise=true only for boundary-accurate sampling (may split the encoder).
    void mark_timestamp(void* heap, uint32_t idx, bool precise = false);

    // convenience: one fused call per dispatch (ordinal-keyed)
    void encode(Pso pso, Kernel k, int ordinal, Grid grid, Threadgroup tg) {
        set_pso(pso); set_argtable_ordinal(ordinal); dispatch(grid, tg); barrier();
    }

  private:
    friend class RawMetalContext;
    explicit StepEncoder(void* impl) : impl_(impl) {}
    void* impl_;  // borrowed encoder state
};

// ── RawMetalContext — owns the Metal-4 device objects + heap + arg tables ─────
class RawMetalContext {
  public:
    struct Impl;  // Obj-C++ guts (defined in mtl4_context.mm)

    // heap_bytes: total single-heap budget (delta sizes from DecodeGeometry + manifest).
    static std::unique_ptr<RawMetalContext> create(
        size_t heap_bytes,
        size_t elastic_budget_bytes = 0);
    ~RawMetalContext();

    RawMetalContext(const RawMetalContext&)            = delete;
    RawMetalContext& operator=(const RawMetalContext&) = delete;

    // ── (1) Heap sub-allocation (delta's blocked signature) ──
    // Bump-allocates `size` (aligned) from the single placement heap. Deterministic
    // offsets in call order. align defaults to 256 (Metal buffer-offset alignment).
    SlotHandle heap_alloc(size_t size, size_t align = 256);

    // CPU-visible standalone storage for channels, IO staging, and other pools
    // intentionally excluded from elastic arenas.
    SlotHandle create_standalone_buffer(size_t size);
    // Private placement-sparse VA backed by lazily-created Shared placement
    // heaps. The VA and gpu_address never change as chunks grow or trim.
    SlotHandle create_elastic_buffer(
        size_t size,
        size_t initial_commit_bytes = 0);
    bool ensure_elastic_buffer(const SlotHandle& h, size_t bytes);
    bool ensure_elastic_buffers_atomically(
        const std::vector<std::pair<SlotHandle, size_t>>& targets);
    bool trim_elastic_buffer(const SlotHandle& h, size_t bytes);
    void release_elastic_buffer(const SlotHandle& h);
    bool zero_buffer_range(const SlotHandle& h, size_t offset, size_t bytes);
    bool copy_buffer_range(
        const SlotHandle& dst,
        size_t dst_offset,
        const SlotHandle& src,
        size_t src_offset,
        size_t bytes);
    size_t elastic_page_bytes() const;
    size_t elastic_budget_pages() const;
    size_t elastic_committed_pages() const;
    void drain_elastic_mappings();
    size_t pending_elastic_release_count() const;

    // Size-classed, residency-stable storage for PTIR command scratch and
    // metadata. Recycle only after the command's completion fence.
    SlotHandle acquire_transient_buffer(size_t size);
    void recycle_transient_buffer(const SlotHandle& h);
    TransientBufferPoolStats transient_buffer_pool_stats() const;
    void set_transient_buffer_pool_limit_for_test(size_t bytes);

    // Add a Shared-storage buffer owned elsewhere (for example, an
    // authoritative PTIR channel ring) to this context's residency set.
    void use_external_buffer(const SlotHandle& h);
    void release_external_buffer(const SlotHandle& h);
    size_t external_buffer_count() const;

    // Phase 3 (review item 4) — release a standalone buffer previously handed
    // out by create_standalone_buffer: drop it from the residency set
    // (removeAllocation + commit) AND from the context's retained-alive array
    // so ARC actually frees the GPU allocation. Without this, `resize_pool`'s
    // repeated grow/shrink would leak the OLD K/V buffers forever (they stay
    // retained + resident), growing GPU memory unbounded. `contents()`/
    // `gpu_address` on `h` are invalid after this call. A no-op for an invalid
    // (zero) handle or one this context never allocated.
    void release_standalone_buffer(const SlotHandle& h);

    // Host-visible allocation probe over all STANDALONE (non-heap) buffers.
    // This includes resident transient-pool buffers until context destruction;
    // use transient_buffer_pool_stats() to distinguish cached from in-use
    // command storage. The fixed placement heap is not counted.
    size_t standalone_buffer_count() const;
    size_t standalone_bytes() const;

    // Make the whole heap resident ONCE (invariant I2). Call after all heap_alloc +
    // all arg_bind, before the first encode.
    void make_resident();

    // ── (2) Argument-table bind, keyed by delta's bind:: enums (built once, I2) ──
    // The arg-table key is the FLAT DISPATCH ORDINAL (beta's DAG walker, 0..321):
    // unique + stable token-to-token. `layer`/`k` are decorative (charlie's dump
    // naming) — they DON'T disambiguate, because within one layer-cycle Rms and
    // Residual each recur, so (kind, layer) collides. Pass the dispatch ordinal as
    // the int param; delta + beta share the same ordinal space.
    void arg_bind(Kernel k, int ordinal, uint8_t bind_index, SlotHandle slot,
                  size_t offset = 0);
    // Explicit ordinal-keyed form (kind elided — the ordinal is the only key).
    void arg_bind_ordinal(int ordinal, uint8_t bind_index, SlotHandle slot,
                          size_t offset = 0);
    // Binding introspection for DAG coverage tests.  It reports host-side
    // table population, not shader reflection, and is safe before residency.
    bool arg_slot_is_bound(int ordinal, uint8_t bind_index) const;
    uint64_t arg_slot_address(int ordinal, uint8_t bind_index) const;
    void release_argtable_ordinal(int ordinal);
    // delta's exact 1-arg-less form for singleton kernels (ordinal = -1).
    void arg_bind(Kernel k, uint8_t bind_index, SlotHandle slot, size_t offset = 0) {
        arg_bind(k, -1, bind_index, slot, offset);
    }

    // ── Shaders (runtime-compiled; no offline toolchain needed) ──
    Pso compile_pso(const std::string& metal_source, const std::string& fn_name,
                    std::string* error = nullptr);
    Pso compile_pso_from_file(const std::string& metal_path, const std::string& fn_name,
                              std::string* error = nullptr);
    // PTIR semantics require strict NaN/tie behavior. This path always passes
    // explicit safe-math options (MTLMathModeSafe, or fastMathEnabled=NO on
    // older SDKs).
    Pso compile_ptir_pso(const std::string& metal_source, const std::string& fn_name,
                         std::string* error = nullptr);
    Pso compile_ptir_pso_from_file(
        const std::string& metal_path,
        const std::string& fn_name,
        std::string* error = nullptr);
    Pso compile_ptir_pso_cached(
        const std::string& metal_source,
        const std::string& fn_name,
        const std::string& archive_path,
        bool* cache_hit = nullptr,
        std::string* error = nullptr);
    void release_pso(Pso pso);
    size_t retained_pso_count() const;
    bool last_ptir_compile_disabled_fast_math() const;
    std::uint64_t device_cache_id() const;

    // ── GPU timestamp attribution (beta's per-dispatch / per-phase timing) ──
    // Allocate an opaque MTL4CounterHeap of `count` timestamp entries (owned by the
    // context; lives until destruction). During encode, StepEncoder::mark_timestamp
    // writes a timestamp at an index; AFTER run_step (GPU complete — the event is
    // already waited) resolve_timestamps copies the `count` resolved GPU timestamps
    // (nanoseconds on this device) into `out`. Returns nullptr on failure.
    void* create_timestamp_heap(uint32_t count);
    void  resolve_timestamps(void* heap, uint32_t count, uint64_t* out);
    void  release_timestamp_heap(void* heap);

    // ── Encode one decode step. `encode_fn` issues the DAG via StepEncoder ──
    // Uses the double-buffered allocator (ab = 0/1) so the harness can overlap
    // encode(N+1) with GPU(N). Returns the encode/GPU split for THIS step.
    StepTiming run_step(const std::function<void(StepEncoder&)>& encode_fn, int ab = 0);
    void force_next_wait_timeout_for_test();

    // ── Pipelined async commit (downclock-ceiling prototype) ──
    // Encode+commit a step WITHOUT waiting for completion (returns the signalled event value),
    // so the next step's commit follows back-to-back and the GPU never drains between steps.
    // The device-fed NextToken removes the host dependency, making this safe in principle; the
    // caller is responsible for the WAR hazard on per-step-mutated argtables/IO (the prototype
    // keeps binds constant = timing-only, valid for the clock question since GPU work is
    // identical regardless of data). Pair with sync_event() to bound in-flight / drain.
    uint64_t commit_step_async(const std::function<void(StepEncoder&)>& encode_fn, int ab = 0);
    // As above, but the committed CB waits for `wait_value` on the queue timeline before it
    // executes (GPU-side serialization for the autoregressive single-stream dependency).
    uint64_t commit_step_async_dep(const std::function<void(StepEncoder&)>& encode_fn, int ab,
                                   uint64_t wait_value);
    // Wait until the queue has signalled >= value (bounds in-flight to `depth` and final drain).
    void     sync_event(uint64_t value);
    uint64_t last_event() const;

    // ── Continuous-async GPU keepalive (downclock proof-of-ceiling) ──
    // Spawns a background thread on a SEPARATE MTL4 command queue that commits a tunable
    // compute-spin dispatch back-to-back with a bounded in-flight depth (no per-CB host
    // wait) so the GPU clock domain never gates between the main loop's per-token drains.
    // This is the EXPERIMENT that proves the gap is 100% DVFS downclock (does gpu_exec
    // reach the 3.78ms hot floor?) — NOT a shippable fix (the resident loop is). Tunables:
    //   spin_iters  — inner loop count per thread (GPU duty per dispatch)
    //   threadgroups — grid width (occupancy)
    //   depth       — max in-flight command buffers (>=2 keeps overlap, never fully drains)
    void start_keepalive(uint32_t spin_iters, uint32_t threadgroups, uint32_t depth);
    void stop_keepalive();

  private:
    RawMetalContext();
    std::unique_ptr<Impl> impl_;
    friend class StepEncoder;
};

}  // namespace pie::metal
