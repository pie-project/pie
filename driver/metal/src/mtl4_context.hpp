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

/// A `.metal` source with its `#include "..."` directives spliced in.
///
/// Metal's runtime compiler resolves no local includes of its own, so this is
/// what lets two kernel files share a definition instead of restating it.
bool read_metal_source(
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
    double gpu_exec_ms = 0.0;  // commit -> event wait (GPU execution, host-observed)
    // GPU-reported execution time for the commit, from the Metal 4 commit
    // feedback handler (GPUEndTime - GPUStartTime). Unlike `gpu_exec_ms` this
    // excludes host wake-up latency. Zero when the feedback had not landed by
    // the time the step returned -- it is delivered asynchronously.
    double gpu_ms      = 0.0;
    bool completed = false;    // event fence reached; command resources may be released
    // The driver stopped waiting for the fence. It used to mean "the first
    // five-second probe expired", which a slow-but-healthy step also does --
    // so the one caller that acted on it killed slow steps, while a step that
    // never completed was retried forever and reported nothing at all. It now
    // means the wait was abandoned, `completed` is false, and the context is
    // finished: its command buffers may still be running, so no allocator it
    // owns can be reset.
    bool timed_out = false;
    bool gpu_error = false;    // commit feedback reported a GPU-side error
    // What the GPU said, when it said anything. `gpu_error` alone sent every
    // caller to the same "timed out before its completion fence" message, so a
    // command buffer that ran, failed, and reported
    // `kIOGPUCommandBufferCallbackErrorOutOfMemory` was indistinguishable from
    // one that never came back.
    std::string gpu_error_text;
    double total_ms()  const { return encode_ms + gpu_exec_ms; }
    bool succeeded() const { return completed && !gpu_error; }
};

// Asynchronous per-commit report from Metal 4's commit feedback handler. The
// driver keeps the most recent one; `event_value` identifies which point on the
// queue timeline it describes.
struct GpuCommitFeedback {
    uint64_t event_value = 0;
    double gpu_start_s = 0.0;
    double gpu_end_s = 0.0;
    double gpu_ms = 0.0;
    bool had_error = false;
    std::string error;
    bool valid() const { return event_value != 0; }
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
    // The encode body is shared by the ordered and unordered runners, and a
    // shared body has to be a free function -- neither runner can host it
    // without the other calling across a class boundary. This is that function,
    // named here only so it can be trusted with the constructor.
    // Both take and return their Metal objects as `void*`, like every other
    // Obj-C handle this header carries, because the header is included by
    // plain C++ translation units that cannot name an `id<>`.
    friend void* encode_one_command_buffer(
        void* ctx_impl, int ab,
        const std::function<void(StepEncoder&)>& encode_fn);
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

    /// What this device is willing to hold resident, in bytes.
    ///
    /// Free-standing because it answers BEFORE a context exists: the question
    /// "will this model fit" has to be asked before nineteen gigabytes are
    /// copied in. Exceeding it does not fail an allocation -- every buffer is
    /// created, every bind succeeds, and then the command buffer comes back
    /// with "The operation couldn't be completed", whose underlying error,
    /// three levels down, is `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
    static size_t device_working_set_bytes();

    /// Test hook: make `device_working_set_bytes` answer `bytes` instead of
    /// asking the device, or 0 to ask it again. A refusal that fires when a
    /// model is too big for the GPU cannot otherwise be exercised on a GPU the
    /// models fit on -- and the alternative, trusting that the check is wired
    /// into every setup path because it was written once, is exactly how
    /// `setup_simple` came to have no check at all.
    static void set_device_working_set_bytes_for_test(size_t bytes);
    ~RawMetalContext();

    RawMetalContext(const RawMetalContext&)            = delete;
    RawMetalContext& operator=(const RawMetalContext&) = delete;

    // ── (1) Heap sub-allocation (delta's blocked signature) ──
    // Bump-allocates `size` (aligned) from the single placement heap. Deterministic
    // offsets in call order. align defaults to 256 (Metal buffer-offset alignment).
    SlotHandle heap_alloc(size_t size, size_t align = 256);

    /// A heap slot MEMOIZED by the argument-table slot it will be bound to.
    ///
    /// Constants are rebound whenever the row count changes, and a fresh
    /// `heap_alloc` per rebind leaks: a batch whose size varies fire to fire
    /// walks the heap until `heap_alloc` returns nothing, and the model fails
    /// to set up its NEXT sequence with "budget too small". The value at a
    /// given (ordinal, index) is always the same size, so the allocation can be
    /// made once and rewritten -- which is safe because a rebind happens
    /// between steps, and a step blocks on its completion fence.
    SlotHandle const_slot(int ordinal, std::uint8_t index, size_t bytes);

    // CPU-visible standalone storage for channels, IO staging, and other pools
    // intentionally excluded from elastic arenas.
    SlotHandle create_standalone_buffer(size_t size);

    /// A buffer over memory this context does NOT own — an mmap of the
    /// checkpoint, in practice. `ptr` must be page-aligned and stay mapped for
    /// the buffer's lifetime.
    ///
    /// This is what weight streaming would be on Apple silicon: a file-backed
    /// mapping wrapped this way is demand-faulted under GPU access and its
    /// pages stay clean, so the kernel evicts them under pressure instead of
    /// the driver paying for every weight, resident, forever.
    SlotHandle wrap_host_memory(void* ptr, size_t size);
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
    void set_memory_pressure_level_for_test(std::uint32_t level);
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
    // One entrypoint to build out of one source file.
    struct PsoFileRequest {
        std::string path;
        std::string function;
    };
    // Build many pipelines in one batch. Each distinct source file is read and
    // turned into an MTLLibrary once even when several entrypoints share it,
    // and, when `archive_path` names a Metal 4 pipeline archive, the compiled
    // binaries are looked up there instead of being rebuilt: pipeline creation,
    // not source parsing, is where a cold start spends its time. The archive is
    // written on the first run that misses it.
    //
    // Compilation is deliberately serial. Metal funnels compiles through its own
    // service, so extra threads measured no faster, and the MTL4Compiler's
    // completion-handler API overruns the stack of Metal's scheduler threads on
    // batches this size.
    //
    // Returns one Pso per request, positionally; a failed entry is invalid and
    // its reason lands in `errors` (sized to match) when provided.
    std::vector<Pso> compile_psos_from_files(
        const std::vector<PsoFileRequest>& requests,
        std::vector<std::string>* errors = nullptr,
        bool use_archive_cache = true);

    // Directory holding the pipeline archives written by
    // `compile_psos_from_files`. Defaults to a per-user cache directory;
    // PIE_METAL_PSO_CACHE overrides it, and setting that to an empty value
    // turns archive caching off.
    static std::string pso_archive_dir();
    // PTIR semantics require strict NaN/tie behavior. This path always passes
    // explicit safe-math options (MTLMathModeSafe, or fastMathEnabled=NO on
    // older SDKs).
    Pso compile_precise_pso_from_file(const std::string& path, const std::string& fn,
                                      std::string* error);
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
    // Largest threadgroup this pipeline can be launched with. A big fused
    // kernel can fall well short of the device maximum on registers alone, and
    // asking for more than it allows fails the dispatch.
    uint32_t pso_max_threads(Pso pso) const;

    void* create_timestamp_heap(uint32_t count);
    void  resolve_timestamps(void* heap, uint32_t count, uint64_t* out);
    void  release_timestamp_heap(void* heap);

    // ── Encode one decode step. `encode_fn` issues the DAG via StepEncoder ──
    // Uses the double-buffered allocator (ab = 0/1) so the harness can overlap
    // encode(N+1) with GPU(N). Returns the encode/GPU split for THIS step.
    StepTiming run_step(const std::function<void(StepEncoder&)>& encode_fn, int ab = 0);
    // Encode N independent command buffers and submit them in ONE
    // `commit:count:options:` call. The command buffers execute without ordering
    // guarantees relative to each other, so `encode_fns` must be mutually
    // hazard-free; a single event signal fences the whole batch. Use this when a
    // pass splits into parallel chunks that would otherwise pay N submits.
    StepTiming run_steps(const std::vector<std::function<void(StepEncoder&)>>& encode_fns,
                         int ab = 0);
    // Encode and run N segments IN ORDER, with the host running between them.
    //
    // `between(i)` runs after segment `i` has completed on the GPU and before
    // segment `i + 1` is encoded, so it may read what the segment computed and
    // may change what the next one reads. It runs after EVERY segment, the last
    // one included, because a caller that pinned something per segment needs
    // somewhere to give the last pin back. That is the whole reason this exists
    // and the one thing neither `run_step` nor `run_steps` can do: both commit
    // everything before waiting for anything, which is right when the host has
    // nothing to add and impossible when it does.
    //
    // The caller pays one submit and one completion wait per segment. Splitting
    // a step that does not need the host is therefore a straight loss; this is
    // for the case where the host holds something the GPU cannot compute for
    // itself -- which, for expert paging, is which weights exist at all.
    //
    // Distinct from `run_steps` in ordering as well as in the callback: those
    // command buffers race each other on purpose, these are serialized by the
    // wait between them.
    StepTiming run_segments(const std::vector<std::function<void(StepEncoder&)>>& encode_fns,
                            const std::function<void(std::size_t)>& between,
                            int ab = 0);
    // Most recent Metal 4 commit feedback (GPU-measured timing, GPU-side error).
    // Delivered asynchronously, so it may lag the last committed step.
    GpuCommitFeedback last_commit_feedback() const;
    // Makes the next wait abandon immediately rather than spend its budget, so
    // the abandon path can be tested without a wedged GPU or a minute of
    // waiting. The context is genuinely finished afterwards, exactly as it
    // would be in the real case.
    void force_next_wait_timeout_for_test();

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
