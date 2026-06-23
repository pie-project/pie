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

#include "decode_abi.hpp"  // delta owns this (pure C++); Region / IoSlot / bind / Kernel

namespace pie_metal_driver::raw_metal {

// ── Opaque handles (borrowed; lifetime owned by RawMetalContext) ──────────────

// A sub-range of the single resident heap. `contents()` is the CPU-visible pointer
// (heap is Shared storage on UMA — valid for weight staging + IO-scalar writes).
struct SlotHandle {
    void*    buffer       = nullptr;  // id<MTLBuffer> (placed sub-buffer), borrowed
    void*    contents_ptr = nullptr;  // CPU pointer into the slot (Shared storage)
    uint64_t gpu_address  = 0;        // base GPU VA of the slot (for setAddress)
    size_t   offset       = 0;        // byte offset within the heap
    size_t   size         = 0;        // slot size in bytes

    bool  valid()    const { return buffer != nullptr; }
    void* contents() const { return contents_ptr; }
};

// A compiled compute pipeline state (built via MTL4Compiler). Borrowed.
struct Pso { void* obj = nullptr; bool valid() const { return obj != nullptr; } };

// 3D launch geometry.
struct Grid { uint32_t x = 1, y = 1, z = 1; };
struct Threadgroup { uint32_t x = 1, y = 1, z = 1; };

// Per-step timing split (manager wants BOTH reported separately).
struct StepTiming {
    double encode_ms   = 0.0;  // begin_step -> end_step (CPU command-buffer build)
    double gpu_exec_ms = 0.0;  // commit -> event wait (GPU execution)
    double total_ms()  const { return encode_ms + gpu_exec_ms; }
};

// ── StepEncoder — the per-dispatch surface beta's executor drives ─────────────
// Mirrors beta's flow: setPSO / setArgumentTable(slot) / dispatchThreads / barrier.
// Obtained from RawMetalContext::begin_step(); finalized by RawMetalContext::end_step().
class StepEncoder {
  public:
    void set_pso(Pso pso);
    // Bind the prebuilt argument table for a (kernel, layer) dispatch instance.
    void set_argtable(Kernel k, int layer = -1);
    void dispatch(Grid grid, Threadgroup tg);
    void barrier();  // device-visibility barrier between dependent dispatches

    // convenience: one fused call per dispatch
    void encode(Pso pso, Kernel k, int layer, Grid grid, Threadgroup tg) {
        set_pso(pso); set_argtable(k, layer); dispatch(grid, tg); barrier();
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
    static std::unique_ptr<RawMetalContext> create(size_t heap_bytes);
    ~RawMetalContext();

    RawMetalContext(const RawMetalContext&)            = delete;
    RawMetalContext& operator=(const RawMetalContext&) = delete;

    // ── (1) Heap sub-allocation (delta's blocked signature) ──
    // Bump-allocates `size` (aligned) from the single placement heap. Deterministic
    // offsets in call order. align defaults to 256 (Metal buffer-offset alignment).
    SlotHandle heap_alloc(size_t size, size_t align = 256);

    // Make the whole heap resident ONCE (invariant I2). Call after all heap_alloc +
    // all arg_bind, before the first encode.
    void make_resident();

    // ── (2) Argument-table bind, keyed by delta's bind:: enums (built once, I2) ──
    // `layer` disambiguates per-layer dispatch instances of the same Kernel kind
    // (e.g. QmvIn recurs across all GDN layers with distinct weight slots).
    // Use layer = -1 for singleton kernels (EmbedGather / FinalRms / Argmax).
    void arg_bind(Kernel k, int layer, uint8_t bind_index, SlotHandle slot,
                  size_t offset = 0);
    // delta's exact 1-arg-less form for singleton kernels (layer = -1).
    void arg_bind(Kernel k, uint8_t bind_index, SlotHandle slot, size_t offset = 0) {
        arg_bind(k, -1, bind_index, slot, offset);
    }

    // ── Shaders (runtime-compiled; no offline toolchain needed) ──
    Pso compile_pso(const std::string& metal_source, const std::string& fn_name,
                    std::string* error = nullptr);
    Pso compile_pso_from_file(const std::string& metal_path, const std::string& fn_name,
                              std::string* error = nullptr);

    // ── Encode one decode step. `encode_fn` issues the DAG via StepEncoder ──
    // Uses the double-buffered allocator (ab = 0/1) so the harness can overlap
    // encode(N+1) with GPU(N). Returns the encode/GPU split for THIS step.
    StepTiming run_step(const std::function<void(StepEncoder&)>& encode_fn, int ab = 0);

  private:
    RawMetalContext();
    std::unique_ptr<Impl> impl_;
    friend class StepEncoder;
};

}  // namespace pie_metal_driver::raw_metal
