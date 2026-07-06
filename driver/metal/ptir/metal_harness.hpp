#pragma once
//
// MetalHarness — a small standalone Metal compute harness for bit-exact PTIR
// sampling-IR op parity testing. Runtime-compiles a .metal source file (this
// box is CLT-only, no offline metal compiler), builds pipelines by function
// name, binds Shared (UMA) buffers, dispatches a 1-D grid, and reads results
// back. The Metal dual of charlie's NVRTC-based CUDA sampling-IR test harness.
//
// Pure C++ interface (no Objective-C types leak through the header) so op tests
// stay backend-agnostic.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ptir_metal {

// One kernel argument. Input buffers are copied host->device before dispatch;
// output buffers are copied device->host after completion. Backed by Shared
// (unified-memory) storage.
struct Arg {
    void* data = nullptr;  // host pointer (read for input, written for output)
    std::size_t bytes = 0;
    bool is_output = false;
    bool is_inout = false;  // upload initial data AND copy back (in-place kernels)

    static Arg in(const void* p, std::size_t n) {
        return Arg{const_cast<void*>(p), n, false, false};
    }
    static Arg out(void* p, std::size_t n) { return Arg{p, n, true, false}; }
    static Arg inout(void* p, std::size_t n) { return Arg{p, n, true, true}; }
};

class MetalHarness {
public:
    MetalHarness();
    ~MetalHarness();
    MetalHarness(const MetalHarness&) = delete;
    MetalHarness& operator=(const MetalHarness&) = delete;

    // True if a default Metal device was created.
    bool ok() const;
    std::string device_name() const;
    // Last error message (compile / pipeline / dispatch failure).
    const std::string& error() const { return error_; }

    // Compile a .metal source file at runtime. Returns false + sets error() on
    // failure. Must succeed before run().
    bool load_library(const std::string& metal_source_path);

    // Dispatch `fn_name` over `grid_threads` threads (1-D). Args are bound at
    // buffer indices 0..args.size()-1 in order. Returns false + sets error()
    // on failure. Output args are populated on success.
    bool run(const std::string& fn_name, std::vector<Arg>& args,
             std::uint32_t grid_threads);

private:
    struct Impl;
    Impl* impl_;
    std::string error_;
};

// Bit-exact comparison of two float vectors: reinterpret each lane as u32 and
// require identical bits (so -inf / NaN / signed zero all compare exactly).
// Returns the number of the first differing lane, or -1 if identical.
long first_bit_diff(const std::vector<float>& a, const std::vector<float>& b);

}  // namespace ptir_metal
