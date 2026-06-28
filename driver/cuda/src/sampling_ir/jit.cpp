// NVRTC JIT + driver-API launch for the sampling-IR CUDA backend (lane L3).
// See jit.hpp for the codegen->JIT seam and the overall design.

#include "sampling_ir/jit.hpp"

#include <nvrtc.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace pie_cuda_driver::sampling_ir::jit {

namespace {

[[noreturn]] void throw_cu(CUresult res, const char* expr, const char* file, int line) {
    const char* name = nullptr;
    const char* desc = nullptr;
    cuGetErrorName(res, &name);
    cuGetErrorString(res, &desc);
    std::ostringstream oss;
    oss << "CUDA driver error: " << (name ? name : "?") << " — "
        << (desc ? desc : "?") << " (" << static_cast<int>(res) << ") at " << file
        << ":" << line << " — " << expr;
    throw std::runtime_error(oss.str());
}

#define CU_CHECK(expr)                                                        \
    do {                                                                      \
        CUresult _res = (expr);                                               \
        if (_res != CUDA_SUCCESS) ::pie_cuda_driver::sampling_ir::jit::throw_cu( \
            _res, #expr, __FILE__, __LINE__);                                 \
    } while (0)

#define NVRTC_CHECK(expr)                                                     \
    do {                                                                      \
        nvrtcResult _res = (expr);                                            \
        if (_res != NVRTC_SUCCESS) {                                          \
            std::ostringstream _oss;                                          \
            _oss << "NVRTC error: " << nvrtcGetErrorString(_res) << " at "    \
                 << __FILE__ << ":" << __LINE__ << " — " << #expr;            \
            throw std::runtime_error(_oss.str());                             \
        }                                                                     \
    } while (0)

// NVRTC-compile a self-contained CUDA-C source to PTX for `arch`
// (e.g. "compute_89"). On failure the compile log is folded into the error.
std::string compile_to_ptx(const std::string& name, const std::string& source,
                           const std::string& arch) {
    nvrtcProgram prog = nullptr;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, source.c_str(), (name + ".cu").c_str(),
                                   0, nullptr, nullptr));

    const std::string arch_opt = "--gpu-architecture=" + arch;
    const char* opts[] = {arch_opt.c_str()};
    const nvrtcResult compile_res =
        nvrtcCompileProgram(prog, 1, opts);

    if (compile_res != NVRTC_SUCCESS) {
        std::size_t log_size = 0;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        if (log_size > 0) nvrtcGetProgramLog(prog, log.data());
        nvrtcDestroyProgram(&prog);
        std::ostringstream oss;
        oss << "NVRTC failed to compile kernel '" << name
            << "': " << nvrtcGetErrorString(compile_res) << "\n--- log ---\n"
            << log;
        throw std::runtime_error(oss.str());
    }

    std::size_t ptx_size = 0;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
    std::string ptx(ptx_size, '\0');
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
    return ptx;
}

// #11 NVRTC PTX-gen pool width. NVRTC serializes internally (~1.9x at N=2 on
// CUDA 13.3, degrades past that), so the default is small — the pool hides
// compile latency behind in-flight steps, it is not a throughput pool. Override
// via `PIE_JIT_POOL_THREADS` for the load-test's pool-width sweep.
unsigned jit_pool_threads() {
    if (const char* env = std::getenv("PIE_JIT_POOL_THREADS")) {
        const int n = std::atoi(env);
        if (n >= 1 && n <= 64) return static_cast<unsigned>(n);
    }
    return 2;
}

}  // namespace

// fnv1a64 is now header-inline in program_hash.hpp (the canonical program hash).

// ───────────────────────────── CompiledProgram ───────────────────────────

CompiledProgram::~CompiledProgram() {
    for (std::size_t i = 0; i < buffer_ptrs_.size(); ++i) {
        if (buffer_owned_[i] && buffer_ptrs_[i]) cuMemFree(buffer_ptrs_[i]);
    }
    for (CUmodule m : modules_) {
        if (m) cuModuleUnload(m);
    }
    if (graph_exec_) cuGraphExecDestroy(graph_exec_);
    if (graph_) cuGraphDestroy(graph_);
}

CUdeviceptr CompiledProgram::device_ptr(BufferId id) const {
    auto it = buffer_slot_.find(id);
    if (it == buffer_slot_.end()) {
        throw std::runtime_error("sampling_ir::JIT: unknown BufferId " +
                                 std::to_string(id));
    }
    return buffer_ptrs_[it->second];
}

// ─────────────────────────────── JitEngine ───────────────────────────────

JitEngine::JitEngine() : pool_(jit_pool_threads()) {
    CUdevice dev = 0;
    CU_CHECK(cuCtxGetDevice(&dev));
    int major = 0, minor = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    arch_ = "compute_" + std::to_string(major) + std::to_string(minor);
}

// #11 finalize (CONTEXT-THREAD ONLY): allocate the buffer table + load each
// pre-built PTX into a module + resolve its CUfunction + build launch params.
// The NVRTC PTX-gen already ran off-thread (pool); `ptx[i]` is kernel `i`'s PTX.
void JitEngine::finalize(const KernelDAG& dag, const std::vector<std::string>& ptx,
                         CompiledProgram& prog) {
    prog.hash_ = dag.hash;

    // 1. Allocate the buffer table. Non-external buffers are owned by the JIT;
    //    external (IO) buffers start null and must be bound before launch.
    //    Batched intermediates are sized per-row × capacity (capacity = 1 at
    //    build; grown per fire by ensure_capacity).
    const std::size_t n_buf = dag.buffers.size();
    prog.buffer_ptrs_.assign(n_buf, 0);
    prog.buffer_sizes_.assign(n_buf, 0);
    prog.buffer_owned_.assign(n_buf, false);
    prog.buffer_batched_.assign(n_buf, false);
    prog.capacity_ = 1;
    for (std::size_t slot = 0; slot < n_buf; ++slot) {
        const BufferDecl& b = dag.buffers[slot];
        if (prog.buffer_slot_.count(b.id)) {
            throw std::runtime_error("sampling_ir::JIT: duplicate BufferId " +
                                     std::to_string(b.id));
        }
        prog.buffer_slot_[b.id] = slot;
        prog.buffer_sizes_[slot] = b.size_bytes;  // per-row if batched
        prog.buffer_batched_[slot] = b.batched;
        if (!b.external && b.size_bytes > 0) {
            const std::size_t bytes =
                b.batched ? b.size_bytes * prog.capacity_ : b.size_bytes;
            CUdeviceptr ptr = 0;
            CU_CHECK(cuMemAlloc(&ptr, bytes));
            prog.buffer_ptrs_[slot] = ptr;
            prog.buffer_owned_[slot] = true;
        }
    }

    // 2. Load each pre-built PTX + resolve its entry, then build the launch
    //    param arrays. Buffer args point into the stable buffer table; scalar
    //    args point into per-kernel scalar backing storage. NO NVRTC here — the
    //    PTX was produced off-thread; this is the context-thread module load.
    prog.modules_.reserve(dag.kernels.size());
    prog.items_.reserve(dag.kernels.size());
    for (std::size_t ki = 0; ki < dag.kernels.size(); ++ki) {
        const KernelDef& k = dag.kernels[ki];
        CUmodule mod = nullptr;
        CU_CHECK(cuModuleLoadData(&mod, ptx[ki].c_str()));
        prog.modules_.push_back(mod);
        CUfunction fn = nullptr;
        CU_CHECK(cuModuleGetFunction(&fn, mod, k.name.c_str()));

        CompiledProgram::LaunchItem item;
        item.fn = fn;
        item.grid = k.grid;
        item.block = k.block;
        item.shared_bytes = k.shared_bytes;
        item.args = k.args;
        item.grid_shape = k.grid_shape;
        item.per_row_len = k.per_row_len;

        std::size_t n_scalar = 0;
        for (const KernelArg& a : k.args) {
            if (a.kind != ArgKind::Buffer) ++n_scalar;  // Scalar + Param slots
        }
        item.scalar_store.reserve(n_scalar);  // stable: no realloc after reserve

        item.params.reserve(k.args.size());
        for (const KernelArg& a : k.args) {
            if (a.kind == ArgKind::Buffer) {
                auto it = prog.buffer_slot_.find(a.buffer);
                if (it == prog.buffer_slot_.end()) {
                    throw std::runtime_error(
                        "sampling_ir::JIT: kernel '" + k.name +
                        "' references unknown BufferId " + std::to_string(a.buffer));
                }
                item.params.push_back(&prog.buffer_ptrs_[it->second]);
            } else {
                // Scalar (const immediate) or Param (per-fire, patched at launch).
                const std::size_t store_idx = item.scalar_store.size();
                item.scalar_store.push_back(
                    a.kind == ArgKind::Scalar ? a.scalar_bits : 0ULL);
                if (a.kind == ArgKind::Param) {
                    item.param_patches.emplace_back(a.param, store_idx);
                }
                item.params.push_back(&item.scalar_store[store_idx]);
            }
        }
        prog.items_.push_back(std::move(item));
    }
}

std::shared_future<JitEngine::PtxResult>
JitEngine::request_compile(std::uint64_t hash, const KernelDAG& dag) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = cache_.find(hash);
    if (it != cache_.end()) return it->second.ptx_fut;  // dedup: one compile per hash

    // Miss: create the Compiling entry + enqueue NVRTC PTX-gen on the pool. The
    // closure owns a shared_ptr<KernelDAG> copy (one per distinct program — not
    // per request) + the arch string by value; it touches NO CUDA context.
    auto dag_sp = std::make_shared<KernelDAG>(dag);
    auto promise = std::make_shared<std::promise<PtxResult>>();
    std::shared_future<PtxResult> fut = promise->get_future().share();

    CacheEntry e;
    e.state = CompileState::Compiling;
    e.ptx_fut = fut;
    cache_.emplace(hash, std::move(e));

    const std::string arch = arch_;  // immutable post-init; capture by value
    pool_.submit([this, dag_sp, promise, arch]() {
        PtxResult r;
        try {
            r.ptx.reserve(dag_sp->kernels.size());
            for (const KernelDef& k : dag_sp->kernels)
                r.ptx.push_back(compile_to_ptx(k.name, k.source, arch));
            r.ok = true;
        } catch (const std::exception& ex) {
            r.ok = false;
            r.error = ex.what();
        }
        // One pool task per DISTINCT program (request_compile dedups) → this
        // counts actual NVRTC runs (the dedup metric), success or failure.
        compiles_run_.fetch_add(1, std::memory_order_relaxed);
        promise->set_value(std::move(r));
    });
    return fut;
}

void JitEngine::prefetch_compile(const KernelDAG& dag) {
    // Fire-and-forget: kick PTX-gen off-thread (idempotent / dedup'd by hash).
    // NO finalize here — the module load happens lazily at first fire on the
    // context thread. Safe from any thread (touches no CUDA context).
    (void)request_compile(dag.hash, dag);
}

CompiledProgram& JitEngine::get_or_compile(const KernelDAG& dag) {
    // CONTEXT-THREAD ONLY: this is the sole finalizer (cuMemAlloc/cuModuleLoadData
    // in finalize()). The defensive Ready re-check below is belt-and-suspenders —
    // it does NOT make a true 2-context-thread call safe (double cuModuleLoadData);
    // the contract is that exactly one thread (the context owner) calls this.
    // Hot path: already finalized → O(1) cache hit. `*prog` is heap-stable (the
    // unique_ptr pointee survives a map rehash), so the returned ref is valid
    // after the lock is released.
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = cache_.find(dag.hash);
        if (it != cache_.end() && it->second.state == CompileState::Ready)
            return *it->second.prog;
    }

    // Ensure PTX-gen is in flight (dedup) and wait for it. Only the first
    // consumer truly blocks; concurrent requesters share the future.
    std::shared_future<PtxResult> fut = request_compile(dag.hash, dag);
    const PtxResult& r = fut.get();

    if (!r.ok) {
        // Failed: record + throw (loud, like the prior synchronous path). The
        // entry stays Failed so repeats fast-fail (no recompile), dedup'd.
        std::lock_guard<std::mutex> lk(mu_);
        CacheEntry& e = cache_[dag.hash];
        e.state = CompileState::Failed;
        e.error = r.error;
        throw std::runtime_error("sampling_ir::JIT compile failed: " + r.error);
    }

    // Defensive: another path may have finalized while we held no lock (only the
    // context thread finalizes, so this is belt-and-suspenders).
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = cache_.find(dag.hash);
        if (it != cache_.end() && it->second.state == CompileState::Ready)
            return *it->second.prog;
    }

    // Finalize on THIS (context) thread WITHOUT holding mu_ across the cu* calls
    // (the cache map is free for concurrent prefetch/request meanwhile).
    auto prog = std::make_unique<CompiledProgram>();
    finalize(dag, r.ptx, *prog);  // cuMemAlloc + cuModuleLoadData — mu_ NOT held
    CompiledProgram& ref = *prog;
    {
        std::lock_guard<std::mutex> lk(mu_);
        CacheEntry& e = cache_[dag.hash];
        e.prog = std::move(prog);
        e.state = CompileState::Ready;
    }
    return ref;
}

void JitEngine::bind_buffer(CompiledProgram& prog, BufferId id, CUdeviceptr ptr) {
    auto it = prog.buffer_slot_.find(id);
    if (it == prog.buffer_slot_.end()) {
        throw std::runtime_error("sampling_ir::JIT: bind_buffer unknown BufferId " +
                                 std::to_string(id));
    }
    const std::size_t slot = it->second;
    if (prog.buffer_owned_[slot] && prog.buffer_ptrs_[slot]) {
        // Replacing a JIT-owned allocation with an external one: free ours.
        cuMemFree(prog.buffer_ptrs_[slot]);
        prog.buffer_owned_[slot] = false;
    }
    prog.buffer_ptrs_[slot] = ptr;  // LaunchItem params already point at this slot
}

void JitEngine::ensure_capacity(CompiledProgram& prog, std::size_t num_rows) {
    if (num_rows <= prog.capacity_) return;
    for (std::size_t slot = 0; slot < prog.buffer_ptrs_.size(); ++slot) {
        if (!prog.buffer_batched_[slot] || !prog.buffer_owned_[slot]) continue;
        const std::size_t per_row = prog.buffer_sizes_[slot];
        if (per_row == 0) continue;
        if (prog.buffer_ptrs_[slot]) cuMemFree(prog.buffer_ptrs_[slot]);
        CUdeviceptr ptr = 0;
        CU_CHECK(cuMemAlloc(&ptr, per_row * num_rows));
        prog.buffer_ptrs_[slot] = ptr;  // launch params hold &buffer_ptrs_[slot]
    }
    prog.capacity_ = num_rows;
}

namespace {
unsigned grid_x_for(GridShape shape, const Dim3& fixed, unsigned block_x,
                    std::uint64_t num_rows, std::uint64_t vocab,
                    std::uint64_t per_row_len) {
    auto ceil_div = [](std::uint64_t a, std::uint64_t b) -> unsigned {
        return b == 0 ? 1u : static_cast<unsigned>((a + b - 1) / b);
    };
    const std::uint64_t rows = num_rows == 0 ? 1 : num_rows;
    switch (shape) {
        case GridShape::Fixed:
            return fixed.x;
        case GridShape::OneBlockPerRow:
            return static_cast<unsigned>(rows);
        case GridShape::GridStrideOverVocab:
            return ceil_div(rows * vocab, block_x);
        case GridShape::GridStrideOverLen:
            return ceil_div(rows * per_row_len, block_x);
    }
    return fixed.x;
}
}  // namespace

void JitEngine::issue_kernels(CompiledProgram& prog, CUstream stream,
                             const std::vector<std::uint64_t>& param_values) {
    const std::uint64_t num_rows =
        kParamNumRows < param_values.size() ? param_values[kParamNumRows] : 1;
    const std::uint64_t vocab =
        kParamVocab < param_values.size() ? param_values[kParamVocab] : 0;
    // Grow JIT-owned batched scratch to fit this fire before any launch.
    ensure_capacity(prog, num_rows == 0 ? 1 : static_cast<std::size_t>(num_rows));

    for (std::size_t slot = 0; slot < prog.buffer_ptrs_.size(); ++slot) {
        if (prog.buffer_sizes_[slot] > 0 && prog.buffer_ptrs_[slot] == 0) {
            throw std::runtime_error(
                "sampling_ir::JIT: launch with unbound buffer at slot " +
                std::to_string(slot));
        }
    }
    for (CompiledProgram::LaunchItem& item : prog.items_) {
        // Patch per-fire scalars (RNG offset, num_rows, …) into the param array.
        for (const auto& [param_id, store_idx] : item.param_patches) {
            if (param_id >= param_values.size()) {
                throw std::runtime_error(
                    "sampling_ir::JIT: launch missing value for Param id " +
                    std::to_string(param_id));
            }
            item.scalar_store[store_idx] = param_values[param_id];
        }
        // Recompute grid.x per fire for batched (M>1) kernels; Fixed keeps the
        // baked grid.
        const unsigned grid_x =
            grid_x_for(item.grid_shape, item.grid, item.block.x, num_rows, vocab,
                       item.per_row_len);
        CU_CHECK(cuLaunchKernel(
            item.fn, grid_x, item.grid.y, item.grid.z, item.block.x,
            item.block.y, item.block.z, item.shared_bytes, stream,
            item.params.empty() ? nullptr : item.params.data(), nullptr));
    }
}

void JitEngine::launch(CompiledProgram& prog, CUstream stream,
                       const std::vector<std::uint64_t>& param_values) {
    issue_kernels(prog, stream, param_values);
}

void JitEngine::instantiate_graph(CompiledProgram& prog, CUstream capture_stream,
                                  const std::vector<std::uint64_t>& param_values) {
    if (prog.graph_exec_) {
        cuGraphExecDestroy(prog.graph_exec_);
        prog.graph_exec_ = nullptr;
    }
    if (prog.graph_) {
        cuGraphDestroy(prog.graph_);
        prog.graph_ = nullptr;
    }
    CU_CHECK(cuStreamBeginCapture(capture_stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    // If capture fails mid-way we must still end it before propagating, else the
    // stream is left in capture state.
    try {
        issue_kernels(prog, capture_stream, param_values);
    } catch (...) {
        CUgraph dead = nullptr;
        cuStreamEndCapture(capture_stream, &dead);
        if (dead) cuGraphDestroy(dead);
        throw;
    }
    CU_CHECK(cuStreamEndCapture(capture_stream, &prog.graph_));
    CU_CHECK(cuGraphInstantiate(&prog.graph_exec_, prog.graph_, 0));
}

void JitEngine::launch_graph(CompiledProgram& prog, CUstream stream) {
    if (!prog.graph_exec_) {
        throw std::runtime_error(
            "sampling_ir::JIT: launch_graph before instantiate_graph");
    }
    CU_CHECK(cuGraphLaunch(prog.graph_exec_, stream));
}

}  // namespace pie_cuda_driver::sampling_ir::jit
