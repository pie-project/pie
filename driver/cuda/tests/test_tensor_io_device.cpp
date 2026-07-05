// #6 WS-L4 (echo) — standalone device de-risk for the tensor-I/O fast-path.
// Verifies, in isolation on a live GPU (no executor / no forward), the primitives
// the executor composes at foxtrot's WIT freeze:
//   * cut #2 INPUT  : arena device_alloc (co-located R12 flag) → write_async H2D +
//                     Model-A self-arm → clear-on-bind.
//   * cut #1 OUTPUT : arena pinned_alloc → eager-D2H read-cache (ordered after a
//                     "forward sample-done" event) → host memcpy == tensor.read.
//   * BOTH arenas   : free→re-alloc reuses the block with ZERO new backing syscall
//                     (proves @ingim's no-per-tensor-cudaMalloc / cudaHostAlloc).
// Exercises the real extern "C" FFI surface bravo consumes, plus the driver-
// internal engine hooks (eager_d2h_after / clear_flag).

#include "sampling_ir/tensor_io.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

using pie_cuda_driver::sampling_ir::TensorIoEngine;

#define CK(x)                                                                  \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA err %s @ %s:%d\n",                      \
                         cudaGetErrorString(e_), __FILE__, __LINE__);          \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define EXPECT(cond, msg)                                                      \
    do {                                                                       \
        if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); return 1; }  \
    } while (0)

static std::uint32_t read_u32_device(const std::uint32_t* d) {
    std::uint32_t h = 0xDEADBEEFu;
    cudaStreamSynchronize(TensorIoEngine::instance().copy_stream());
    cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost);
    return h;
}

// (a2) completion host-func: fires once the copy-stream eager-D2H has drained.
static void CUDART_CB completion_cb(void* ud) {
    *static_cast<int*>(ud) = 1;
}

int main() {
    const std::size_t N = 256;
    const std::size_t bytes = N * sizeof(float);

    // ── cut #2: INPUT late-channel ───────────────────────────────────────────
    void* d_dst = nullptr;
    std::uint32_t* d_flag = nullptr;
    pie_device_alloc(bytes, &d_dst, &d_flag);
    EXPECT(d_dst && d_flag, "device_alloc returned null");
    EXPECT(read_u32_device(d_flag) == 0u, "flag not CLEAR at construct");

    std::vector<float> src(N);
    for (std::size_t i = 0; i < N; ++i) src[i] = float(i) * 1.5f + 0.25f;
    void* w_ev = pie_tensor_write_async(d_dst, src.data(), bytes, d_flag);
    pie_event_sync(w_ev);

    std::vector<float> back(N, -1.0f);
    CK(cudaMemcpy(back.data(), d_dst, bytes, cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < N; ++i)
        EXPECT(back[i] == src[i], "H2D write mismatch");
    EXPECT(read_u32_device(d_flag) == 1u, "flag not READY after write (self-arm)");

    TensorIoEngine::instance().clear_flag(d_flag);
    EXPECT(read_u32_device(d_flag) == 0u, "flag not CLEAR after clear_flag");

    // ── cut #1: OUTPUT read-cache (eager-D2H ordered after a produced event) ──
    void* d_out = nullptr;
    CK(cudaMalloc(&d_out, bytes));  // stands in for the ws sampler-output buffer
    std::vector<float> outpat(N);
    for (std::size_t i = 0; i < N; ++i) outpat[i] = float(N - i) * 0.5f - 3.0f;
    cudaStream_t fwd;
    CK(cudaStreamCreateWithFlags(&fwd, cudaStreamNonBlocking));
    CK(cudaMemcpyAsync(d_out, outpat.data(), bytes, cudaMemcpyHostToDevice, fwd));
    cudaEvent_t produced;
    CK(cudaEventCreateWithFlags(&produced, cudaEventDisableTiming));
    CK(cudaEventRecord(produced, fwd));

    void* pinned = pie_pinned_alloc(bytes);
    EXPECT(pinned != nullptr, "pinned_alloc returned null");
    cudaEvent_t d2h_ev = TensorIoEngine::instance().eager_d2h_after(
        pinned, d_out, bytes, produced);
    pie_event_sync(static_cast<void*>(d2h_ev));
    const float* pf = static_cast<const float*>(pinned);
    for (std::size_t i = 0; i < N; ++i)
        EXPECT(pf[i] == outpat[i], "eager-D2H read-cache mismatch");

    // ── arena reuse: free → re-alloc reuses the block, no new backing syscall ──
    const std::size_t dev_calls0 = TensorIoEngine::instance().device_backing_alloc_calls();
    pie_device_free(d_dst, d_flag);
    void* d_dst2 = nullptr;
    std::uint32_t* d_flag2 = nullptr;
    pie_device_alloc(bytes, &d_dst2, &d_flag2);
    EXPECT(d_dst2 == d_dst, "device arena did not reuse the freed block");
    EXPECT(TensorIoEngine::instance().device_backing_alloc_calls() == dev_calls0,
           "device arena issued a new cudaMalloc on reuse");

    const std::size_t pin_calls0 = TensorIoEngine::instance().pinned_backing_alloc_calls();
    pie_pinned_free(pinned);
    void* pinned2 = pie_pinned_alloc(bytes);
    EXPECT(pinned2 == pinned, "pinned arena did not reuse the freed block");
    EXPECT(TensorIoEngine::instance().pinned_backing_alloc_calls() == pin_calls0,
           "pinned arena issued a new cudaHostAlloc on reuse");

    // ── cut #1 batch: eager_d2h_outputs (N program outputs → N pinned dsts) +
    //    enqueue_completion (the (a2) forward-done host-func fires post-D2H) ──
    {
        void* d_src0 = nullptr;
        void* d_src1 = nullptr;
        CK(cudaMalloc(&d_src0, bytes));
        CK(cudaMalloc(&d_src1, bytes));
        std::vector<float> pat0(N), pat1(N);
        for (std::size_t i = 0; i < N; ++i) {
            pat0[i] = float(i) + 0.5f;
            pat1[i] = float(i) * -2.0f;
        }
        cudaStream_t s;
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        CK(cudaMemcpyAsync(d_src0, pat0.data(), bytes, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(d_src1, pat1.data(), bytes, cudaMemcpyHostToDevice, s));
        cudaEvent_t sample_done;
        CK(cudaEventCreateWithFlags(&sample_done, cudaEventDisableTiming));
        CK(cudaEventRecord(sample_done, s));

        void* pdst0 = pie_pinned_alloc(bytes);
        void* pdst1 = pie_pinned_alloc(bytes);
        const std::uint64_t dptrs[2] = {reinterpret_cast<std::uint64_t>(pdst0),
                                        reinterpret_cast<std::uint64_t>(pdst1)};
        const std::uint32_t dlens[2] = {static_cast<std::uint32_t>(bytes),
                                        static_cast<std::uint32_t>(bytes)};
        const void* srcs[2] = {d_src0, d_src1};
        const std::size_t nbs[2] = {bytes, bytes};
        cudaEvent_t t_d2h = TensorIoEngine::instance().eager_d2h_outputs(
            dptrs, dlens, srcs, nbs, 2, sample_done);
        int fired = 0;
        TensorIoEngine::instance().enqueue_completion(completion_cb, &fired);
        CK(cudaStreamSynchronize(TensorIoEngine::instance().copy_stream()));
        EXPECT(fired == 1, "enqueue_completion host-func did not fire post-D2H");
        const float* f0 = static_cast<const float*>(pdst0);
        const float* f1 = static_cast<const float*>(pdst1);
        for (std::size_t i = 0; i < N; ++i) {
            EXPECT(f0[i] == pat0[i], "batch eager-D2H output 0 mismatch");
            EXPECT(f1[i] == pat1[i], "batch eager-D2H output 1 mismatch");
        }
        CK(cudaEventDestroy(t_d2h));
        CK(cudaEventDestroy(sample_done));
        CK(cudaStreamDestroy(s));
        CK(cudaFree(d_src0));
        CK(cudaFree(d_src1));
        pie_pinned_free(pdst0);
        pie_pinned_free(pdst1);
    }

    pie_device_free(d_dst2, d_flag2);
    pie_pinned_free(pinned2);
    CK(cudaEventDestroy(produced));
    CK(cudaStreamDestroy(fwd));
    CK(cudaFree(d_out));

    std::printf(
        "TENSOR_IO_OK MATCH=true "
        "(write+self-arm, eager-D2H read-cache, clear-on-bind, "
        "device+pinned arena reuse: %zu/%zu backing syscalls)\n",
        TensorIoEngine::instance().device_backing_alloc_calls(),
        TensorIoEngine::instance().pinned_backing_alloc_calls());
    return 0;
}
