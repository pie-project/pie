// SupergraphBuilder smoke: one capture, IF/ELSE + a NESTED conditional,
// four predicate combinations replayed through the single exec. The
// in-tree restatement of tools/supergraph-poc against the real builder.

#include <cstdio>
#include <cstdint>

#include <cuda_runtime.h>

#include "batch/supergraph.hpp"
#include "cuda_check.hpp"

using pie_cuda_driver::batch::SupergraphBuilder;

namespace {

__global__ void add_k(float* x, float v) { x[0] += v; }

float run_once(cudaGraphExec_t exec, cudaStream_t s, std::uint8_t* preds_d,
               std::uint8_t p0, std::uint8_t p1, float* buf) {
    const std::uint8_t host[2] = {p0, p1};
    CUDA_CHECK(cudaMemset(buf, 0, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(preds_d, host, 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaGraphLaunch(exec, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    float out = 0.f;
    CUDA_CHECK(cudaMemcpy(&out, buf, sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

}  // namespace

int main() {
    float* buf = nullptr;
    std::uint8_t* preds_d = nullptr;
    CUDA_CHECK(cudaMalloc(&buf, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&preds_d, 2));
    cudaStream_t s{};
    CUDA_CHECK(cudaStreamCreate(&s));

    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    cudaGraph_t graph = nullptr;
    {
        SupergraphBuilder b(s, preds_d);
        add_k<<<1, 1, 0, b.stream()>>>(buf, 1.f);

        auto outer = b.open_cond(/*pred_slot=*/0, /*with_else=*/true);
        b.begin_body(outer.if_body);
        {
            add_k<<<1, 1, 0, b.stream()>>>(buf, 10.f);
            // Nested conditional INSIDE the if-arm (the A1 stack shape).
            auto inner = b.open_cond(/*pred_slot=*/1, /*with_else=*/false);
            b.begin_body(inner.if_body);
            add_k<<<1, 1, 0, b.stream()>>>(buf, 100.f);
            b.end_body();
            b.close_cond(inner);
            add_k<<<1, 1, 0, b.stream()>>>(buf, 1000.f);
        }
        b.end_body();
        b.begin_body(outer.else_body);
        add_k<<<1, 1, 0, b.stream()>>>(buf, 100000.f);
        b.end_body();
        b.close_cond(outer);
        add_k<<<1, 1, 0, b.stream()>>>(buf, 10000.f);
    }
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    cudaGraphExec_t exec{};
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    struct Case {
        std::uint8_t p0, p1;
        float expect;
    };
    // base 1 + tail 10000 always; if-arm adds 10 + (p1 ? 100 : 0) + 1000;
    // else-arm adds 100000.
    const Case cases[] = {
        {0, 0, 1.f + 100000.f + 10000.f},
        {0, 1, 1.f + 100000.f + 10000.f},
        {1, 0, 1.f + 10.f + 1000.f + 10000.f},
        {1, 1, 1.f + 10.f + 100.f + 1000.f + 10000.f},
    };
    bool ok = true;
    for (const Case& c : cases) {
        const float got = run_once(exec, s, preds_d, c.p0, c.p1, buf);
        std::printf("preds=(%d,%d): got %.0f expect %.0f\n", c.p0, c.p1, got,
                    c.expect);
        ok = ok && got == c.expect;
    }
    std::printf("%s\n", ok ? "SUPERGRAPH-BUILDER-OK" : "BUILDER-MISMATCH");
    return ok ? 0 : 1;
}
