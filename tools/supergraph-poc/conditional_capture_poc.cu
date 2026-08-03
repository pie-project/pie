// Supergraph PoC: ONE captured CUDA graph whose branch is a conditional
// node driven by DEVICE-resident predicate, replayed with both values.
#include <cuda_runtime.h>
#include <cstdio>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("ERR %s line %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); return 1; } } while(0)

__global__ void set_cond_from_device(cudaGraphConditionalHandle h,
                                     const unsigned char* pred) {
    cudaGraphSetConditional(h, pred[0]);
}
__global__ void add_k(float* x, float v) { x[0] += v; }

int main() {
    float* buf; unsigned char* pred;
    CK(cudaMalloc(&buf, 4)); CK(cudaMalloc(&pred, 1));
    CK(cudaMemset(buf, 0, 4));
    cudaStream_t s, s2;
    CK(cudaStreamCreate(&s)); CK(cudaStreamCreate(&s2));

    cudaGraph_t graph;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    add_k<<<1,1,0,s>>>(buf, 1.0f);

    cudaStreamCaptureStatus st; cudaGraph_t cg;
    const cudaGraphNode_t* deps; size_t ndeps;
    CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &cg, &deps, nullptr, &ndeps));
    cudaGraphConditionalHandle h;
    CK(cudaGraphConditionalHandleCreate(&h, cg, 0, 0));
    set_cond_from_device<<<1,1,0,s>>>(h, pred);

    CK(cudaStreamGetCaptureInfo(s, &st, nullptr, &cg, &deps, nullptr, &ndeps));
    cudaGraphNodeParams np = {};
    np.type = cudaGraphNodeTypeConditional;
    np.conditional.handle = h;
    np.conditional.type = cudaGraphCondTypeIf;
    np.conditional.size = 2;  // IF/ELSE two-body form (CUDA 12.8+)
    cudaGraphNode_t cond_node;
    CK(cudaGraphAddNode(&cond_node, cg, deps, nullptr, ndeps, &np));

    // IF body: +10
    CK(cudaStreamBeginCaptureToGraph(s2, np.conditional.phGraph_out[0],
                                     nullptr, nullptr, 0,
                                     cudaStreamCaptureModeGlobal));
    add_k<<<1,1,0,s2>>>(buf, 10.0f);
    cudaGraph_t out0; CK(cudaStreamEndCapture(s2, &out0));
    // ELSE body: +1000
    CK(cudaStreamBeginCaptureToGraph(s2, np.conditional.phGraph_out[1],
                                     nullptr, nullptr, 0,
                                     cudaStreamCaptureModeGlobal));
    add_k<<<1,1,0,s2>>>(buf, 1000.0f);
    cudaGraph_t out1; CK(cudaStreamEndCapture(s2, &out1));

    CK(cudaStreamUpdateCaptureDependencies(s, &cond_node, nullptr, 1,
                                           cudaStreamSetCaptureDependencies));
    add_k<<<1,1,0,s>>>(buf, 100.0f);
    CK(cudaStreamEndCapture(s, &graph));

    cudaGraphExec_t exec;
    CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    unsigned char v;
    v = 0; CK(cudaMemcpy(pred, &v, 1, cudaMemcpyHostToDevice));
    CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
    float r0; CK(cudaMemcpy(&r0, buf, 4, cudaMemcpyDeviceToHost));
    v = 1; CK(cudaMemcpy(pred, &v, 1, cudaMemcpyHostToDevice));
    CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
    float r1; CK(cudaMemcpy(&r1, buf, 4, cudaMemcpyDeviceToHost));

    // replay1 (pred=0): 1 + 1000 + 100 = 1101
    // replay2 (pred=1): +1 +10 +100    = 1212 total
    printf("replay pred=0: %.0f (expect 1101)\n", r0);
    printf("replay pred=1 cumulative: %.0f (expect 1212)\n", r1);
    printf("%s\n", (r0 == 1101.f && r1 == 1212.f) ? "SUPERGRAPH-POC-OK"
                                                  : "POC-MISMATCH");
    return 0;
}
