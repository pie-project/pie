// The supergraph capture scaffolding (S2) — see supergraph.hpp.

#include "batch/supergraph.hpp"

#include <stdexcept>

#include "cuda_check.hpp"

namespace pie_cuda_driver::batch {

namespace {

__global__ void supergraph_set_cond_kernel(cudaGraphConditionalHandle h,
                                           const std::uint8_t* preds,
                                           int slot) {
    cudaGraphSetConditional(h, preds[slot]);
}

}  // namespace

void launch_supergraph_set_cond(cudaGraphConditionalHandle handle,
                                const std::uint8_t* preds_d,
                                int slot,
                                cudaStream_t stream) {
    supergraph_set_cond_kernel<<<1, 1, 0, stream>>>(handle, preds_d, slot);
    CUDA_CHECK(cudaGetLastError());
}

SupergraphBuilder::SupergraphBuilder(cudaStream_t capture_stream,
                                     const std::uint8_t* preds_d)
    : root_(capture_stream), preds_d_(preds_d) {
    active_.push_back(root_);
}

SupergraphBuilder::~SupergraphBuilder() {
    for (cudaStream_t s : body_streams_) {
        cudaStreamDestroy(s);  // best-effort in a destructor
    }
}

cudaStream_t SupergraphBuilder::stream() const { return active_.back(); }

SupergraphBuilder::Cond SupergraphBuilder::open_cond(int pred_slot,
                                                     bool with_else) {
    if (pred_slot < 0 || pred_slot >= kSupergraphPredSlots) {
        throw std::runtime_error("supergraph: pred slot out of range");
    }
    const cudaStream_t s = stream();
    // The handle belongs to whichever graph this stream is capturing —
    // the root graph at depth 0, an arm's body graph when nested.
    cudaStreamCaptureStatus st{};
    cudaGraph_t capturing = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    size_t ndeps = 0;
    CUDA_CHECK(cudaStreamGetCaptureInfo(s, &st, nullptr, &capturing, &deps,
                                        nullptr, &ndeps));
    if (st != cudaStreamCaptureStatusActive) {
        throw std::runtime_error("supergraph: open_cond outside a capture");
    }
    cudaGraphConditionalHandle handle{};
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, capturing, 0, 0));
    launch_supergraph_set_cond(handle, preds_d_, pred_slot, s);

    CUDA_CHECK(cudaStreamGetCaptureInfo(s, &st, nullptr, &capturing, &deps,
                                        nullptr, &ndeps));
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = with_else ? 2 : 1;
    Cond out;
    CUDA_CHECK(
        cudaGraphAddNode(&out.node, capturing, deps, nullptr, ndeps, &params));
    out.if_body = params.conditional.phGraph_out[0];
    if (with_else) out.else_body = params.conditional.phGraph_out[1];
    return out;
}

void SupergraphBuilder::begin_body(cudaGraph_t body) {
    const std::size_t depth = active_.size() - 1;  // bodies opened so far
    if (depth >= body_streams_.size()) {
        cudaStream_t s{};
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        body_streams_.push_back(s);
    }
    cudaStream_t s = body_streams_[depth];
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(s, body, nullptr, nullptr, 0,
                                             cudaStreamCaptureModeGlobal));
    active_.push_back(s);
}

void SupergraphBuilder::end_body() {
    if (active_.size() <= 1) {
        throw std::runtime_error("supergraph: end_body underflow");
    }
    cudaGraph_t out = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(active_.back(), &out));
    active_.pop_back();
}

void SupergraphBuilder::close_cond(const Cond& cond) {
    CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
        stream(), const_cast<cudaGraphNode_t*>(&cond.node), nullptr, 1,
        cudaStreamSetCaptureDependencies));
}

}  // namespace pie_cuda_driver::batch
