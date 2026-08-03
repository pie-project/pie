#pragma once

// batch/: the unionized supergraph's capture scaffolding (S2 of the
// supergraph ladder, north-star-dsl.md "The supergraph directive").
//
// A supergraph is ONE captured CUDA graph per (R, N) bucket whose
// attachment branches — the declared trace's Guard vocabulary — are
// conditional IF/ELSE nodes. The predicates live in a DEVICE-resident
// word the replay path updates per fire; a graph-embedded kernel
// (`supergraph_set_cond`) reads a slot and arms the conditional handle,
// so a replay takes the fire's arms with no host round-trip and no
// recapture. tools/supergraph-poc proves the primitives on this
// deployment (CUDA 13.0, driver 580, sm_89).
//
// The builder wraps the CUDA 12.4+ capture-time conditional dance:
//   open_cond():  handle on the CURRENT capturing graph (works at any
//                 nesting depth — GetCaptureInfo answers for whichever
//                 stream is capturing), the set_cond kernel, the
//                 conditional node inserted with the running deps;
//   begin_body(): BeginCaptureToGraph on the next pooled depth stream;
//   end_body():   EndCapture of that stream;
//   close_cond(): the outer stream's capture deps collapse onto the
//                 conditional node, so post-branch work follows it.
//
// Guard nesting (A1: the walk keeps a stack) maps to a stack of body
// captures over the depth-indexed stream pool. The generated
// `..._supergraph_build` functions (emitter mode, S2) call exactly this
// surface; nothing here knows about models.

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace pie_cuda_driver::batch {

// Predicate slots in the device word, by the trace's wire pred kinds
// (forward/src/trace.rs GuardPred::wire): HasWriteDesc=0, TokensLE=1,
// TokensGT=2, WantsAttnScore=3, HasCustomMask=4, (5 reserved),
// HasLora=6. Slots 7+: the Peel endpoint bits (S4 note: a MIXED hooked
// fire — 0 < fast_rows < N — cannot replay a baked window; eligibility
// keeps it eager until row windows learn to read device memory).
inline constexpr int kSupergraphPredSlots = 9;
inline constexpr int kPredSlotPeelAllFast = 7;  // fast_rows == N
inline constexpr int kPredSlotPeelAllHooked = 8;  // fast_rows == 0

// Launch the graph-embedded predicate reader: arms `handle` from
// `preds_d[slot]` on `stream` (which must be capturing — the launch
// becomes the conditional's upstream node). Implemented in
// supergraph.cu (needs a __global__).
void launch_supergraph_set_cond(cudaGraphConditionalHandle handle,
                                const std::uint8_t* preds_d,
                                int slot,
                                cudaStream_t stream);

class SupergraphBuilder {
public:
    struct Cond {
        cudaGraphNode_t node = nullptr;
        cudaGraph_t if_body = nullptr;
        cudaGraph_t else_body = nullptr;  // null unless with_else
    };

    // `capture_stream` must already be inside cudaStreamBeginCapture.
    // `preds_d` is the fire-updated device predicate word.
    SupergraphBuilder(cudaStream_t capture_stream,
                      const std::uint8_t* preds_d);
    ~SupergraphBuilder();

    SupergraphBuilder(const SupergraphBuilder&) = delete;
    SupergraphBuilder& operator=(const SupergraphBuilder&) = delete;

    // The stream launches should currently target: the root capture
    // stream at depth 0, the innermost body stream inside a body.
    cudaStream_t stream() const;

    // Insert a conditional keyed on `pred_slot` at the current capture
    // position. Arms are then filled via begin_body/end_body on
    // `if_body` (and `else_body` when `with_else`).
    Cond open_cond(int pred_slot, bool with_else);

    // Capture into an arm's body graph (push) / finish it (pop).
    void begin_body(cudaGraph_t body);
    void end_body();

    // Collapse the current stream's capture dependencies onto the
    // conditional node so subsequent work depends on the whole branch.
    void close_cond(const Cond& cond);

private:
    cudaStream_t root_;
    const std::uint8_t* preds_d_;
    // Depth-indexed pool for body captures (guard nesting is shallow —
    // the A1 walk's stack depth; grown on demand).
    std::vector<cudaStream_t> body_streams_;
    std::vector<cudaStream_t> active_;  // capture stack, innermost last
};

}  // namespace pie_cuda_driver::batch
