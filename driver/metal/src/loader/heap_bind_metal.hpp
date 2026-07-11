#pragma once
// heap_bind_metal.hpp — the Metal-side staging/bind surface (declarations) for decode_run.
//
// heap_bind.hpp stays PURE (no Metal) so heap_bind_probe can verify the name registry
// against the real checkpoint with zero Metal deps. The actual heap_alloc/memcpy/arg_bind
// surface (which needs alpha's RawMetalContext + SlotHandle) lives here, included by
// heap_bind.cpp (definitions) and decode_run.cpp (the headline driver).

#include <unordered_map>
#include <vector>
#include <string>

#include "decode_abi.hpp"
#include "heap_layout.hpp"      // HeapPlan
#include "mtl4_context.hpp"     // RawMetalContext, SlotHandle

namespace pie::metal {

class SafetensorsView;
struct Dispatch;  // beta: decode_step.hpp

// Every slot delta allocates: the bind pass + the beta scratch handoff read from this.
struct BoundDecode {
    HeapPlan plan;

    // load-once weights, keyed by HF tensor name (tied lm_head appears once).
    std::unordered_map<std::string, SlotHandle> weights;

    // GDN persistent state, per layer (only GDN layers populated).
    struct GdnState { SlotHandle conv_state, conv_state_out, recurrent_state, conv_bias_zero; };
    std::vector<GdnState> gdn;        // size n_layers; full-attn entries unused

    // paged KV, per layer (only full-attn layers populated).
    struct KvSlots { SlotHandle k_pages, v_pages; };
    std::vector<KvSlots> kv;          // size n_layers; GDN entries unused

    // IO region (I1 per-token buffers + I3 logits). Indexed by IoSlot.
    SlotHandle io[kIoSlotCount];

    // device-argmax substrate (I3; allocated always, bound + read only when with_argmax):
    // argmax_params = ArgmaxParams const (vocab + inline EOS ids; Shared, runtime-writable),
    // eos_flag = u32[max_tokens] OUT (1 if NextToken[r] ∈ eos_ids). Inert at M=1/no-argmax.
    SlotHandle argmax_params;
    SlotHandle eos_flag;

    // activation ping-pong pool handed to beta (he assigns X/Out per dispatch).
    SlotHandle scratch[SCRATCH_POOL];
};

// Stage all weights/state/KV/IO/scratch into the single resident heap (allocation order
// follows the region plan). The view must outlive nothing (zero-copy mmap is read here).
BoundDecode stage_decode_weights(RawMetalContext& ctx, const SafetensorsView& view,
                                 const DecodeGeometry& g, const HeapPlan& plan);

// Walk beta's DAG; bind delta's weight/state/KV/IO slots for each dispatch by ordinal.
void bind_decode_dag(RawMetalContext& ctx, const BoundDecode& b,
                     const std::vector<Dispatch>& dag, const DecodeGeometry& g,
                     bool gdn_prep = false);

}  // namespace pie::metal
