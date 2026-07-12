#pragma once
// heap_bind.hpp — delta's weight-staging + arg-table binding for the raw-Metal decode.
//
// Runtime canonical weight names are emitted by the Rust storage compiler.
// This header only maps kernel operand slots to those stable runtime names;
// checkpoint naming and quantization knowledge do not live in the driver.
//        bind_decode_dag(...)             -> walk beta's build_decode_dag, bind each
//                                            dispatch's weight/state/KV/IO slots by ORDINAL
//                                            (beta separately binds the X/Out scratch).
//
// Ownership seam (manager): delta lays out the heap + binds the LOAD-ONCE weights, the
// persistent GDN state, the KV pages, and the IO scalars. beta binds the per-dispatch
// activation/scratch X/Out (his WAR/WAW ping-pong schedule) over the SAME ordinals.
//
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "decode_abi.hpp"

namespace pie::metal {

// ── 1. Pure name registry (no Metal) ─────────────────────────────────────────

// One load-once weight a dispatch binds: which kernel bind-index <- which HF tensor.
struct WeightBind {
    uint8_t     bind_index;  // bind::<Kind> slot
    std::string tensor;      // HF checkpoint tensor name
};

// The load-once WEIGHT tensors a single dispatch (kind at `layer`) binds. `layer` = -1
// for singletons (EmbedGather/FinalRms/QmvLmHead/Argmax). Returns empty for weight-less
// kinds (QSplit/Rope/RopeK/Sdpa-Q/AttnGate/SiluMul/Residual/KvAppend/Argmax) — their
// non-weight slots (scratch X/Out, KV pages, IO scalars, const params) are bound
// elsewhere (beta's scratch; delta's KV/IO in bind_decode_dag).
std::vector<WeightBind> weight_binds(Kernel kind, int layer, const DecodeGeometry& g,
                                     bool gdn_prep = false);

}  // namespace pie::metal
