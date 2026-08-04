#pragma once
// decode_consts.hpp — delta's geometry-derived CONST-PARAM binder for the raw-Metal decode.
//
// Gap closed: the ported kernels take geometry constants (RmsParams, GdnCoreParams,
// GatedRmsParams, qmv in/out_vec, dense K/N, rope scale/base/head_dim, sdpa gqa/strides/
// scale, kv_append head_dim/strides, embed hidden, q_split head_dim) that are bound by
// NO other lane — not the weight registry (heap_bind), not beta's scratch schedule, not
// the KV/state/IO binds. Since alpha's `arg_bind_ordinal` is setAddress-only (no setBytes),
// an unbound `constant&` arg is a GPU fault, not a compile error.
//
// These values are token-INDEPENDENT (I2-safe: identical every step → CB stays byte-
// identical) with ONE exception, the mixture's routing, which is a function of the batch
// width -- see `bind_token_consts`. Even that stays I2-safe: rebinding rewrites a const
// slot's CONTENTS at an address the argument table already holds, so no encoded byte
// moves. Otherwise purely geometry-derived → delta's heap/geometry lane. This module
// walks
// beta's build_decode_dag and, for each dispatch, allocates a tiny resident slot holding
// the constant value(s) and binds it by ORDINAL at the kernel's const buffer index. Call
// AFTER stage_decode_weights + bind_decode_dag + bind_scratch, BEFORE make_resident.
//
// All struct layouts + buffer indices below are replicated EXACTLY from the ported
// kernels (rms_norm.metal, gdn_core.metal, gated_rms.metal, quantized_qmv.metal,
// rope.metal, sdpa_vector.metal, kv_append.metal, embed_gather.metal,
// attn_gate.metal q_gate_split) and the bind:: enums in decode_abi.hpp.

#include <vector>

#include "decode_abi.hpp"

namespace pie::metal {

class RawMetalContext;
struct Dispatch;  // beta: decode_step.hpp

// A projection's in/out vector lengths, from geometry (matches the staged
// weight shapes).  The split-K rule needs K, not just the output width.
struct KN { int K, N; };
KN qmv_kn(Kernel k, const DecodeGeometry& g);

// Allocate + fill + bind every const-param buffer for every dispatch in the DAG, by
// ordinal. `max_ctx` MUST match the value passed to plan_heap (KV cache stride basis).
// Returns the number of const slots allocated (for accounting / the heap budget check).
// Allocate + fill + bind every const-param buffer for every dispatch in the DAG, by
// ordinal. `max_ctx` MUST match the value passed to plan_heap (KV cache stride basis).
// `n_tokens` is the batch width this DAG will be fired at; every constant here is
// width-invariant except the mixture's routing, which `bind_token_consts` owns.
// Returns the number of const slots allocated (for accounting / the heap budget check).
int bind_decode_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                       const DecodeGeometry& g, int max_ctx, bool gdn_prep = false,
                       int n_tokens = 1);

// Rebind ONLY the constants the batch width can change -- the mixture's routing.
// `bind_decode_consts` calls this, so a freshly bound DAG is complete; the fire path
// calls it alone when the token count changes, which is allocation-free (const slots
// are cached by (ordinal, index) and overwritten in place).
int bind_token_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                      const DecodeGeometry& g, int n_tokens);

// Heap headroom (bytes) the const region needs on top of plan_heap().total. Conservative
// upper bound (one 256-aligned slot per possible const buffer across the whole DAG).
size_t decode_consts_budget(const std::vector<Dispatch>& dag);

}  // namespace pie::metal
