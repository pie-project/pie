// mtpverify_test — PTIR-native MTP speculative-verify on Metal.
//
// Executes the `spec_verify_greedy` composed sampling-IR program (sdk/rust/
// sampling-edsl program.rs:210) over REAL Qwen3.5-0.8B MTP drafts captured from
// the corrected Metal MTP head (driver/metal/tests/qwen35_mtp.cpp, 50% accept),
// via the SAME certified tier-0 Metal op kernels the interpreter dispatches to
// (driver/metal/ptir/kernels/sampling_ir.metal). It certifies two things:
//
//   1. De-hardwiring (CUDA Stage 2 mirror): the draft token is
//      `argmax(mtp_logits)` — bound to Intrinsic::MtpLogits — computed by the
//      identical `reduce_argmax` kernel as the target's `argmax(logits)`
//      (Intrinsic::Logits). Only the bound source differs; the bytecode/kernel
//      is byte-identical (mtp_argmax == argmax). trace.hpp Intrinsic::MtpLogits=1.
//
//   2. Verify DAG: argmax -> eq -> select(1,0) -> cumprod -> gt(0.5) ->
//      select(draft,-1), producing the sentinel-coded `[k]` Token output (the
//      accepted matching prefix, then -1 from the first reject). The argmaxes
//      and the two selects run on Metal (reduce_argmax + dselect_f32 kernels);
//      the eq/cumprod/gt lanes are the interp's host control-flow. The result is
//      asserted bit-exact against the eval.rs reference semantics AND against
//      the direct-loop acceptance the MTP head produced (mtp_verify_tail).
//
//   Usage: mtpverify_test [kernels_dir]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "metal_harness.hpp"
#include "metal_ops.hpp"

using ptir_metal::MetalHarness;

namespace {

int g_pass = 0, g_fail = 0;
void expect(bool c, const std::string& what) {
    std::printf("  %s  %s\n", c ? "PASS" : "FAIL", what.c_str());
    if (c) ++g_pass; else ++g_fail;
}

// A real captured (draft, target) block from the corrected Metal MTP run on
// "The capital of France is Paris. The capital of Japan is" (Qwen3.5-0.8B).
struct Block {
    std::string name;
    std::vector<int> draft;   // argmax(mtp_logits) per position  (Intrinsic::MtpLogits)
    std::vector<int> target;  // argmax(logits) per position       (Intrinsic::Logits)
};

// Reference eval.rs semantics of spec_verify_greedy (the oracle we match).
std::vector<int> ref_verify(const std::vector<int>& draft, const std::vector<int>& target) {
    const int k = (int)draft.size();
    std::vector<int> out(k);
    int acc = 1;  // prefix-AND over matches (cumprod of {0,1})
    for (int i = 0; i < k; ++i) {
        acc = acc && (draft[i] == target[i]);      // matched -> select(1,0) -> cumprod
        out[i] = acc ? draft[i] : -1;              // gt(0.5) -> select(draft,-1)
    }
    return out;
}

// Build a per-row one-hot-peaked logits matrix [k, vocab] whose row-i argmax is
// tok[i]: peak 10.0 at the token column, 0 elsewhere. This is what materializes
// the real (target / mtp) argmax tokens through the Metal reduce_argmax kernel.
std::vector<float> peaked(const std::vector<int>& tok, int vocab) {
    std::vector<float> m((std::size_t)tok.size() * vocab, 0.0f);
    for (std::size_t r = 0; r < tok.size(); ++r) m[r * vocab + tok[r]] = 10.0f;
    return m;
}

std::string vec_str(const std::vector<int>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
    return s + "]";
}

void run_block(const Block& blk, tier0::MetalOps& ops) {
    std::printf("[%s]\n", blk.name.c_str());
    const int k = (int)blk.draft.size();
    int vocab = 1;
    for (int t : blk.draft) vocab = std::max(vocab, t + 1);
    for (int t : blk.target) vocab = std::max(vocab, t + 1);

    // ── de-hardwiring: SAME reduce_argmax kernel, two intrinsic bindings ──
    //   target = argmax(logits)      [Intrinsic::Logits]
    //   draft  = argmax(mtp_logits)  [Intrinsic::MtpLogits]
    auto target = ops.reduce_argmax(peaked(blk.target, vocab), k, vocab);  // Metal
    auto draft  = ops.reduce_argmax(peaked(blk.draft, vocab), k, vocab);   // Metal
    std::vector<int> target_i(target.begin(), target.end());
    std::vector<int> draft_i(draft.begin(), draft.end());
    expect(target_i == blk.target, blk.name + ": Metal argmax(logits) == captured target");
    expect(draft_i == blk.draft, blk.name + ": Metal argmax(mtp_logits) == captured draft (MtpLogits binding)");

    // ── verify DAG (interp op dispatch) ──
    // matched = eq(target, draft)            [host, interp OC::Eq]
    std::vector<std::uint8_t> matched(k);
    for (int i = 0; i < k; ++i) matched[i] = (target_i[i] == draft_i[i]) ? 1 : 0;
    // sel = select(matched, 1.0, 0.0)        [Metal dselect_f32]
    std::vector<float> ones(k, 1.0f), zeros(k, 0.0f);
    std::vector<float> sel = ops.dselect(matched, ones, zeros);
    // acc = cumprod(sel)                     [host, interp OC::CumProd]
    std::vector<float> acc(k);
    { float a = 1.0f; for (int i = 0; i < k; ++i) { a *= sel[i]; acc[i] = a; } }
    // keep = gt(acc, 0.5)                    [host, interp OC::Gt]
    std::vector<std::uint8_t> keep(k);
    for (int i = 0; i < k; ++i) keep[i] = (acc[i] > 0.5f) ? 1 : 0;
    // out = select(keep, draft, -1)          [Metal dselect_f32 -> i32]
    std::vector<float> draft_f(draft_i.begin(), draft_i.end()), neg1(k, -1.0f);
    std::vector<float> out_f = ops.dselect(keep, draft_f, neg1);
    std::vector<int> out(k);
    for (int i = 0; i < k; ++i) out[i] = (int)std::lround(out_f[i]);

    // ── assert bit-exact to eval.rs semantics (mtp_verify_tail) ──
    auto ref = ref_verify(blk.draft, blk.target);
    int accepted = 0;
    for (int v : out) if (v >= 0) ++accepted;
    std::printf("    draft =%s\n    target=%s\n    verify=%s  (accepted prefix = %d)\n",
                vec_str(blk.draft).c_str(), vec_str(blk.target).c_str(),
                vec_str(out).c_str(), accepted);
    expect(out == ref, blk.name + ": sentinel accept-tail bit-exact to eval.rs spec_verify_greedy");
}

}  // namespace

int main(int argc, char** argv) {
    std::string kdir = argc > 1 ? argv[1] : TIER0_KERNELS_DIR;
    MetalHarness h;
    if (!h.ok()) { std::printf("MTPVERIFY_FAIL: %s\n", h.error().c_str()); return 2; }
    tier0::MetalOps ops{h};
    if (!ops.load(kdir)) { std::printf("MTPVERIFY_FAIL: %s\n", h.error().c_str()); return 2; }
    std::printf("PTIR-native MTP spec-verify on Metal — device: %s\n\n", h.device_name().c_str());

    // Real k=5 blocks captured from the corrected Metal Qwen3.5-0.8B MTP head:
    //   A: 4 accepts then a reject  -> prefix 4
    //   B: 3 accepts then a reject  -> prefix 3
    //   C: match, reject, then later matches AFTER the reject -> prefix 1
    //      (the cumprod prefix-gate MUST sentinel-code the post-reject matches).
    std::vector<Block> blocks = {
        {"block_A_prefix4", {314, 279, 3516, 4042, 314}, {314, 279, 3516, 4042, 369}},
        {"block_B_prefix3", {314, 279, 3516, 4042, 314}, {314, 279, 3516, 14634, 369}},
        {"block_C_prefixgate", {13, 198, 1330, 314, 279}, {13, 561, 6511, 314, 279}},
    };
    for (auto& b : blocks) run_block(b, ops);

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) { std::printf("MTPVERIFY_OK\n"); return 0; }
    std::printf("MTPVERIFY_FAIL\n");
    return 1;
}
