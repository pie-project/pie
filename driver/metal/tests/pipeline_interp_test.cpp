// PTIR host-interpreter conformance + classification gate (metal_ptir_plan.md
// Phase 1, G1.1 / G1.5). Pure host, no Metal/Apple/checkpoint dependency —
// this binary always builds and runs (unlike `metal_direct_stub_test` it
// needs no `PIE_METAL_BUILD_STUB_TESTS` gate).
//
// Two halves:
//   1. Injected-logits golden-vector replay: decodes echo's ACTUAL container
//      + PTIB sidecar bytes (the same cross-language vectors the CUDA driver
//      pins against, interface/ptir/tests/golden-ptir/*.txt — vendored
//      inline as hex, transcribed byte-for-byte from those files) and drives
//      them through `pie::metal::pipeline::step()` with a `PassInputs`
//      binding real logits, matching interp.rs's canonical `take`/`committed`
//      results exactly (greedy_argmax, section3_masked_gumbel [temperature +
//      grammar-mask + gumbel], mtp_verify_tail [Stage-2 MtpLogits K-row
//      matrix read], matrix_mask_apply_packed).
//   2. Rejection honesty (G1.5): `classify_exec_plan` over hand-built traces
//      proves HostInput / per-layer taps / unsupported intrinsics
//      (hidden/query/value-head) still hard-reject with a precise reason,
//      while Intrinsic(Logits)/Intrinsic(MtpLogits) roots are `executable`
//      with the matching `needs_logits`/`needs_mtp_logits` flag set.

#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "pipeline/interp.hpp"

using namespace pie::metal::pipeline;

namespace {

int g_pass = 0, g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}

std::vector<std::uint8_t> hex_to_bytes(const std::string& h) {
    std::vector<std::uint8_t> b;
    for (std::size_t i = 0; i + 1 < h.size(); i += 2)
        b.push_back(static_cast<std::uint8_t>(std::stoul(h.substr(i, 2), nullptr, 16)));
    return b;
}

bool build(const std::string& name, const std::string& chex, const std::string& shex,
          ExecPlan& out) {
    auto cb = hex_to_bytes(chex), sb = hex_to_bytes(shex);
    std::string err;
    if (!build_exec_plan(cb.data(), cb.size(), sb.data(), sb.size(), out, &err)) {
        expect(false, name + ": build_exec_plan (" + err + ")");
        return false;
    }
    return true;
}

// ── greedy_argmax: chan0 seeded embed token [1]; argmax(logits), V=8. ──
void run_greedy_argmax() {
    std::printf("[greedy_argmax]\n");
    ExecPlan plan;
    if (!build("greedy_argmax", "5054495201000000000000000200000001000000010000000101010000000100000000010101010000000100000002000000000000000306000000a000000002010000000800000039000000000108000000330100000039020000000101000000920000000003000000920100000003000000", "5054494201000000fe598742954369ff0200000000000200000000000000ff00010000000301010000000304000000000201000000080000000001080000000100010101000000", plan)) return;
    expect(plan.executable && plan.needs_logits && !plan.needs_mtp_logits,
          "classified executable, needs_logits (no reject)");

    std::map<std::uint32_t, std::shared_ptr<ChannelState>> externs;
    std::map<std::uint32_t, Value> seeds{{0, Value::i32({1})}};
    InterpInstance inst = make_instance(plan, externs, seeds);

    auto fire = [&](std::vector<float> logits, std::int32_t want) {
        PassInputs in;
        in.logits = logits.data();
        in.rows = 1;
        in.vocab = 8;
        StepResult r = step(inst, plan, in);
        Value v;
        const HostOp rc = host_take(inst, plan, 1, v);
        expect(r.ok && r.committed && rc == HostOp::Ok && !v.i.empty() && v.i[0] == want,
              "token == " + std::to_string(want) + " (committed=" +
                  (r.committed ? "T" : "F") + ")");
    };
    fire({0, 1, 9, 2, 0, 0, 0, 3}, 2);  // echo step 0
    fire({7, 1, 0, 2, 0, 0, 0, 3}, 0);  // echo step 1
}

// ── section3_masked_gumbel: overview §3 — greedy + grammar mask + gumbel
//    (temperature/RNG path), V=32. Late-mask miss (dummy-run) + recovery. ──
void run_section3_masked_gumbel() {
    std::printf("[section3_masked_gumbel]\n");
    ExecPlan plan;
    if (!build("section3_masked_gumbel", "505449520100000000000000050000000300000001000000010101000000010000000001010101000000010000000200030120000000010000000100020101000000010000000001020102000000010000000001000000000000010102010200000000000000010000000500030000000313000000a0000000020100000020000000390000000001200000009004000000900200000071020000000120000000018100000080ff200300000001000000050000001006000000040000003307000000640200000010020000000900000092040000000a0000003908000000010100000092000000000b0000009003000000810201000000100c0000000d00000092030000000e00000092010000000b000000", "5054494201000000ed743a6c49ec7ff90500000000000000010500000000000000ff0003000000ff0004000000030002000000030001000000030101000000030f00000000020100000020000000000120000000020102000000030120000000000120000000000000012000000000012000000001000201020000000201020000000101010000000201010000000200020101000000", plan)) return;
    expect(plan.executable && plan.needs_logits, "classified executable, needs_logits");

    std::map<std::uint32_t, std::shared_ptr<ChannelState>> externs;
    std::map<std::uint32_t, Value> seeds{
        {0, Value::i32({1})}, {3, Value::u32({1})}, {4, Value::u32({1234, 0})}};
    InterpInstance inst = make_instance(plan, externs, seeds);

    std::vector<float> logits(32, 0.0f);
    logits[7] = 100.0f;
    PassInputs in;
    in.logits = logits.data();
    in.rows = 1;
    in.vocab = 32;

    std::vector<std::uint8_t> allow_all(32, 1);
    expect(host_put(inst, plan, 2, Value::boolean(allow_all)) == HostOp::Ok,
          "host_put chan2 allow_all");
    StepResult r0 = step(inst, plan, in);
    Value v;
    HostOp rc = host_take(inst, plan, 1, v);
    expect(r0.ok && r0.committed && rc == HostOp::Ok && v.i[0] == 7,
          "step 0: token == 7 (committed=" + std::string(r0.committed ? "T" : "F") + ")");

    // Mask channel now empty (consumed) -> late-mask MISS, dummy-run.
    StepResult r1 = step(inst, plan, in);
    expect(r1.ok && !r1.committed, "step 1: late-mask dummy-run (committed=false)");

    std::vector<std::uint8_t> allow_only3(32, 0);
    allow_only3[3] = 1;
    host_put(inst, plan, 2, Value::boolean(allow_only3));
    StepResult r2 = step(inst, plan, in);
    rc = host_take(inst, plan, 1, v);
    expect(r2.ok && r2.committed && rc == HostOp::Ok && v.i[0] == 3,
          "step 2: recover, token == 3 (committed=" + std::string(r2.committed ? "T" : "F") + ")");
}

// ── mtp_verify_tail: echo's §6.1 match-verify K=3 cross-backend anchor.
//    Exercises the Stage-2 [K,vocab] MtpLogits MATRIX read at
//    mtp_draft_row=4 within the SAME logits buffer as the [4,8] verify
//    positions (rows 0..4) — the exact PassInputs single-buffer contract
//    this increment adds (§5.3), no MTP model involved (synthetic draft
//    logits validate the READ semantics only). ──
void run_mtp_verify_tail() {
    std::printf("[mtp_verify_tail]\n");
    ExecPlan plan;
    if (!build("mtp_verify_tail", "50544952010000000000000004000000000000000100000001010300000001000000000103020400000008000000010000000100010104000000010000000200010103000000010000000200031c000000a0000000020400000008000000a001000002030000000800000090010000008100000080ff38030000000204000000080000002002000000000000000400000033050000009000000000640300000060060000000800000018090000000700000081000000803f810000000000380b0000000103000000380c0000000103000000200a0000000d0000000e000000410f00000030100000000711000000023812000000010400000064040000001713000000140000008101ffffffff201500000006000000160000009202000000170000003301000000920000000018000000920300000018000000", "5054494201000000142e9ff7c60808b00400000000000000040000000100000003000000000003000200000003010300000003010100000003190000000002040000000800000000020300000008000000030204000000080000000000000204000000080000000002040000000800000001010400000001010300000002010300000001010300000003010300000000000000000103000000000103000000000103000000000103000000000002000201040000000201040000000301040000000100010104000000010103000000", plan)) return;
    expect(plan.executable && plan.needs_logits && plan.needs_mtp_logits,
          "classified executable, needs_logits + needs_mtp_logits");

    std::map<std::uint32_t, std::shared_ptr<ChannelState>> externs;
    std::map<std::uint32_t, Value> seeds{{0, Value::i32({3, 5, 6})}};
    InterpInstance inst = make_instance(plan, externs, seeds);

    std::vector<std::uint8_t> mask = {
        1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 1, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1,
    };
    expect(host_put(inst, plan, 1, Value::boolean(mask)) == HostOp::Ok, "host_put chan1 mask");

    // [logits(4x8) | mtp_logits(3x8)] packed into one base; MtpLogits reads
    // the 3 draft rows at mtp_draft_row = 4.
    std::vector<float> packed = {
        0, 0, 0, 9, 0, 0, 0, 0,  0, 0, 0, 0, 0, 9, 0, 0,
        0, 0, 1, 0, 0, 0, 9, 0,  0, 0, 0, 0, 9, 0, 0, 0,
        0, 7, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 7, 0, 0, 0,
        7, 0, 0, 0, 0, 0, 0, 0,
    };
    PassInputs in;
    in.logits = packed.data();
    in.rows = 7;
    in.vocab = 8;
    in.mtp_draft_row = 4;
    StepResult r = step(inst, plan, in);
    expect(r.ok && r.committed, "step 0: committed=true (err=" + r.error + ")");

    Value acc, mtp;
    expect(host_take(inst, plan, 2, acc) == HostOp::Ok &&
              acc.i.size() == 4 && acc.i[0] == 3 && acc.i[1] == 5 && acc.i[2] == 2 && acc.i[3] == -1,
          "take chan2 accept-prefix == [3,5,2,-1]");
    expect(host_take(inst, plan, 3, mtp) == HostOp::Ok &&
              mtp.i.size() == 3 && mtp.i[0] == 1 && mtp.i[1] == 4 && mtp.i[2] == 0,
          "take chan3 mtp-argmax == [1,4,0]");
}

// ── matrix_mask_apply_packed: a [2,8] matrix mask_apply_packed + argmax,
//    the MATRIX (not vector) variant of the epilogue op family. ──
void run_matrix_mask_apply_packed() {
    std::printf("[matrix_mask_apply_packed]\n");
    ExecPlan plan;
    if (!build("matrix_mask_apply_packed", "5054495201000000000000000100000000000000010000000101020000000100000002000306000000a0000000020200000008000000810228000000390100000001010000006500000000020000003303000000920000000004000000", "50544942010000000be0db922b5f4f3201000000000100000000000000030101000000030500000000020200000008000000020002010100000000020200000008000000010102000000", plan)) return;
    expect(plan.executable && plan.needs_logits, "classified executable, needs_logits");

    std::map<std::uint32_t, std::shared_ptr<ChannelState>> externs;
    std::map<std::uint32_t, Value> seeds;  // no seeds — the channel is host_reader-only
    InterpInstance inst = make_instance(plan, externs, seeds);

    std::vector<float> logits = {0.0, 0.0, 9.0, 1.0, 0.0, 2.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 4.0, 0.0, 3.0, 0.0, 9.0};
    PassInputs in;
    in.logits = logits.data();
    in.rows = 2;
    in.vocab = 8;
    StepResult r = step(inst, plan, in);
    expect(r.ok && r.committed, "step 0: committed=true (err=" + r.error + ")");
    Value out;
    expect(host_take(inst, plan, 0, out) == HostOp::Ok && out.i.size() == 2 && out.i[0] == 5 &&
              out.i[1] == 3,
          "take chan0 == [5,3]");
}

// ── Rejection honesty (G1.5): classify_exec_plan over hand-built traces. ──
void run_classification_rejections() {
    std::printf("[classification: rejection honesty]\n");
    {
        ExecPlan plan;
        cptir::Value v;
        v.source = cptir::ValueSource::HostInput;
        plan.trace.values.push_back(v);
        classify_exec_plan(plan);
        expect(!plan.executable && plan.reject_reason.find("host input") != std::string::npos,
              "HostInput root rejects with a host-input reason (" + plan.reject_reason + ")");
    }

    {
        ExecPlan plan;
        cptir::Stage st;
        st.kind = cptir::StageKind::OnAttn;
        plan.trace.stages.push_back(st);
        classify_exec_plan(plan);
        expect(!plan.executable && plan.reject_reason.find("per-layer") != std::string::npos,
              "OnAttn stage rejects with a per-layer-tap reason (" + plan.reject_reason + ")");
    }
    {
        ExecPlan plan;
        cptir::Stage st;
        st.kind = cptir::StageKind::OnAttnProj;
        plan.trace.stages.push_back(st);
        classify_exec_plan(plan);
        expect(!plan.executable && plan.reject_reason.find("per-layer") != std::string::npos,
              "OnAttnProj stage rejects with a per-layer-tap reason (" + plan.reject_reason + ")");
    }
    for (const auto intr : {cptir::Intrinsic::Hidden, cptir::Intrinsic::Query,
                            cptir::Intrinsic::ValueHead}) {
        ExecPlan plan;
        cptir::Value v;
        v.source = cptir::ValueSource::Intrinsic;
        v.intrinsic = intr;
        plan.trace.values.push_back(v);
        classify_exec_plan(plan);
        expect(!plan.executable && !plan.needs_logits && !plan.needs_mtp_logits &&
                  plan.reject_reason.find("intrinsic") != std::string::npos,
              "unsupported intrinsic (" + std::to_string(static_cast<int>(intr)) +
                  ") rejects with an intrinsic reason (" + plan.reject_reason + ")");
    }
    {
        // Logits/MtpLogits: executable=true, classified as "needs forward",
        // never a hard reject — the whole point of the C2 split (§5.3).
        ExecPlan plan;
        cptir::Value logits_v;
        logits_v.source = cptir::ValueSource::Intrinsic;
        logits_v.intrinsic = cptir::Intrinsic::Logits;
        plan.trace.values.push_back(logits_v);
        cptir::Value mtp_v;
        mtp_v.source = cptir::ValueSource::Intrinsic;
        mtp_v.intrinsic = cptir::Intrinsic::MtpLogits;
        plan.trace.values.push_back(mtp_v);
        classify_exec_plan(plan);
        expect(plan.executable && plan.needs_logits && plan.needs_mtp_logits &&
                  plan.reject_reason.empty(),
              "Logits + MtpLogits roots classify as executable + needs_forward, no reject");
    }
}

void run_multistage_pivot_payload() {
    std::printf("[pivot_threshold: predicate payload is a plain global ValueId]\n");
    Trace trace;
    trace.values.resize(5);
    trace.values[3].type.dtype = cptir::DType::F32;
    trace.values[3].type.shape.dims = {3};

    cptir::Op op;
    op.code = cptir::OpCode::PivotThreshold;
    op.args = {3};
    op.result_id = 4;
    op.result_count = 1;
    op.predicate.tag = cptir::PredTag::ProbGe;
    // Already a GLOBAL trace id (container_to_trace / bound.hpp remaps the
    // wire's stage-local id through gid() before this point, exactly like any
    // other op operand) — eval_op must dereference it directly, with no
    // further stage-base rebasing of its own.
    op.predicate.payload = 2;

    std::vector<Value> vals(5);
    vals[0] = Value::f32({100.0f});  // A different (wrong) value: must not be read.
    vals[2] = Value::f32({0.5f});
    vals[3] = Value::f32({0.2f, 0.6f, 0.8f});
    std::string error;
    const bool ok = detail::eval_op(op, trace, vals, error);
    expect(ok && vals[4].b == std::vector<std::uint8_t>({0, 1, 1}),
           "predicate payload dereferences the global id directly (no eval_op-side rebasing)");
}

}  // namespace

int main() {
    run_greedy_argmax();
    run_section3_masked_gumbel();
    run_mtp_verify_tail();
    run_matrix_mask_apply_packed();
    run_classification_rejections();
    run_multistage_pivot_payload();
    std::printf("\n==== pipeline_interp_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
