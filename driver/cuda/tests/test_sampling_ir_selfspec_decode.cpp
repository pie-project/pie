// Lane L2 / charlie (#31 STOP-#3 boundary guard): the C++ v4 reader must decode
// the `SelfSpecDraftInput` readiness (bytecode bit 6, READY_SELFSPEC_BIT=0x40) the
// Rust encoder emits — it's a DUAL Rust-encoder/C++-reader contract, and the
// skeleton's Rust-only round-trip is necessary-not-sufficient. Before the reader
// fix, the real `mtp_self_spec_greedy` bytecode FAILED to decode driver-side:
// reader masked only 0x80 → 0x40 survived into parse_dtype (tags 0..=3) →
// DecodeError::UnknownTag "input dtype". This pure-host boundary test pins:
//   (1) decode succeeds (0x40 masked off),
//   (2) the draft Host binding is stamped HostAvailability::SelfSpecDraftInput
//       (the marker echo's resolver redirects to pi.tokens+sample_row+1), and
//   (3) lower classifies it HostLate (device-resident, not HostSubmit fail-loud).
// Bytecode = the REAL `mtp_self_spec_greedy(vocab=32, k=4)` (sampling-edsl), the
// program the executor actually fires. No CUDA.
#include <cstdio>
#include <vector>

#include "sampling_ir/ir.hpp"
#include "sampling_ir/reader.hpp"
#include "sampling_ir/codegen.hpp"

using namespace pie_cuda_driver::sampling_ir;

namespace {
int g_fail = 0;
#define EXPECT(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); ++g_fail; } \
    } while (0)

// `mtp_self_spec_greedy(32, 4)` lowered v4 bytecode (sampling-edsl). The draft
// input's dtype byte is 0x41 = I32(0x01) | READY_SELFSPEC_BIT(0x40).
const unsigned char SELFSPEC_BC[] = {
    0x50,0x53,0x49,0x52,0x04,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x0f,0x00,0x00,0x00,
    0x01,0x00,0x00,0x00,0x00,0x02,0x04,0x00,0x00,0x00,0x20,0x00,0x00,0x00,0x41,0x01,
    0x04,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x00,0x80,0x01,0x00,0x00,0x00,0x33,0x00,
    0x00,0x00,0x00,0x18,0x02,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x81,0x00,0x00,0x00,
    0x80,0x3f,0x38,0x04,0x00,0x00,0x00,0x01,0x04,0x00,0x00,0x00,0x81,0x00,0x00,0x00,
    0x00,0x00,0x38,0x06,0x00,0x00,0x00,0x01,0x04,0x00,0x00,0x00,0x20,0x03,0x00,0x00,
    0x00,0x05,0x00,0x00,0x00,0x07,0x00,0x00,0x00,0x41,0x08,0x00,0x00,0x00,0x81,0x00,
    0x00,0x00,0x00,0x3f,0x16,0x09,0x00,0x00,0x00,0x0a,0x00,0x00,0x00,0x81,0x01,0xff,
    0xff,0xff,0xff,0x38,0x0c,0x00,0x00,0x00,0x01,0x04,0x00,0x00,0x00,0x20,0x0b,0x00,
    0x00,0x00,0x01,0x00,0x00,0x00,0x0d,0x00,0x00,0x00,0x0e,0x00,0x00,0x00,0x00,
};

// Slot manifest (one Binding per input, slot order): the [k,vocab] target logits
// intrinsic + the [k] i32 draft (Host). The draft's host_avail is overridden by
// the bytecode's 0x40 bit → SelfSpecDraftInput (readiness is a program property).
std::vector<Binding> manifest() {
    std::vector<Binding> sb(2);
    sb[0].tag = BindingTag::Intrinsic;
    sb[0].intrinsic = Intrinsic::Logits;
    sb[1].tag = BindingTag::Host;
    sb[1].host_key = 1;
    sb[1].host_avail = HostAvailability::SubmitBound;  // bytecode 0x40 must override
    return sb;
}

// Same binding shape as `manifest()`, in the codegen-facing ProgramManifest form
// that lower_bytecode_v4 consumes.
ProgramManifest lower_manifest() {
    ProgramManifest m;
    { InputBind b; b.kind = BindKind::Logits; m.push_back(b); }
    { InputBind b; b.kind = BindKind::HostTensor; b.host_key = 1; m.push_back(b); }
    return m;
}
}  // namespace

int main() {
    // (1)+(2) decode succeeds and stamps the draft SelfSpecDraftInput.
    Program prog;
    DecodeError err;
    const bool ok = decode_v4(SELFSPEC_BC, sizeof(SELFSPEC_BC), manifest(), prog, &err);
    EXPECT(ok, "decode_v4 must succeed (0x40 masked off, no UnknownTag)");

    bool found_draft = false;
    for (const Input& in : prog.inputs) {
        if (in.binding.tag != BindingTag::Host) continue;
        found_draft = true;
        EXPECT(in.binding.host_avail == HostAvailability::SelfSpecDraftInput,
               "draft Host binding must be stamped SelfSpecDraftInput (the resolver marker)");
    }
    EXPECT(found_draft, "program must have the Host draft input");

    // (3) lower classifies the draft device-resident (HostLate), not HostSubmit.
    LowerResult lr = lower_bytecode_v4(SELFSPEC_BC, sizeof(SELFSPEC_BC), lower_manifest(), LowerOptions{});
    EXPECT(lr.ok, "lower_bytecode_v4 must succeed");
    bool found_hostlate = false;
    for (const BufferDecl& b : lr.dag.buffers)
        if (b.cls == BufferClass::HostLate) found_hostlate = true;
    EXPECT(found_hostlate, "the draft buffer must classify HostLate (device-resident)");

    if (g_fail == 0) {
        std::fprintf(stderr, "sampling_ir_selfspec_decode: OK\n");
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_selfspec_decode: %d failure(s)\n", g_fail);
    return 1;
}
