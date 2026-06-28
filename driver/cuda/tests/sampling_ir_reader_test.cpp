// Unit test for the PSIR v1 bytecode reader (reader.cpp). Decodes authoritative
// golden vectors produced by alpha's Rust encoder (sampling_ir_golden_bytecode.h)
// and asserts the parsed IR graph matches, plus negative/structural cases. Pure
// C++ (no CUDA); runs fast under CTest.

#include <cstdio>
#include <cstring>
#include <string>

#include "sampling_ir/ir.hpp"
#include "sampling_ir/reader.hpp"
#include "sampling_ir_golden_bytecode.h"

using namespace pie_cuda_driver::sampling_ir;

namespace {
int g_fail = 0, g_checks = 0;
void expect(bool c, const char* what) {
    ++g_checks;
    if (!c) { ++g_fail; std::fprintf(stderr, "  FAIL: %s\n", what); }
}

template <std::size_t N>
bool decode_arr(const unsigned char (&a)[N], Program& p, DecodeError& e) {
    return decode(reinterpret_cast<const std::uint8_t*>(a), N, p, &e);
}

void test_argmax() {
    std::printf("[GV_ARGMAX]\n");
    Program p; DecodeError e;
    expect(decode_arr(GV_ARGMAX, p, e), "argmax decodes");
    expect(p.inputs.size() == 1, "1 input");
    expect(p.slots.size() == 1, "1 slot");
    if (p.inputs.size() == 1) {
        const Input& in = p.inputs[0];
        expect(in.id == 0, "input id 0");
        expect(in.ty.dtype == DType::F32, "logits F32");
        expect(in.ty.shape.tag == ShapeTag::Vector && in.ty.shape.a == 32000, "Vector{32000}");
        expect(in.binding.tag == BindingTag::Intrinsic && in.binding.intrinsic == Intrinsic::Logits,
               "Intrinsic(Logits)");
    }
    if (p.slots.size() == 1 && p.slots[0].ops.size() == 1) {
        const Op& op = p.slots[0].ops[0];
        expect(op.code == OpCode::ReduceArgmax, "op = ReduceArgmax");
        expect(op.a == 0, "operand v=0");
        expect(op.result_id == 1 && op.result_count == 1, "result id 1");
        expect(p.slots[0].outputs.size() == 1 && p.slots[0].outputs[0].value == 1 &&
                   p.slots[0].outputs[0].kind == OutputKind::Token, "output = id 1 (Token)");
    } else {
        expect(false, "argmax slot has 1 op");
    }
}

void test_sample() {
    std::printf("[GV_SAMPLE]\n");
    Program p; DecodeError e;
    expect(decode_arr(GV_SAMPLE, p, e), "sample decodes");
    expect(p.inputs.size() == 3, "3 inputs");
    if (p.inputs.size() == 3) {
        expect(p.inputs[1].binding.tag == BindingTag::Const, "input1 Const");
        expect(p.inputs[1].binding.lit.dtype == DType::F32, "const F32");
        expect(p.inputs[1].binding.lit.as_f32() == 0.7f, "const value 0.7");
        expect(p.inputs[2].binding.tag == BindingTag::Host, "input2 Host");
        expect(p.inputs[2].binding.host_key == 42, "host key 42");
        expect(p.inputs[2].binding.host_avail == HostAvailability::SubmitBound, "submit-bound");
    }
    if (p.slots.size() == 1) {
        const Slot& s = p.slots[0];
        expect(s.ops.size() == 8, "8 ops");
        // ops: Div(0,1)=3, Exp(3)=4, ReduceSum(4)=5, Div(4,5)=6, SortDesc(6)=7&8,
        //      Pivot(7,CummassLe(1))=9, Rng(2,Gumbel)=10, ReduceArgmax(10)=11
        if (s.ops.size() == 8) {
            expect(s.ops[0].code == OpCode::Div && s.ops[0].a == 0 && s.ops[0].b == 1 && s.ops[0].result_id == 3, "Div(0,1)->3");
            expect(s.ops[4].code == OpCode::SortDesc && s.ops[4].result_id == 7 && s.ops[4].result_count == 2, "SortDesc->7,8");
            expect(s.ops[5].code == OpCode::PivotThreshold && s.ops[5].a == 7 && s.ops[5].result_id == 9, "Pivot(7)->9");
            expect(s.ops[5].predicate.tag == PredTag::CummassLe && s.ops[5].predicate.payload == 1, "CummassLe(id1)");
            expect(s.ops[6].code == OpCode::Rng && s.ops[6].a == 2 && s.ops[6].rng_kind == RngKind::Gumbel && s.ops[6].result_id == 10, "Rng(seed2,Gumbel)->10");
            expect(s.ops[6].shape.tag == ShapeTag::Vector && s.ops[6].shape.a == 32000, "Rng shape Vector{32000}");
            expect(s.ops[7].code == OpCode::ReduceArgmax && s.ops[7].a == 10 && s.ops[7].result_id == 11, "ReduceArgmax(10)->11");
        }
        expect(s.outputs.size() == 2 && s.outputs[0].value == 11 && s.outputs[1].value == 8,
               "outputs [11,8]");
    }
}

void test_temp() {
    std::printf("[GV_TEMP]\n");
    Program p; DecodeError e;
    expect(decode_arr(GV_TEMP, p, e), "temp decodes");
    expect(p.inputs.size() == 3, "3 inputs");
    if (!p.slots.empty() && p.slots[0].ops.size() == 4) {
        const Slot& s = p.slots[0];
        expect(s.ops[0].code == OpCode::Div && s.ops[0].result_id == 3, "Div->3");
        expect(s.ops[1].code == OpCode::Rng && s.ops[1].a == 2 && s.ops[1].result_id == 4, "Rng(seed2)->4");
        expect(s.ops[2].code == OpCode::Add && s.ops[2].a == 3 && s.ops[2].b == 4 && s.ops[2].result_id == 5, "Add(3,4)->5");
        expect(s.ops[3].code == OpCode::ReduceArgmax && s.ops[3].a == 5 && s.ops[3].result_id == 6, "Argmax(5)->6");
        expect(s.outputs.size() == 1 && s.outputs[0].value == 6, "output [6]");
    } else {
        expect(false, "temp has 4 ops");
    }
}

void test_allops() {
    std::printf("[GV_ALLOPS] (layout coverage)\n");
    Program p; DecodeError e;
    expect(decode_arr(GV_ALLOPS, p, e), "allops decodes");
    if (p.slots.empty()) { expect(false, "allops slot"); return; }
    const Slot& s = p.slots[0];
    expect(s.ops.size() == 30, "30 ops decoded");

    // Every op tag must appear at least once (reader switch coverage).
    bool seen[256] = {false};
    for (const Op& op : s.ops) seen[static_cast<std::uint8_t>(op.code)] = true;
    const std::uint8_t tags[] = {
        0x01,0x02,0x03,0x04,0x05,0x06, 0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,
        0x20, 0x30,0x31,0x32,0x33,0x38, 0x40,0x41, 0x50,0x58, 0x60,0x61,0x62,0x63, 0x70};
    bool all = true;
    for (std::uint8_t t : tags) if (!seen[t]) all = false;
    expect(all, "all 30 op tags covered");

    // SSA counter: SortDesc reserves 2 consecutive ids; the next op's result id
    // must be SortDesc.result_id + 2.
    for (std::size_t i = 0; i + 1 < s.ops.size(); ++i) {
        if (s.ops[i].code == OpCode::SortDesc) {
            expect(s.ops[i].result_count == 2, "SortDesc result_count 2");
            expect(s.ops[i + 1].result_id == s.ops[i].result_id + 2, "id advances by 2 after SortDesc");
        }
    }
    // Broadcast (0x38) carries a shape; spot-check it parsed a Vector{4}.
    for (const Op& op : s.ops) {
        if (op.code == OpCode::Broadcast) {
            expect(op.shape.tag == ShapeTag::Vector && op.shape.a == 4, "Broadcast shape Vector{4}");
        }
    }
}

void test_negatives() {
    std::printf("[negative cases]\n");
    // bad magic
    {
        unsigned char b[sizeof(GV_ARGMAX)];
        std::memcpy(b, GV_ARGMAX, sizeof(b));
        b[0] = 'X';
        Program p; DecodeError e;
        expect(!decode(reinterpret_cast<std::uint8_t*>(b), sizeof(b), p, &e) && e.code == DecodeError::BadMagic,
               "bad magic rejected");
    }
    // truncated header
    {
        Program p; DecodeError e;
        expect(!decode(reinterpret_cast<const std::uint8_t*>(GV_ARGMAX), 10, p, &e) &&
                   e.code == DecodeError::UnexpectedEof,
               "truncated rejected (EOF)");
    }
    // unknown opcode: op tag at offset header(16)+input(16)+n_ops(4) = 36
    {
        unsigned char b[sizeof(GV_ARGMAX)];
        std::memcpy(b, GV_ARGMAX, sizeof(b));
        const std::size_t op_off = 16 + 16 + 4;
        expect(b[op_off] == 0x33, "op tag at expected offset");
        b[op_off] = 0xEE;
        Program p; DecodeError e;
        expect(!decode(reinterpret_cast<std::uint8_t*>(b), sizeof(b), p, &e) &&
                   e.code == DecodeError::UnknownOpcode,
               "unknown opcode rejected");
    }
    // bad version
    {
        unsigned char b[sizeof(GV_ARGMAX)];
        std::memcpy(b, GV_ARGMAX, sizeof(b));
        b[4] = 0x01;  // version low byte -> 1 (v1 no longer supported; reader is v2)
        Program p; DecodeError e;
        expect(!decode(reinterpret_cast<std::uint8_t*>(b), sizeof(b), p, &e) &&
                   e.code == DecodeError::BadVersion,
               "bad version rejected");
    }
}

void test_rowbcast() {
    std::printf("[GV_GATHERCOLS] (v3 op 0x64)\n");
    {
        Program p; DecodeError e;
        expect(decode_arr(GV_GATHERCOLS, p, e), "gathercols decodes");
        if (!p.slots.empty() && !p.slots[0].ops.empty()) {
            const Op& op = p.slots[0].ops[0];
            expect(op.code == OpCode::GatherCols, "op = GatherCols (0x64)");
            expect(op.a == 0 && op.b == 1, "GatherCols src=0 idx=1");
        } else expect(false, "gathercols slot");
    }
    std::printf("[GV_ROWBCAST] (v3 op 0x39)\n");
    Program p; DecodeError e;
    expect(decode_arr(GV_ROWBCAST, p, e), "rowbcast decodes (v3)");
    if (p.slots.empty()) { expect(false, "rowbcast slot"); return; }
    const Slot& s = p.slots[0];
    // ops: Exp(0)=1, ReduceSum(1)=2, RowBroadcast(2,len=8)=3, Div(1,3)=4
    bool found = false;
    for (const Op& op : s.ops) {
        if (op.code == OpCode::RowBroadcast) {
            found = true;
            expect(op.a == 2, "RowBroadcast per_row = id2");
            expect(op.imm == 8, "RowBroadcast len immediate = 8");
        }
    }
    expect(found, "RowBroadcast (0x39) op present");
}

}  // namespace

int main() {
    test_argmax();
    test_sample();
    test_temp();
    test_rowbcast();
    test_allops();
    test_negatives();
    std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
    if (g_fail == 0) std::printf("ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
