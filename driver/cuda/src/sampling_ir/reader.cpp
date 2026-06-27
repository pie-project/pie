#include "sampling_ir/reader.hpp"

#include <cstring>

namespace pie_cuda_driver::sampling_ir {

namespace {

// Forward cursor over [p, end). Every read is bounds-checked; the first
// out-of-bounds read sets `ok = false` and subsequent reads return 0 so the
// caller can bail after the fact without UB.
struct Cursor {
    const std::uint8_t* p;
    const std::uint8_t* end;
    bool ok = true;

    bool have(std::size_t n) const { return static_cast<std::size_t>(end - p) >= n; }

    std::uint8_t u8() {
        if (!ok || !have(1)) { ok = false; return 0; }
        return *p++;
    }
    std::uint16_t u16() {
        if (!ok || !have(2)) { ok = false; return 0; }
        std::uint16_t v;
        std::memcpy(&v, p, 2);
        p += 2;
        return v;  // host is little-endian (x86_64 / sm_89 host)
    }
    std::uint32_t u32() {
        if (!ok || !have(4)) { ok = false; return 0; }
        std::uint32_t v;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }
};

bool parse_dtype(std::uint8_t tag, DType& out) {
    if (tag > 3) return false;
    out = static_cast<DType>(tag);
    return true;
}

bool read_shape(Cursor& c, Shape& out) {
    std::uint8_t tag = c.u8();
    std::uint32_t a = c.u32();
    std::uint32_t b = c.u32();
    if (!c.ok || tag > 3) return false;
    out.tag = static_cast<ShapeTag>(tag);
    out.a = a;
    out.b = b;
    return true;
}

bool read_literal(Cursor& c, Literal& out) {
    std::uint8_t dt = c.u8();
    std::uint32_t bits = c.u32();
    DType d;
    if (!c.ok || !parse_dtype(dt, d)) return false;
    out.dtype = d;
    out.bits = bits;
    return true;
}

bool read_input(Cursor& c, std::uint32_t expected_id, Input& out, DecodeError* err) {
    out.id = c.u32();
    if (c.ok && out.id != expected_id) {
        if (err) { err->code = DecodeError::BadInputId; err->detail = "Input.id != index"; }
        return false;
    }
    std::uint8_t dt = c.u8();
    if (!parse_dtype(dt, out.ty.dtype)) {
        if (err) { err->code = DecodeError::UnknownTag; err->detail = "input dtype"; }
        return false;
    }
    if (!read_shape(c, out.ty.shape)) {
        if (err) { err->code = DecodeError::UnknownTag; err->detail = "input shape"; }
        return false;
    }
    std::uint8_t btag = c.u8();
    switch (btag) {
        case 0: {  // Const
            out.binding.tag = BindingTag::Const;
            if (!read_literal(c, out.binding.lit)) {
                if (err) { err->code = DecodeError::UnknownTag; err->detail = "const literal dtype"; }
                return false;
            }
            break;
        }
        case 1: {  // Intrinsic
            out.binding.tag = BindingTag::Intrinsic;
            std::uint8_t in = c.u8();
            if (in != 0) {
                if (err) { err->code = DecodeError::UnknownTag; err->detail = "intrinsic"; }
                return false;
            }
            out.binding.intrinsic = Intrinsic::Logits;
            break;
        }
        case 2: {  // Host
            out.binding.tag = BindingTag::Host;
            out.binding.host_key = c.u32();
            std::uint8_t av = c.u8();
            if (av > 1) {
                if (err) { err->code = DecodeError::UnknownTag; err->detail = "host availability"; }
                return false;
            }
            out.binding.host_avail = static_cast<HostAvailability>(av);
            break;
        }
        case 3: {  // Output
            out.binding.tag = BindingTag::Output;
            out.binding.out_program = c.u32();
            out.binding.out_value = c.u32();
            break;
        }
        default:
            if (err) { err->code = DecodeError::UnknownTag; err->detail = "binding tag"; }
            return false;
    }
    return c.ok;
}

bool read_predicate(Cursor& c, Predicate& out) {
    std::uint8_t tag = c.u8();
    std::uint32_t payload = c.u32();
    if (!c.ok || tag > 2) return false;
    out.tag = static_cast<PredTag>(tag);
    out.payload = payload;
    return true;
}

// Decode one op (tag already known to be a valid OpCode) by reading its fixed
// operand layout (BYTECODE.md §4).
bool read_op_operands(Cursor& c, OpCode code, Op& op, DecodeError* err) {
    op.code = code;
    switch (code) {
        // unary: a
        case OpCode::Exp: case OpCode::Log: case OpCode::Neg: case OpCode::Recip:
        case OpCode::Abs: case OpCode::Sign:
        case OpCode::ReduceSum: case OpCode::ReduceMax: case OpCode::ReduceMin:
        case OpCode::ReduceArgmax:
        case OpCode::CumSum: case OpCode::CumProd:
        case OpCode::SortDesc:
            op.a = c.u32();
            break;
        // binary: a, b
        case OpCode::Add: case OpCode::Sub: case OpCode::Mul: case OpCode::Div:
        case OpCode::MaxElem: case OpCode::MinElem:
        case OpCode::Gt: case OpCode::Ge: case OpCode::Eq:
        case OpCode::Gather: case OpCode::GatherRow: case OpCode::GatherCols:
            op.a = c.u32();
            op.b = c.u32();
            break;
        // ternary: a, b, c
        case OpCode::Select:
        case OpCode::ScatterAdd: case OpCode::ScatterSet:
            op.a = c.u32();
            op.b = c.u32();
            op.c = c.u32();
            break;
        // broadcast: scalar(a), shape
        case OpCode::Broadcast:
            op.a = c.u32();
            if (!read_shape(c, op.shape)) {
                if (err) { err->code = DecodeError::UnknownTag; err->detail = "broadcast shape"; }
                return false;
            }
            break;
        // row-broadcast (v3): per_row(a), len immediate → Matrix{rows=per_row.len, len}
        case OpCode::RowBroadcast:
            op.a = c.u32();
            op.imm = c.u32();
            break;
        // pivot: input(a), predicate
        case OpCode::PivotThreshold:
            op.a = c.u32();
            if (!read_predicate(c, op.predicate)) {
                if (err) { err->code = DecodeError::UnknownTag; err->detail = "predicate tag"; }
                return false;
            }
            break;
        // rng: seed(a), kind, shape
        case OpCode::Rng: {
            op.a = c.u32();
            std::uint8_t k = c.u8();
            if (k > 1) {
                if (err) { err->code = DecodeError::UnknownTag; err->detail = "rng kind"; }
                return false;
            }
            op.rng_kind = static_cast<RngKind>(k);
            if (!read_shape(c, op.shape)) {
                if (err) { err->code = DecodeError::UnknownTag; err->detail = "rng shape"; }
                return false;
            }
            break;
        }
    }
    return c.ok;
}

bool is_known_opcode(std::uint8_t tag, OpCode& out) {
    switch (tag) {
        case 0x01: case 0x02: case 0x03: case 0x04: case 0x05: case 0x06:
        case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15:
        case 0x16: case 0x17: case 0x18:
        case 0x20:
        case 0x30: case 0x31: case 0x32: case 0x33: case 0x38: case 0x39:
        case 0x40: case 0x41:
        case 0x50: case 0x58:
        case 0x60: case 0x61: case 0x62: case 0x63: case 0x64:
        case 0x70:
            out = static_cast<OpCode>(tag);
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// PSIR v4 (shape-typed, binding-free). Header(20) / InputDecl[] / Op[] /
// Output[]. Input(0x80)/Const(0x81) are leaf ops in the flat SSA space; the
// per-slot binding (logits intrinsic / host tensor) is supplied out-of-band in
// `slot_bindings` (the EDSL manifest). Shimmed into the in-memory Program: leaf
// Input/Const ops → Program.inputs (value-ids first), compute ops → one Slot,
// v4 SSA ids remapped to the inputs-first convention the codegen expects.
// ---------------------------------------------------------------------------

// v4 shape: rank:u8 | dims[rank]:u32. rank 0=scalar, 1=vector, 2=matrix{rows,len}.
bool read_shape_v4(Cursor& c, Shape& out) {
    std::uint8_t rank = c.u8();
    if (!c.ok) return false;
    if (rank == 0) { out = Shape::scalar(); return true; }
    if (rank == 1) { std::uint32_t d0 = c.u32(); out = Shape::vector(d0); return c.ok; }
    if (rank == 2) { std::uint32_t d0 = c.u32(); std::uint32_t d1 = c.u32(); out = Shape::matrix(d0, d1); return c.ok; }
    return false;
}

// v4 op operands. Identical to v3 except Broadcast/Rng use the v4 shape, and Rng
// carries a static `stream` (no seed operand, Model B).
bool read_op_operands_v4(Cursor& c, OpCode code, Op& op, DecodeError* err) {
    op.code = code;
    if (code == OpCode::Broadcast) {
        op.a = c.u32();
        if (!read_shape_v4(c, op.shape)) { if (err) { err->code = DecodeError::UnknownTag; err->detail = "broadcast shape"; } return false; }
        return c.ok;
    }
    if (code == OpCode::Rng) {
        op.stream = c.u32();
        if (!read_shape_v4(c, op.shape)) { if (err) { err->code = DecodeError::UnknownTag; err->detail = "rng shape"; } return false; }
        std::uint8_t k = c.u8();
        if (!c.ok || k > 1) { if (err) { err->code = DecodeError::UnknownTag; err->detail = "rng kind"; } return false; }
        op.rng_kind = static_cast<RngKind>(k);
        return c.ok;
    }
    // every other op has the v3 operand layout (raw value-ids, no shape).
    return read_op_operands(c, code, op, err);
}

// Remap the value-id operands an op actually uses (arity-aware) through `m`.
void remap_op_v4(Op& op, const std::vector<std::uint32_t>& m) {
    auto r = [&](std::uint32_t v) { return v < m.size() ? m[v] : v; };
    switch (op.code) {
        case OpCode::Select: case OpCode::ScatterAdd: case OpCode::ScatterSet:
            op.c = r(op.c); [[fallthrough]];
        case OpCode::Add: case OpCode::Sub: case OpCode::Mul: case OpCode::Div:
        case OpCode::MaxElem: case OpCode::MinElem:
        case OpCode::Gt: case OpCode::Ge: case OpCode::Eq:
        case OpCode::Gather: case OpCode::GatherRow: case OpCode::GatherCols:
            op.b = r(op.b); [[fallthrough]];
        case OpCode::Exp: case OpCode::Log: case OpCode::Neg: case OpCode::Recip:
        case OpCode::Abs: case OpCode::Sign:
        case OpCode::ReduceSum: case OpCode::ReduceMax: case OpCode::ReduceMin:
        case OpCode::ReduceArgmax: case OpCode::CumSum: case OpCode::CumProd:
        case OpCode::SortDesc: case OpCode::Broadcast: case OpCode::RowBroadcast:
            op.a = r(op.a);
            break;
        case OpCode::PivotThreshold:
            op.a = r(op.a);
            if (op.predicate.tag == PredTag::CummassLe || op.predicate.tag == PredTag::ProbGe)
                op.predicate.payload = r(op.predicate.payload);
            break;
        case OpCode::Rng:   // v4: no value-id operands (ambient seed + static stream)
            break;
    }
}

bool read_program_v4(const std::uint8_t* data, std::size_t len,
                     const std::vector<Binding>& slot_bindings, Program& out, DecodeError* err) {
    out.inputs.clear();
    out.slots.clear();
    out.ambient_seed = true;   // v4 = Model B: RNG seed is ambient (no Op::Rng operand)

    Cursor c{data, data + len};
    if (!c.have(4) || std::memcmp(c.p, "PSIR", 4) != 0) {
        if (err) { err->code = DecodeError::BadMagic; err->detail = "magic != PSIR"; }
        return false;
    }
    c.p += 4;
    std::uint16_t version = c.u16();
    if (!c.ok || version != 4) {
        if (err) { err->code = DecodeError::BadVersion; err->detail = "version " + std::to_string(version); }
        return false;
    }
    (void)c.u16();  // flags
    std::uint32_t n_inputs = c.u32();
    std::uint32_t n_ops = c.u32();
    std::uint32_t n_outputs = c.u32();
    if (!c.ok) { if (err) { err->code = DecodeError::UnexpectedEof; err->detail = "v4 header"; } return false; }
    if (slot_bindings.size() != n_inputs) {
        if (err) { err->code = DecodeError::BadInputId; err->detail = "manifest/n_inputs mismatch"; }
        return false;
    }

    // InputDecl[]: dtype:u8 | shape_v4 — typed slots, binding from `slot_bindings`.
    std::vector<ValueType> slot_ty(n_inputs);
    for (std::uint32_t i = 0; i < n_inputs; ++i) {
        std::uint8_t dt = c.u8();
        if (!parse_dtype(dt, slot_ty[i].dtype)) { if (err) { err->code = DecodeError::UnknownTag; err->detail = "input dtype"; } return false; }
        if (!read_shape_v4(c, slot_ty[i].shape)) { if (err) { err->code = DecodeError::UnknownTag; err->detail = "input shape"; } return false; }
    }

    // Pass 1: parse ops, classify Input(0x80)/Const(0x81)/compute, track v4 SSA ids.
    struct Parsed { int kind; std::uint32_t v4_id; std::uint32_t slot_idx; Literal lit; Op op; };
    std::vector<Parsed> parsed;
    parsed.reserve(n_ops);
    std::uint32_t v4_id = 0, n_consts = 0;
    for (std::uint32_t i = 0; i < n_ops; ++i) {
        std::uint8_t tag = c.u8();
        if (!c.ok) { if (err) { err->code = DecodeError::UnexpectedEof; err->detail = "op tag"; } return false; }
        if (tag == 0x80) {            // Op::Input(idx)
            std::uint32_t idx = c.u32();
            if (!c.ok || idx >= n_inputs) { if (err) { err->code = DecodeError::BadInputId; err->detail = "input(idx)"; } return false; }
            parsed.push_back({0, v4_id++, idx, {}, {}});
        } else if (tag == 0x81) {     // Op::Const(lit)
            Literal lit;
            if (!read_literal(c, lit)) { if (err) { err->code = DecodeError::UnknownTag; err->detail = "const literal"; } return false; }
            parsed.push_back({1, v4_id++, 0, lit, {}});
            ++n_consts;
        } else {                      // compute op
            OpCode code;
            if (!is_known_opcode(tag, code)) { if (err) { err->code = DecodeError::UnknownTag; err->detail = "opcode " + std::to_string(tag); } return false; }
            Op op;
            if (!read_op_operands_v4(c, code, op, err)) {
                if (err && err->code == DecodeError::None) { err->code = DecodeError::UnexpectedEof; err->detail = "op operands"; }
                return false;
            }
            op.result_count = op_result_count(code);
            parsed.push_back({2, v4_id, 0, {}, op});
            v4_id += op.result_count;
        }
    }

    // Pass 2: assign in-memory ids — leaves (Input/Const) first (0..n_leaf-1),
    // compute ops after; build the v4→in-memory SSA map.
    std::uint32_t n_leaf = n_inputs + n_consts;
    std::vector<std::uint32_t> v4_to_mine(v4_id, 0);
    out.inputs.reserve(n_leaf);
    std::uint32_t leaf_id = 0, compute_id = n_leaf;
    for (const Parsed& p : parsed) {
        if (p.kind == 0) {
            Input in; in.id = leaf_id; in.ty = slot_ty[p.slot_idx]; in.binding = slot_bindings[p.slot_idx];
            out.inputs.push_back(in);
            v4_to_mine[p.v4_id] = leaf_id++;
        } else if (p.kind == 1) {
            Input in; in.id = leaf_id;
            in.binding.tag = BindingTag::Const; in.binding.lit = p.lit;
            in.ty.dtype = p.lit.dtype; in.ty.shape = Shape::scalar();
            out.inputs.push_back(in);
            v4_to_mine[p.v4_id] = leaf_id++;
        } else {
            v4_to_mine[p.v4_id] = compute_id;
            compute_id += p.op.result_count;
        }
    }

    // Pass 2b: emit compute ops with remapped operands + result ids.
    out.slots.resize(1);
    Slot& slot = out.slots[0];
    for (const Parsed& p : parsed) {
        if (p.kind != 2) continue;
        Op op = p.op;
        op.result_id = v4_to_mine[p.v4_id];
        remap_op_v4(op, v4_to_mine);
        slot.ops.push_back(op);
    }

    // Output[]: value:u32 | kind:u8.
    slot.outputs.reserve(n_outputs);
    for (std::uint32_t i = 0; i < n_outputs; ++i) {
        std::uint32_t val = c.u32();
        std::uint8_t kind = c.u8();
        if (!c.ok || kind > 6) { if (err) { err->code = DecodeError::UnknownTag; err->detail = "output kind"; } return false; }
        Output o; o.value = (val < v4_to_mine.size() ? v4_to_mine[val] : val);
        o.kind = static_cast<OutputKind>(kind);
        slot.outputs.push_back(o);
    }

    return c.ok;
}


bool read_slot(Cursor& c, std::uint32_t n_inputs, Slot& out, DecodeError* err) {
    std::uint32_t n_ops = c.u32();
    if (!c.ok) { if (err) { err->code = DecodeError::UnexpectedEof; err->detail = "slot n_ops"; } return false; }
    out.ops.reserve(n_ops);

    // SSA counter: resets to n_inputs at the start of each slot (§3).
    std::uint32_t next_id = n_inputs;
    for (std::uint32_t i = 0; i < n_ops; ++i) {
        std::uint8_t tag = c.u8();
        if (!c.ok) { if (err) { err->code = DecodeError::UnexpectedEof; err->detail = "op tag"; } return false; }
        OpCode code;
        if (!is_known_opcode(tag, code)) {
            if (err) { err->code = DecodeError::UnknownOpcode; err->detail = "op tag " + std::to_string(tag); }
            return false;
        }
        Op op;
        if (!read_op_operands(c, code, op, err)) {
            if (err && err->code == DecodeError::None) { err->code = DecodeError::UnexpectedEof; err->detail = "op operands"; }
            return false;
        }
        op.result_count = op_result_count(code);
        op.result_id = next_id;
        next_id += op.result_count;
        out.ops.push_back(op);
    }

    std::uint32_t n_outputs = c.u32();
    if (!c.ok) { if (err) { err->code = DecodeError::UnexpectedEof; err->detail = "slot n_outputs"; } return false; }
    out.outputs.reserve(n_outputs);
    for (std::uint32_t i = 0; i < n_outputs; ++i) {
        Output o;
        o.value = c.u32();              // value id
        std::uint8_t kind = c.u8();     // kind tag (PSIR v2)
        if (!c.ok) { if (err) { err->code = DecodeError::UnexpectedEof; err->detail = "slot output"; } return false; }
        if (kind > 6) {
            if (err) { err->code = DecodeError::UnknownTag; err->detail = "output kind " + std::to_string(kind); }
            return false;
        }
        o.kind = static_cast<OutputKind>(kind);
        out.outputs.push_back(o);
    }
    return true;
}

// Structural v4 walk (no binding / no SSA remap) that locates every
// `PivotThreshold` `RankLe(k)` predicate: records the byte offset of its u32
// payload immediate (`offsets`) and/or its value (`values`). Returns false on a
// non-v4 / malformed buffer. ONLY `RankLe` payloads are immediates — the
// `CummassLe`/`ProbGe` predicates carry value-ids (left untouched), so this is
// precise, not over-broad (matches `remap_op_v4`). Used by canonicalize_op_shape
// (k-invariance) + extract_rank_le_k (the baked top-k cutoff).
bool scan_rank_le(const std::uint8_t* data, std::size_t len,
                  std::vector<std::size_t>* offsets,
                  std::vector<std::uint32_t>* values) {
    Cursor c{data, data + len};
    if (!c.have(4) || std::memcmp(c.p, "PSIR", 4) != 0) return false;
    c.p += 4;
    std::uint16_t version = c.u16();
    (void)c.u16();  // flags
    if (!c.ok || version < 4) return false;  // op-shape canonicalization is v4-only
    std::uint32_t n_inputs = c.u32();
    std::uint32_t n_ops = c.u32();
    (void)c.u32();  // n_outputs
    if (!c.ok) return false;
    // InputDecl[]: dtype:u8 | shape_v4 — advance past, no binding needed.
    for (std::uint32_t i = 0; i < n_inputs; ++i) {
        (void)c.u8();  // dtype
        Shape sh;
        if (!read_shape_v4(c, sh)) return false;
    }
    // ops[]: classify Input(0x80)/Const(0x81)/compute, capturing RankLe payloads.
    for (std::uint32_t i = 0; i < n_ops; ++i) {
        std::uint8_t tag = c.u8();
        if (!c.ok) return false;
        if (tag == 0x80) { (void)c.u32(); continue; }                    // Input(idx)
        if (tag == 0x81) { Literal lit; if (!read_literal(c, lit)) return false; continue; }  // Const
        OpCode code;
        if (!is_known_opcode(tag, code)) return false;
        if (code == OpCode::PivotThreshold) {
            (void)c.u32();  // a (input value-id)
            std::uint8_t ptag = c.u8();
            if (!c.ok || ptag > 2) return false;
            const std::size_t payload_off = static_cast<std::size_t>(c.p - data);
            std::uint32_t payload = c.u32();
            if (!c.ok) return false;
            if (ptag == static_cast<std::uint8_t>(PredTag::RankLe)) {
                if (offsets) offsets->push_back(payload_off);
                if (values) values->push_back(payload);
            }
        } else {
            Op op;
            DecodeError ignore;
            if (!read_op_operands_v4(c, code, op, &ignore)) return false;
        }
    }
    return c.ok;
}

}  // namespace

bool decode(const std::uint8_t* data, std::size_t len, Program& out, DecodeError* err) {
    if (err) { err->code = DecodeError::None; err->detail.clear(); }
    out.inputs.clear();
    out.slots.clear();

    Cursor c{data, data + len};

    // Header (16 bytes).
    if (!c.have(4) || std::memcmp(c.p, "PSIR", 4) != 0) {
        if (err) { err->code = DecodeError::BadMagic; err->detail = "magic != PSIR"; }
        return false;
    }
    c.p += 4;
    std::uint16_t version = c.u16();
    if (!c.ok) { if (err) { err->code = DecodeError::UnexpectedEof; err->detail = "header"; } return false; }
    // Accept PSIR versions in [MIN_READ_VERSION=2, VERSION=3]; v2 streams decode
    // identically (v3 is additive: +RowBroadcast op, matrix/cross-program gates).
    if (version < 2 || version > 3) {
        if (err) { err->code = DecodeError::BadVersion; err->detail = "version " + std::to_string(version); }
        return false;
    }
    (void)c.u16();  // flags (reserved)
    std::uint32_t n_inputs = c.u32();
    std::uint32_t n_slots = c.u32();
    if (!c.ok) { if (err) { err->code = DecodeError::UnexpectedEof; err->detail = "header counts"; } return false; }

    out.inputs.reserve(n_inputs);
    for (std::uint32_t i = 0; i < n_inputs; ++i) {
        Input in;
        if (!read_input(c, i, in, err)) {
            if (err && err->code == DecodeError::None) { err->code = DecodeError::UnexpectedEof; err->detail = "input"; }
            return false;
        }
        out.inputs.push_back(in);
    }

    out.slots.reserve(n_slots);
    for (std::uint32_t i = 0; i < n_slots; ++i) {
        Slot s;
        if (!read_slot(c, n_inputs, s, err)) return false;
        out.slots.push_back(std::move(s));
    }

    return true;
}

bool decode_v4(const std::uint8_t* data, std::size_t len,
               const std::vector<Binding>& slot_bindings, Program& out, DecodeError* err) {
    if (err) { err->code = DecodeError::None; err->detail.clear(); }
    return read_program_v4(data, len, slot_bindings, out, err);
}

std::vector<std::uint8_t> canonicalize_op_shape(const std::uint8_t* data, std::size_t len) {
    std::vector<std::uint8_t> out(data, data + len);
    std::vector<std::size_t> offsets;
    // Non-v4 / malformed → return the unchanged copy (it won't match the canonical
    // table, so it falls through to CustomJIT — correct for a genuine custom).
    if (!scan_rank_le(data, len, &offsets, nullptr)) return out;
    for (std::size_t off : offsets) {
        if (off + 4 <= out.size()) {
            out[off] = out[off + 1] = out[off + 2] = out[off + 3] = 0;  // RankLe(k) → RankLe(0)
        }
    }
    return out;
}

std::optional<std::uint32_t> extract_rank_le_k(const std::uint8_t* data, std::size_t len) {
    std::vector<std::uint32_t> values;
    if (!scan_rank_le(data, len, nullptr, &values) || values.empty()) return std::nullopt;
    // TopK / TopKTopP bake exactly one RankLe (the group-uniform top-k cutoff).
    return values.front();
}

}  // namespace pie_cuda_driver::sampling_ir
