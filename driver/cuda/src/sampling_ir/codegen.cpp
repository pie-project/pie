#include "sampling_ir/codegen.hpp"

#include <cstdio>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "sampling_ir/reader.hpp"

// Sampling-IR codegen (Lane L2 / charlie, W2): lowers a decoded PSIR program
// (ir.hpp) into a DAG of fused CUDA-C kernels (codegen.hpp / delta's JIT).
//
// MVP model (design §3.3): single slot, single position, M=1 decode. Shapes are
// static (baked from the IR). Each kernel is one block of PIE_IR_BLOCK (256)
// threads (LaunchShape::OneBlockPerRow). Every SSA value is materialized to a
// typed device buffer; ops are emitted as sequential, __syncthreads-separated
// stages that call the W1 primitive prelude helpers. The bf16 intrinsic logits
// are cast to an f32 intermediate once, up front, so all downstream ops see f32.
// A kernel boundary is cut at a data-dependent barrier — a GatherRow whose row
// index is a computed (reduced) value — matching delta's tested cross-kernel
// device-pointer hand-off.

namespace pie_cuda_driver::sampling_ir {

namespace {

enum class Phys { F32, I32, U32, BF16 };

const char* phys_ptr_cty(Phys p) {
    switch (p) {
        case Phys::F32: return "float*";
        case Phys::I32: return "int*";
        case Phys::U32: return "unsigned int*";
        case Phys::BF16: return "const unsigned short*";  // bf16 reinterpreted (no header)
    }
    return "float*";
}

DType phys_to_dtype(Phys p) {
    switch (p) {
        case Phys::F32: return DType::F32;
        case Phys::I32: return DType::I32;
        case Phys::U32: return DType::U32;
        case Phys::BF16: return DType::F32;  // external; size irrelevant (not allocated)
    }
    return DType::F32;
}

Phys dtype_to_phys(DType d) {
    switch (d) {
        case DType::F32: return Phys::F32;
        case DType::I32: return Phys::I32;
        case DType::U32: return Phys::U32;
        case DType::Bool: return Phys::F32;  // bool stored as 0.0f/1.0f
    }
    return Phys::F32;
}

std::string fmt_f32(float v) {
    // Non-finite literals can't be written as a decimal token; emit the exact
    // bit pattern via the device intrinsic so the value round-trips precisely.
    if (!(v == v)) return "(__int_as_float(0x7fc00000))";          // NaN
    if (v == std::numeric_limits<float>::infinity()) return "(__int_as_float(0x7f800000))";
    if (v == -std::numeric_limits<float>::infinity()) return "(__int_as_float(0xff800000))";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    std::string s(buf);
    // %.9g of an integral value (e.g. 1.0 -> "1") lacks a '.'/'e', so "1f" would
    // be an invalid user-defined literal. Ensure a decimal point before the 'f'.
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos)
        s += ".0";
    return s + "f";
}

// Per-SSA-value lowering info.
struct VInfo {
    ValueType ty;
    bool is_const = false;
    Literal lit;
    BufferId buffer = 0;   // valid iff !is_const
    Phys phys = Phys::F32;
    bool scalar = false;   // Shape::Scalar
    std::uint32_t len = 1; // last-axis length (1 for scalar)
    std::uint32_t rows = 1;     // matrix row count (1 for scalar/vector/indices)
    bool row_indexed = false;   // one element per matrix row (produced by a
                                // per-row reduce/argmax of a Matrix). In an
                                // R-block kernel it is indexed by blockIdx.x.
};

struct Codegen {
    const Program& prog;
    const Slot& slot;
    std::vector<ValueType> vtype;     // indexed by SSA value id
    std::vector<VInfo> vinfo;         // indexed by SSA value id
    std::vector<BufferDecl> buffers;  // by BufferId == index
    std::vector<Phys> buf_phys;       // by BufferId
    std::uint32_t logits_value = UINT32_MAX;  // SSA id of intrinsic logits, if any
    BufferId logits_raw = UINT32_MAX;          // external bf16 buffer
    BufferId logits_f32 = UINT32_MAX;          // f32 cast intermediate (when materialized)
    bool inline_logits_cast = false;           // fold bf16→f32 into consumer loops
    bool batched = false;                      // M>1: one block per batch row
    bool ambient_seed = false;                 // v4 Model B: RNG seed from RowSeed buffer
    BufferId row_seed_buffer = UINT32_MAX;     // ambient per-row seed S (external, bind-only)
    std::vector<std::uint32_t> use_count;      // consumers per SSA value (ops + outputs)
    std::vector<bool> inlined;                 // value folded into its consumer (no buffer/write)
    // Passthrough outputs: a program output whose value is an input leaf (never
    // an op result), e.g. grammar_submit_with_logits' raw-logits output aliases
    // the IntrinsicLogits input. The op-result loop never materializes these, so
    // each is copied — RAW, in the input's native storage dtype (bf16 for logits,
    // honoring #28's no-f32-[vocab] rule; the host converts on readback) — into
    // its own BufferClass::Output buffer by a tail kernel.
    std::vector<std::pair<BufferId, BufferId>> passthrough_outputs;  // (source buffer, output buffer)
    bool ok = true;
    std::string err;

    Codegen(const Program& p, bool batched_ = false)
        : prog(p), slot(p.slots.front()), batched(batched_), ambient_seed(p.ambient_seed) {}

    void fail(const std::string& m) { if (ok) { ok = false; err = m; } }

    // ---- fusion analysis (map→reduce / map→map) -----------------------------
    // An op is elementwise-inlinable if it produces one value per element from
    // its operands without cross-element work. Such a value, if consumed exactly
    // once and not a program output, is folded into its consumer's loop instead
    // of being materialized to a global buffer — eliminating a full-vocab global
    // round-trip (the B≥128 traffic that puts IR behind the fused hardwired
    // sampler). Only enabled for the batched (M>1) path, where it matters; the
    // M=1 / matrix-spec paths keep materializing (already cache-friendly / small).
    static bool is_elementwise(OpCode c) {
        switch (c) {
            case OpCode::Exp: case OpCode::Log: case OpCode::Neg: case OpCode::Recip:
            case OpCode::Abs: case OpCode::Sign:
            case OpCode::Add: case OpCode::Sub: case OpCode::Mul: case OpCode::Div:
            case OpCode::MaxElem: case OpCode::MinElem:
            case OpCode::Gt: case OpCode::Ge: case OpCode::Eq:
            case OpCode::Select:
            case OpCode::Broadcast: case OpCode::RowBroadcast:
            case OpCode::Rng:
            case OpCode::MaskApply:
                return true;
            default:
                return false;
        }
    }

    const Op* producer(ValueId id) {
        for (const Op& op : slot.ops)
            if (id >= op.result_id && id < op.result_id + op.result_count) return &op;
        return nullptr;
    }

    // Ops that read all their value operands through elem_expr (so an inlined
    // operand folds in correctly). Everything else (pivot/scan/sort/gather/
    // scatter/rowbroadcast) needs a materialized buffer pointer for its input.
    static bool consumer_accepts_inline(OpCode c) {
        switch (c) {
            case OpCode::Exp: case OpCode::Log: case OpCode::Neg: case OpCode::Recip:
            case OpCode::Abs: case OpCode::Sign:
            case OpCode::Add: case OpCode::Sub: case OpCode::Mul: case OpCode::Div:
            case OpCode::MaxElem: case OpCode::MinElem:
            case OpCode::Gt: case OpCode::Ge: case OpCode::Eq:
            case OpCode::Select: case OpCode::Broadcast:
            case OpCode::ReduceSum: case OpCode::ReduceMax: case OpCode::ReduceMin:
            case OpCode::ReduceArgmax:
                return true;
            default:
                return false;
        }
    }

    void analyze_fusion() {
        std::uint32_t total = static_cast<std::uint32_t>(vtype.size());
        use_count.assign(total, 0);
        for (const Op& op : slot.ops)
            for (ValueId v : op_operands(op)) ++use_count[v];
        for (const Output& o : slot.outputs) ++use_count[o.value];

        inlined.assign(total, false);

        // Determine the intrinsic-logits value id (the input bound to
        // Intrinsic(Logits)), needed for the inline-cast gate below.
        for (std::uint32_t i = 0; i < prog.n_inputs(); ++i)
            if (prog.inputs[i].binding.tag == BindingTag::Intrinsic) logits_value = i;

        // Inline bf16→f32 logits cast: fold the cast into consuming loops (read
        // bf16 in registers) instead of materializing a full-vocab f32 buffer —
        // valid iff every consumer of the logits value reads it through elem_expr
        // (elementwise/reduction). A pivot/scan/sort/gather consumer rescans the
        // row, so it needs the materialized f32; keep the cast there. Gated to the
        // batched (M>1) path: it uses elem_expr-based fused reductions, so the
        // inlined cast threads through; the M=1 vector path reduces via a buffer
        // pointer (no element hook) and keeps materializing. This makes a batched
        // argmax a single bf16-read pass == hardwired (closes the B≥128 tail).
        if (batched && logits_value != UINT32_MAX) {
            bool all_inline = true;
            for (const Op& op : slot.ops)
                for (ValueId v : op_operands(op))
                    if (v == logits_value && !consumer_accepts_inline(op.code)) all_inline = false;
            for (const Output& o : slot.outputs)
                if (o.value == logits_value) all_inline = false;  // direct output must materialize
            inline_logits_cast = all_inline;
        }

        if (!batched) return;   // map-chain fusion only on the M>1 path
        std::unordered_map<ValueId, bool> is_output;
        for (const Output& o : slot.outputs) is_output[o.value] = true;

        // Map each single-use value to the op that consumes it.
        std::vector<const Op*> consumer(total, nullptr);
        for (const Op& op : slot.ops)
            for (ValueId v : op_operands(op))
                if (v < total) consumer[v] = &op;

        for (const Op& op : slot.ops) {
            if (op.result_count != 1) continue;            // multi-result never inlined
            ValueId r = op.result_id;
            if (!is_elementwise(op.code)) continue;
            if (use_count[r] != 1) continue;               // duplicating would recompute
            if (is_output.count(r)) continue;              // outputs must materialize
            const Op* c = consumer[r];
            if (!c || !consumer_accepts_inline(c->code)) continue;  // consumer needs a buffer
            // RowBroadcast reads its per_row operand by row-index, not elem_expr;
            // Rng's operand is a seed (never an op result). Both excluded above.
            inlined[r] = true;
        }
    }

    // Value-ids an op directly reads (mirrors pie_sampling_ir Op::operands():
    // immediates like RowBroadcast.len are excluded; predicate value-id payloads
    // included). Used for the late-input first-use barrier analysis.
    std::vector<ValueId> op_operands(const Op& op) {
        switch (op.code) {
            case OpCode::Select:
            case OpCode::ScatterAdd: case OpCode::ScatterSet:
                return {op.a, op.b, op.c};
            case OpCode::Add: case OpCode::Sub: case OpCode::Mul: case OpCode::Div:
            case OpCode::MaxElem: case OpCode::MinElem:
            case OpCode::Gt: case OpCode::Ge: case OpCode::Eq:
            case OpCode::Gather: case OpCode::GatherRow: case OpCode::GatherCols:
            case OpCode::MaskApply:   // (logits, mask)
                return {op.a, op.b};
            case OpCode::PivotThreshold:
                // #25: all three predicates carry a value-id payload (RankLe `k`
                // included — host-submit, no longer a baked immediate).
                return {op.a, op.predicate.payload};
            case OpCode::Rng:
                // v4 ambient (Model B): no value-id operand (seed is launch-side).
                // v2/v3: op.a is the Host seed input.
                return ambient_seed ? std::vector<ValueId>{} : std::vector<ValueId>{op.a};
            default:
                return {op.a};  // unary / reduce / scan / sort / broadcast / rowbcast
        }
    }

    BufferId new_buffer(BufferClass cls, Phys phys, std::uint32_t elem_count,
                        std::uint32_t input_id = 0, std::uint32_t output_index = 0,
                        OutputKind output_kind = OutputKind::Token) {
        BufferId id = static_cast<BufferId>(buffers.size());
        BufferDecl b;
        b.id = id;
        b.cls = cls;
        b.dtype = phys_to_dtype(phys);
        b.elem_count = elem_count;
        b.input_id = input_id;
        b.output_index = output_index;
        b.output_kind = output_kind;
        b.batched = batched;   // M>1: per-row elem_count, JIT sizes num_rows×
        buffers.push_back(b);
        buf_phys.push_back(phys);
        return id;
    }

    // ---- type inference (BYTECODE.md §4) -----------------------------------
    static Shape broadcast(const Shape& a, const Shape& b) {
        if (a.tag == ShapeTag::Scalar) return b;
        return a;  // b scalar or equal
    }

    void infer_types() {
        std::uint32_t n = prog.n_inputs();
        std::uint32_t total = n;
        for (const Op& op : slot.ops) total += op.result_count;
        vtype.assign(total, ValueType{});
        for (std::uint32_t i = 0; i < n; ++i) vtype[i] = prog.inputs[i].ty;

        for (const Op& op : slot.ops) {
            ValueId r = op.result_id;
            auto sh = [&](ValueId v) { return vtype[v].shape; };
            auto dt = [&](ValueId v) { return vtype[v].dtype; };
            switch (op.code) {
                case OpCode::Exp: case OpCode::Log: case OpCode::Recip:
                    vtype[r] = {sh(op.a), DType::F32}; break;
                case OpCode::Neg: case OpCode::Abs: case OpCode::Sign:
                    vtype[r] = {sh(op.a), dt(op.a)}; break;
                case OpCode::Add: case OpCode::Sub: case OpCode::Mul:
                case OpCode::MaxElem: case OpCode::MinElem:
                    vtype[r] = {broadcast(sh(op.a), sh(op.b)), dt(op.a)}; break;
                case OpCode::Div:
                    vtype[r] = {broadcast(sh(op.a), sh(op.b)), DType::F32}; break;
                case OpCode::MaskApply:
                    // result ≡ logits SHAPE (op.a); mask (op.b) is [ceil(len/32)] u32,
                    // word-indexed — NOT broadcast with logits, so it never shapes the
                    // result. dtype = F32 (the codegen's working logit dtype; the bf16
                    // `0xFF80` contract is the storage view on a final bf16 output write).
                    vtype[r] = {sh(op.a), DType::F32}; break;
                case OpCode::Gt: case OpCode::Ge: case OpCode::Eq:
                    vtype[r] = {broadcast(sh(op.a), sh(op.b)), DType::Bool}; break;
                case OpCode::Select:
                    vtype[r] = {broadcast(broadcast(sh(op.a), sh(op.b)), sh(op.c)), dt(op.b)}; break;
                case OpCode::ReduceSum: case OpCode::ReduceMax: case OpCode::ReduceMin: {
                    Shape s = sh(op.a);
                    Shape o = (s.tag == ShapeTag::Matrix) ? Shape::vector(s.a) : Shape::scalar();
                    vtype[r] = {o, dt(op.a)}; break;
                }
                case OpCode::ReduceArgmax: {
                    Shape s = sh(op.a);
                    Shape o = (s.tag == ShapeTag::Matrix) ? Shape::vector(s.a) : Shape::scalar();
                    vtype[r] = {o, DType::I32}; break;
                }
                case OpCode::Broadcast:
                    vtype[r] = {op.shape, dt(op.a)}; break;
                case OpCode::RowBroadcast: {
                    // per_row Vector{rows} → Matrix{rows, len=imm}, dtype = per_row's.
                    std::uint32_t rows = sh(op.a).last_len();
                    vtype[r] = {Shape::matrix(rows, op.imm), dt(op.a)}; break;
                }
                case OpCode::CumSum: case OpCode::CumProd:
                    vtype[r] = {sh(op.a), DType::F32}; break;
                case OpCode::SortDesc: {
                    std::uint32_t len = sh(op.a).last_len();
                    vtype[r] = {Shape::vector(len), DType::F32};
                    vtype[r + 1] = {Shape::indices(len), DType::U32};
                    break;
                }
                case OpCode::PivotThreshold:
                    vtype[r] = {sh(op.a), DType::Bool}; break;
                case OpCode::Gather:
                    vtype[r] = {Shape::vector(sh(op.b).last_len()), dt(op.a)}; break;
                case OpCode::GatherRow:
                    vtype[r] = {Shape::vector(sh(op.a).last_len()), dt(op.a)}; break;
                case OpCode::GatherCols:
                    // src Matrix{rows,len}, idx Vector{rows} → Vector{rows}.
                    vtype[r] = {Shape::vector(sh(op.a).rows()), dt(op.a)}; break;
                case OpCode::ScatterAdd: case OpCode::ScatterSet:
                    vtype[r] = {Shape::vector(sh(op.a).last_len()), dt(op.a)}; break;
                case OpCode::Rng:
                    vtype[r] = {op.shape, DType::F32}; break;
            }
        }
    }

    // ---- buffer allocation --------------------------------------------------
    void build_buffers() {
        std::uint32_t n = prog.n_inputs();
        vinfo.assign(vtype.size(), VInfo{});

        // Inputs.
        for (std::uint32_t i = 0; i < n; ++i) {
            const Input& in = prog.inputs[i];
            VInfo& vi = vinfo[i];
            vi.ty = in.ty;
            vi.scalar = in.ty.shape.is_scalar();
            vi.len = in.ty.shape.last_len();
            vi.rows = in.ty.shape.rows();
            switch (in.binding.tag) {
                case BindingTag::Const:
                    vi.is_const = true;
                    vi.lit = in.binding.lit;
                    break;
                case BindingTag::Intrinsic: {
                    // Raw bf16 logits (external) + (unless inlined) an f32 cast
                    // intermediate. For a Matrix{rows,len} intrinsic (spec-verify
                    // block) both buffers hold rows*len elements, row-major.
                    logits_value = i;
                    std::uint32_t ec = vi.rows * vi.len;
                    logits_raw = new_buffer(BufferClass::IntrinsicLogits, Phys::BF16, ec, i);
                    // Carry the intrinsic kind (Logits | MtpLogits) so delta's
                    // jit_backend wire can route MtpLogits to the draft row.
                    buffers[logits_raw].intrinsic_kind = in.binding.intrinsic;
                    if (inline_logits_cast) {
                        // No materialized f32 — consumers read bf16 inline (cast
                        // in registers via melem_leaf's logits special-case).
                        vi.buffer = logits_raw;
                        vi.phys = Phys::F32;   // logically f32 to consumers
                    } else {
                        logits_f32 = new_buffer(BufferClass::Intermediate, Phys::F32, ec);
                        vi.buffer = logits_f32;   // consumers read the f32 cast
                        vi.phys = Phys::F32;
                    }
                    break;
                }
                case BindingTag::Host: {
                    Phys ph = dtype_to_phys(in.ty.dtype);
                    BufferClass cls = (in.binding.host_avail == HostAvailability::LateBound)
                                          ? BufferClass::HostLate : BufferClass::HostSubmit;
                    std::uint32_t ec = in.ty.shape.tag == ShapeTag::Matrix
                                           ? in.ty.shape.a * in.ty.shape.b
                                           : in.ty.shape.last_len();
                    vi.buffer = new_buffer(cls, ph, ec, i);
                    vi.phys = ph;
                    break;
                }
                case BindingTag::Output:
                    fail("Output-bound inputs (cross-program) not supported in MVP");
                    break;
            }
        }
        if (!ok) return;

        // Map slot outputs → (output index, kind).
        std::unordered_map<ValueId, std::uint32_t> output_index;
        std::unordered_map<ValueId, OutputKind> output_kind;
        for (std::uint32_t i = 0; i < slot.outputs.size(); ++i) {
            output_index[slot.outputs[i].value] = i;
            output_kind[slot.outputs[i].value] = slot.outputs[i].kind;
        }

        // Op results.
        for (const Op& op : slot.ops) {
            bool is_reduce = op.code == OpCode::ReduceSum || op.code == OpCode::ReduceMax ||
                             op.code == OpCode::ReduceMin || op.code == OpCode::ReduceArgmax;
            bool is_gathercols = op.code == OpCode::GatherCols;
            bool matrix_operand = vinfo[op.a].rows > 1;
            for (std::uint32_t k = 0; k < op.result_count; ++k) {
                ValueId r = op.result_id + k;
                VInfo& vi = vinfo[r];
                vi.ty = vtype[r];
                vi.scalar = vtype[r].shape.is_scalar();
                vi.len = vtype[r].shape.last_len();
                vi.rows = vtype[r].shape.rows();
                vi.phys = dtype_to_phys(vtype[r].dtype);
                // A per-row reduce/argmax/gather-cols of a Matrix produces a
                // Vector{rows} written one element per block row → row-indexed.
                if ((is_reduce || is_gathercols) && matrix_operand) vi.row_indexed = true;
                // Inlined (fused) values are never materialized — no buffer.
                if (r < inlined.size() && inlined[r]) { vi.buffer = UINT32_MAX; continue; }
                std::uint32_t ec = vtype[r].shape.tag == ShapeTag::Matrix
                                       ? vtype[r].shape.a * vtype[r].shape.b
                                       : vtype[r].shape.last_len();
                auto it = output_index.find(r);
                if (it != output_index.end())
                    vi.buffer = new_buffer(BufferClass::Output, vi.phys, ec, 0, it->second,
                                           output_kind[r]);
                else
                    vi.buffer = new_buffer(BufferClass::Intermediate, vi.phys, ec);
            }
        }

        // Passthrough outputs: a program output whose value is an INPUT leaf (or
        // any value the op-result loop above didn't give a BufferClass::Output) —
        // e.g. grammar_submit_with_logits declares `output(logits, Logits)`, which
        // aliases the IntrinsicLogits input and is never an op result, so without
        // this it gets no Output buffer → the declared-vs-compiled output count
        // mismatches (n_out=1 vs 2) and the readback is dropped. Materialize a
        // dedicated Output buffer per such output; a tail kernel copies the source
        // value into it (analyze_fusion forces the logits f32-cast to materialize
        // when logits is a direct output, so the source buffer is always live).
        for (std::uint32_t oi = 0; oi < slot.outputs.size(); ++oi) {
            ValueId v = slot.outputs[oi].value;
            if (v >= vinfo.size()) continue;
            BufferId vb = vinfo[v].buffer;
            bool has_output_buffer = vb != UINT32_MAX && vb < buffers.size() &&
                                     buffers[vb].cls == BufferClass::Output;
            if (has_output_buffer) continue;   // op-result output, already materialized
            // RAW passthrough: copy the input's NATIVE storage (bf16 for the
            // intrinsic logits — #28 no-f32-[vocab] rule, the host converts on
            // readback; the host buffer's dtype otherwise), NOT the f32 view.
            BufferId src = (v == logits_value && logits_raw != UINT32_MAX) ? logits_raw : vb;
            if (src == UINT32_MAX || src >= buffers.size()) continue;
            std::uint32_t ec = vtype[v].shape.tag == ShapeTag::Matrix
                                   ? vtype[v].shape.a * vtype[v].shape.b
                                   : vtype[v].shape.last_len();
            BufferId ob = new_buffer(BufferClass::Output, buf_phys[src], ec, 0, oi,
                                     slot.outputs[oi].kind);
            passthrough_outputs.push_back({src, ob});
        }

        // Ambient RNG seed (v4 Model B): one external per-row RowSeed buffer S[N],
        // bound by the executor (LaunchArgs.row_seeds); Op::Rng reads S[r]. Created
        // once iff the program draws RNG. No Op::Input/value-id — it's launch-side.
        if (ambient_seed) {
            for (const Op& op : slot.ops) {
                if (op.code == OpCode::Rng) {
                    row_seed_buffer = new_buffer(BufferClass::RowSeed, Phys::U32, 1);
                    break;
                }
            }
        }
    }

    // ---- partitioning (fusion): cut at geometry changes + data barriers -----
    // Each op runs with a "block count": 1 for vector/scalar ops (single block),
    // or `rows` for a per-row matrix op (reduce/scan/argmax over a Matrix, or an
    // elementwise op producing a Matrix). A kernel is one geometry, so a change
    // in block-count cuts a boundary — e.g. a matrix per-row argmax (grid=rows)
    // feeding vector ops (grid=1) becomes two kernels, the matrix result vector
    // crossing the boundary via its device buffer. The GatherRow data-dependent
    // barrier (reduce result used as a row index) also cuts.
    std::uint32_t op_block_rows(const Op& op) {
        switch (op.code) {
            case OpCode::ReduceSum: case OpCode::ReduceMax: case OpCode::ReduceMin:
            case OpCode::ReduceArgmax: case OpCode::CumSum: case OpCode::CumProd:
            case OpCode::GatherCols:
                return vinfo[op.a].rows > 1 ? vinfo[op.a].rows : 1;
            default:
                return vinfo[op.result_id].rows > 1 ? vinfo[op.result_id].rows : 1;
        }
    }

    // A kernel is "matrix per-row" (grid=rows, emit_op_matrix path) iff it carries
    // matrix-shaped work — distinguished by SHAPE CLASS, not row count, so that a
    // Matrix{1,N} op at k=1 is NOT merged into a single-block vector kernel (it
    // still needs the per-row emit path). Matrix iff: a matrix-only op
    // (RowBroadcast/GatherCols), a Matrix result, or a per-row reduce/gather
    // result (row_indexed — one element per matrix row).
    bool op_is_matrix(const Op& op) {
        if (op.code == OpCode::RowBroadcast || op.code == OpCode::GatherCols) return true;
        const VInfo& rv = vinfo[op.result_id];
        return rv.ty.shape.tag == ShapeTag::Matrix || rv.row_indexed;
    }

    std::vector<std::vector<std::uint32_t>> partition() {
        std::vector<std::vector<std::uint32_t>> groups;
        groups.emplace_back();
        std::uint32_t n = prog.n_inputs();
        // Batched (M>1): every op is per-batch-row → one kernel, grid=num_rows.
        // (Core samplers have no cross-row barrier; a GatherRow-at-computed-index
        // would need special handling — reject it for batched lowering.)
        if (batched) {
            for (std::uint32_t i = 0; i < slot.ops.size(); ++i) groups.back().push_back(i);
            return groups;
        }
        std::uint32_t cur_geom = 0;  // 0 = unset
        for (std::uint32_t i = 0; i < slot.ops.size(); ++i) {
            const Op& op = slot.ops[i];
            std::uint32_t rows_geom = op_block_rows(op);
            // Tag matrix-shaped ops so a matrix↔vector transition cuts a kernel
            // boundary even when rows==1 (k=1) — a single-row matrix kernel still
            // uses the per-row emit path and can't share a vector kernel.
            std::uint32_t geom = op_is_matrix(op) ? (rows_geom | 0x80000000u) : rows_geom;
            bool barrier = (op.code == OpCode::GatherRow) && (op.b >= n);
            bool geom_change = (cur_geom != 0 && geom != cur_geom);
            if ((barrier || geom_change) && !groups.back().empty()) groups.emplace_back();
            groups.back().push_back(i);
            cur_geom = geom;
        }
        return groups;
    }

    // ---- late-input first-consuming-kernel stamp (echo/delta perf barrier) ---
    // For each HostLate input buffer, find the first op (in op order) that
    // directly reads its value-id, map that op to its kernel group, and stamp
    // `first_consuming_kernel` on the BufferDecl. This is the device-side mirror
    // of alpha's Rust `late_values()` first-use analysis — delta's split-launch
    // injects the late value just before that kernel. Correctness does not
    // depend on it (single-stream H2D→DAG is ordered); it's the overlap-upstream
    // perf refinement. Output-ref inputs (cross-program) get the same treatment
    // when supported.
    void stamp_late_buffers(const std::vector<std::vector<std::uint32_t>>& groups) {
        // op index → kernel group index.
        std::vector<std::uint32_t> op_kernel(slot.ops.size(), 0);
        for (std::uint32_t g = 0; g < groups.size(); ++g)
            for (std::uint32_t oi : groups[g]) op_kernel[oi] = g;

        for (std::uint32_t in = 0; in < prog.n_inputs(); ++in) {
            if (prog.inputs[in].binding.tag != BindingTag::Host) continue;
            if (prog.inputs[in].binding.host_avail != HostAvailability::LateBound) continue;
            BufferId b = vinfo[in].buffer;
            // first op that reads input value-id `in`.
            for (std::uint32_t oi = 0; oi < slot.ops.size(); ++oi) {
                bool reads = false;
                for (ValueId v : op_operands(slot.ops[oi])) if (v == in) { reads = true; break; }
                if (reads) { buffers[b].first_consuming_kernel = op_kernel[oi]; break; }
            }
        }
    }
    std::string ref_f32(ValueId id, const std::string& j) {
        const VInfo& vi = vinfo[id];
        if (vi.is_const) {
            switch (vi.lit.dtype) {
                case DType::F32: return fmt_f32(vi.lit.as_f32());
                case DType::I32: return "(float)(" + std::to_string(vi.lit.as_i32()) + ")";
                case DType::U32: return "(float)(" + std::to_string(vi.lit.as_u32()) + "u)";
                case DType::Bool: return vi.lit.as_bool() ? "1.0f" : "0.0f";
            }
        }
        std::string idx = vi.scalar ? "0" : j;
        // Inlined bf16 logits cast (read raw bf16, convert in registers).
        if (inline_logits_cast && id == logits_value)
            return "pie_ir_bf16_to_f32(b" + std::to_string(logits_raw) + "[" + idx + "])";
        std::string e = "b" + std::to_string(vi.buffer) + "[" + idx + "]";
        if (vi.phys == Phys::I32 || vi.phys == Phys::U32) return "(float)(" + e + ")";
        return e;
    }

    std::string buf_name(ValueId id) { return "b" + std::to_string(vinfo[id].buffer); }

    // ---- matrix emission (one block per row; r = blockIdx.x) ----------------
    // `cur_rows` > 1 selects the matrix path: each block handles one matrix row.
    std::uint32_t cur_rows = 1;

    // Float element access for value `id` at (row r, col j) inside a per-row
    // kernel. In batched (M>1) mode every value is per-batch-row: scalar → [r],
    // vector{len} → [r*len+j]. In matrix mode: row-indexed vectors index by r;
    // matrices by r*len+j; scalars by 0; plain vectors by j (shared across rows).
    // Leaf per-element read of value `id` (const or materialized buffer) inside
    // a per-row kernel. In batched (M>1) mode every value is per-batch-row:
    // scalar → [r], vector{len} → [r*len+j]. In matrix mode: row-indexed vectors
    // index by r; matrices by r*len+j; scalars by 0; plain vectors by j.
    std::string melem_leaf(ValueId id) {
        const VInfo& vi = vinfo[id];
        if (vi.is_const) {
            switch (vi.lit.dtype) {
                case DType::F32: return fmt_f32(vi.lit.as_f32());
                case DType::I32: return "(float)(" + std::to_string(vi.lit.as_i32()) + ")";
                case DType::U32: return "(float)(" + std::to_string(vi.lit.as_u32()) + "u)";
                case DType::Bool: return vi.lit.as_bool() ? "1.0f" : "0.0f";
            }
        }
        std::string idx;
        if (batched) {
            idx = vi.scalar ? "r" : ("r*" + std::to_string(vi.len) + "+j");
        } else if (vi.scalar) idx = "0";
        else if (vi.row_indexed) idx = "r";
        else if (vi.rows > 1) idx = "r*" + std::to_string(vi.len) + "+j";
        else idx = "j";
        // Inlined bf16 logits cast (read raw bf16, convert in registers).
        if (inline_logits_cast && id == logits_value)
            return "pie_ir_bf16_to_f32(b" + std::to_string(logits_raw) + "[" + idx + "])";
        std::string e = "b" + std::to_string(vi.buffer) + "[" + idx + "]";
        if (vi.phys == Phys::I32 || vi.phys == Phys::U32) return "(float)(" + e + ")";
        return e;
    }

    // Integer element access for value `id` — the analog of `melem_leaf` for an
    // integer value (e.g. the top-k `k` predicate operand), returning the RAW
    // integer (no float cast). A predicate `k` is a scalar (→ "0"/"r" batched) or
    // a per-row `[rows]` vector (→ "r"); it is never per-column, so `j` is never
    // referenced in the reduce-style (per-block) site where this is emitted.
    std::string melem_int(ValueId id) {
        const VInfo& vi = vinfo[id];
        if (vi.is_const) {
            if (vi.lit.dtype == DType::U32) return std::to_string(vi.lit.as_u32()) + "u";
            return std::to_string(vi.lit.as_i32());
        }
        std::string idx;
        if (batched) {
            idx = vi.scalar ? "r" : ("r*" + std::to_string(vi.len) + "+j");
        } else if (vi.scalar) idx = "0";
        else if (vi.row_indexed) idx = "r";
        else if (vi.rows > 1) idx = "r*" + std::to_string(vi.len) + "+j";
        else idx = "j";
        return "b" + std::to_string(vi.buffer) + "[" + idx + "]";
    }
    // producers marked `inlined` (folding the map chain into the consumer's
    // loop, no global round-trip); otherwise a leaf read. For non-batched paths
    // `inlined` is all-false so this is exactly `melem_leaf`.
    std::string elem_expr(ValueId id) {
        if (id < inlined.size() && inlined[id]) {
            const Op* op = producer(id);
            if (op) return op_compute_expr(*op);
        }
        return melem_leaf(id);
    }
    // Kept name for existing call sites — now the recursive form.
    std::string melem(ValueId id) { return elem_expr(id); }

    // The per-element RHS expression of an elementwise op (in terms of
    // elem_expr(operands)) — the single source of truth for both inlining and
    // the materialized write.
    std::string op_compute_expr(const Op& op) {
        auto un = [&](const char* fn){ return std::string(fn) + "(" + elem_expr(op.a) + ")"; };
        auto bn = [&](const char* fn){ return std::string(fn) + "(" + elem_expr(op.a) + ", " + elem_expr(op.b) + ")"; };
        auto cm = [&](const char* c){ return "((" + elem_expr(op.a) + " " + c + " " + elem_expr(op.b) + ") ? 1.0f : 0.0f)"; };
        switch (op.code) {
            case OpCode::Exp:   return un("pie_ir_exp");
            case OpCode::Log:   return un("pie_ir_log");
            case OpCode::Neg:   return un("pie_ir_neg");
            case OpCode::Recip: return un("pie_ir_recip");
            case OpCode::Abs:   return un("pie_ir_abs");
            case OpCode::Sign:  return un("pie_ir_sign");
            case OpCode::Add:     return bn("pie_ir_add");
            case OpCode::Sub:     return bn("pie_ir_sub");
            case OpCode::Mul:     return bn("pie_ir_mul");
            case OpCode::Div:     return bn("pie_ir_div");
            case OpCode::MaxElem: return bn("pie_ir_max");
            case OpCode::MinElem: return bn("pie_ir_min");
            case OpCode::Gt: return cm(">");
            case OpCode::Ge: return cm(">=");
            case OpCode::Eq: return cm("==");
            case OpCode::Select:
                return "((" + elem_expr(op.a) + " != 0.0f) ? " + elem_expr(op.b) + " : " + elem_expr(op.c) + ")";
            case OpCode::Broadcast:
                return elem_expr(op.a);                       // scalar replicated
            case OpCode::MaskApply: {
                // out = bit_j(mask) ? logits : -inf. mask (op.b) = packed u32,
                // word-indexed (mask[j>>5] bit j&31); logits (op.a) = f32 elem.
                // In batched, each sequence carries its own grammar mask → r*len.
                const VInfo& mv = vinfo[op.b];
                std::string maskbase = "b" + std::to_string(mv.buffer);
                if (batched) maskbase += " + r*" + std::to_string(mv.len);
                return "pie_ir_mask_apply(" + maskbase + ", j, " + elem_expr(op.a) + ")";
            }
            case OpCode::RowBroadcast: {
                const VInfo& pv = vinfo[op.a];                // Vector{rows} per-row
                if (pv.is_const) return melem_leaf(op.a);
                std::string e = "b" + std::to_string(pv.buffer) + "[r]";
                if (pv.phys == Phys::I32 || pv.phys == Phys::U32) e = "(float)(" + e + ")";
                return e;
            }
            case OpCode::Rng: {
                const char* fn = (op.rng_kind == RngKind::Gumbel) ? "pie_ir_gumbel" : "pie_ir_hash_uniform";
                std::uint32_t L = vinfo[op.result_id].len;
                std::string col = batched ? "j" : ("r*" + std::to_string(L) + "+j");
                if (ambient_seed) {  // v4 Model B: ambient per-row seed S[r] + static stream
                    std::string ridx = batched ? "r" : "0";
                    return std::string(fn) + "(pie_ir_seed_eff_stream(b" + std::to_string(row_seed_buffer)
                         + "[" + ridx + "], " + std::to_string(op.stream) + "u), " + col + ")";
                }
                std::string seed;
                if (batched && !vinfo[op.a].is_const)
                    seed = "b" + std::to_string(vinfo[op.a].buffer) + "[r]";
                else
                    seed = "(unsigned int)" + ref_int(op.a);
                return std::string(fn) + "(pie_ir_seed_eff(" + seed + "), " + col + ")";
            }
            default:
                return melem_leaf(op.result_id);  // unreachable (only elementwise inlined)
        }
    }

    // Write-target index expression for a per-row-kernel result value.
    std::string mwidx(ValueId id) {
        const VInfo& vi = vinfo[id];
        if (batched) return vi.scalar ? "r" : ("r*" + std::to_string(vi.len) + "+j");
        if (vi.row_indexed) return "r";
        if (vi.rows > 1) return "r*" + std::to_string(vi.len) + "+j";
        return "j";
    }

    void emit_op_matrix(std::ostringstream& s, const Op& op) {
        ValueId r = op.result_id;
        std::uint32_t L = vinfo[r].len;
        // Inlined (fused) elementwise op: its value is folded into the consumer
        // via elem_expr — emit nothing, materialize nothing.
        if (r < inlined.size() && inlined[r]) return;

        // Materialized elementwise write: one expression, value via op_compute_expr
        // (operands fold in any inlined producers).
        auto mat_elem = [&]() {
            s << "  for (int j = tid; j < " << L << "; j += 256) " << buf_name(r)
              << "[" << mwidx(r) << "] = " << op_compute_expr(op) << ";\n  __syncthreads();\n";
        };
        // Fused per-row reduction over the input row: compute each element's
        // value inline (elem_expr — folds the whole map chain, no materialized
        // input buffer) and reduce the thread-local partial.
        std::uint32_t Lin = vinfo[op.a].len;
        // Per-row reduce/argmax write index. Each block (row r) collapses its
        // input row to exactly ONE value, so it writes the row's single slot: "r".
        // NOT mwidx() — that is the elementwise "r*len+j", whose per-element j has
        // no surrounding loop in a single-thread reduce write, so emitting it
        // yields an undefined `j` (a k=1 single-row matrix lowered batched gives
        // "r*1+j" → NVRTC compile failure). Matrix row_indexed results already map
        // to "r"; this makes the batched per-row reduce do the same.
        const std::string rrow = "r";
        auto reduce_fused = [&](const char* init, const char* acc, const char* reducer) {
            s << "  { float _acc = " << init << ";\n"
              << "    for (int j = tid; j < " << Lin << "; j += 256) { float _v = " << elem_expr(op.a)
              << "; " << acc << " }\n"
              << "    float _res = " << reducer << "(_acc);\n"
              << "    if (tid == 0) " << buf_name(r) << "[" << rrow << "] = _res; __syncthreads(); }\n";
        };
        switch (op.code) {
            case OpCode::Exp: case OpCode::Log: case OpCode::Neg: case OpCode::Recip:
            case OpCode::Abs: case OpCode::Sign:
            case OpCode::Add: case OpCode::Sub: case OpCode::Mul: case OpCode::Div:
            case OpCode::MaxElem: case OpCode::MinElem:
            case OpCode::Gt: case OpCode::Ge: case OpCode::Eq:
            case OpCode::Select:
                mat_elem(); break;
            case OpCode::ReduceSum: reduce_fused("0.0f", "_acc += _v;", "pie_ir_block_sum_reduce"); break;
            case OpCode::ReduceMax: reduce_fused("pie_ir_neg_inf()", "_acc = fmaxf(_acc, _v);", "pie_ir_block_max_reduce"); break;
            case OpCode::ReduceMin: reduce_fused("pie_ir_pos_inf()", "_acc = fminf(_acc, _v);", "pie_ir_block_min_reduce"); break;
            case OpCode::ReduceArgmax: {
                s << "  { float _best = pie_ir_neg_inf(); int _bi = 0x7fffffff;\n"
                  << "    for (int j = tid; j < " << Lin << "; j += 256) { float _v = " << elem_expr(op.a)
                  << "; if (_v > _best || (_v == _best && j < _bi)) { _best = _v; _bi = j; } }\n"
                  << "    int _res = pie_ir_block_argmax_reduce(_best, _bi);\n"
                  << "    if (tid == 0) " << buf_name(r) << "[" << rrow << "] = _res; __syncthreads(); }\n";
                break;
            }
            case OpCode::Broadcast: case OpCode::RowBroadcast: case OpCode::Rng:
                mat_elem(); break;
            case OpCode::CumSum: case OpCode::CumProd: {
                int is_prod = (op.code == OpCode::CumProd) ? 1 : 0;
                s << "  pie_ir_block_inclusive_scan(" << buf_name(op.a) << " + r*" << Lin << ", "
                  << buf_name(r) << " + r*" << L << ", " << Lin << ", " << is_prod
                  << ");\n  __syncthreads();\n";
                break;
            }
            case OpCode::PivotThreshold: {
                // Per-row sort-free threshold over the operand row slice, then a
                // per-row mask. Identical to the vector path but offset by r.
                std::uint32_t Lin = vinfo[op.a].len;
                s << "  { float _t;\n";
                switch (op.predicate.tag) {
                    case PredTag::RankLe:
                        // #25: k is a value-id (host-submit U32), read per-row.
                        s << "    _t = pie_ir_pivot_topk_radix(" << buf_name(op.a) << " + r*" << Lin
                          << ", " << Lin << ", " << melem_int(op.predicate.payload) << ");\n";
                        break;
                    case PredTag::CummassLe:
                        s << "    _t = pie_ir_pivot_topp_radix(" << buf_name(op.a) << " + r*" << Lin
                          << ", " << Lin << ", " << melem(op.predicate.payload) << ");\n";
                        break;
                    case PredTag::ProbGe:
                        s << "    _t = " << melem(op.predicate.payload) << ";\n";
                        break;
                }
                s << "    __syncthreads();\n"
                  << "    for (int j = tid; j < " << Lin << "; j += 256) " << buf_name(r)
                  << "[r*" << Lin << "+j] = (" << buf_name(op.a) << "[r*" << Lin
                  << "+j] >= _t) ? 1.0f : 0.0f;\n    __syncthreads(); }\n";
                break;
            }
            case OpCode::GatherCols: {
                // out[r] = src[r, idx[r]] ; fill-0 if idx[r] OOB. One element per
                // row (row-indexed), written by thread 0.
                std::uint32_t Lin = vinfo[op.a].len;
                std::string idx;
                const VInfo& iv = vinfo[op.b];
                if (iv.is_const) idx = std::to_string(iv.lit.as_i32());
                else {
                    idx = "b" + std::to_string(iv.buffer) + "[r]";
                    if (iv.phys == Phys::F32) idx = "(int)(" + idx + ")";
                }
                s << "  if (tid == 0) { int _c = " << idx << ";\n"
                  << "    " << buf_name(r) << "[r] = (_c >= 0 && _c < " << Lin << ") ? "
                  << buf_name(op.a) << "[r*" << Lin << "+_c] : 0.0f; }\n  __syncthreads();\n";
                break;
            }
            default:
                fail("op not supported in matrix (per-row) kernel");
                break;
        }
    }

    // ---- per-op emission ----------------------------------------------------
    void emit_unary(std::ostringstream& s, const char* fn, const Op& op) {
        ValueId r = op.result_id;
        std::uint32_t L = vinfo[r].len;
        s << "  for (int j = tid; j < " << L << "; j += 256) "
          << buf_name(r) << "[j] = " << fn << "(" << ref_f32(op.a, "j") << ");\n"
          << "  __syncthreads();\n";
    }
    void emit_binary(std::ostringstream& s, const char* fn, const Op& op) {
        ValueId r = op.result_id;
        std::uint32_t L = vinfo[r].len;
        s << "  for (int j = tid; j < " << L << "; j += 256) "
          << buf_name(r) << "[j] = " << fn << "(" << ref_f32(op.a, "j") << ", "
          << ref_f32(op.b, "j") << ");\n"
          << "  __syncthreads();\n";
    }
    void emit_cmp(std::ostringstream& s, const char* cmp, const Op& op) {
        ValueId r = op.result_id;
        std::uint32_t L = vinfo[r].len;
        s << "  for (int j = tid; j < " << L << "; j += 256) "
          << buf_name(r) << "[j] = (" << ref_f32(op.a, "j") << " " << cmp << " "
          << ref_f32(op.b, "j") << ") ? 1.0f : 0.0f;\n"
          << "  __syncthreads();\n";
    }
    void emit_reduce(std::ostringstream& s, const char* fn, const Op& op) {
        ValueId r = op.result_id;
        std::uint32_t L = vinfo[op.a].len;
        s << "  { float _v = " << fn << "(" << buf_name(op.a) << ", " << L << ");\n"
          << "    if (tid == 0) " << buf_name(r) << "[0] = _v; __syncthreads(); }\n";
    }
    void emit_op(std::ostringstream& s, const Op& op) {
        switch (op.code) {
            case OpCode::Exp:   emit_unary(s, "pie_ir_exp", op); break;
            case OpCode::Log:   emit_unary(s, "pie_ir_log", op); break;
            case OpCode::Neg:   emit_unary(s, "pie_ir_neg", op); break;
            case OpCode::Recip: emit_unary(s, "pie_ir_recip", op); break;
            case OpCode::Abs:   emit_unary(s, "pie_ir_abs", op); break;
            case OpCode::Sign:  emit_unary(s, "pie_ir_sign", op); break;
            case OpCode::Add:     emit_binary(s, "pie_ir_add", op); break;
            case OpCode::Sub:     emit_binary(s, "pie_ir_sub", op); break;
            case OpCode::Mul:     emit_binary(s, "pie_ir_mul", op); break;
            case OpCode::Div:     emit_binary(s, "pie_ir_div", op); break;
            case OpCode::MaxElem: emit_binary(s, "pie_ir_max", op); break;
            case OpCode::MinElem: emit_binary(s, "pie_ir_min", op); break;
            case OpCode::Gt: emit_cmp(s, ">", op); break;
            case OpCode::Ge: emit_cmp(s, ">=", op); break;
            case OpCode::Eq: emit_cmp(s, "==", op); break;
            case OpCode::Select: {
                ValueId r = op.result_id;
                std::uint32_t L = vinfo[r].len;
                s << "  for (int j = tid; j < " << L << "; j += 256) "
                  << buf_name(r) << "[j] = ((" << ref_f32(op.a, "j") << " != 0.0f) ? "
                  << ref_f32(op.b, "j") << " : " << ref_f32(op.c, "j") << ");\n"
                  << "  __syncthreads();\n";
                break;
            }
            case OpCode::MaskApply: {
                // Materialized (M=1): out[j] = bit_j(mask) ? logits[j] : -inf. mask
                // (op.b) packed u32 word-indexed; result F32 (downstream reads f32;
                // -inf truncates to bf16 0xFF80 only on a final bf16 output write).
                ValueId r = op.result_id;
                std::uint32_t L = vinfo[r].len;
                s << "  for (int j = tid; j < " << L << "; j += 256) "
                  << buf_name(r) << "[j] = pie_ir_mask_apply(" << buf_name(op.b) << ", j, "
                  << ref_f32(op.a, "j") << ");\n"
                  << "  __syncthreads();\n";
                break;
            }
            case OpCode::ReduceSum: emit_reduce(s, "pie_ir_block_sum", op); break;
            case OpCode::ReduceMax: emit_reduce(s, "pie_ir_block_max", op); break;
            case OpCode::ReduceMin: emit_reduce(s, "pie_ir_block_min", op); break;
            case OpCode::ReduceArgmax: {
                ValueId r = op.result_id;
                std::uint32_t L = vinfo[op.a].len;
                s << "  { int _v = pie_ir_block_argmax(" << buf_name(op.a) << ", " << L << ");\n"
                  << "    if (tid == 0) " << buf_name(r) << "[0] = _v; __syncthreads(); }\n";
                break;
            }
            case OpCode::Broadcast: {
                ValueId r = op.result_id;
                std::uint32_t L = vinfo[r].len;
                s << "  for (int j = tid; j < " << L << "; j += 256) "
                  << buf_name(r) << "[j] = " << ref_f32(op.a, "0") << ";\n"
                  << "  __syncthreads();\n";
                break;
            }
            case OpCode::CumSum: case OpCode::CumProd: {
                ValueId r = op.result_id;
                std::uint32_t L = vinfo[op.a].len;
                int is_prod = (op.code == OpCode::CumProd) ? 1 : 0;
                s << "  pie_ir_block_inclusive_scan(" << buf_name(op.a) << ", "
                  << buf_name(r) << ", " << L << ", " << is_prod << ");\n"
                  << "  __syncthreads();\n";
                break;
            }
            case OpCode::SortDesc: {
                ValueId r = op.result_id;
                std::uint32_t L = vinfo[op.a].len;
                if (L > 1024) { fail("SortDesc length > 1024 not supported in MVP"); break; }
                s << "  pie_ir_block_sort_desc(" << buf_name(op.a) << ", "
                  << buf_name(r) << ", (int*)" << buf_name(r + 1) << ", " << L << ");\n"
                  << "  __syncthreads();\n";
                break;
            }
            case OpCode::PivotThreshold: {
                ValueId r = op.result_id;
                std::uint32_t L = vinfo[op.a].len;
                s << "  { float _t;\n";
                switch (op.predicate.tag) {
                    case PredTag::RankLe:
                        // #25: k is a value-id (host-submit U32 scalar), read [0].
                        s << "    _t = pie_ir_pivot_topk_radix(" << buf_name(op.a) << ", " << L
                          << ", " << ref_int(op.predicate.payload) << ");\n";
                        break;
                    case PredTag::CummassLe:
                        s << "    _t = pie_ir_pivot_topp_radix(" << buf_name(op.a) << ", " << L
                          << ", " << ref_f32(op.predicate.payload, "0") << ");\n";
                        break;
                    case PredTag::ProbGe:
                        s << "    _t = " << ref_f32(op.predicate.payload, "0") << ";\n";
                        break;
                }
                s << "    __syncthreads();\n"
                  << "    for (int j = tid; j < " << L << "; j += 256) " << buf_name(r)
                  << "[j] = (" << buf_name(op.a) << "[j] >= _t) ? 1.0f : 0.0f;\n"
                  << "    __syncthreads(); }\n";
                break;
            }
            case OpCode::Gather: {
                ValueId r = op.result_id;
                std::uint32_t src_len = vinfo[op.a].len;
                std::uint32_t n = vinfo[r].len;
                s << "  pie_ir_gather(" << buf_name(op.a) << ", " << src_len << ", (const int*)"
                  << buf_name(op.b) << ", " << buf_name(r) << ", " << n << ");\n"
                  << "  __syncthreads();\n";
                break;
            }
            case OpCode::GatherRow: {
                ValueId r = op.result_id;
                std::uint32_t nrows = vinfo[op.a].ty.shape.rows();
                std::uint32_t ncols = vinfo[r].len;
                s << "  pie_ir_gather_row(" << buf_name(op.a) << ", " << nrows << ", (int)"
                  << ref_int(op.b) << ", " << ncols << ", " << buf_name(r) << ");\n"
                  << "  __syncthreads();\n";
                break;
            }
            case OpCode::ScatterAdd: case OpCode::ScatterSet: {
                ValueId r = op.result_id;
                std::uint32_t base_len = vinfo[op.a].len;
                std::uint32_t nidx = vinfo[op.b].len;
                const char* fn = (op.code == OpCode::ScatterAdd) ? "pie_ir_scatter_add"
                                                                 : "pie_ir_scatter_set";
                s << "  for (int j = tid; j < " << base_len << "; j += 256) " << buf_name(r)
                  << "[j] = " << ref_f32(op.a, "j") << ";\n"
                  << "  __syncthreads();\n"
                  << "  " << fn << "(" << buf_name(r) << ", " << base_len << ", (const int*)"
                  << buf_name(op.b) << ", " << buf_name(op.c) << ", " << nidx << ");\n"
                  << "  __syncthreads();\n";
                break;
            }
            case OpCode::Rng: {
                ValueId r = op.result_id;
                std::uint32_t L = vinfo[r].len;
                const char* fn = (op.rng_kind == RngKind::Gumbel) ? "pie_ir_gumbel"
                                                                  : "pie_ir_hash_uniform";
                if (ambient_seed) {  // v4 Model B: ambient seed (single block → S[0])
                    s << "  { unsigned long long _se = pie_ir_seed_eff_stream(b" << row_seed_buffer
                      << "[0], " << op.stream << "u);\n";
                } else {
                    s << "  { unsigned long long _se = pie_ir_seed_eff((unsigned int)" << ref_int(op.a)
                      << ");\n";
                }
                s << "    for (int j = tid; j < " << L << "; j += 256) " << buf_name(r)
                  << "[j] = " << fn << "(_se, j);\n"
                  << "    __syncthreads(); }\n";
                break;
            }
            case OpCode::RowBroadcast:
            case OpCode::GatherCols:
                // Both always produce/consume a Matrix → per-row (matrix) kernel,
                // never the single-block vector path. Unreachable here.
                fail("matrix-only op in a single-block (vector) kernel");
                break;
        }
    }

    // integer scalar reference (seed, row index): const or buffer[0].
    std::string ref_int(ValueId id) {
        const VInfo& vi = vinfo[id];
        if (vi.is_const) {
            if (vi.lit.dtype == DType::U32) return std::to_string(vi.lit.as_u32()) + "u";
            return std::to_string(vi.lit.as_i32());
        }
        return "b" + std::to_string(vi.buffer) + "[0]";
    }

    // Kernel parameter pointer type for a buffer. bf16 is read-only `const` when
    // it's an external input (the intrinsic logits), but a WRITABLE `unsigned
    // short*` when codegen produces it — e.g. the raw-bf16 logits-passthrough
    // Output buffer, which a tail kernel writes (a const ptr there won't compile).
    const char* param_ptr_cty(BufferId id) {
        if (buf_phys[id] == Phys::BF16 &&
            (buffers[id].cls == BufferClass::Output ||
             buffers[id].cls == BufferClass::Intermediate))
            return "unsigned short*";
        return phys_ptr_cty(buf_phys[id]);
    }

    // Collect the buffers a kernel touches (operands + results) in ascending id
    // order, and emit its source. `rows` is the kernel's block geometry (1 =
    // vector single-block; >1 = one block per matrix row).
    KernelDesc emit_kernel(std::size_t kidx, const std::vector<std::uint32_t>& op_idx,
                           bool emit_logits_cast, std::uint32_t rows, bool is_matrix) {
        cur_rows = rows;
        const bool mat = !batched && is_matrix;   // matrix per-row path (even at rows==1, k=1)
        const bool per_row = batched || mat;   // one block per row (matrix or batch)
        std::ostringstream body;
        if (batched) {
            // grid = num_rows exactly (LaunchShape::OneBlockPerRow), so r is
            // always in range — no bound check needed.
            body << "  const int r = blockIdx.x;\n  const int tid = threadIdx.x;\n";
        } else if (mat) {
            body << "  const int r = blockIdx.x;\n  if (r >= " << rows << ") return;\n"
                 << "  const int tid = threadIdx.x;\n";
        } else {
            body << "  const int tid = threadIdx.x;\n";
        }

        std::vector<bool> used(buffers.size(), false);
        auto use = [&](BufferId id) { if (id < used.size()) used[id] = true; };
        // Recursively mark the leaf buffers a value reads: an inlined value has
        // no buffer of its own — it folds to its operands' buffers.
        std::function<void(ValueId)> mark_used = [&](ValueId v) {
            if (v >= vinfo.size() || vinfo[v].is_const) return;
            if (v < inlined.size() && inlined[v]) {
                const Op* p = producer(v);
                if (p) {
                    // An inlined ambient Rng reads the external RowSeed buffer.
                    if (ambient_seed && p->code == OpCode::Rng && row_seed_buffer != UINT32_MAX)
                        use(row_seed_buffer);
                    for (ValueId o : op_operands(*p)) mark_used(o);
                }
                return;
            }
            use(vinfo[v].buffer);
        };

        if (emit_logits_cast) {
            use(logits_raw);
            use(logits_f32);
            if (per_row) {
                // Each block casts its own row (logits is [rows|num_rows, len]).
                std::uint32_t L = vinfo[logits_value].len;
                body << "  for (int j = tid; j < " << L << "; j += 256) b" << logits_f32
                     << "[r*" << L << "+j] = pie_ir_bf16_to_f32(b" << logits_raw << "[r*" << L
                     << "+j]);\n  __syncthreads();\n";
            } else {
                std::uint32_t L = buffers[logits_f32].elem_count;
                body << "  for (int j = tid; j < " << L << "; j += 256) b" << logits_f32
                     << "[j] = pie_ir_bf16_to_f32(b" << logits_raw << "[j]);\n"
                     << "  __syncthreads();\n";
            }
        }

        for (std::uint32_t oi : op_idx) {
            const Op& op = slot.ops[oi];
            // Mark operand leaf buffers (recursing through inlined producers).
            for (ValueId v : op_operands(op)) mark_used(v);
            // A materialized ambient Rng reads the external RowSeed buffer.
            if (ambient_seed && op.code == OpCode::Rng && row_seed_buffer != UINT32_MAX)
                use(row_seed_buffer);
            // Mark materialized result buffers (inlined results have none).
            for (std::uint32_t k = 0; k < op.result_count; ++k) {
                ValueId rr = op.result_id + k;
                if (!(rr < inlined.size() && inlined[rr])) use(vinfo[rr].buffer);
            }

            if (per_row) emit_op_matrix(body, op);
            else emit_op(body, op);
            if (!ok) break;
        }

        // Build ordered param list + args from used buffers.
        KernelDesc kd;
        kd.entry_name = "psir_k" + std::to_string(kidx);
        if (mat) {
            kd.shape = LaunchShape::Custom;
            kd.custom_grid = rows;
            kd.custom_block = kSamplingIrBlock;
        } else {
            // Batched: grid = num_rows (dynamic, JIT computes from Param{0}).
            // M=1: grid = 1.
            kd.shape = LaunchShape::OneBlockPerRow;
        }

        std::ostringstream params;
        bool first = true;
        for (BufferId id = 0; id < buffers.size(); ++id) {
            if (!used[id]) continue;
            if (!first) params << ", ";
            first = false;
            params << param_ptr_cty(id) << " b" << id;
            kd.args.push_back(KernelArg::Buf(id));
        }

        std::ostringstream src;
        src << primitive_prelude() << "\n"
            << "extern \"C\" __global__ void " << kd.entry_name << "(" << params.str() << ") {\n"
            << body.str()
            << "}\n";
        kd.source = src.str();
        return kd;
    }

    // Tail kernel that copies each passthrough output's source value into its
    // dedicated BufferClass::Output buffer (see build_buffers). Emitted after the
    // op kernels so the source (e.g. the materialized logits f32-cast) is live.
    KernelDesc emit_passthrough_kernel(std::size_t kidx) {
        std::ostringstream body;
        if (batched) body << "  const int r = blockIdx.x;\n";
        body << "  const int tid = threadIdx.x;\n";

        std::vector<bool> used(buffers.size(), false);
        auto use = [&](BufferId id) { if (id < used.size()) used[id] = true; };
        for (const auto& pt : passthrough_outputs) {
            BufferId src = pt.first;
            BufferId ob = pt.second;
            std::uint32_t L = buffers[ob].elem_count ? buffers[ob].elem_count : 1u;
            std::string widx = batched ? ("r*" + std::to_string(L) + "+j") : "j";
            // Raw, same-dtype element copy (bf16→bf16 for logits) — a verbatim
            // passthrough; the host reinterprets per the output's storage dtype.
            body << "  for (int j = tid; j < " << L << "; j += 256) b" << ob
                 << "[" << widx << "] = b" << src << "[" << widx << "];\n";
            use(ob);
            use(src);
        }
        body << "  __syncthreads();\n";

        KernelDesc kd;
        kd.entry_name = "psir_k" + std::to_string(kidx);
        kd.shape = LaunchShape::OneBlockPerRow;   // grid = num_rows (batched) / 1 (M=1)
        std::ostringstream params;
        bool first = true;
        for (BufferId id = 0; id < buffers.size(); ++id) {
            if (!used[id]) continue;
            if (!first) params << ", ";
            first = false;
            params << param_ptr_cty(id) << " b" << id;
            kd.args.push_back(KernelArg::Buf(id));
        }
        std::ostringstream src;
        src << primitive_prelude() << "\n"
            << "extern \"C\" __global__ void " << kd.entry_name << "(" << params.str() << ") {\n"
            << body.str()
            << "}\n";
        kd.source = src.str();
        return kd;
    }
};

}  // namespace

LowerResult lower(const Program& program, const LowerOptions& opts) {
    LowerResult res;
    if (program.slots.empty()) { res.error = "program has no slots"; return res; }
    if (program.slots.size() != 1) { res.error = "MVP supports a single slot only"; return res; }

    Codegen cg(program, opts.batched);
    cg.infer_types();
    cg.analyze_fusion();
    cg.build_buffers();
    if (!cg.ok) { res.error = cg.err; return res; }

    // Batched (M>1) lowering assumes per-batch-row INDEPENDENT work (one block per
    // batch row, no cross-row geometry). A matrix program (a `[k, vocab]` intrinsic
    // — spec-verify) instead carries a per-row MATRIX axis whose rows feed cross-row
    // ops (per-row argmax → `[k]` vector → cumprod accept-scan → reduce). Folding
    // that into one per-block batched kernel silently miscompiles the cross-row scan
    // (each block sees only its own row). Reject matrix work here so the backend's
    // M=1 fallback (jit_backend.cpp) re-lowers it on the correct per-row matrix path
    // (Custom grid = k, baked from the bytecode; cross-row ops cut to their own
    // single-block kernel). The standard batched samplers (temp/min-p/grammar/
    // mirostat) are `[vocab]` vector programs with no matrix work — unaffected.
    if (opts.batched) {
        bool matrix_work = false;
        for (const VInfo& vi : cg.vinfo)
            if (vi.ty.shape.tag == ShapeTag::Matrix) { matrix_work = true; break; }
        if (!matrix_work)
            for (const Op& op : cg.slot.ops)
                if (cg.op_is_matrix(op)) { matrix_work = true; break; }
        if (matrix_work) {
            res.error = "batched lowering does not support matrix (per-row) programs";
            return res;   // ok stays false → backend re-lowers M=1
        }
    }

    auto groups = cg.partition();

    // The materialized logits cast goes in the first group that references logits
    // — skipped entirely when the cast is inlined into consumer loops.
    bool cast_emitted = false;
    for (std::size_t g = 0; g < groups.size(); ++g) {
        bool emit_cast = false;
        if (!cg.inline_logits_cast && cg.logits_value != UINT32_MAX && !cast_emitted) {
            for (std::uint32_t oi : groups[g]) {
                const Op& op = cg.slot.ops[oi];
                if (op.a == cg.logits_value || op.b == cg.logits_value || op.c == cg.logits_value) {
                    emit_cast = true;
                    break;
                }
            }
        }
        std::uint32_t rows = groups[g].empty() ? 1 : cg.op_block_rows(cg.slot.ops[groups[g][0]]);
        bool is_matrix = !groups[g].empty() && cg.op_is_matrix(cg.slot.ops[groups[g][0]]);
        KernelDesc kd = cg.emit_kernel(g, groups[g], emit_cast, rows, is_matrix);
        if (!cg.ok) { res.error = cg.err; return res; }
        if (emit_cast) cast_emitted = true;
        res.dag.kernels.push_back(std::move(kd));
    }

    // Tail copy kernel for passthrough outputs (input-leaf program outputs that
    // the op-result loop never materialized — e.g. raw-logits in [Token, Logits]).
    if (!cg.passthrough_outputs.empty()) {
        KernelDesc kd = cg.emit_passthrough_kernel(groups.size());
        if (!cg.ok) { res.error = cg.err; return res; }
        res.dag.kernels.push_back(std::move(kd));
    }

    // Stamp the late-input inject barriers (first-consuming-kernel) before
    // handing the buffer table off.
    cg.stamp_late_buffers(groups);

    res.dag.buffers = std::move(cg.buffers);
    res.ok = true;
    return res;
}

LowerResult lower(const Program& program) {
    return lower(program, LowerOptions{});
}

LowerResult lower_bytecode(const std::uint8_t* data, std::size_t len, const LowerOptions& opts) {
    Program p;
    DecodeError de;
    if (!decode(data, len, p, &de)) {
        LowerResult r;
        r.error = "bytecode decode failed: " + de.detail;
        return r;
    }
    return lower(p, opts);
}

LowerResult lower_bytecode(const std::uint8_t* data, std::size_t len) {
    return lower_bytecode(data, len, LowerOptions{});
}

LowerResult lower_bytecode_v4(const std::uint8_t* data, std::size_t len,
                              const ProgramManifest& manifest, const LowerOptions& opts) {
    // Map the codegen-facing manifest (Logits | HostTensor) → in-memory bindings.
    std::vector<Binding> slot_bindings;
    slot_bindings.reserve(manifest.size());
    for (const InputBind& b : manifest) {
        Binding bd;
        if (b.kind == BindKind::Logits) {
            bd.tag = BindingTag::Intrinsic;
            bd.intrinsic = b.intrinsic_kind;   // Logits | MtpLogits (manifest-only)
        } else {
            bd.tag = BindingTag::Host;
            bd.host_key = b.host_key;
            bd.host_avail = b.ready;
        }
        slot_bindings.push_back(bd);
    }
    Program p;
    DecodeError de;
    if (!decode_v4(data, len, slot_bindings, p, &de)) {
        LowerResult r;
        r.error = "v4 bytecode decode failed: " + de.detail;
        return r;
    }
    return lower(p, opts);
}

}  // namespace pie_cuda_driver::sampling_ir
