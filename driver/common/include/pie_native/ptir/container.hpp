#pragma once

// PTIR trace-container reader (`PTIR` v1) — the shared pure-host driver-side
// decoder for echo's wire format (`interface/sampling-ir/PTIR-CONTAINER.md`),
// consumed by both the CUDA and Metal drivers.
// Decodes the stage-tagged program blob delta's SDK emits into a structural
// in-memory Container, and derives the per-channel readiness table (C2's
// producer) the stage-runner consumes.
//
// Per the manager's Option-B ruling, per-value (shape, dtype) types are NOT
// re-derived here (echo's `bind()`/`infer.rs` is the single inference oracle and
// its typed BoundTrace reaches the driver at registration). This reader carries
// the shapes the container *does* encode (channel decls, broadcast/reshape/rng/
// intrinsic_val target shapes, const dtype) and the op structure; the typed
// side-table binds on top. Structural conformance is gated against echo's
// golden containers (tests/golden-ptir): container_hash + readiness table match
// byte-for-byte.
//
// Header-only, host C++ (no CUDA) — pulls op_table.hpp for tags/enums only.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "pie_native/ptir/op_table.hpp"
#include "pie_native/ptir/ptir_abi.h"

namespace pie_native::ptir::container {

// ── FNV-1a 64-bit — the container identity hash (C3), same as program_hash ──
inline std::uint64_t fnv1a64(const std::uint8_t* data, std::size_t len) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < len; ++i) { h ^= data[i]; h *= 0x100000001b3ULL; }
    return h;
}

struct CShape {
    std::uint8_t rank = 0;
    std::uint32_t dims[4] = {0, 0, 0, 0};
};

struct CChannel {
    std::uint8_t  dtype = 0;
    CShape        shape;
    std::uint32_t capacity = 1;
    std::uint8_t  host_role = 0;    // PtirHostRole
    std::uint8_t  seeded = 0;
    std::int8_t   extern_dir = -1;  // -1 private, 0 import, 1 export
    std::string   extern_name;
};

struct CPort {
    std::uint8_t  port = 0;         // PtirPort
    bool          is_const = false;
    std::uint32_t chan = 0;         // src == channel
    // const payload (src == const) is skipped structurally (types via Option-B).
};

// A decoded op: its tag + the channel it touches (chan ops) + a compact operand
// record. Enough for cursor advance, readiness derivation, and typed lowering.
struct COp {
    std::uint8_t  tag = 0;
    std::int64_t  chan = -1;        // chan_take/read/put target, else -1
    std::uint16_t name_idx = 0;     // kernel_call / sink_call name table index
    std::vector<std::uint32_t> args;
    std::uint32_t results = 1;      // SSA ids defined
    // op-specific immediates (filled per tag; container_to_trace consumes them):
    std::uint8_t  lit_dtype = 0;    // const literal dtype
    std::uint32_t lit_bits = 0;     // const literal raw bits / cast dtype target
    std::uint16_t intr = 0;         // intrinsic_val id
    std::uint8_t  dtype = 0;        // intrinsic_val / cast / rng element dtype
    CShape        shape;            // broadcast/reshape/rng/intrinsic target shape
    std::uint32_t imm = 0;          // top_k k / iota len / rng stream
    std::uint8_t  kind = 0;         // rng kind (0 uniform, 1 gumbel)
    std::uint8_t  pred_tag = 0;     // pivot_threshold predicate tag
    std::uint32_t pred_payload = 0; // pivot_threshold predicate payload (value id / imm)
};

struct CStage {
    std::uint8_t stage = 0;         // PtirStage
    std::vector<COp> ops;
};

struct Container {
    std::vector<std::string> names;
    std::vector<CChannel>    channels;
    std::vector<CPort>       ports;
    std::vector<CStage>      stages;
    std::uint64_t            hash = 0;   // fnv1a64 over the raw bytes
};

// Readiness direction (mirrors PtirDirection).
enum class Direction : std::uint8_t { NeedsFull = 0, NeedsEmpty = 1 };

struct ReadinessEntry {
    std::uint32_t chan = 0;
    std::uint8_t  phase = 0;        // PtirStage or PTIR_PHASE_DESCRIPTOR (0xFF)
    Direction     dir = Direction::NeedsFull;
};

struct DecodeError {
    bool        ok = true;
    std::string detail;
};

// ── cursor ──
namespace detail {
struct Cur {
    const std::uint8_t* p;
    std::size_t n;
    std::size_t i = 0;
    bool err = false;
    bool need(std::size_t k) {
        if (i > n || k > n - i) {
            err = true;
            return false;
        }
        return true;
    }
    std::uint8_t u8() { if (!need(1)) return 0; return p[i++]; }
    std::uint16_t u16() { if (!need(2)) return 0; std::uint16_t v; std::memcpy(&v, p + i, 2); i += 2; return v; }
    std::uint32_t u32() { if (!need(4)) return 0; std::uint32_t v; std::memcpy(&v, p + i, 4); i += 4; return v; }
    void skip(std::size_t k) { if (need(k)) i += k; }
    CShape shape() { CShape s; s.rank = u8(); if (s.rank > 4) { err = true; return s; } for (int d = 0; d < s.rank; ++d) s.dims[d] = u32(); return s; }
};

// Advance past one op, filling `op`. Operand layout per PTIR-CONTAINER.md §3.
inline void decode_op(Cur& c, COp& op) {
    op.tag = c.u8();
    switch (op.tag) {
        // unary a:u32
        case PTIR_OP_EXP: case PTIR_OP_LOG: case PTIR_OP_NEG: case PTIR_OP_RECIP:
        case PTIR_OP_ABS: case PTIR_OP_SIGN: case PTIR_OP_NOT: case PTIR_OP_TRANSPOSE:
        case PTIR_OP_REDUCE_SUM: case PTIR_OP_REDUCE_MAX: case PTIR_OP_REDUCE_MIN:
        case PTIR_OP_REDUCE_ARGMAX: case PTIR_OP_CUMSUM: case PTIR_OP_CUMPROD:
        case PTIR_OP_SORT_DESC:
            op.args = {c.u32()};
            op.results = (op.tag == PTIR_OP_SORT_DESC) ? 2 : 1;
            break;
        case PTIR_OP_CAST:  op.args = {c.u32()}; op.dtype = c.u8(); break;      // value, dtype
        // binary a,b
        case PTIR_OP_ADD: case PTIR_OP_SUB: case PTIR_OP_MUL: case PTIR_OP_DIV:
        case PTIR_OP_MAX_ELEM: case PTIR_OP_MIN_ELEM: case PTIR_OP_REM:
        case PTIR_OP_GT: case PTIR_OP_GE: case PTIR_OP_EQ: case PTIR_OP_NE:
        case PTIR_OP_LT: case PTIR_OP_LE: case PTIR_OP_AND: case PTIR_OP_OR:
        case PTIR_OP_MATMUL: case PTIR_OP_GATHER: case PTIR_OP_GATHER_ROW:
        case PTIR_OP_MASK_APPLY_PACKED:
            op.args = {c.u32(), c.u32()};
            break;
        case PTIR_OP_SELECT: case PTIR_OP_SCATTER_ADD: case PTIR_OP_SCATTER_SET:
            op.args = {c.u32(), c.u32(), c.u32()};
            break;
        case PTIR_OP_BROADCAST: case PTIR_OP_RESHAPE:                          // value, shape
            op.args = {c.u32()}; op.shape = c.shape(); break;
        case PTIR_OP_TOP_K: op.args = {c.u32()}; op.imm = c.u32(); op.results = 2; break;  // input, k
        case PTIR_OP_PIVOT_THRESHOLD:                                          // input, predicate(5B)
            op.args = {c.u32()}; op.pred_tag = c.u8(); op.pred_payload = c.u32(); break;
        case PTIR_OP_IOTA: op.imm = c.u32(); break;                            // len (immediate)
        case PTIR_OP_RNG:  op.imm = c.u32(); op.shape = c.shape(); op.kind = c.u8(); break;   // stream, shape, kind
        case PTIR_OP_RNG_KEYED: op.args = {c.u32()}; op.shape = c.shape(); op.kind = c.u8(); break; // state, shape, kind
        case PTIR_OP_CONST: op.lit_dtype = c.u8(); op.lit_bits = c.u32(); break;  // literal(5B)
        case PTIR_OP_CHAN_TAKE: case PTIR_OP_CHAN_READ:
            op.chan = c.u32(); break;
        case PTIR_OP_CHAN_PUT:
            op.chan = c.u32(); op.args = {c.u32()}; op.results = 0; break;
        case PTIR_OP_INTRINSIC_VAL: op.intr = c.u16(); op.dtype = c.u8(); op.shape = c.shape(); break;  // intr, dtype, shape
        case PTIR_OP_KERNEL_CALL: {                                            // name, dtype, shape, n_args, args
            op.name_idx = c.u16(); op.dtype = c.u8(); op.shape = c.shape();
            std::uint8_t n = c.u8();
            for (std::uint8_t k = 0; k < n; ++k) op.args.push_back(c.u32());
            break;
        }
        case PTIR_OP_SINK_CALL: {                                              // name, n_args, args
            op.name_idx = c.u16();
            std::uint8_t n = c.u8();
            for (std::uint8_t k = 0; k < n; ++k) op.args.push_back(c.u32());
            op.results = 0;
            break;
        }
        default:
            c.err = true; break;   // unknown op tag
    }
}
}  // namespace detail

// Decode a `PTIR` v1 container. On success returns true and fills `out` (incl.
// its `hash`); on failure returns false and (if non-null) fills `err`.
inline bool decode(const std::uint8_t* data, std::size_t len, Container& out, DecodeError* err = nullptr) {
    auto fail = [&](const char* m) { if (err) { err->ok = false; err->detail = m; } return false; };
    out = {};
    detail::Cur c{data, len};
    if (len < 24) return fail("short header");
    if (std::memcmp(data, PTIR_MAGIC, 4) != 0) return fail("bad magic");
    c.skip(4);
    std::uint16_t version = c.u16();
    if (version != PTIR_VERSION && version != 2) return fail("bad version");
    c.u16();  // flags
    std::uint32_t n_names = c.u32(), n_channels = c.u32(), n_ports = c.u32(), n_stages = c.u32();
    const std::uint32_t n_externs = version == 2 ? c.u32() : 0;

    for (std::uint32_t i = 0; i < n_names; ++i) {
        std::uint16_t l = c.u16();
        if (!c.need(l)) return fail("name overrun");
        out.names.emplace_back(reinterpret_cast<const char*>(data + c.i), l);
        c.i += l;
    }
    for (std::uint32_t i = 0; i < n_channels; ++i) {
        CChannel ch;
        ch.dtype = c.u8();
        ch.shape = c.shape();
        ch.capacity = c.u32();
        ch.host_role = c.u8();
        ch.seeded = c.u8();
        out.channels.push_back(ch);
        if (c.err) return fail("channel overrun");
    }
    for (std::uint32_t i = 0; i < n_ports; ++i) {
        CPort p;
        p.port = c.u8();
        std::uint8_t src = c.u8();
        if (src == 0) { p.is_const = false; p.chan = c.u32(); }
        else {  // const: dtype, shape, data[numel*elem]
            p.is_const = true;
            std::uint8_t dt = c.u8();
            CShape s = c.shape();
            std::uint64_t numel = s.rank == 0 ? 1 : 1;
            for (int d = 0; d < s.rank; ++d) numel *= s.dims[d];
            std::size_t elem = (dt == PTIR_DT_BOOL) ? 1 : 4;
            c.skip(numel * elem);
        }
        out.ports.push_back(p);
        if (c.err) return fail("port overrun");
    }
    for (std::uint32_t i = 0; i < n_stages; ++i) {
        CStage st;
        st.stage = c.u8();
        std::uint32_t n_ops = c.u32();
        for (std::uint32_t k = 0; k < n_ops; ++k) {
            COp op;
            detail::decode_op(c, op);
            if (c.err) return fail("op overrun / unknown tag");
            st.ops.push_back(std::move(op));
        }
        out.stages.push_back(std::move(st));
    }
    std::vector<bool> has_extern(out.channels.size(), false);
    for (std::uint32_t i = 0; i < n_externs; ++i) {
        const std::uint16_t name = c.u16();
        const std::uint8_t direction = c.u8();
        const std::uint32_t channel = c.u32();
        if (c.err || name >= out.names.size() || direction > 1 ||
            channel >= out.channels.size() || has_extern[channel]) {
            return fail("invalid extern declaration");
        }
        has_extern[channel] = true;
        out.channels[channel].extern_dir = static_cast<std::int8_t>(direction);
        out.channels[channel].extern_name = out.names[name];
    }
    if (c.i != len) return fail("trailing bytes");
    out.hash = fnv1a64(data, len);
    return true;
}

// Derive the per-channel readiness table (C2): the FIRST op touching each
// channel across the phase order prologue → descriptor → on_attn_proj → on_attn
// → epilogue names its required bit (take/read or a port consumption ⇒
// NeedsFull; a leading put ⇒ NeedsEmpty). One entry per channel touched.
inline std::vector<ReadinessEntry> derive_readiness(const Container& c) {
    std::vector<ReadinessEntry> out;
    std::vector<std::uint8_t> seen(c.channels.size(), 0);
    auto touch = [&](std::uint32_t chan, std::uint8_t phase, Direction dir) {
        if (chan >= c.channels.size() || seen[chan]) return;
        seen[chan] = 1;
        out.push_back({chan, phase, dir});
    };
    auto scan_stage = [&](std::uint8_t stage_tag) {
        for (const CStage& st : c.stages) {
            if (st.stage != stage_tag) continue;
            for (const COp& op : st.ops) {
                if (op.chan < 0) continue;
                Direction d = (op.tag == PTIR_OP_CHAN_PUT) ? Direction::NeedsEmpty : Direction::NeedsFull;
                touch((std::uint32_t)op.chan, stage_tag, d);
            }
        }
    };
    // phase order: prologue → descriptor → on_attn_proj → on_attn → epilogue
    scan_stage(PTIR_STAGE_PROLOGUE);
    for (const CPort& p : c.ports)                       // descriptor: port consumes/peeks ⇒ full
        if (!p.is_const) touch(p.chan, PTIR_PHASE_DESCRIPTOR, Direction::NeedsFull);
    scan_stage(PTIR_STAGE_ON_ATTN_PROJ);
    scan_stage(PTIR_STAGE_ON_ATTN);
    scan_stage(PTIR_STAGE_EPILOGUE);
    return out;
}

}  // namespace pie_native::ptir::container
