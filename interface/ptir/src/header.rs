//! Deterministic C header generation from the op table and registries —
//! `include/ptir_abi.h` is checked in and kept in sync by the
//! `ptir_header_uptodate` test (regenerate with `PTIR_REGEN=1 cargo test -p
//! pie-sampling-ir ptir_header`). The C++ driver includes that header, so op
//! ids / stage tags / port ids cannot drift between Rust and CUDA.

use alloc::format;
use alloc::string::String;

use super::op::{IntrinsicId, OP_TABLE, VARIADIC};
use super::registry::{KNOWN_SINKS, PHASE_DESCRIPTOR_TAG, Port, Stage};
use crate::PTIR_VERSION;

/// Render `include/ptir_abi.h`. Pure function of the tables — byte-stable.
pub fn generate_c_header() -> String {
    let mut s = String::new();
    s.push_str("// ptir_abi.h — GENERATED from pie-sampling-ir `src/ptir/{op,registry}.rs`.\n");
    s.push_str(
        "// DO NOT EDIT. Regenerate: PTIR_REGEN=1 cargo test -p pie-sampling-ir ptir_header\n",
    );
    s.push_str("// Container layout: interface/sampling-ir/PTIR-CONTAINER.md\n");
    s.push_str("#pragma once\n#include <stdint.h>\n\n");

    s.push_str(&format!(
        "#define PTIR_MAGIC \"PTIR\"\n#define PTIR_VERSION {PTIR_VERSION}\n"
    ));
    s.push_str(&format!(
        "#define PTIB_MAGIC \"PTIB\" // bound-trace typed sidecar (PTIR-CONTAINER.md section 7)\n#define PTIB_VERSION {}\n",
        super::sidecar::PTIB_VERSION
    ));
    s.push_str(&format!(
        "// v1.1 extern channels (PTIR-CONTAINER.md section 6b): wire-version 2 iff externs\n#define PTIR_VERSION_EXTERN {}\nenum PtirExternDir : uint8_t {{ PTIR_EXTERN_IMPORT = 0, PTIR_EXTERN_EXPORT = 1 }};\n\n",
        crate::PTIR_VERSION_EXTERN
    ));
    s.push_str(&format!(
        "#define PTIR_COMPILER_VERSION {}\n#define PTIR_REGION_PLAN_VERSION {}\n#define PTIR_LANE_TABLE_ABI_VERSION {}\n\n",
        crate::compiler::COMPILER_VERSION,
        crate::compiler::REGION_PLAN_VERSION,
        crate::compiler::LANE_TABLE_ABI_VERSION
    ));
    s.push_str(
        "enum PtirSymbolicExtent : uint8_t {\n\
  PTIR_EXTENT_KV_LEN = 0,\n\
  PTIR_EXTENT_PAGE_COUNT = 1,\n\
  PTIR_EXTENT_ROW_COUNT = 2,\n\
  PTIR_EXTENT_TOKEN_COUNT = 3,\n\
  PTIR_EXTENT_SAMPLED_ROWS = 4,\n\
  PTIR_EXTENT_QUERY_LEN = 5,\n\
  PTIR_EXTENT_KEY_LEN = 6,\n\
};\n\n\
enum PtirScheduleTemplate : uint8_t {\n\
  PTIR_SCHEDULE_EFFECTS = 0,\n\
  PTIR_SCHEDULE_ONE_CTA_PER_ROW = 1,\n\
  PTIR_SCHEDULE_HIERARCHICAL_ROW = 2,\n\
  PTIR_SCHEDULE_LIBRARY = 3,\n\
};\n\n\
enum PtirLibraryOp : uint8_t {\n\
  PTIR_LIBRARY_NUCLEUS_SAMPLE = 0,\n\
  PTIR_LIBRARY_TOP_K = 1,\n\
  PTIR_LIBRARY_SORT = 2,\n\
  PTIR_LIBRARY_SCAN = 3,\n\
  PTIR_LIBRARY_MATMUL = 4,\n\
  PTIR_LIBRARY_SECOND_PARTY = 5,\n\
};\n\n\
typedef struct PtirLaneTableHeader {\n\
  uint32_t abi_version;\n\
  uint32_t lane_count;\n\
  uint32_t channel_slots_per_lane;\n\
  uint32_t flags;\n\
} PtirLaneTableHeader;\n\n\
typedef struct PtirLaneRecord {\n\
  uint64_t logits_base;\n\
  uint32_t logits_row_offset;\n\
  uint32_t logits_row_count;\n\
  uint32_t kv_len;\n\
  uint32_t page_count;\n\
  uint32_t row_count;\n\
  uint32_t token_count;\n\
  uint32_t sampled_rows;\n\
  uint32_t query_len;\n\
  uint32_t key_len;\n\
  uint32_t channel_slot_offset;\n\
  uint64_t rng_state;\n\
  uint64_t commit_slot;\n\
  uint64_t active_row_mask;\n\
} PtirLaneRecord;\n\n\
typedef struct PtirLaneChannelSlot {\n\
  uint64_t committed_cell;\n\
  uint64_t pending_cell;\n\
  uint64_t expected_head;\n\
  uint64_t expected_tail;\n\
} PtirLaneChannelSlot;\n\n",
    );

    s.push_str("// ── op tags (X-macro: name, tag, value-operands, results; 0xFF = variadic) ──\n");
    s.push_str("#define PTIR_OP_LIST(X) \\\n");
    for (i, op) in OP_TABLE.iter().enumerate() {
        let cont = if i + 1 == OP_TABLE.len() { "" } else { " \\" };
        s.push_str(&format!(
            "  X({}, 0x{:02X}, {}, {}){}\n",
            op.name,
            op.tag,
            if op.val_operands == VARIADIC {
                String::from("0xFF")
            } else {
                format!("{}", op.val_operands)
            },
            op.results,
            cont
        ));
    }
    s.push_str("\nenum PtirOpTag : uint8_t {\n");
    for op in OP_TABLE {
        s.push_str(&format!(
            "  PTIR_OP_{} = 0x{:02X},\n",
            op.name.to_uppercase(),
            op.tag
        ));
    }
    s.push_str("};\n\n");

    s.push_str(
        "// ── dtypes (channel decls may also carry PTIR_DT_ACT = late-bound activation) ──\n",
    );
    s.push_str("enum PtirDType : uint8_t {\n  PTIR_DT_F32 = 0,\n  PTIR_DT_I32 = 1,\n  PTIR_DT_U32 = 2,\n  PTIR_DT_BOOL = 3,\n  PTIR_DT_ACT = 4,\n};\n\n");

    s.push_str("// ── stages (per-layer taps: ON_ATTN_PROJ, ON_ATTN) ──\n");
    s.push_str("enum PtirStage : uint8_t {\n");
    for st in [
        Stage::Prologue,
        Stage::OnAttnProj,
        Stage::OnAttn,
        Stage::Epilogue,
    ] {
        s.push_str(&format!(
            "  PTIR_STAGE_{} = {},\n",
            st.name().to_uppercase(),
            st as u8
        ));
    }
    s.push_str("};\n");
    s.push_str(&format!(
        "// readiness-table phase tag for the descriptor (not a program stage)\n#define PTIR_PHASE_DESCRIPTOR 0x{PHASE_DESCRIPTOR_TAG:02X}\n\n"
    ));

    s.push_str("// ── descriptor ports (token family CONSUMES, geometry/masks PEEK) ──\n");
    s.push_str("enum PtirPort : uint8_t {\n");
    for p in [
        Port::EmbedTokens,
        Port::EmbedIndptr,
        Port::Positions,
        Port::Pages,
        Port::PageIndptr,
        Port::KvLen,
        Port::WSlot,
        Port::WOff,
        Port::Readout,
        Port::AttnMask,
    ] {
        s.push_str(&format!(
            "  PTIR_PORT_{} = {},\n",
            p.name().to_uppercase(),
            p as u8
        ));
    }
    s.push_str("};\n\n");

    s.push_str("// ── first-party value intrinsics (op 0xA0 payload) ──\n");
    s.push_str("enum PtirIntrinsic : uint16_t {\n");
    for i in [
        IntrinsicId::Logits,
        IntrinsicId::MtpLogits,
        IntrinsicId::Hidden,
        IntrinsicId::Query,
        IntrinsicId::ValueHead,
        IntrinsicId::Layer,
        IntrinsicId::MtpDrafts,
    ] {
        s.push_str(&format!(
            "  PTIR_INTR_{} = {},\n",
            i.name().to_uppercase(),
            i as u16
        ));
    }
    s.push_str("};\n\n");

    s.push_str("// ── channel host roles / readiness direction / lowering classes ──\n");
    s.push_str("enum PtirHostRole : uint8_t { PTIR_HOST_NONE = 0, PTIR_HOST_WRITER = 1, PTIR_HOST_READER = 2 };\n");
    s.push_str("enum PtirDirection : uint8_t { PTIR_NEEDS_FULL = 0, PTIR_NEEDS_EMPTY = 1 };\n");
    s.push_str("enum PtirChannelClass : uint8_t { PTIR_CHAN_FULL_RING = 0, PTIR_CHAN_IN_PLACE = 1, PTIR_CHAN_IN_PLACE_UNDO = 2 };\n\n");

    s.push_str("// ── well-known first-party sink names (scope: 0 = pass-wide/prologue-only, 1 = attention) ──\n");
    for (name, scope) in KNOWN_SINKS {
        s.push_str(&format!(
            "#define PTIR_SINK_{} \"{}\" // scope {}\n",
            name.to_uppercase(),
            name,
            *scope as u8
        ));
    }
    s.push_str("\n// ── numeric contract (T8 replay determinism; golden interp is normative) ──\n");
    s.push_str("// argmax: lower index wins ties; NaN never selected (all-NaN row -> index 0).\n");
    s.push_str("// sort_desc/top_k: descending, ties -> lower original index first; NaN sorts below -inf.\n");
    s.push_str(
        "// rank_le(k): #strictly-greater < k (ties may admit > k elements at the boundary).\n",
    );
    s.push_str("// cummass_le(p): inclusive nucleus (keep while exclusive prefix mass < p). prob_ge: >=.\n");
    s.push_str("// rng_keyed(state=[key,ctr]): seed64 = splitmix64((key<<32)|ctr); u(j) = hash_uniform(seed64, j)\n");
    s.push_str("//   with splitmix64/hash_uniform exactly as interface/ptir/src/rng.rs; gumbel = -log(-log(u)).\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_deterministic() {
        assert_eq!(generate_c_header(), generate_c_header());
    }
}
