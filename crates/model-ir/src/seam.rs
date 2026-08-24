#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cap {
    Transform,

    Observe,

    Scores,

    PageMaskSink,

    Put,

    Sample,

    Emit,
}

pub struct Def {
    pub name: &'static str,

    pub sees: &'static [&'static str],
    pub caps: &'static [Cap],

    pub position: Option<Position>,

    pub sink: Option<&'static str>,
}

#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub after: &'static [&'static str],
    pub before: &'static [&'static str],
}

pub const ATTN_Q: Def = Def {
    name: "attn.q",
    sees: &["q"],
    caps: &[Cap::Observe, Cap::PageMaskSink],
    position: None,

    sink: Some("attention.pages"),
};

pub const ATTN_OUT: Def = Def {
    name: "attn.out",
    sees: &["a"],
    caps: &[Cap::Observe, Cap::Scores],
    position: None,
    sink: None,
};

pub const ATTN_QV: Def = Def {
    name: "attn.qv",
    sees: &["q", "v"],
    caps: &[Cap::Transform],

    position: Some(Position {
        after: &["gemm.matmul", "gemm.matmul_acc", "matmul", "layout.split_qkv"],
        before: &[
            "norm.add_bias",
            "norm.rmsnorm",
            "norm.rmsnorm_no_scale",
            "rmsnorm",
            "rope",
            "rope.full",
            // THE SAME ADMITTED SET UNDER TWO SPELLINGS. `admits` matches a
            // routine's claim either whole or by its first dotted segment,
            // so the bare entry used to cover the core append and the
            // `kv_append.mla` / `.index` / `.pool` sub-families at once. The
            // core's claim is `attention.kv_append` now — under `attention`,
            // where the bare entry no longer reaches it — so it is spelled
            // whole, and the bare one stays for the three sub-families that
            // still read their role there. `attention` itself is NOT the
            // entry: it would admit the attention statement the seam is
            // required to sit before.
            "attention.kv_append",
            "kv_append",
        ],
    }),
    sink: None,
};

pub const RECURRENT: Def = Def {
    name: "recurrent",
    sees: &["mixed"],
    caps: &[Cap::Observe],
    position: None,
    sink: None,
};

pub const IN: Def = Def {
    name: "in",
    sees: &[],
    caps: &[Cap::Put, Cap::Emit],
    position: None,
    sink: None,
};

pub const OUT: Def = Def {
    name: "out",
    sees: &["logits"],
    caps: &[Cap::Observe, Cap::Sample, Cap::Put, Cap::Emit],
    position: None,
    sink: None,
};

// `check_plan` AND ITS THREE HELPERS STOOD HERE — the seam-ordering walk
// over `crate::trace::ForwardPlan`: does each seam sit after the op whose
// value it observes and before the next op that consumes it. It read a
// statement's role through `kernels::claim_of`, which read
// `KernelSig::canon`.
//
// It went with its subject. `ForwardPlan` is the LEGACY traced form and
// `TraceBuilder::finish` was the only thing that ever called this; R3 deleted
// `model-dsl-legacy`'s `Trace`, which was the only thing that ever built a
// `TraceBuilder`. A check with no plan to check is not a check.
//
// THE DEFS BELOW ARE NOT LEGACY and stay: `model_dsl::forward` records `IN`
// and `OUT` by name on every traced plan, and `driver-cuda`/`baker-smoke`
// both find the logits by asking for `OUT.name`.

pub const ALL: &[&Def] = &[&IN, &ATTN_QV, &ATTN_Q, &ATTN_OUT, &RECURRENT, &OUT];

pub fn by_name(name: &str) -> Option<&'static Def> {
    ALL.iter().copied().find(|d| d.name == name)
}


