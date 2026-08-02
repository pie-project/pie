//! ② KERNEL SIGNATURES — the vocabulary. The rows live with the kernels
//! (`.wiki/tart/dsl.md` ②).
//!
//! `dsl::cuda` has ten wrappers over five attention kernels because
//! `_region` / `_planned` / `_capture` / `_dequant` encode the DISPATCH
//! CONTEXT in the wrapper name. The context is a property of the call site;
//! what belongs to the kernel is its symbol and its contract. A [`KernelSig`]
//! is that contract, once per symbol.
//!
//! Four declarations, each replacing something that is a hand-written runtime
//! rule today:
//!
//! | declaration | replaces |
//! |---|---|
//! | `whole`   | `if c.head_dim_padded \|\| (window_one && c.xqa_decode)` in the model body |
//! | `lacks`   | "a score-wanting program under XQA fails loudly PTIR-side" (a C++ throw) |
//! | `needs`   | the prepare a stated kernel obligates, named nowhere |
//! | `sink`    | `emit_cuda::emit_masked_pages_bracket`'s hardcoded page substitution |
//!
//! `whole` is CHECKED at trace time — which is load time, since a declaration
//! is traced when the model loads. The other three are declared but not yet
//! consumed: `needs`/`sink` are the emitter's knowledge until the launch ABI
//! flattens (migration step 6), and `lacks` needs the deployment's
//! servable-seam set, which is the support-matrix work. Declaring them first
//! is the point — the table is where they land, and it exists.
//!
//! ## Why this is its own crate
//!
//! The rows are in [`kernels-cuda`](../kernels_cuda/index.html) and
//! [`kernels-metal`](../kernels_metal/index.html), one crate per backend,
//! each beside the `.cu`/`.metal` it describes — so a new kernel is one
//! source file and one table row in the same directory and the same diff
//! hunk. Both tables have to be written in the same words, and neither
//! backend owns those words, so they are here.
//!
//! Bare-named for the same reason [`driver`](../driver/index.html) is: it is
//! the shared floor under a `-`-prefixed pair, holding what both members
//! speak rather than anything either one does. Nothing depends on it but the
//! two tables and the compiler that reads them, and it depends on nothing at
//! all — a row must be writable next to its kernel without dragging a
//! dependency graph along.

/// A capability a seam may ask of the kernel covering its rows. Named after
/// the seam vocabulary (`.wiki/tart/dsl.md` ①), because that is what a
/// `lacks` line refuses to serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cap {
    /// The attention scores, published for an `attn.out` observer.
    Scores,
    /// The page-mask sink an `attn.q` tap writes.
    PageMaskSink,
}

/// The host-side plan a kernel's contract obligates: stated so a reader of
/// the model text can see which prepare a launch drags in, rather than
/// reading the driver to find out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prepare {
    /// No host plan.
    None,
    /// The FlashInfer decode plan (per fire, per layer group).
    DecodePlan,
    /// The FlashInfer ragged prefill plan.
    PrefillPlan,
    /// The custom-mask plan (`attn_page_mask`'s consumer).
    CustomPlan,
    /// XQA's fire-wide prepare — R-shaped, so it cannot be built per row
    /// window. This is why `xqa_decode` is also `whole`.
    FireWide,
    /// MLA's plan (`ops::plan_attention_mla_bf16`), which is its own kind
    /// rather than a FlashInfer plan under another name: it is built from
    /// `kv_lora_rank` and `qk_rope_head_dim` — a latent KV geometry no other
    /// prepare here has a field for — and it is cached in an `MlaPlanCache`
    /// the dispatch borrows, not in the shared attention workspace.
    MlaPlan,
}

/// One kernel's contract.
pub struct KernelSig {
    /// The dsl-side name (what a model text spells).
    pub name: &'static str,
    /// The C++ launcher symbol the trace records.
    pub symbol: &'static str,
    /// The kernel REFUSES a row split: it may not be stated inside a peel's
    /// regions, because its addressing (a fire-wide prepare, a padded staging
    /// buffer) is not row-offsettable. `model-compiler`'s `OpKind::Peel` is
    /// the op this refuses, and its `check_plan` is what enforces the refusal.
    pub whole: bool,
    /// The host plan its contract obligates.
    pub needs: Prepare,
    /// Capabilities this kernel cannot serve — a seam asking for one of these
    /// over rows this kernel covers is unservable.
    pub lacks: &'static [Cap],
    /// Where a sink-writing seam's output lands, if this kernel accepts one
    /// (`sink pages -> kv.pages`).
    pub sink: Option<&'static str>,
    /// The operand index this kernel ACCUMULATES INTO, if it is in-place.
    ///
    /// `launch_residual_add_bf16(y, x, n)` writes its result over `y`, so
    /// its row says `in_place = 0`. That is a fact about the KERNEL and
    /// not about any statement using it — every call of it is in-place —
    /// which is why it lives here rather than at the call site.
    ///
    /// `lower::Buffers` is what reads it: an in-place op's output takes
    /// its operand's OFFSET instead of an allocation of its own. Without
    /// that, a text accumulating into a WINDOW (gemma3n's per-layer
    /// embedding, added back into K-1 corrected AltUp streams) would
    /// produce fresh values nothing downstream reads, and the streams
    /// would silently stay pre-update.
    pub in_place: Option<u32>,
    /// On a union tail layer this dispatch pairs the DEPTH PREFIX plan (and
    /// its dedicated workspace) instead of the fire's own plan.
    ///
    /// This was the `PrefixPlanSwap` half of the retired per-op `DepthRole` —
    /// a word the IR carried on one launch per layer of every depth-declaring
    /// trace, restating a fact about the KERNEL. Migration step 5 moved it
    /// here.
    pub depth_prefix_plan: bool,
}

/// Declare one kernel. The syntax is `.wiki/tart/dsl.md` ②'s, minus the
/// operand shapes: those stay with the emitter until the launch ABI flattens,
/// and stating them twice would be the duplication this redesign exists to
/// remove.
///
/// Exported so the two backend tables can declare rows in the same words. It
/// names [`KernelSig`], [`Prepare`] and [`Cap`] through `$crate`, so a table
/// crate needs no `use` beyond the macro itself.
#[macro_export]
macro_rules! kernel {
    ($name:ident $symbol:literal $(, $key:ident = $value:expr)* $(,)?) => {
        $crate::KernelSig {
            name: stringify!($name),
            symbol: $symbol,
            $($key: $value,)*
            ..$crate::KernelSig {
                name: "",
                symbol: "",
                whole: false,
                needs: $crate::Prepare::None,
                lacks: &[],
                sink: None,
                in_place: None,
                depth_prefix_plan: false,
            }
        }
    };
}

/// The contract for one symbol, in `table`.
///
/// A linear scan: the tables are ~100 and ~20 rows, and the call sites are
/// load-time (a declaration is traced when the model loads), not per-fire.
pub fn sig_in(table: &'static [KernelSig], symbol: &str) -> Option<&'static KernelSig> {
    table.iter().find(|k| k.symbol == symbol)
}
