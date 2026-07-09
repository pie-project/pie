//! **Linear-model fold-commit surface exercise** (echo — In Gim's linear-model
//! WIT revisit), written on the **raw WIT surface** (no `Context`/`Forward`
//! facade). A canonical ① low-level rewrite per the SDK-minimization directive
//! (`ptir-sdk-minimization-audit`): it proves an inferlet author can drive a
//! MODEL-AGNOSTIC speculative-commit loop against the kept keep-core primitives
//! + the raw WIT ops directly:
//!
//!   - `model::is_linear()` — the class gate (linear/recurrent vs attention);
//!   - `model::rs_state_size()` / `rs_fold_granularity()` / `rs_buffer_page_size()`
//!     — the RS shaping caps used to size the buffer + validate a fold;
//!   - `working_set::RsWorkingSet` — the folded state + un-folded buffered suffix;
//!   - `geometry::*` (keep-core) — the KV read/write page split for the prefill;
//!   - `ForwardPass::rs_working_set(set, start, len)` — write the buffered RS
//!     window UN-FOLDED (uncertain);
//!   - `ForwardPass::rs_fold_buffered(n_acc)` — the linear-model COMMIT: fold
//!     ONLY the accepted prefix into the recurrent state, irreversibly; the
//!     un-folded tail stays uncertain.
//!
//! The commit is expressed model-agnostically: `if model::is_linear() {
//! pass.rs_fold_buffered(n_acc) }` — on attention models the runtime commits via
//! KV-slot reuse instead, so the same loop is correct on both classes. The
//! runtime lowers `rs_fold_buffered(n)` to `rs_fold_lens` + `RS_FLAG_FOLD` over
//! the buffered slabs (api/inference.rs).
//!
//! Returns the model's RS caps so a harness can read them back. On a pure-
//! attention mock (`is_linear() == false`, `rs_state_size() == 0`) the fold
//! branch is skipped and the pass is an ordinary prefill.

use inferlet::geometry;
use inferlet::inference::ForwardPass;
use inferlet::working_set::{KvWorkingSet, RsWorkingSet};
use inferlet::{model, Result};

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    // How many draft tokens the (pretend) verify accepted — the kept prefix to
    // commit. Bare-integer input → n_acc; anything else → 1.
    let n_acc: u32 = input.trim().parse().unwrap_or(1);

    // ── The class gate + RS shaping caps ────────────────────────────────
    let linear = model::is_linear();
    let state_size = model::rs_state_size();
    let gran = model::rs_fold_granularity();
    let buf_page = model::rs_buffer_page_size();

    // Raw KV working set (binds the single served model implicitly).
    let kv = KvWorkingSet::new();
    let page = kv.page_size();

    let prompt = model::encode("hello world");
    let prompt = if prompt.is_empty() { vec![0u32] } else { prompt };
    let n = prompt.len() as u32;

    // A recurrent-state working set (folded state + un-folded buffered suffix).
    // Constructed unconditionally; only attached on linear models.
    let rs = RsWorkingSet::new();

    // One forward, prefilling `[0, n)` from an empty KV. The keep-core geometry
    // helper computes the read/write page split (here: all-write, no read
    // context) and grows the slot array; the RS window + fold ride the raw pass.
    let geom = geometry::ensure_pages(&kv, geometry::kv_write_geometry(0, n, page))?;

    let pass = ForwardPass::new();
    geometry::attach_kv_write(&pass, &kv, &geom);

    let positions: Vec<u32> = (0..n).collect();
    pass.input_tokens(&prompt, &positions);

    // On a linear model: write the `[0, n)` window into the RS buffer UN-FOLDED,
    // then commit only the accepted prefix (`n_acc`) into the folded recurrent
    // state — the un-folded tail (`[n_acc, n)`) stays uncertain.
    if linear {
        pass.rs_working_set(&rs, 0, n);
        // Fold-commit is irreversible — never fold a rejected draft. Clamp the
        // kept prefix to the buffered window for safety.
        let commit = n_acc.min(n).max(1);
        pass.rs_fold_buffered(commit);
    }

    // No sampler: a prefill/flush fires and finalizes host-side (no output to
    // await), matching the facade's no-sampler execute path.
    pass.execute();

    Ok(format!(
        "linearfold: is_linear={linear} rs_state_size={state_size} \
         rs_fold_granularity={gran} rs_buffer_page_size={buf_page} n_acc={n_acc}"
    ))
}
