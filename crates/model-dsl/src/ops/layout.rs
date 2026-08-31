//! The `Layout` family: the embedding gather and the shape-only splits and
//! selects that move rows around without touching their values.

use super::*;

/// **A GATHER LANDS ON ITS IDS' AXIS.** `y` carries `ids`' row space, not
/// `Dim::Tokens` — which is what lets one op serve the token embedding and the
/// tower's learned position embedding (multimodal §9.2) without either
/// knowing about the other.
///
/// **THIS WAS A LITERAL `Dim::Tokens` AND THAT WAS A BUG WAITING FOR A SECOND
/// AXIS.** Every text that existed when it was written passed
/// [`Input::tokens`](crate::Input::tokens), whose row space IS `Dim::Tokens`,
/// so the two readings agreed everywhere and nothing could tell them apart. A
/// patch-axis gather under the old spelling would have minted a TOKEN
/// rectangle: cut at the token window, sized by `max_tokens`, and assigned to
/// the trunk's capture unit — wrong in three ways and loud in none of them.
/// `a_gather_lands_on_its_ids_axis` is the test that keeps them apart now.
/// **THE GATHER THAT CONCATENATES** ([`Layout::EmbedConcat`]): `y[r] =
/// table[ids[r, 0]] ‖ … ‖ table[ids[r, heads−1]]` — the PLE n-gram read,
/// every hashed head's row side by side. `ids` is `[rows, heads]` `i32`
/// ([`attn::ple_ngram_ids`](super::attn::ple_ngram_ids)'s answer), and
/// `heads` is read off its width, for [`embed_weighted`]'s reason.
pub fn embed_concat(ids: &Value, table: &Weight, vocab: u32) -> Value {
    let r = ids.rec();
    let y = r.fresh(tensor(
        ids.rows(),
        ids.width() * table.dim(1),
        table.compute_dtype(),
    ));
    r.push(
        Layout::EmbedConcat {
            ids: ids.id(),
            table: r.weight(table),
            vocab,
            y: y.id(),
        },
        &[ids],
    );
    y
}

pub fn embed(ids: &Value, table: &Weight, vocab: u32) -> Value {
    let r = ids.rec();
    let y = r.fresh(tensor(ids.rows(), table.dim(1), table.compute_dtype()));
    r.push(
        Layout::Embed {
            ids: ids.id(),
            table: r.weight(table),
            vocab,
            y: y.id(),
        },
        &[ids],
    );
    y
}

/// **THE GATHER THAT INTERPOLATES** (multimodal §9.2): `y[r] = Σₜ
/// weights[r, t] · table[ids[r, t]]`.
///
/// The tower's learned position grid is stored at `num_grid_per_side²` and
/// resampled to each image's grid, so a patch row reads `taps` rows of it
/// under `taps` weights. `ids` is `[Dim::Patches, taps]` `i32`
/// ([`Input::patch_embed_rows`](crate::Input::patch_embed_rows)), `weights` is
/// `[Dim::Patches, taps]` `f32`
/// ([`Input::patch_embed_weights`](crate::Input::patch_embed_weights)), and
/// `taps` is read off their width — 4 for bilinear, 16 for bicubic.
///
/// **THE NATIVE GRID WRITES [`embed`] INSTEAD**, with a `[Dim::Patches]` id
/// vector and no weight stream at all: the resample is the identity there, and
/// the cheap path is this op's ABSENCE rather than a degenerate use of it.
pub fn embed_weighted(ids: &Value, weights: &Value, table: &Weight, vocab: u32) -> Value {
    let r = ids.rec();
    let y = r.fresh(tensor(ids.rows(), table.dim(1), table.compute_dtype()));
    r.push(
        Layout::EmbedWeighted {
            ids: ids.id(),
            weights: weights.id(),
            table: r.weight(table),
            vocab,
            y: y.id(),
        },
        &[ids, weights],
    );
    y
}

pub fn split_qkv(packed: &Value, q_width: u32, kv_width: u32) -> (Value, Value, Value) {
    let r = packed.rec();
    let q = r.fresh(tensor(packed.rows(), q_width, packed.dtype()));
    let k = r.fresh(tensor(packed.rows(), kv_width, packed.dtype()));
    let v = r.fresh(tensor(packed.rows(), kv_width, packed.dtype()));
    r.push(
        Layout::SplitQkv {
            packed: packed.id(),
            q_width,
            kv_width,
            q: q.id(),
            k: k.id(),
            v: v.id(),
        },
        &[packed],
    );
    (q, k, v)
}

pub fn split_q_gate(packed: &Value, head_dim: u32) -> (Value, Value) {
    let r = packed.rec();
    let head_dim64 = u64::from(head_dim);
    let half = packed.width() / (2 * head_dim64) * head_dim64;
    let q = r.fresh(tensor(packed.rows(), half, packed.dtype()));
    let gate = r.fresh(tensor(packed.rows(), half, packed.dtype()));
    r.push(
        Layout::SplitQGate {
            packed: packed.id(),
            head_dim,
            q: q.id(),
            gate: gate.id(),
        },
        &[packed],
    );
    (q, gate)
}

pub fn split_rows(x: &Value, width: u32) -> (Value, Value) {
    let r = x.rec();
    let left = r.fresh(tensor(x.rows(), width, x.dtype()));
    let right = r.fresh(tensor(x.rows(), x.width() - u64::from(width), x.dtype()));
    r.push(
        Layout::SplitRows {
            x: x.id(),
            width,
            left: left.id(),
            right: right.id(),
        },
        &[x],
    );
    (left, right)
}

pub fn select(table: &Value, layer: u32, width: u32) -> Value {
    let r = table.rec();
    let y = r.fresh(tensor(table.rows(), width, table.dtype()));
    r.push(
        Layout::Select {
            table: table.id(),
            layer,
            width,
            y: y.id(),
        },
        &[table],
    );
    y
}

/// **THE SPATIAL POOL** (multimodal §6.5, §7.4): the mean of every `side ·
/// side` consecutive rows of `x`, one output row each.
///
/// gemma4's tower folds each `k × k` square of the patch grid into one soft
/// token; `side` is `pooling_kernel_size` — the checkpoint's own number — and
/// `side == 1` is the identity, which the upstream pooler also treats as one
/// (it skips itself when the patch count already equals the soft-token count).
///
/// The result carries `x`'s TYPE and therefore `x`'s row space: the same
/// `Dim::Patches` rectangle, cut at the same patch window, with its leading
/// `rows / side²` rows written. What makes a 2-D pool this 1-D fold is the
/// submission's patch ORDER — each `side × side` square contiguous, which is
/// §2's merge-block-major statute at `side` instead of 2 — and that is the one
/// thing this op asks of anything outside the plan.
pub fn pool_rows(x: &Value, side: u32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Layout::PoolRows {
            x: x.id(),
            side,
            y: y.id(),
        },
        &[x],
    );
    y
}

/// **THE MERGING FOLD** (multimodal §8.1, §8.3): every `side · side`
/// consecutive rows of `x` laid END TO END, one output row of `side²` times
/// the width.
///
/// qwen's spatial merger — `Qwen3_5VisionPatchMerger.forward`'s opening
/// `x.view(-1, hidden_size · spatial_merge_size²)`, which is why
/// `merger.linear_fc1.weight` is `[4·hidden, 4·hidden]`. `side` is
/// `spatial_merge_size`.
///
/// **THE ONE FOLD THAT CANNOT REUSE `x`'s TYPE**: the result is
/// `[Dim::Patches, side² · width]`. Same row space and therefore the same
/// window and capture unit, with its leading `rows / side²` rows written —
/// and the rows past them want a `-1` in `patch_routes` and
/// [`scatter_live_rows`], which is §8.6's whole subject.
///
/// **THE NORM GOES BEFORE THIS OR AFTER IT, AND THE CHECKPOINT SAYS WHICH.**
/// `Qwen3_5VisionPatchMerger` norms the UNMERGED rows by default and the
/// merged ones under `use_postshuffle_norm` — so a text writes
/// `layernorm_no_scale` at `hidden` before this call, or at `side² · hidden`
/// after it, and never guesses.
pub fn merge_rows(x: &Value, side: u32) -> Value {
    let r = x.rec();
    let y = r.fresh(tensor(
        x.rows(),
        u64::from(side) * u64::from(side) * x.width(),
        x.dtype(),
    ));
    r.push(
        Layout::MergeRows {
            x: x.id(),
            side,
            y: y.id(),
        },
        &[x],
    );
    y
}

/// **THE EMBED MERGE, WITH A DROP SENTINEL** (multimodal §8.6):
/// [`scatter_rows`] exactly, plus a NEGATIVE `routes` entry meaning "this row
/// has no destination".
///
/// What a compacting fold ([`pool_rows`], [`merge_rows`]) owes the scatter.
/// Those ops write `rows / side²` rows and leave the rest of the patch
/// rectangle as the arena left it; `routes` is `[Dim::Patches]`, one entry per
/// row of the FULL rectangle, so the tail rows need a value that means
/// nowhere. `-1` is the one
/// [`Input::adapter_routes`](crate::Input::adapter_routes) already uses for
/// "no bank".
///
/// **A TEXT PICKS ONE OF THE TWO SCATTERS AND THE SHELL READS THE PICK.** The
/// plain scatter still refuses a negative route host-side; the sentinel is
/// admitted only for a plan that declares THIS op. So a tower that folds
/// declares this one, a tower that does not keeps the other, and neither can
/// get the other's leniency by accident.
pub fn scatter_live_rows(src: &Value, routes: &Value, y: &Value) -> Value {
    let r = y.rec();
    let y_out = r.fresh(y.ty().clone());
    r.push(
        Layout::ScatterLiveRows {
            src: src.id(),
            routes: routes.id(),
            y: y.id(),
            y_out: y_out.id(),
        },
        &[src, routes, y],
    );
    y_out
}

/// **THE EMBED MERGE**: the tower's rows written into the token rows the
/// image placeholders occupy (multimodal §2's third op).
///
/// `src` is a patch rectangle, `y` a token one, and `routes`
/// ([`Input::patch_routes`](crate::Input::patch_routes)) says which token row
/// each tower row lands in. Returns the merged embedding — a fresh `Value`
/// aliased onto `y`'s slot, which is the in-place pair every `*_out` field in
/// this IR constructs, because a scatter writes SOME rows of its output and
/// the rest are the ones `layout.embed` already wrote.
pub fn scatter_rows(src: &Value, routes: &Value, y: &Value) -> Value {
    let r = y.rec();
    let y_out = r.fresh(y.ty().clone());
    r.push(
        Layout::ScatterRows {
            src: src.id(),
            routes: routes.id(),
            y: y.id(),
            y_out: y_out.id(),
        },
        &[src, routes, y],
    );
    y_out
}
