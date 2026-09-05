//! The `Layout` family: the embedding gather and the shape-only splits and
//! selects that move rows around without touching their values.

use super::*;

/// A gather lands on its ids' axis: `y` carries `ids`' row space, not
/// `Dim::Tokens`.
///
/// `Layout::EmbedConcat`: `y[r] = table[ids[r, 0]] ++ ... ++ table[ids[r,
/// heads-1]]` — the PLE n-gram read, every hashed head's row side by side.
/// `ids` is `[rows, heads]` i32; `heads` is read off its width.
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

/// The gather that interpolates: `y[r] = sum_t weights[r, t] * table[ids[r, t]]`.
///
/// The tower's learned position grid is resampled to each image's grid, so a
/// patch row reads `taps` rows under `taps` weights (`ids`/`weights` are
/// `[Dim::Patches, taps]`, `taps` read off their width). The native grid uses
/// [`embed`] instead, since the resample there is the identity.
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

/// The spatial pool: the mean of every `side * side` consecutive rows of `x`,
/// one output row each. `side == 1` is the identity. The result carries `x`'s
/// type and row space, with its leading `rows / side^2` rows written; this
/// relies on the submission laying each `side * side` square out contiguously.
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

/// The merging fold: every `side * side` consecutive rows of `x` laid end to
/// end, one output row of `side^2` times the width. The result is
/// `[Dim::Patches, side^2 * width]` with its leading `rows / side^2` rows
/// written; the rest want a `-1` in `patch_routes` (see [`scatter_live_rows`]).
/// Whether the norm goes before or after this call is checkpoint-specific.
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

/// The per-row argmax of each of `xs`, side by side: `[rows, xs.len()]` i32.
/// `argmax(&[&logits])` is the token a draft chain feeds its next step;
/// `argmax(&[&l0, &l1, &l2])` is the `[rows, depth]` drafts plane
/// [`seam::MTP_DRAFTS`](crate::seam::MTP_DRAFTS) exports. Every `xs` shares
/// one row space.
pub fn argmax(xs: &[&Value]) -> Value {
    let first = xs.first().expect("an argmax over at least one value");
    let r = first.rec();
    let y = r.fresh(tensor(first.rows(), xs.len() as u64, Dtype::I32));
    r.push(
        Layout::Argmax {
            xs: xs.iter().map(|x| x.id()).collect(),
            y: y.id(),
        },
        xs,
    );
    y
}

/// The `k` largest entries of every row, sorted descending — `(values
/// [rows, k] f32, indices [rows, k] i32)`. Ties to the lower column, a NaN
/// never chosen: the argmax rule, so the first column IS [`argmax`].
pub fn topk(x: &Value, k: u32) -> (Value, Value) {
    let r = x.rec();
    let values = r.fresh(tensor(x.rows(), u64::from(k), Dtype::F32));
    let indices = r.fresh(tensor(x.rows(), u64::from(k), Dtype::I32));
    r.push(
        Layout::TopK {
            x: x.id(),
            k,
            values: values.id(),
            indices: indices.id(),
        },
        &[x],
    );
    (values, indices)
}

/// [`scatter_rows`] plus a negative `routes` entry meaning "this row has no
/// destination" — what a compacting fold ([`pool_rows`], [`merge_rows`])
/// owes the scatter, since `routes` has one entry per row of the full
/// rectangle but the fold only writes `rows / side^2` of them. The plain
/// scatter still refuses a negative route; only a plan declaring this op
/// admits the sentinel.
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

/// The tower's rows written into the token rows the image placeholders
/// occupy. `src` is a patch rectangle, `y` a token one, `routes` says which
/// token row each tower row lands in. Returns a fresh `Value` aliased onto
/// `y`'s slot: a scatter writes only some rows, the rest are `layout.embed`'s.
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
