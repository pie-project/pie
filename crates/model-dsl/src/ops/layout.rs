//! The `Layout` family: the embedding gather and the shape-only splits and
//! selects that move rows around without touching their values.

use super::*;

pub fn embed(ids: &Value, table: &Weight, vocab: u32) -> Value {
    let r = ids.rec();
    let y = r.fresh(tensor(Dim::Tokens, table.dim(1), table.compute_dtype()));
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
