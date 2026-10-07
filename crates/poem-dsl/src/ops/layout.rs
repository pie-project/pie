use super::*;

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

pub fn gather_rows(x: &Value, rows: &Value) -> Value {
    let r = x.rec();
    let y = r.fresh(tensor(Dim::Readouts, x.width(), x.dtype()));
    r.push(
        Layout::GatherRows {
            x: x.id(),
            rows: rows.id(),
            y: y.id(),
        },
        &[x, rows],
    );
    y
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

pub fn pack_rows(x: &Value, perm: &Value) -> Value {
    let r = x.rec();
    assert_eq!(
        x.rows(),
        perm.rows(),
        "a permutation is over the rows it packs"
    );
    let y = r.fresh(x.ty().clone());
    r.push(
        Layout::PackRows {
            x: x.id(),
            perm: perm.id(),
            y: y.id(),
        },
        &[x, perm],
    );
    y
}

pub fn unpack_rows(x: &Value, perm: &Value) -> Value {
    let r = x.rec();
    assert_eq!(
        x.rows(),
        perm.rows(),
        "a permutation is over the rows it unpacks"
    );
    let y = r.fresh(x.ty().clone());
    r.push(
        Layout::UnpackRows {
            x: x.id(),
            perm: perm.id(),
            y: y.id(),
        },
        &[x, perm],
    );
    y
}
