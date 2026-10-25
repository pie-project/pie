use super::*;

pub fn embed_concat(ids: &Value, table: &Weight, vocab: u32) -> Value {
    let r = ids.rec();
    let y = r.fresh(tensor(
        ids.rows(),
        ids.width() * table.dim(1),
        table.compute_dtype(),
    ));
    r.push(
        Gather::Embed {
            ids: ids.id(),
            table: r.weight(table),
            scales: (table.dtype == Dtype::Mxfp8).then(|| r.weight_plane(table, poem_ir::SCALES)),
            vocab,
            y: y.id(),
            weights: None,
            concat: true,
        },
        &[ids],
    );
    y
}

pub fn embed(ids: &Value, table: &Weight, vocab: u32) -> Value {
    let r = ids.rec();
    let y = r.fresh(tensor(ids.rows(), table.dim(1), table.compute_dtype()));
    r.push(
        Gather::Embed {
            ids: ids.id(),
            table: r.weight(table),
            vocab,
            y: y.id(),
            weights: None,
            scales: None,
            concat: false,
        },
        &[ids],
    );
    y
}

pub fn embed_weighted(ids: &Value, weights: &Value, table: &Weight, vocab: u32) -> Value {
    let r = ids.rec();
    let y = r.fresh(tensor(ids.rows(), table.dim(1), table.compute_dtype()));
    r.push(
        Gather::Embed {
            ids: ids.id(),
            weights: Some(weights.id()),
            table: r.weight(table),
            vocab,
            y: y.id(),
            scales: None,
            concat: false,
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
        Gather::SplitQkv {
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

/// `packed`'s `heads` (query, key-value) heads, queries and keys
/// `widths[0]` wide and values `widths[1]`, as q, k and v of `head_dim`-wide
/// heads zero past their widths.
pub fn split_qkv_padded(
    packed: &Value,
    heads: [u32; 2],
    widths: [u32; 2],
    head_dim: u32,
) -> (Value, Value, Value) {
    let r = packed.rec();
    // (Saturating: a pattern's holes stand for any count.)
    let (q_width, kv_width) = (
        heads[0].saturating_mul(head_dim),
        heads[1].saturating_mul(head_dim),
    );
    let q = r.fresh(tensor(packed.rows(), q_width, packed.dtype()));
    let k = r.fresh(tensor(packed.rows(), kv_width, packed.dtype()));
    let v = r.fresh(tensor(packed.rows(), kv_width, packed.dtype()));
    r.push(
        Gather::SplitQkvPadded {
            packed: packed.id(),
            heads,
            widths,
            head_dim,
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
        Gather::SplitQGate {
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
    let rest = x.width().checked_sub(u64::from(width)).unwrap_or_else(|| {
        panic!(
            "split_rows at {width} of a {}-wide row: the split is past the row",
            x.width()
        )
    });
    let left = r.fresh(tensor(x.rows(), width, x.dtype()));
    let right = r.fresh(tensor(x.rows(), rest, x.dtype()));
    r.push(
        Gather::SplitRows {
            x: x.id(),
            width,
            left: left.id(),
            right: right.id(),
        },
        &[x],
    );
    (left, right)
}

/// The rows of `x` at `rows`, one a row of `rows` (a readout's row, or a
/// token's lane, say); a value of one axis (a row's token) gathers to one
/// axis too.
pub fn gather_rows(x: &Value, rows: &Value) -> Value {
    let r = x.rec();
    let y = r.fresh(match x.ty() {
        Ty::Tensor { shape, dtype } if shape.len() == 1 => Ty::Tensor {
            shape: vec![rows.rows()],
            dtype: *dtype,
        },
        _ => tensor(rows.rows(), x.width(), x.dtype()),
    });
    r.push(
        Gather::Rows {
            x: x.id(),
            rows: rows.id(),
            y: y.id(),
        },
        &[x, rows],
    );
    y
}

/// Each of `lanes`' rows' count of the rows `csr` (a selection's
/// `group_indptr`) packs in its lane's group (`groups`, the
/// `group_of_lane`), `[rows, 1]` f32: a peer lane's rows, read from the
/// group structure.
pub fn group_rows(csr: &Value, groups: &Value, lanes: &Value) -> Value {
    for (what, v, rows) in [
        ("csr", csr, Dim::LanesPlus(1)),
        ("groups", groups, Dim::Lanes),
    ] {
        assert!(
            v.rows() == rows && v.dtype() == Dtype::I32,
            "`gather::group_rows`'s {what} is a {rows:?} i32 table, not {:?}",
            v.ty()
        );
    }
    assert!(
        lanes.rows().axis() == Some(poem_ir::RowAxis::Tokens) && lanes.dtype() == Dtype::I32,
        "`gather::group_rows` counts for each token row's lane, not {:?}",
        lanes.ty()
    );
    let r = lanes.rec();
    let y = r.fresh(tensor(lanes.rows(), 1u64, Dtype::F32));
    r.push(
        Gather::GroupRows {
            csr: csr.id(),
            groups: groups.id(),
            lanes: lanes.id(),
            y: y.id(),
        },
        &[csr, groups, lanes],
    );
    y
}

pub fn select(table: &Value, layer: u32, width: u32) -> Value {
    let r = table.rec();
    let y = r.fresh(tensor(table.rows(), width, table.dtype()));
    r.push(
        Gather::Select {
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
        Gather::PoolRows {
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
        Gather::MergeRows {
            x: x.id(),
            side,
            y: y.id(),
        },
        &[x],
    );
    y
}

/// `xs` side by side: a row's columns of each, in order.
pub fn concat_cols(xs: &[&Value]) -> Value {
    let first = xs.first().expect("a concat of at least one value");
    let r = first.rec();
    let width: u64 = xs.iter().map(|x| x.width()).sum();
    let y = r.fresh(tensor(first.rows(), width, first.dtype()));
    r.push(
        Gather::ConcatCols {
            xs: xs.iter().map(|x| x.id()).collect(),
            y: y.id(),
        },
        xs,
    );
    y
}

pub fn scatter_live_rows(src: &Value, routes: &Value, y: &Value) -> Value {
    let r = y.rec();
    let y_out = r.fresh(y.ty().clone());
    r.push(
        Gather::ScatterLiveRows {
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
        Gather::ScatterRows {
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
        Gather::PackRows {
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
        Gather::UnpackRows {
            x: x.id(),
            perm: perm.id(),
            y: y.id(),
        },
        &[x, perm],
    );
    y
}

/// Patch rows of raw RGB bytes (`patch` × `patch` pixels, HWC) to the
/// tower's rows: normalized by `mean` and `std` per channel, columns in
/// `order`, the frame repeated `temporal` times.
pub fn pixels(x: &Value, mean: [f32; 3], std: [f32; 3], order: PixelOrder, temporal: u32) -> Value {
    let r = x.rec();
    let width = x.width();
    let patch = ((width / 3) as f64).sqrt() as u64;
    assert!(
        3 * patch * patch == width,
        "pixel rows are 3 · patch² bytes wide, not {width}"
    );
    let y = r.fresh(tensor(x.rows(), u64::from(temporal) * width, Dtype::Bf16));
    r.push(
        Gather::Pixels {
            x: x.id(),
            patch: patch as u32,
            mean,
            std,
            order,
            temporal,
            y: y.id(),
        },
        &[x],
    );
    y
}

/// Position-table taps for each patch, from its grid position, its image's
/// grid and the patch segments: `(ids, weights)` for `embed_weighted`.
pub fn grid_taps(
    positions: &Value,
    grids: &Value,
    segments: &Value,
    kind: TapKind,
    side: u32,
) -> (Value, Value) {
    let r = positions.rec();
    let taps = u64::from(kind.taps());
    let ids = r.fresh(tensor(positions.rows(), taps, Dtype::I32));
    let weights = r.fresh(tensor(positions.rows(), taps, Dtype::F32));
    r.push(
        Gather::GridTaps {
            positions: positions.id(),
            grids: grids.id(),
            segments: segments.id(),
            kind,
            side,
            ids: ids.id(),
            weights: weights.id(),
        },
        &[positions, grids, segments],
    );
    (ids, weights)
}

#[allow(clippy::too_many_arguments)]
pub fn ngram_ids(
    ids: &Value,
    state: ValueId,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    map: Option<&Weight>,
) -> Value {
    let r = ids.rec();
    let ngram_ids = r.fresh(tensor(ids.rows(), primes.len() as u64, Dtype::I32));
    r.push(
        Gather::NgramIds {
            ids: ids.id(),
            state,
            eos,
            mults: mults.to_vec(),
            primes: primes.to_vec(),
            offsets: offsets.to_vec(),
            heads_per_ngram,
            map: map.map(|w| r.weight(w)),
            ngram_ids: ngram_ids.id(),
        },
        &[ids],
    );
    ngram_ids
}

#[allow(clippy::too_many_arguments)]
pub fn ngram_ids_chunked(
    ids: &Value,
    state: ValueId,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    map: Option<&Weight>,
) -> Value {
    let r = ids.rec();
    let ngram_ids = r.fresh(tensor(ids.rows(), primes.len() as u64, Dtype::I32));
    r.push(
        Gather::NgramIdsChunked {
            ids: ids.id(),
            state,
            eos,
            mults: mults.to_vec(),
            primes: primes.to_vec(),
            offsets: offsets.to_vec(),
            heads_per_ngram,
            map: map.map(|w| r.weight(w)),
            ngram_ids: ngram_ids.id(),
        },
        &[ids],
    );
    ngram_ids
}
