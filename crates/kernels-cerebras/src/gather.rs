//! Gathers, splits, row moves and embedding lookups.

pub mod ngram;

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, Cx, expect};
use crate::error::{Error, refuse};
use crate::library::Kernel;
use crate::phase::{HostOp, Shard};
use crate::plan::{Footprint, Held, Spread};
use crate::tensor::Tensor;

/// How many PEs hold `table` for a gather, each a whole number of rows,
/// beside `beside` words (the ids and two copies of the output), the parts
/// summed on the fabric; `None` sends the gather to the host.
fn embed_pes(ctx: &Ctx<'_>, table: Tensor, beside: u64) -> Option<u32> {
    let width = u64::from(table.width);
    ctx.plan().spread(&Spread {
        units: table.rows,
        collective: true,
        held: &|n| {
            let share = u64::from(table.rows / n) * width;
            Held {
                words: share + beside,
                array: share,
            }
        },
    })
}

/// The rows of a row-wise copy over PEs: `rg` row groups when every buffer
/// has `rows` rows and a group's words fit a PE, else one PE when the
/// whole fits, else `None` (the host copies).
fn copy_split(
    ctx: &Ctx<'_>,
    op: &'static str,
    rows: u32,
    width: u32,
    same_rows: bool,
) -> Option<u32> {
    // The source row and its parts.
    let per_row = 2 * u64::from(width);
    if same_rows {
        return ctx
            .plan()
            .tile(
                op,
                Footprint {
                    rows,
                    width: 1,
                    per_row,
                    per_col: 0,
                    whole: 0,
                    merge: 0,
                    unit: 1,
                },
            )
            .ok()
            .map(|(rg, _)| rg);
    }
    ctx.plan()
        .holds(Held {
            words: per_row * u64::from(rows),
            array: u64::from(width) * u64::from(rows),
        })
        .then_some(1)
}

/// A gather of `table` rows into `y` (`y_words` a PE) over `n` PEs holding
/// the table by rows: every PE writes its part (zeros where it holds no
/// row) and the row of PEs adds the parts into the first PE's `y`;
/// returns the pointer the kernel writes.
fn embed_over(
    cx: &mut Cx<'_>,
    n: u32,
    table: &crate::phase::Buf,
    y: &crate::phase::Buf,
    y_words: u64,
) -> Arg {
    if n == 1 {
        return Arg::Ptr(y.clone());
    }
    cx.over(n);
    cx.shard(table, Shard::Rows(n));
    cx.shard(
        y,
        Shard::Roots {
            rows: 1,
            cols: 1,
            depth: n,
        },
    );
    let partial = cx.scratch("partial", y_words);
    cx.fabric_sum(&partial, y, y_words, (n, 1));
    Arg::Scratch(partial, "f32")
}

/// `y[r] = table[ids[r]]`; ids outside `[0, min(vocab, table.rows))` read row 0.
pub fn embed(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    expect(OP, ids, &[Dtype::I32])?;
    if table.dtype != y.dtype {
        return Err(refuse(
            OP,
            format!("table is {:?}, y is {:?}", table.dtype, y.dtype),
        ));
    }
    if vocab == 0 {
        return Err(refuse(OP, "the embedding table states zero rows"));
    }
    if table.width != y.width || ids.elements() < u64::from(y.rows) {
        return Err(refuse(
            OP,
            format!(
                "table is {}x{}, y is {}x{}, {} ids",
                table.rows,
                table.width,
                y.rows,
                y.width,
                ids.elements()
            ),
        ));
    }
    let width = y.width;
    let limit = vocab.min(table.rows);
    // The table by rows over a row of PEs, each gathering what it holds, the
    // parts added on the fabric; a table no such row holds (the vocabulary
    // under the simulator's budget) is gathered by the host.
    let y_words = y.elements();
    let pes = embed_pes(ctx, table, ids.elements() + 2 * y_words);
    ctx.emit(&mut |cx| {
        let ib = cx.read(ids)?;
        let tb = cx.read(table)?;
        let yb = cx.write(y)?;
        let Some(n) = pes else {
            cx.host(HostOp::Embed {
                ids: ib.name,
                table: tb.name,
                y: yb.name,
                rows: y.rows,
                width,
                limit,
            });
            return Ok(());
        };
        let out = embed_over(cx, n, &tb, &yb, y_words);
        let share = table.rows / n;
        cx.call(
            Kernel::EmbedPart,
            vec![
                Arg::Ptr(ib),
                Arg::Ptr(tb),
                out,
                Arg::Int(i64::from(y.rows)),
                Arg::Int(i64::from(width)),
                Arg::Int(i64::from(limit)),
                Arg::PeTimes(i64::from(share)),
                Arg::Int(i64::from(share)),
            ],
        );
        Ok(())
    })
}

/// Cuts a `[q_h ‖ gate_h]`-per-head row into its query and gate halves.
pub fn split_q_gate(
    ctx: &Ctx<'_>,
    packed: Tensor,
    head_dim: u32,
    q: Tensor,
    gate: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.split_q_gate";
    expect(OP, packed, &[Dtype::Bf16, Dtype::F32])?;
    if head_dim == 0 || !packed.width.is_multiple_of(2 * head_dim) {
        return Err(refuse(
            OP,
            format!("packed is {} wide for head_dim {head_dim}", packed.width),
        ));
    }
    let heads = packed.width / (2 * head_dim);
    for (what, t) in [("q", q), ("gate", gate)] {
        if t.dtype != packed.dtype || t.width != heads * head_dim || t.rows < packed.rows {
            return Err(refuse(
                OP,
                format!(
                    "{what} is {}x{} {:?}; {}x{} wanted",
                    t.rows,
                    t.width,
                    t.dtype,
                    packed.rows,
                    heads * head_dim
                ),
            ));
        }
    }
    let rows = packed.rows;
    let split = copy_split(
        ctx,
        OP,
        rows,
        packed.width,
        q.rows == rows && gate.rows == rows,
    );
    ctx.emit(&mut |cx| {
        let pb = cx.read(packed)?;
        let qb = cx.write(q)?;
        let gb = cx.write(gate)?;
        let Some(rg) = split else {
            cx.host(HostOp::SplitQGate {
                packed: pb.name,
                q: qb.name,
                gate: gb.name,
                rows,
                heads,
                head_dim,
            });
            return Ok(());
        };
        if rg > 1 {
            cx.over(rg);
            for b in [&pb, &qb, &gb] {
                cx.shard(b, Shard::Rows(rg));
            }
        }
        cx.call(
            Kernel::SplitQGate,
            vec![
                Arg::Ptr(pb),
                Arg::Ptr(qb),
                Arg::Ptr(gb),
                Arg::Int(i64::from(rows / rg)),
                Arg::Int(i64::from(heads)),
                Arg::Int(i64::from(head_dim)),
            ],
        );
        Ok(())
    })
}

/// `left = x[:, :width]`, `right = x[:, width:]`.
pub fn split_rows(
    ctx: &Ctx<'_>,
    x: Tensor,
    width: u32,
    left: Tensor,
    right: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.split_rows";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    if width == 0 || width >= x.width {
        return Err(refuse(OP, format!("cut at {width} of {} columns", x.width)));
    }
    let rw = x.width - width;
    for (what, t, w) in [("left", left, width), ("right", right, rw)] {
        if t.dtype != x.dtype || t.width != w || t.rows < x.rows {
            return Err(refuse(
                OP,
                format!(
                    "{what} is {}x{} {:?}; {}x{w} wanted",
                    t.rows, t.width, t.dtype, x.rows
                ),
            ));
        }
    }
    let rows = x.rows;
    let split = copy_split(
        ctx,
        OP,
        rows,
        x.width,
        left.rows == rows && right.rows == rows,
    );
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let lb = cx.write(left)?;
        let rb = cx.write(right)?;
        let Some(rg) = split else {
            cx.host(HostOp::SplitRows {
                x: xb.name,
                left: lb.name,
                right: rb.name,
                rows,
                width: x.width,
                cut: width,
            });
            return Ok(());
        };
        if rg > 1 {
            cx.over(rg);
            for b in [&xb, &lb, &rb] {
                cx.shard(b, Shard::Rows(rg));
            }
        }
        cx.call(
            Kernel::SplitRows,
            vec![
                Arg::Ptr(xb),
                Arg::Ptr(lb),
                Arg::Ptr(rb),
                Arg::Int(i64::from(rows / rg)),
                Arg::Int(i64::from(x.width)),
                Arg::Int(i64::from(width)),
            ],
        );
        Ok(())
    })
}

/// `tight[i] = wide[index[i]]`; an index outside `wide` is clamped into it.
pub fn gather_rows(ctx: &Ctx<'_>, wide: Tensor, index: Tensor, tight: Tensor) -> Result<(), Error> {
    const OP: &str = "gather.rows";
    expect(OP, wide, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, index, &[Dtype::I32])?;
    if tight.dtype != wide.dtype
        || tight.width != wide.width
        || index.elements() < u64::from(tight.rows)
    {
        return Err(refuse(
            OP,
            format!(
                "wide {}x{}, tight {}x{}, {} indices",
                wide.rows,
                wide.width,
                tight.rows,
                tight.width,
                index.elements()
            ),
        ));
    }
    let last = wide.rows as i64 - 1;
    ctx.emit(&mut |cx| {
        let wb = cx.read(wide)?;
        let ib = cx.read(index)?;
        let tb = cx.write(tight)?;
        cx.call(
            Kernel::GatherRows,
            vec![
                Arg::Ptr(wb),
                Arg::Ptr(ib),
                Arg::Ptr(tb),
                Arg::Int(i64::from(tight.rows)),
                Arg::Int(i64::from(wide.width)),
                Arg::Int(last),
            ],
        );
        Ok(())
    })
}

/// `y[routes[i]] = src[i]` for every route inside `y`; other rows of `y` keep.
pub fn scatter_live_rows(
    ctx: &Ctx<'_>,
    src: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.scatter_live_rows";
    expect(OP, src, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, routes, &[Dtype::I32])?;
    if y.dtype != src.dtype || y.width != src.width || routes.elements() < u64::from(src.rows) {
        return Err(refuse(
            OP,
            format!(
                "src {}x{}, y {}x{}, {} routes",
                src.rows,
                src.width,
                y.rows,
                y.width,
                routes.elements()
            ),
        ));
    }
    ctx.emit(&mut |cx| {
        let sb = cx.read(src)?;
        let rb = cx.read(routes)?;
        cx.read(y)?;
        let yb = cx.write(y)?;
        cx.call(
            Kernel::ScatterLiveRows,
            vec![
                Arg::Ptr(sb),
                Arg::Ptr(rb),
                Arg::Ptr(yb),
                Arg::Int(i64::from(src.rows)),
                Arg::Int(i64::from(src.width)),
                Arg::Int(i64::from(y.rows)),
            ],
        );
        Ok(())
    })
}

/// `y[o] = x[o * side²] ‖ … ‖ x[(o + 1) * side² - 1]`; leftover rows drop.
pub fn merge_rows(ctx: &Ctx<'_>, x: Tensor, side: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "gather.merge_rows";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    // `side²` rows of `x` become one row of `y`; `y`'s rows past the merged
    // count keep what they held.
    let per = side.checked_mul(side).unwrap_or(0);
    let out = x.rows.checked_div(per).unwrap_or(0);
    if y.dtype != x.dtype || y.width != x.width * per || out == 0 || y.rows < out {
        return Err(refuse(
            OP,
            format!(
                "x {}x{}, y {}x{}, side {side}",
                x.rows, x.width, y.rows, y.width
            ),
        ));
    }
    let n = u64::from(out) * u64::from(y.width);
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let yb = cx.write(y)?;
        cx.call(
            Kernel::Copy,
            vec![
                Arg::Ptr(yb),
                Arg::Int(0),
                Arg::Ptr(xb),
                Arg::Int(0),
                Arg::Int(n as i64),
            ],
        );
        Ok(())
    })
}

/// `y[n] = table[ids[n, 0]] ‖ table[ids[n, 1]] ‖ …`; an id outside the
/// vocabulary lands a zero slice. The table is vocabulary-sized: the host
/// gathers the rows.
pub fn embed_concat(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    expect(OP, ids, &[Dtype::I32])?;
    if table.dtype != y.dtype {
        return Err(refuse(
            OP,
            format!("table is {:?}, y is {:?}", table.dtype, y.dtype),
        ));
    }
    if vocab == 0 {
        return Err(refuse(OP, "the embedding table states zero rows"));
    }
    let heads = ids.width;
    if heads == 0 || ids.rows != y.rows || y.width != heads * table.width {
        return Err(refuse(
            OP,
            format!(
                "ids are {}x{}, table rows {} wide, y is {}x{}",
                ids.rows, ids.width, table.width, y.rows, y.width
            ),
        ));
    }
    let (rows, width) = (y.rows, table.width);
    let limit = vocab.min(table.rows);
    let y_words = y.elements();
    let pes = embed_pes(ctx, table, ids.elements() + 2 * y_words);
    ctx.emit(&mut |cx| {
        let ib = cx.read(ids)?;
        let tb = cx.read(table)?;
        let yb = cx.write(y)?;
        let Some(n) = pes else {
            cx.host(HostOp::EmbedConcat {
                ids: ib.name,
                table: tb.name,
                y: yb.name,
                rows,
                heads,
                width,
                limit,
            });
            return Ok(());
        };
        let out = embed_over(cx, n, &tb, &yb, y_words);
        let share = table.rows / n;
        cx.call(
            Kernel::EmbedConcatPart,
            vec![
                Arg::Ptr(ib),
                Arg::Ptr(tb),
                out,
                Arg::Int(i64::from(rows)),
                Arg::Int(i64::from(heads)),
                Arg::Int(i64::from(width)),
                Arg::Int(i64::from(limit)),
                Arg::PeTimes(i64::from(share)),
                Arg::Int(i64::from(share)),
            ],
        );
        Ok(())
    })
}

/// Cuts `packed = [q | k | v]` rows into `q` (`q_width`), `k` and `v`
/// (`kv_width` each): pure data movement between phases, so the host cuts.
pub fn split_qkv(
    ctx: &Ctx<'_>,
    packed: Tensor,
    q_width: u32,
    kv_width: u32,
    q: Tensor,
    k: Tensor,
    v: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.split_qkv";
    expect(OP, packed, &[Dtype::Bf16, Dtype::F32])?;
    if q_width == 0 || kv_width == 0 || packed.width != q_width + 2 * kv_width {
        return Err(refuse(
            OP,
            format!(
                "packed is {} wide for q {q_width} and kv {kv_width}",
                packed.width
            ),
        ));
    }
    for (what, t, w) in [("q", q, q_width), ("k", k, kv_width), ("v", v, kv_width)] {
        if t.dtype != packed.dtype || t.width != w || t.rows < packed.rows {
            return Err(refuse(
                OP,
                format!(
                    "{what} is {}x{} {:?}; {}x{w} wanted",
                    t.rows, t.width, t.dtype, packed.rows
                ),
            ));
        }
    }
    let rows = packed.rows;
    let split = copy_split(
        ctx,
        OP,
        rows,
        packed.width,
        q.rows == rows && k.rows == rows && v.rows == rows,
    );
    ctx.emit(&mut |cx| {
        let pb = cx.read(packed)?;
        let qb = cx.write(q)?;
        let kb = cx.write(k)?;
        let vb = cx.write(v)?;
        let Some(rg) = split else {
            cx.host(HostOp::SplitQkv {
                packed: pb.name,
                q: qb.name,
                k: kb.name,
                v: vb.name,
                rows,
                q_width,
                kv_width,
            });
            return Ok(());
        };
        if rg > 1 {
            cx.over(rg);
            for b in [&pb, &qb, &kb, &vb] {
                cx.shard(b, Shard::Rows(rg));
            }
        }
        cx.call(
            Kernel::SplitQkv,
            vec![
                Arg::Ptr(pb),
                Arg::Ptr(qb),
                Arg::Ptr(kb),
                Arg::Ptr(vb),
                Arg::Int(i64::from(rows / rg)),
                Arg::Int(i64::from(q_width)),
                Arg::Int(i64::from(kv_width)),
            ],
        );
        Ok(())
    })
}

/// `y[n] = Σ_t weights[n, t] · table[ids[n, t]]`, accumulated in f32 in tap
/// order; an id outside the vocabulary reads row 0. The table is the
/// vocabulary-sized embedding, so the host gathers.
pub fn embed_weighted(
    ctx: &Ctx<'_>,
    ids: Tensor,
    weights: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    expect(OP, ids, &[Dtype::I32])?;
    expect(OP, weights, &[Dtype::Bf16, Dtype::F32])?;
    if table.dtype != y.dtype || table.width != y.width {
        return Err(refuse(
            OP,
            format!(
                "table is {}x{} {:?}, y is {}x{} {:?}",
                table.rows, table.width, table.dtype, y.rows, y.width, y.dtype
            ),
        ));
    }
    if ids.rows != y.rows || weights.rows != y.rows || weights.width != ids.width || ids.width == 0
    {
        return Err(refuse(
            OP,
            format!(
                "ids {}x{}, weights {}x{} for {} rows",
                ids.rows, ids.width, weights.rows, weights.width, y.rows
            ),
        ));
    }
    let y_words = y.elements();
    let limit = vocab.min(table.rows);
    let pes = embed_pes(
        ctx,
        table,
        ids.elements() + weights.elements() + 2 * y_words,
    );
    ctx.emit(&mut |cx| {
        let ib = cx.read(ids)?;
        let wb = cx.read(weights)?;
        let tb = cx.read(table)?;
        let yb = cx.write(y)?;
        let Some(n) = pes else {
            cx.host(HostOp::EmbedWeighted {
                ids: ib.name,
                weights: wb.name,
                table: tb.name,
                y: yb.name,
                rows: y.rows,
                taps: ids.width,
                width: y.width,
                limit,
            });
            return Ok(());
        };
        let out = embed_over(cx, n, &tb, &yb, y_words);
        let share = table.rows / n;
        cx.call(
            Kernel::EmbedWeightedPart,
            vec![
                Arg::Ptr(ib),
                Arg::Ptr(wb),
                Arg::Ptr(tb),
                out,
                Arg::Int(i64::from(y.rows)),
                Arg::Int(i64::from(ids.width)),
                Arg::Int(i64::from(y.width)),
                Arg::Int(i64::from(limit)),
                Arg::PeTimes(i64::from(share)),
                Arg::Int(i64::from(share)),
            ],
        );
        Ok(())
    })
}
