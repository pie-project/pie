//! Gathers, splits and row moves.

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, Cx, expect};
use crate::error::{Error, refuse};
use crate::program::{FabricOp, FabricStep, Guarded, HostOp, Shard};
use crate::tensor::Tensor;

/// The most PEs an embedding table spreads over for a fabric gather (one
/// rectangle row of partials).
const EMBED_PES: u32 = 256;

/// How many PEs hold `table` for a gather, each a whole number of rows
/// within an array, beside `beside` words (the ids and two copies of the
/// output); `None` sends the gather to the host.
fn embed_pes(table: Tensor, beside: u64) -> Option<u32> {
    use crate::linear::gemm::{ARRAY_WORDS, FABRIC_SUM_WORDS, pe_words, resident_pes};
    let budget = pe_words();
    // The resident row: the table over its PEs, or whole on each.
    if let Some(n) = resident_pes() {
        let fits = |n: u32| {
            table.rows.is_multiple_of(n) && {
                let share = u64::from(table.rows / n) * u64::from(table.width);
                let extra = if n > 1 { FABRIC_SUM_WORDS } else { 0 };
                share <= ARRAY_WORDS && share + beside + extra <= budget
            }
        };
        return [n, 1].into_iter().find(|n| fits(*n));
    }
    (1..=EMBED_PES.min(table.rows))
        .filter(|n| table.rows.is_multiple_of(*n))
        .find(|n| {
            let share = u64::from(table.rows / n) * u64::from(table.width);
            let extra = if *n > 1 { FABRIC_SUM_WORDS } else { 0 };
            share <= ARRAY_WORDS && share + beside + extra <= budget
        })
}

/// The rows of a row-wise copy over PEs: `rg` row groups when every buffer
/// has `rows` rows and a group's words fit a PE, else one PE when the
/// whole fits, else `None` (the host copies).
fn copy_split(op: &'static str, rows: u32, per_row: u64, same_rows: bool) -> Option<u32> {
    use crate::linear::gemm::pe_words;
    if same_rows {
        return crate::cx::tile_split(op, rows, 1, per_row, 0, 0, 1).ok().map(|(rg, _)| rg);
    }
    (per_row * u64::from(rows) <= pe_words()).then_some(1)
}

/// A gather of `table` rows into `y` (`y_words` a PE) over `n` PEs holding
/// the table by rows: every PE writes its part (zeros where it holds no
/// row) and the row of PEs adds the parts into the first PE's `y`;
/// returns the pointer the kernel writes.
fn embed_over(
    cx: &mut Cx<'_>,
    n: u32,
    table: &crate::program::Buf,
    y: &crate::program::Buf,
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
    // Under the resident placement every PE reads `y` in place after:
    // the sum comes back to the row.
    let resident = crate::linear::gemm::resident_pes().is_some();
    cx.fabric_sum_with(&partial, y, y_words, (n, 1), resident);
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
    const OP: &str = "layout.embed";
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
    let pes = embed_pes(table, ids.elements() + 2 * y_words);
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
        cx.library("k_embed_part");
        cx.call(
            "k_embed_part",
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
    const OP: &str = "layout.split_q_gate";
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
        OP,
        rows,
        2 * u64::from(packed.width),
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
        cx.library("k_split_q_gate");
        cx.call(
            "k_split_q_gate",
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
    const OP: &str = "layout.split_rows";
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
        OP,
        rows,
        2 * u64::from(x.width),
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
        cx.library("k_split_rows");
        cx.call(
            "k_split_rows",
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
    const OP: &str = "layout.gather_rows";
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
        cx.library("k_gather_rows");
        cx.call(
            "k_gather_rows",
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
    const OP: &str = "layout.scatter_live_rows";
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
        cx.library("k_scatter_live_rows");
        cx.call(
            "k_scatter_live_rows",
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
    const OP: &str = "layout.merge_rows";
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
            "k_copy",
            vec![Arg::Ptr(yb), Arg::Int(0), Arg::Ptr(xb), Arg::Int(0), Arg::Int(n as i64)],
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
    const OP: &str = "layout.embed_concat";
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
    let pes = embed_pes(table, ids.elements() + 2 * y_words);
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
        cx.library("k_embed_concat_part");
        cx.call(
            "k_embed_concat_part",
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

/// The `k` largest of each row of `x`, largest first: ties go to the lower
/// column, NaN is never picked, and a slot no value fills answers value 0
/// at column 0. Values land f32, indices i32.
pub fn topk(
    ctx: &Ctx<'_>,
    x: Tensor,
    k: u32,
    values: Tensor,
    indices: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.topk";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, values, &[Dtype::F32])?;
    expect(OP, indices, &[Dtype::I32])?;
    let (rows, width) = (x.rows, x.width);
    if rows == 0 || width == 0 || k == 0 || k > width {
        return Err(refuse(OP, format!("x is {rows}x{width}, k = {k}")));
    }
    if values.rows != rows || values.width != k || indices.rows != rows || indices.width != k {
        return Err(refuse(
            OP,
            format!(
                "the values are {}x{} and the indices {}x{} for {rows} rows of {k}",
                values.rows, values.width, indices.rows, indices.width
            ),
        ));
    }
    // Rows over PEs (a row whole on its PE); a row past a PE ranks on the
    // host.
    let split = crate::cx::tile_split(OP, rows, width, u64::from(width) + 2 * u64::from(k), 0, 0, width).ok();
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let vb = cx.write(values)?;
        let ib = cx.write(indices)?;
        let Some((rg, _)) = split else {
            cx.host(HostOp::TopK {
                x: xb.name,
                values: vb.name,
                indices: ib.name,
                rows,
                width,
                k,
            });
            return Ok(());
        };
        if rg > 1 {
            cx.over(rg);
            for b in [&xb, &vb, &ib] {
                cx.shard(b, Shard::Rows(rg));
            }
        }
        cx.library("k_topk");
        cx.call(
            "k_topk",
            vec![
                Arg::Ptr(xb),
                Arg::Ptr(vb),
                Arg::Ptr(ib),
                Arg::Int(i64::from(rows / rg)),
                Arg::Int(i64::from(width)),
                Arg::Int(i64::from(k)),
            ],
        );
        Ok(())
    })
}

/// Writes the index of each row's largest value into column `column` of the
/// i32 `y`; ties go to the lowest index, NaN is never picked, an all-NaN row
/// answers 0. The rows ranked are the vocabulary-sized logits the host
/// already holds, so the host ranks them.
pub fn argmax(ctx: &Ctx<'_>, x: Tensor, column: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.argmax";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, y, &[Dtype::I32])?;
    if x.rows == 0 || x.width == 0 {
        return Err(refuse(OP, format!("x is {}x{}", x.rows, x.width)));
    }
    if column >= y.width {
        return Err(refuse(
            OP,
            format!(
                "column {column} is outside the {}-wide plane it writes",
                y.width
            ),
        ));
    }
    if x.rows != y.rows {
        return Err(refuse(
            OP,
            format!("{} rows ranked into {} rows", x.rows, y.rows),
        ));
    }
    // Each column block's PE finds its greatest lane and index (a pair a
    // row); the blocks of a row group (one rectangle row) gather those on
    // the row's first PE, which picks among them into its rows of y.
    let fabric = crate::linear::gemm::fabric_sum();
    let whole = if fabric { crate::linear::gemm::FABRIC_SUM_WORDS } else { 0 };
    let (rg, cg) = crate::cx::tile_split(OP, x.rows, x.width, u64::from(x.width) + 4, 0, whole, 1)?;
    let (rows_local, width_local) = (x.rows / rg, x.width / cg);
    let pes = rg * cg;
    if cg > 1 && !fabric {
        return argmax_host_merge(ctx, x, column, y, rg, cg);
    }
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        cx.read(y)?;
        let yb = cx.write(y)?;
        let parts = cx.scratch("am_parts", 2 * u64::from(rows_local));
        if pes > 1 {
            cx.over(pes);
            cx.shard(
                &xb,
                Shard::Tile {
                    rows: rg,
                    cols: cg,
                    segments: 1,
                },
            );
            cx.shard(
                &yb,
                Shard::Roots {
                    rows: rg,
                    cols: 1,
                    depth: cg,
                },
            );
        }
        cx.library("k_argmax_part");
        cx.library("k_argmax_merge");
        cx.call(
            "k_argmax_part",
            vec![
                Arg::Ptr(xb),
                Arg::Scratch(parts.clone(), "f32"),
                Arg::Int(i64::from(rows_local)),
                Arg::Int(i64::from(width_local)),
            ],
        );
        if cg == 1 {
            cx.call(
                "k_argmax_merge",
                vec![
                    Arg::Scratch(parts.clone(), "f32"),
                    Arg::Ptr(yb),
                    Arg::Int(1),
                    Arg::Int(i64::from(rows_local)),
                    Arg::Int(i64::from(width_local)),
                    Arg::Int(i64::from(column)),
                    Arg::Int(i64::from(y.width)),
                ],
            );
            return Ok(());
        }
        let all = cx.scratch("am_all", 2 * u64::from(rows_local) * u64::from(cg));
        let steps = vec![FabricStep {
            calls: Vec::new(),
            op: FabricOp::Gather {
                send: parts.clone(),
                recv: all.clone(),
                count: 2 * u64::from(rows_local),
            },
        }];
        let finish = vec![Guarded::root(
            "k_argmax_merge",
            vec![
                Arg::Scratch(all.clone(), "f32"),
                Arg::Ptr(yb.clone()),
                Arg::Int(i64::from(cg)),
                Arg::Int(i64::from(rows_local)),
                Arg::Int(i64::from(width_local)),
                Arg::Int(i64::from(column)),
                Arg::Int(i64::from(y.width)),
            ],
        )];
        cx.fabric((cg, rg), steps, finish);
        Ok(())
    })
}

/// The argmax with the blocks' pairs merged on the host
/// (`PIE_CEREBRAS_FABRIC_SUM=0`).
fn argmax_host_merge(
    ctx: &Ctx<'_>,
    x: Tensor,
    column: u32,
    y: Tensor,
    rg: u32,
    cg: u32,
) -> Result<(), Error> {
    let (rows_local, width_local) = (x.rows / rg, x.width / cg);
    let pes = rg * cg;
    let parts_name = format!("am_{}_{}_{column}", x.buf, y.buf);
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let parts = cx
            .program()
            .declare(&parts_name, Dtype::F32, x.rows, 2 * cg, true)?;
        if pes > 1 {
            cx.over(pes);
            let tile = Shard::Tile {
                rows: rg,
                cols: cg,
                segments: 1,
            };
            cx.shard(&xb, tile);
            cx.shard(&parts, tile);
        }
        cx.library("k_argmax_part");
        cx.call(
            "k_argmax_part",
            vec![
                Arg::Ptr(xb),
                Arg::Ptr(parts),
                Arg::Int(i64::from(rows_local)),
                Arg::Int(i64::from(width_local)),
            ],
        );
        Ok(())
    })?;
    ctx.emit(&mut |cx| {
        let parts = cx
            .program()
            .declare(&parts_name, Dtype::F32, x.rows, 2 * cg, true)?;
        cx.read(y)?;
        let yb = cx.write(y)?;
        cx.host(HostOp::ArgmaxMerge {
            parts: parts.name,
            y: yb.name,
            rows: x.rows,
            cg,
            block: width_local,
            column,
            y_width: y.width,
        });
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
    const OP: &str = "layout.split_qkv";
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
        OP,
        rows,
        2 * u64::from(packed.width),
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
        cx.library("k_split_qkv");
        cx.call(
            "k_split_qkv",
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
    const OP: &str = "layout.embed_weighted";
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
    let pes = embed_pes(table, ids.elements() + weights.elements() + 2 * y_words);
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
        cx.library("k_embed_weighted_part");
        cx.call(
            "k_embed_weighted_part",
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
