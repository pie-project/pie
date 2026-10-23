//! Dense projections: `y = act · wᵀ`.

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::library::Kernel;
use crate::phase::{HostOp, Shard};
use crate::plan::Product;
use crate::tensor::Tensor;

/// `y[m, n] = Σ_k act[m, k] · w[n, k]` over the first `y.rows` rows of `act`,
/// accumulated in f32.
pub fn matmul(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul", act, w, y)
}

/// The vocabulary projection; same contract as [`matmul`].
pub fn lm_head(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.lm_head", act, w, y)
}

fn act_x_wt(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    expect(op, act, &[Dtype::Bf16, Dtype::F32])?;
    expect(op, w, &[Dtype::Bf16, Dtype::F32])?;
    expect(op, y, &[Dtype::Bf16, Dtype::F32])?;
    let (k, n) = (act.width, y.width);
    if w.width != k || w.rows != n {
        return Err(refuse(
            op,
            format!(
                "w is {}x{}; act is {k} wide and y is {n} wide",
                w.rows, w.width
            ),
        ));
    }
    if y.rows > act.rows {
        return Err(refuse(
            op,
            format!("y has {} rows, act has {}", y.rows, act.rows),
        ));
    }
    let m = y.rows;
    if m == 0 {
        return Ok(());
    }
    // A weight larger than one PE's share is split over a row of PEs: each
    // holds `n / pes` of its rows and produces those columns of `y`; the
    // activations are whole on every PE. A weight no row of PEs holds (the
    // vocabulary projection), or one whose extents divide into no grid
    // within a PE (a 4304-wide projection: 4304 = 16 · 269), is multiplied
    // on the host.
    if let Some(pes) = ctx.plan().fabric().resident_pes {
        return resident_matmul(ctx, op, act, w, y, m, k, n, pes);
    }
    ctx.emit(&mut |cx| {
        let ab = cx.read(act)?;
        // The weight packed (two bf16 a word) when the engine keeps it so:
        // half the words a share, the kernel widening as it reads.
        let half = cx.read_half(w)?;
        let packed = half.is_some();
        let product = Product { m, k, n, packed };
        let Some((rg, cg, kg)) = ctx.plan().grid(product) else {
            let wb = cx.read(w)?;
            let yb = cx.write(y)?;
            cx.host(HostOp::Matmul {
                act: ab.name,
                w: wb.name,
                y: yb.name,
                m,
                k,
                n,
            });
            return Ok(());
        };
        let wb = match half {
            Some(b) => b,
            None => cx.read(w)?,
        };
        let yb = cx.write(y)?;
        // A depth split leaves every PE a partial of its y block. On the
        // fabric, the `kg` PEs of a block form one row of a `kg × (rg · cg)`
        // rectangle and add their partials into the row's first PE; else
        // every partial comes down and the host adds them.
        let fabric = kg > 1 && cx.plan().fabric().fabric_sum;
        if rg > 1 {
            // PE `x` of the `rg × cg × kg` grid: rows `x / (cg·kg)`, output
            // columns `(x / kg) % cg`, depth slice `x % kg`.
            cx.over(rg * cg * kg);
            cx.shard(
                &ab,
                Shard::RowsDepth {
                    rows: rg,
                    cols: cg,
                    depth: kg,
                },
            );
            cx.shard(
                &wb,
                Shard::ColsDepth {
                    rows: rg,
                    cols: cg,
                    depth: kg,
                },
            );
            let shard = if fabric {
                Shard::Roots {
                    rows: rg,
                    cols: cg,
                    depth: kg,
                }
            } else {
                Shard::SumGrid {
                    rows: rg,
                    cols: cg,
                    depth: kg,
                }
            };
            cx.shard(&yb, shard);
        } else if kg > 1 {
            // PE `x` of the `cg × kg` grid: output columns `x / kg`, depth
            // slice `x % kg`.
            cx.over(cg * kg);
            cx.shard(
                &ab,
                Shard::ColsBy {
                    parts: kg,
                    period: 1,
                },
            );
            cx.shard(&wb, Shard::Grid { rows: cg, cols: kg });
            let shard = if fabric {
                Shard::Roots {
                    rows: 1,
                    cols: cg,
                    depth: kg,
                }
            } else {
                Shard::SumCols {
                    parts: cg,
                    period: kg,
                }
            };
            cx.shard(&yb, shard);
        } else if cg > 1 {
            cx.over(cg);
            cx.shard(&wb, Shard::Rows(cg));
            cx.shard(&yb, Shard::Cols(cg));
        }
        let kernel = if packed {
            Kernel::MatmulPacked
        } else {
            Kernel::Matmul
        };
        let count = u64::from(m / rg) * u64::from(n / cg);
        let out = if fabric {
            let partial = cx.scratch("partial", count);
            cx.fabric_sum(&partial, &yb, count, (kg, rg * cg));
            Arg::Scratch(partial, "f32")
        } else {
            Arg::Ptr(yb.clone())
        };
        cx.call(
            kernel,
            vec![
                Arg::Ptr(ab),
                Arg::Ptr(wb),
                out,
                Arg::Int(i64::from(m / rg)),
                Arg::Int(i64::from(k / kg)),
                Arg::Int(i64::from(n / cg)),
            ],
        );
        Ok(())
    })
}

/// Words the collectives library and a reduce's partial copy cost a PE
/// beside its shares: a depth-split grid on the fabric fits its `y` twice
/// (the partial and the sum) and the library's buffers and code.
pub const FABRIC_SUM_WORDS: u64 = 1536;

/// The resident form of `y = act · wᵀ` over a row of `pes` PEs: the
/// activations whole on every PE, the weight by rows (output columns)
/// across the row when its rows divide so and a share fits, each PE's
/// column block of `y` gathered on the first PE and broadcast, so every PE
/// ends with the whole `y`; a weight the row cannot split is whole on
/// every PE (when it fits) and so is the product.
#[allow(clippy::too_many_arguments)]
fn resident_matmul(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
    m: u32,
    k: u32,
    n: u32,
    pes: u32,
) -> Result<(), Error> {
    use crate::phase::{FabricOp, FabricStep, Guarded};
    let budget = ctx.plan().fabric().pe_words;
    let (m64, k64, n64) = (u64::from(m), u64::from(k), u64::from(n));
    let a_words = u64::from(act.rows) * k64;
    let y_words = m64 * n64;
    let split = n.is_multiple_of(pes) && {
        let share = (n64 / u64::from(pes)) * k64;
        share <= ARRAY_WORDS
            && a_words + share + 2 * y_words + y_words / u64::from(pes) + FABRIC_SUM_WORDS <= budget
    };
    if !split && (a_words + n64 * k64 + y_words > budget || n64 * k64 > ARRAY_WORDS) {
        return Err(refuse(
            op,
            format!(
                "the resident row of {pes} PEs holds neither a {n}x{k} weight whole nor by rows ({n} rows over {pes})"
            ),
        ));
    }
    ctx.emit(&mut |cx| {
        let ab = cx.read(act)?;
        let half = cx.read_half(w)?;
        let packed = half.is_some();
        let wb = match half {
            Some(b) => b,
            None => cx.read(w)?,
        };
        let yb = cx.write(y)?;
        cx.over(pes);
        let kernel = if packed {
            Kernel::MatmulPacked
        } else {
            Kernel::Matmul
        };
        if !split {
            cx.call(
                kernel,
                vec![
                    Arg::Ptr(ab),
                    Arg::Ptr(wb),
                    Arg::Ptr(yb),
                    Arg::Int(i64::from(m)),
                    Arg::Int(i64::from(k)),
                    Arg::Int(i64::from(n)),
                ],
            );
            return Ok(());
        }
        let nl = n / pes;
        cx.shard(&wb, Shard::Rows(pes));
        let part = cx.scratch("part", m64 * u64::from(nl));
        let all = cx.scratch("all", y_words);
        cx.call(
            kernel,
            vec![
                Arg::Ptr(ab),
                Arg::Ptr(wb),
                Arg::Scratch(part.clone(), "f32"),
                Arg::Int(i64::from(m)),
                Arg::Int(i64::from(k)),
                Arg::Int(i64::from(nl)),
            ],
        );
        let steps = vec![
            FabricStep {
                calls: Vec::new(),
                op: FabricOp::Gather {
                    send: part.clone(),
                    recv: all.clone(),
                    count: m64 * u64::from(nl),
                },
            },
            FabricStep {
                calls: Vec::new(),
                op: FabricOp::Broadcast {
                    buf: all.clone(),
                    count: y_words,
                },
            },
        ];
        let finish = vec![Guarded::all(
            Kernel::ColsUnblock,
            vec![
                Arg::Scratch(all.clone(), "f32"),
                Arg::Ptr(yb.clone()),
                Arg::Int(i64::from(pes)),
                Arg::Int(i64::from(m)),
                Arg::Int(i64::from(nl)),
            ],
        )];
        cx.collectives((pes, 1), steps, finish);
        // `y` lands with the broadcast: its rounding follows the steps.
        cx.round_after_steps(&yb);
        Ok(())
    })
}

/// The most words one array holds: a word address of 16 bits (the probe's
/// 8448-word array fails a 14-bit-plus-shift relocation).
pub const ARRAY_WORDS: u64 = 8192;
