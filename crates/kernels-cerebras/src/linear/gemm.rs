//! Dense projections: `y = act · wᵀ`.

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::program::{HostOp, Shard};
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
    if let Some(pes) = resident_pes() {
        return resident_matmul(ctx, op, act, w, y, m, k, n, pes);
    }
    ctx.emit(&mut |cx| {
        let ab = cx.read(act)?;
        // The weight packed (two bf16 a word) when the engine keeps it so:
        // half the words a share, the kernel widening as it reads.
        let half = cx.read_half(w)?;
        let packed = half.is_some();
        let w_words = if packed { w.elements() / 2 } else { w.elements() };
        let grid = if w_words > u64::from(max_pes()) * shard_words() {
            None
        } else {
            grid(m, k, n, packed)
        };
        let Some((rg, cg, kg)) = grid else {
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
        let fabric = kg > 1 && fabric_sum();
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
            cx.library("k_half");
            cx.library("k_matmul_packed")
        } else {
            cx.library("k_matmul")
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

/// Whether bf16 weights live on the device packed two halves a word
/// (`PIE_CEREBRAS_HALF_WEIGHTS=1`): half the memory, the kernels widening
/// as they read.
pub fn half_weights() -> bool {
    std::env::var("PIE_CEREBRAS_HALF_WEIGHTS").is_ok_and(|v| v != "0")
}

/// Whether bf16 kv pools live on the device packed two halves a word
/// (`PIE_CEREBRAS_HALF_POOL=1`): half the memory a page, attention widening
/// a key or value row as it reads it, `kv_append` packing as it lands.
pub fn half_pool() -> bool {
    std::env::var("PIE_CEREBRAS_HALF_POOL").is_ok_and(|v| v != "0")
}

/// The resident placement: every phase of a fire on one row of this many
/// PEs (`PIE_CEREBRAS_RESIDENT_PES`), activations whole on every PE, the
/// weights by rows across the row, a matmul's column blocks gathered back
/// to every PE, lanes padded to the row; `None` is the per-phase
/// placement.
pub fn resident_pes() -> Option<u32> {
    std::env::var("PIE_CEREBRAS_RESIDENT_PES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|v| *v > 1)
}

/// Whether a depth-split matmul adds its partials on the fabric
/// (`PIE_CEREBRAS_FABRIC_SUM=0` leaves the adding to the host).
pub fn fabric_sum() -> bool {
    std::env::var("PIE_CEREBRAS_FABRIC_SUM")
        .ok()
        .is_none_or(|v| v != "0")
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
    use crate::program::{FabricOp, FabricStep, Guarded};
    let budget = pe_words();
    let (m64, k64, n64) = (u64::from(m), u64::from(k), u64::from(n));
    let a_words = u64::from(act.rows) * k64;
    let y_words = m64 * n64;
    let split = n.is_multiple_of(pes) && {
        let share = (n64 / u64::from(pes)) * k64;
        share <= ARRAY_WORDS && a_words + share + 2 * y_words + y_words / u64::from(pes) + FABRIC_SUM_WORDS <= budget
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
            cx.library("k_half");
            cx.library("k_matmul_packed")
        } else {
            cx.library("k_matmul")
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
        cx.library("k_cols_unblock");
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
            "k_cols_unblock",
            vec![
                Arg::Scratch(all.clone(), "f32"),
                Arg::Ptr(yb.clone()),
                Arg::Int(i64::from(pes)),
                Arg::Int(i64::from(m)),
                Arg::Int(i64::from(nl)),
            ],
        )];
        cx.fabric((pes, 1), steps, finish);
        // `y` lands with the broadcast: its rounding follows the steps.
        cx.round_after_steps(&yb);
        Ok(())
    })
}

/// The `(row groups, column groups, depth groups)` grid of PEs a `m × k`
/// by `n × k` product spreads over. PE `(r, c, d)` holds rows `r` and depth
/// slice `d` of the activations, weight rows `c` at depth `d`, and a
/// partial sum of output block `(r, c)`, which the host adds over `d`: the
/// words moved are the activations once per column group, the weight once
/// per row group, and the output once per depth group. The grid moving the
/// fewest words whose shares fit [`pe_words`] (arrays within
/// [`ARRAY_WORDS`]) on at most [`MAX_PES`] PEs wins, the fewer PEs the
/// better among equals.
/// A grid `(rg, cg, kg)` with its `(words moved, PEs)` key.
type GridPick = ((u32, u32, u32), (u64, u32));

fn grid(m: u32, k: u32, n: u32, packed: bool) -> Option<(u32, u32, u32)> {
    let budget = pe_words();
    let most = max_pes();
    let fabric = fabric_sum();
    let per = if packed { 2 } else { 1 };
    let divisors = |x: u32| (1..=x).filter(move |d| x.is_multiple_of(*d));
    let (m64, k64, n64) = (u64::from(m), u64::from(k), u64::from(n));
    let mut best: Option<GridPick> = None;
    for rg in divisors(m) {
        for cg in divisors(n) {
            for kg in divisors(k) {
                if packed && !(k / kg).is_multiple_of(2) {
                    continue;
                }
                let a = (m64 / u64::from(rg)) * (k64 / u64::from(kg));
                let w = (n64 / u64::from(cg)) * (k64 / u64::from(kg)) / per;
                let y = (m64 / u64::from(rg)) * (n64 / u64::from(cg));
                let extra = if kg > 1 && fabric { y + FABRIC_SUM_WORDS } else { 0 };
                if a.max(w).max(y) > ARRAY_WORDS || a + w + y + extra > budget || rg * cg * kg > most {
                    continue;
                }
                let moved = m64 * k64 * u64::from(cg)
                    + n64 * k64 * u64::from(rg)
                    + m64 * n64 * u64::from(kg);
                let key = (moved, rg * cg * kg);
                if best.is_none_or(|(_, k)| key < k) {
                    best = Some(((rg, cg, kg), key));
                }
                break;
            }
        }
    }
    best.map(|(g, _)| g)
}

/// The most elements of one buffer a PE holds before it is split
/// (`PIE_CEREBRAS_SHARD_WORDS` overrides it: a small budget makes a small
/// model take every split form).
pub fn shard_words() -> u64 {
    env_budget("PIE_CEREBRAS_SHARD_WORDS", SHARD_WORDS_DEFAULT)
}

const SHARD_WORDS_DEFAULT: u64 = 4096;

/// The most data words one phase places on a PE altogether
/// (`PIE_CEREBRAS_PE_WORDS` overrides it).
pub fn pe_words() -> u64 {
    env_budget("PIE_CEREBRAS_PE_WORDS", PE_WORDS_DEFAULT)
}

fn env_budget(var: &str, default: u64) -> u64 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

/// The most PEs one phase spreads over (a rectangle up to 16 rows tall:
/// `rect_of`); a weight past `max_pes() × shard_words()` (the vocabulary
/// projection, under the default) runs on the host. The default suits the
/// simulator; a wafer takes `PIE_CEREBRAS_MAX_PES` in the hundreds of
/// thousands, and the projection with it.
pub const MAX_PES: u32 = 2048;

/// `MAX_PES`, or `PIE_CEREBRAS_MAX_PES`.
#[must_use]
pub fn max_pes() -> u32 {
    std::env::var("PIE_CEREBRAS_MAX_PES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(MAX_PES)
}

/// The most data words one phase places on a PE altogether (its share of
/// every buffer plus scratch). A probe (`tests/pe_memory_probe.rs`) links
/// 10240 words of data beside the memcpy module alone; the rest of a PE's
/// memory is code, stack and the task tables, and the kernels' code takes
/// its share below that.
const PE_WORDS_DEFAULT: u64 = 8704;

/// The most words one array holds: a word address of 16 bits (the probe's
/// 8448-word array fails a 14-bit-plus-shift relocation).
pub const ARRAY_WORDS: u64 = 8192;
