//! How a rectangle becomes a launch.
//!
//! A lowered `Launch` gives rows and layers — an **iteration space**. A
//! dispatch needs a thread grid and a threadgroup. Something has to turn one
//! into the other, and *where that something lives* decides whether the
//! executor is a loop or a switch.
//!
//! # The rule is named, and the rule stays a function
//!
//! The obvious move is to put the geometry on the row as numbers, or as a
//! little expression grammar the row can spell in `const`. Both were tried on
//! paper and both are worse than what is here.
//!
//! Numbers cannot work: a kernel's geometry is a function of the fire — rows,
//! widths, head counts — so a row would have to state a formula, not a value.
//!
//! A grammar can express every rule in the driver today; they are all
//! `source → max → min → divide-rounding-up → multiply`. But writing
//! `Term { floor: 1, cap: 1024, div_ceil: 32, mul: 32 }` **loses the sentence
//! that says why**, and in this codebase those sentences are load-bearing:
//! `grid::qmv`'s doc records that its round-up is the difference between
//! computing every output and silently dropping the last few. A grammar buys
//! `const` and pays in explanation.
//!
//! So: the row names a [`Rule`], and the rule remains the documented function
//! it already is. The consequences are the point —
//!
//! * **The driver's dispatch is a loop.** `sig.launch.eval(dims)` for every
//!   launch, with no per-family branch and no per-kernel arm.
//! * **The match is arm-per-RULE, not arm-per-kernel.** Sixteen arms, shared by
//!   every family, every text and every backend that reuses the vocabulary. A
//!   new kernel that launches like an existing one costs zero arms — it names
//!   the rule.
//! * **Every doc comment survives**, beside the code it explains, which is
//!   where this project keeps its arguments.
//!
//! # Where this belongs
//!
//! On [`KernelSig`], as `launch = Rule::Qmv` beside `whole`, `needs` and
//! `lacks` — a launch shape is a contract fact exactly as those are. It is
//! here rather than in `kernels` because the vocabulary had to be shown to
//! cover the existing rules before the tables adopt it, and the test below is
//! that proof: **every rule the driver hand-writes today is reproduced through
//! this enum, exactly.**
//!
//! [`KernelSig`]: https://docs.rs/kernels

pub use kernels::LaunchRule as Rule;

use crate::lowering::grid::{self as shapes, Launch};

/// The fire-time quantities a launch rule may read.
///
/// Named rather than positional because a rule takes two or three of them and
/// two adjacent `u32`s that can be swapped is the defect `.wiki/driver/progress-metal.md`
/// records in `plan_heap`. Every field is a fact the lowering or the geometry
/// already states; nothing here is derived by the driver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dims {
    /// Rows the rectangle covers.
    pub rows: u32,
    /// Elements per row of the operand that sizes the launch — a projection's
    /// output width, a norm's row width, an MLP's intermediate. The launch's
    /// last widthed operand, which is its last OUTPUT.
    pub width: u32,
    /// Elements per row of the launch's first widthed operand — its first
    /// INPUT.
    ///
    /// Most rules size on the output, because most statements read narrow and
    /// write wide or the same. A statement that reads ONE packed buffer and
    /// writes several sizes on the input instead: its outputs are each a
    /// fraction of the work, so no one of them spells the grid. Both numbers
    /// are the trace's; neither is derived here.
    pub in_width: u32,
    /// Query heads.
    pub q_heads: u32,
    /// Key/value heads.
    pub kv_heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Elements one reduction spans, when that is not the whole row.
    ///
    /// A hidden-state norm reduces over its row and this equals
    /// [`width`](Self::width); a QK-norm reduces over one HEAD of a stacked
    /// projection and it does not. The kernel already knew — `rms.metal`
    /// gives threadgroup `gid` the span `gid * axis_size` — and the grid did
    /// not, so a launch covered one reduction where the statement stated
    /// thirty-two.
    ///
    /// Read off the statement's own scalar run through
    /// [`KernelSig::grid_param`](crate::lowering::kernels::KernelSig::grid_param),
    /// which is the same channel gemma-4's per-layer `rotary_dims` takes and
    /// for the same reason: one fire-wide number cannot be both of a model's
    /// two shapes.
    pub axis: u32,
    /// Channels a partial rope rotates.
    pub rotary_dims: u32,
    /// Experts the router scores.
    pub n_experts: u32,
    /// Experts each token routes to.
    pub experts_per_token: u32,
}

/// Why a rule could not produce a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ungeometric {
    /// The row states no rule, so nothing can be dispatched from it.
    Unstated,
    /// A tiled GEMM over a row count its tile does not divide.
    ///
    /// `qmm_t.metal`'s own header states the precondition: *"The driver only
    /// selects this kernel when M % BM == 0 ... so every tile is full and the
    /// `load_unsafe` path is the only one reachable; the row count lives in
    /// the grid."* There is no M argument, so a partial tile is not a shorter
    /// tile -- it is a full tile reading and writing rows that are not there.
    ///
    /// This refuses instead of substituting, and both substitutions have been
    /// tried against a real checkpoint. Handing the GEMM's symbol the MATVEC's
    /// grid made a two-token prefill entirely NaN. Rounding the row axis up
    /// made it finite and WRONG -- q_proj came out 1.258 where the matvec and
    /// MLX both say 1.320 -- which is the worse of the two, because the arena
    /// is laid out back to back and fourteen rows of overrun land on the next
    /// value.
    ///
    /// The answer is a text that states the pair with a predicate on rows, the
    /// way the DSL states every other polymorphism. A DRIVER that picks
    /// between two kernels is the thing this crate exists to remove, so it
    /// refuses and says why.
    PartialTile {
        /// Rows the fire has.
        rows: u32,
        /// Rows the tile needs a multiple of.
        tile: u32,
    },
    /// A rotation over a tensor whose row is not a whole number of heads.
    ///
    /// [`Rule::Rope`] takes its head axis from the operand it rotates rather
    /// than from the fire, because a fire has TWO head counts and the rotation
    /// is stated once per tensor. A row width the head width does not divide
    /// means the operand is not head-shaped, so there is no head axis to take
    /// and every grid this rule could return would be a guess.
    Unheaded {
        /// The operand's row width.
        width: u32,
        /// Elements per head.
        head_dim: u32,
    },
}

/// The launch a rule produces for `dims`.
///
/// A free function rather than an inherent method because [`Rule`] is
/// `kernels`' — the table STATES the rule and this backend is what knows the
/// arithmetic, which is the same split `Prepare` and `Source` already make.
///
/// # One rule, both lanes
///
/// The driver carries a *second* set of geometry functions for the batched
/// lane (`batch/dispatch_mb.rs`), and the planning documents recorded "which
/// of the two a row means" as an open question to answer before M>1 could be
/// dispatched.
///
/// Measured, the question dissolves. Every M=1 function is its M>1 function
/// **at one row** — `qmv(w)` is `qmv_mb(w, 1)`, `residual(w)` is
/// `elementwise_mb(w, 1)`, `kv_append(hd, kvh)` is the paged append at `n = 1`
/// — and where the two lanes genuinely differ they are *different kernels with
/// different names* (`affine_qmv_fast` against `affine_qmm_t`,
/// `embed_gather_4bit` against `embed_gather_mb_4bit`), each stating its own
/// row. So a row never has to say which lane it means: **the lane is
/// `dims.rows`**, and the symbol is the rest.
///
/// The tests below hold both ends of that claim — every rule reproduces its
/// M=1 function at one row, and its M>1 function above one.
///
/// # Errors
///
/// [`Ungeometric::Unstated`] when the row has not named a rule. That is drift,
/// not a runtime condition: a symbol reached dispatch whose contract does not
/// say how to launch it.
pub fn eval(rule: Rule, dims: Dims) -> Result<Launch, Ungeometric> {
    // A rectangle covers at least one row; zero would produce a grid of no
    // threads, which runs nothing and reports success.
    let rows = dims.rows.max(1);
    Ok(match rule {
        Rule::Unstated => return Err(Ungeometric::Unstated),
        Rule::Qmv => shapes::qmv_mb(dims.width, rows),
        Rule::Qmm => {
            let bm = shapes::qmm_bm(rows);
            let bn = widest_column_tile(dims.width);
            // See `Ungeometric::PartialTile`. Both substitutions were tried
            // against a real checkpoint and both were wrong; this refuses.
            if bn == 0 {
                shapes::qmv_mb(dims.width, rows)
            } else if rows.is_multiple_of(bm) {
                shapes::qmm_t(dims.width, rows, bn, bm)
            } else {
                return Err(Ungeometric::PartialTile { rows, tile: bm });
            }
        }
        Rule::Rms => shapes::rms(dims.width, dims.axis, rows),
        Rule::Rope => rope_rows(dims.rotary_dims, rope_heads(dims)?, rows),
        Rule::Elementwise => shapes::elementwise_mb(dims.width, rows),
        Rule::ElementwiseRows => embed_rows(dims.width, rows),
        Rule::SplitPacked => embed_rows(dims.in_width, rows),
        Rule::PerHead => per_head_rows(dims.head_dim, dims.kv_heads, rows),
        Rule::SdpaVector => sdpa_rows(dims.q_heads, rows),
        Rule::PerHeadElementwise => shapes::attn_gate(dims.q_heads, dims.head_dim),
        Rule::GatedRms => shapes::gated_rms(dims.kv_heads, dims.head_dim),
        Rule::RouterLane => shapes::router_topk(dims.n_experts, rows),
        // ONE threadgroup whatever the rows: see [`Rule::RouterSort`].
        Rule::RouterSort => shapes::route_sort(dims.n_experts),
        Rule::RouteRows => shapes::route_rows(dims.width, rows),
        Rule::RoutedQmv => shapes::routed_qmv(dims.width, dims.experts_per_token, rows),
    })
}

/// The widest 16/32/64 column tile that divides `out_vec`, or zero.
///
/// Wider is strictly fewer dequantizations of each weight tile —
/// `dispatch_mb::qmm_bn`'s finding, without its `min_batch` gate, which is a
/// per-family tuning number the lowering does not state.
fn widest_column_tile(out_vec: u32) -> u32 {
    [16, 32, 64]
        .into_iter()
        .rfind(|bn| out_vec.is_multiple_of(*bn))
        .unwrap_or(0)
}

// The row-aware shapes live HERE rather than in `batch/dispatch.rs`, and that
// is the point: the launch arithmetic is legitimate backend knowledge and
// stays, while the DAG builder beside it retires. Each is its M=1 sibling's
// generalisation and reduces to it at one row — proved below.

/// How many heads THIS launch's tensor has — which is not the fire's
/// `q_heads`.
///
/// `dsl::metal::rope` states the rotation TWICE, once for q and once for k,
/// and a grouped-query deployment gives the two different head counts:
/// llama-3.2 has 32 query heads and 8 key heads. This rule used to answer
/// `dims.q_heads` for both, which is wrong for k in two ways at once.
///
/// The grid covers heads the tensor does not have — 32 over an 8-head buffer,
/// so twenty-four heads' worth of rotation lands past the end of k and on
/// whatever the arena put next. And `rope_neox_mb` derives its per-row stride
/// from `grid.y`, so row *m* of k is written at `m * q_heads * head_dim` in a
/// tensor whose rows are `kv_heads * head_dim` apart.
///
/// At ONE row the stride is never applied and only the overrun happens, which
/// is why a decode agreed with MLX token for token while a prefill left every
/// key row after the first unrotated — and rope at position zero is the
/// identity, so row zero agreed too and nothing downstream disagreed loudly
/// enough to name the cause.
///
/// The launch's own operand already carries the answer: [`Dims::width`] is the
/// row width of the tensor being rotated, so the head count is
/// `width / head_dim`. For q that is `q_heads` and nothing changes; for k it
/// is `kv_heads`, which is what the kernel needed all along.
fn rope_heads(dims: Dims) -> Result<u32, Ungeometric> {
    if dims.head_dim == 0 || dims.width == 0 || !dims.width.is_multiple_of(dims.head_dim) {
        return Err(Ungeometric::Unheaded {
            width: dims.width,
            head_dim: dims.head_dim,
        });
    }
    Ok(dims.width / dims.head_dim)
}

/// Rope with the row on the third axis: `x` = frequency index, `y` = head,
/// `z` = row. In place, so it is dispatched once for Q and once for K.
fn rope_rows(rotary_dims: u32, n_heads: u32, rows: u32) -> Launch {
    let half = rotary_dims / 2;
    Launch {
        grid: [half, n_heads, rows],
        tg: [half, 1, 1],
    }
}

/// A gather whose rows are NOT contiguous, so the row gets its own axis
/// instead of stacking flat: one thread per (channel, row).
fn embed_rows(width: u32, rows: u32) -> Launch {
    Launch {
        grid: [width, rows, 1],
        tg: [256, 1, 1],
    }
}

/// The per-head scatter with the row on the third axis — the q/k/v split and
/// the KV append, paged or not.
fn per_head_rows(head_dim: u32, n_heads: u32, rows: u32) -> Launch {
    Launch {
        grid: [head_dim, n_heads, rows],
        tg: [head_dim, 1, 1],
    }
}

/// Single-pass attention with the row on the second axis: one 1024-thread
/// threadgroup per (query head, row).
fn sdpa_rows(n_q_heads: u32, rows: u32) -> Launch {
    Launch {
        grid: [n_q_heads * 1024, rows, 1],
        tg: [1024, 1, 1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims() -> Dims {
        Dims {
            rows: 7,
            width: 4096,
            in_width: 4096,
            q_heads: 16,
            kv_heads: 4,
            head_dim: 128,
            // A norm over the whole row, which is what most statements are.
            // The QK-norm case has its own fixture below.
            axis: 4096,
            rotary_dims: 64,
            n_experts: 128,
            experts_per_token: 8,
        }
    }

    /// The proof the vocabulary is right: every rule the driver hand-writes
    /// today is reachable through the enum and produces the same launch —
    /// **at one row**, which is the M=1 lane.
    ///
    /// If this holds, moving the rule onto the row is mechanical — the row
    /// names a variant and nothing else changes. If a new kernel ever needs a
    /// launch no variant produces, that is a new variant with its own
    /// documented function, not a special case in the executor.
    #[test]
    fn every_rule_reproduces_the_function_the_driver_already_uses() {
        let d = Dims { rows: 1, ..dims() };
        for (rule, expected) in [
            (Rule::Qmv, shapes::qmv(d.width)),
            (Rule::Rms, shapes::rms(d.width, d.axis, 1)),
            (Rule::Rope, shapes::rope(d.rotary_dims, d.width / d.head_dim)),
            (Rule::Elementwise, shapes::residual(d.width)),
            (Rule::ElementwiseRows, shapes::embed(d.width)),
            (Rule::PerHead, shapes::kv_append(d.head_dim, d.kv_heads)),
            (Rule::SdpaVector, shapes::sdpa(d.q_heads)),
            (
                Rule::PerHeadElementwise,
                shapes::attn_gate(d.q_heads, d.head_dim),
            ),
            (Rule::GatedRms, shapes::gated_rms(d.kv_heads, d.head_dim)),
            (Rule::RouterLane, shapes::router_topk(d.n_experts, 1)),
            (Rule::RouterSort, shapes::route_sort(d.n_experts)),
            (Rule::RouteRows, shapes::route_rows(d.width, 1)),
            (
                Rule::RoutedQmv,
                shapes::routed_qmv(d.width, d.experts_per_token, 1),
            ),
        ] {
            assert_eq!(
                eval(rule, d).expect("a stated rule evaluates"),
                expected,
                "{rule:?} does not reproduce its M=1 function"
            );
        }
    }

    /// The other end of the same claim, and the one that retires
    /// `batch/dispatch_mb.rs`'s vocabulary: **above one row a rule reproduces
    /// the BATCHED function**, so a row never has to say which lane it means.
    #[test]
    fn every_rule_reproduces_the_batched_function_above_one_row() {
        let d = dims();
        let n = d.rows;
        assert!(n > 1, "the batched lane needs more than one row");
        for (rule, expected) in [
            (Rule::Qmv, shapes::qmv_mb(d.width, n)),
            (Rule::Elementwise, shapes::elementwise_mb(d.width, n)),
            (Rule::Rms, shapes::rms_mb(d.width, 1, n)),
            // The shapes whose batched form puts the row on its own axis.
            // `mb_geometry`'s arms are the reference for each.
            (Rule::ElementwiseRows, Launch {
                grid: [d.width, n, 1],
                tg: [256, 1, 1],
            }),
            (Rule::PerHead, Launch {
                grid: [d.head_dim, d.kv_heads, n],
                tg: [d.head_dim, 1, 1],
            }),
            (Rule::Rope, Launch {
                grid: [d.rotary_dims / 2, d.width / d.head_dim, n],
                tg: [d.rotary_dims / 2, 1, 1],
            }),
            (Rule::SdpaVector, Launch {
                grid: [d.q_heads * 1024, n, 1],
                tg: [1024, 1, 1],
            }),
            // The router, which was NOT in this list and was wrong because of
            // it. `route.metal` reads its row from `tgid.y` and the rule
            // returned `grid: [w, 1, 1]`, so a mixture PREFILL routed row 0
            // only: every other row kept whatever `expert_ids` held from the
            // last layer, and the FFN then ran those rows through the wrong
            // experts. A finite, plausible, different model.
            //
            // Absent from the M>1 list is exactly how it stayed wrong -- the
            // M=1 list has it and is right there, so the rule looked covered.
            (Rule::RouterLane, Launch {
                grid: [shapes::router_lane_width(d.n_experts), n, 1],
                tg: [shapes::router_lane_width(d.n_experts), 1, 1],
            }),
            // And its twin, which must NOT move. `route_sort` reduces across
            // every (row, slot) pair through threadgroup atomics; one copy
            // per row would have each clearing and rewriting the permutation
            // the others read. The two shared `RouterLane` until the row axis
            // landed, which is why they are two rows now.
            (Rule::RouterSort, shapes::route_sort(d.n_experts)),
        ] {
            assert_eq!(
                eval(rule, d).expect("a stated rule evaluates"),
                expected,
                "{rule:?} does not reproduce its M>1 function"
            );
        }
    }

    /// **Every rule must either scale with the row count or say why not.**
    ///
    /// This is the guard the `RouterLane` defect earned. That rule dropped
    /// the row axis, so a mixture prefill routed row 0 and ran every other
    /// row on its experts — and it went unseen because the rule was simply
    /// *absent* from the batched list above. A list you have to remember to
    /// add to does not catch the thing you forgot.
    ///
    /// So this enumerates the vocabulary instead: a rule whose launch does
    /// not move between one row and many is a rule that ignores its rows,
    /// and it has to be on `ROW_INVARIANT` with a reason. Adding a variant
    /// to `LaunchRule` fails this test until someone answers the question.
    #[test]
    fn a_rule_that_ignores_its_rows_has_to_say_so() {
        /// The rules that genuinely do not move with the row count.
        const ROW_INVARIANT: &[(Rule, &str)] = &[
            (
                Rule::RouterSort,
                "ONE threadgroup reduces across every (row, slot) pair through \
                 threadgroup atomics and stripes them over its own lanes; a \
                 copy per row would clear the permutation the others read",
            ),
            (
                Rule::PerHeadElementwise,
                "per-head pointwise over the head geometry; no kernel row \
                 claims it yet, so its row behaviour is unmeasured -- see \
                 UNCLAIMED",
            ),
            (
                Rule::GatedRms,
                "GDN's gated norm over the value heads; no kernel row claims \
                 it yet, so its row behaviour is unmeasured -- see UNCLAIMED",
            ),
        ];

        let one = Dims { rows: 1, ..dims() };
        let many = Dims { rows: 8, ..dims() };
        for &rule in Rule::ALL {
            if rule == Rule::Unstated {
                continue;
            }
            let (a, b) = (eval(rule, one), eval(rule, many));
            let invariant = ROW_INVARIANT.iter().find(|(r, _)| *r == rule);
            match (a, b) {
                (Ok(a), Ok(b)) if a == b => assert!(
                    invariant.is_some(),
                    "{rule:?} launches the same rectangle for 1 row and 8. \
                     Either it drops the row axis -- which is the RouterLane \
                     defect, where every row but the first got the first \
                     row's answer -- or it is genuinely row-invariant and \
                     belongs on ROW_INVARIANT with the reason."
                ),
                (Ok(_), Ok(_)) => assert!(
                    invariant.is_none(),
                    "{rule:?} is on ROW_INVARIANT and its launch moves with \
                     the row count anyway; the reason there is stale"
                ),
                // A rule that refuses one of the two is stating a geometry
                // question, not ignoring rows. `Qmm` refuses a partial tile.
                _ => {}
            }
        }
    }

    /// The GEMM is a DIFFERENT KERNEL, not the matvec launched wider — which
    /// is what makes the batched lane a row's statement rather than a mode the
    /// driver picks.
    #[test]
    fn the_gemm_tiles_over_rows_and_refuses_a_partial_one() {
        let d = Dims {
            rows: 32,
            width: 4096,
            ..dims()
        };
        assert_eq!(
            eval(Rule::Qmm, d).expect("stated"),
            shapes::qmm_t(4096, 32, 64, shapes::qmm_bm(32)),
            "a divisible shape tiles"
        );
        // A row count no rung divides is still the GEMM's: the last tile is
        // PARTIAL and the kernel bounds-checks within it.
        //
        // This expected `shapes::qmv_mb` — the matvec grid — on the theory
        // that it "computes all of it, slower". It does not: `affine_qmm_t`
        // reads its tile FROM the grid, so a matvec grid points it at a tiling
        // that is not there, and a two-token prefill against a real checkpoint
        // came back entirely NaN. `QMM_BMS` starts at sixteen, so EVERY
        // prefill shorter than sixteen rows took that path.
        let ragged = Dims { rows: 3, ..d };
        assert_eq!(
            eval(Rule::Qmm, ragged),
            Err(Ungeometric::PartialTile { rows: 3, tile: 16 }),
            "an indivisible shape refuses rather than substituting"
        );
    }

    #[test]
    fn a_rectangle_of_no_rows_still_launches_one() {
        // Zero rows would multiply a grid to nothing, and a dispatch of no
        // threads runs nothing and reports success.
        let none = Dims { rows: 0, ..dims() };
        let one = Dims { rows: 1, ..dims() };
        assert_eq!(eval(Rule::Elementwise, none), eval(Rule::Elementwise, one));
    }

    #[test]
    fn the_shapes_that_share_a_function_are_one_variant_not_three() {
        // `residual`, `embed` and `silu_mul` are the same 256-wide pointwise
        // launch; the C++ and the driver spell it three times. A kernel that
        // launches like an existing one should cost zero arms.
        let d = Dims { rows: 1, ..dims() };
        let ew = eval(Rule::Elementwise, d).expect("stated");
        assert_eq!(ew, shapes::residual(d.width));
        assert_eq!(ew, shapes::embed(d.width));
        assert_eq!(ew, shapes::silu_mul(d.width));

        // The same for the per-head pair: the q/k/v split and the KV append
        // launch identically, over whichever head count they address.
        assert_eq!(
            eval(Rule::PerHead, d).expect("stated"),
            shapes::q_split(d.head_dim, d.kv_heads),
            "one rule, read with the head count the operand names"
        );
    }

    #[test]
    fn an_unstated_rule_refuses_rather_than_launching_something_plausible() {
        // The default. A symbol whose contract does not say how to launch it
        // has reached dispatch by drift, and a guessed grid is a kernel that
        // runs over the wrong extent — which the hardware does not report.
        assert_eq!(
            eval(Rule::default(), dims()),
            Err(Ungeometric::Unstated),
            "unstated must not fall back to anything"
        );
    }

    #[test]
    fn a_rule_reads_only_the_dims_it_names() {
        // Changing a dimension a rule does not use must not move its launch.
        // This is what makes `Dims` safe to grow: a new field cannot silently
        // change an existing rule's geometry.
        let d = dims();
        let wider = Dims {
            n_experts: 999,
            experts_per_token: 3,
            ..d
        };
        assert_eq!(eval(Rule::Rms, d), eval(Rule::Rms, wider));
        assert_eq!(eval(Rule::SdpaVector, d), eval(Rule::SdpaVector, wider));
        assert_ne!(
            eval(Rule::RouterLane, d),
            eval(Rule::RouterLane, wider),
            "and one that DOES name it must move"
        );
    }

    /// **Q and K rotate over their OWN head counts, not the fire's.**
    ///
    /// A grouped-query fire states the rotation twice with the same `Dims`
    /// except for `width` — q's row is `q_heads * head_dim` and k's is
    /// `kv_heads * head_dim`. If the rule reads `q_heads` for both, k's launch
    /// covers heads k does not have and, on the batched kernel, strides its
    /// rows by q's width. That is the shape of a prefill that leaves every key
    /// row after the first unrotated.
    #[test]
    fn the_rotation_takes_its_head_axis_from_the_tensor_it_turns() {
        let d = Dims {
            rows: 4,
            head_dim: 128,
            rotary_dims: 128,
            q_heads: 32,
            kv_heads: 8,
            ..dims()
        };
        let q = eval(Rule::Rope, Dims {
            width: d.q_heads * d.head_dim,
            ..d
        })
        .expect("q rotates");
        let k = eval(Rule::Rope, Dims {
            width: d.kv_heads * d.head_dim,
            ..d
        })
        .expect("k rotates");

        assert_eq!(q.grid, [64, 32, 4], "q covers its own thirty-two heads");
        assert_eq!(k.grid, [64, 8, 4], "k covers its own eight, not q's");
        assert_ne!(
            q.grid[1], k.grid[1],
            "a grouped-query fire has two head counts and one of them is not \
             the answer for both"
        );
    }

    /// A tensor whose row is not a whole number of heads has no head axis to
    /// take, so the rule refuses rather than returning a guessed grid.
    #[test]
    fn a_rotation_over_an_unheaded_row_refuses() {
        let d = Dims {
            width: 100,
            head_dim: 128,
            ..dims()
        };
        assert_eq!(
            eval(Rule::Rope, d),
            Err(Ungeometric::Unheaded {
                width: 100,
                head_dim: 128
            })
        );
    }

    /// Two rules sit on `ROW_INVARIANT` for a reason that is not "this rule
    /// does not move with rows" but "nothing dispatches it, so nobody has
    /// looked". That is a claim about the KERNEL TABLE, written in prose, in a
    /// list whose other entries are permanent — and prose does not fail when
    /// it stops being true.
    ///
    /// This is not hypothetical. `Rule::Rope` was measured once, against a
    /// decode, and its grid took the head axis from the fire's `q_heads`. Rows
    /// were then stated on it — five of them — and nothing re-asked whether the
    /// launch arithmetic still fitted, because the only record that it had been
    /// measured at one row was a comment. A grouped-query prefill rotated k
    /// over q's head count and strided its rows by q's width, and every key row
    /// after the first came out unrotated.
    ///
    /// So the exemption expires by itself: the moment a row claims one of these
    /// rules, the rule is dispatchable, its row behaviour is measurable, and
    /// this fails until somebody measures it.
    #[test]
    fn an_exemption_for_being_unreachable_expires_when_a_row_claims_the_rule() {
        /// Exempt only because no kernel names them. Remove a rule from here
        /// when a row claims it, and give `ROW_INVARIANT` the real reason or
        /// take it off that list.
        const UNCLAIMED: &[Rule] = &[Rule::PerHeadElementwise, Rule::GatedRms];

        for &rule in UNCLAIMED {
            let claimed: Vec<&str> = kernels_metal::KERNELS
                .iter()
                .filter(|k| k.launch == rule)
                .map(|k| k.name)
                .collect();
            assert!(
                claimed.is_empty(),
                "{rule:?} is exempted from the row-axis check only because no \
                 kernel row claimed it, and {claimed:?} now does. Its launch \
                 ignores `dims.rows`, so every row of a prefill would get row \
                 zero's rectangle. Measure what the kernel does with M>1 before \
                 this dispatches: either the rule grows a row axis, or it earns \
                 a real place on ROW_INVARIANT."
            );
        }
    }

    /// **A QK-norm reduces over a head, and the grid has to cover all of
    /// them.**
    ///
    /// `rms.metal` gives threadgroup `gid` the span `gid * axis_size`, so a
    /// launch over a 8192-wide Q whose norm spans 256 channels needs 32
    /// threadgroups per row. It used to get one per row, sized on the width:
    /// head 0 was normalized and the other 31 were left as the projection
    /// wrote them.
    ///
    /// Nothing reported it and nothing could. The tensor is fully written —
    /// just not fully normalized — so no NaN, no zero, no unwritten slot.
    /// It surfaced only where the missing threadgroups happen to be whole
    /// tokens: a prefill whose second row came back all zeros.
    ///
    /// Falsified by the shape of the assertion: sizing on the width instead
    /// of the axis gives `[1024 * rows, 1, 1]`, which is what this rejects.
    #[test]
    fn a_qk_norm_covers_every_head_and_a_row_norm_is_the_case_where_it_is_one() {
        let q_width = 8192;
        let head_dim = 256;
        let rows = 3;
        let d = Dims {
            rows,
            width: q_width,
            axis: head_dim,
            ..dims()
        };
        let launch = eval(Rule::Rms, d).expect("a norm has a grid");
        let threads = head_dim / 4;
        assert_eq!(launch.tg, [threads, 1, 1], "a threadgroup spans one HEAD");
        assert_eq!(
            launch.grid,
            [threads * (q_width / head_dim) * rows, 1, 1],
            "one threadgroup per head per row, not one per row"
        );
        assert_eq!(
            launch.grid[0] / launch.tg[0],
            (q_width / head_dim) * rows,
            "32 heads x 3 rows of reductions"
        );

        // THE SAME FUNCTION at `axis == width`, which is every hidden-state
        // norm in the tree and the reason this change moves nothing else.
        let whole = Dims {
            rows,
            width: 4096,
            axis: 4096,
            ..dims()
        };
        assert_eq!(
            eval(Rule::Rms, whole).expect("a norm has a grid"),
            Launch {
                grid: [1024 * rows, 1, 1],
                tg: [1024, 1, 1],
            },
            "a norm spanning its row is the one-head case"
        );
    }
}

#[cfg(test)]
mod stated_head_width {
    use super::*;

    /// **A rope counts heads by the width the kernel is told, not the fire's.**
    ///
    /// `rope_neox_mb` finds a token's row by multiplying the two back
    /// together — `row_base = m * n_head * head_dim` — so the grid's head
    /// count and the kernel's head width are one quantity stated twice. When
    /// they disagree the product is not the row.
    ///
    /// gemma-4 is the deployment that disagrees: 512-wide heads on its
    /// full-attention layers, 256-wide on its sliding ones. Against the
    /// fire's 256 a 16 384-wide Q counted 64 heads, the kernel multiplied 64
    /// by its own 512 and stepped 32 768 elements per row — twice the row.
    /// Row 1 was written over row 2, and every row past the second landed off
    /// the end of the tensor entirely. The rotation applied to almost
    /// nothing, in every fire, decode included.
    #[test]
    fn a_rope_steps_exactly_one_row_per_token() {
        let width = 16_384;
        for (head_dim, heads) in [(512, 32), (256, 64)] {
            let dims = Dims {
                rows: 3,
                width,
                head_dim,
                rotary_dims: 128,
                in_width: width,
                q_heads: 32,
                kv_heads: 4,
                axis: width,
                n_experts: 0,
                experts_per_token: 0,
            };
            let launch = eval(Rule::Rope, dims).expect("a rope grids");
            assert_eq!(
                launch.grid[1], heads,
                "a {width}-wide tensor of {head_dim}-wide heads has {heads} of them"
            );
            // The property that matters, stated as the kernel computes it.
            assert_eq!(
                launch.grid[1] * head_dim,
                width,
                "the head count times the head width is one row, or the \
                 kernel's `m * n_head * head_dim` is not the mth row"
            );
        }
    }
}
