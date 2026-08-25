//! The mixture's six points, on the GPU, for the first time.
//!
//! `moe/route.metal` and `moe/select.metal` answer `moe.topk_softmax`,
//! `moe.topk_sigmoid`, `moe.topk_sqrt_softplus`, `moe.matmul_select`,
//! `moe.weighted_sum` and `moe.sigmoid_gate_add` -- six of the fifty-one
//! points this plane claims. The two ranked routers and the dense routed
//! projection are new; none of the six had ever produced a number.
//!
//! # A router is an INDEX, and an index has no tolerance
//!
//! Everything else this sweep measures answers a number, and a number gets a
//! bound. A router answers `expert_ids` first, and picking the wrong expert
//! is not a small error -- it is a different matrix. So the ids are compared
//! with `assert_eq!` and the weights with a bound, and the fixture's logits
//! are drawn DISTINCT within a row so that the ranking has one right answer
//! rather than a tie the two implementations may break differently.
//!
//! That distinctness is checked rather than assumed: [`Router::of`] asserts
//! it, because a fixture that quietly grew a tie would turn this file into a
//! test of which lane wins a `simd_min`.
//!
//! # The fan-out that exceeds the expert count
//!
//! `route.metal` states the contract in its own header: "A ROW WHOSE FAN-OUT
//! EXCEEDS ITS EXPERT COUNT parks its spare slots on expert 0 with weight
//! zero. Repeating the last winner would double-count it in the fold, and
//! leaving the slot unwritten hands the combine whatever the arena held -- an
//! id that indexes a bank out of bounds." Nothing else in the tree fires that
//! shape, so the sigmoid router is fired at `k = 8` against `n = 5` below and
//! the parked slots are checked by value.
//!
//! # The bias shifts the RANKING and not the weight
//!
//! `router_topk_sqrt_softplus` adds DeepSeek's correction bias to the value it
//! compares and publishes a weight recomputed from the logit alone. The
//! shader's header says a body that published the biased weight would
//! "reweight every expert by its own bias, and the model still produces text",
//! which is exactly the shape of defect that survives every check except a
//! comparison. It is one of the mutations, and the fixture's bias is large
//! enough that it changes which experts win -- a bias too small to move the
//! ranking would leave the two spellings agreeing.
//!
//! # Two activation shapes, because `matmul_select` takes two
//!
//! `moe.matmul_select`'s activation is either one row per TOKEN -- a gate or
//! up projection, where every slot of a token reads the same row, slot stride
//! zero -- or one row per ROUTE, which is what a down projection reads out of
//! the stack the activation before it wrote. `kernels_metal::moe::selected`
//! picks between them on `x.rows`, and the shader header notes that reading
//! slot 0 for every expert "is not a crash -- it is k copies of the first
//! expert's activation, which survives all the way to a plausible wrong
//! token". Both shapes are fired.

#![cfg(target_vendor = "apple")]

mod plane;

use driver_metal::skip::skipped;
use plane::{Arg, Rig};

const FILE_ROUTE: &str = "moe/route.metal";
const FILE_SELECT: &str = "moe/select.metal";

/// Forty experts against a threadgroup of sixty-four: one whole simdgroup,
/// one that is three quarters live, and twenty-four lanes that hold no
/// expert at all.
const E: usize = 40;
const K: usize = 6;
const ROWS: usize = 5;

/// The routed projection's rectangle.
const EXPERTS: usize = 4;
const IN_WIDTH: usize = 20;
const OUT_WIDTH: usize = 5;
const TOKENS: usize = 3;
const FAN: usize = 2;
const ROUTES: usize = TOKENS * FAN;

/// DeepSeek's renormalisation gain, and a value that is not one so that
/// dropping it is visible.
const SCALING: f32 = 2.5;

const POISON: f32 = -99.0;

/// Relative to the widest weight in the plane.
///
/// A router's weights land in f32 rather than bf16 -- `Moe::topk_softmax`
/// declares them `Out<Tensor<f32>>` because a router weight is a probability
/// -- so this is the one place in the sweep where the difference between a
/// fast transcendental and a correctly rounded one is visible at all.
/// `router_topk` spells its softmax with `fast::exp` and the two ranked
/// routers use `metal::exp`, `metal::log` and `metal::sqrt`.
///
/// MEASURED, and the measurement is the interesting part: `2^-22` is one and
/// a half f32 ulps, and the SOFTMAX -- the one that uses the fast
/// exponential -- sits at 0.29 of it while the sigmoid router sits at 0.87.
/// Metal's `fast::exp` is not measurably worse than its precise one here,
/// which is worth knowing and is not what the name suggests.
///
/// [`plane::tolerance_holds`] keeps this honest: the five comparisons in this
/// file take between 0.23 and 0.87 of it.
const WEIGHT_BOUND: f32 = 1.0 / 4_194_304.0;

/// One bf16 step of the dot product's own magnitude, the same bound
/// `device_dense` takes over the same shape of contraction.
const SCALE_BOUND: f32 = 1.0 / 256.0;

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_softmax_router_normalises_over_the_experts_it_picked() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `moe.topk_softmax` was not fired");
        return;
    };
    let fx = Router::of(E);
    let root = plane::kernels_dir();
    let (ids, weights) = fx.softmax(&rig, root.as_path());
    let (want_ids, want_weights) = fx.softmax_model();

    assert_eq!(
        ids, want_ids,
        "`moe.topk_softmax` picks the k largest logits"
    );
    fx.weights_agree(&weights, &want_weights, "moe.topk_softmax");

    // The knock-out, aimed one lane over. The winner then survives every
    // round and the row routes k copies of one expert, which is a routing
    // that produces text.
    fx.softmax_bites(
        &rig,
        &want_ids,
        &want_weights,
        "if (lid == winner_of_round) v = NEG_INF;\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n  }\n\n  if (lid == 0) {",
        "if (lid == winner_of_round + 1u) v = NEG_INF;\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n  }\n\n  if (lid == 0) {",
    );

    // The denominator, taken over the logits rather than over their
    // exponentials. Still a normalisation, still sums to one, and not a
    // softmax.
    fx.softmax_bites(
        &rig,
        &want_ids,
        &want_weights,
        "for (uint r = 0; r < k; ++r) sum += fast::exp(chosen[r] - mx);",
        "for (uint r = 0; r < k; ++r) sum += chosen[r];",
    );

    // The row's stride into its own results. `k` is the fan-out and the row
    // pitch of both output planes, and dropping it overlaps every row with
    // the next.
    fx.softmax_bites(
        &rig,
        &want_ids,
        &want_weights,
        "expert_ids += size_t(row) * size_t(k);",
        "expert_ids += size_t(row);",
    );

    // The logits' own pitch. `logits_pitch` of zero means the pitch IS the
    // expert count, and `moe.topk_softmax` binds the count rather than zero
    // -- so a body that ignored the argument would read every row from the
    // same forty logits shifted by one.
    fx.softmax_bites(
        &rig,
        &want_ids,
        &want_weights,
        "logits += size_t(row) * size_t(logits_pitch != 0u ? logits_pitch : n);",
        "logits += size_t(row);",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_sigmoid_router_renormalises_only_when_the_statement_asks() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `moe.topk_sigmoid` was not fired");
        return;
    };
    let fx = Router::of(E);
    let root = plane::kernels_dir();

    for renormalize in [false, true] {
        let (ids, weights) = fx.sigmoid(&rig, root.as_path(), K, renormalize);
        let (want_ids, want_weights) = fx.sigmoid_model(K, renormalize);
        assert_eq!(
            ids, want_ids,
            "`moe.topk_sigmoid` picks the k largest gates, renormalize={renormalize}"
        );
        fx.weights_agree(
            &weights,
            &want_weights,
            &format!("moe.topk_sigmoid, renormalize={renormalize}"),
        );
    }

    // The renormalisation, inverted. `scaling / sum` is a row whose weights
    // sum to `scaling`; `scaling * sum` is a row that grows with its own
    // confidence.
    fx.sigmoid_bites(
        &rig,
        true,
        "? scaling / sum : scaling;",
        "? scaling * sum : scaling;",
    );

    // The ranking, inverted: the row then routes its k WORST experts, which
    // is a mixture that still mixes.
    fx.sigmoid_bites(
        &rig,
        false,
        "float v = lid < n ? router_sigmoid(float(logits[lid])) : NEG_INF;",
        "float v = lid < n ? -router_sigmoid(float(logits[lid])) : NEG_INF;",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn a_fan_out_wider_than_the_bank_parks_its_spare_slots_on_expert_zero() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: the router's fan-out contract was not fired");
        return;
    };
    // Five experts and eight slots: three of the eight have no expert to
    // take, and the shader's own header says what must happen to them.
    const NARROW: usize = 5;
    const WIDE: usize = 8;
    let fx = Router::of(NARROW);
    let root = plane::kernels_dir();

    let (ids, weights) = fx.sigmoid(&rig, root.as_path(), WIDE, true);
    let (want_ids, want_weights) = fx.sigmoid_model(WIDE, true);
    assert_eq!(ids, want_ids, "the spare slots park on expert zero");
    fx.weights_agree(&weights, &want_weights, "moe.topk_sigmoid at k > n");

    for r in 0..ROWS {
        for slot in NARROW..WIDE {
            assert_eq!(
                ids[r * WIDE + slot],
                0,
                "slot {slot} of row {r} has no expert and must park on zero"
            );
            assert_eq!(
                weights[r * WIDE + slot],
                0.0,
                "a parked slot weighs nothing, or the fold double-counts it"
            );
        }
    }
    plane::measured(
        "moe.topk_sigmoid at k > n",
        "three spare slots a row, parked on expert 0 at weight 0",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_sqrt_softplus_router_ranks_with_the_bias_and_publishes_without_it() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `moe.topk_sqrt_softplus` was not fired");
        return;
    };
    let fx = Router::of(E);
    let root = plane::kernels_dir();
    let (ids, weights) = fx.sqrt_softplus(&rig, root.as_path(), true);
    let (want_ids, want_weights) = fx.sqrt_softplus_model(true);

    assert_eq!(
        ids, want_ids,
        "`moe.topk_sqrt_softplus` ranks by the biased value"
    );
    fx.weights_agree(&weights, &want_weights, "moe.topk_sqrt_softplus");

    // THE DEFECT THE HEADER NAMES. Publishing the ranked value instead of the
    // logit's own reweights every expert by its own bias, and the ids do not
    // move at all -- so nothing but a weight comparison can see it.
    fx.softplus_bites(
        &rig,
        "const float w = sqrt_softplus(float(logits[uint(expert_ids[r])]));",
        "const float w = sqrt_softplus(float(logits[uint(expert_ids[r])])) + correction[uint(expert_ids[r])];",
    );

    // The square root, dropped. `log(1 + exp(x))` is monotone in the same
    // direction, so the RANKING is untouched and only the weights move.
    fx.softplus_bites(
        &rig,
        "return metal::sqrt(max(sp, 0.0f));",
        "return max(sp, 0.0f);",
    );

    // The bias, dropped from the ranking. This is the other half of the
    // header's claim, and it moves the ids rather than the weights.
    fx.softplus_bites(
        &rig,
        "float v = lid < n ? sqrt_softplus(float(logits[lid])) + correction[lid] : NEG_INF;",
        "float v = lid < n ? sqrt_softplus(float(logits[lid])) : NEG_INF;",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_routed_projection_reads_the_expert_the_route_names() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `moe.matmul_select` was not fired");
        return;
    };
    let fx = Select::of();
    let root = plane::kernels_dir();

    // The gate/up shape: one activation row per TOKEN, slot stride zero.
    let got = fx.fire(&rig, root.as_path(), Shape::PerToken);
    fx.agrees(
        &got,
        Shape::PerToken,
        "moe.matmul_select, one row per token",
    );

    // The down shape: one activation row per ROUTE, so the slot strides.
    let got = fx.fire(&rig, root.as_path(), Shape::PerRoute);
    fx.agrees(
        &got,
        Shape::PerRoute,
        "moe.matmul_select, one row per route",
    );

    // The expert, folded into the element offset. `bank` is `[E, N, K]` and
    // this is the classic way to read expert 0's weights for every expert.
    fx.bites(
        &rig,
        Shape::PerToken,
        "bank + (size_t(uint(e)) * size_t(out_width) + size_t(out_row)) * size_t(in_width)",
        "bank + size_t(out_row) * size_t(in_width)",
    );

    // The slot, dropped from the activation's address. Zero on the gate arm
    // and `I` on the down arm, so this is a no-op for the first shape and k
    // copies of the first slot's activation for the second.
    fx.bites(
        &rig,
        Shape::PerRoute,
        "x + size_t(route / k) * size_t(x_row_stride) + size_t(route % k) * size_t(x_slot_stride);",
        "x + size_t(route / k) * size_t(x_row_stride);",
    );

    // The negative id's zero, dropped. An untouched row is whatever the arena
    // held, which in bf16 can be inf -- and the fold that follows multiplies
    // it by a weight and adds it.
    fx.bites(
        &rig,
        Shape::PerToken,
        "if (lane == 0) y[at] = bfloat(0);\n    return;",
        "return;",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_expert_fold_weights_each_slot_where_the_router_left_it() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `moe.weighted_sum` was not fired");
        return;
    };
    let fx = Fold::of();
    let root = plane::kernels_dir();
    let got = fx.combine(&rig, root.as_path());
    fx.agrees(&got, &fx.combine_model(), "moe.weighted_sum");

    // The weight, dropped: the fold becomes a sum, which is what a mixture
    // looks like when every expert is equally right.
    fx.combine_bites(
        &rig,
        "acc += expert_weights[at] * float(y[at * size_t(width) + size_t(c)]);",
        "acc += float(y[at * size_t(width) + size_t(c)]);",
    );

    // The token's own base into the routed stack. `y` is `[rows * k, width]`
    // in (token, slot) order, so `row * k` is where a token's slots begin and
    // `row` is where some other token's do.
    fx.combine_bites(
        &rig,
        "const size_t base = size_t(row) * size_t(k);",
        "const size_t base = size_t(row);",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_shared_expert_gate_is_one_sigmoid_per_token() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `moe.sigmoid_gate_add` was not fired");
        return;
    };
    let fx = Fold::of();
    let root = plane::kernels_dir();
    let got = fx.gate(&rig, root.as_path());
    fx.agrees(&got, &fx.gate_model(), "moe.sigmoid_gate_add");

    // The sigmoid, dropped. `route.metal`'s header: "The sigmoid is computed
    // in float from a bf16 logit. That matters at the tails" -- and a raw
    // logit in its place is a gate that is neither bounded nor positive.
    fx.gate_bites(
        &rig,
        "const float g = 1.0f / (1.0f + metal::exp(-float(gate[row])));\n  const uint at = row * width + c;",
        "const float g = float(gate[row]);\n  const uint at = row * width + c;",
    );

    // The gate, applied to the routed half as well. It gates the SHARED
    // expert; scaling the mixture's own output by it is a line that reads
    // like an algebraic simplification and is not one.
    fx.gate_bites(
        &rig,
        "out[at] = static_cast<bfloat>(float(routed[at]) + g * float(shared[at]));\n}",
        "out[at] = static_cast<bfloat>(g * (float(routed[at]) + float(shared[at])));\n}",
    );

    // One gate per TOKEN, broadcast over the row. Indexing it by the column
    // instead reads `width` gate values where there is one, which is what
    // `attn_gate` would have done and why this is its own kernel.
    fx.gate_bites(
        &rig,
        "const float g = 1.0f / (1.0f + metal::exp(-float(gate[row])));\n  const uint at = row * width + c;",
        "const float g = 1.0f / (1.0f + metal::exp(-float(gate[c])));\n  const uint at = row * width + c;",
    );
}

// ── the routers ─────────────────────────────────────────────────────────────

/// One rectangle of router logits, and the correction bias beside it.
struct Router {
    logits: Vec<f32>,
    correction: Vec<f32>,
    experts: usize,
}

impl Router {
    /// Logits DISTINCT within a row, so the ranking has one right answer.
    fn of(experts: usize) -> Self {
        let logits: Vec<f32> = (0..ROWS * experts)
            .map(|i| {
                let (r, e) = (i / experts, i % experts);
                plane::narrowed(((r * 13 + e * 29) % 97) as f32 * 0.05 - 2.4)
            })
            .collect();
        for r in 0..ROWS {
            let row = &logits[r * experts..(r + 1) * experts];
            for i in 0..experts {
                for j in i + 1..experts {
                    assert_ne!(
                        row[i], row[j],
                        "row {r} ties experts {i} and {j}, so its ranking has \
                         no single right answer and this file would be \
                         measuring a tie-break"
                    );
                }
            }
        }
        Self {
            // Large enough to move the ranking: a bias that only nudged the
            // weights would leave the biased and unbiased spellings agreeing
            // about which experts win.
            correction: (0..experts)
                .map(|e| ((e * 17) % 23) as f32 * 0.06 - 0.6)
                .collect(),
            logits,
            experts,
        }
    }

    /// The k largest, lowest index first on a tie.
    fn ranked(&self, by: impl Fn(usize, usize) -> f32, fan: usize) -> Vec<usize> {
        let picks = fan.min(self.experts);
        let mut ids = Vec::with_capacity(ROWS * fan);
        for r in 0..ROWS {
            let mut order: Vec<usize> = (0..self.experts).collect();
            order.sort_by(|a, b| {
                by(r, *b)
                    .partial_cmp(&by(r, *a))
                    .expect("no NaN in a router's logits")
                    .then(a.cmp(b))
            });
            ids.extend_from_slice(&order[..picks]);
            ids.extend(std::iter::repeat_n(0usize, fan - picks));
        }
        ids
    }

    fn softmax_model(&self) -> (Vec<i32>, Vec<f32>) {
        let ids = self.ranked(|r, e| self.logits[r * self.experts + e], K);
        let mut weights = vec![0.0; ROWS * K];
        for r in 0..ROWS {
            let chosen: Vec<f32> = (0..K)
                .map(|s| self.logits[r * self.experts + ids[r * K + s]])
                .collect();
            let mx = chosen.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
            let sum: f32 = chosen.iter().map(|v| plane::exp32(v - mx)).sum();
            for s in 0..K {
                weights[r * K + s] = plane::exp32(chosen[s] - mx) / sum;
            }
        }
        (ids.iter().map(|e| *e as i32).collect(), weights)
    }

    fn sigmoid_model(&self, fan: usize, renormalize: bool) -> (Vec<i32>, Vec<f32>) {
        let gate =
            |r: usize, e: usize| 1.0 / (1.0 + plane::exp32(-self.logits[r * self.experts + e]));
        let ids = self.ranked(gate, fan);
        let picks = fan.min(self.experts);
        let mut weights = vec![0.0; ROWS * fan];
        for r in 0..ROWS {
            let mut sum = 0.0f32;
            for s in 0..picks {
                let w = gate(r, ids[r * fan + s]);
                weights[r * fan + s] = w;
                sum += w;
            }
            let scale = if renormalize && sum > 0.0 {
                SCALING / sum
            } else {
                SCALING
            };
            for s in 0..fan {
                weights[r * fan + s] *= scale;
            }
        }
        (ids.iter().map(|e| *e as i32).collect(), weights)
    }

    fn sqrt_softplus_model(&self, renormalize: bool) -> (Vec<i32>, Vec<f32>) {
        let ids = self.ranked(
            |r, e| sqrt_softplus(self.logits[r * self.experts + e]) + self.correction[e],
            K,
        );
        let mut weights = vec![0.0; ROWS * K];
        for r in 0..ROWS {
            let mut sum = 0.0f32;
            for s in 0..K {
                // WITHOUT THE BIAS.
                let w = sqrt_softplus(self.logits[r * self.experts + ids[r * K + s]]);
                weights[r * K + s] = w;
                sum += w;
            }
            let scale = if renormalize && sum > 0.0 {
                SCALING / sum
            } else {
                SCALING
            };
            for s in 0..K {
                weights[r * K + s] *= scale;
            }
        }
        (ids.iter().map(|e| *e as i32).collect(), weights)
    }

    fn softmax(&self, rig: &Rig, root: &std::path::Path) -> (Vec<i32>, Vec<f32>) {
        let logits = plane::alloc_bf16(&rig.context, &self.logits, "logits");
        let ids = plane::alloc_i32(&rig.context, &[-7; ROWS * K], "expert_ids");
        let weights = plane::alloc_f32(&rig.context, &[POISON; ROWS * K], "expert_weights");
        // Slot 3 is `per_expert_scale`, which the unscaled instantiation
        // never dereferences -- the claim body binds an absent buffer there
        // and this binds a real one, which is the same statement.
        let scale = plane::alloc_bf16(&rig.context, &[1.0], "per_expert_scale");
        plane::fire(
            rig,
            root,
            FILE_ROUTE,
            "router_topk_f32w_bfloat16",
            [lanes(self.experts), ROWS as u32, 1],
            [lanes(self.experts), 1, 1],
            &[
                Arg::Buf(&logits),
                Arg::Buf(&ids),
                Arg::Buf(&weights),
                Arg::Buf(&scale),
                Arg::U32(self.experts as u32),
                Arg::U32(K as u32),
                Arg::U32(0),
                Arg::U32(self.experts as u32),
            ],
        );
        (
            plane::read_i32(&ids, ROWS * K),
            plane::read_f32(&weights, ROWS * K),
        )
    }

    fn sigmoid(
        &self,
        rig: &Rig,
        root: &std::path::Path,
        fan: usize,
        renormalize: bool,
    ) -> (Vec<i32>, Vec<f32>) {
        let experts = self.experts;
        let logits = plane::alloc_bf16(&rig.context, &self.logits, "logits");
        let ids = plane::alloc_i32(&rig.context, &vec![-7; ROWS * fan], "expert_ids");
        let weights = plane::alloc_f32(&rig.context, &vec![POISON; ROWS * fan], "expert_weights");
        plane::fire(
            rig,
            root,
            FILE_ROUTE,
            "router_topk_sigmoid",
            [lanes(experts), ROWS as u32, 1],
            [lanes(experts), 1, 1],
            &[
                Arg::Buf(&logits),
                Arg::Buf(&ids),
                Arg::Buf(&weights),
                Arg::U32(experts as u32),
                Arg::U32(fan as u32),
                Arg::U32(u32::from(renormalize)),
                Arg::F32(SCALING),
            ],
        );
        (
            plane::read_i32(&ids, ROWS * fan),
            plane::read_f32(&weights, ROWS * fan),
        )
    }

    fn sqrt_softplus(
        &self,
        rig: &Rig,
        root: &std::path::Path,
        renormalize: bool,
    ) -> (Vec<i32>, Vec<f32>) {
        let logits = plane::alloc_bf16(&rig.context, &self.logits, "logits");
        let correction = plane::alloc_f32(&rig.context, &self.correction, "correction");
        let ids = plane::alloc_i32(&rig.context, &[-7; ROWS * K], "expert_ids");
        let weights = plane::alloc_f32(&rig.context, &[POISON; ROWS * K], "expert_weights");
        plane::fire(
            rig,
            root,
            FILE_ROUTE,
            "router_topk_sqrt_softplus",
            [lanes(self.experts), ROWS as u32, 1],
            [lanes(self.experts), 1, 1],
            &[
                Arg::Buf(&logits),
                Arg::Buf(&correction),
                Arg::Buf(&ids),
                Arg::Buf(&weights),
                Arg::U32(self.experts as u32),
                Arg::U32(K as u32),
                Arg::U32(u32::from(renormalize)),
                Arg::F32(SCALING),
            ],
        );
        (
            plane::read_i32(&ids, ROWS * K),
            plane::read_f32(&weights, ROWS * K),
        )
    }

    fn weights_agree(&self, got: &[f32], want: &[f32], what: &str) {
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = plane::worst(got, want, scale);
        assert!(
            worst <= WEIGHT_BOUND,
            "{what}: weight {} is {worst} of the widest weight away from the \
             model, past the {WEIGHT_BOUND} this device was measured at",
            plane::worst_at(got, want, scale)
        );
        plane::tolerance_holds(worst, WEIGHT_BOUND, what);
        plane::measured(
            what,
            &format!("worst {worst} against the weight bound {WEIGHT_BOUND}"),
        );
    }

    fn softmax_bites(
        &self,
        rig: &Rig,
        want_ids: &[i32],
        want_weights: &[f32],
        from: &str,
        to: &str,
    ) {
        let root = plane::mutant(FILE_ROUTE, from, to);
        let (ids, weights) = self.softmax(rig, root.path());
        let scale = want_weights.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = plane::worst(&weights, want_weights, scale);
        assert!(
            ids != want_ids || worst > WEIGHT_BOUND,
            "replacing `{from}` with `{to}` left every id and every weight \
             where they were, so the comparison above would not have caught it"
        );
        plane::measured(
            "router_topk_f32w_bfloat16",
            &format!(
                "`{from}` -> `{to}`: {} ids move, worst weight {worst}",
                ids.iter().zip(want_ids).filter(|(a, b)| a != b).count()
            ),
        );
    }

    fn sigmoid_bites(&self, rig: &Rig, renormalize: bool, from: &str, to: &str) {
        let root = plane::mutant(FILE_ROUTE, from, to);
        let (ids, weights) = self.sigmoid(rig, root.path(), K, renormalize);
        let (want_ids, want_weights) = self.sigmoid_model(K, renormalize);
        let scale = want_weights.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = plane::worst(&weights, &want_weights, scale);
        assert!(
            ids != want_ids || worst > WEIGHT_BOUND,
            "replacing `{from}` with `{to}` left every id and every weight \
             where they were, so the comparison above would not have caught it"
        );
        plane::measured(
            "router_topk_sigmoid",
            &format!(
                "`{from}` -> `{to}`: {} ids move, worst weight {worst}",
                ids.iter().zip(&want_ids).filter(|(a, b)| a != b).count()
            ),
        );
    }

    fn softplus_bites(&self, rig: &Rig, from: &str, to: &str) {
        let root = plane::mutant(FILE_ROUTE, from, to);
        let (ids, weights) = self.sqrt_softplus(rig, root.path(), true);
        let (want_ids, want_weights) = self.sqrt_softplus_model(true);
        let scale = want_weights.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = plane::worst(&weights, &want_weights, scale);
        assert!(
            ids != want_ids || worst > WEIGHT_BOUND,
            "replacing `{from}` with `{to}` left every id and every weight \
             where they were, so the comparison above would not have caught it"
        );
        plane::measured(
            "router_topk_sqrt_softplus",
            &format!(
                "`{from}` -> `{to}`: {} ids move, worst weight {worst}",
                ids.iter().zip(&want_ids).filter(|(a, b)| a != b).count()
            ),
        );
    }
}

/// `sqrt(log(1 + exp(x)))`, saturated at zero, with the shader's guard.
fn sqrt_softplus(x: f32) -> f32 {
    let sp = if x > 20.0 {
        x
    } else {
        (1.0 + plane::exp32(x)).ln()
    };
    sp.max(0.0).sqrt()
}

/// One lane per expert, rounded up to whole simdgroups.
fn lanes(experts: usize) -> u32 {
    (experts as u32).min(1024).div_ceil(32) * 32
}

// ── the routed projection ───────────────────────────────────────────────────

/// Which of the two activation layouts `moe.matmul_select` was handed.
#[derive(Clone, Copy)]
enum Shape {
    /// One row per token: a gate or up projection, slot stride zero.
    PerToken,
    /// One row per route: a down projection, slot stride `I`.
    PerRoute,
}

/// The bank, the routes, and both shapes of activation.
struct Select {
    bank: Vec<f32>,
    per_token: Vec<f32>,
    per_route: Vec<f32>,
    routes: Vec<i32>,
}

impl Select {
    fn of() -> Self {
        Self {
            bank: bf16_draw(EXPERTS * OUT_WIDTH * IN_WIDTH, 5, 0.6),
            per_token: bf16_draw(TOKENS * IN_WIDTH, 11, 1.0),
            per_route: bf16_draw(ROUTES * IN_WIDTH, 17, 1.0),
            // One route the router refused, so the zeroing branch is fired.
            routes: vec![2, 0, 1, 3, -1, 2],
        }
    }

    fn activation(&self, shape: Shape) -> &[f32] {
        match shape {
            Shape::PerToken => &self.per_token,
            Shape::PerRoute => &self.per_route,
        }
    }

    /// `y[route, n] = sum_k bank[routes[route], n, k] * x[row(route), k]`.
    fn model(&self, shape: Shape) -> (Vec<f32>, Vec<f32>) {
        let x = self.activation(shape);
        let mut y = vec![0.0; ROUTES * OUT_WIDTH];
        let mut scale = vec![0.0; ROUTES * OUT_WIDTH];
        for route in 0..ROUTES {
            let e = self.routes[route];
            if e < 0 {
                // A refused route is zeroed rather than left alone, so its
                // scale is one: what a kernel that skipped the write would
                // leave behind is measured against the number it should have
                // written, not against a magnitude it never accumulated.
                for n in 0..OUT_WIDTH {
                    scale[route * OUT_WIDTH + n] = 1.0;
                }
                continue;
            }
            let row = match shape {
                Shape::PerToken => route / FAN,
                Shape::PerRoute => route,
            };
            for n in 0..OUT_WIDTH {
                let (mut acc, mut mag) = (0.0f32, 0.0f32);
                for k in 0..IN_WIDTH {
                    let term = self.bank[((e as usize * OUT_WIDTH) + n) * IN_WIDTH + k]
                        * x[row * IN_WIDTH + k];
                    acc += term;
                    mag += term.abs();
                }
                y[route * OUT_WIDTH + n] = acc;
                scale[route * OUT_WIDTH + n] = mag;
            }
        }
        (y, scale)
    }

    fn fire(&self, rig: &Rig, root: &std::path::Path, shape: Shape) -> Vec<f32> {
        let (row_stride, slot_stride) = match shape {
            Shape::PerToken => (IN_WIDTH, 0),
            Shape::PerRoute => (IN_WIDTH * FAN, IN_WIDTH),
        };
        let x = plane::alloc_bf16(&rig.context, self.activation(shape), "x");
        let bank = plane::alloc_bf16(&rig.context, &self.bank, "bank");
        let routes = plane::alloc_i32(&rig.context, &self.routes, "routes");
        let y = plane::alloc_bf16(&rig.context, &[POISON; ROUTES * OUT_WIDTH], "y");
        plane::fire(
            rig,
            root,
            FILE_SELECT,
            "select_gemv",
            [(OUT_WIDTH * 32) as u32, ROUTES as u32, 1],
            [128, 1, 1],
            &[
                Arg::Buf(&x),
                Arg::Buf(&bank),
                Arg::Buf(&routes),
                Arg::Buf(&y),
                Arg::I32(IN_WIDTH as i32),
                Arg::I32(OUT_WIDTH as i32),
                Arg::I32(FAN as i32),
                Arg::I32(row_stride as i32),
                Arg::I32(slot_stride as i32),
            ],
        );
        plane::read_bf16(&y, ROUTES * OUT_WIDTH)
    }

    fn worst(&self, got: &[f32], shape: Shape) -> f32 {
        let (want, scale) = self.model(shape);
        got.iter()
            .zip(&want)
            .zip(&scale)
            .map(|((g, w), s)| (g - w).abs() / s)
            .fold(0.0, f32::max)
    }

    fn agrees(&self, got: &[f32], shape: Shape, what: &str) {
        let worst = self.worst(got, shape);
        assert!(
            worst <= SCALE_BOUND,
            "{what}: the widest element is {worst} of the dot product's own \
             scale, past the {SCALE_BOUND} one bf16 step allows"
        );
        plane::measured(
            what,
            &format!("worst {worst} against the scale bound {SCALE_BOUND}"),
        );
    }

    fn bites(&self, rig: &Rig, shape: Shape, from: &str, to: &str) {
        let root = plane::mutant(FILE_SELECT, from, to);
        let got = self.fire(rig, root.path(), shape);
        let worst = self.worst(&got, shape);
        assert!(
            worst > SCALE_BOUND,
            "replacing `{from}` with `{to}` left every element within {worst} \
             of the scale bound, so the comparison above would not have \
             caught it"
        );
        plane::measured(
            "select_gemv",
            &format!("`{from}` -> `{to}`: worst {worst} against the scale bound {SCALE_BOUND}"),
        );
    }
}

// ── the two folds ───────────────────────────────────────────────────────────

/// The routed stack, its weights, and the shared expert's three planes.
struct Fold {
    routed: Vec<f32>,
    weights: Vec<f32>,
    shared: Vec<f32>,
    gate: Vec<f32>,
}

const WIDTH: usize = 30;

impl Fold {
    fn of() -> Self {
        Self {
            routed: bf16_draw(TOKENS * FAN * WIDTH, 3, 1.0),
            weights: (0..TOKENS * FAN)
                .map(|i| 0.2 + 0.13 * ((i * 5) % 7) as f32)
                .collect(),
            shared: bf16_draw(TOKENS * WIDTH, 23, 1.0),
            gate: bf16_draw(TOKENS, 29, 3.0),
        }
    }

    fn combine_model(&self) -> Vec<f32> {
        let mut y = vec![0.0; TOKENS * WIDTH];
        for t in 0..TOKENS {
            for c in 0..WIDTH {
                let mut acc = 0.0f32;
                for s in 0..FAN {
                    acc += self.weights[t * FAN + s] * self.routed[(t * FAN + s) * WIDTH + c];
                }
                y[t * WIDTH + c] = plane::narrowed(acc);
            }
        }
        y
    }

    fn gate_model(&self) -> Vec<f32> {
        let mut y = vec![0.0; TOKENS * WIDTH];
        for t in 0..TOKENS {
            let g = 1.0 / (1.0 + plane::exp32(-self.gate[t]));
            for c in 0..WIDTH {
                // `routed` is reused as the mixture's own output: this fold
                // is `y = routed + sigmoid(gate) * shared` and the two
                // rectangles are the same shape.
                y[t * WIDTH + c] =
                    plane::narrowed(self.routed[t * WIDTH + c] + g * self.shared[t * WIDTH + c]);
            }
        }
        y
    }

    fn combine(&self, rig: &Rig, root: &std::path::Path) -> Vec<f32> {
        let routed = plane::alloc_bf16(&rig.context, &self.routed, "routed");
        let weights = plane::alloc_f32(&rig.context, &self.weights, "expert_weights");
        let y = plane::alloc_bf16(&rig.context, &[POISON; TOKENS * WIDTH], "y");
        plane::fire(
            rig,
            root,
            FILE_ROUTE,
            "expert_combine",
            [WIDTH as u32, TOKENS as u32, 1],
            [WIDTH.min(256) as u32, 1, 1],
            &[
                Arg::Buf(&routed),
                Arg::Buf(&weights),
                Arg::Buf(&y),
                Arg::I32(WIDTH as i32),
                Arg::I32(FAN as i32),
            ],
        );
        plane::read_bf16(&y, TOKENS * WIDTH)
    }

    fn gate(&self, rig: &Rig, root: &std::path::Path) -> Vec<f32> {
        let routed = plane::alloc_bf16(&rig.context, &self.routed[..TOKENS * WIDTH], "routed");
        let shared = plane::alloc_bf16(&rig.context, &self.shared, "shared");
        let gate = plane::alloc_bf16(&rig.context, &self.gate, "gate");
        let y = plane::alloc_bf16(&rig.context, &[POISON; TOKENS * WIDTH], "y");
        plane::fire(
            rig,
            root,
            FILE_ROUTE,
            "shared_expert_combine",
            [WIDTH as u32, TOKENS as u32, 1],
            [WIDTH.min(256) as u32, 1, 1],
            &[
                Arg::Buf(&routed),
                Arg::Buf(&shared),
                Arg::Buf(&gate),
                Arg::Buf(&y),
                Arg::I32(WIDTH as i32),
            ],
        );
        plane::read_bf16(&y, TOKENS * WIDTH)
    }

    fn agrees(&self, got: &[f32], want: &[f32], what: &str) {
        let (widest, at, inexact) = plane::ulp_spread(got, want);
        assert!(
            widest <= 1,
            "{what}: element {at} is {widest} bf16 steps from the model -- {} \
             against {}",
            got[at],
            want[at]
        );
        plane::measured(
            what,
            &format!(
                "{widest} bf16 steps at worst, {inexact} of {} elements inexact",
                got.len()
            ),
        );
    }

    fn combine_bites(&self, rig: &Rig, from: &str, to: &str) {
        let root = plane::mutant(FILE_ROUTE, from, to);
        let got = self.combine(rig, root.path());
        let (widest, _, _) = plane::ulp_spread(&got, &self.combine_model());
        assert!(
            widest > 1,
            "replacing `{from}` with `{to}` moved the fold by {widest} bf16 \
             steps, so the comparison above would not have caught it"
        );
        plane::measured(
            "expert_combine",
            &format!("`{from}` -> `{to}` moves the fold {widest} bf16 steps"),
        );
    }

    fn gate_bites(&self, rig: &Rig, from: &str, to: &str) {
        let root = plane::mutant(FILE_ROUTE, from, to);
        let got = self.gate(rig, root.path());
        let (widest, _, _) = plane::ulp_spread(&got, &self.gate_model());
        assert!(
            widest > 1,
            "replacing `{from}` with `{to}` moved the gated sum by {widest} \
             bf16 steps, so the comparison above would not have caught it"
        );
        plane::measured(
            "shared_expert_combine",
            &format!("`{from}` -> `{to}` moves the gated sum {widest} bf16 steps"),
        );
    }
}

/// A draw a bf16 buffer holds exactly.
fn bf16_draw(n: usize, seed: usize, gain: f32) -> Vec<f32> {
    (0..n)
        .map(|i| plane::narrowed((((i * 7 + seed * 13) % 17) as f32 - 8.0) / 8.5 * gain))
        .collect()
}
