//! The gated-delta-net core, on the GPU, for the first time.
//!
//! `ssm/gdn_core.metal` and `ssm/gdn_prep.metal` are sixteen entry points
//! compiled into every build this driver ships and signed into its archive.
//!
//! WHEN THIS FILE WAS WRITTEN all sixteen were dark, and this header said so:
//! `qwen_3_5`'s Metal projection refused the family, so no statement named
//! them and no dispatch had ever opened one. They were the largest dark block
//! in the tree and the ledger's own word for them was WRITTEN.
//!
//! Every clause of that has since been falsified by work, which is the only
//! way a paragraph like it is supposed to end. `model-dsl/src/metal.rs:2079`
//! onward projects the family; `driver-metal`'s `lowering/consts.rs` builds
//! its parameter struct and `lowering/routine.rs` crosses it; `ssm.rs` fires
//! it at seven sites; and `routine::DARK` is now ONE row long and that row is
//! `silu_mul_strided`. None of the sixteen are on it, and none are among the
//! four names `kernels-metal`'s `UNFIRED` still carries.
//!
//! The paragraph is kept rather than deleted because the test below is only
//! legible with it: everything the file does by hand, it does because at the
//! time nothing else could, and a reader who finds hand-built `Dispatch`es in
//! a family the lowering now reaches should know they are not a workaround
//! for a gap that still exists. They are the measurement that closed it.
//!
//! Written is not measured. Nothing on any backend has asked a GDN kernel for
//! a number — not a reference, not a differential, not a shape. The lowering
//! work that would make them reachable is real work, and it would land on top
//! of four hundred lines of arithmetic that has never been executed. This test
//! executes it, the way `device_attention` executed the paged attention family:
//! by building the `Dispatch` by hand, so reachability is not a prerequisite
//! for correctness.
//!
//! # What the kernel is
//!
//! One dispatch that fuses the whole decode core: a `Kc`-tap causal
//! convolution with SiLU over the in-projection's mixed q/k/v, an l2 norm of q
//! and k with q pre-scaled by `1/sqrt(Dk)`, the gate
//! `decay = exp(-exp(A_log) * softplus(a + dt_bias))` and
//! `beta = sigmoid(b)`, one step of the delta rule against a persistent
//! `[slots, Hv, Dv, Dk]` recurrent state, and a shift-and-append of the
//! convolution history into a separate output slab.
//!
//! # The two claims, and why both are needed
//!
//! A CPU reference says the arithmetic is the arithmetic. It is worth having
//! because every term above is a place a port can be subtly wrong -- a
//! softplus that overflows, an `eps` inside or outside the square root, a
//! decay applied after the delta instead of before -- and none of those would
//! make the kernel crash.
//!
//! The DIFFERENTIAL says something the reference cannot. `gdn_core` and
//! `gdn_core_slotted` are one template at `SLOTTED = false` and `true`, and
//! the file's claim about them is precise: "slot = b_idx (the sealed M=1 path;
//! slot_ids never read -> byte-identical)". That is a claim that two entry
//! points are one kernel, and this tree has already been bitten once by
//! believing such a claim -- see `sdpa_paged_mma`, where three bodies that
//! answered the same softmax did not share the same contract. So the slotted
//! form is fired three ways: at identity slots, where it must agree with the
//! unslotted form to the LAST BIT rather than to a tolerance; and at a
//! permuted slot map over a permuted state slab, where it must still answer
//! the same thing, which is the only test that the indirection is an
//! indirection and not a second name for `b_idx`.
//!
//! # The GQA repeat, which is why `Hk != Hv` here
//!
//! The header records a bug this family already had: every kernel indexed q
//! and k by the VALUE head, which is the same expression as indexing by the
//! key head only when `Hv == Hk`. That is true of qwen3.5-0.8B and of
//! `Qwen3.6-35B-A3B` and false of `Qwen3.6-27B` (`Hk = 16`, `Hv = 48`), where
//! a v-head past the sixteenth read its q from inside the K and V regions of
//! the same convolution output -- in bounds, finite, and wrong.
//!
//! A fixture at `rep == 1` cannot see that, and `rep == 1` is what every
//! checkpoint anyone has run locally has. So this one is `Hk = 2, Hv = 4`,
//! `rep = 2`, and the reference indexes q/k by `hv / rep` because that is what
//! the fix says. Setting it back to `hv` is a one-character injection that the
//! test fails on.
//!
//! # The tolerance, which was wrong before the kernel was
//!
//! This test passed the first time it was ever run, which is not a thing to
//! be pleased about. Five injections were then put through it, and one of the
//! first two got through: scaling the reference's `decay` by 1.01 PASSED.
//!
//! The bound was `max(|want|/64, 1/128)`, reasoned from bf16's half-ulp with
//! room for the reduction order. Reasoned, not measured. What this device
//! actually delivers over this fixture is 3.1e-3 of `max(|want|, 1/16)` on
//! the output and 6.8e-3 on the state, so the bound was better than twice the
//! truth in its relative term and twelve times it at its floor -- and a
//! one-percent error in the central gate of the whole kernel lived in that
//! gap, undetectable.
//!
//! Worse, the test already had a check whose stated job was to prevent
//! exactly this, copied from `device_attention`, where it also stood in three
//! places. It asserted that a perturbation of twice the bound exceeds the
//! bound. That is `2b > b`: true of every bound that was ever written,
//! including the loose one, and it passed all the way through. A tolerance
//! check that does not read the device measures nothing, and its presence is
//! worse than its absence because it is read as assurance.
//!
//! Both are fixed, and the fix is a claim rather than a comment: the bound is
//! `max(|want|, 1/16)/96`, and the tail measures the worst element of both
//! slabs against its own bound and requires the ratio to land between an
//! eighth and one. Too low and the tolerance has started accepting errors the
//! kernel is not making; too high and it is one flaky element from red. A
//! hand that loosens the bound to quiet a failure trips the floor instead of
//! getting away with it -- verified by loosening it sixteenfold, which now
//! fails at 0.041 of the bound rather than passing.
//!
//! `device_attention`'s three copies got the same treatment. Their bounds
//! turned out to be honest -- 0.45 of the bound at worst -- so nothing there
//! was wrong except the sentence claiming it had been checked.

#![cfg(target_vendor = "apple")]

use driver_metal::bind::encode::{Params, Pipelines, encode};
use driver_metal::device::{Allocation, ArgumentTable, Context, Stepper};
use driver_metal::layout::region::Region as _;
use driver_metal::lowering::dispatch::{Dispatch, ParamSlot, Touches};
use driver_metal::lowering::executor::{BoundArg, Slice};

/// One key head's width, and `32 * n_per_t` with `n_per_t = Dk / 32`.
use driver_metal::skip::skipped;

const DK: usize = 128;
/// Value channels a head owns. A multiple of the threadgroup's four
/// simdgroups, because the tile is `tg = {32, 4, 1}` over this axis.
const DV: usize = 8;
const HK: usize = 2;
const HV: usize = 4;
/// `Hv / Hk`. Two, deliberately: see the module doc.
const REP: usize = HV / HK;
const KC: usize = 4;
const ROWS: usize = 2;
const SLOTS: usize = 3;
const Q_OFF: usize = 0;
const K_OFF: usize = HK * DK;
const V_OFF: usize = 2 * HK * DK;
const CDIM: usize = 2 * HK * DK + HV * DV;
const EPS: f32 = 1e-6;

/// A value generator whose period shares no factor with any stride here.
///
/// `CDIM`, `DK`, `DV` and `KC` are 544, 128, 8 and 4, so a period sharing a
/// factor with them would hand two different channels the same number and a
/// mis-indexed read would land on the value it should have had. Seventeen is
/// prime and divides none of them. The values stay inside `[-1, 1]` so the
/// convolution's SiLU and the gate's `exp` are in the range a checkpoint puts
/// them in rather than saturated, which is where a wrong sign hides.
fn spread(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = ((i * 7 + seed * 13) % 17) as f32;
            (t - 8.0) / 8.5
        })
        .collect()
}

/// The kernel, in Rust, from `gdn_core.metal`'s body.
struct Reference {
    core_out: Vec<f32>,
    rstate: Vec<f32>,
    new_conv_state: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn reference(
    mixed: &[f32],
    conv_state: &[f32],
    rstate_in: &[f32],
    conv_w: &[f32],
    conv_b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    a_gate: &[f32],
    b_gate: &[f32],
    slot_of: &[usize],
) -> Reference {
    let inv_sqrt_dk = 1.0 / (DK as f32).sqrt();
    let mut core_out = vec![0.0f32; ROWS * HV * DV];
    let mut rstate = rstate_in.to_vec();
    let mut new_conv_state = vec![0.0f32; SLOTS * KC * CDIM];

    // `conv_state` is READ-ONLY and `new_conv_state` is a separate slab, so
    // every channel of every slot the fire touches is written and the rest
    // keeps whatever it was allocated with. The kernel writes q and k from the
    // first v-head of a key group alone and v from every `(hv, dv)`, which is
    // full coverage exactly once -- a reference that wrote them per v-head
    // would agree anyway, so the test checks the UNTOUCHED slot too.
    let convsilu = |c: usize, b: usize, slot: usize| -> f32 {
        let mut acc = conv_b[c];
        for j in 0..KC - 1 {
            acc += conv_state[(slot * KC + (j + 1)) * CDIM + c] * conv_w[c * KC + j];
        }
        acc += mixed[b * CDIM + c] * conv_w[c * KC + (KC - 1)];
        acc / (1.0 + (-acc).exp())
    };

    for b in 0..ROWS {
        let slot = slot_of[b];
        for hv in 0..HV {
            // `hv / rep`, NOT `hv`. The one-character difference the header
            // calls "in bounds, finite, and wrong".
            let hk = hv / REP;
            let qraw: Vec<f32> = (0..DK)
                .map(|d| convsilu(Q_OFF + hk * DK + d, b, slot))
                .collect();
            let kraw: Vec<f32> = (0..DK)
                .map(|d| convsilu(K_OFF + hk * DK + d, b, slot))
                .collect();
            let qsq: f32 = qraw.iter().map(|x| x * x).sum();
            let ksq: f32 = kraw.iter().map(|x| x * x).sum();
            let qinv = inv_sqrt_dk / (qsq + EPS).sqrt();
            let kinv = 1.0 / (ksq + EPS).sqrt();
            let q: Vec<f32> = qraw.iter().map(|x| x * qinv).collect();
            let k: Vec<f32> = kraw.iter().map(|x| x * kinv).collect();

            let ad = a_gate[b * HV + hv] + dt_bias[hv];
            let softplus = ad.max(0.0) + (1.0 + (-ad.abs()).exp()).ln();
            let decay = (-a_log[hv].exp() * softplus).exp();
            let beta = 1.0 / (1.0 + (-b_gate[b * HV + hv]).exp());

            for dv in 0..DV {
                let vval = convsilu(V_OFF + hv * DV + dv, b, slot);
                let base = ((slot * HV + hv) * DV + dv) * DK;
                let st = &mut rstate[base..base + DK];
                let mut kv = 0.0f32;
                for (i, s) in st.iter_mut().enumerate() {
                    *s *= decay;
                    kv += *s * k[i];
                }
                let delta = (vval - kv) * beta;
                let mut out = 0.0f32;
                for (i, s) in st.iter_mut().enumerate() {
                    *s += k[i] * delta;
                    out += *s * q[i];
                }
                core_out[(b * HV + hv) * DV + dv] = out;
            }

            let mut writeback = |c: usize| {
                for j in 0..KC - 1 {
                    new_conv_state[(slot * KC + j) * CDIM + c] =
                        conv_state[(slot * KC + (j + 1)) * CDIM + c];
                }
                new_conv_state[(slot * KC + (KC - 1)) * CDIM + c] = mixed[b * CDIM + c];
            };
            if hv % REP == 0 {
                for d in 0..DK {
                    writeback(Q_OFF + hk * DK + d);
                    writeback(K_OFF + hk * DK + d);
                }
            }
            for dv in 0..DV {
                writeback(V_OFF + hv * DV + dv);
            }
        }
    }

    Reference {
        core_out,
        rstate,
        new_conv_state,
    }
}

/// What came back from one fire.
struct Answer {
    core_out: Vec<f32>,
    rstate: Vec<f32>,
    new_conv_state: Vec<f32>,
}

/// One dispatch of one entry point, into fresh state.
///
/// `rstate` is IN PLACE -- the kernel reads and writes the same slab -- so
/// every arm gets its own copy or the second arm would step a state the first
/// one already advanced. That is not a detail of the test: it is the reason
/// the family needs a ping-pong `new_conv_state` while the recurrent state
/// does not, and firing two arms against one slab is the mistake the shader's
/// own header spends a paragraph on.
#[allow(clippy::too_many_arguments)]
/// The eleven `GdnShape::params` scalars, ONE BUFFER EACH, starting at `base`.
///
/// THERE WAS A STRUCT HERE and every fixture in this file staged one: a single
/// `packed: true` slot pointing at eleven four-byte fields, matching a
/// `GdnCoreParams` that `ssm/gdn_params.h` declared and that Vulkan and wgpu
/// each kept their own copy of. `gdn_core.metal`'s header records its removal
/// and the reason -- the routines name all eleven as `Const<i32>`/`Const<f32>`
/// marks, so the contract is a SIGNATURE and there is no struct left to keep
/// three copies of in step.
///
/// THE SHADERS MOVED AND THIS FILE DID NOT, which is the only reason the three
/// tests below were red. A packed slot bound at 11 handed `gdn_core_bfloat16`
/// a POINTER where it declares `const constant int& Dk`, and left Dv through
/// inv_sqrt_dk bound to nothing at all. The failure was not a refusal: the
/// pipeline builds, the dispatch encodes, and the kernel reads a garbage
/// extent and writes nothing -- "row 0 v-head 0 channel 0 is 0 and the
/// reference is 0.12245929", an all-zero output with no error anywhere.
///
/// AND `slot_ids` MOVED WITH THEM, which this file had equally missed. It was
/// declared past the struct and now sits WHERE THE STRUCT DID, so the slotted
/// forms bind it at 12 (prep) and 10 (recurrent) rather than 13 and 11. That
/// is the one place the numbering moved, and `gdn_core.metal`'s header says
/// so in as many words.
///
/// `at` walks the SAME staged run the struct was staged from, four bytes a
/// field in `GdnShape::params` order, because that order was always the
/// statement's and the removal did not touch it.
fn gdn_scalar_slots(base: usize) -> Vec<ParamSlot> {
    (0..11)
        .map(|i| ParamSlot {
            slot: base + i,
            at: (i as u32) * 4,
            bytes: 4,
            packed: false,
            value: Some(i as u8),
        })
        .collect()
}

/// [`gdn_scalar_slots`] plus the prefill scan's `row_pitch` and `n_scan`,
/// which follow the eleven at words 11 and 12 of the same run.
fn gdn_prefill_slots(base: usize) -> Vec<ParamSlot> {
    let mut slots = gdn_scalar_slots(base);
    slots.push(ParamSlot {
        slot: base + 11,
        at: 44,
        bytes: 4,
        packed: false,
        value: Some(11),
    });
    slots.push(ParamSlot {
        slot: base + 12,
        at: 48,
        bytes: 4,
        packed: false,
        value: Some(12),
    });
    slots
}

fn fire(
    context: &Context,
    compiler: &driver_metal::program::Compiler,
    entrypoint: &str,
    mixed: &[f32],
    conv_state: &[f32],
    rstate_in: &[f32],
    conv_w: &[f32],
    conv_b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    a_gate: &[f32],
    b_gate: &[f32],
    slot_ids: Option<&[u32]>,
) -> Answer {
    let mixed_a = alloc_bf16(context, mixed, "mixed");
    let conv_state_a = alloc_f32(context, conv_state, "conv_state");
    let rstate_a = alloc_f32(context, rstate_in, "rstate");
    let core_out_a = alloc_bf16(context, &vec![0.0; ROWS * HV * DV], "core_out");
    let conv_w_a = alloc_bf16(context, conv_w, "conv_w");
    let conv_b_a = alloc_bf16(context, conv_b, "conv_b");
    let a_log_a = alloc_f32(context, a_log, "A_log");
    let dt_bias_a = alloc_bf16(context, dt_bias, "dt_bias");
    let a_gate_a = alloc_bf16(context, a_gate, "a_gate");
    let b_gate_a = alloc_bf16(context, b_gate, "b_gate");
    // Poisoned, so a channel the kernel never writes is visible as poison
    // rather than as a plausible zero.
    let new_conv_a = alloc_f32(context, &vec![-99.0; SLOTS * KC * CDIM], "new_conv_state");
    let slot_a = slot_ids.map(|s| alloc_words(context, s, "slot_ids"));

    // `gdn_core` sealed: eleven buffers then eleven scalars, ending at 21.
    // Slotted puts `slot_ids` at 11 -- where the struct used to sit -- and
    // pushes the scalars to 12..22.
    let (wide, scalar_base) = if slot_ids.is_some() {
        (23, 12)
    } else {
        (22, 11)
    };
    let mut args = vec![
        BoundArg {
            slice: Slice {
                address: core_out_a.gpu_address(),
                bytes: 1 << 20,
            },
            width: 0,
        };
        wide
    ];
    let mut bind = vec![
        (0usize, mixed_a.gpu_address()),
        (1, conv_state_a.gpu_address()),
        (2, rstate_a.gpu_address()),
        (3, core_out_a.gpu_address()),
        (4, conv_w_a.gpu_address()),
        (5, conv_b_a.gpu_address()),
        (6, a_log_a.gpu_address()),
        (7, dt_bias_a.gpu_address()),
        (8, a_gate_a.gpu_address()),
        (9, b_gate_a.gpu_address()),
        (10, new_conv_a.gpu_address()),
    ];
    if let Some(s) = slot_a.as_ref() {
        // 11, not 12: `slot_ids` sits where the params struct did.
        bind.push((11, s.gpu_address()));
    }
    for (slot, address) in &bind {
        args[*slot] = BoundArg {
            slice: Slice {
                address: *address,
                bytes: 1 << 20,
            },
            width: 0,
        };
    }

    // The eleven scalars in `GdnShape::params` order, bound one buffer each
    // now, the way the paged attention rows always bound theirs.
    let params = vec![
        DK as u32,
        DV as u32,
        HK as u32,
        HV as u32,
        CDIM as u32,
        KC as u32,
        Q_OFF as u32,
        K_OFF as u32,
        V_OFF as u32,
        EPS.to_bits(),
        (1.0f32 / (DK as f32).sqrt()).to_bits(),
    ];
    let param_slots = gdn_scalar_slots(scalar_base);

    let dispatch = Dispatch {
        symbol: entrypoint,
        file: "ssm/gdn_core.metal",
        stamp: "",
        // Threads, not groups: one simdgroup per `(row, v-head, v-channel)`,
        // four of them to a threadgroup over the `dv` axis.
        grid: [32, DV as u32, (ROWS * HV) as u32],
        threadgroup: [32, 4, 1],
        touches: Touches::everything(&args),
        args,
        params,
        param_slots,
        layers: 0..1,
        op: 0,
    };

    let mut pipelines = Pipelines::new(kernels_dir());
    pipelines
        .ensure(context, compiler, std::slice::from_ref(&dispatch))
        .unwrap_or_else(|why| panic!("`{entrypoint}` builds a pipeline: {why}"));
    let staged =
        Params::stage(context, std::slice::from_ref(&dispatch)).expect("the scalars stage");
    let table = ArgumentTable::new(context, wide).expect("a table as wide as the row");
    let mut stepper = Stepper::new(context).expect("a stepper");
    stepper
        .run(|encoder| {
            encode(
                encoder,
                &table,
                &pipelines,
                &staged,
                std::slice::from_ref(&dispatch),
            )
        })
        .unwrap_or_else(|why| panic!("`{entrypoint}` fires: {why}"));

    Answer {
        core_out: read_bf16(&core_out_a, ROWS * HV * DV),
        rstate: read_f32(&rstate_a, SLOTS * HV * DV * DK),
        new_conv_state: read_f32(&new_conv_a, SLOTS * KC * CDIM),
    }
}

/// What the pair left behind, including the scratch the fused kernel has no
/// equivalent of.
struct Pair {
    answer: Answer,
    pre_gate_tail: Vec<f32>,
}

/// The SPLIT path: `gdn_prep` then `gdn_core_recurrent`, two dispatches over
/// one scratch, in one encoder with the barrier `encode` puts between every
/// pair of dispatches.
///
/// `gdn_prep.metal` opens with a claim, and it is the strongest claim any
/// header in this tree makes: "Bit-exactness vs the single fused gdn_core:
/// pre_q/pre_k are stored fp32 with the SAME 32-lane simd_sum reduction over
/// Dk=128, so the values the recurrent kernel reads are IDENTICAL to the
/// in-kernel `sh_q/sh_k` floats." Two kernels, one number, to the bit.
///
/// This tree has been bitten by exactly that shape of claim before -- three
/// paged attention bodies that answered the same softmax and did not share
/// the same contract -- so the claim is worth firing rather than reading. A
/// tolerance would not test it: the interesting failure is a reduction that
/// associates differently, which lands a few ulps away and inside any bound
/// wide enough for bf16.
///
/// The three ABIs in this family are still three ABIs, and the paragraph that
/// stood here counted them in the struct era: "`gdn_core` puts params at
/// buffer 11 and `slot_ids` at 12; `gdn_prep` puts them at 12 and 13; and
/// `gdn_core_recurrent` puts params at 10 and `slot_ids` at 11 -- which is
/// where `gdn_core` puts its PARAMS. Nothing enforces that they agree,
/// because nothing has ever bound them."
///
/// THE LAST SENTENCE WAS RIGHT AND THIS FILE PAID FOR IT. Nothing bound them,
/// the shaders retired the struct, and these fixtures went on staging one --
/// so every index above is now off by the width of a struct that no longer
/// exists. What each entrypoint reads is written out at
/// [`gdn_scalar_slots`]: eleven loose scalars after the last buffer, with
/// `slot_ids` where the struct used to be. Three ABIs and no shared
/// declaration is still the state of things; the difference is only that the
/// count is derived from a base here instead of typed out five times.
#[allow(clippy::too_many_arguments)]
fn fire_pair(
    context: &Context,
    compiler: &driver_metal::program::Compiler,
    mixed: &[f32],
    conv_state: &[f32],
    rstate_in: &[f32],
    conv_w: &[f32],
    conv_b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    a_gate: &[f32],
    b_gate: &[f32],
    slot_ids: Option<&[u32]>,
) -> Pair {
    let mixed_a = alloc_bf16(context, mixed, "mixed");
    let conv_state_a = alloc_f32(context, conv_state, "conv_state");
    let rstate_a = alloc_f32(context, rstate_in, "rstate");
    let core_out_a = alloc_bf16(context, &vec![0.0; ROWS * HV * DV], "core_out");
    let conv_w_a = alloc_bf16(context, conv_w, "conv_w");
    let conv_b_a = alloc_bf16(context, conv_b, "conv_b");
    let a_log_a = alloc_f32(context, a_log, "A_log");
    let dt_bias_a = alloc_bf16(context, dt_bias, "dt_bias");
    let a_gate_a = alloc_bf16(context, a_gate, "a_gate");
    let b_gate_a = alloc_bf16(context, b_gate, "b_gate");
    let new_conv_a = alloc_f32(context, &vec![-99.0; SLOTS * KC * CDIM], "new_conv_state");
    let pre_q_a = alloc_f32(context, &vec![-99.0; ROWS * HV * DK], "pre_q");
    let pre_k_a = alloc_f32(context, &vec![-99.0; ROWS * HV * DK], "pre_k");
    // The header sizes this `[R, 2*Hv + Hv*Dv]` and calls the tail "precomputed
    // V". The decode prep writes `pre_gate[2*n]` and `pre_gate[2*n+1]` and
    // nothing else, and the decode recurrent computes its own `vval` with its
    // own `convsilu`. So allocate what the header asks for, POISON it, and let
    // the assertion say which of the two is true.
    let pre_gate_a = alloc_f32(context, &vec![-99.0; ROWS * (2 * HV + HV * DV)], "pre_gate");
    let slot_a = slot_ids.map(|s| alloc_words(context, s, "slot_ids"));

    let params = vec![
        DK as u32,
        DV as u32,
        HK as u32,
        HV as u32,
        CDIM as u32,
        KC as u32,
        Q_OFF as u32,
        K_OFF as u32,
        V_OFF as u32,
        EPS.to_bits(),
        (1.0f32 / (DK as f32).sqrt()).to_bits(),
    ];
    let fill = |bind: &[(usize, u64)], wide: usize| -> Vec<BoundArg> {
        let mut args = vec![
            BoundArg {
                slice: Slice {
                    address: core_out_a.gpu_address(),
                    bytes: 1 << 20,
                },
                width: 0,
            };
            wide
        ];
        for (slot, address) in bind {
            args[*slot] = BoundArg {
                slice: Slice {
                    address: *address,
                    bytes: 1 << 20,
                },
                width: 0,
            };
        }
        args
    };

    let mut prep_bind = vec![
        (0usize, mixed_a.gpu_address()),
        (1, conv_state_a.gpu_address()),
        (2, conv_w_a.gpu_address()),
        (3, conv_b_a.gpu_address()),
        (4, a_log_a.gpu_address()),
        (5, dt_bias_a.gpu_address()),
        (6, a_gate_a.gpu_address()),
        (7, b_gate_a.gpu_address()),
        (8, pre_q_a.gpu_address()),
        (9, pre_k_a.gpu_address()),
        (10, pre_gate_a.gpu_address()),
        (11, new_conv_a.gpu_address()),
    ];
    let mut rec_bind = vec![
        (0usize, mixed_a.gpu_address()),
        (1, conv_state_a.gpu_address()),
        (2, rstate_a.gpu_address()),
        (3, core_out_a.gpu_address()),
        (4, conv_w_a.gpu_address()),
        (5, conv_b_a.gpu_address()),
        (6, pre_q_a.gpu_address()),
        (7, pre_k_a.gpu_address()),
        (8, pre_gate_a.gpu_address()),
        (9, new_conv_a.gpu_address()),
    ];
    let (prep_wide, rec_wide) = if let Some(s) = slot_a.as_ref() {
        prep_bind.push((12, s.gpu_address()));
        rec_bind.push((10, s.gpu_address()));
        (13, 11)
    } else {
        (12, 10)
    };
    // Those two are now the SCALAR BASE for each half, not the row width: the
    // eleven follow whatever the last buffer was, so the row runs eleven
    // further than it used to.
    let (prep_base, rec_base) = (prep_wide, rec_wide);
    let (prep_wide, rec_wide) = (prep_base + 11, rec_base + 11);
    let suffix = if slot_ids.is_some() { "_slotted" } else { "" };
    let prep_sym = format!("gdn_prep{suffix}_bfloat16");
    let rec_sym = format!("gdn_core_recurrent{suffix}_bfloat16");

    let dispatches = vec![
        Dispatch {
            symbol: &prep_sym,
            file: "ssm/gdn_prep.metal",
            stamp: "",
            // One simdgroup per `(row, v-head)`: the q/k path computed exactly
            // once, where the fused kernel computes it once per `dv` TILE.
            grid: [32, 1, (ROWS * HV) as u32],
            threadgroup: [32, 1, 1],
            touches: Touches::everything(&fill(&prep_bind, prep_wide)),
            args: fill(&prep_bind, prep_wide),
            params: params.clone(),
            param_slots: gdn_scalar_slots(prep_base),
            layers: 0..1,
            op: 0,
        },
        Dispatch {
            symbol: &rec_sym,
            file: "ssm/gdn_prep.metal",
            stamp: "",
            grid: [32, DV as u32, (ROWS * HV) as u32],
            threadgroup: [32, 4, 1],
            touches: Touches::everything(&fill(&rec_bind, rec_wide)),
            args: fill(&rec_bind, rec_wide),
            params: params.clone(),
            param_slots: gdn_scalar_slots(rec_base),
            layers: 0..1,
            op: 0,
        },
    ];

    let mut pipelines = Pipelines::new(kernels_dir());
    pipelines
        .ensure(context, compiler, &dispatches)
        .unwrap_or_else(|why| panic!("the GDN pair builds pipelines: {why}"));
    let staged = Params::stage(context, &dispatches).expect("the scalars stage");
    let table = ArgumentTable::new(context, prep_wide.max(rec_wide))
        .expect("a table as wide as the wider row");
    let mut stepper = Stepper::new(context).expect("a stepper");
    stepper
        .run(|encoder| encode(encoder, &table, &pipelines, &staged, &dispatches))
        .unwrap_or_else(|why| panic!("the GDN pair fires: {why}"));

    Pair {
        answer: Answer {
            core_out: read_bf16(&core_out_a, ROWS * HV * DV),
            rstate: read_f32(&rstate_a, SLOTS * HV * DV * DK),
            new_conv_state: read_f32(&new_conv_a, SLOTS * KC * CDIM),
        },
        pre_gate_tail: read_f32(&pre_gate_a, ROWS * (2 * HV + HV * DV))[ROWS * 2 * HV..].to_vec(),
    }
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_gdn_core_answers_its_own_arithmetic() {
    let Ok(context) = Context::new() else {
        skipped("no Metal 4 device");
        return;
    };
    let compiler = driver_metal::program::Compiler::new(&context).expect("a compiler");

    let mixed = spread(ROWS * CDIM, 1);
    let conv_state = spread(SLOTS * KC * CDIM, 2);
    let rstate_in = spread(SLOTS * HV * DV * DK, 3);
    let conv_w = spread(CDIM * KC, 5);
    let conv_b = spread(CDIM, 7);
    // `A_log` is exponentiated twice over, so a spread centred on zero puts
    // `exp(A_log)` near one and the decay across the interesting part of its
    // range rather than pinned at 0 or 1.
    let a_log: Vec<f32> = (0..HV).map(|h| h as f32 * 0.25 - 0.5).collect();
    let dt_bias = spread(HV, 11);
    let a_gate = spread(ROWS * HV, 13);
    let b_gate = spread(ROWS * HV, 17);

    // The sealed path: slot == row.
    let identity: Vec<usize> = (0..ROWS).collect();
    let plain = fire(
        &context,
        &compiler,
        "gdn_core_bfloat16",
        &mixed,
        &conv_state,
        &rstate_in,
        &conv_w,
        &conv_b,
        &a_log,
        &dt_bias,
        &a_gate,
        &b_gate,
        None,
    );
    let want = reference(
        &mixed,
        &conv_state,
        &rstate_in,
        &conv_w,
        &conv_b,
        &a_log,
        &dt_bias,
        &a_gate,
        &b_gate,
        &identity,
    );

    // bfloat16 out of a chain that runs a convolution, a SiLU, an l2 norm, a
    // double exponential and a hundred-and-twenty-eight-term dot product.
    //
    // This bound is MEASURED, not guessed. The first version of it was
    // `max(|want|/64, 1/128)`, reasoned from bf16's half-ulp of 2^-9 with
    // room for the reduction order -- and a fault injection that scaled the
    // decay by 1.01 PASSED underneath it. What the device actually delivers
    // over this fixture is 3.1e-3 of `max(|want|, 1/16)` on the output and
    // 6.8e-3 on the state; the old bound admitted 1.6e-2 plus a 7.8e-3 floor,
    // better than twice the truth in the relative term and twelve times it at
    // the floor, which is exactly the room a one-percent error hid in.
    //
    // The form is a single relative bound against a floored magnitude rather
    // than a max of two terms, because the state elements that matter here are
    // small and an absolute floor is where the slack collects. The factor 96
    // leaves 1.5x over the worst element observed, and the tail of this test
    // asserts that headroom stays small so this comment cannot rot back into
    // a guess.
    let bound = |want: f32| want.abs().max(1.0 / 16.0) / 96.0;
    for b in 0..ROWS {
        for hv in 0..HV {
            for dv in 0..DV {
                let at = (b * HV + hv) * DV + dv;
                assert!(
                    (plain.core_out[at] - want.core_out[at]).abs() <= bound(want.core_out[at]),
                    "gdn_core_bfloat16: row {b} v-head {hv} channel {dv} is {} and the \
                     reference is {}",
                    plain.core_out[at],
                    want.core_out[at]
                );
            }
        }
    }
    for (i, w) in want.rstate.iter().enumerate() {
        assert!(
            (plain.rstate[i] - w).abs() <= bound(*w),
            "gdn_core_bfloat16: recurrent state element {i} is {} and the reference is {w}; \
             the state is fp32 and read back in place, so this is the kernel's own \
             arithmetic and not a narrowing",
            plain.rstate[i]
        );
    }
    // Only the slots the fire touched. A slot no row names keeps its poison,
    // which is the claim that the writeback is keyed by SLOT and not by row.
    for slot in 0..SLOTS {
        for i in 0..KC * CDIM {
            let at = slot * KC * CDIM + i;
            if identity.contains(&slot) {
                assert!(
                    (plain.new_conv_state[at] - want.new_conv_state[at]).abs() < 1.0 / 256.0,
                    "gdn_core_bfloat16: conv history element {i} of slot {slot} is {} and the \
                     reference shifts it to {}",
                    plain.new_conv_state[at],
                    want.new_conv_state[at]
                );
            } else {
                assert!(
                    (plain.new_conv_state[at] + 99.0).abs() < 1.0 / 256.0,
                    "gdn_core_bfloat16 wrote {} into element {i} of slot {slot}, which no row \
                     names; the conv writeback is supposed to be keyed by the slot",
                    plain.new_conv_state[at]
                );
            }
        }
    }

    // The slotted twin at IDENTITY slots. The file's claim is not "close": it
    // is that `slot_ids` is never read on the sealed path and the two entry
    // points are one kernel, so the bar here is every bit of every output.
    let sealed: Vec<u32> = (0..ROWS as u32).collect();
    let same = fire(
        &context,
        &compiler,
        "gdn_core_slotted_bfloat16",
        &mixed,
        &conv_state,
        &rstate_in,
        &conv_w,
        &conv_b,
        &a_log,
        &dt_bias,
        &a_gate,
        &b_gate,
        Some(&sealed),
    );
    assert_eq!(
        same.core_out, plain.core_out,
        "`gdn_core_slotted` at identity slots answered a different core output than \
         `gdn_core`; they are one template at `SLOTTED = false` and `true` and the file \
         claims the sealed path is byte-identical"
    );
    assert_eq!(
        same.rstate, plain.rstate,
        "`gdn_core_slotted` at identity slots left a different recurrent state"
    );
    assert_eq!(
        same.new_conv_state, plain.new_conv_state,
        "`gdn_core_slotted` at identity slots left a different conv history"
    );

    // And the indirection has to BE one. The same two rows, pointed at slots
    // 2 and 0, with the state slabs permuted to match: same answer, different
    // slabs. A `slot_ids` the kernel ignored would read slots 0 and 1 here and
    // answer something else entirely.
    let map: Vec<usize> = vec![2, 0];
    let mut conv_moved = vec![0.0f32; SLOTS * KC * CDIM];
    let mut rstate_moved = vec![0.0f32; SLOTS * HV * DV * DK];
    for (row, &slot) in map.iter().enumerate() {
        for j in 0..KC {
            let from = (row * KC + j) * CDIM;
            let to = (slot * KC + j) * CDIM;
            conv_moved[to..to + CDIM].copy_from_slice(&conv_state[from..from + CDIM]);
        }
        let span = HV * DV * DK;
        rstate_moved[slot * span..(slot + 1) * span]
            .copy_from_slice(&rstate_in[row * span..(row + 1) * span]);
    }
    let ids: Vec<u32> = map.iter().map(|s| *s as u32).collect();
    let moved = fire(
        &context,
        &compiler,
        "gdn_core_slotted_bfloat16",
        &mixed,
        &conv_moved,
        &rstate_moved,
        &conv_w,
        &conv_b,
        &a_log,
        &dt_bias,
        &a_gate,
        &b_gate,
        Some(&ids),
    );
    assert_eq!(
        moved.core_out, plain.core_out,
        "the same two rows over the same state, placed in slots {map:?} instead of 0 and 1, \
         answered a different core output; the slot map is supposed to be where the state \
         lives and nothing else"
    );
    for (row, &slot) in map.iter().enumerate() {
        let span = HV * DV * DK;
        assert_eq!(
            &moved.rstate[slot * span..(slot + 1) * span],
            &plain.rstate[row * span..(row + 1) * span],
            "row {row}'s advanced state did not land in slot {slot}"
        );
    }

    // The bound has to discriminate, and the check that says so has to be
    // about the device.
    //
    // The first version of this tail asserted that a perturbation of twice the
    // bound exceeds the bound. That is true of EVERY bound -- it is `2b > b`
    // wearing a fixture -- so it passed just as happily under the loose bound
    // that let a one-percent decay error through. It measured nothing.
    //
    // What actually keeps a tolerance honest is headroom: the slack between
    // what the hardware delivers and what the assertions admit. Below an
    // eighth, this test is one flaky element from red; above it, the bound has
    // started to accept errors the kernel is not making. So measure the worst
    // element of both slabs against its own bound and require the ratio to sit
    // in that band. A future hand that loosens `bound` to quiet a failure
    // trips this instead, and a device more accurate than this one tightens
    // the bound rather than silently widening the gap.
    let mut worst = 0.0f32;
    for (i, w) in want.core_out.iter().enumerate() {
        worst = worst.max((plain.core_out[i] - w).abs() / bound(*w));
    }
    for (i, w) in want.rstate.iter().enumerate() {
        worst = worst.max((plain.rstate[i] - w).abs() / bound(*w));
    }
    tolerance_holds(worst, "the fused GDN core against its reference");
}

/// How much of its own tolerance the worst element used, and the band that has
/// to hold. See the note on the tolerance in this file's header for why the
/// check that used to stand here -- perturb by twice the bound, assert the
/// perturbation exceeds the bound -- measured nothing at all.
fn tolerance_holds(worst: f32, what: &str) {
    // Exact agreement is not a loose bound; it is the absence of anything to
    // bound. Two paths that produce identical bits have no headroom to
    // measure, and on a device whose reduction happens to associate the same
    // way as the oracle's that is the RIGHT answer rather than a suspicious
    // one. The floor is about a bound that admits errors, and there is no
    // error here to admit.
    //
    // This matters because these bands were measured on one Mac. A different
    // GPU can land closer to the oracle than this one does -- the prefill
    // scan is already within a single fp32 ulp of the walked decode, and one
    // ulp from zero is not far -- and a floor that failed on perfect
    // agreement would turn a better device into a red build.
    if worst == 0.0 {
        return;
    }
    assert!(
        worst <= 1.0,
        "{what}: the worst element used {worst} of its bound, so an assertion above \
         passed by an accident of iteration order"
    );
    assert!(
        worst >= 0.125,
        "{what}: the worst element used only {worst} of its bound, so the tolerance is \
         more than eight times the arithmetic this device actually delivers, which is \
         the room a wrong kernel hides in -- tighten the bound rather than trusting it"
    );
}

/// The split path answers the fused path, to the bit.
///
/// `gdn_prep.metal` exists because the fused `gdn_core` recomputes the
/// dv-independent q/k path once per `dv` TILE -- its threadgroup share caps
/// the redundancy at 4x, and splitting into two dispatches bridged by global
/// scratch takes it to 1x. That is a performance argument. The correctness
/// argument the header makes alongside it is much stronger, and it is the one
/// worth firing: the split is BIT-EXACT, because `pre_q`/`pre_k` are fp32 and
/// reduced by the same 32-lane `simd_sum` the fused kernel uses for
/// `sh_q`/`sh_k`.
///
/// So this asserts equality and not agreement. A tolerance would not test the
/// claim at all: the failure a split like this actually has is a reduction
/// that associates differently, which lands a few ulps out and inside any
/// bound wide enough for a bf16 store.
///
/// Four more entrypoints stopped being merely written here, and they stopped
/// with the harder of the two available claims proved.
///
/// This used to say they "leave the DARK ledger's UNEXECUTED column". There
/// was no such column -- `routine::DARK` is a stem and a sentence, two fields
/// and never three -- and the phrase was describing a wider ledger, from
/// before the rows retired, that counted compiled-and-never-run separately
/// from crossed. What replaced it is split in two: `DARK` holds only what no
/// routine can express, and what is merely never FIRED is counted in
/// `kernels-metal`'s `UNFIRED`. These four are on neither.
#[test]
#[ignore = "needs a Metal 4 device"]
fn the_split_gdn_pair_is_the_fused_kernel_to_the_bit() {
    let Ok(context) = Context::new() else {
        skipped("no Metal 4 device");
        return;
    };
    let compiler = driver_metal::program::Compiler::new(&context).expect("a compiler");

    let mixed = spread(ROWS * CDIM, 1);
    let conv_state = spread(SLOTS * KC * CDIM, 2);
    let rstate_in = spread(SLOTS * HV * DV * DK, 3);
    let conv_w = spread(CDIM * KC, 5);
    let conv_b = spread(CDIM, 7);
    let a_log: Vec<f32> = (0..HV).map(|h| h as f32 * 0.25 - 0.5).collect();
    let dt_bias = spread(HV, 11);
    let a_gate = spread(ROWS * HV, 13);
    let b_gate = spread(ROWS * HV, 17);

    let fused = fire(
        &context,
        &compiler,
        "gdn_core_bfloat16",
        &mixed,
        &conv_state,
        &rstate_in,
        &conv_w,
        &conv_b,
        &a_log,
        &dt_bias,
        &a_gate,
        &b_gate,
        None,
    );
    let split = fire_pair(
        &context,
        &compiler,
        &mixed,
        &conv_state,
        &rstate_in,
        &conv_w,
        &conv_b,
        &a_log,
        &dt_bias,
        &a_gate,
        &b_gate,
        None,
    );

    assert_eq!(
        split.answer.core_out, fused.core_out,
        "`gdn_prep` + `gdn_core_recurrent` answered a different core output than the \
         fused `gdn_core`; the header's claim is bit-exactness and this is not even \
         equality"
    );
    assert_eq!(
        split.answer.rstate, fused.rstate,
        "the split pair advanced the recurrent state differently from the fused kernel"
    );
    // The coverage claim, which the fused kernel makes alone and the pair
    // makes between them: `gdn_prep` writes every q and k channel exactly
    // once (only the first value head of each key group does it), and
    // `gdn_core_recurrent` writes the v channel of every `(hv, dv)`. Between
    // them that is every channel of `conv_dim`, and a poison surviving here
    // is a channel the split forgot.
    assert_eq!(
        split.answer.new_conv_state, fused.new_conv_state,
        "the split pair left a different convolution history; the q/k channels are \
         `gdn_prep`'s and the v channels are `gdn_core_recurrent`'s, and between them \
         they are supposed to cover exactly what the fused kernel covers"
    );

    // The stale sizing. The header says prep emits
    // `pre_gate[R, 2*Hv + Hv*Dv]`, "{decay,beta} then precomputed V". The
    // decode prep writes two floats per `(row, v-head)` and no V at all --
    // that tail belongs to the PREFILL prep, which really does precompute V
    // at `pre_gate + t*pitch + 2*Hv`. The scratch was allocated at the
    // header's size and poisoned, so the poison surviving is the proof.
    assert!(
        split.pre_gate_tail.iter().all(|v| (*v + 99.0).abs() < 1e-6),
        "the decode `gdn_prep` wrote into the `Hv*Dv` tail the header calls \
         \"precomputed V\"; either the header is right and this test is wrong, or \
         {} of {} tail floats are no longer poison",
        split
            .pre_gate_tail
            .iter()
            .filter(|v| (**v + 99.0).abs() >= 1e-6)
            .count(),
        split.pre_gate_tail.len()
    );

    // And the slotted pair is the slotted fused kernel, over a permuted map,
    // for the same reason and to the same standard.
    let map: Vec<usize> = vec![2, 0];
    let mut conv_moved = vec![0.0f32; SLOTS * KC * CDIM];
    let mut rstate_moved = vec![0.0f32; SLOTS * HV * DV * DK];
    for (row, &slot) in map.iter().enumerate() {
        for j in 0..KC {
            let from = (row * KC + j) * CDIM;
            let to = (slot * KC + j) * CDIM;
            conv_moved[to..to + CDIM].copy_from_slice(&conv_state[from..from + CDIM]);
        }
        let span = HV * DV * DK;
        rstate_moved[slot * span..(slot + 1) * span]
            .copy_from_slice(&rstate_in[row * span..(row + 1) * span]);
    }
    let ids: Vec<u32> = map.iter().map(|s| *s as u32).collect();
    let fused_slotted = fire(
        &context,
        &compiler,
        "gdn_core_slotted_bfloat16",
        &mixed,
        &conv_moved,
        &rstate_moved,
        &conv_w,
        &conv_b,
        &a_log,
        &dt_bias,
        &a_gate,
        &b_gate,
        Some(&ids),
    );
    let split_slotted = fire_pair(
        &context,
        &compiler,
        &mixed,
        &conv_moved,
        &rstate_moved,
        &conv_w,
        &conv_b,
        &a_log,
        &dt_bias,
        &a_gate,
        &b_gate,
        Some(&ids),
    );
    assert_eq!(
        split_slotted.answer.core_out, fused_slotted.core_out,
        "`gdn_prep_slotted` + `gdn_core_recurrent_slotted` answered differently from \
         `gdn_core_slotted` over slot map {map:?}; the split and the slot map are \
         independent axes and this is where they cross"
    );
    assert_eq!(
        split_slotted.answer.rstate, fused_slotted.rstate,
        "the slotted split pair advanced the state differently"
    );
    assert_eq!(
        split_slotted.answer.new_conv_state, fused_slotted.new_conv_state,
        "the slotted split pair left a different convolution history"
    );

    // And the two dispatches are two. `gdn_core_recurrent` reads `pre_q`,
    // `pre_k` and `pre_gate` and computes none of them, so a fire that
    // dropped the prep -- or ran it after -- would read poison and answer
    // something nowhere near. That the answers above are BIT-equal to a
    // kernel that computes q/k itself is the whole proof that the barrier
    // `encode` puts between every pair of dispatches is the RAW edge the
    // header says the host DAG must insert.
    // And the crossing was a crossing. The permuted fire must answer the SAME
    // core output -- that is the claim -- so equality there proves nothing
    // about whether the slot map did anything. What has to differ is where
    // the advanced state LANDED: row 0 wrote slot 0 in the sealed fire and
    // slot 2 in the permuted one. If those slabs matched, `slot_ids` was
    // ignored and both halves of this test were reading the same fire twice.
    assert_ne!(
        split.answer.rstate, split_slotted.answer.rstate,
        "the sealed and permuted split fires left the state in the same places, so \
         `slot_ids` moved nothing and the crossing above proved nothing"
    );
    let span = HV * DV * DK;
    for (row, &slot) in map.iter().enumerate() {
        assert_eq!(
            &split_slotted.answer.rstate[slot * span..(slot + 1) * span],
            &split.answer.rstate[row * span..(row + 1) * span],
            "the split pair put row {row}'s advanced state somewhere other than slot \
             {slot}"
        );
    }
}

/// Tokens in the prompt this prefill scans. Greater than `KC` so the
/// convolution window walks off the persisted history and onto the prompt --
/// at `T_SCAN <= KC` every tap still comes from `conv_state` and the whole
/// point of the prefill form is invisible.
const T_SCAN: usize = 5;

/// The PROMPT's row stride, in activation elements.
///
/// `mixed` alone: it is how far apart the in-projection's token rows sit, and
/// a prefill's rectangle is a window into a packed projection. The fp32
/// scratch used to share it -- `pitch_f = row_pitch / 2`, one number for four
/// buffers -- which held only while the widest scratch row fit inside the
/// projection. `pre_gate` carries `2*Hv + Hv*Dv` floats and qwen3-next's is
/// 8320 against a conv_dim of 8192, so every token's value channels landed on
/// the next token's gates. Each buffer is packed at its own width now.
///
/// Not a round number, and larger than `CDIM`, because a pitch that happens
/// to equal a tensor's width cannot tell a pitched read from a packed one.
const ROW_PITCH: usize = 1100;

/// One token through the fused `gdn_core`, at `ROWS = 1` and one slot.
///
/// The oracle for the prefill path. `gdn_core` is the kernel the split pair
/// was proved bit-equal to, and it is proved against a CPU reference of the
/// whole core, so a prefill that answers this answers the arithmetic.
#[allow(clippy::too_many_arguments)]
fn fire_fused_one(
    context: &Context,
    compiler: &driver_metal::program::Compiler,
    mixed_row: &[f32],
    conv_state: &[f32],
    rstate_in: &[f32],
    w: &Weights,
    a_gate_row: &[f32],
    b_gate_row: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mixed_a = alloc_bf16(context, mixed_row, "mixed");
    let conv_state_a = alloc_f32(context, conv_state, "conv_state");
    let rstate_a = alloc_f32(context, rstate_in, "rstate");
    let core_out_a = alloc_bf16(context, &[0.0; HV * DV], "core_out");
    let conv_w_a = alloc_bf16(context, &w.conv_w, "conv_w");
    let conv_b_a = alloc_bf16(context, &w.conv_b, "conv_b");
    let a_log_a = alloc_f32(context, &w.a_log, "A_log");
    let dt_bias_a = alloc_bf16(context, &w.dt_bias, "dt_bias");
    let a_gate_a = alloc_bf16(context, a_gate_row, "a_gate");
    let b_gate_a = alloc_bf16(context, b_gate_row, "b_gate");
    let new_conv_a = alloc_f32(context, &vec![-99.0; KC * CDIM], "new_conv_state");

    let bind = [
        (0usize, mixed_a.gpu_address()),
        (1, conv_state_a.gpu_address()),
        (2, rstate_a.gpu_address()),
        (3, core_out_a.gpu_address()),
        (4, conv_w_a.gpu_address()),
        (5, conv_b_a.gpu_address()),
        (6, a_log_a.gpu_address()),
        (7, dt_bias_a.gpu_address()),
        (8, a_gate_a.gpu_address()),
        (9, b_gate_a.gpu_address()),
        (10, new_conv_a.gpu_address()),
    ];
    let mut args = vec![
        BoundArg {
            slice: Slice {
                address: core_out_a.gpu_address(),
                bytes: 1 << 20,
            },
            width: 0,
        };
        12
    ];
    for (slot, address) in &bind {
        args[*slot] = BoundArg {
            slice: Slice {
                address: *address,
                bytes: 1 << 20,
            },
            width: 0,
        };
    }
    let dispatch = Dispatch {
        symbol: "gdn_core_bfloat16",
        file: "ssm/gdn_core.metal",
        stamp: "",
        grid: [32, DV as u32, HV as u32],
        threadgroup: [32, 4, 1],
        touches: Touches::everything(&args),
        args,
        params: gdn_params(),
        param_slots: gdn_scalar_slots(11),
        layers: 0..1,
        op: 0,
    };
    let mut pipelines = Pipelines::new(kernels_dir());
    pipelines
        .ensure(context, compiler, std::slice::from_ref(&dispatch))
        .expect("the fused core builds");
    let staged =
        Params::stage(context, std::slice::from_ref(&dispatch)).expect("the scalars stage");
    let table = ArgumentTable::new(context, 22).expect("a table");
    let mut stepper = Stepper::new(context).expect("a stepper");
    stepper
        .run(|encoder| {
            encode(
                encoder,
                &table,
                &pipelines,
                &staged,
                std::slice::from_ref(&dispatch),
            )
        })
        .expect("the fused core fires");
    (
        read_bf16(&core_out_a, HV * DV),
        read_f32(&rstate_a, HV * DV * DK),
        read_f32(&new_conv_a, KC * CDIM),
    )
}

/// The whole prompt through `gdn_prep_prefill` + one scan tiling.
///
/// Two dispatches again, and the same barrier between them, but the scratch
/// is row-pitched over tokens and the scan walks all `T_SCAN` of them inside
/// one kernel with the recurrent state living in registers for the duration.
#[allow(clippy::too_many_arguments)]
fn fire_prefill(
    context: &Context,
    compiler: &driver_metal::program::Compiler,
    lanes: u32,
    vrows: u32,
    prompt: &[f32],
    conv_state: &[f32],
    rstate_in: &[f32],
    w: &Weights,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mixed_a = alloc_bf16(context, prompt, "mixed");
    let conv_state_a = alloc_f32(context, conv_state, "conv_state");
    let rstate_a = alloc_f32(context, rstate_in, "rstate");
    let core_out_a = alloc_bf16(context, &vec![0.0; T_SCAN * HV * DV], "core_out");
    let conv_w_a = alloc_bf16(context, &w.conv_w, "conv_w");
    let conv_b_a = alloc_bf16(context, &w.conv_b, "conv_b");
    let a_log_a = alloc_f32(context, &w.a_log, "A_log");
    let dt_bias_a = alloc_bf16(context, &w.dt_bias, "dt_bias");
    let a_gate_a = alloc_bf16(context, &w.a_gate_rows, "a_gate");
    let b_gate_a = alloc_bf16(context, &w.b_gate_rows, "b_gate");
    let new_conv_a = alloc_f32(context, &vec![-99.0; KC * CDIM], "new_conv_state");
    // Every scratch row is packed at its OWN width -- `row_pitch` describes
    // `mixed` and nothing else. See `gdn_prep.metal:377`: one shared pitch
    // has to clear the widest row, and `pre_gate`'s `2*Hv + Hv*Dv` beats a
    // real stack's conv_dim.
    let qk_pitch = HV * DK;
    let g_pitch = 2 * HV + HV * DV;
    let pre_q_a = alloc_f32(context, &vec![-99.0; T_SCAN * qk_pitch], "pre_q");
    let pre_k_a = alloc_f32(context, &vec![-99.0; T_SCAN * qk_pitch], "pre_k");
    let pre_gate_a = alloc_f32(context, &vec![-99.0; T_SCAN * g_pitch], "pre_gate");
    // ONE SEAT PER ROW, all the same seat: this prompt is one request.
    //
    // This staged a single word, because both prefill kernels read
    // `slot_ids[0]` and nothing else -- which is also why a fire holding two
    // requests convolved the second over the first's tokens. They read the
    // table per row now, and the row where the seat changes is the request
    // boundary.
    let slot_a = alloc_words(context, &[0u32; T_SCAN], "slot_ids");

    let fill = |bind: &[(usize, u64)], wide: usize| -> Vec<BoundArg> {
        let mut args = vec![
            BoundArg {
                slice: Slice {
                    address: core_out_a.gpu_address(),
                    bytes: 1 << 20,
                },
                width: 0,
            };
            wide
        ];
        for (slot, address) in bind {
            args[*slot] = BoundArg {
                slice: Slice {
                    address: *address,
                    bytes: 1 << 20,
                },
                width: 0,
            };
        }
        args
    };
    // `row_pitch` and `n_scan` follow the eleven at words 11 and 12 of the
    // same statement run. They were always loose `constant int&` operands
    // beside the packed struct; now that the eleven are loose too, all
    // thirteen are one uniform run and `gdn_prefill_slots` writes it.
    let scalars = {
        let mut v = gdn_params();
        v.push(ROW_PITCH as u32);
        v.push(T_SCAN as u32);
        v
    };
    let prep_bind = [
        (0usize, mixed_a.gpu_address()),
        (1, conv_state_a.gpu_address()),
        (2, conv_w_a.gpu_address()),
        (3, conv_b_a.gpu_address()),
        (4, a_log_a.gpu_address()),
        (5, dt_bias_a.gpu_address()),
        (6, a_gate_a.gpu_address()),
        (7, b_gate_a.gpu_address()),
        (8, pre_q_a.gpu_address()),
        (9, pre_k_a.gpu_address()),
        (10, pre_gate_a.gpu_address()),
        (11, new_conv_a.gpu_address()),
        (12, slot_a.gpu_address()),
    ];
    let scan_bind = [
        (2usize, rstate_a.gpu_address()),
        (3, core_out_a.gpu_address()),
        (6, pre_q_a.gpu_address()),
        (7, pre_k_a.gpu_address()),
        (8, pre_gate_a.gpu_address()),
        (10, slot_a.gpu_address()),
    ];
    let scan_sym = format!("gdn_core_recurrent_prefill_bfloat16_l_{lanes}_v_{vrows}");
    // `LANES` lanes own one `dv` row, so `32/LANES` rows share a simdgroup and
    // each walks `VROWS` of them; `grid.y` is what is left of `Dv`.
    let per_y = (32 / lanes) * vrows;
    let grid_y = (DV as u32).div_ceil(per_y);
    let dispatches = vec![
        Dispatch {
            symbol: "gdn_prep_prefill_bfloat16",
            file: "ssm/gdn_prep.metal",
            stamp: "",
            grid: [32, 1, (T_SCAN * HV) as u32],
            threadgroup: [32, 1, 1],
            touches: Touches::everything(&fill(&prep_bind, 26)),
            args: fill(&prep_bind, 26),
            params: scalars.clone(),
            // Twelve buffers, `slot_ids` at 12, the eleven at 13..23, then
            // row_pitch and n_scan at 24 and 25.
            param_slots: gdn_prefill_slots(13),
            layers: 0..1,
            op: 0,
        },
        Dispatch {
            symbol: &scan_sym,
            file: "ssm/gdn_prep.metal",
            stamp: "",
            grid: [32, grid_y, HV as u32],
            threadgroup: [32, 1, 1],
            touches: Touches::everything(&fill(&scan_bind, 24)),
            args: fill(&scan_bind, 24),
            params: scalars.clone(),
            // `slot_ids` at 10, the eleven at 11..21, then row_pitch and
            // n_scan at 22 and 23.
            param_slots: gdn_prefill_slots(11),
            layers: 0..1,
            op: 0,
        },
    ];

    let mut pipelines = Pipelines::new(kernels_dir());
    pipelines
        .ensure(context, compiler, &dispatches)
        .unwrap_or_else(|why| panic!("`{scan_sym}` builds a pipeline: {why}"));
    let staged = Params::stage(context, &dispatches).expect("the scalars stage");
    let table = ArgumentTable::new(context, 26).expect("a table as wide as the prep row");
    let mut stepper = Stepper::new(context).expect("a stepper");
    stepper
        .run(|encoder| encode(encoder, &table, &pipelines, &staged, &dispatches))
        .unwrap_or_else(|why| panic!("`{scan_sym}` fires: {why}"));

    (
        read_bf16(&core_out_a, T_SCAN * HV * DV),
        read_f32(&rstate_a, HV * DV * DK),
        read_f32(&new_conv_a, KC * CDIM),
    )
}

/// Everything both paths hold constant, in the two layouts they want it in.
struct Weights {
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    a_gate_rows: Vec<f32>,
    b_gate_rows: Vec<f32>,
}

fn gdn_params() -> Vec<u32> {
    vec![
        DK as u32,
        DV as u32,
        HK as u32,
        HV as u32,
        CDIM as u32,
        KC as u32,
        Q_OFF as u32,
        K_OFF as u32,
        V_OFF as u32,
        EPS.to_bits(),
        (1.0f32 / (DK as f32).sqrt()).to_bits(),
    ]
}

/// Every tiling of the prefill scan, `(LANES, VROWS)`.
///
/// Nine of the family's sixteen entrypoints are this one template. `LANES` is
/// how many lanes own a `dv` row and `VROWS` is how many rows one lane group
/// walks, and the header argues for both at length from measurements. Neither
/// is supposed to change a number.
const TILINGS: [(u32, u32); 9] = [
    (4, 1),
    (8, 1),
    (8, 2),
    (16, 1),
    (16, 2),
    (16, 4),
    (32, 2),
    (32, 4),
    (32, 8),
];

/// The prefill scan is the decode pair, walked.
///
/// This is the claim the whole prefill half of the family rests on, and
/// nothing has ever asked for it. A prompt walked token by token serializes
/// the GDN pair behind a barrier per token -- the header measures 1224
/// dispatches in a strict chain, 25 ms of a 60 ms prefill -- and the prefill
/// form removes that by making the convolution token-parallel and running the
/// recurrent scan in registers inside one kernel. The header's word for the
/// result is that "the arithmetic is unchanged".
///
/// Unchanged from WHAT is the point. So the oracle here is not a second CPU
/// reference: it is the decode path itself, `gdn_core` fired once per token
/// with the convolution history ping-ponged forward between fires, which is
/// exactly what the per-token path does in production. That kernel is proved
/// against a CPU reference of the whole core in
/// `the_gdn_core_answers_its_own_arithmetic`, so agreeing with it is agreeing
/// with the arithmetic, and disagreeing with it is the failure that would
/// actually ship: a prefill that fills a state the decode then continues from.
///
/// `T_SCAN = 5` against `KC = 4`, so the convolution window walks off the
/// persisted history and onto the prompt partway through. At `T_SCAN <= KC`
/// every tap still comes from `conv_state` and the token-parallel FIR -- the
/// first of the header's two observations -- is never exercised.
///
/// # What each of the nine has to satisfy
///
/// `LANES` changes the reduction: `gdn_row_sum<LANES>` is an xor tree over
/// exactly that many lanes and `n_per_t = Dk/LANES` sequential terms feed it,
/// so two tilings with different `LANES` associate a 128-term dot product
/// differently. `VROWS` does not: it only changes how many independent rows a
/// lane group carries, which is the header's whole argument for it.
///
/// That predicts a tolerance on everything and bit-equality only within a
/// width. Half of it survived contact with the device.
///
/// `core_out` is stored bf16, and the difference between a four-lane
/// reduction and a thirty-two-lane one is smaller than that store can
/// express: all nine tilings answer the walked decode on every one of 160
/// channels EXACTLY, and every tiling answers every other one exactly. So the
/// output is asserted with `assert_eq!` and no bound at all -- across both
/// parameters, which is stronger than this test was written to ask for.
///
/// `rstate` is fp32 and keeps what the output rounds off. It is bit-equal
/// across `VROWS` and NOT across `LANES`, exactly as predicted, and it sits
/// one fp32 ulp -- 2^-23, measured -- from the walked decode. That is the
/// bound, and the widths are held to it and no tighter.
#[test]
#[ignore = "needs a Metal 4 device"]
fn the_prefill_scan_answers_the_decode_walked_token_by_token() {
    let Ok(context) = Context::new() else {
        skipped("no Metal 4 device");
        return;
    };
    let compiler = driver_metal::program::Compiler::new(&context).expect("a compiler");

    let w = Weights {
        conv_w: spread(CDIM * KC, 5),
        conv_b: spread(CDIM, 7),
        a_log: (0..HV).map(|h| h as f32 * 0.25 - 0.5).collect(),
        dt_bias: spread(HV, 11),
        // Hv per token, which is the width `in_proj_a` and `in_proj_b`
        // produce. These were `T_SCAN * ROW_PITCH` and the decode oracle read
        // them at that stride too, so the prefill prep's `a_gate[row_t + hv]`
        // -- the PROMPT's pitch on a buffer Hv wide -- read exactly what the
        // oracle expected and the pair agreed on the wrong layout.
        a_gate_rows: spread(T_SCAN * HV, 13),
        b_gate_rows: spread(T_SCAN * HV, 17),
    };
    // The prompt, row-pitched. Only the first `CDIM` of each stride is read,
    // and the rest exists to make a packed read answer differently.
    let prompt = spread(T_SCAN * ROW_PITCH, 1);
    let conv_state0 = spread(KC * CDIM, 2);
    let rstate0 = spread(HV * DV * DK, 3);

    // The oracle: the decode kernel, once per token, with the convolution
    // history ping-ponged and the recurrent state carried forward.
    let mut conv = conv_state0.clone();
    let mut rst = rstate0.clone();
    let mut want_out: Vec<f32> = Vec::with_capacity(T_SCAN * HV * DV);
    for t in 0..T_SCAN {
        // Two strides, and they are different buffers: the prompt is pitched
        // and the gates are packed at Hv.
        let base = t * ROW_PITCH;
        let gate = t * HV;
        let (out, next_rst, next_conv) = fire_fused_one(
            &context,
            &compiler,
            &prompt[base..base + CDIM],
            &conv,
            &rst,
            &w,
            &w.a_gate_rows[gate..gate + HV],
            &w.b_gate_rows[gate..gate + HV],
        );
        want_out.extend_from_slice(&out);
        rst = next_rst;
        conv = next_conv;
    }

    // MEASURED, and much sharper than the guess it replaced.
    //
    // This was first written expecting `core_out` to need a tolerance:
    // `gdn_row_sum<LANES>` is an xor tree over exactly that many lanes with
    // `n_per_t = Dk/LANES` sequential terms feeding it, so four lanes and
    // thirty-two associate the same 128 products differently and ought to
    // land a few ulps apart. They do -- in fp32. The output is stored bf16,
    // which rounds every one of those differences away: all nine tilings
    // answer the walked decode on 160 of 160 channels EXACTLY, and so does
    // every tiling against every other.
    //
    // So `core_out` is asserted with `assert_eq!` and no bound at all. The
    // recurrent state is fp32 and keeps what the output loses: it differs by
    // 1.1920929e-7 of its magnitude, which is 2^-23 -- one ulp, in the last
    // place, and the same for every tiling. The bound is one doubling of
    // that, so the worst element uses half of it.
    let bound = |want: f32| want.abs().max(1.0 / 16.0) / f32::from(1u16 << 12) / 1024.0;
    /// One tiling's fire: `(LANES, VROWS, core_out, rstate, new_conv_state)`.
    type Tiled = (u32, u32, Vec<f32>, Vec<f32>, Vec<f32>);
    let mut answers: Vec<Tiled> = Vec::new();
    let mut worst = 0.0f32;
    for (lanes, vrows) in TILINGS {
        let (out, rstate, new_conv) = fire_prefill(
            &context,
            &compiler,
            lanes,
            vrows,
            &prompt,
            &conv_state0,
            &rstate0,
            &w,
        );
        assert_eq!(
            out, want_out,
            "`gdn_core_recurrent_prefill_bfloat16_l_{lanes}_v_{vrows}` answered \
             differently from the decode kernel walked token by token over the same \
             prompt. The output is bf16, which rounds away every difference the \
             reduction widths make, so this is not a tolerance question: the two \
             paths either compute the same thing or they do not"
        );
        for (i, want) in rst.iter().enumerate() {
            worst = worst.max((rstate[i] - want).abs() / bound(*want));
            assert!(
                (rstate[i] - want).abs() <= bound(*want),
                "`..._l_{lanes}_v_{vrows}`: recurrent state element {i} is {} after the \
                 scan and {want} after {T_SCAN} decode steps, which is further apart \
                 than the one fp32 ulp the reduction widths can explain; a prefill that \
                 leaves a different state hands the decode a prompt it did not see",
                rstate[i]
            );
        }
        // The convolution history the next token continues from. The prefill
        // writes it once, from the last scanned token, taking the whole
        // `Kc`-tap window straight out of the prompt; the decode reaches the
        // same place by shifting and appending `T_SCAN` times.
        assert_eq!(
            new_conv, conv,
            "`..._l_{lanes}_v_{vrows}`: the prefill left a different convolution \
             history than {T_SCAN} ping-ponged decode steps. Only the last scanned \
             token carries it forward, and it must carry forward exactly what the \
             walk arrived at"
        );
        answers.push((lanes, vrows, out, rstate, new_conv));
    }
    tolerance_holds(worst, "the prefill state against the walked decode");

    // And the nine are one kernel, in the two different senses the two
    // parameters earn.
    //
    // The OUTPUT is bit-identical across every tiling, `LANES` and `VROWS`
    // alike, because it is stored bf16 and the reduction widths differ by
    // less than that store can express.
    //
    // The STATE is not, and asserting that it was is how this test found out.
    // `rstate` is fp32, so it keeps what the output rounds off: `_l_8_v_1`
    // and `_l_4_v_1` leave genuinely different bits. What is invariant there
    // is `VROWS` -- same reduction width, same association, same bits --
    // which is exactly the claim the header makes for it: register blocking
    // that changes how many independent rows a lane group carries and nothing
    // about what is summed. So `VROWS` is asserted to the bit within a width,
    // and the widths are held to the one-ulp bound above and no tighter.
    for (lanes, vrows, out, rstate, _) in &answers {
        assert_eq!(
            out, &answers[0].2,
            "`_l_{lanes}_v_{vrows}` and `_l_{}_v_{}` answered differently. The output \
             is bf16 and no reduction width in this family moves a value that far",
            answers[0].0, answers[0].1
        );
        let (_, first_v, _, first_rstate, _) = answers
            .iter()
            .find(|(l, ..)| l == lanes)
            .expect("its own width");
        assert_eq!(
            rstate, first_rstate,
            "`_l_{lanes}_v_{vrows}` and `_l_{lanes}_v_{first_v}` left different \
             recurrent states. They share a reduction width, so `VROWS` changed an \
             association order, and `VROWS` is supposed to change only how many \
             independent rows one lane group carries"
        );
    }

    // The widths really are different reductions, or the `VROWS` equality
    // above is the trivial kind and the one-ulp bound is measuring nothing.
    assert!(
        answers
            .iter()
            .any(|(_, _, _, rstate, _)| rstate != &answers[0].3),
        "every tiling left bit-identical state, across widths as well as within them. \
         `gdn_row_sum<4>` and `gdn_row_sum<32>` associate the same 128 products \
         differently, so this means the fixture cannot expose an association order and \
         the `VROWS` equality proved nothing"
    );

    // The one thing bit-equality between nine tilings cannot say is whether
    // any of them ran. A `dv_base >= Dv` early return, a `grid.y` computed
    // wrong, a scan that never entered its token loop -- all nine would agree
    // beautifully on the poison they left. The `assert_eq!` against the walked
    // decode is what rules that out, and this is the check that the walk is
    // not itself trivial: five tokens have to move the state somewhere the
    // prompt's first token alone would not.
    assert_ne!(
        answers[0].3, rstate0,
        "the scan left the recurrent state exactly as it found it over {T_SCAN} \
         tokens, so nothing above compared two computations"
    );
}

fn kernels_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels")
}

fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = ((bits >> 16) & 1) + 0x7fff;
    ((bits.wrapping_add(round)) >> 16) as u16
}

fn from_bf16(x: u16) -> f32 {
    f32::from_bits(u32::from(x) << 16)
}

fn alloc_bf16(context: &Context, values: &[f32], what: &'static str) -> Allocation {
    let narrow: Vec<u16> = values.iter().copied().map(to_bf16).collect();
    let bytes = std::mem::size_of_val(narrow.as_slice()) as u64;
    let a = Allocation::new(context, bytes.max(4), what).expect("an allocation");
    unsafe {
        a.write(0, cast(&narrow)).expect("the halves fit");
    }
    a
}

fn alloc_f32(context: &Context, values: &[f32], what: &'static str) -> Allocation {
    let bytes = std::mem::size_of_val(values) as u64;
    let a = Allocation::new(context, bytes.max(4), what).expect("an allocation");
    unsafe {
        a.write(
            0,
            core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), bytes as usize),
        )
        .expect("the floats fit");
    }
    a
}

fn alloc_words(context: &Context, values: &[u32], what: &'static str) -> Allocation {
    let bytes = std::mem::size_of_val(values) as u64;
    let a = Allocation::new(context, bytes.max(4), what).expect("an allocation");
    unsafe {
        a.write(
            0,
            core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), bytes as usize),
        )
        .expect("the words fit");
    }
    a
}

fn read_bf16(a: &Allocation, n: usize) -> Vec<f32> {
    let words = unsafe { core::slice::from_raw_parts(a.contents().as_ptr().cast::<u16>(), n) };
    words.iter().copied().map(from_bf16).collect()
}

fn read_f32(a: &Allocation, n: usize) -> Vec<f32> {
    unsafe { core::slice::from_raw_parts(a.contents().as_ptr().cast::<f32>(), n) }.to_vec()
}

fn cast(v: &[u16]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
