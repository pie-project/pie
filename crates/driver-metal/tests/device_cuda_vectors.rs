//! The Metal shaders against the numbers the REAL CUDA kernels produced.
//!
//! Every other file in this sweep compares a Metal kernel to a model written
//! in Rust from that kernel's own body. That catches a great deal and it
//! cannot catch one thing: a TRANSCRIPTION error. `kernels-metal` was written
//! last week by reading `kernels-cuda`, and a model written by reading
//! `kernels-metal` agrees with `kernels-metal` about everything the reading
//! got wrong.
//!
//! # How a CUDA reference gets onto an Apple machine
//!
//! It does not run here -- there is no CUDA on a Mac, so the wgpu and Vulkan
//! planes' trick of firing the shader and its CUDA twin on the same card is
//! not available. What IS available is the shape
//! `crates/driver-cuda/tests/oracle/` already uses for the cuBLAS service:
//! run the real kernel on a real card, record what went in and what came out,
//! and carry the transcript.
//!
//! `tests/fixtures/cuda_reference.cu` is that program, `tests/fixtures/run.sh`
//! regenerates it, and the four `.txt` files beside them are what it produced
//! on an L40S at nvcc 13.0. Each `case` is one launch of a `__global__` out of
//! `crates/kernels-cuda/kernels/**` -- the same code cuda serves with -- and
//! each file's header names the kernel, the shape and every parameter.
//!
//! # THIS WAS EXPECTED TO NEED A TOLERANCE AND MOSTLY DOES NOT
//!
//! The fixtures' own header warns not to compare bit for bit, and the reasons
//! are real: the Metal sources deliberately spell things differently wherever
//! the difference is below the storage floor. `fast::exp` against `expf`,
//! `precise::tanh` against `tanhf`, `fast::cos` against `__sincosf`,
//! `exp2(-d * log2(theta))` against `powf(theta, -d)`,
//! `metal::log(1 + exp(x))` against `log1pf(expf(x))`, and
//! `packed_gptoss_swiglu`'s `g * (1 / (1 + exp))` against
//! `chunked_gpt_oss_glu`'s `g / (1 + exp)`. Two vendors' hardware
//! transcendentals, two compilers, two instruction sets.
//!
//! **Fourteen of the sixteen bf16-valued cases below agree to the BIT
//! anyway**, and the seven rotations are the ones worth staring at: four
//! different transcendentals between the two planes, and all 360 channels of
//! each of the seven came back with the same sixteen bits. What is doing that
//! is the storage element -- bf16 has eight mantissa bits and two f32 answers
//! within an ulp of each other round to the same one -- and it is why this
//! comparison is worth making at all rather than being drowned in
//! tolerance.
//!
//! So the allowances are stated PER CASE and most of them are zero, and each
//! non-zero one names the spelling that makes it non-zero. [`exactly`] then
//! pins the count: a hand that widens an allowance, or a shader change that
//! stops agreeing, has to come and edit a number that says how many cases
//! were exact.
//!
//! The two groups that do not agree to the bit are the ones whose answer is
//! not stored in bf16. A router's weight is `Out<Tensor<f32>>` -- a
//! probability, kept at full width -- so the last two bits of `exp` survive
//! into the comparison; and GELU's `1 + tanh` cancels, which `device_packed`
//! measures at three f32 ulps of one.
//!
//! Where the answer is an INDEX rather than a number -- a router's
//! `expert_ids` -- the comparison is `assert_eq!`, and every id in every one
//! of the seven router cases matched. That is the half of a router that has
//! no tolerance at all: picking the wrong expert is not a small error, it is
//! a different matrix.
//!
//! # The mutations are here too, and they are here for a specific reason
//!
//! The per-point files already sabotage these shaders against their models.
//! What that does not prove is that THIS file's comparison is live: a fixture
//! read through the wrong array name, or a case whose expected output happens
//! to be what an unwritten buffer holds, is a test that passes without ever
//! looking at the device. So one mutation per group is fired here as well,
//! against the fixture rather than against a model.
//!
//! # What has no vector, and why
//!
//! `router_topk`'s `softmax_over_all != 0` arm: no CUDA router takes the
//! softmax over every expert and then selects, so there is nothing to compare
//! it to. `rope_neox_mb` at a rotary narrower than the head: no CUDA kernel
//! pairs `(i, i + rotary/2)` while dividing the exponent by `rotary`, so the
//! `neox_mb_partial` case below was taken by driving CUDA's `rotate` at
//! `head_dim = rotary` over the leading channels of each head -- the same
//! rotation channel for channel, and the one place in the generator where a
//! host repack stands between the two. The fixture's header states it.
//! `gated_delta`, `kda` and the two gemm roads have no vector at all; their
//! evidence is `device_recurrence` and `device_dense`.

#![cfg(target_vendor = "apple")]

mod plane;

use driver_metal::skip::skipped;
use plane::vectors::{Case, Vectors};
use plane::{Arg, Rig};

const FILE_PACKED: &str = "mlp/packed.metal";
const FILE_ROUTE: &str = "moe/route.metal";
const FILE_NEOX: &str = "rope/neox.metal";
const FILE_CONV: &str = "ssm/causal_conv1d.metal";

/// What a case may differ by, as a fraction of the widest element cuda wrote.
///
/// Zero means BIT-EXACT and most of these are zero. The three that are not
/// are the three places the two planes deliberately spell one function two
/// ways, and each says which.
const EXACT: f32 = 0.0;

/// `chunked_gpt_oss_glu` computes `g / (1 + exp(-alpha g))` and
/// `packed_gptoss_swiglu` computes `g * (1 / (1 + fast::exp(-alpha g)))` --
/// a division against a reciprocal-then-multiply, which is one f32 rounding
/// apart before the bf16 store rounds it away.
const GPTOSS_SPELLING: f32 = 1.0 / 4_000_000_000.0;

/// `precise::tanh` against `tanhf`. GELU's `1 + tanh(inner)` cancels for a
/// negative gate -- `device_packed`'s header measures it at three f32 ulps
/// of one -- so a single ulp of disagreement about `tanh` is a third of an
/// answer whose magnitude is a ten-thousandth of the plane's. Measured at
/// `2.05e-6`, which is 0.54 of this.
///
/// SiTU uses the same `precise::tanh` twice and agrees to the bit anyway: it
/// has no cancellation to amplify the disagreement through, which is the
/// whole of the difference between the two activations here.
const TANH_CANCELLATION: f32 = 1.0 / 262_144.0;

/// `fast::exp`, `metal::log` and `metal::sqrt` against `expf`, `log1pf` and
/// `sqrtf`, landing in an f32 output rather than a bf16 one -- which is why
/// the routers are the only group here that does not agree to the bit.
///
/// `sqrt_softplus` is the widest of the five at `2.42e-7`, and the shader's
/// own header predicted it: "`log(1 + exp(x))` and not cuda's
/// `log1pf(expf(x))`: MSL's math library has no `log1p`." Two ulps of one
/// f32, on a weight, from a library function that does not exist on this
/// plane. The softmax sits at `1.58e-7` and the sigmoid router at `8.1e-8`.
const ROUTER_LIBM: f32 = 1.0 / 2_097_152.0;

/// The mutations below have to clear this to prove the comparison is live.
/// It is deliberately not any of the allowances above: a mutation that only
/// just exceeded a case's own allowance would be a weak proof.
const MUTATION_FLOOR: f32 = 1.0 / 256.0;

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_packed_activations_answer_what_cuda_answered() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: the packed activations were not fired against cuda");
        return;
    };
    let vectors = Vectors::read("mlp_packed.txt");
    let root = plane::kernels_dir();

    let mut exact = 0;
    for (case, allow) in [
        ("packed_swiglu", EXACT),
        ("packed_swiglu_clamp", EXACT),
        ("packed_gptoss_swiglu", GPTOSS_SPELLING),
        ("packed_geglu_tanh", TANH_CANCELLATION),
        // SiTU takes two `precise::tanh` and agrees to the bit: there
        // is no cancellation in `beta tanh(g/beta) sigmoid(g)` for an
        // ulp to be amplified through.
        ("packed_situ", EXACT),
        ("packed_situ_uncapped", EXACT),
    ] {
        let fx = vectors.case(case);
        let got = activation(&rig, root.as_path(), case, fx);
        exact += usize::from(agrees(&got, fx.f32s("y"), allow, &format!("cuda/{case}")));
    }
    exactly(exact, 4, "the packed activations");

    // The fixture path itself, sabotaged: if this file were reading the wrong
    // array or comparing a buffer nothing wrote, the mutation would not move
    // the answer.
    let fx = vectors.case("packed_swiglu");
    let mutant = plane::mutant(
        FILE_PACKED,
        "out[row + i] = static_cast<T>((g / (1.0f + metal::exp(-g))) * u);",
        "out[row + i] = static_cast<T>((g / (1.0f + metal::exp(-g))) + u);",
    );
    let bent = activation(&rig, mutant.path(), "packed_swiglu", fx);
    bites(&bent, fx.f32s("y"), "packed_swiglu");
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_routers_pick_the_experts_cuda_picked() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: the routers were not fired against cuda");
        return;
    };
    let vectors = Vectors::read("moe_router.txt");
    let root = plane::kernels_dir();

    // `router_topk` is spelled like CUDA's WARP form -- select on the raw
    // logits, exponentiate only the k -- while the rowed block kernel
    // softmaxes over every expert and renormalises the k out of it. The two
    // are the same function and the fixture carries both, so both are fired
    // and the block form is the harder of the two to agree with.
    let mut exact = 0;
    for case in ["topk_softmax_warp", "topk_softmax_block"] {
        let fx = vectors.case(case);
        let (ids, weights) = softmax(&rig, root.as_path(), fx);
        assert_eq!(ids, fx.i32s("topk_idx"), "cuda/{case} picks these experts");
        exact += usize::from(agrees(
            &weights,
            fx.f32s("topk_w"),
            ROUTER_LIBM,
            &format!("cuda/{case}"),
        ));
    }

    for case in [
        "topk_sigmoid_renorm",
        "topk_sigmoid_plain",
        "topk_sigmoid_fanout",
    ] {
        let fx = vectors.case(case);
        let (ids, weights) = sigmoid(&rig, root.as_path(), fx);
        assert_eq!(ids, fx.i32s("topk_idx"), "cuda/{case} picks these experts");
        exact += usize::from(agrees(
            &weights,
            fx.f32s("topk_w"),
            ROUTER_LIBM,
            &format!("cuda/{case}"),
        ));
    }

    for case in ["topk_sqrtsoftplus_bias", "topk_sqrtsoftplus_plain"] {
        let fx = vectors.case(case);
        let (ids, weights) = softplus(&rig, root.as_path(), fx);
        assert_eq!(ids, fx.i32s("topk_idx"), "cuda/{case} picks these experts");
        exact += usize::from(agrees(
            &weights,
            fx.f32s("topk_w"),
            ROUTER_LIBM,
            &format!("cuda/{case}"),
        ));
    }
    // EVERY EXPERT ID MATCHED and no weight did, which is exactly the shape
    // this group should have: the selection is integer arithmetic over the
    // same logits and cannot differ, while the weight is a transcendental in
    // f32 and cannot help but.
    exactly(exact, 0, "the routers' weights");

    // The published weight, taken with the bias on it -- the defect
    // `route.metal`'s own header names, which moves no id at all.
    let fx = vectors.case("topk_sqrtsoftplus_bias");
    let mutant = plane::mutant(
        FILE_ROUTE,
        "const float w = sqrt_softplus(float(logits[uint(expert_ids[r])]));",
        "const float w = sqrt_softplus(float(logits[uint(expert_ids[r])])) + correction[uint(expert_ids[r])];",
    );
    let (ids, weights) = softplus(&rig, mutant.path(), fx);
    assert_eq!(
        ids,
        fx.i32s("topk_idx"),
        "publishing the biased weight must not move an id"
    );
    bites(&weights, fx.f32s("topk_w"), "router_topk_sqrt_softplus");
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_rotations_answer_what_cuda_rotated() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: the rotations were not fired against cuda");
        return;
    };
    let vectors = Vectors::read("rope_neox.txt");
    let root = plane::kernels_dir();

    let mut exact = 0;
    for case in [
        "neox_mb_full",
        "neox_mb_partial",
        "neox_prop_mb",
        "neox_last_mb",
        "neox_last_mb_interleaved",
        "neox_yarn_mb",
        "neox_yarn_mb_interleaved",
    ] {
        let fx = vectors.case(case);
        let got = rotate(&rig, root.as_path(), case, fx);
        exact += usize::from(agrees(
            &got,
            fx.f32s("q_out"),
            EXACT,
            &format!("cuda/{case}"),
        ));
    }
    // ALL SEVEN, which is the most surprising line in this file. Metal
    // spells the frequency `exp2(-d * log2(theta))` and CUDA spells it
    // `powf(theta, -d)`; Metal turns with `fast::cos`/`fast::sin` and CUDA
    // with `__sincosf`. Four different transcendentals, and the bf16 store
    // rounds every disagreement between them away.
    exactly(exact, 7, "the rotations");

    // The rotation's own sign, against cuda's numbers this time.
    let fx = vectors.case("neox_mb_full");
    let mutant = plane::mutant(
        FILE_NEOX,
        "const float y2 = x1 * sintheta + x2 * costheta;",
        "const float y2 = x1 * sintheta - x2 * costheta;",
    );
    let bent = rotate(&rig, mutant.path(), "neox_mb_full", fx);
    bites(&bent, fx.f32s("q_out"), "neox_mb_bfloat16");
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_convolution_answers_what_cuda_convolved() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: the convolution was not fired against cuda");
        return;
    };
    let vectors = Vectors::read("ssm_conv.txt");
    let root = plane::kernels_dir();

    let mut exact = 0;
    let fx = vectors.case("conv_update");
    let (y, slab) = conv_step(&rig, root.as_path(), fx);
    exact += usize::from(agrees(&y, fx.f32s("y"), EXACT, "cuda/conv_update"));
    // CUDA shifts its slab in place and Metal writes a second plane, so the
    // two hold the same rectangle at the same element -- every value in the
    // fixture is bf16-exact and the copy rounds nothing.
    assert_eq!(
        slab,
        fx.f32s("new_conv_state"),
        "cuda/conv_update leaves this slab, and it is a copy"
    );
    plane::measured("cuda/conv_update, the slab", "bit-exact against cuda");

    for case in ["conv_chunked", "conv_chunked_empty"] {
        let fx = vectors.case(case);
        let (y, slab) = conv_window(&rig, root.as_path(), fx);
        exact += usize::from(agrees(&y, fx.f32s("y"), EXACT, &format!("cuda/{case}")));
        assert_eq!(
            slab,
            fx.f32s("new_conv_state"),
            "cuda/{case} leaves this slab, and it is a copy"
        );
        plane::measured(&format!("cuda/{case}, the slab"), "bit-exact against cuda");
    }

    // The causal window, one token late, against cuda's numbers.
    let fx = vectors.case("conv_chunked");
    let mutant = plane::mutant(
        FILE_CONV,
        "const int src = t - (width - 1) + k;",
        "const int src = t - width + k;",
    );
    exactly(exact, 3, "the convolution");

    let (y, _) = conv_window(&rig, mutant.path(), fx);
    bites(&y, fx.f32s("y"), "causal_conv1d_chunked_bfloat16");
}

// ── the comparisons ─────────────────────────────────────────────────────────

/// The widest disagreement, as a fraction of the widest element cuda wrote.
fn worst(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    plane::worst(got, want, scale.max(f32::MIN_POSITIVE))
}

/// Compare one case, and answer whether it was BIT-EXACT.
///
/// `allow` is zero for most of what follows, which is the whole point: a
/// Metal shader and a CUDA kernel that were never compiled by the same
/// toolchain, never linked against the same libm and never ran on the same
/// vendor's silicon land on the same bits, and the strongest thing this file
/// can say is that they do. Where they do not, the allowance carries the
/// reason beside it -- and every reason in this file is a spelling the two
/// planes chose deliberately and differently.
fn agrees(got: &[f32], want: &[f32], allow: f32, what: &str) -> bool {
    let seen = worst(got, want);
    assert!(
        seen <= allow,
        "{what}: element {} is {seen} of the widest element away from what \
         cuda answered, past the {allow} this case is allowed",
        plane::worst_at(got, want, 1.0)
    );
    plane::measured(
        what,
        &if allow == 0.0 {
            "bit-exact against cuda".to_string()
        } else {
            format!("worst {seen}, allowed {allow}")
        },
    );
    seen == 0.0
}

/// How many of a group's cases agreed to the bit, stated rather than
/// counted afterwards.
///
/// This is the tripwire that keeps the allowances above honest. A hand that
/// turns a bit-exact case into an approximate one -- by widening its
/// allowance, or by changing a shader so that it no longer agrees -- has to
/// come here and say so, and the number is the one thing in this file a
/// reader can check at a glance.
fn exactly(seen: usize, want: usize, group: &str) {
    assert_eq!(
        seen, want,
        "{want} of {group}'s cases were bit-exact against cuda when this was \
         written and {seen} are now"
    );
    plane::measured(group, &format!("{seen} cases bit-exact against cuda"));
}

fn bites(got: &[f32], want: &[f32], symbol: &str) {
    let seen = worst(got, want);
    assert!(
        seen > MUTATION_FLOOR,
        "the mutation moved the answer {seen} of the plane's widest element, \
         which is inside {MUTATION_FLOOR} -- so the comparison against cuda \
         would not have caught it"
    );
    plane::measured(
        symbol,
        &format!("the mutation moves it {seen} against cuda's numbers"),
    );
}

// ── the fires ───────────────────────────────────────────────────────────────

/// One packed activation, at the shape and the scalars the case states.
fn activation(rig: &Rig, root: &std::path::Path, case: &str, fx: &Case) -> Vec<f32> {
    let packed = fx.f32s("packed");
    let i = fx.at("i");
    let rows = packed.len() / (2 * i);
    let (symbol, extra): (&'static str, Vec<Arg<'_>>) = match case {
        "packed_swiglu" => ("packed_swiglu_bfloat16", vec![]),
        "packed_swiglu_clamp" => (
            "packed_swiglu_clamp_bfloat16",
            vec![Arg::F32(fx.f32("limit"))],
        ),
        "packed_gptoss_swiglu" => (
            "packed_gptoss_swiglu_bfloat16",
            vec![Arg::F32(fx.f32("limit")), Arg::F32(fx.f32("alpha"))],
        ),
        "packed_geglu_tanh" => ("packed_geglu_tanh_bfloat16", vec![]),
        "packed_situ" | "packed_situ_uncapped" => (
            "packed_situ_bfloat16",
            // `linear_beta` is what `Mlp::situ` and `packed.metal` call
            // `up_cap`; the uncapped case states zero, which is the branch.
            vec![Arg::F32(fx.f32("beta")), Arg::F32(fx.f32("linear_beta"))],
        ),
        other => panic!("no entry point for case `{other}`"),
    };
    let src = plane::alloc_bf16(&rig.context, packed, "packed");
    let out = plane::alloc_bf16(&rig.context, &vec![0.0; rows * i], "y");
    let mut args = vec![Arg::Buf(&src), Arg::Buf(&out), Arg::U32(i as u32)];
    args.extend_from_slice(&extra);
    plane::fire(
        rig,
        root,
        FILE_PACKED,
        symbol,
        [i as u32, rows as u32, 1],
        [i.min(256) as u32, 1, 1],
        &args,
    );
    plane::read_bf16(&out, rows * i)
}

fn softmax(rig: &Rig, root: &std::path::Path, fx: &Case) -> (Vec<i32>, Vec<f32>) {
    let logits = fx.f32s("logits");
    let (n, k) = (fx.at("num_experts"), fx.at("k"));
    let rows = logits.len() / n;
    let l = plane::alloc_bf16(&rig.context, logits, "logits");
    let ids = plane::alloc_i32(&rig.context, &vec![-7; rows * k], "expert_ids");
    let weights = plane::alloc_f32(&rig.context, &vec![0.0; rows * k], "expert_weights");
    // Slot 3 is `per_expert_scale`, which the unscaled instantiation never
    // dereferences.
    let scale = plane::alloc_bf16(&rig.context, &[1.0], "per_expert_scale");
    plane::fire(
        rig,
        root,
        FILE_ROUTE,
        "router_topk_f32w_bfloat16",
        [lanes(n), rows as u32, 1],
        [lanes(n), 1, 1],
        &[
            Arg::Buf(&l),
            Arg::Buf(&ids),
            Arg::Buf(&weights),
            Arg::Buf(&scale),
            Arg::U32(n as u32),
            Arg::U32(k as u32),
            Arg::U32(0),
            Arg::U32(n as u32),
        ],
    );
    (
        plane::read_i32(&ids, rows * k),
        plane::read_f32(&weights, rows * k),
    )
}

fn sigmoid(rig: &Rig, root: &std::path::Path, fx: &Case) -> (Vec<i32>, Vec<f32>) {
    let logits = fx.f32s("logits");
    let (e, k) = (fx.at("e"), fx.at("k"));
    let rows = logits.len() / e;
    let l = plane::alloc_bf16(&rig.context, logits, "logits");
    let ids = plane::alloc_i32(&rig.context, &vec![-7; rows * k], "expert_ids");
    let weights = plane::alloc_f32(&rig.context, &vec![0.0; rows * k], "expert_weights");
    plane::fire(
        rig,
        root,
        FILE_ROUTE,
        "router_topk_sigmoid",
        [lanes(e), rows as u32, 1],
        [lanes(e), 1, 1],
        &[
            Arg::Buf(&l),
            Arg::Buf(&ids),
            Arg::Buf(&weights),
            Arg::U32(e as u32),
            Arg::U32(k as u32),
            Arg::U32(fx.i32("renormalize") as u32),
            Arg::F32(fx.f32("routed_scaling_factor")),
        ],
    );
    (
        plane::read_i32(&ids, rows * k),
        plane::read_f32(&weights, rows * k),
    )
}

fn softplus(rig: &Rig, root: &std::path::Path, fx: &Case) -> (Vec<i32>, Vec<f32>) {
    let logits = fx.f32s("logits");
    let (e, k) = (fx.at("e"), fx.at("k"));
    let rows = logits.len() / e;
    let l = plane::alloc_bf16(&rig.context, logits, "logits");
    let bias = plane::alloc_f32(&rig.context, fx.f32s("correction_bias"), "correction");
    let ids = plane::alloc_i32(&rig.context, &vec![-7; rows * k], "expert_ids");
    let weights = plane::alloc_f32(&rig.context, &vec![0.0; rows * k], "expert_weights");
    plane::fire(
        rig,
        root,
        FILE_ROUTE,
        "router_topk_sqrt_softplus",
        [lanes(e), rows as u32, 1],
        [lanes(e), 1, 1],
        &[
            Arg::Buf(&l),
            Arg::Buf(&bias),
            Arg::Buf(&ids),
            Arg::Buf(&weights),
            Arg::U32(e as u32),
            Arg::U32(k as u32),
            Arg::U32(fx.i32("renormalize") as u32),
            Arg::F32(fx.f32("routed_scaling_factor")),
        ],
    );
    (
        plane::read_i32(&ids, rows * k),
        plane::read_f32(&weights, rows * k),
    )
}

/// One in-place rotation, at the entry point and the scalars the case states.
fn rotate(rig: &Rig, root: &std::path::Path, case: &str, fx: &Case) -> Vec<f32> {
    let q_in = fx.f32s("q_in");
    let heads = fx.at("num_q_heads");
    let head_dim = fx.at("head_dim");
    let rows = q_in.len() / (heads * head_dim);
    let x = plane::alloc_bf16(&rig.context, q_in, "q");
    let positions = plane::alloc_i32(&rig.context, fx.i32s("positions"), "positions");

    let (symbol, rotary, extra): (&'static str, usize, Vec<Arg<'_>>) = match case {
        "neox_mb_full" | "neox_mb_partial" => (
            "neox_mb_bfloat16",
            fx.at("rotary_dim"),
            vec![
                Arg::F32(fx.f32("scale")),
                Arg::F32(fx.f32("base")),
                Arg::I32(head_dim as i32),
            ],
        ),
        "neox_prop_mb" => (
            "neox_prop_mb_bfloat16",
            fx.at("rotary_dim"),
            vec![
                Arg::F32(fx.f32("scale")),
                Arg::F32(fx.f32("base")),
                Arg::I32(head_dim as i32),
            ],
        ),
        "neox_last_mb" | "neox_last_mb_interleaved" => (
            "neox_last_mb_bfloat16",
            fx.at("rotary_dim"),
            vec![
                Arg::F32(fx.f32("base")),
                Arg::I32(head_dim as i32),
                Arg::I32(fx.i32("interleaved")),
            ],
        ),
        "neox_yarn_mb" | "neox_yarn_mb_interleaved" => (
            "neox_yarn_mb_bfloat16",
            head_dim,
            vec![
                Arg::F32(fx.f32("base")),
                Arg::I32(head_dim as i32),
                Arg::F32(fx.f32("factor")),
                Arg::F32(fx.f32("low_dim")),
                Arg::F32(fx.f32("high_dim")),
                Arg::F32(fx.f32("mscale")),
                Arg::I32(fx.i32("interleaved")),
            ],
        ),
        other => panic!("no entry point for case `{other}`"),
    };

    let mut args = vec![Arg::Buf(&x), Arg::Buf(&positions)];
    args.extend_from_slice(&extra);
    plane::fire(
        rig,
        root,
        FILE_NEOX,
        symbol,
        [(rotary / 2) as u32, heads as u32, rows as u32],
        [(rotary / 2) as u32, 1, 1],
        &args,
    );
    plane::read_bf16(&x, q_in.len())
}

fn conv_step(rig: &Rig, root: &std::path::Path, fx: &Case) -> (Vec<f32>, Vec<f32>) {
    let (r, c, k) = (fx.at("r"), fx.at("c"), fx.at("k"));
    let slab = fx.f32s("conv_state");
    let x = plane::alloc_bf16(&rig.context, fx.f32s("x"), "x");
    let weight = plane::alloc_bf16(&rig.context, fx.f32s("weight"), "conv weight");
    let state = plane::alloc_f32(&rig.context, slab, "conv_state");
    // Seeded with what it reads: `Pool::carry_forward`'s invariant, and what
    // makes an in-place shift and a ping-pong plane the same answer.
    let fresh = plane::alloc_f32(&rig.context, slab, "new_conv_state");
    // The fixture records seats as i32 and the kernel reads `uint`; every
    // seat is non-negative, so the bits are the same number.
    let seats = plane::alloc_i32(&rig.context, fx.i32s("slot_ids"), "slots");
    let y = plane::alloc_bf16(&rig.context, &vec![0.0; r * c], "y");
    plane::fire(
        rig,
        root,
        FILE_CONV,
        "causal_conv1d_bfloat16",
        [c as u32, r as u32, 1],
        [c.min(256) as u32, 1, 1],
        &[
            Arg::Buf(&x),
            Arg::Buf(&weight),
            Arg::Buf(&state),
            Arg::Buf(&fresh),
            Arg::Buf(&seats),
            Arg::Buf(&y),
            Arg::I32(c as i32),
            Arg::I32(k as i32),
        ],
    );
    (
        plane::read_bf16(&y, r * c),
        plane::read_f32(&fresh, slab.len()),
    )
}

fn conv_window(rig: &Rig, root: &std::path::Path, fx: &Case) -> (Vec<f32>, Vec<f32>) {
    let (c, k) = (fx.at("c"), fx.at("k"));
    let indptr = fx.i32s("qo_indptr");
    let requests = indptr.len() - 1;
    let tokens = fx.f32s("x").len() / c;
    let slab = fx.f32s("conv_state");
    let x = plane::alloc_bf16(&rig.context, fx.f32s("x"), "x");
    let csr = plane::alloc_i32(&rig.context, indptr, "indptr");
    let weight = plane::alloc_bf16(&rig.context, fx.f32s("weight"), "conv weight");
    let state = plane::alloc_f32(&rig.context, slab, "conv_state");
    let fresh = plane::alloc_f32(&rig.context, slab, "new_conv_state");
    // One seat per TOKEN here, which is what the chunked kernel indexes by
    // its window's first row.
    let seats = plane::alloc_i32(&rig.context, fx.i32s("slots"), "slots");
    let y = plane::alloc_bf16(&rig.context, &vec![0.0; tokens * c], "y");
    plane::fire(
        rig,
        root,
        FILE_CONV,
        "causal_conv1d_chunked_bfloat16",
        [c as u32, requests as u32, 1],
        [c.min(256) as u32, 1, 1],
        &[
            Arg::Buf(&x),
            Arg::Buf(&csr),
            Arg::Buf(&weight),
            Arg::Buf(&state),
            Arg::Buf(&fresh),
            Arg::Buf(&seats),
            Arg::Buf(&y),
            Arg::I32(c as i32),
            Arg::I32(k as i32),
        ],
    );
    (
        plane::read_bf16(&y, tokens * c),
        plane::read_f32(&fresh, slab.len()),
    )
}

/// One lane per expert, rounded up to whole simdgroups.
fn lanes(experts: usize) -> u32 {
    (experts as u32).min(1024).div_ceil(32) * 32
}
