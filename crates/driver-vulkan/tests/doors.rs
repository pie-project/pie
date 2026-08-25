//! What a claim body reaches when the doors it asks through actually answer.
//!
//! `kernels_vulkan::points::Staged` declared five doors and implemented every
//! one of them as an unconditional refusal. The consequence was not that five
//! functions were missing. It was that the attention, mlp, moe and layout claim
//! bodies call one of those doors on their FIRST LINE, so those points
//! compiled, counted as answered in the claim census, and refused at run time
//! before binding anything — and the plane read as far more complete than it
//! was. Two independent readers found the same thing and neither could measure
//! it, because measuring it needs something to be on the other side of the
//! door.
//!
//! This file is that something. It is a recording [`Encode`] — no device, no
//! `slangc`, no plan — that answers `staged` and `windowed` the way the real
//! encoder does and writes down every fire a body asks for. What the sweeps
//! below then ask is the only question that distinguishes an answered door
//! from a decorative one: **does the body get past its first line, and is what
//! it binds there what the compiled module declares?**
//!
//! # Why the SPIR-V is the reference and not a transcription
//!
//! An argument list a body states and a binding table a shader declares are two
//! halves of one ABI written in two languages, and nothing but a sweep makes
//! them agree. `tests/rules.rs` opens on the same reasoning and gives it in
//! full. So the assertions here do not say *"`silu_mul_strided_bfloat16` takes
//! three buffers"* — they read the module and ask whether the body's list is
//! the shape the module wants. A body that bound one buffer too few would still
//! be past its first line and would still look answered; this is what says it
//! is not.
//!
//! # No GPU
//!
//! The modules are `kernels-vulkan`'s embedded table, read through this
//! crate's own SPIR-V loader. Nothing here submits anything.

use std::cell::RefCell;

use driver_vulkan::{Declared, spirv};
use kernels::plane::Cache;
use kernels::points::{Attention, Layout, Mlp, Moe, Rope, Ssm};
use kernels_vulkan::Capability;
use kernels_vulkan::plane::{ArgValue, Const, Encode, Fire, In, InOut, Out};
use kernels_vulkan::points::{Handle, bf16};
use kernels_vulkan::views::{AttnFireView, MaskView, PagedKvView, RecurrentView, SplitView};

/// Skip with a reason when there are no modules, rather than pass silently.
///
/// The same guard `tests/rules.rs` uses, for the same reason: a sweep over an
/// empty table is a green run that checked nothing.
macro_rules! modules {
    () => {
        if !kernels_vulkan::embedded() {
            eprintln!(
                "no modules: build with `-p driver-vulkan --features native` \
                 (or any profile that pulls kernels-vulkan/native) and have \
                 `slangc` on PATH"
            );
            return;
        }
    };
}

/// One fire a body asked for.
struct Asked {
    fire: Fire,
    args: Vec<ArgValue>,
}

/// An [`Encode`] that answers the doors and records the fires.
///
/// The two doors are answered the way `driver_vulkan::encode::Encoder` answers
/// them and not more generously, because a fixture that answered everything
/// would prove that a body compiles rather than that it reaches:
///
/// * `staged` holds the five names [`driver_vulkan::binding::FireTable::named`]
///   translates and refuses every other, so `rope.yarn`'s
///   `rope.yarn_inv_freq` is refused here exactly as the driver refuses it.
/// * `windowed` mints a handle out of a base the real encoder also keeps
///   separate (`encode::WINDOW`), and records the byte the window opened at,
///   which is the number the whole `mlp` cut turns on.
///
/// `resolve` refuses: a claim body has no column to resolve, which is the
/// finding the `stream` door's own refusal recorded.
struct Recorder {
    fires: RefCell<Vec<Asked>>,
    windows: RefCell<Vec<(u32, u64)>>,
}

/// Where this fixture's window handles start.
///
/// Not `encode::WINDOW`'s value and deliberately not: what matters is that the
/// two spaces are disjoint, and a fixture that borrowed the constant would pass
/// if the constant were wrong.
const WINDOW: u32 = 1 << 20;

impl Recorder {
    fn new() -> Self {
        Self {
            fires: RefCell::new(Vec::new()),
            windows: RefCell::new(Vec::new()),
        }
    }

    /// The one fire a body was expected to ask for, or a failure naming how
    /// many it asked for instead.
    fn only(&self) -> Asked {
        let mut fires = self.fires.borrow_mut();
        assert_eq!(fires.len(), 1, "a body stated a different number of fires");
        fires.pop().expect("one fire")
    }
}

impl Encode for Recorder {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), kernels::plane::Refusal> {
        self.fires.borrow_mut().push(Asked {
            fire,
            args: args.to_vec(),
        });
        Ok(())
    }

    fn staged(&self, name: &'static str) -> Result<u32, kernels::plane::Refusal> {
        // The five names this driver stages a table for, and nothing else.
        let at = [
            "positions",
            "token_ids",
            "request_of_token",
            "sampling_indices",
            "rope.frequencies",
        ]
        .iter()
        .position(|held| *held == name)
        .ok_or(kernels::plane::Refusal::Unstated {
            what: "a runtime stream this driver stages no table for",
        })?;
        Ok(900 + at as u32)
    }

    fn windowed(&self, of: u32, at: u64) -> Result<u32, kernels::plane::Refusal> {
        let mut windows = self.windows.borrow_mut();
        let n = windows.len();
        windows.push((of, at));
        Ok(WINDOW + n as u32)
    }
}

/// What the module under an entrypoint declares.
fn declared(entrypoint: &str) -> Declared {
    let code = kernels_vulkan::code(entrypoint, Capability::Baseline)
        .unwrap_or_else(|| panic!("`{entrypoint}` is not a module this build produced"));
    let words = spirv::words(code).expect("a built module is whole words");
    spirv::declared(&words).expect("a built module is well formed")
}

/// The two halves an argument list splits into, the way
/// `driver_vulkan::encode` splits it: [`ArgValue::Buffer`] takes a descriptor
/// and every other variant is a word of the scalar block.
fn split(args: &[ArgValue]) -> (usize, usize) {
    let buffers = args
        .iter()
        .filter(|a| matches!(a, ArgValue::Buffer { .. }))
        .count();
    (buffers, args.len() - buffers)
}

/// Hold one recorded fire against the module it names.
///
/// The descriptor count is `bindings` minus the holes, which is the number
/// `encode::fire` itself checks against — `slangc` drops the declaration of a
/// buffer a variant never reads, so the count of decorated slots and one past
/// the highest are different numbers and using the wrong one is a crash.
///
/// The scalar count is the number of MEMBERS the push block declares. Every
/// scalar this plane passes is four bytes wide, so the block's offsets are
/// `0, 4, 8, ...` and a member count that matches is a run that lands on its
/// own fields; the offsets are checked rather than assumed.
fn matches_module(asked: &Asked, entrypoint: &str) {
    assert_eq!(
        asked.fire.entrypoint, entrypoint,
        "the body fired a different entrypoint"
    );
    let d = declared(entrypoint);
    let (buffers, scalars) = split(&asked.args);
    assert_eq!(
        buffers,
        d.bindings as usize - d.holes(),
        "{entrypoint}: the body binds {buffers} buffers and the module \
         decorates {} slots",
        d.bindings as usize - d.holes()
    );
    assert_eq!(
        scalars,
        d.push_offsets.len(),
        "{entrypoint}: the body states {scalars} scalars and the module's push \
         block has {} members",
        d.push_offsets.len()
    );
    let packed: Vec<u32> = (0..scalars as u32).map(|i| i * 4).collect();
    assert_eq!(
        d.push_offsets, packed,
        "{entrypoint}: the module's push members are not at the offsets a run \
         of four-byte scalars lands on, so a body packing end to end would \
         write a field into its neighbour"
    );
    // A grid of zero on any axis is `vkCmdDispatch(0, ..)`, which is legal
    // Vulkan that runs nothing and reports success. `encode::fire` refuses it;
    // a body that computed one has already lost.
    assert!(
        !asked.fire.lanes.contains(&0),
        "{entrypoint}: the body asked for a grid with a zero axis"
    );
}

/// A rectangle of `rows x width` at handle `h`.
fn input(h: u32, rows: i32, width: i32) -> In<Handle<bf16>> {
    In {
        ptr: Handle::new(h),
        rows,
        width,
    }
}

fn output(h: u32, rows: i32, width: i32) -> Out<Handle<bf16>> {
    Out {
        ptr: Handle::new(h),
        rows,
        width,
    }
}

/// THE DOOR THAT WAS THE WHOLE MLP FAMILY'S FIRST LINE.
///
/// `mlp::halves` cuts a packed `[gate | up]` row, and the cut used to be
/// `ctx.window(..)` against a refusal that said *"a descriptor names a whole
/// allocation"*. That was never true of this driver — `device::Bound` is a
/// buffer, an offset AND a range — and the fix is not only opening the window.
/// A packed row's second half is the second half of EVERY row, so the base
/// address is half the answer and the PITCH is the other half: the body has to
/// fire an entrypoint that takes one.
///
/// So this asserts three things at once, and the third is the one that would
/// have been silently wrong: the window opens at the intermediate width in
/// BYTES, the entrypoint is the strided arm, and its argument list is the
/// shape that arm declares.
#[test]
fn the_mlp_cut_opens_a_window_and_fires_the_arm_that_takes_a_pitch() {
    modules!();
    for (point, entrypoint) in [
        ("mlp.swiglu", "silu_mul_strided_bfloat16"),
        ("mlp.geglu_tanh_packed", "geglu_tanh_strided_bfloat16"),
    ] {
        let rec = Recorder::new();
        let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
        // 8 rows of `[gate | up]` at an intermediate width of 128.
        let packed = input(0, 8, 256);
        let y = output(1, 8, 128);
        let fired = if point == "mlp.swiglu" {
            Mlp::swiglu::<bf16>(ctx, packed, 128, y)
        } else {
            Mlp::geglu_tanh_packed::<bf16>(ctx, packed, 128, y)
        };
        fired.unwrap_or_else(|e| panic!("{point} refused past its cut: {e}"));

        assert_eq!(
            *rec.windows.borrow(),
            vec![(0, 256)],
            "{point}: the window did not open on the packed operand at the \
             intermediate width in bytes"
        );
        let asked = rec.only();
        matches_module(&asked, entrypoint);
        // Gate is the operand itself and up is the window: two descriptors onto
        // one allocation, which is the whole content of the door.
        assert_eq!(
            asked.args[0],
            ArgValue::Buffer {
                handle: 0,
                writes: false,
                rows: 0,
                width: 0
            },
            "{point}: the gate half is not the packed operand"
        );
        assert_eq!(
            asked.args[1],
            ArgValue::Buffer {
                handle: WINDOW,
                writes: false,
                rows: 0,
                width: 0
            },
            "{point}: the up half is not the window that was opened"
        );
        // `{ width, rows, gate_pitch, up_pitch, out_pitch }`: both halves are
        // read at the PACKED pitch and the result is written at its own.
        assert_eq!(
            &asked.args[3..],
            &[
                ArgValue::I32(128),
                ArgValue::I32(8),
                ArgValue::I32(256),
                ArgValue::I32(256),
                ArgValue::I32(128),
            ],
            "{point}: the pitches are not the ones the cut states"
        );
        // Flat over `rows * width`, which is what both arms index by.
        assert_eq!(asked.fire.lanes, [128 * 8, 1, 1], "{point}: the grid moved");
    }
}

/// THE FIVE THINGS AN SDPA ARM READS THAT ITS POINT DOES NOT DECLARE.
///
/// `attention.decode` declares `q`, the pool row, `window`, `head_dim`,
/// `sm_scale` and `o`. It also reads the positions table, the request-of-token
/// table, the mask triple, the split policy and the pool's KV head count, and
/// every one of those used to be asked for through a door that refused —
/// `ctx.stream("positions")` on the body's first line.
///
/// `driver-wgpu` had already settled the answer and this plane takes it: the
/// pool MARK carries the whole fire. So the door is not answered here, it is
/// retired, and what this asserts is that retiring it works — the body reaches
/// its fire and binds the ten descriptors the module decorates.
#[test]
fn the_attention_arms_read_their_fire_off_the_pool_mark() {
    modules!();
    let rec = Recorder::new();
    let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
    let view = fire_view(1);
    let fired = Attention::decode::<bf16>(
        ctx,
        input(0, 4, 512),
        Cache {
            ptr: std::ptr::from_ref(&view),
        },
        0,
        128,
        0.088_388_35,
        output(1, 4, 512),
    );
    fired.unwrap_or_else(|e| panic!("attention.decode refused past its facts: {e}"));
    let asked = rec.only();
    matches_module(&asked, "sdpa_paged_decode_bfloat16_d_128");
    // Four query heads of 128 against one KV head is the gqa the body derives,
    // and it is derived from a number `pool_heads` used to refuse outright.
    assert!(
        asked.args.contains(&ArgValue::I32(4)),
        "the body did not state the gqa its pool implies"
    );
}

/// The same mark, on the arm that appends rather than attends.
///
/// `attention.kv_append` needs the pool's `(kv_heads, head_dim)` and nothing
/// else off the fire, and it is the body that shows the retired door was
/// carrying a real fact rather than a formality: it checks the appended row
/// against `kv_heads * head_dim` and refuses a mismatch, which it could not do
/// while `pool_heads` answered `Unstated`.
#[test]
fn the_append_arm_checks_its_row_against_a_head_geometry_it_can_now_read() {
    modules!();
    let rec = Recorder::new();
    let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
    let view = fire_view(1);
    let cache = Cache {
        ptr: std::ptr::from_ref(&view),
    };
    Attention::kv_append::<bf16>(ctx, input(0, 4, 128), input(1, 4, 128), cache)
        .expect("a row of one 128-wide kv head appends");
    matches_module(&rec.only(), "kv_append_paged_bfloat16");

    // And the mismatch it can now catch.
    let rec = Recorder::new();
    let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
    let wrong = Attention::kv_append::<bf16>(ctx, input(0, 4, 256), input(1, 4, 256), cache);
    assert_eq!(
        wrong,
        Err(kernels::plane::Refusal::Narrow {
            what: "the appended key row, against the pool's head geometry",
            at: 256,
        }),
        "a row twice the pool's width was accepted"
    );
}

/// THE DOORS THAT STILL REFUSE, AND WHAT EACH ONE IS WAITING FOR.
///
/// A refusal that has been measured is worth as much as an answer and is worth
/// nothing at all if it drifts, so each of these is held against the sentence
/// it now gives. What they have in common is that none of them is a door this
/// plane forgot to open — each names something outside the door: a table the
/// driver does not stage, a mark the point does not declare, a permutation
/// nothing writes, an entrypoint the shader tree does not stamp.
#[test]
fn the_survivors_name_what_is_missing_rather_than_what_was_not_written() {
    modules!();
    let rec = Recorder::new();
    let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;

    // `rope.frequencies` is a table this driver stages; `rope.yarn_inv_freq` is
    // not, and it is not a spelling mistake for it — one is llama-3's
    // piecewise-in-wavelength rescale and the other is YaRN's ramp. The door
    // works and the table is absent, which is why this refusal comes out of the
    // driver's own name table rather than out of the plane.
    let refused = Rope::yarn::<bf16>(
        ctx,
        InOut {
            ptr: Handle::new(0),
            rows: 4,
            width: 512,
        },
        InOut {
            ptr: Handle::new(1),
            rows: 4,
            width: 512,
        },
        In {
            ptr: Handle::new(2),
            rows: 4,
            width: 1,
        },
        128,
        10000.0,
        1.0,
        32.0,
        1.0,
        1.0,
        4096,
        false,
    );
    assert_eq!(
        refused,
        Err(kernels::plane::Refusal::Unstated {
            what: "a runtime stream this driver stages no table for",
        }),
        "rope.yarn refused for a reason other than the missing YaRN ladder"
    );
    assert!(rec.fires.borrow().is_empty(), "rope.yarn fired anyway");

    // One address where three planes are read. The point marks this operand
    // `Const<Tensor<T>>`, and the floor HAS a mark for a weight that crosses in
    // planes — `Const<Bank<R>>`, which `moe.matmul_select_bias` uses — so this
    // is a statement about `crates/kernels`, not about this plane.
    let bank = Layout::embed::<bf16>(
        ctx,
        In {
            ptr: Handle::new(0),
            rows: 4,
            width: 1,
        },
        Const::new(Handle::new(1)),
        32000,
        output(2, 4, 512),
    );
    assert!(
        matches!(bank, Err(kernels::plane::Refusal::Unstated { what }) if what.contains("Const<Bank<R>>")),
        "layout.embed refused for a reason other than the single-plane mark: {bank:?}"
    );

    // A permutation nothing on this plane writes. `combine_sorted` folds
    // through `inv[row * top_k + e]` and `route_sort` is what fills one; the
    // claim census has no point that fires it. A scratch door would hand this
    // body uninitialised memory and the fold would report success.
    let inv = Moe::weighted_sum::<bf16>(
        ctx,
        input(0, 8, 512),
        In {
            ptr: Handle::new(1),
            rows: 8,
            width: 1,
        },
        output(2, 4, 512),
    );
    assert!(
        matches!(inv, Err(kernels::plane::Refusal::Unstated { what }) if what.contains("route_sort")),
        "moe.weighted_sum refused for a reason other than the unwritten \
         permutation: {inv:?}"
    );

    // The window opens; the entrypoint cannot read it. `gptoss_swiglu_bfloat16`
    // indexes gate, up and out by one flat id, so the pitch has nowhere to go —
    // and `gated.slang` stamps no strided gpt-oss arm the way it stamps a
    // strided geglu and a strided silu.
    let gptoss =
        Mlp::swiglu_clamp_alpha::<bf16>(ctx, input(0, 8, 256), 128, 7.0, 1.702, output(1, 8, 128));
    assert!(
        matches!(gptoss, Err(kernels::plane::Refusal::Absent { what }) if what.contains("no strided gpt-oss arm")),
        "mlp.swiglu_clamp_alpha refused for a reason other than the missing \
         arm: {gptoss:?}"
    );
    // The flat arm is still what a caller with two SEPARATE halves fires, and
    // the module is still there — the refusal is about the packed row, not
    // about the entrypoint being absent.
    assert_eq!(declared("gptoss_swiglu_bfloat16").push_offsets, vec![0, 4]);
}

/// A pool row and the per-fire planes an sdpa arm reads, at one KV head of 128.
fn fire_view(kv: u32) -> AttnFireView {
    AttnFireView {
        kv: PagedKvView {
            keys: kernels::shader::Tensor::new(10),
            values: kernels::shader::Tensor::new(11),
            page_indices: kernels::shader::Tensor::new(12),
            page_indptr: kernels::shader::Tensor::new(13),
            write_page: kernels::shader::Tensor::new(14),
            write_offset: kernels::shader::Tensor::new(15),
            page_size: 16,
            seq_stride: kernels::shader::Usize(128),
            head_stride: kernels::shader::Usize(128),
        },
        positions: kernels::shader::Tensor::new(16),
        request_of_token: kernels::shader::Tensor::new(17),
        mask: MaskView {
            mask: kernels::shader::Tensor::new(18),
            enabled: kernels::shader::Tensor::new(19),
            stride: 0,
        },
        split: SplitView {
            partials: kernels::shader::Tensor::new(20),
            splits: 1,
        },
        kv_heads: kv as i32,
        head_dim: 128,
    }
}

/// A rectangle of `rows x width` at handle `h`, at an element the caller names.
///
/// The bf16 [`input`]/[`output`] pair above cannot serve the ssm family: three
/// of its five points state an `f32` operand or result beside their bf16 ones,
/// and `gates` in particular is the operand whose ELEMENT is the difference
/// between the packed row `ssm.gdn_prep` writes and the compact planes the
/// staged shaders keep.
fn read<T: kernels::points::Scalar>(h: u32, rows: i32, width: i32) -> In<Handle<T>> {
    In {
        ptr: Handle::new(h),
        rows,
        width,
    }
}

fn write<T: kernels::points::Scalar>(h: u32, rows: i32, width: i32) -> Out<Handle<T>> {
    Out {
        ptr: Handle::new(h),
        rows,
        width,
    }
}

/// The three slabs and the slot table an ssm point reads off its cache mark.
///
/// `driver_vulkan::baker::views::recurrent` builds exactly this out of
/// `Pools::slab`, and answers `None` for every layer today because nothing in
/// the driver allocates one. So this fixture is not standing in for a pool that
/// exists -- it is standing in for one that does not, which is why the bodies
/// below can be measured at all.
fn recurrent_view() -> RecurrentView {
    RecurrentView {
        state: kernels::shader::Tensor::new(30),
        slots: kernels::shader::Tensor::new(31),
        conv_state: kernels::shader::Tensor::new(32),
        new_conv_state: kernels::shader::Tensor::new(33),
    }
}

/// THE FIVE SSM POINTS BIND WHAT THEIR MODULES DECLARE.
///
/// Every one of them reaches a fire -- none refuses on a door -- and every one
/// binds the descriptor count and packs the scalar run its `.slang` stamps.
/// That is the whole ABI, and it is two halves written in two languages: the
/// argument list here and the binding table there.
///
/// The grids are asserted too, because a body's `Fire::apply` states LANES and
/// the driver divides by the module's own `[numthreads]`. A conv over 130
/// channels is 130 lanes and three workgroups; a scan is one workgroup on x
/// with the value head on y and the token or request on z.
#[test]
fn every_ssm_point_this_plane_claims_binds_what_its_module_declares() {
    modules!();
    let view = recurrent_view();
    let state = Cache {
        ptr: std::ptr::from_ref(&view),
    };

    // `ssm.causal_conv1d`: 130 channels of 3 rows, four taps.
    let rec = Recorder::new();
    let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
    Ssm::causal_conv1d::<bf16>(
        ctx,
        input(0, 3, 130),
        Const::new(Handle::new(1)),
        state,
        4,
        output(2, 3, 130),
    )
    .expect("ssm.causal_conv1d reaches its fire");
    let asked = rec.only();
    matches_module(&asked, "causal_conv1d_bfloat16");
    assert_eq!(asked.fire.lanes, [130, 3, 1], "the conv's grid moved");
    assert_eq!(
        &asked.args[6..],
        &[ArgValue::I32(130), ArgValue::I32(4)],
        "the conv states its channel row and its tap count, in that order"
    );

    // `ssm.causal_conv1d_chunked`: the same row, over a CSR of three requests.
    let rec = Recorder::new();
    let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
    Ssm::causal_conv1d_chunked::<bf16>(
        ctx,
        input(0, 7, 130),
        read::<i32>(1, 3, 1),
        Const::new(Handle::new(2)),
        state,
        4,
        output(3, 7, 130),
    )
    .expect("ssm.causal_conv1d_chunked reaches its fire");
    let asked = rec.only();
    matches_module(&asked, "causal_conv1d_chunked_bfloat16");
    assert_eq!(
        asked.fire.lanes,
        [130, 3, 1],
        "the chunked conv covers REQUESTS on y and not tokens"
    );

    // `ssm.gdn_prep`: ONE packed operand, ONE packed result, ONE launch, and
    // `v_heads` read off half the operand's width rather than restated.
    let rec = Recorder::new();
    let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
    Ssm::gdn_prep::<bf16>(
        ctx,
        input(0, 5, 74),
        Const::new(Handle::new(1)),
        Const::new(Handle::new(2)),
        write::<f32>(3, 5, 74),
    )
    .expect("ssm.gdn_prep reaches its fire");
    let asked = rec.only();
    matches_module(&asked, "gdn_ba_gates_bfloat16");
    assert_eq!(
        asked.fire.lanes,
        [37, 5, 1],
        "the gate row is one lane per VALUE HEAD and not per element of the \
         packed row"
    );
    assert_eq!(
        &asked.args[4..],
        &[ArgValue::I32(37)],
        "the seam is the one number this kernel is told, and it is half the \
         operand's width"
    );

    // `ssm.gated_delta` and its chunked twin.
    let rec = Recorder::new();
    let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
    Ssm::gated_delta::<bf16>(
        ctx,
        input(0, 3, 576),
        input(1, 3, 192),
        read::<f32>(2, 3, 8),
        state,
        2,
        4,
        96,
        48,
        write::<f32>(3, 3, 192),
    )
    .expect("ssm.gated_delta reaches its fire");
    let asked = rec.only();
    matches_module(&asked, "gated_delta_bfloat16");
    assert_eq!(
        asked.fire.lanes,
        [128, 4, 3],
        "the scan is one workgroup per (value head, token)"
    );
    assert_eq!(
        &asked.args[5..],
        &[
            ArgValue::I32(2),
            ArgValue::I32(4),
            ArgValue::I32(96),
            ArgValue::I32(48)
        ],
        "the scan states the four head numbers in the order its push block \
         declares them"
    );

    let rec = Recorder::new();
    let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
    Ssm::gated_delta_chunked::<bf16>(
        ctx,
        input(0, 7, 576),
        read::<i32>(1, 3, 1),
        input(2, 7, 192),
        read::<f32>(3, 7, 8),
        state,
        2,
        4,
        96,
        48,
        write::<f32>(4, 7, 192),
    )
    .expect("ssm.gated_delta_chunked reaches its fire");
    let asked = rec.only();
    matches_module(&asked, "gated_delta_chunked_bfloat16");
    assert_eq!(
        asked.fire.lanes,
        [128, 4, 3],
        "the chunked scan covers REQUESTS on z and not tokens"
    );
}

/// THE PACKED ROW IS CUT BY THE KERNEL AND NEVER BY THE EXECUTOR.
///
/// `ssm.gdn_prep` gets ONE `Encode::fire` and opens NO window. That is the
/// dense-rectangles rule as this plane can state it: `mlp.swiglu` above cuts a
/// packed `[gate | up]` row with `ctx.window(..)` because its two halves are two
/// OPERANDS of one kernel, and this does not, because `[b | a]` is one operand
/// whose seam the kernel is told.
///
/// Both facts are asserted and not one: a body that fired twice over two halves
/// would still bind the right descriptors on each fire, and `matches_module`
/// alone would pass it.
#[test]
fn the_gdn_prep_packing_is_never_cut_by_a_window_or_a_second_fire() {
    modules!();
    let rec = Recorder::new();
    let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
    Ssm::gdn_prep::<bf16>(
        ctx,
        input(0, 5, 74),
        Const::new(Handle::new(1)),
        Const::new(Handle::new(2)),
        write::<f32>(3, 5, 74),
    )
    .expect("ssm.gdn_prep reaches its fire");
    assert!(
        rec.windows.borrow().is_empty(),
        "the packed `[b | a]` row was cut by the executor"
    );
    assert_eq!(
        rec.fires.borrow().len(),
        1,
        "the packed `[b | a]` row was answered by more than one launch"
    );
    // And the operand the kernel is handed is the whole rectangle, at the
    // handle the statement named.
    let asked = rec.only();
    assert_eq!(
        asked.args[0],
        ArgValue::Buffer {
            handle: 0,
            writes: false,
            rows: 0,
            width: 0
        },
        "the operand bound is not the one the point declared"
    );

    // An odd `[b | a]` row does not halve into value heads, and the body says
    // so rather than rounding.
    let rec = Recorder::new();
    let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
    let odd = Ssm::gdn_prep::<bf16>(
        ctx,
        input(0, 5, 75),
        Const::new(Handle::new(1)),
        Const::new(Handle::new(2)),
        write::<f32>(3, 5, 75),
    );
    assert_eq!(
        odd,
        Err(kernels::plane::Refusal::Narrow {
            what: "the `[b | a]` projection's row, which halves into the value heads",
            at: 75,
        }),
        "an odd projection row was accepted"
    );
}

/// THE TWO SSM POINTS THIS PLANE DOES NOT CLAIM SAY SO BY NAME.
///
/// `ssm.kda_step` and `ssm.kda_chunked` are NOT in the `#[claims]` block, so
/// they fall through to the default body `#[points]` writes and refuse
/// `Refusal::unclaimed`. That is the difference between a backlog and the
/// defect this file opens on: a point written INTO the block as a refusal would
/// be in `SSM_CLAIMS`, would count in `model_ir::kernels::point_claims`, and
/// would make this plane read as answering a scan it does not have.
///
/// So the census is asserted beside the refusal -- five names and not seven.
#[test]
fn the_kimi_pair_is_unclaimed_rather_than_claimed_and_refusing() {
    let view = recurrent_view();
    let state = Cache {
        ptr: std::ptr::from_ref(&view),
    };
    let rec = Recorder::new();
    let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
    let refused = Ssm::kda_step::<bf16>(
        ctx,
        input(0, 3, 288),
        input(1, 3, 96),
        input(2, 3, 2),
        Const::new(Handle::new(3)),
        Const::new(Handle::new(4)),
        state,
        2,
        48,
        1e-6,
        write::<f32>(5, 3, 96),
    );
    assert_eq!(
        refused,
        Err(kernels::plane::Refusal::unclaimed("ssm.kda_step")),
        "ssm.kda_step refused for a reason other than being unclaimed"
    );
    assert!(rec.fires.borrow().is_empty(), "ssm.kda_step fired anyway");

    assert_eq!(
        kernels_vulkan::ssm::SSM_CLAIMS,
        [
            "ssm.causal_conv1d",
            "ssm.causal_conv1d_chunked",
            "ssm.gdn_prep",
            "ssm.gated_delta",
            "ssm.gated_delta_chunked",
        ],
        "the ssm claim census is not the five points this plane stamps a \
         shader for"
    );
    assert!(
        model_ir::kernels::point_claims(model_ir::kernels::Backend::Vulkan)
            .contains(&"ssm.gdn_prep"),
        "`SSM_CLAIMS` is not joined into the table a lane resolves against"
    );
    assert!(
        !model_ir::kernels::point_claims(model_ir::kernels::Backend::Vulkan)
            .contains(&"ssm.kda_step"),
        "an unclaimed point reached the claim table anyway"
    );
}

/// A SCAN HANDED NO CARRY REFUSES RATHER THAN BINDING NOTHING.
///
/// `Pools::slab` answers `None` for every layer on this driver, and its own doc
/// says why that must refuse: *"a scan handed a null carry answers fluently and
/// wrongly"*. This is the other side of that sentence -- the body checks the
/// mark before it binds anything, so the refusal names the view rather than
/// arriving as a descriptor built on a null handle.
#[test]
fn a_recurrent_point_handed_no_carry_names_the_view_it_wanted() {
    modules!();
    let rec = Recorder::new();
    let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
    let refused = Ssm::causal_conv1d::<bf16>(
        ctx,
        input(0, 3, 130),
        Const::new(Handle::new(1)),
        Cache {
            ptr: std::ptr::null(),
        },
        4,
        output(2, 3, 130),
    );
    assert_eq!(
        refused,
        Err(kernels::plane::Refusal::Null {
            what: "the recurrent view this statement names",
        }),
        "a null carry was bound rather than refused"
    );
    assert!(
        rec.fires.borrow().is_empty(),
        "the conv fired without a carry"
    );
}

/// THE SIX POINTS OF THE GEMM/LAYOUT/NORM WAVE, HELD AGAINST THEIR MODULES.
///
/// `tests/device_gemm.rs` fires five entrypoints and checks the NUMBERS. What
/// it cannot check is that the claim BODIES ask for those entrypoints with the
/// argument lists the modules declare — it builds its own push blocks and binds
/// its own buffers, because a device test that went through a claim body would
/// need an encoder with a device behind it.
///
/// So this is the other half, and it is the half that catches an ABI drift: the
/// real bodies run, the [`Recorder`] writes down what each asked for, and
/// [`matches_module`] holds the ask against the compiled SPIR-V. A body binding
/// one buffer too few or packing a scalar into its neighbour's field is past
/// its first line, looks answered, and would still be wrong.
///
/// THE THREE GEMM POINTS ARE ONE ARITHMETIC and are checked as three anyway.
/// `lm_head` and `attention_landing` forward to `matmul` on this plane exactly
/// as they do on `kernels-cuda`, so what is being asserted is that the
/// forwarding is REAL — that all three reach a module rather than two of them
/// reaching a stub — and, for `attention_landing`, that its extra `layer`
/// operand does not become a fourth push word.
#[test]
fn the_gemm_layout_and_norm_wave_binds_what_its_modules_declare() {
    modules!();
    use kernels::points::{Gemm, Norm};

    // Both gemm arms, selected the way the body selects them: the tile at or
    // above `TILE_M` rows and the vector arm below it.
    for (rows, entrypoint) in [
        (64, "dense_gemm_t_bfloat16_bm_32_bn_32"),
        (
            kernels_vulkan::gemm::TILE_M,
            "dense_gemm_t_bfloat16_bm_32_bn_32",
        ),
        (kernels_vulkan::gemm::TILE_M - 1, "dense_gemv_t_bfloat16"),
        (1, "dense_gemv_t_bfloat16"),
    ] {
        for point in ["matmul", "lm_head", "attention_landing"] {
            let rec = Recorder::new();
            let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
            let act = input(0, rows, 512);
            let w = Const::new(Handle::new(1));
            let y = output(2, rows, 47);
            let fired = match point {
                "matmul" => Gemm::matmul::<bf16>(ctx, act, w, y),
                "lm_head" => Gemm::lm_head::<bf16>(ctx, act, w, y),
                _ => Gemm::attention_landing::<bf16>(ctx, act, w, 3, y),
            };
            fired.unwrap_or_else(|e| panic!("gemm.{point} at {rows} rows refused: {e}"));
            let asked = rec.only();
            matches_module(&asked, entrypoint);
            // `layer` is a statement TAG and not an operand: the generated
            // dispatch reads it off `Op::layer`, and a body that pushed it
            // would give this module a fourth scalar it does not declare.
            let (buffers, scalars) = split(&asked.args);
            assert_eq!(buffers, 3, "gemm.{point}: act, w and y");
            assert_eq!(
                scalars, 3,
                "gemm.{point}: m, n and k -- `layer` is the statement's tag, \
                 not a fourth word"
            );
        }
    }

    // `layout.split_rows`: 8 rows of 461 cut at 197, so neither half is a
    // multiple of the workgroup and neither is even.
    {
        let rec = Recorder::new();
        let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
        Layout::split_rows::<bf16>(
            ctx,
            input(0, 8, 461),
            197,
            output(1, 8, 197),
            output(2, 8, 264),
        )
        .expect("layout.split_rows");
        matches_module(&rec.only(), "split_rows_bfloat16");
    }

    // `layout.select`: one layer of a 7-layer relay.
    {
        let rec = Recorder::new();
        let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
        Layout::select::<bf16>(ctx, input(0, 8, 7 * 53), 5, 53, output(1, 8, 53))
            .expect("layout.select");
        matches_module(&rec.only(), "select_slice_bfloat16");
    }

    // `norm.mul_scalar`: the STATED arm, which is where the bindings renumber.
    // `norm.scale` reads its factor from binding 1 and writes at 2; with the
    // factor on the push range there is no binding 1 to leave empty, so the
    // output moves down. A body that kept `scale`'s three-buffer list would
    // bind one descriptor more than this module decorates.
    {
        let rec = Recorder::new();
        let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;
        Norm::mul_scalar::<bf16>(
            ctx,
            45.254_834,
            InOut {
                ptr: Handle::new(0),
                rows: 8,
                width: 461,
            },
        )
        .expect("norm.mul_scalar");
        let asked = rec.only();
        matches_module(&asked, "layer_scalar_mul_stated_bfloat16");
        let (buffers, scalars) = split(&asked.args);
        assert_eq!(buffers, 2, "the read view and the write view of one handle");
        assert_eq!(scalars, 1, "the factor, on the push range");
    }
}

/// THE TWO CUTS REFUSE A RECTANGLE THAT DOES NOT ADD UP, and say which.
///
/// Both bodies derive their extents from the OUTPUTS and check them against
/// the source, which is the only place a caller's mistake can still be caught:
/// once a fire is recorded, a cut that ran off the end is a copy nothing
/// reports. These are the refusals `kernels-cuda` and `kernels-metal` state in
/// the same words, so a plane drifting from them is drifting from the point's
/// meaning and not only from a message.
#[test]
fn the_two_cuts_refuse_a_rectangle_that_does_not_add_up() {
    modules!();
    let rec = Recorder::new();
    let ctx: &kernels_vulkan::plane::Ctx<'_> = &rec;

    // Halves that do not sum to the row they divide.
    let short = Layout::split_rows::<bf16>(
        ctx,
        input(0, 8, 461),
        197,
        output(1, 8, 197),
        output(2, 8, 200),
    );
    assert!(
        matches!(short, Err(kernels::plane::Refusal::Narrow { what, .. })
                 if what.contains("against the row they divide")),
        "split_rows accepted halves that do not sum to the row: {short:?}"
    );

    // A left half that is not the width the cut states.
    let stated = Layout::split_rows::<bf16>(
        ctx,
        input(0, 8, 461),
        196,
        output(1, 8, 197),
        output(2, 8, 264),
    );
    assert!(
        matches!(stated, Err(kernels::plane::Refusal::Narrow { what, .. })
                 if what.contains("the width this cut states")),
        "split_rows accepted a left half other than the stated width: {stated:?}"
    );

    // A layer whose slice runs off the end of the relayed row. 7 layers of 53
    // is 371 wide, so layer 7 begins exactly at the end.
    let past = Layout::select::<bf16>(ctx, input(0, 8, 7 * 53), 7, 53, output(1, 8, 53));
    assert!(
        matches!(past, Err(kernels::plane::Refusal::Narrow { what, .. })
                 if what.contains("does not reach this layer's slice")),
        "select accepted a layer past the end of the relayed row: {past:?}"
    );

    assert!(
        rec.fires.borrow().is_empty(),
        "a refused cut fired anyway, which is the one outcome that cannot be \
         reported downstream"
    );
}
