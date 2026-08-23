pub mod routine;
pub mod views;

pub mod attn;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod ptir;
pub mod quant;
pub mod rope;
pub mod sample;
pub mod ssm;

pub type Plane = crate::routine::Metal;

#[cfg(not(target_family = "wasm"))]
#[::linkme::distributed_slice]
pub static METAL_ROUTINES: [::kernels::routine::Routine<Plane>];

#[cfg(not(target_family = "wasm"))]
pub use METAL_ROUTINES as ROUTINES;

#[cfg(target_family = "wasm")]
#[doc(hidden)]
pub struct Registered(pub ::kernels::routine::Routine<Plane>);

#[cfg(target_family = "wasm")]
::inventory::collect!(Registered);

pub fn rows() -> impl Iterator<Item = &'static ::kernels::routine::Routine<Plane>> {
    #[cfg(not(target_family = "wasm"))]
    {
        ROUTINES.iter()
    }
    #[cfg(target_family = "wasm")]
    {
        ::inventory::iter::<Registered>.into_iter().map(|r| &r.0)
    }
}

mod stamped {
    include!(concat!(env!("OUT_DIR"), "/entrypoints.rs"));
}

pub use stamped::STAMPED;

#[must_use]
pub fn declared() -> Vec<kernels::routine::Declared> {
    rows()
        .map(kernels::routine::Routine::declared)
        .chain(ELSEWHERE.iter().copied())
        .collect()
}

// ── `norm::rms_rope`, DECLARED HERE BECAUSE METAL HAS NO ARM ─────────────
//
// The fused RMS-norm + RoPE is a Vulkan/wgpu routine; Metal only names it, so
// the cross-backend signature checker knows the symbol exists and can hold
// this plane accountable to the same shape the other two backends implement.
// The declaration BELOW MUST TRACK THE `#[routine]` on the real arms
// (see `crates/kernels-vulkan/src/norm.rs::rms_rope` and
// `crates/kernels-wgpu/src/norm.rs::rms_rope`) byte-for-byte, because that is
// what the checker compares against. When the sweep on the two live arms
// widened the parameter list from `(x, w)` to the full RoPE contract, this
// stub was left carrying the old two-operand shape and the validator refused
// with `signature violations in this declaration ... rms_rope_bfloat16`.
//
// # How each column is derived, in the order the macro would emit it
//
// `#[routine]` on
//
//     pub fn rms_rope(
//         ctx: &Ctx<'_>,
//         x: InOut<Tensor<bf16>>,
//         w: Const<Tensor<bf16>>,
//         eps: Const<f32>,
//         axis: Const<i32>,
//         w_stride: Const<u32>,
//         plus_one: Const<u32>,
//         gain: Const<f32>,
//         row_pitch: Const<i32>,
//         rotary: Const<i32>,
//         scale: Const<f32>,
//         base_or_mscale: Const<f32>,
//         positions: In<Tensor<i32>>,
//         rows: Const<i32>,
//     ) -> Result<(), Refusal>
//
// walks the parameters after `ctx` and derives three parallel columns:
//
// * `args` — the `Ty` each parameter's `Arg` impl states. `InOut<Tensor<bf16>>`
//   uses `E::TY_MUT` = `Bf16sMut` (the mark says "writes"); `Const<Tensor<bf16>>`
//   uses `E::TY_CONST` = `Bf16s`; `In<Tensor<i32>>` uses `I32s`; a scalar
//   `Const<f32>` / `Const<i32>` / `Const<u32>` uses `F32` / `I32` / `U32`
//   respectively.
// * `sources` — `resolve()` in `crates/kernels/src/routine.rs` runs FOUR
//   independent counters (ins, outs, weights, params — `Param` and `ParamF32`
//   share one) in signature order. Every `Const<f32>` scalar takes the next
//   params slot as `Slot(ParamF32, n)`; every `Const<i32>`/`Const<u32>` scalar
//   takes it as `Slot(Param, n)`; a `Const<Tensor<E>>` takes the next weight
//   slot; `In` and `Out` take their own counters; and `InOut` claims BOTH ins
//   and outs at once, which the source spells as `Alias(i_at, o_at)`. That is
//   why `x` here becomes `Alias(0, 0)` while `positions` — the second `In`
//   argument, after `x`'s implicit `In` half — is `Slot(In, 1)`.
// * `derived` — the parameter's Rust NAME and whether its type admits a null.
//   None of these parameters are `Option<_>` or `MaybeConst<_>`, so every row
//   is `nullable: false`.
//
// The scalar params-run indices below are 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 in the
// order the parameters appear: `eps` (ParamF32, 0), `axis` (Param, 1),
// `w_stride` (Param, 2), `plus_one` (Param, 3), `gain` (ParamF32, 4),
// `row_pitch` (Param, 5), `rotary` (Param, 6), `scale` (ParamF32, 7),
// `base_or_mscale` (ParamF32, 8), and — AFTER the `In<Tensor<i32>>` positions
// stream, which touches no scalar counter — `rows` (Param, 9). Both readings
// share one counter (the params run is a byte channel and `ParamF32` is only
// how the caller reads a word out of it), which is what makes the indices
// interleave across the two `Kind`s.
//
// `whole` and `depth_prefix_plan` are both false (the real arms carry no such
// attribute), and there is no canon role, so `canon` is `None`.
pub const ELSEWHERE: &[kernels::routine::Declared] = &[kernels::routine::Declared {
    name: "rms_rope",
    namespace: "norm",
    // THE VULKAN/WGPU TWINS' COLUMN, verbatim: this plane has no routine
    // for the fused form and declares the shape so a metal-family trace
    // can state it -- and a declaration two marks short of the twins had
    // `shader_backends_agree` refusing the whole table. The thirteen
    // marks: x (in place), the weight, the nine scalars in the twins'
    // Const order, the positions stream, and the spliced rows.
    args: &[
        kernels::Ty::Bf16sMut,
        kernels::Ty::Bf16s,
        kernels::Ty::F32,
        kernels::Ty::I32,
        kernels::Ty::U32,
        kernels::Ty::U32,
        kernels::Ty::F32,
        kernels::Ty::I32,
        kernels::Ty::I32,
        kernels::Ty::F32,
        kernels::Ty::F32,
        kernels::Ty::I32s,
        kernels::Ty::I32,
    ],
    sources: &[
        Some(kernels::Source::Alias(0, 0)),
        Some(kernels::Source::Slot(kernels::Kind::Weight, 0)),
        Some(kernels::Source::Slot(kernels::Kind::ParamF32, 0)),
        Some(kernels::Source::Slot(kernels::Kind::Param, 1)),
        Some(kernels::Source::Slot(kernels::Kind::Param, 2)),
        Some(kernels::Source::Slot(kernels::Kind::Param, 3)),
        Some(kernels::Source::Slot(kernels::Kind::ParamF32, 4)),
        Some(kernels::Source::Slot(kernels::Kind::Param, 5)),
        Some(kernels::Source::Slot(kernels::Kind::Param, 6)),
        Some(kernels::Source::Slot(kernels::Kind::ParamF32, 7)),
        Some(kernels::Source::Slot(kernels::Kind::ParamF32, 8)),
        Some(kernels::Source::Slot(kernels::Kind::In, 1)),
        Some(kernels::Source::Slot(kernels::Kind::Param, 9)),
    ],
    whole: false,
    depth_prefix_plan: false,
    derived: &[
        kernels::Derived {
            name: "x",
            nullable: false,
        },
        kernels::Derived {
            name: "w",
            nullable: false,
        },
        kernels::Derived {
            name: "eps",
            nullable: false,
        },
        kernels::Derived {
            name: "axis",
            nullable: false,
        },
        kernels::Derived {
            name: "w_stride",
            nullable: false,
        },
        kernels::Derived {
            name: "plus_one",
            nullable: false,
        },
        kernels::Derived {
            name: "gain",
            nullable: false,
        },
        kernels::Derived {
            name: "row_pitch",
            nullable: false,
        },
        kernels::Derived {
            name: "rotary",
            nullable: false,
        },
        kernels::Derived {
            name: "scale",
            nullable: false,
        },
        kernels::Derived {
            name: "base_or_mscale",
            nullable: false,
        },
        kernels::Derived {
            name: "positions",
            nullable: false,
        },
        kernels::Derived {
            name: "rows",
            nullable: false,
        },
    ],
    canon: None,
}];

/// Every point this crate can reach, however it comes to exist.
///
/// TWO SOURCES, because a point is no longer always the file's. `STAMPED` is
/// what `build.rs` reads out of the `.metal` tree, and a family the HOST
/// stamps has left that tree by design -- `quant/qmm_t.metal` holds the
/// template and the `PIE_STAMP_qmm_t` macro, and the names it used to
/// instantiate for itself are composed by [`quant::qmm_point`] at the fire.
/// See [`kernels::routine::Fire::stamp`].
///
/// [`quant::composed`] is the product of the same axes through the same
/// constructor a fire calls, so a name here is a name a fire can reach BY
/// CONSTRUCTION rather than by a list agreeing with one. A family that moves
/// out of its file the same way adds its walker here and nothing else
/// changes.
///
/// NOT [`shaders`], which stays the file's own list: that pair is what a
/// device compiles, and a composed point compiles from its STAMP rather than
/// from the file as it stands. This is the census -- who exists -- and that
/// is the build list.
fn census() -> impl Iterator<Item = &'static str> {
    STAMPED
        .iter()
        .map(|(_, name)| *name)
        .chain(quant::composed().into_iter().map(|(_, name)| name))
}

pub fn entrypoints() -> Vec<String> {
    let mut out: Vec<String> = census().map(str::to_owned).collect();
    out.sort();
    out
}

#[must_use]
pub fn kernel_of(symbol: &str) -> Option<&'static str> {
    static CENSUS: std::sync::OnceLock<std::collections::BTreeSet<&'static str>> =
        std::sync::OnceLock::new();
    let census = CENSUS.get_or_init(|| {
        // [`census`] AND NOT `STAMPED`, and the difference is not cosmetic.
        // `model-ir`'s `kernels::check_plan` asks this function whether a
        // launched symbol is declared, and while the census was the file's
        // list alone every affine GEMM in every `*.metal.*` text answered no
        // -- `llama_like.metal.prefill: launches
        // "affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_32", which no metal
        // kernel declares`, at load, for a point that compiles.
        census().chain(DECLARED_ELSEWHERE.iter().copied()).collect()
    });

    rows()
        .map(|r| r.name)
        .chain(ELSEWHERE.iter().map(|d| d.name))
        .filter(|name| {
            symbol == *name || (census.contains(symbol) && at_word_boundary(symbol, name))
        })
        .max_by_key(|name| name.len())
}

#[must_use]
pub fn shaders() -> Vec<(&'static str, &'static str)> {
    let mut out: Vec<(&str, &str)> = STAMPED.to_vec();
    out.sort_unstable();
    out
}

pub const DECLARED_ELSEWHERE: &[&str] = &["rms_rope_bfloat16"];

fn at_word_boundary(symbol: &str, name: &str) -> bool {
    let mut from = 0;
    while let Some(at) = symbol[from..].find(name) {
        let start = from + at;
        let end = start + name.len();
        let before = start == 0 || symbol.as_bytes()[start - 1] == b'_';
        let after = end == symbol.len() || symbol.as_bytes()[end] == b'_';
        if before && after {
            return true;
        }
        from = start + 1;
    }
    false
}

pub trait RoutineElem: kernels::Elem {}

impl<T: kernels::Elem> RoutineElem for T {}
