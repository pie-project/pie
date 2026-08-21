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

#[must_use]
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

pub const ELSEWHERE: &[kernels::routine::Declared] = &[kernels::routine::Declared {
    name: "rms_rope",
    namespace: "norm",
    args: &[kernels::Ty::Bf16sMut, kernels::Ty::Bf16s],
    sources: &[
        Some(kernels::Source::Alias(0, 0)),
        Some(kernels::Source::Slot(kernels::Kind::Weight, 0)),
    ],
    whole: false,
    depth_prefix_plan: false,
    derived: &[
        kernels::Derived { name: "x", nullable: false },
        kernels::Derived { name: "w", nullable: false },
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
