pub mod plane;
pub mod routine;
pub mod views;

pub mod attn;
pub mod dist;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod quant;
pub mod rope;
pub mod ssm;

pub type Plane = crate::routine::Metal;

mod stamped {
    include!(concat!(env!("OUT_DIR"), "/entrypoints.rs"));
}

pub use stamped::STAMPED;

/// The claims this plane answers by SYMBOL rather than by point.
///
/// TWO ROWS, AND EACH ONE IS A RESOLUTION EVERY METAL LANE DEPENDS ON.
/// `model_compiler::sweep::resolve` asks a family's `*_CLAIMS` first and this
/// table second, and exactly two points on this plane are in that position:
/// `layout.embed` through `layout::embed_gather_mb_4bit` and
/// `moe.weighted_sum` through `moe::combine_sorted`. Every other name that
/// used to sit here is a point the claim tables now answer, which is why the
/// routine fold moved no lane; dropping one of THESE drops a resolution from
/// every metal lane that embeds or folds experts, which is all of them.
///
/// They land as claims when the floor carries `Bank<R: Repr>` (the embed's
/// three-operand affine bank) and when a point can state the permutation
/// `moe/route.metal`'s sorted combine reads; both gaps are written up in
/// their impl blocks' headers.
///
/// A TABLE AND NOT A COLUMN ON A ROUTINE ROW. This was the `canon` column of
/// the last two `#[routine]`s in this crate, read through a linkme registry
/// four crates carried so that this one question could be asked of it. The
/// registry, the attribute and its parser are all deleted; two
/// `(claim, symbol)` pairs are what was left of them.
pub const CANON: &[(&str, &str)] = &[
    ("layout.embed", "layout::embed_gather_mb_4bit"),
    ("moe.weighted_sum", "moe::combine_sorted"),
];

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
pub fn shaders() -> Vec<(&'static str, &'static str)> {
    let mut out: Vec<(&str, &str)> = STAMPED.to_vec();
    out.sort_unstable();
    out
}


pub trait RoutineElem: kernels::Elem {}

impl<T: kernels::Elem> RoutineElem for T {}
