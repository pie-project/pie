pub mod points;

pub mod plane;
pub mod points_dispatch;
pub mod views;

pub mod attn;
pub mod dist;
pub mod gemm;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod quant;
pub mod rope;
pub mod ssm;

mod stamped {
    include!(concat!(env!("OUT_DIR"), "/entrypoints.rs"));
}

pub use stamped::STAMPED;

pub const CANON: &[(&str, &str)] = &[];

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
