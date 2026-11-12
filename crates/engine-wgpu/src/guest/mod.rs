//! The shared guest forms ([`engine_gpu::guest`]) bound to WebGPU: a
//! stage's WGSL is what the pipeline takes, its constant infinities read
//! from private cells.

pub use engine_gpu::guest::{ENTRY, Refused, step_source};

pub type Compiled = engine_gpu::guest::Compiled<crate::device::Api>;
pub type Lowered = engine_gpu::guest::Lowered<String>;
pub type Forms = engine_gpu::guest::Forms<crate::device::Api>;

impl engine_gpu::guest::Lower for crate::device::Api {
    type Code = String;

    fn lower(_stage: usize, source: String) -> Result<String, Refused> {
        let source = for_tint(source);
        tracing::trace!(source = %source, "guest stage source");
        Ok(source)
    }
}

fn for_tint(source: String) -> String {
    const FOLDED: [(&str, &str, &str); 2] = [
        (
            "bitcast<f32>(0x7F800000u)",
            "bitcast<f32>(pie_pos_inf_bits)",
            "var<private> pie_pos_inf_bits : u32 = 0x7F800000u;\n",
        ),
        (
            "bitcast<f32>(0xFF800000u)",
            "bitcast<f32>(pie_neg_inf_bits)",
            "var<private> pie_neg_inf_bits : u32 = 0xFF800000u;\n",
        ),
    ];
    let mut out = source;
    let mut header = String::new();
    for (folded, read, decl) in FOLDED {
        if out.contains(folded) {
            out = out.replace(folded, read);
            header.push_str(decl);
        }
    }
    header + &out
}

#[cfg(feature = "wgpu")]
pub mod run;
#[cfg(feature = "wgpu")]
pub mod session;
#[cfg(feature = "wgpu")]
pub mod widen;

#[cfg(feature = "wgpu")]
pub(crate) fn validated<T>(
    core: &crate::device::ctx::Core,
    what: &'static str,
    build: impl FnOnce() -> T,
) -> std::result::Result<T, String> {
    let scope = core.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let built = build();
    match crate::device::host::block_on(what, scope.pop()) {
        Ok(Some(error)) => Err(error.to_string()),
        Ok(None) => Ok(built),
        Err(fault) => Err(fault.to_string()),
    }
}
