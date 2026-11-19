pub use eta_ir::Dtype;

#[allow(non_upper_case_globals)]
pub const f32: Dtype = Dtype::F32;
#[allow(non_upper_case_globals)]
pub const i32: Dtype = Dtype::I32;
#[allow(non_upper_case_globals)]
pub const u32: Dtype = Dtype::U32;
#[allow(non_upper_case_globals)]
pub const bool: Dtype = Dtype::Bool;
/// The storage dtype of vocab-sized intermediates: two bytes an element,
/// computed in f32 and rounded to nearest even on each store; a reduction
/// over it accumulates in f32 and returns f32 (`eta_ir::types`).
#[allow(non_upper_case_globals)]
pub const bf16: Dtype = Dtype::Bf16;
