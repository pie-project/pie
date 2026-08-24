use core::ffi::c_void;



pub const LORA_SITE_Q: u64 = 1 << 0;

pub const LORA_SITE_K: u64 = 1 << 1;

pub const LORA_SITE_V: u64 = 1 << 2;

pub const LORA_SITE_O: u64 = 1 << 3;

pub const LORA_SITE_GATE_UP: u64 = 1 << 4;

pub const LORA_SITE_DOWN: u64 = 1 << 5;

pub const LORA_SITES_KNOWN: u64 =
    LORA_SITE_Q | LORA_SITE_K | LORA_SITE_V | LORA_SITE_O | LORA_SITE_GATE_UP | LORA_SITE_DOWN;

pub const LORA_SITES_CONSUMED: u64 = LORA_SITE_Q | LORA_SITE_V;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum LoraForm {
    #[default]
    LowRank = 0,
    Scale = 1,
}

#[derive(Debug, Clone, Copy)]
pub struct LoraLaneView {
    pub a: *const c_void,
    pub b: *const c_void,
    pub sites_bits: u64,
    pub token_start: u32,
    pub token_count: u32,
    pub num_layers: u32,
    pub rank: u32,
    pub d_in: u32,
    pub d_out: u32,
    pub form: LoraForm,
}

#[derive(Debug, Clone, Copy)]
pub struct Lane {
    pub view: LoraLaneView,
    pub a_bf16: *mut c_void,
    pub b_bf16: *mut c_void,
    pub xa_offset: usize,
    pub grouped: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Group {
    pub rank: i32,
    pub d_in: i32,
    pub d_out: i32,
    pub members: Vec<usize>,
    pub nq: i32,
    pub nv: i32,
    pub m: Vec<i32>,
    pub mq: Vec<i32>,
    pub mv: Vec<i32>,
    pub slab_off: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Staged<'a> {
    pub lanes: &'a [Lane],
    pub groups: &'a [Group],
    pub ptr_slab: *mut c_void,
    pub slab_stride: usize,
}

pub fn bf16_row(base: *const c_void, row: u32, width: i32) -> *const c_void {
    let off = row as usize * usize::try_from(width.max(0)).unwrap_or(0) * 2;

    unsafe { base.cast::<u8>().add(off).cast() }
}
