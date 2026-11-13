//! The readout widen a guest pass reads bf16 rows through: its WGSL and
//! config.

pub const WG: u32 = 256;

/// The widen's WGSL with binding 2 (its config) in address space `$cfg`.
macro_rules! widen_source {
    ($cfg:literal) => {
        concat!(
            r#"
@group(0) @binding(0) var<storage, read>       src : array<u32>;
@group(0) @binding(1) var<storage, read_write> dst : array<u32>;

struct Cfg {
  src_word : u32,
  dst_word : u32,
  lanes    : u32,
  pad      : u32,
};
@group(0) @binding(2) "#,
            $cfg,
            r#" cfg : Cfg;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= cfg.lanes) { return; }
  let w = src[cfg.src_word + (i >> 1u)];
  var half : u32 = w & 0xffffu;
  if ((i & 1u) == 1u) { half = w >> 16u; }
  dst[cfg.dst_word + i] = half << 16u;
}
"#
        )
    };
}

/// The widen with its config in a storage buffer: Vulkan binds binding 2
/// as one.
pub const STORAGE_SOURCE: &str = widen_source!("var<storage, read>");

/// The widen with its config in a uniform buffer: WebGPU binds binding 2
/// as one.
pub const UNIFORM_SOURCE: &str = widen_source!("var<uniform>");

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Cfg {
    pub src_word: u32,
    pub dst_word: u32,
    pub lanes: u32,
    pub pad: u32,
}

#[must_use]
pub fn config_bytes(cfg: Cfg) -> [u8; size_of::<Cfg>()] {
    let mut out = [0u8; size_of::<Cfg>()];
    out[0..4].copy_from_slice(&cfg.src_word.to_le_bytes());
    out[4..8].copy_from_slice(&cfg.dst_word.to_le_bytes());
    out[8..12].copy_from_slice(&cfg.lanes.to_le_bytes());
    out[12..16].copy_from_slice(&cfg.pad.to_le_bytes());
    out
}
