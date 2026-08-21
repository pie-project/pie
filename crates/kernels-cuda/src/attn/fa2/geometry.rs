
use core::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvWidth(pub u32);

impl KvWidth {

    pub const BF16: Self = Self(2);

    pub const POINTER: u32 = 8;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Device {

    pub cc_major: u32,
    pub max_smem_per_sm: u32,
    pub max_smem_per_block_optin: u32,
}

impl Device {

    pub const L40S: Self =
        Self { cc_major: 8, max_smem_per_sm: 102_400, max_smem_per_block_optin: 101_376 };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {

    DecodeBdxOverWarp { head_dim: u32, bdx: u32 },
    DecodeGroupSize { group_size: u32 },
    DecodeEmptyHeadDim,
    PrefillCtaTileQ { cta_tile_q: u32 },
    PrefillKvTileTooLarge { head_dim: u32, cta_tile_q: u32, fixed_smem: u32, per_mma_kv: u32 },
    PrefillTraitsInvalid { cta_tile_q: u32, num_mma_kv: u32, num_mma_d_vo: u32 },
    PrefillSmemOverBudget { smem_bytes: u32, limit: u32 },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Refusal::DecodeBdxOverWarp { head_dim, bdx } => write!(
                f,
                "fa2 decode: head_dim {head_dim} gives bdx={bdx}, and decode.cuh:765 \
                 static_asserts bdx <= 32 -- head_dim must be a multiple of 32"
            ),
            Refusal::DecodeGroupSize { group_size } => write!(
                f,
                "fa2 decode: GQA group {group_size} is outside DISPATCH_GQA_GROUP_SIZE's \
                 {{1,2,3,4,8}} (utils.cuh:164); 5/6/7 route to the prefill path"
            ),
            Refusal::DecodeEmptyHeadDim => {
                f.write_str("fa2 decode: head_dim 0 has no vec_size (decode.cuh:762)")
            }
            Refusal::PrefillCtaTileQ { cta_tile_q } => write!(
                f,
                "fa2 prefill: cta_tile_q {cta_tile_q} is outside DISPATCH_CTA_TILE_Q's \
                 {{16,32,64,128}} (utils.cuh:135)"
            ),
            Refusal::PrefillKvTileTooLarge { head_dim, cta_tile_q, fixed_smem, per_mma_kv } => {
                write!(
                    f,
                    "fa2 prefill: even NUM_MMA_KV=1 does not fit for head_dim={head_dim} \
                     cta_tile_q={cta_tile_q} (fixed {fixed_smem} B + {per_mma_kv} B per tile) \
                     -- prefill.cuh:4270"
                )
            }
            Refusal::PrefillTraitsInvalid { cta_tile_q, num_mma_kv, num_mma_d_vo } => write!(
                f,
                "fa2 prefill: KernelTraits::IsInvalid() for cta_tile_q={cta_tile_q} \
                 num_mma_kv={num_mma_kv} num_mma_d_vo={num_mma_d_vo} (prefill.cuh:221)"
            ),
            Refusal::PrefillSmemOverBudget { smem_bytes, limit } => write!(
                f,
                "fa2 prefill: SharedStoragePaged is {smem_bytes} B, over this part's \
                 {limit} B opt-in limit (prefill.cuh:4298)"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeGeometry {

    pub num_stages_smem: u32,
    pub tile_size_per_bdx: u32,
    pub vec_size: u32,
    pub bdx: u32,
    pub bdy: u32,
    pub bdz: u32,
    pub num_threads: u32,
    pub smem_bytes: u32,
    pub head_dim: u32,
}

impl DecodeGeometry {

    pub const fn derive(
        head_dim: u32,
        group_size: u32,
        kv: KvWidth,
        dev: Device,
    ) -> Result<Self, Refusal> {
        if head_dim == 0 {
            return Err(Refusal::DecodeEmptyHeadDim);
        }
        let a = 16 / kv.0;
        let b = head_dim / 32;
        let vec_size = if a > b { a } else { b };
        if vec_size == 0 {
            return Err(Refusal::DecodeEmptyHeadDim);
        }
        let bdx = head_dim / vec_size;
        if bdx > 32 {
            return Err(Refusal::DecodeBdxOverWarp { head_dim, bdx });
        }
        if !matches!(group_size, 1 | 2 | 3 | 4 | 8) {
            return Err(Refusal::DecodeGroupSize { group_size });
        }
        let bdy = group_size;
        let lanes = bdx * bdy;
        let num_threads = if lanes > 128 { lanes } else { 128 };
        let bdz = num_threads / lanes;
        let tile_size_per_bdx = if group_size == 1 { if kv.0 == 1 { 2 } else { 4 } } else { 1 };
        let num_stages_smem = if dev.cc_major >= 8 { 2 } else { 1 };
        let staged = 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim * kv.0;
        let offsets = tile_size_per_bdx * num_threads * KvWidth::POINTER;
        let exchange = 2 * bdy * bdz * 4;
        let tail = if offsets > exchange { offsets } else { exchange };
        Ok(Self {
            num_stages_smem,
            tile_size_per_bdx,
            vec_size,
            bdx,
            bdy,
            bdz,
            num_threads,
            smem_bytes: staged + tail,
            head_dim,
        })
    }

    #[must_use]
    pub const fn block(&self) -> [u32; 3] {
        [self.bdx, self.bdy, self.bdz]
    }

    #[must_use]
    pub const fn grid(padded_batch_size: u32, num_kv_heads: u32) -> [u32; 3] {
        [padded_batch_size, num_kv_heads, 1]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Occupancy {

    pub blocks_per_sm: u32,
    pub num_sm: u32,
}

impl Occupancy {

    #[must_use]
    pub const fn max_grid_size(self) -> u32 {
        self.blocks_per_sm * self.num_sm
    }
}

#[allow(clippy::if_same_then_else)]
const fn num_warps_q(cta_tile_q: u32) -> u32 {
    if cta_tile_q == 32 {

        1
    } else if cta_tile_q > 16 {
        4
    } else {
        1
    }
}

const fn num_warps_kv(cta_tile_q: u32) -> u32 {
    4 / num_warps_q(cta_tile_q)
}

#[allow(clippy::if_same_then_else)]
const fn num_mma_q(cta_tile_q: u32) -> u32 {
    if cta_tile_q == 32 {

        2
    } else if cta_tile_q > 64 {
        2
    } else {
        1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillGeometry {

    pub cta_tile_q: u32,
    pub num_mma_q: u32,
    pub num_mma_kv: u32,
    pub num_mma_d_qk: u32,
    pub num_mma_d_vo: u32,
    pub num_warps_q: u32,
    pub num_warps_kv: u32,
    pub cta_tile_kv: u32,
    pub smem_bytes: u32,
    pub head_dim: u32,
}

impl PrefillGeometry {

    pub const ECHO_TEMPLATE: &'static str =
        "&::pie::attn::fa2::smem_bytes_paged";

    pub const fn derive(
        head_dim: u32,
        cta_tile_q: u32,
        kv: KvWidth,
        use_fp16_qk_reduction: bool,
        dev: Device,
    ) -> Result<Self, Refusal> {
        if !matches!(cta_tile_q, 16 | 32 | 64 | 128) {
            return Err(Refusal::PrefillCtaTileQ { cta_tile_q });
        }
        let q_width = 2u32;

        let vo_split_layout = kv.0 == 2 && head_dim >= 512 && cta_tile_q == 32;
        let (num_mma_q, num_warps_q_, num_warps_kv_) = if vo_split_layout {
            (1, 2, 2)
        } else {
            (num_mma_q(cta_tile_q), num_warps_q(cta_tile_q), num_warps_kv(cta_tile_q))
        };

        let num_mma_d_qk = head_dim / 16;
        let num_mma_d_vo = head_dim / 16;

        let use_repack = kv.0 == 1 && head_dim != 64 && head_dim <= 256 && cta_tile_q > 16;
        let kv_shared = num_mma_d_vo > 16
            && num_mma_d_vo.is_multiple_of(num_warps_kv_)
            && (kv.0 == 2 || cta_tile_q > 16);
        let vo_split_dispatch =
            num_mma_d_vo > 16 && num_mma_d_vo.is_multiple_of(num_warps_kv_);

        let per_mma_kv =
            (if kv_shared {
                head_dim * 16 * num_warps_kv_ * kv.0
            } else {
                (head_dim + head_dim) * 16 * num_warps_kv_ * kv.0
            }) + (if use_repack { head_dim * 16 * num_warps_kv_ * q_width } else { 0 })
                + (if vo_split_dispatch { cta_tile_q * num_warps_kv_ * 16 * q_width } else { 0 });

        let vo_split_fixed =
            if vo_split_dispatch { num_warps_kv_ * cta_tile_q * 8 + 2048 } else { 0 };
        let shared_rope_freq = 0;
        let fixed_smem = cta_tile_q * head_dim * q_width + vo_split_fixed + shared_rope_freq;

        let min_valid_mma_kv = if kv.0 == 1 && num_warps_q_ > 2 { num_warps_q_ / 2 } else { 1 };
        let ctas_per_sm = if dev.max_smem_per_sm >= 2 * (fixed_smem + min_valid_mma_kv * per_mma_kv)
        {
            2
        } else {
            1
        };
        let per_block = {
            let a = dev.max_smem_per_sm / ctas_per_sm;
            if a < dev.max_smem_per_block_optin { a } else { dev.max_smem_per_block_optin }
        };
        let _ = use_fp16_qk_reduction;
        let max_mma_kv_reg = 8 / num_mma_q;
        if per_block <= fixed_smem || (per_block - fixed_smem) < per_mma_kv {
            return Err(Refusal::PrefillKvTileTooLarge {
                head_dim,
                cta_tile_q,
                fixed_smem,
                per_mma_kv,
            });
        }
        let max_mma_kv_smem = (per_block - fixed_smem) / per_mma_kv;
        let budget =
            if max_mma_kv_smem < max_mma_kv_reg { max_mma_kv_smem } else { max_mma_kv_reg };
        let num_mma_kv = if budget >= 8 {
            8
        } else if budget >= 4 {
            4
        } else if budget >= 2 {
            2
        } else {
            1
        };

        let num_mma_d_vo_tile = if num_mma_d_vo > 16 { 16 } else { num_mma_d_vo };
        let num_mma_d_vo_per_warp =
            if vo_split_dispatch { num_mma_d_vo / num_warps_kv_ } else { num_mma_d_vo };
        let reg_frags = if vo_split_dispatch { num_mma_d_vo_per_warp } else { num_mma_d_vo_tile };
        let invalid = (if head_dim >= 512 { cta_tile_q > 32 } else { cta_tile_q == 32 })
            || num_mma_d_vo < 4
            || (num_mma_d_vo == 4 && num_mma_kv % 2 == 1)
            || num_mma_q * (8 * reg_frags + 2 * 4 * num_mma_kv) >= 256;
        if invalid {
            return Err(Refusal::PrefillTraitsInvalid { cta_tile_q, num_mma_kv, num_mma_d_vo });
        }

        let cta_tile_kv = num_mma_kv * num_warps_kv_ * 16;
        let smem_bytes = Self::shared_storage_paged(
            cta_tile_q,
            cta_tile_kv,
            head_dim,
            num_warps_kv_,
            kv,
            q_width,
        );
        if smem_bytes > dev.max_smem_per_block_optin {
            return Err(Refusal::PrefillSmemOverBudget {
                smem_bytes,
                limit: dev.max_smem_per_block_optin,
            });
        }
        Ok(Self {
            cta_tile_q,
            num_mma_q,
            num_mma_kv,
            num_mma_d_qk,
            num_mma_d_vo,
            num_warps_q: num_warps_q_,
            num_warps_kv: num_warps_kv_,
            cta_tile_kv,
            smem_bytes,
            head_dim,
        })
    }

    #[must_use]
    pub const fn shared_storage_paged(
        cta_tile_q: u32,
        cta_tile_kv: u32,
        head_dim: u32,
        num_warps_kv: u32,
        kv: KvWidth,
        q_width: u32,
    ) -> u32 {

        const fn align16(n: u32) -> u32 {
        n.div_ceil(16) * 16
        }

        let kv_share_shape = head_dim / 16 > 16 && (head_dim / 16).is_multiple_of(num_warps_kv);
        let vo_split = kv_share_shape;
        let v_share_active = kv_share_shape && (kv.0 == 2 || cta_tile_q > 16);

        let mut a = 0;
        a = align16(a) + cta_tile_q * head_dim * q_width;
        a = align16(a) + cta_tile_kv * head_dim * kv.0;
        a = align16(a) + if v_share_active { kv.0 } else { cta_tile_kv * head_dim * kv.0 };
        let a = align16(a);

        let sync_o_elems = if num_warps_kv == 1 || vo_split {
            1
        } else {
            num_warps_kv * cta_tile_q * if head_dim > 256 { 256 } else { head_dim }
        };
        let sync_md_elems = if num_warps_kv == 1 { 1 } else { num_warps_kv * cta_tile_q };
        let mut b = 0;
        b = align16(b) + sync_o_elems * 4;
        b = align16(b) + sync_md_elems * 8;
        let b = align16(b);

        let c = align16(cta_tile_q * head_dim * q_width);

        let mut off = if a > b { a } else { b };
        if c > off {
            off = c;
        }

        off = align16(off) + 1;
        off = align16(off) + 1;
        off = align16(off) + q_width;
        off = align16(off) + if vo_split { cta_tile_q * cta_tile_kv * q_width } else { q_width };
        off = align16(off) + if vo_split { num_warps_kv * cta_tile_q * 8 } else { 8 };
        align16(off)
    }

    #[must_use]
    pub const fn block(&self) -> [u32; 3] {
        [32, self.num_warps_q, self.num_warps_kv]
    }

    #[must_use]
    pub const fn grid(padded_batch_size: u32, num_kv_heads: u32) -> [u32; 3] {
        [padded_batch_size, 1, num_kv_heads]
    }
}
