//! The XLA kernels' profile decisions and [`Tuning`], the values one load
//! runs under. The defaults are the ones measured on a TPU v6e.

use profile::{Decision, Profile, Registry};

/// How a routed (mixture-of-experts) matmul is formulated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.moe.path",
    default = r#"{ use = "auto", gmm = true }"#,
    tune = search
)]
pub enum MoePath {
    /// Chosen per fire from the bank's bytes, the pairs and the cost model
    /// (`kernel.moe.cost.*`); `gmm` lets it take the grouped-matmul kernel.
    Auto {
        #[constant(default = true)]
        gmm: bool,
    },
    /// Each pair's expert sliced out of the bank and multiplied by its row.
    Slice,
    /// The loop over the experts the fire routes to.
    Loop,
    /// The loop over active experts, or the whole bank when more are active
    /// than the loop is worth.
    Active,
    /// Every row against every expert.
    All,
    /// The ragged dot.
    Ragged,
    /// The grouped-matmul kernel, where the bank is one it reads; else
    /// every expert.
    Gmm,
}

/// Tiles the loop over active experts multiplies per step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.loop_unroll", constant(range(1..=8)), default = 2, tune = search)]
pub struct MoeLoopUnroll(pub u32);

/// The most rows one tile of the loop over active experts holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.loop_tile_max", constant(range(8..=128)), default = 128, tune = search)]
pub struct MoeLoopTileMax(pub u32);

/// Rows of one tile of the grouped-matmul kernel: fewer waste less padding
/// per expert, more fill the MXU (128 wide on v5e, 256 on v6e). A multiple
/// of 16, since bf16 rows pack in pairs of sublanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.gmm_tm", constant(set(16, 32, 64, 128)), default = 16, tune = search)]
pub struct MoeGmmTm(pub u32);

/// Row tiles one grid step of the grouped-matmul kernel takes, all of one
/// expert's run, through one dot (a dense bank's routed MLP): the MXU loads
/// each weight tile once for them all, and the steps' overhead is paid once
/// (v6e, a 2048-token fire of Qwen3.6-35B-A3B's 32-expert cut: its gate-up
/// 213 -> 147 us, its down 139 -> 89). 1 takes a step a tile (a v5e's
/// 128-wide MXU loses by it: Gemma-4-26B-A4B's 1792-row routed MLP 2.7 ->
/// 3.3 ms).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.gmm_run", constant(set(1, 2, 4, 8)), default = 1, tune = search)]
pub struct MoeGmmRun(pub u32);

/// The fewest tokens a fire holds for its routed MLP's grouped matmuls to
/// take `kernel.moe.gmm_run` tiles a step: below them a step takes one (v5e:
/// Gemma-4-26B-A4B's 32-expert cut at 640 tokens 838 -> 862 us at two a
/// step, at 1024 984 -> 965).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.gmm_run_from", constant(range(0..=8192)), default = 0, tune = search)]
pub struct MoeGmmRunFrom(pub u32);

/// Weight bytes one block of the grouped-matmul kernel holds, in MiB (two
/// are in flight, plus a bf16 copy): bigger blocks take fewer DMA steps and
/// more VMEM.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.gmm_block_mib", constant(range(1..=64)), default = 12, tune = search)]
pub struct MoeGmmBlockMib(pub u32);

/// Tokens a block of a routed MLP's sum takes (0: one one-hot dot over every
/// row): each block copies the runs its tokens hold of each expert's rows
/// and sums them by its own dot (`moe_gmm::SegmentSum`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.segment_tokens", constant(set(0, 128, 256)), default = 0, tune = search)]
pub struct MoeSegmentTokens(pub u32);

/// The fewest tokens a fire holds for its routed MLP's sum to take the
/// blocks (`kernel.moe.segment_tokens`): below them the one-hot dot or the
/// pick is cheaper (v5e: Qwen3.6-35B-A3B's 16-expert cut at 1088 tokens
/// 690 -> 725 us through the blocks, at 1280 747 -> 733).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.segment_from", constant(range(0..=8192)), default = 1024, tune = search)]
pub struct MoeSegmentFrom(pub u32);

/// Whether a routed MLP's gate-up grouped matmul lands its rows' GLU, each
/// times its weight, in the kernel (1), or its packed halves for the GLU
/// after (0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.gmm_glu", constant(set(0, 1)), default = 0, tune = search)]
pub struct MoeGmmGlu(pub u32);

/// The fewest tokens a fire holds for its routed MLP's two legs to run as
/// one kernel (`moe_gmm::Gmm::mlp_kernel`, its GLU rows kept in VMEM between
/// them); 0: never.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.gmm_fuse_from", constant(range(0..=8192)), default = 0, tune = search)]
pub struct MoeGmmFuseFrom(pub u32);

macro_rules! moe_costs {
    ($($ty:ident = $id:literal, $lo:literal ..= $hi:literal, $default:literal, $doc:literal;)*) => {
        $(
            #[doc = $doc]
            #[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
            #[decision(id = $id, constant(range($lo..=$hi)), default = $default, tune = search)]
            pub struct $ty(pub u32);
        )*
    };
}

// The automatic formulation's cost model (`linear::moe`).
moe_costs! {
    MoeDenseAllMi = "kernel.moe.cost.dense_all_mi", 1..=4096, 128,
        "Up to this many Mi f32 partials, every row runs against every expert.";
    MoeSliceMib = "kernel.moe.cost.slice_mib", 1..=1024, 18,
        "What a dot over one sliced expert costs at least, in MiB of bank streamed.";
    MoeSlicePairs = "kernel.moe.cost.slice_pairs", 1..=64, 8,
        "Up to this many routed pairs, each pair's expert is sliced out.";
    MoeLoopTileNs = "kernel.moe.cost.loop_tile_ns", 1..=1_000_000, 14_000,
        "What one row tile of the loop over active experts costs at least, in ns.";
    MoeLoopRowNs = "kernel.moe.cost.loop_row_ns", 0..=10_000, 10,
        "What each row of such a tile adds, in ns.";
    MoeSliceBytesPerNs = "kernel.moe.cost.slice_bytes_per_ns", 1..=100_000, 1_200,
        "The rate a dot over a sliced expert streams at, in bytes a ns.";
    MoeWholeBytesPerNs = "kernel.moe.cost.whole_bytes_per_ns", 1..=100_000, 1_450,
        "The rate one dot over the whole bank streams at, in bytes a ns.";
    MoeWholePartialFs = "kernel.moe.cost.whole_partial_fs", 0..=1_000_000, 5_500,
        "What each picked partial of that dot adds, in fs.";
    MoeOnehotMi = "kernel.moe.cost.onehot_mi", 0..=4096, 64,
        "Up to this many Mi cells of sorted rows by tokens, a routed MLP sums each token's rows by one dot against them, one-hot; past it, it picks them out.";
}

/// How a quantized projection's rows meet its group factors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.quant.decode",
    default = r#"{ use = "grouped", max_cells = 2048, max_rows = 64 }"#,
    tune = search
)]
pub enum QuantDecode {
    /// Group-batched (each group contracted on its own, the factors scaling
    /// the partials) while `rows · groups <= max_cells` for a packed bank and
    /// `rows <= max_rows` for a native one; past that the weight is decoded
    /// once and contracted densely.
    Grouped {
        #[constant(range(0..=1_048_576), default = 2048)]
        max_cells: u32,
        #[constant(range(0..=4096), default = 64)]
        max_rows: u32,
    },
    /// Always decoded once and contracted densely.
    Dense,
}

/// From this many rows, a rope over heads narrower than a lane tile turns the
/// row as it lies (`elemwise::rope::turn_flat`); 0 never does. A v6e gains
/// (gpt-oss 4096-row prefill: 125.5 -> 123.4 ms a fire); a v5e loses
/// (1792 rows: 65.5 -> 67.6 ms).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.rope.flat_rows", constant(set(0, 1024, 2048, 4096)), default = 0, tune = search)]
pub struct RopeFlatRows(pub u32);

/// Up to this many rows per lane, an `auto` attention walk gathers each
/// row's keys; past it, row blocks walk the key axis. 0 always blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.attn.per_row_max", constant(range(0..=64)), default = 2, tune = search)]
pub struct AttnPerRowMax(pub u32);

/// Mi elements one attention chunk's largest intermediate may hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.attn.budget_mi", constant(range(1..=256)), default = 16, tune = search)]
pub struct AttnBudgetMi(pub u32);

/// From this many rows a decode reads its keys through the page-copy kernel
/// (`attn::pages`) rather than the per-row walk. 0 never does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.attn.decode_pages", constant(set(0, 1, 8, 16, 32, 64, 128)), default = 0, tune = search)]
pub struct AttnDecodePages(pub u32);

/// Query rows per block of the blocked attention walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.attn.block_rows", constant(set(128, 256, 512, 1024, 2048)), default = 512, tune = search)]
pub struct AttnBlockRows(pub u32);

/// Keys per chunk of the blocked attention walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.attn.block_keys", constant(set(128, 256, 512, 1024, 2048)), default = 512, tune = search)]
pub struct AttnBlockKeys(pub u32);

/// The most row blocks of the blocked attention walk that run one after
/// another in line rather than as a loop: in line, no block's result is
/// written into a copy of the whole carried output, at a larger program
/// (more host memory to compile).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.attn.unrolled_blocks", constant(range(1..=16)), default = 16, tune = search)]
pub struct AttnUnrolledBlocks(pub u32);

/// Tokens per chunk of the scalar-decay (GDN) chunked scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ssm.chunk", constant(set(16, 32, 64, 128, 256)), default = 64, tune = search)]
pub struct SsmChunk(pub u32);

/// The GDN chunk once a fire's lanes average at least this many rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ssm.long_chunk", constant(set(64, 128, 256)), default = 128, tune = search)]
pub struct SsmLongChunk(pub u32);

/// From this many lanes a GDN decode step takes each head's `q·S` and
/// `k·S` and its `k ⊗ δ` update through the MXU, a lane's state stored back
/// while the next lane steps (three state buffers). 0 never does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ssm.step_mxu", constant(set(0, 8, 16, 32, 64)), default = 0, tune = search)]
pub struct SsmStepMxu(pub u32);

/// Tokens per chunk of the per-channel-decay (KDA) chunked scan, whose
/// intra-chunk decay is a `[C, C, d]` product.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ssm.kda_chunk", constant(set(8, 16, 32)), default = 16, tune = search)]
pub struct SsmKdaChunk(pub u32);

pub fn register(registry: &mut Registry) {
    registry
        .add::<MoePath>()
        .add::<MoeLoopUnroll>()
        .add::<MoeLoopTileMax>()
        .add::<MoeGmmTm>()
        .add::<MoeGmmRun>()
        .add::<MoeGmmRunFrom>()
        .add::<MoeGmmBlockMib>()
        .add::<MoeSegmentTokens>()
        .add::<MoeSegmentFrom>()
        .add::<MoeGmmGlu>()
        .add::<MoeGmmFuseFrom>()
        .add::<MoeDenseAllMi>()
        .add::<MoeSliceMib>()
        .add::<MoeSlicePairs>()
        .add::<MoeLoopTileNs>()
        .add::<MoeLoopRowNs>()
        .add::<MoeSliceBytesPerNs>()
        .add::<MoeWholeBytesPerNs>()
        .add::<MoeWholePartialFs>()
        .add::<MoeOnehotMi>()
        .add::<QuantDecode>()
        .add::<AttnPerRowMax>()
        .add::<AttnBudgetMi>()
        .add::<AttnDecodePages>()
        .add::<AttnBlockRows>()
        .add::<RopeFlatRows>()
        .add::<AttnBlockKeys>()
        .add::<AttnUnrolledBlocks>()
        .add::<SsmChunk>()
        .add::<SsmLongChunk>()
        .add::<SsmKdaChunk>()
        .add::<SsmStepMxu>();
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeCosts {
    pub dense_all: i64,
    pub slice_bytes: i64,
    pub slice_pairs: i64,
    pub loop_tile_ns: i64,
    pub loop_row_ns: i64,
    pub slice_bytes_per_ns: i64,
    pub whole_bytes_per_ns: i64,
    pub whole_partial_fs: i64,
    pub onehot_cells: i64,
}

/// The most `rows · groups` of a packed bank and the most rows of a native
/// one that run group-batched (0 never).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grouped {
    pub max_cells: i64,
    pub max_rows: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunks {
    /// Elements one chunk's largest intermediate may hold.
    pub budget: i64,
    pub block_rows: i64,
    pub block_keys: i64,
    /// Row blocks run in line (see [`AttnUnrolledBlocks`]).
    pub unrolled: i64,
    /// Whether a sliding-window decode walk stops after the last chunk any
    /// row admits a key in (`attn::paged`'s window walk). Always on from a
    /// profile; off walks every chunk of the window, as the walk did before
    /// the bound, which a test sets to check the bound changes no bit.
    pub window_reach: bool,
    /// Rows from which a decode takes the page-copy kernel (see
    /// [`AttnDecodePages`]).
    pub decode_pages: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SsmChunks {
    pub chunk: i64,
    pub long_chunk: i64,
    pub kda_chunk: i64,
    /// The lanes from which a GDN decode step runs through the MXU
    /// ([`SsmStepMxu`]).
    pub step_mxu: i64,
}

/// The kernel decisions one load runs under, and the device's VMEM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    pub moe_path: MoePath,
    pub moe_loop_unroll: i64,
    pub moe_loop_tile_max: i64,
    pub moe_gmm_tm: i64,
    pub moe_gmm_run: i64,
    pub moe_gmm_run_from: i64,
    pub moe_gmm_block_bytes: i64,
    pub moe_segment_tokens: i64,
    pub moe_segment_from: i64,
    pub moe_gmm_glu: bool,
    pub moe_gmm_fuse_from: i64,
    pub moe_costs: MoeCosts,
    pub grouped: Grouped,
    pub attn_per_row_max: i64,
    pub rope_flat_rows: i64,
    pub chunks: Chunks,
    pub ssm: SsmChunks,
    pub vmem_bytes: i64,
}

impl Tuning {
    pub const V6E_VMEM: i64 = 128 << 20;

    /// The registry defaults on a v6e.
    pub const DEFAULT: Tuning = Tuning {
        moe_path: MoePath::Auto { gmm: true },
        moe_loop_unroll: 2,
        moe_loop_tile_max: 128,
        moe_gmm_tm: 16,
        moe_gmm_run: 1,
        moe_gmm_run_from: 0,
        moe_gmm_block_bytes: 12 << 20,
        moe_segment_tokens: 0,
        moe_segment_from: 1024,
        moe_gmm_glu: false,
        moe_gmm_fuse_from: 0,
        moe_costs: MoeCosts {
            dense_all: 1 << 27,
            slice_bytes: 18 << 20,
            slice_pairs: 8,
            loop_tile_ns: 14_000,
            loop_row_ns: 10,
            slice_bytes_per_ns: 1_200,
            whole_bytes_per_ns: 1_450,
            whole_partial_fs: 5_500,
            onehot_cells: 64 << 20,
        },
        grouped: Grouped {
            max_cells: 2048,
            max_rows: 64,
        },
        attn_per_row_max: 2,
        rope_flat_rows: 0,
        chunks: Chunks {
            budget: 1 << 24,
            block_rows: 512,
            block_keys: 512,
            unrolled: 16,
            window_reach: true,
            decode_pages: 0,
        },
        ssm: SsmChunks {
            chunk: 64,
            long_chunk: 128,
            kda_chunk: 16,
            step_mxu: 0,
        },
        vmem_bytes: Self::V6E_VMEM,
    };

    #[must_use]
    pub fn on(profile: &Profile, vmem_bytes: i64) -> Tuning {
        let n = |v: u32| i64::from(v);
        Tuning {
            moe_path: *profile.get::<MoePath>(),
            moe_loop_unroll: n(profile.get::<MoeLoopUnroll>().0),
            moe_loop_tile_max: n(profile.get::<MoeLoopTileMax>().0),
            moe_gmm_tm: n(profile.get::<MoeGmmTm>().0),
            moe_gmm_run: n(profile.get::<MoeGmmRun>().0),
            moe_gmm_run_from: n(profile.get::<MoeGmmRunFrom>().0),
            moe_gmm_block_bytes: n(profile.get::<MoeGmmBlockMib>().0) << 20,
            moe_segment_tokens: n(profile.get::<MoeSegmentTokens>().0),
            moe_segment_from: n(profile.get::<MoeSegmentFrom>().0),
            moe_gmm_glu: profile.get::<MoeGmmGlu>().0 == 1,
            moe_gmm_fuse_from: n(profile.get::<MoeGmmFuseFrom>().0),
            moe_costs: MoeCosts {
                dense_all: n(profile.get::<MoeDenseAllMi>().0) << 20,
                slice_bytes: n(profile.get::<MoeSliceMib>().0) << 20,
                slice_pairs: n(profile.get::<MoeSlicePairs>().0),
                loop_tile_ns: n(profile.get::<MoeLoopTileNs>().0),
                loop_row_ns: n(profile.get::<MoeLoopRowNs>().0),
                slice_bytes_per_ns: n(profile.get::<MoeSliceBytesPerNs>().0),
                whole_bytes_per_ns: n(profile.get::<MoeWholeBytesPerNs>().0),
                whole_partial_fs: n(profile.get::<MoeWholePartialFs>().0),
                onehot_cells: n(profile.get::<MoeOnehotMi>().0) << 20,
            },
            grouped: match *profile.get::<QuantDecode>() {
                QuantDecode::Grouped {
                    max_cells,
                    max_rows,
                } => Grouped {
                    max_cells: n(max_cells),
                    max_rows: n(max_rows),
                },
                QuantDecode::Dense => Grouped {
                    max_cells: 0,
                    max_rows: 0,
                },
            },
            attn_per_row_max: n(profile.get::<AttnPerRowMax>().0),
            rope_flat_rows: n(profile.get::<RopeFlatRows>().0),
            chunks: Chunks {
                budget: n(profile.get::<AttnBudgetMi>().0) << 20,
                block_rows: n(profile.get::<AttnBlockRows>().0),
                block_keys: n(profile.get::<AttnBlockKeys>().0),
                unrolled: n(profile.get::<AttnUnrolledBlocks>().0),
                window_reach: true,
                decode_pages: n(profile.get::<AttnDecodePages>().0),
            },
            ssm: SsmChunks {
                chunk: n(profile.get::<SsmChunk>().0),
                long_chunk: n(profile.get::<SsmLongChunk>().0),
                kda_chunk: n(profile.get::<SsmKdaChunk>().0),
                step_mxu: n(profile.get::<SsmStepMxu>().0),
            },
            vmem_bytes,
        }
    }

    #[must_use]
    pub fn moe_gmm(&self) -> bool {
        matches!(self.moe_path, MoePath::Auto { gmm: true } | MoePath::Gmm)
    }

    /// The scoped VMEM a Mosaic kernel may ask for: 110 of a v6e's 128 MiB,
    /// in proportion elsewhere.
    #[must_use]
    pub fn vmem_cap(&self) -> i64 {
        self.vmem_bytes / 128 * 110
    }

    /// Bytes of states a decode-form delta rule's round may gather, so the
    /// round stays on chip.
    #[must_use]
    pub fn decode_round_bytes(&self) -> i64 {
        self.vmem_bytes / 4
    }
}

impl Default for Tuning {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_tuning_is_the_registry_defaults() {
        let mut r = Registry::new("xla", &[], &["rows"]);
        register(&mut r);
        assert_eq!(Tuning::on(&r.defaults(), Tuning::V6E_VMEM), Tuning::DEFAULT);
    }

    #[test]
    fn a_v6e_keeps_the_limits_it_was_measured_under() {
        assert_eq!(Tuning::DEFAULT.vmem_cap(), 110 << 20);
        assert_eq!(Tuning::DEFAULT.decode_round_bytes(), 32 << 20);
    }
}
