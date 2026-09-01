use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FileId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InstrId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Axis(pub u8);

/// What a checkpoint tensor holds — [`dtype::Dtype`], re-exported under the
/// name this crate has always spelled it.
///
/// The enum ITSELF stood here, one of two saying the same thing: the IR's
/// `model_ir::Dtype` named the compute and storage elements over a vocabulary
/// that overlapped this one without containing it (it knew `Fp4` and `Mxfp4`,
/// it did not know the wide ints or `Bool`), so every edge between a checkpoint
/// and a plan was a hand-written table. There is one enum now, and the two
/// spellings that differed (`BF16`, `F8E4M3`, `F8E5M2`, `E8M0`) survive as
/// serde aliases so a plan recorded under them still reads.
pub use dtype::Dtype as DType;

/// Whether a checkpoint storing `dtype` ships a separate block-scale tensor
/// alongside it.
///
/// A *format* fact, not a device one: DeepSeek- and GLM-style FP8
/// checkpoints carry one scale per `[B, B]` tile of the weight, so an FP8
/// tensor is never self-describing. The block size `B` is what the
/// consuming kernel fixes, and that is on the target
/// ([`crate::plan::StorageTarget::block_scale_rows`]); which dtypes
/// arrive that way is here, because it is true of the file no matter who
/// reads it.
///
/// A free function rather than the inherent method it was: [`DType`] is defined
/// in a leaf crate that knows nothing about checkpoints, and this is a fact
/// about checkpoints.
#[must_use]
pub fn is_block_scaled(dtype: DType) -> bool {
    matches!(dtype, DType::E4m3 | DType::E5m2)
}

/// Which on-disk format a checkpoint file is.
///
/// One variant per format the loader can read, which is every format
/// `ztensor-compat` projects — the loader enables all of them. `Unknown` is
/// therefore not a "cannot read this" marker; it is what a *newer* zTensor
/// would report for a format this build has no name for yet, and
/// `checkpoint_format` has a test that keeps it unreachable until then.
///
/// New variants are appended rather than slotted in beside their siblings:
/// these numbers cross the C ABI, so an engine compiled against an older header
/// must keep reading every value it knew as the number it knew.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckpointFormat {
    Safetensors,
    Gguf,
    Unknown,
    /// The loader's own container (`.zt`), including a root that names shards.
    Zt,
    /// NumPy's zip archive (`.npz`).
    Npz,
    /// PyTorch's pickle archive (`.pt`).
    Pt,
    /// HDF5 (`.h5`), including Keras checkpoints.
    Hdf5,
    /// ONNX protobuf (`.onnx`), read for its initializers.
    Onnx,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    Cuda,
    Metal,
    Vulkan,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantScheme {
    None,
    Fp8E4M3,
    Fp8E5M2,
    Int8Symmetric,
    Int8Asymmetric,
    AwqInt4,
    GptqInt4,
    Mxfp4E2M1E8M0,
    MlxAffineU4,
    GgufQ4_0,
    /// Two bits per weight, affine, in sixteen sub-blocks of sixteen.
    ///
    /// The lowest-precision block ggml stores self-contained, and the reason
    /// the K family stops here rather than at Q4_K: an IQ scheme of the same
    /// width indexes a lattice compiled into llama.cpp, so its block does not
    /// carry its own values. This one does.
    ///
    /// Unlike [`Self::GgufQ4K`] the super-block scales sit at the END of the
    /// payload, after the sub-block scales and the quants.
    GgufQ2K,
    /// Three bits per weight, symmetric, in sixteen sub-blocks of sixteen.
    ///
    /// The third bit lives in a separate 32-byte mask, and it reads
    /// INVERTED: a set bit means the element keeps its two-bit value, and a
    /// clear one means four is subtracted from it.
    GgufQ3K,
    GgufQ4K,
    GgufQ5_0,
    GgufQ5K,
    GgufQ8_0,
    /// 4-bit integers biased by 8, eight to a 32-bit word, low nibble first.
    ///
    /// An element is `nibble - 8`, so the stored range `0..=15` reads as
    /// `-8..=7`. The group scales are a separate tensor rather than part of
    /// this name -- a contract pairs the two with [`crate::contract::Expr`]'s
    /// `Scale` -- which is why, unlike [`Self::AwqInt4`] and
    /// [`Self::GptqInt4`], there is no zero-point tensor implied here: the
    /// zero point *is* the 8.
    ///
    /// New variants go on the end. The FFI discriminants follow declaration
    /// order and the C++ side reads them as integers, so inserting one in the
    /// middle renumbers every scheme after it.
    Int4B8,
    /// llama.cpp's `block_q6_K`: a 256-element super-block of 6-bit codes,
    /// scaled in sixteens.
    ///
    /// 210 bytes -- 128 of low nibbles, 64 of high pairs, 16 signed sub-block
    /// scales and one half-precision super-block scale -- measured off
    /// `llama-2-7b.Q4_0.gguf` rather than read off a struct definition.
    ///
    /// Carried because a "Q4_0" release is not uniformly Q4_0: that file
    /// holds 225 Q4_0 tensors and exactly one Q6_K, `output.weight`, and
    /// without this row the missing scheme refused the whole 3.8 GB file at
    /// parse. Naming a scheme is what lets its bytes be COPIED; decoding them
    /// is a separate claim this does not make.
    GgufQ6K,
    /// llama.cpp's `block_q4_1`: 20 bytes per 32 elements — an F16 scale, an
    /// F16 offset and sixteen packed bytes, each element `nibble × d + m`.
    ///
    /// The affine sibling of [`Self::GgufQ4_0`]. Its offset is *added*, where
    /// a K-quant's minimum is subtracted, so the two cannot share a decoder
    /// arm however similar they look.
    GgufQ4_1,
    /// llama.cpp's `block_q5_1`: 24 bytes per 32 elements — [`Self::GgufQ4_1`]
    /// plus a 32-bit plane of fifth bits.
    ///
    /// Carried because a K-quant release is not uniformly K-quant, the same
    /// way `GgufQ6K` is: `qwen2.5-0.5b-instruct-q5_k_m.gguf` holds 12 Q5_K
    /// tensors, 12 Q6_K and **133 Q5_1**, so without this row the file is
    /// refused at parse and the most common Q5 release in the wild cannot be
    /// read at all.
    GgufQ5_1,
    /// llama.cpp's `block_iq4_nl`: 18 bytes per 32 elements — an F16 scale
    /// and sixteen packed bytes of 4-bit indices.
    ///
    /// The same size as [`Self::GgufQ4_0`] and a different thing entirely.
    /// Q4_0's nibble is a *number*, offset by eight; this one is an INDEX
    /// into a sixteen-entry table of non-uniform levels. "NL" is non-linear:
    /// the levels crowd near zero, where weights are dense, and spread out
    /// toward the tails. Reading these codes as Q4_0's would decode every
    /// element to the wrong magnitude while looking entirely plausible.
    ///
    /// The table is llama.cpp's `kvalues_iq4nl` and is compiled in rather
    /// than shipped, which is what makes this decodable where the IQ2/IQ3
    /// lattice schemes are not: sixteen values fit in a line, a lattice does
    /// not.
    GgufIq4Nl,
    /// llama.cpp's `block_iq4_xs`: 136 bytes per 256 elements — eight
    /// sub-blocks of 32 over [`Self::GgufIq4Nl`]'s table.
    ///
    /// The sub-block scale is six bits split across two planes: four low bits
    /// in `scales_l`, packed two per byte, and two high bits in `scales_h`,
    /// packed eight per `u16`. It is read as `ls - 32`, so it is signed and
    /// the super-block scale multiplies it.
    ///
    /// Measured on `Llama-3.2-1B-Instruct-UD-Q2_K_XL.gguf`, which holds five
    /// of these beside IQ2_S, IQ2_XS, IQ3_S and IQ3_XXS. Naming this one does
    /// not name those: they index a lattice, and the lattice is not in the
    /// file.
    GgufIq4Xs,
    /// llama.cpp's `block_mxfp4`: 17 bytes per 32 elements — one E8M0 scale
    /// byte followed by sixteen bytes of packed E2M1 nibbles.
    ///
    /// The same *numeric* format as [`Self::Mxfp4E2M1E8M0`] and a different
    /// *byte* layout, which is the only thing a loader addresses bytes with.
    /// OCP Microscaling stores the codes and the scales as two planes, so a
    /// tensor of N elements is N/2 bytes of data beside a separate scale
    /// tensor; ggml interleaves the scale into each block, so the same tensor
    /// is N × 17/32 bytes with no companion at all.
    ///
    /// The two shared one variant until this was measured. `gguf.mxfp4/1` was
    /// read as `Mxfp4E2M1E8M0`, which reports no block layout, so the tensor
    /// passed through byte for byte and was written back out under the OCP
    /// profile `zt.mx/1`. On `gpt-oss-20b-MXFP4.gguf` that produced 72 objects
    /// whose declared extent was 132,710,400 bytes over a stored span of
    /// 141,004,800 — a ratio of exactly 17/16 — so **597 MB of payload sat
    /// outside what the manifest described**, and every block after the first
    /// was at the wrong offset for any consumer computing addresses from the
    /// profile. Naming the layout separately is what makes the two spans agree.
    GgufMxfp4,
    /// llama.cpp's `block_iq2_xxs`: 66 bytes per 256 elements.
    ///
    /// The first of the schemes that quantize a *direction* rather than a
    /// magnitude. A byte does not hold a weight here — it holds an address
    /// into a 256-entry table of eight-element points, chosen offline against
    /// real weight distributions, and the block's scale and a seven-bit sign
    /// index place that point. Two bits a weight buys nothing unless the
    /// points are good, and the points are what "IQ" names.
    ///
    /// The table is not in the file. It is compiled in, from `gguf-py`'s
    /// `iq2xxs_grid`, which is why these landed after the IQ4 schemes rather
    /// than beside them — see `crates/checkpoint/src/executor/iq_grid.rs`.
    GgufIq2Xxs,
    /// llama.cpp's `block_iq2_xs`: 74 bytes per 256 elements over a 512-entry
    /// grid.
    ///
    /// Twice [`Self::GgufIq2Xxs`]'s grid, so the point index takes nine bits
    /// and the sign index moves up to the top seven of the same `u16`. The
    /// eight extra bytes over `IQ2_XXS` are per-16 scales instead of per-32.
    GgufIq2Xs,
    /// llama.cpp's `block_iq2_s`: 82 bytes per 256 elements over a 1024-entry
    /// grid.
    ///
    /// The widest IQ2 grid, addressed by ten bits — eight in `qs` and two more
    /// in a `qh` plane. It also stops packing signs through a parity index and
    /// stores all eight outright, which is what the extra bytes over
    /// [`Self::GgufIq2Xs`] buy.
    GgufIq2S,
    /// llama.cpp's `block_iq3_xxs`: 98 bytes per 256 elements.
    ///
    /// The IQ3 grids hold **four** components per point, not eight, so a point
    /// index covers four weights and 64 of them cover the block. Everything
    /// else is shaped like [`Self::GgufIq2Xxs`] — four seven-bit sign indices
    /// and a four-bit scale to a `u32` — with the scale factor doubled to
    /// match a grid whose components run to 62.
    GgufIq3Xxs,
    /// llama.cpp's `block_iq3_s`: 110 bytes per 256 elements over a 512-entry
    /// four-component grid.
    ///
    /// The one scheme here whose scale is an odd integer, `1 + 2s`, over a grid
    /// of the odd numbers 1 through 15. Its `qh` plane contributes a single bit
    /// per point where [`Self::GgufIq2S`]'s contributes two, and its signs are
    /// stored outright.
    GgufIq3S,
}

impl QuantScheme {
    pub fn default_bits(self) -> u8 {
        match self {
            Self::AwqInt4
            | Self::GptqInt4
            | Self::Mxfp4E2M1E8M0
            | Self::MlxAffineU4
            | Self::GgufQ4_0
            | Self::GgufQ4_1
            | Self::GgufQ4K
            | Self::GgufIq4Nl
            | Self::GgufIq4Xs
            | Self::GgufMxfp4
            | Self::Int4B8 => 4,
            Self::GgufQ2K | Self::GgufIq2Xxs | Self::GgufIq2Xs | Self::GgufIq2S => 2,
            Self::GgufQ3K | Self::GgufIq3Xxs | Self::GgufIq3S => 3,
            Self::GgufQ5_0 | Self::GgufQ5_1 | Self::GgufQ5K => 5,
            Self::GgufQ6K => 6,
            Self::Fp8E4M3
            | Self::Fp8E5M2
            | Self::Int8Symmetric
            | Self::Int8Asymmetric
            | Self::GgufQ8_0
            | Self::None => 8,
        }
    }

    /// The block a GGUF-family scheme stores, as `(elements, bytes)`, or
    /// `None` for a scheme whose payload is a plain bit-packing.
    ///
    /// GGUF blocks carry their scales *inside* the payload — Q4_0 is one F16
    /// scale and sixteen packed bytes per 32 elements — so their size is not
    /// `elements × bits / 8` and a span computed that way reads short. These
    /// are the GGML reference layouts, and this table is the only place in
    /// the crate that states them — `file/zt.rs` and `file/write.rs` carry the
    /// `gguf.q4_0/1` encoding *name* and ask here for its size.
    ///
    /// Answering here is also what makes a scheme decodable: the host `Decode`
    /// sizes its blocks from this and dispatches on the same set, so a scheme
    /// listed here without a decoder is a compile error rather than a run that
    /// reads the wrong number of bytes.
    pub fn block_layout(self) -> Option<(u64, u64)> {
        match self {
            Self::GgufQ4_0 => Some((32, 18)),
            Self::GgufQ4_1 => Some((32, 20)),
            Self::GgufQ5_0 => Some((32, 22)),
            Self::GgufQ5_1 => Some((32, 24)),
            Self::GgufQ8_0 => Some((32, 34)),
            Self::GgufQ2K => Some((256, 84)),
            Self::GgufQ3K => Some((256, 110)),
            Self::GgufQ4K => Some((256, 144)),
            Self::GgufQ5K => Some((256, 176)),
            Self::GgufQ6K => Some((256, 210)),
            Self::GgufIq4Nl => Some((32, 18)),
            Self::GgufIq4Xs => Some((256, 136)),
            Self::GgufMxfp4 => Some((32, 17)),
            Self::GgufIq2Xxs => Some((256, 66)),
            Self::GgufIq2Xs => Some((256, 74)),
            Self::GgufIq2S => Some((256, 82)),
            Self::GgufIq3Xxs => Some((256, 98)),
            Self::GgufIq3S => Some((256, 110)),
            _ => None,
        }
    }

    /// Whether this scheme keeps its scales *inside* its payload.
    ///
    /// The same question [`block_layout`](Self::block_layout) answers with
    /// sizes, asked by the callers that only want the yes or no. It is worth a
    /// name because three separate policies turn on it, and reading
    /// `block_layout().is_some()` at each one states the mechanism where the
    /// decision belongs:
    ///
    /// - `pie model import` **keeps** such a tensor as the source wrote it,
    ///   rather than unpacking it to BF16 (`contract::materialize`).
    /// - A family author **decodes** one before publishing it, because it is
    ///   about to be bound (`model::shared::builder`).
    /// - The loader **refuses** to hand one to a device that has no kernel
    ///   able to read it (`plan::passes::validate`).
    ///
    /// All three follow from the one fact, so all three should ask for it by
    /// name. Note the two MXFP4 spellings are not the same answer:
    /// `GgufMxfp4` interleaves its scale with its codes and is self-contained,
    /// while `Mxfp4E2M1E8M0` is the OCP form with a separate scale plane and
    /// is not.
    #[must_use]
    pub fn is_self_contained(self) -> bool {
        self.block_layout().is_some()
    }

    pub fn default_group_size(self) -> u32 {
        match self {
            Self::AwqInt4 | Self::GptqInt4 | Self::Mxfp4E2M1E8M0 | Self::Int4B8 => 32,
            Self::MlxAffineU4 => 64,
            Self::GgufQ4_0
            | Self::GgufQ4_1
            | Self::GgufQ4K
            | Self::GgufQ5_0
            | Self::GgufQ5_1
            | Self::GgufQ5K
            | Self::GgufIq4Nl
            | Self::GgufIq4Xs
            | Self::GgufMxfp4 => 32,
            // Sixteen, not the 32 its neighbours use: Q6_K carries one scale
            // per sixteen elements, and Q2_K and Q3_K cut their super-block
            // into sixteen sub-blocks of sixteen the same way. Inert for
            // extents either way, since a scheme with a `block_layout`
            // answers through that.
            Self::GgufQ2K | Self::GgufQ3K | Self::GgufQ6K => 16,
            // The IQ lattice schemes have no group size in the sense the
            // affine ones do: a grid point covers eight elements (IQ2) or four
            // (IQ3), and scales run per 16 or 32. Inert either way, since a
            // scheme with a `block_layout` answers extents through that.
            Self::GgufIq2Xxs
            | Self::GgufIq2Xs
            | Self::GgufIq2S
            | Self::GgufIq3Xxs
            | Self::GgufIq3S => 32,
            Self::Fp8E4M3
            | Self::Fp8E5M2
            | Self::Int8Symmetric
            | Self::Int8Asymmetric
            | Self::GgufQ8_0
            | Self::None => 1,
        }
    }
}

/// The tiled affine layout's two geometry constants, restated nowhere: they
/// live beside the [`DType::U4g64tiled`] variant that names the layout, and
/// this crate's [`RepackLayout::TiledAffineU4Weight`] checks a declared target
/// shape against them.
pub use dtype::{TILED_BAND, TILED_STEP};

/// Which backend kernel a [`Expr::Repack`](crate::contract::Expr::Repack) names.
///
/// The whole of what a repack says in a contract. Everything a kernel also
/// needs -- how many rows, how many columns, which rows -- is either the
/// operand's type or the declared output's, so the plan builder derives it and
/// a contract never repeats it.
///
/// Every value here names a kernel, and there is deliberately no `None` and no
/// [`Default`]: a repack with no layout is not a repack, so the algebra should
/// not be able to hold one. The discriminants start at 1 for the same reason
/// the enum is total — zero is what an uninitialized FFI field carries, so it
/// must decode as an error rather than as the first kernel in the list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RepackLayout {
    MarlinMxfp4Weight = 1,
    MarlinMxfp4Scale = 2,
    /// **THE FOUR-BIT AFFINE CODE PLANE IN m16n8k16 FRAGMENT ORDER**
    /// (§J4b) — `kernels_cuda::linear::tiled`'s `repack_affine_tiled`, whose
    /// banner states the map and whose two readers are the tiled GEMM and
    /// the tiled decode point.
    ///
    /// Operand `[rows, k]` of four-bit affine codes, target `[rows padded to
    /// a whole 16-column band, k]` of the same. The rows are the
    /// PROJECTION's output columns, so the padding is the band quantum and
    /// the tail decodes to a zero weight.
    ///
    /// It is a plain matrix and not a batch, unlike the two above: a dense
    /// projection has no expert axis, and declaring a leading `1` would be a
    /// dimension the algebra could disagree with itself about.
    TiledAffineU4Weight = 3,
    /// **THE FACTOR PLANE BESIDE IT** — `repack_factors_tiled`, which is a
    /// transpose of the (column, group) rectangle inside each 16-column
    /// band and nothing else. One layout for the scales and the biases
    /// alike: they are the same rectangle in the same order, and a second
    /// row here would be one name for one permutation.
    TiledAffineFactor = 4,
}

/// A repack as the *executor* needs it: the layout plus the geometry.
///
/// Not part of the contract. Every field but `layout` is derived by
/// [`plan::compile`](crate::plan::compile) from the operand's type and the
/// declaration, which is what keeps the algebra from restating in integers what
/// it already says in nodes. A contract selects rows with
/// [`Expr::Slice`](crate::contract::Expr::Slice),
/// [`Expr::Shard`](crate::contract::Expr::Shard) and
/// [`Expr::Stride`](crate::contract::Expr::Stride); by the time a spec exists
/// the operand is exactly the block the kernel reads.
///
/// `target_rows`/`target_cols` may exceed the source's: a layout with a tile
/// quantum declares the padded shape and the kernel zero-fills the tail, which
/// is the one geometric fact that is the kernel's and not the algebra's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RepackSpec {
    pub layout: RepackLayout,
    pub batch: u32,
    pub source_rows: u32,
    pub target_rows: u32,
    pub source_cols: u32,
    pub target_cols: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantSpec {
    pub scheme: QuantScheme,
    pub logical_dtype: DType,
    pub bits_per_element: u8,
    pub group_size: u32,
    pub channel_axis: Option<Axis>,
}

impl QuantSpec {
    pub fn normalized(mut self) -> Self {
        if self.bits_per_element == 0 {
            self.bits_per_element = self.scheme.default_bits();
        }
        if self.group_size == 0 {
            self.group_size = self.scheme.default_group_size();
        }
        self
    }

    /// The width of one element when the payload is a plain array, or `None`
    /// when it is not addressable that way.
    ///
    /// A blocked scheme answers `None` whatever its bit width. Q8_0 is the
    /// reason this is stated rather than implied: it is the one GGUF scheme
    /// whose bits divide by eight, so a width derived from bits alone comes
    /// back as `Some(1)` and every caller then sizes its span as
    /// `elements × 1`, which is short by the F16 scale in each block. The
    /// others fall out only because four, five and six bits do not divide.
    /// Having a block is the property that matters, so it is asked first.
    pub fn dense_element_bytes(&self) -> Option<u64> {
        if self.scheme.is_self_contained() {
            return None;
        }
        let bits = self.normalized_bits();
        if bits.is_multiple_of(8) {
            Some(u64::from(bits / 8))
        } else {
            None
        }
    }

    /// The block a GGUF-family scheme stores, as `(elements, bytes)`, or
    /// `None` for a scheme whose payload is a plain bit-packing.
    ///
    /// GGUF blocks carry their scales *inside* the payload — Q4_0 is one F16
    /// scale and sixteen packed bytes per 32 elements — so their size is not
    /// `elements × bits / 8` and a span computed that way reads short. These
    /// are the GGML reference layouts, and this table is the only place in
    /// the crate that states them — `file/zt.rs` and `file/write.rs` carry the
    /// `gguf.q4_0/1` encoding *name* and ask here for its size.
    pub fn block_layout(&self) -> Option<(u64, u64)> {
        self.scheme.block_layout()
    }

    pub fn normalized_bits(&self) -> u8 {
        if self.bits_per_element == 0 {
            self.scheme.default_bits()
        } else {
            self.bits_per_element
        }
    }

    pub fn normalized_group_size(&self) -> u32 {
        if self.group_size == 0 {
            self.scheme.default_group_size()
        } else {
            self.group_size
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Encoding {
    Raw(DType),
    Quant(QuantSpec),
}

impl Encoding {
    /// The type one element reads as. For a quantized encoding this is the
    /// logical type the elements decode to, not their storage width.
    pub fn dtype(&self) -> DType {
        match self {
            Encoding::Raw(dtype) => *dtype,
            Encoding::Quant(spec) => spec.logical_dtype,
        }
    }
}

pub fn normalize_encoding(encoding: &Encoding) -> Encoding {
    match encoding {
        Encoding::Raw(dtype) => Encoding::Raw(*dtype),
        Encoding::Quant(spec) => Encoding::Quant(spec.clone().normalized()),
    }
}

/// Whether a declared tensor is something the engine binds, or a name the
/// contract needed for itself.
///
/// The algebra has no `let`: the only way to use a subexpression twice, or to
/// feed one entry's result into another's [`Expr::Scale`](crate::contract::Expr::Scale) factors, is to give it
/// a name. Without this, every such name is also a runtime weight — a stacked
/// slab of dequantization factors ends up in the persistent arena and stays
/// there for the life of the process, and the bind namespace fills with
/// tensors no kernel will ever ask for.
///
/// `Internal` is that name without those consequences. It resolves through
/// [`Expr::Out`](crate::contract::Expr::Out) like any other, but the plan emits no `Finalize` for it, so the
/// engine never sees it and its buffer stays a temporary the memory planner may
/// reuse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    /// A runtime weight. The engine binds it by name.
    #[default]
    Public,
    /// A name for the contract's own use. Not bound, not persistent.
    Internal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDecl {
    pub id: TensorId,
    pub name: String,
    /// What this rank holds, in elements — the band a
    /// [`Expr::Shard`](crate::contract::Expr::Shard) cut out, not the whole
    /// tensor the contract that asked for it declares.
    pub shape: Vec<i64>,
    pub encoding: Encoding,
    pub alignment: u32,
    /// Whether the engine binds this name. See [`Visibility`].
    #[serde(default, skip_serializing_if = "Visibility::is_public")]
    pub visibility: Visibility,
}

impl Visibility {
    pub fn is_public(&self) -> bool {
        matches!(self, Visibility::Public)
    }
}

impl TensorDecl {
    pub fn dtype(&self) -> DType {
        self.encoding.dtype()
    }
}

/// A declared shape read as the `[rows, cols]` rectangle every encode kernel
/// walks: the LAST axis is the contracted one, and every axis before it folds
/// into the row count.
///
/// **The fold is not a convenience, it is the layout.** A dense tensor is
/// row-major, so `[experts, rows, cols]` and `[experts * rows, cols]` are the
/// same bytes in the same order, and a kernel that indexes `row * cols + c` —
/// which is every quantizer in the tree, on both the host and the device — has
/// already been walking the folded rectangle for every rank it was ever
/// handed. What was rank-2-only was the ARITHMETIC ABOVE it: `for_encode`
/// destructured `[rows, cols]` and refused anything else, so a contract that
/// stacked experts and then asked for runtime quantization got a refusal
/// naming a restriction the kernels did not have.
///
/// Rank 0 and rank 1 are `None` rather than folded. A rank-1 tensor has no
/// axis left over to hold a per-row scale, so the caller's question ("what
/// rectangle do I scale") has no answer for it, and inventing `rows = 1` would
/// turn a shapeless declaration into a plausible one-row weight.
#[must_use]
pub fn rectangle(shape: &[i64]) -> Option<(i64, i64)> {
    let (&cols, lead) = shape.split_last()?;
    if lead.is_empty() {
        return None;
    }
    let rows = lead
        .iter()
        .try_fold(1i64, |acc, dim| acc.checked_mul(*dim))?;
    Some((rows, cols))
}

/// A block-scaled scales shape: the payload's leading axes, then one entry
/// per group along the contracted axis.
///
/// Its own function because two sides state it and they must not drift: the
/// plan compiler BUILDS it (`plan::build::ScaleLayout::for_encode`) and the
/// host executor CHECKS the buffer it was handed against it before it writes
/// a byte.
#[must_use]
pub fn grouped_shape(lead: &[i64], groups: i64) -> Vec<i64> {
    let mut shape = lead.to_vec();
    shape.push(groups);
    shape
}

pub fn tensor_nbytes(shape: &[i64], element_bytes: u64) -> Option<u64> {
    tensor_elements(shape)?.checked_mul(element_bytes)
}

pub fn tensor_elements(shape: &[i64]) -> Option<u64> {
    let mut elements = 1u64;
    for dim in shape {
        let dim = u64::try_from(*dim).ok()?;
        elements = elements.checked_mul(dim)?;
    }
    Some(elements)
}

pub fn encoding_dense_element_bytes(encoding: &Encoding) -> Option<u64> {
    match encoding {
        Encoding::Raw(dtype) => Some(dtype.bytes_ceil()),
        Encoding::Quant(spec) => spec.dense_element_bytes(),
    }
}

pub fn encoding_nbytes(shape: &[i64], encoding: &Encoding) -> Option<u64> {
    match encoding {
        Encoding::Raw(dtype) => tensor_nbytes(shape, dtype.bytes_ceil()),
        Encoding::Quant(spec) => {
            let spec = spec.clone().normalized();
            // A blocked scheme's scales live inside the payload, so its span
            // is blocks × block bytes, not elements × bits.
            if let Some((block_elements, block_bytes)) = spec.block_layout() {
                let elements = tensor_elements(shape)?;
                return elements.div_ceil(block_elements).checked_mul(block_bytes);
            }
            if let Some(element_bytes) = spec.dense_element_bytes() {
                return tensor_nbytes(shape, element_bytes);
            }
            let elements = tensor_elements(shape)?;
            let bits = elements.checked_mul(u64::from(spec.bits_per_element))?;
            Some(bits.div_ceil(8))
        }
    }
}

/// How a scale tensor's entries map onto the tensor they scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantGranularity {
    /// One scale per row of `channel_axis`.
    PerChannel,
    /// One scale per `group_size` elements along the axis after `channel_axis`.
    PerGroup,
}

/// What the engine's kernels expect a scale tensor to hold by the time they read
/// it.
///
/// Not derivable from the scale tensor itself — its dtype says how the bytes are
/// stored, not how the kernel wants them — so whoever declared the scale states
/// it. The engine used to infer it from `group_size == 32`, which was true only
/// because MXFP4 is the one scheme with that group size today.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleForm {
    /// Consumed as raw E8M0 exponent bytes. The MXFP4 GEMM, the dequant kernels
    /// and `make_expert_weight_view` all require U8 and assert on anything else.
    RawE8M0,
    /// Consumed as F32 multipliers. Whatever the scales were stored as (E8M0
    /// bytes, BF16, or F32 already) is expanded before the GEMM sees them.
    F32Factors,
    /// Consumed as BF16 multipliers that are only half of the dequantization:
    /// the scheme is affine, so a second tensor holds the zero point each group
    /// is offset by, and an element is `code * scale + zero`.
    ///
    /// The zero point is named by [`QuantAttachment::zero_point_tensor`] rather
    /// than implied by a suffix, for the same reason the scale is: a kernel that
    /// cannot find it does not read a coarser weight, it reads a wrong one.
    ///
    /// New variants go on the end. The FFI discriminants follow declaration
    /// order and the C++ side reads them as integers.
    Bf16AffineFactors,
}
