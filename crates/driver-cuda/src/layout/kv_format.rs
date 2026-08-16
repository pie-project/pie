//! The KV cache storage format: what one page costs, and how a config string
//! names it.
//!
//! Port of `driver-cuda/csrc/src/store/kv_cache_format.{hpp,cpp}`.
//!
//! # One table instead of three lists
//!
//! The C++ spells the format catalogue three times: a chain of `if (v == ...)`
//! in the parser, a hand-written comma-separated string in
//! `valid_kv_cache_dtype_values()`, and a `try`/`catch` round-trip in
//! `is_valid_kv_cache_dtype()`. The first two are in sync today, by hand, and
//! nothing checks that they still are -- adding a format and forgetting the
//! string produces an error message that omits the option the user was
//! supposed to discover.
//!
//! Here [`KvCacheFormat::CATALOGUE`] is the single list, the parser is a
//! lookup in it, the error message is built from it, and
//! [`KvCacheFormat::is_valid_name`] is the parser returning `is_ok`. A test
//! pins the rendered string against the C++ literal, so the two stay
//! byte-identical while both exist.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// How a format quantizes, which is what the attention kernels switch on.
///
/// Discriminants mirror `kernels-cuda/csrc/src/kernels/kv_cache_view.hpp` --
/// the archive crate's host header, deleted with it at `85c6c674b` -- which
/// declared them without explicit values, so they are 0..4 in declaration
/// order, and this enum must keep that order for the same FFI reason
/// [`DType`] must keep its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KvCacheScheme {
    /// Stored in the activation dtype; no scales, no dequantization.
    Native = 0,
    /// FP8 with one scale for the whole tensor, known ahead of time.
    Fp8PerTensor = 1,
    /// INT8 with a scale per (token, head).
    Int8PerTokenHead = 2,
    /// FP8 with a scale per (token, head).
    Fp8PerTokenHead = 3,
    /// FP4 packed two-per-byte, with a scale per (token, head, block).
    Fp4Block = 4,
}

/// Where the side-scale buffer's entries live, and therefore how many there
/// are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KvCacheScaleLayout {
    /// No side scales at all; the scale buffer is not allocated.
    None = 0,
    /// One scale per (token, head).
    PerTokenHead = 1,
    /// One scale per (token, head, block-of-head-dim).
    PerTokenHeadBlock = 2,
}

/// The default block width when a `PerTokenHeadBlock` format does not state
/// one.
///
/// The C++ writes this as a bare `16` inside `scale_bytes_per_page`'s
/// `block_size > 0 ? block_size : 16`. It is only reachable through a
/// hand-built format, since every catalogue entry selecting
/// `PerTokenHeadBlock` sets `block_size = 16` explicitly -- but it is a number
/// that changes a page size, so it gets a name.
const DEFAULT_SCALE_BLOCK: u32 = 16;

/// A KV cache storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheFormat {
    name: &'static str,
    scheme: KvCacheScheme,
    scale_layout: KvCacheScaleLayout,
    storage_dtype: DType,
    block_size: u32,
}

impl Default for KvCacheFormat {
    /// Native bf16 -- the C++ `KvCacheFormat{}` default, which is what its
    /// parser returns for `auto`, `bf16` and `bfloat16` alike.
    fn default() -> Self {
        Self::BF16
    }
}

impl KvCacheFormat {
    /// Every format the driver accepts, in the order the C++ lists them.
    ///
    /// The order is the error message's order, so it is pinned by test rather
    /// than free. Aliases are separate rows: `bfloat16` resolves to the same
    /// format as `bf16` but is its own accepted spelling, and both appear in
    /// the message because the C++ message lists both.
    pub const CATALOGUE: &'static [(&'static str, KvCacheFormat)] = &[
        ("auto", Self::BF16),
        ("bf16", Self::BF16),
        ("bfloat16", Self::BF16),
        ("fp8_e4m3", Self::FP8_E4M3),
        ("fp8_e5m2", Self::FP8_E5M2),
        ("int8_per_token_head", Self::INT8_PER_TOKEN_HEAD),
        ("fp8_per_token_head", Self::FP8_PER_TOKEN_HEAD),
        ("fp4_e2m1", Self::FP4_E2M1),
        ("nvfp4", Self::NVFP4),
    ];

    /// Native bf16.
    pub const BF16: Self = Self {
        name: "bf16",
        scheme: KvCacheScheme::Native,
        scale_layout: KvCacheScaleLayout::None,
        storage_dtype: DType::Bf16,
        block_size: 0,
    };
    /// FP8 E4M3, one scale for the tensor.
    pub const FP8_E4M3: Self = Self {
        name: "fp8_e4m3",
        scheme: KvCacheScheme::Fp8PerTensor,
        scale_layout: KvCacheScaleLayout::None,
        storage_dtype: DType::Fp8E4M3,
        block_size: 0,
    };
    /// FP8 E5M2, one scale for the tensor.
    pub const FP8_E5M2: Self = Self {
        name: "fp8_e5m2",
        scheme: KvCacheScheme::Fp8PerTensor,
        scale_layout: KvCacheScaleLayout::None,
        storage_dtype: DType::Fp8E5M2,
        block_size: 0,
    };
    /// INT8 with a scale per (token, head).
    pub const INT8_PER_TOKEN_HEAD: Self = Self {
        name: "int8_per_token_head",
        scheme: KvCacheScheme::Int8PerTokenHead,
        scale_layout: KvCacheScaleLayout::PerTokenHead,
        storage_dtype: DType::Int8,
        block_size: 0,
    };
    /// FP8 E4M3 with a scale per (token, head).
    pub const FP8_PER_TOKEN_HEAD: Self = Self {
        name: "fp8_per_token_head",
        scheme: KvCacheScheme::Fp8PerTokenHead,
        scale_layout: KvCacheScaleLayout::PerTokenHead,
        storage_dtype: DType::Fp8E4M3,
        block_size: 0,
    };
    /// FP4 E2M1 in 16-wide blocks.
    pub const FP4_E2M1: Self = Self {
        name: "fp4_e2m1",
        scheme: KvCacheScheme::Fp4Block,
        scale_layout: KvCacheScaleLayout::PerTokenHeadBlock,
        storage_dtype: DType::Uint8,
        block_size: 16,
    };
    /// The same format as [`Self::FP4_E2M1`] under NVIDIA's spelling.
    ///
    /// A separate constant only because the C++ writes the *requested* alias
    /// into `name`, so `nvfp4` and `fp4_e2m1` produce formats that differ in
    /// that one string. Preserved rather than normalised: the name reaches
    /// logs and the plan, and a rewrite that changes what a log says is a
    /// rewrite that cannot be diffed against the original.
    pub const NVFP4: Self = Self {
        name: "nvfp4",
        ..Self::FP4_E2M1
    };

    /// Build a format from its fields, as the C++'s aggregate initialisation
    /// does.
    ///
    /// `KvCache::allocate(DType)` synthesises a format this way rather than
    /// looking one up by name, so the combinations reachable here are not
    /// limited to [`Self::CATALOGUE`]: a scale layout on a native dtype, or a
    /// blocked layout with `block_size == 0`, are both constructible and both
    /// change what a cache allocates.
    #[must_use]
    pub const fn from_parts(
        name: &'static str,
        scheme: KvCacheScheme,
        scale_layout: KvCacheScaleLayout,
        storage_dtype: DType,
        block_size: u32,
    ) -> Self {
        Self {
            name,
            scheme,
            scale_layout,
            storage_dtype,
            block_size,
        }
    }

    /// The format `KvCache::allocate(DType)` synthesises for a bare dtype.
    ///
    /// Native scheme, no scales, and a name that is `"bf16"` by literal rather
    /// than by `dtype_name` -- the two agree for BF16, but the C++ writes the
    /// literal, so a change to `dtype_name` would not follow here.
    #[must_use]
    pub const fn for_storage_dtype(dtype: DType) -> Self {
        Self {
            name: if matches!(dtype, DType::Bf16) {
                "bf16"
            } else {
                dtype.name()
            },
            scheme: KvCacheScheme::Native,
            scale_layout: KvCacheScaleLayout::None,
            storage_dtype: dtype,
            block_size: 0,
        }
    }

    /// The format's name, as the C++ records it.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// How this format quantizes.
    #[must_use]
    pub const fn scheme(&self) -> KvCacheScheme {
        self.scheme
    }

    /// Where the side scales live.
    #[must_use]
    pub const fn scale_layout(&self) -> KvCacheScaleLayout {
        self.scale_layout
    }

    /// The element type actually stored.
    #[must_use]
    pub const fn storage_dtype(&self) -> DType {
        self.storage_dtype
    }

    /// The scale block width, or `0` when the format has no blocked scales.
    #[must_use]
    pub const fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Is this the plain bf16 format?
    #[must_use]
    pub const fn is_native_bf16(&self) -> bool {
        matches!(self.scheme, KvCacheScheme::Native) && matches!(self.storage_dtype, DType::Bf16)
    }

    /// Does this format allocate a side-scale buffer?
    #[must_use]
    pub const fn has_side_scales(&self) -> bool {
        !matches!(self.scale_layout, KvCacheScaleLayout::None)
    }

    /// Storage elements in one token/head row.
    ///
    /// Equal to `head_dim` except for FP4, which packs two logical values per
    /// byte and therefore needs `ceil(head_dim / 2)` of them. This is the
    /// counterpart to [`DType::size_bytes`] reporting the byte rather than the
    /// logical element: one of the two has to know about the packing, and it
    /// is this one.
    #[must_use]
    pub const fn storage_head_dim(&self, head_dim: u32) -> u64 {
        match self.scheme {
            KvCacheScheme::Fp4Block => (head_dim as u64).div_ceil(2),
            _ => head_dim as u64,
        }
    }

    /// Bytes in one K page, or equivalently one V page.
    #[must_use]
    pub const fn kv_bytes_per_page(&self, page_size: u32, num_kv_heads: u32, head_dim: u32) -> u64 {
        page_size as u64
            * num_kv_heads as u64
            * self.storage_head_dim(head_dim)
            * self.storage_dtype.size_bytes() as u64
    }

    /// Bytes in one K-scale page, or zero when the format has no side scales.
    ///
    /// Scales are FP32 in this milestone, which the C++ states as a literal
    /// `dtype_bytes(DType::FP32)` rather than a field -- so per-format scale
    /// dtypes would be a change here, not a config.
    #[must_use]
    pub const fn scale_bytes_per_page(
        &self,
        page_size: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> u64 {
        let scales_per_head = match self.scale_layout {
            KvCacheScaleLayout::None => return 0,
            KvCacheScaleLayout::PerTokenHead => 1,
            KvCacheScaleLayout::PerTokenHeadBlock => {
                let block = if self.block_size > 0 {
                    self.block_size
                } else {
                    DEFAULT_SCALE_BLOCK
                };
                (head_dim as u64).div_ceil(block as u64)
            }
        };
        page_size as u64 * num_kv_heads as u64 * scales_per_head * DType::Fp32.size_bytes() as u64
    }

    /// Bytes for one whole page: K and V, plus both their scale planes.
    #[must_use]
    pub const fn total_bytes_per_page(
        &self,
        page_size: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> u64 {
        2 * self.kv_bytes_per_page(page_size, num_kv_heads, head_dim)
            + 2 * self.scale_bytes_per_page(page_size, num_kv_heads, head_dim)
    }

    /// Resolve a config string, case-insensitively. An empty string means
    /// `auto`.
    ///
    /// # The dropped parameter
    ///
    /// The C++ takes a second argument, `activation_dtype`, and does nothing
    /// with it: under `auto` it tests whether the activation dtype is bf16 and
    /// returns `KvCacheFormat{}` if it is, then returns `KvCacheFormat{}` if
    /// it is not, with a comment conceding the point ("Keep auto behavior
    /// identical to the historical path"). Every other branch ignores it
    /// outright. It cannot change the result, so it is not a parameter here.
    ///
    /// Dropped rather than kept-and-ignored deliberately: a parameter that
    /// looks like it selects something and does not is worse than no
    /// parameter, and this is exactly the sort of thing that survives a
    /// transliteration and dies in a rewrite.
    pub fn from_name(value: &str) -> Result<Self> {
        let requested = if value.is_empty() { "auto" } else { value };
        Self::CATALOGUE
            .iter()
            .find(|(alias, _)| alias.eq_ignore_ascii_case(requested))
            .map(|&(_, format)| format)
            .ok_or_else(|| {
                Error::invalid(
                    "kv_cache_format_from_string",
                    format!(
                        "invalid kv_cache_dtype '{value}'; expected one of: {}",
                        Self::valid_names()
                    ),
                )
            })
    }

    /// Whether [`Self::from_name`] would accept this string.
    #[must_use]
    pub fn is_valid_name(value: &str) -> bool {
        Self::from_name(value).is_ok()
    }

    /// Every accepted spelling, comma-separated, for an error message.
    ///
    /// Built from [`Self::CATALOGUE`] rather than written out, which is the
    /// point of the catalogue existing.
    #[must_use]
    pub fn valid_names() -> String {
        Self::CATALOGUE
            .iter()
            .map(|(alias, _)| *alias)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rendered_valid_names_still_match_the_cpp_string_exactly() {
        // The literal from `kv_cache_format.cpp`'s
        // `valid_kv_cache_dtype_values()`, which the C++ maintains by hand.
        // While both exist they must produce the same message, so this is a
        // differential pin, not a formatting preference.
        assert_eq!(
            KvCacheFormat::valid_names(),
            "auto, bf16, bfloat16, fp8_e4m3, fp8_e5m2, \
             int8_per_token_head, fp8_per_token_head, fp4_e2m1, nvfp4"
        );
    }

    #[test]
    fn every_catalogue_alias_parses_back_to_its_own_entry() {
        // The property the single table buys: parser and message cannot
        // disagree, because there is nothing for them to disagree about.
        for &(alias, expected) in KvCacheFormat::CATALOGUE {
            assert_eq!(
                KvCacheFormat::from_name(alias).unwrap(),
                expected,
                "alias {alias}"
            );
            assert!(KvCacheFormat::is_valid_name(alias));
        }
    }

    #[test]
    fn parsing_is_case_insensitive_and_empty_means_auto() {
        assert_eq!(
            KvCacheFormat::from_name("BF16").unwrap(),
            KvCacheFormat::BF16
        );
        assert_eq!(
            KvCacheFormat::from_name("Fp8_E4M3").unwrap(),
            KvCacheFormat::FP8_E4M3
        );
        assert_eq!(
            KvCacheFormat::from_name("NVFP4").unwrap(),
            KvCacheFormat::NVFP4
        );
        assert_eq!(KvCacheFormat::from_name("").unwrap(), KvCacheFormat::BF16);
        assert_eq!(
            KvCacheFormat::from_name("auto").unwrap(),
            KvCacheFormat::default()
        );
    }

    #[test]
    fn an_unknown_name_is_refused_and_the_message_lists_the_alternatives() {
        let err = KvCacheFormat::from_name("fp6").unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("fp6"),
            "the message must quote what was asked for: {text}"
        );
        assert!(
            text.contains("nvfp4"),
            "and list what was available: {text}"
        );
        assert!(!KvCacheFormat::is_valid_name("fp6"));
    }

    #[test]
    fn the_two_fp4_spellings_differ_in_name_and_in_nothing_else() {
        // Preserved C++ behaviour, and the one place `name` is not simply a
        // function of the format.
        let a = KvCacheFormat::FP4_E2M1;
        let b = KvCacheFormat::NVFP4;
        assert_ne!(a.name(), b.name());
        assert_eq!(a.scheme(), b.scheme());
        assert_eq!(a.scale_layout(), b.scale_layout());
        assert_eq!(a.storage_dtype(), b.storage_dtype());
        assert_eq!(a.block_size(), b.block_size());
        assert_eq!(
            a.total_bytes_per_page(16, 8, 128),
            b.total_bytes_per_page(16, 8, 128)
        );
    }

    #[test]
    fn bf16_pages_are_the_plain_product_with_no_scale_plane() {
        let f = KvCacheFormat::BF16;
        assert!(f.is_native_bf16());
        assert!(!f.has_side_scales());
        // 16 tokens * 8 heads * 128 dims * 2 bytes
        assert_eq!(f.kv_bytes_per_page(16, 8, 128), 16 * 8 * 128 * 2);
        assert_eq!(f.scale_bytes_per_page(16, 8, 128), 0);
        // K and V only; the scale planes contribute nothing.
        assert_eq!(f.total_bytes_per_page(16, 8, 128), 2 * 16 * 8 * 128 * 2);
    }

    #[test]
    fn fp4_packs_two_values_per_byte_and_rounds_an_odd_head_dim_up() {
        let f = KvCacheFormat::FP4_E2M1;
        assert_eq!(f.storage_head_dim(128), 64);
        assert_eq!(
            f.storage_head_dim(127),
            64,
            "ceil, so the last value keeps a byte"
        );
        assert_eq!(f.storage_head_dim(1), 1);
        assert_eq!(f.storage_head_dim(0), 0);
        // Half the bytes of an int8 cache of the same logical shape, which is
        // the entire reason the packing exists.
        assert_eq!(
            f.kv_bytes_per_page(16, 8, 128),
            KvCacheFormat::INT8_PER_TOKEN_HEAD.kv_bytes_per_page(16, 8, 128) / 2
        );
    }

    #[test]
    fn only_fp4_packs_the_head_dim() {
        for &(alias, f) in KvCacheFormat::CATALOGUE {
            if f.scheme() == KvCacheScheme::Fp4Block {
                continue;
            }
            assert_eq!(f.storage_head_dim(128), 128, "{alias} must not pack");
        }
    }

    #[test]
    fn per_token_head_scales_are_one_fp32_per_token_and_head() {
        let f = KvCacheFormat::INT8_PER_TOKEN_HEAD;
        assert!(f.has_side_scales());
        assert_eq!(f.scale_bytes_per_page(16, 8, 128), 16 * 8 * 4);
        // Independent of head_dim -- that is what "per token head" means, and
        // getting it wrong would scale the scale plane with the model.
        assert_eq!(f.scale_bytes_per_page(16, 8, 4096), 16 * 8 * 4);
    }

    #[test]
    fn blocked_scales_are_one_fp32_per_sixteen_dims_rounded_up() {
        let f = KvCacheFormat::FP4_E2M1;
        assert_eq!(f.block_size(), 16);
        assert_eq!(f.scale_bytes_per_page(16, 8, 128), 16 * 8 * 8 * 4);
        // 17 dims still needs two blocks; a truncating divide would under-size
        // the buffer by one scale per head and corrupt the last block.
        assert_eq!(f.scale_bytes_per_page(1, 1, 17), 2 * 4);
        assert_eq!(f.scale_bytes_per_page(1, 1, 16), 4);
        assert_eq!(f.scale_bytes_per_page(1, 1, 1), 4);
    }

    #[test]
    fn a_blocked_format_with_no_block_size_falls_back_to_sixteen() {
        // Only reachable through a hand-built format; the C++ spells the
        // fallback as a bare literal inside the size computation, so it is
        // pinned here rather than left to be rediscovered.
        let f = KvCacheFormat {
            block_size: 0,
            ..KvCacheFormat::FP4_E2M1
        };
        assert_eq!(
            f.scale_bytes_per_page(1, 1, 64),
            KvCacheFormat::FP4_E2M1.scale_bytes_per_page(1, 1, 64)
        );
    }

    #[test]
    fn a_total_page_is_two_kv_planes_and_two_scale_planes() {
        // Two of each, because K and V are stored and scaled separately. The
        // planner multiplies this by page count, so a factor dropped here is a
        // cache half the size it thinks it is.
        for &(alias, f) in KvCacheFormat::CATALOGUE {
            let kv = f.kv_bytes_per_page(16, 8, 128);
            let scale = f.scale_bytes_per_page(16, 8, 128);
            assert_eq!(
                f.total_bytes_per_page(16, 8, 128),
                2 * kv + 2 * scale,
                "{alias}"
            );
        }
    }

    #[test]
    fn a_zero_page_costs_nothing_in_every_format() {
        for &(alias, f) in KvCacheFormat::CATALOGUE {
            assert_eq!(f.total_bytes_per_page(0, 8, 128), 0, "{alias}");
            assert_eq!(f.total_bytes_per_page(16, 0, 128), 0, "{alias}");
        }
    }
}
