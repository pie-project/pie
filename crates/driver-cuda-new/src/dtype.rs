//! Storage element types, and how many bytes one of them costs.
//!
//! A faithful mirror of `kernels-cuda/csrc/src/tensor.hpp`'s `DType`, right
//! down to the discriminants -- see [`DType`] for why that is not incidental.
//!
//! # Why not one of the four `DType`s the workspace already has
//!
//! Because none of them is this one:
//!
//! * `tensor_ir::DType` is the DSL's arithmetic vocabulary: `f32`, `i32`,
//!   `u32`, `bool`. It describes what a kernel *computes in*, and it has no
//!   notion of a packed nibble or an exponent-only scale byte because nothing
//!   computes in those.
//! * `driver_abi::KvDtype` is the cross-node *wire* vocabulary, and is
//!   deliberately narrow -- five variants, no `FP8_E5M2`, no `UINT8`, both of
//!   which a KV cache format here can select. Widening it to fit would put
//!   local storage decisions into a type whose job is to be stable across a
//!   network.
//! * `model_loader::DType` is the closest in spirit -- storage dtypes with a
//!   `bytes()` -- but it describes what is *in a checkpoint file*, and it
//!   arrives through a crate carrying `ztensor`, `safetensors`, `gguf`,
//!   `pickle`, `hdf5` and `onnx`. That is a large dependency for a shell, and
//!   the wrong direction of dependency besides.
//!
//! The right long-run answer is probably that the loader's and this one's
//! converge. Until something forces that, this is the enum the C++ it is
//! replacing actually uses, which is the property that matters while both are
//! alive.

/// Element type of a stored tensor.
///
/// The discriminants are the C++ enum's, exactly, and that is load-bearing
/// rather than tidy: while the migration is in flight a `DType` will be handed
/// across the FFI boundary as a `uint8_t`, and the two sides agreeing on the
/// *numbers* is what makes that a cast instead of a translation table. A
/// variant may be added at the end; none may be renumbered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum DType {
    /// bfloat16.
    Bf16 = 0,
    /// IEEE half.
    Fp16 = 1,
    /// IEEE single.
    Fp32 = 2,
    /// Signed 8-bit integer.
    Int8 = 3,
    /// Signed 32-bit integer.
    Int32 = 4,
    /// Signed 64-bit integer.
    Int64 = 5,
    /// Unsigned 8-bit integer.
    Uint8 = 6,
    /// FP8 with a 4-bit exponent and 3-bit mantissa.
    Fp8E4M3 = 7,
    /// FP8 with a 5-bit exponent and 2-bit mantissa.
    Fp8E5M2 = 8,
    /// Marlin-packed INT4: one byte holds two nibbles.
    Int4Packed = 9,
    /// Marlin-packed MXFP4 (E2M1 values with E8M0 block scales).
    Mxfp4Packed = 10,
    /// OCP Microscaling's exponent-only scale byte: `b` denotes `2^(b - 127)`.
    /// Only ever a block-scale companion, never a weight.
    E8M0 = 11,
}

impl DType {
    /// Every dtype, in discriminant order.
    pub const ALL: &'static [DType] = &[
        DType::Bf16,
        DType::Fp16,
        DType::Fp32,
        DType::Int8,
        DType::Int32,
        DType::Int64,
        DType::Uint8,
        DType::Fp8E4M3,
        DType::Fp8E5M2,
        DType::Int4Packed,
        DType::Mxfp4Packed,
        DType::E8M0,
    ];

    /// Bytes one *storage* element occupies.
    ///
    /// Storage, not logical: the packed types report 1, because one byte is
    /// what they take, even though it carries two logical values. Callers that
    /// need the logical count divide separately -- which is exactly what
    /// [`crate::store::KvCacheFormat::storage_head_dim`] does.
    ///
    /// Total rather than a `Result`: the C++ closes an exhaustive `switch`
    /// with `throw std::runtime_error("unknown dtype")`, which is unreachable
    /// there and does not exist here, because `match` on a Rust enum has no
    /// unknown arm to fall out of.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::Int64 => 8,
            DType::Fp32 | DType::Int32 => 4,
            DType::Bf16 | DType::Fp16 => 2,
            DType::Int8
            | DType::Uint8
            | DType::Fp8E4M3
            | DType::Fp8E5M2
            | DType::Int4Packed
            | DType::Mxfp4Packed
            | DType::E8M0 => 1,
        }
    }

    /// The lowercase name the C++ `dtype_name` prints.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            DType::Bf16 => "bf16",
            DType::Fp16 => "fp16",
            DType::Fp32 => "fp32",
            DType::Int8 => "int8",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::Uint8 => "u8",
            DType::Fp8E4M3 => "fp8e4m3",
            DType::Fp8E5M2 => "fp8e5m2",
            DType::Int4Packed => "int4-packed",
            DType::Mxfp4Packed => "mxfp4-packed",
            DType::E8M0 => "e8m0",
        }
    }

    /// The dtype a C++ `uint8_t` tag names, or `None` if it names none.
    #[must_use]
    pub fn from_tag(tag: u8) -> Option<DType> {
        DType::ALL.get(usize::from(tag)).copied()
    }

    /// The tag the C++ side sees.
    #[must_use]
    pub const fn tag(self) -> u8 {
        self as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tags_are_the_cpp_enums_discriminants() {
        // Pinned literally, against `kernels-cuda/csrc/src/tensor.hpp`. These
        // numbers cross the FFI boundary during the migration, so a
        // reordering here is a silent misinterpretation there -- the kind that
        // shows up as a wrong-sized allocation rather than a compile error.
        assert_eq!(DType::Bf16.tag(), 0);
        assert_eq!(DType::Fp16.tag(), 1);
        assert_eq!(DType::Fp32.tag(), 2);
        assert_eq!(DType::Int8.tag(), 3);
        assert_eq!(DType::Int32.tag(), 4);
        assert_eq!(DType::Int64.tag(), 5);
        assert_eq!(DType::Uint8.tag(), 6);
        assert_eq!(DType::Fp8E4M3.tag(), 7);
        assert_eq!(DType::Fp8E5M2.tag(), 8);
        assert_eq!(DType::Int4Packed.tag(), 9);
        assert_eq!(DType::Mxfp4Packed.tag(), 10);
        assert_eq!(DType::E8M0.tag(), 11);
    }

    #[test]
    fn all_is_indexed_by_tag_which_is_what_makes_from_tag_a_lookup() {
        for (i, &d) in DType::ALL.iter().enumerate() {
            assert_eq!(usize::from(d.tag()), i, "{d:?} is out of position in ALL");
            assert_eq!(DType::from_tag(d.tag()), Some(d));
        }
        assert_eq!(DType::from_tag(12), None);
        assert_eq!(DType::from_tag(u8::MAX), None);
    }

    #[test]
    fn sizes_match_the_cpp_switch_including_the_packed_types() {
        assert_eq!(DType::Bf16.size_bytes(), 2);
        assert_eq!(DType::Fp16.size_bytes(), 2);
        assert_eq!(DType::Fp32.size_bytes(), 4);
        assert_eq!(DType::Int8.size_bytes(), 1);
        assert_eq!(DType::Int32.size_bytes(), 4);
        assert_eq!(DType::Int64.size_bytes(), 8);
        assert_eq!(DType::Uint8.size_bytes(), 1);
        assert_eq!(DType::Fp8E4M3.size_bytes(), 1);
        assert_eq!(DType::Fp8E5M2.size_bytes(), 1);
        // The packed pair report the byte they occupy, not the two logical
        // values in it. Everything downstream sizes buffers off this.
        assert_eq!(DType::Int4Packed.size_bytes(), 1);
        assert_eq!(DType::Mxfp4Packed.size_bytes(), 1);
        assert_eq!(DType::E8M0.size_bytes(), 1);
    }

    #[test]
    fn the_names_are_the_cpp_dtype_name_spellings_verbatim() {
        // Not a naming convention: these are the literals in
        // `kernels-cuda/csrc/src/tensor.hpp`'s `dtype_name`, and they are
        // inconsistent -- `u8` is abbreviated where `int8` is not, the FP8
        // pair drops the underscore that the KV format names keep, and the
        // packed pair uses a hyphen. Regularising any of them changes what a
        // synthesised `KvCacheFormat::name` reports.
        assert_eq!(DType::Uint8.name(), "u8");
        assert_eq!(DType::Fp8E4M3.name(), "fp8e4m3");
        assert_eq!(DType::Fp8E5M2.name(), "fp8e5m2");
        assert_eq!(DType::Int4Packed.name(), "int4-packed");
        assert_eq!(DType::Mxfp4Packed.name(), "mxfp4-packed");
        assert_eq!(DType::Int8.name(), "int8");
    }

    #[test]
    fn every_dtype_has_a_distinct_name() {
        let mut names: Vec<&str> = DType::ALL.iter().map(|d| d.name()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "two dtypes share a name");
    }
}
