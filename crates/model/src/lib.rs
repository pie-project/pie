pub mod adapter;
pub mod contract;
pub mod deepseek_v4;
pub mod gemma_4;
pub mod glm_5;
pub mod gpt_oss;
pub mod kimi_k3;
pub mod qwen_3;
pub mod template;

use checkpoint::contract::ModelContract;
use checkpoint::types::{DType, Encoding, QuantScheme, QuantSpec};
use model_dsl::Dtype;

/// The vocabulary a caller needs to USE the catalog's fourth column, through
/// the same door the column comes out of. A party that holds a [`ClassifyFn`]
/// has to build the [`Request`] it takes, and making it name `model_dsl` for
/// that would put the authoring eDSL in the dependency graph of everyone who
/// wants a lane's word — the runtime's fire path, which authors nothing.
pub use model_dsl::{ClassifyFn, Request};

/// One shipping SKU: its name, the tensor-parallel width it was traced for,
/// its trace, and how it sorts a request into the fact word a lane carries.
///
/// THE FOURTH COLUMN IS WHAT LETS A LANE STATE ITS CLASS. Nothing outside a
/// family's own module can say which bit `qo_one` is — a plan's
/// `Guard::Fact(bit)` numbers its bits and stops there — so before this column
/// existed the runtime's fire path submitted every lane as word 0, the
/// all-false class, and a decode lane composed as a prefill one. `catalog!`
/// closes each family's `Classify::of(..).word()` into a plain pointer here,
/// and the bit numbering stays private.
pub type Row = (&'static str, u32, model_dsl::TraceFn, model_dsl::ClassifyFn);

pub type ImportRow = (
    &'static str,
    fn(&ztensor::Source) -> Result<ModelContract, contract::ModelError>,
);

#[must_use]
pub fn catalog() -> Vec<Row> {
    [
        deepseek_v4::CATALOG,
        gemma_4::CATALOG,
        glm_5::CATALOG,
        gpt_oss::CATALOG,
        kimi_k3::CATALOG,
        qwen_3::CATALOG,
    ]
    .concat()
}

#[must_use]
pub fn imports() -> Vec<ImportRow> {
    [
        deepseek_v4::IMPORTS,
        gemma_4::IMPORTS,
        glm_5::IMPORTS,
        gpt_oss::IMPORTS,
        kimi_k3::IMPORTS,
        qwen_3::IMPORTS,
    ]
    .concat()
}

#[must_use]
pub fn trace_of(sku: &str) -> Option<model_dsl::TraceFn> {
    catalog()
        .into_iter()
        .find(|(n, ..)| *n == sku)
        .map(|(_, _, trace, _)| trace)
}

/// How `sku` sorts a request into the fact word its lanes carry.
///
/// Keyed by the same string as [`trace_of`], off the same rows, because a
/// build that classified a lane for one model and traced another would compose
/// a fire out of windows the plan does not have.
#[must_use]
pub fn classify_of(sku: &str) -> Option<model_dsl::ClassifyFn> {
    catalog()
        .into_iter()
        .find(|(n, ..)| *n == sku)
        .map(|(_, _, _, classify)| classify)
}

#[must_use]
pub fn import_of(
    sku: &str,
) -> Option<fn(&ztensor::Source) -> Result<ModelContract, contract::ModelError>> {
    imports()
        .into_iter()
        .find(|(n, _)| *n == sku)
        .map(|(_, make)| make)
}

/// The dtype the planes BESIDE a bank of `banks` are stated in.
///
/// **A NORM IS NOT A BANK, AND A QUANTIZED SKU IS NOT A SECOND FAMILY TEXT.**
/// A family's `new` takes one weight representation and stamps it on every
/// plane it declares, which was right while every representation stored itself
/// verbatim. `Mxfp4` and `MlxU4` do not: they are a bank's codes, they come
/// with companion planes, and no checkpoint in either scheme quantizes a
/// layernorm — MLX's own rule is that a group of sixty-four codes needs
/// sixty-four columns to group, and a `[hidden]` norm has one axis and no
/// contracted one at all.
///
/// So the text asks here rather than forking. `layer.0.q_proj` is stated in
/// `banks` and `layer.0.mixer_norm` in what `banks` MULTIPLIES AS, which for
/// every unpacked representation is `banks` itself — so a bf16 SKU declares
/// exactly the weights it always declared, byte for byte, and the quantized
/// row beside it is the same sentences with one word changed.
pub(crate) fn dense(banks: Dtype) -> Dtype {
    model_dsl::compute_dtype(banks)
        .unwrap_or_else(|| panic!("`{banks:?}` is not a weight representation a family declares"))
}

pub(crate) fn encoding(dtype: Dtype) -> Encoding {
    match dtype {
        Dtype::Bf16 => Encoding::Raw(DType::Bf16),
        Dtype::F16 => Encoding::Raw(DType::F16),
        Dtype::F32 => Encoding::Raw(DType::F32),
        Dtype::I32 => Encoding::Raw(DType::I32),
        Dtype::U32 => Encoding::Raw(DType::U32),
        Dtype::U8 => Encoding::Raw(DType::U8),
        Dtype::I8 => Encoding::Raw(DType::I8),
        Dtype::Fp8E4m3 => Encoding::Raw(DType::Fp8E4m3),
        Dtype::E8m0 => Encoding::Raw(DType::E8m0),
        // The six the checkpoint vocabulary brought when the two dtype enums
        // merged. Each stores itself verbatim, so each is `Raw` of itself —
        // the same answer every row above gives.
        Dtype::Fp8E5m2 => Encoding::Raw(DType::Fp8E5m2),
        Dtype::I64 => Encoding::Raw(DType::I64),
        Dtype::I16 => Encoding::Raw(DType::I16),
        Dtype::U64 => Encoding::Raw(DType::U64),
        Dtype::U16 => Encoding::Raw(DType::U16),
        Dtype::Bool => Encoding::Raw(DType::Bool),
        Dtype::Mxfp4 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: None,
        }),
        // MLX's affine U4, in the loader's own vocabulary: sixty-four codes
        // under one bf16 scale and one bf16 offset, dequantized to bf16. The
        // channel axis is stated by whoever holds the shape —
        // `contract::grouped` — because a rank is not a fact about a scheme.
        Dtype::MlxU4 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 64,
            channel_axis: None,
        }),
        // **THE SAME SCHEME AT TWICE THE WIDTH.** `MlxAffineU4` names the
        // arithmetic — affine codes, sixty-four to a group, one bf16 scale and
        // one bf16 offset apiece — and `bits_per_element` has always been the
        // field that says how wide a code is, which is why an eight-bit MLX
        // bank needs no scheme of its own. `Landing::affine_point_of` reports
        // this number to the engine and `kernels_metal::linear::quant` stamps
        // a point at both widths, so the two travel the whole way down as one
        // path with a number in it. See `dtype::Dtype::MlxU8` for the
        // checkpoint that mixes them.
        Dtype::MlxU8 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 8,
            group_size: 64,
            channel_axis: None,
        }),
        Dtype::Fp4 => panic!(
            "`Dtype::Fp4` names a kv-page quantization scheme, not a stored \
             weight plane; no load contract declares one"
        ),
    }
}

pub fn identify(src: &ztensor::Source) -> Result<&'static str, Unmatched> {
    let mut misses: Vec<(&'static str, String)> = Vec::new();
    let rows = catalog();

    for (sku, import) in imports() {
        if rows.iter().any(|row| row.0 == sku && row.1 > 1) {
            continue;
        }
        match import(src) {
            Ok(_) => return Ok(sku),
            Err(why) => misses.push((sku, why.to_string())),
        }
    }
    Err(Unmatched { misses })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unmatched {
    pub misses: Vec<(&'static str, String)>,
}

impl std::fmt::Display for Unmatched {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "this checkpoint matches no SKU this build ships")?;
        for (sku, why) in &self.misses {
            write!(f, "\n  {sku}: {why}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Unmatched {}
