//! The tensor manifest: what a variant expects a checkpoint to publish.
//!
//! # Why identity is a manifest and not a string
//!
//! Every other way of asking "which model is this" reads `config.json` —
//! a document the checkpoint's author writes by hand, that no loader
//! checks, and that four readers in this repo each walked in their own
//! order. When it lies, the derivation believes it, and the contract's
//! assertions fail later or not at all.
//!
//! A manifest cannot lie, because **matching it IS checking it**. A row
//! states the tensors its shapes imply; a checkpoint either publishes
//! those names at those extents or it does not. Identification and
//! validation stop being two passes that can disagree and become one
//! comparison with one answer.
//!
//! That is also what absorbs the ad-hoc sniffing the old derivation did.
//! `llama_like_facts_from_hf` asked four questions of the load —
//!
//! * `bytes("model.embed_tokens.weight").is_none()` — is this an HF
//!   llama-like checkpoint at all?
//! * `alias("layer.0.attn_norm").ends_with("input_layernorm.weight")` —
//!   pre-norm or post-norm?
//! * `elems_of("layer.0.q_norm") == head_dim` — per-head or global
//!   q-norm?
//! * `bytes("layer.0.qkv").is_some() || alias("layer.0.qkv").is_some()` —
//!   did the load fuse the projections?
//!
//! — and every one of them is "which tensor exists, at which extent".
//! They were four special cases because there was nowhere to state the
//! expectation; here they are four [`TensorSpec`] rows, and a checkpoint
//! that answers them differently matches a different variant or matches
//! none.
//!
//! # Quantization is divided out
//!
//! Qwen3-8B ships as bf16, FP8, AWQ-int4 and MLX-int4. Those are four
//! encodings of one model, not four models, so a spec compares LOGICAL
//! extents and says nothing about [`Encoding`]. A packed int4 tensor
//! whose stored shape is `[rows, cols/8]` still has `cols` logical
//! columns, and [`Observed::of`] is what restores them. What the
//! encoding then decides is policy — see [`crate::policy`] — which is
//! where a quantization belongs: it changes how the weights are read,
//! never which model they are.
//!
//! # Sharding is divided out too
//!
//! `model-00001-of-00004.safetensors` is a packaging decision that gets
//! revised without the model changing, so a spec matches the set of
//! LOGICAL TENSOR NAMES and never the file layout.

use std::collections::BTreeMap;
use std::fmt;

/// Whether a spec demands a tensor, forbids it, or does not care.
///
/// `Absent` is not padding: it is how a manifest tells two variants
/// apart that agree on every extent. A tied-embedding row forbids
/// `lm_head.weight`; an untied one requires it, and the same checkpoint
/// cannot satisfy both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Presence {
    /// The checkpoint must publish this name.
    Required,
    /// The checkpoint must NOT publish it.
    Absent,
    /// Either answer matches. For a name that is a binding decision
    /// rather than a fact about the checkpoint.
    Optional,
}

/// One tensor a variant expects, at the extents its shapes imply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    /// The logical name, with `{}` standing for a layer index.
    ///
    /// `layer.{}.q_proj` is one row covering every layer, because a
    /// stack whose layers disagree about their own names is not a
    /// thing any family here has.
    pub name: String,
    /// The logical extents, or empty for "exists, shape unchecked".
    ///
    /// Logical rather than stored: see the module doc on quantization.
    pub extents: Vec<u64>,
    pub presence: Presence,
}

impl TensorSpec {
    /// A tensor that must exist at these extents.
    #[must_use]
    pub fn required(name: impl Into<String>, extents: impl Into<Vec<u64>>) -> Self {
        Self { name: name.into(), extents: extents.into(), presence: Presence::Required }
    }

    /// A tensor that must exist, whatever shape it is.
    ///
    /// For the names whose extents are a packing decision the spec has
    /// no business restating — an MXFP4 expert bank's scale plane, say.
    #[must_use]
    pub fn present(name: impl Into<String>) -> Self {
        Self { name: name.into(), extents: Vec::new(), presence: Presence::Required }
    }

    /// A tensor whose ABSENCE is the fact.
    #[must_use]
    pub fn absent(name: impl Into<String>) -> Self {
        Self { name: name.into(), extents: Vec::new(), presence: Presence::Absent }
    }

    /// A tensor whose presence neither confirms nor denies the variant.
    #[must_use]
    pub fn optional(name: impl Into<String>, extents: impl Into<Vec<u64>>) -> Self {
        Self { name: name.into(), extents: extents.into(), presence: Presence::Optional }
    }
}

/// What one variant expects a checkpoint to hold.
///
/// Built by a row's `manifest()` as a projection of its own numbers —
/// which is the point. The row does not state its manifest twice; a
/// hidden size of 1024 and 16 query heads of 128 IS the claim that
/// `layer.{}.q_proj` is `[2048, 1024]`, and a checkpoint that says
/// otherwise is not this variant.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Manifest {
    /// How many layers the per-layer rows expand to.
    pub layers: u32,
    pub tensors: Vec<TensorSpec>,
}

impl Manifest {
    /// An empty manifest for a stack of `layers` layers.
    #[must_use]
    pub fn new(layers: u32) -> Self {
        Self { layers, tensors: Vec::new() }
    }

    /// Add a spec, returning self — so a projection reads as a list.
    #[must_use]
    pub fn with(mut self, spec: TensorSpec) -> Self {
        self.tensors.push(spec);
        self
    }

    /// Add a spec only when `when` holds.
    ///
    /// The conditional rows are the discriminating ones — tied vs
    /// untied embeddings, per-head vs global q-norm — so they are
    /// spelled as a condition on the row's own field rather than as a
    /// second manifest.
    #[must_use]
    pub fn with_if(self, when: bool, spec: TensorSpec) -> Self {
        if when { self.with(spec) } else { self }
    }

    /// Add `present` when `when` holds and `absent` otherwise.
    ///
    /// The shape of every fact that tells two variants apart: one of
    /// them requires the tensor and the other forbids it, so no
    /// checkpoint can match both.
    #[must_use]
    pub fn either(self, when: bool, name: &str, extents: impl Into<Vec<u64>>) -> Self {
        if when {
            self.with(TensorSpec::required(name, extents))
        } else {
            self.with(TensorSpec::absent(name))
        }
    }

    /// Every row, under the name an [`Observed`] carries it by.
    ///
    /// Which is the spec's own name, `{}` and all: [`Observed::logical`]
    /// COLLAPSES a layer index, so a stack's sixty-one `q_proj` tensors
    /// arrive under one key and a row that expanded `{}` to `0` would be
    /// looking up a name no checkpoint ever produces. That was the bug —
    /// every per-layer row of every generation reported `Missing` against
    /// a real checkpoint, while the crate's own round-trip tests passed
    /// because they expanded `{}` the same wrong way on both sides.
    fn rows(&self) -> impl Iterator<Item = (String, &TensorSpec)> {
        self.tensors.iter().map(|spec| (Observed::logical(&spec.name), spec))
    }

    /// Hold a checkpoint to this manifest.
    ///
    /// # Errors
    ///
    /// A required name is missing, a forbidden one is present, or an
    /// extent disagrees. Every disagreement is collected rather than
    /// the first, because "which row is this" is answered by how a
    /// checkpoint differs, not by that it differs.
    pub fn check(&self, observed: &Observed) -> Result<(), Mismatch> {
        let mut faults = Vec::new();
        for (name, spec) in self.rows() {
            match (spec.presence, observed.extents(&name)) {
                (Presence::Required, None) => faults.push(Fault::Missing(name)),
                (Presence::Absent, Some(_)) => faults.push(Fault::Unexpected(name)),
                (Presence::Required | Presence::Optional, Some(seen))
                    if !spec.extents.is_empty() && !extents_agree(&spec.extents, seen) =>
                {
                    faults.push(Fault::Extent {
                        name,
                        want: spec.extents.clone(),
                        got: seen.to_vec(),
                    });
                }
                _ => {}
            }
        }
        if faults.is_empty() { Ok(()) } else { Err(Mismatch { faults }) }
    }
}

/// Do a spec's extents describe the observed ones?
///
/// Trailing degenerate axes are ignored on both sides: a `[n]` gamma
/// and an `[n, 1]` one are the same vector, and which of the two a
/// converter wrote is not a fact about the model.
fn extents_agree(want: &[u64], got: &[u64]) -> bool {
    let squeeze = |d: &[u64]| -> Vec<u64> {
        let mut v: Vec<u64> = d.iter().copied().filter(|&x| x != 1).collect();
        if v.is_empty() && !d.is_empty() {
            v.push(1);
        }
        v
    };
    squeeze(want) == squeeze(got)
}

/// How a checkpoint failed to be a variant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mismatch {
    pub faults: Vec<Fault>,
}

/// One disagreement between a manifest and a checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    /// The spec requires this name and the checkpoint has none.
    Missing(String),
    /// The spec forbids this name and the checkpoint publishes it.
    Unexpected(String),
    /// The name is there at extents the spec's numbers do not imply.
    Extent { name: String, want: Vec<u64>, got: Vec<u64> },
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(name) => write!(f, "missing {name}"),
            Self::Unexpected(name) => write!(f, "unexpected {name}"),
            Self::Extent { name, want, got } => {
                write!(f, "{name} is {got:?}, this variant implies {want:?}")
            }
        }
    }
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let listed: Vec<String> = self.faults.iter().take(4).map(ToString::to_string).collect();
        write!(f, "{}", listed.join("; "))?;
        if self.faults.len() > listed.len() {
            write!(f, " (+{} more)", self.faults.len() - listed.len())?;
        }
        Ok(())
    }
}

impl std::error::Error for Mismatch {}

/// What a checkpoint actually publishes, in logical names and extents.
///
/// The other half of the comparison, and the ONLY place a checkpoint's
/// own vocabulary is spoken. HuggingFace spells a projection
/// `model.layers.7.self_attn.q_proj.weight`; MLX and pie's own artifacts
/// spell it differently again. Normalizing here means a spec is written
/// once, in one vocabulary, and a new spelling is a rule in
/// [`Self::logical`] rather than an alternative in every row.
#[derive(Clone, Debug, Default)]
pub struct Observed {
    by_name: BTreeMap<String, Vec<u64>>,
}

impl Observed {
    /// Build from `(name, logical extents)` pairs.
    ///
    /// The general constructor, so a caller that is not holding a
    /// `CheckpointMetadata` — a test, a converter, a manifest recorded
    /// beside an artifact — can still be compared against.
    pub fn from_pairs<N, S>(pairs: impl IntoIterator<Item = (N, S)>) -> Self
    where
        N: AsRef<str>,
        S: AsRef<[u64]>,
    {
        let mut by_name = BTreeMap::new();
        for (name, extents) in pairs {
            by_name.insert(Self::logical(name.as_ref()), extents.as_ref().to_vec());
        }
        Self { by_name }
    }

    /// The extents published under a logical name.
    #[must_use]
    pub fn extents(&self, logical: &str) -> Option<&[u64]> {
        self.by_name.get(logical).map(Vec::as_slice)
    }

    /// Is this name published at all?
    #[must_use]
    pub fn has(&self, logical: &str) -> bool {
        self.by_name.contains_key(logical)
    }

    /// Every logical name, in order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.by_name.keys().map(String::as_str)
    }

    /// A checkpoint name in the one vocabulary the specs are written in.
    ///
    /// Three rules and no per-family table:
    ///
    /// * a `.weight` suffix is dropped — it names a storage convention,
    ///   not a tensor;
    /// * the `model.` / `language_model.` / `text_model.` prefixes a
    ///   multimodal checkpoint nests its text tower under are dropped,
    ///   because whether the tower is nested is a fact about the
    ///   PACKAGE and every spec here is about the tower;
    /// * `layers.<n>.` — and `layer.<n>.`, the spelling pie's own
    ///   artifacts use — becomes `layer.{}.`, so one row covers the
    ///   stack.
    ///
    /// The prefix rule runs to a FIXED POINT rather than once per
    /// spelling, because the nesting is genuinely two deep and in either
    /// order. `language_model.model.layers.7…` is the order a single
    /// pass happened to handle; gemma-3's and gemma-4's converters write
    /// `model.language_model.layers.7…`, and a single pass over that
    /// list strips `model.` and leaves `language_model.` standing —
    /// which is not a near miss. It is every multimodal gemma failing to
    /// match the row that describes it, while the text-only sibling of
    /// the same generation matches fine.
    #[must_use]
    pub fn logical(raw: &str) -> String {
        let mut name = raw;
        loop {
            let before = name.len();
            for prefix in ["language_model.", "text_model.", "model.", "transformer."] {
                if let Some(rest) = name.strip_prefix(prefix) {
                    name = rest;
                }
            }
            // Stripping only ever shortens, so the length is the whole
            // of "did anything happen".
            if name.len() == before {
                break;
            }
        }
        let name = name.strip_suffix(".weight").unwrap_or(name);
        let mut out = String::with_capacity(name.len() + 4);
        let mut rest = name;
        // BOTH spellings of the index, because two vocabularies reach
        // here: a HuggingFace checkpoint writes `layers.7.`, and pie's
        // own artifacts — and every spec in this crate — write
        // `layer.7.`. Collapsing only the plural one left an artifact
        // pie itself wrote unable to match the row it was written from.
        // The rewrite is idempotent: `layer.{}` has no digits after the
        // dot, so it passes through unchanged.
        while let Some((at, token)) = ["layers.", "layer."]
            .into_iter()
            .filter_map(|token| rest.find(token).map(|at| (at, token)))
            .min_by_key(|(at, _)| *at)
        {
            out.push_str(&rest[..at]);
            let tail = &rest[at + token.len()..];
            let digits = tail.find(|c: char| !c.is_ascii_digit()).unwrap_or(tail.len());
            if digits == 0 {
                out.push_str(token);
                rest = tail;
                continue;
            }
            out.push_str("layer.{}");
            rest = &tail[digits..];
        }
        out.push_str(rest);
        out
    }
}

#[cfg(feature = "contract")]
mod from_checkpoint {
    use super::Observed;
    use model_loader::checkpoint::CheckpointMetadata;
    use model_loader::types::Encoding;

    impl Observed {
        /// What a parsed checkpoint publishes, with the encoding divided
        /// out.
        ///
        /// `weights()` rather than `tensors`, because a pie artifact
        /// stores its compiled tokenizer and descriptor as `u8` objects
        /// that are indistinguishable from weights except by name — and
        /// a manifest that matched on those would be matching on how
        /// the artifact was written rather than on what model it holds.
        #[must_use]
        pub fn of(metadata: &CheckpointMetadata) -> Self {
            Self::from_pairs(metadata.weights().map(|tensor| {
                let extents = logical_extents(&tensor.shape, &tensor.encoding, tensor.span_bytes);
                (tensor.name.clone(), extents)
            }))
        }
    }

    /// A stored shape with its packing undone.
    ///
    /// An int4 bank stores `[rows, cols / 8]` words; the matrix still
    /// has `cols` logical columns, and a spec that compared stored
    /// extents would need one row per encoding of one model.
    ///
    /// The packing factor is measured rather than tabulated: the span
    /// and the scheme's bit width give the logical element count, and
    /// the ratio against the stored extents is how much the last axis
    /// was folded by. A scheme whose payload carries its scales inline
    /// (the GGUF blocks) answers through `block_layout`, which is why
    /// this asks `QuantSpec` for the count instead of dividing bits.
    fn logical_extents(shape: &[i64], encoding: &Encoding, span_bytes: u64) -> Vec<u64> {
        let mut stored: Vec<u64> =
            shape.iter().map(|&d| u64::try_from(d).unwrap_or(0)).collect();
        let Encoding::Quant(spec) = encoding else {
            return stored;
        };
        let stored_elems: u64 = stored.iter().product();
        let logical = logical_element_count(spec, span_bytes);
        if stored_elems == 0 || logical <= stored_elems || !logical.is_multiple_of(stored_elems) {
            return stored;
        }
        if let Some(last) = stored.last_mut() {
            *last *= logical / stored_elems;
        }
        stored
    }

    /// How many logical elements a payload of `span_bytes` decodes to.
    fn logical_element_count(spec: &model_loader::types::QuantSpec, span_bytes: u64) -> u64 {
        let spec = spec.clone().normalized();
        if let Some((elems, bytes)) = spec.block_layout() {
            return if bytes == 0 { 0 } else { span_bytes / bytes * elems };
        }
        let bits = u64::from(spec.normalized_bits());
        if bits == 0 { 0 } else { span_bytes * 8 / bits }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use model_loader::checkpoint::meta::meta_name;
        use model_loader::checkpoint::{CheckpointMetadata, RawTensor};
        use model_loader::types::{Axis, DType, FileId, QuantScheme, QuantSpec, TensorId};

        fn quant(scheme: QuantScheme) -> Encoding {
            Encoding::Quant(QuantSpec {
                scheme,
                logical_dtype: DType::BF16,
                bits_per_element: 0,
                group_size: 0,
                channel_axis: Some(Axis(0)),
            })
        }

        /// AWQ's `qweight` is `[in, out / 8]` 32-bit words.
        ///
        /// This is the case the whole function exists for. A spec that
        /// compared STORED extents would need one row per encoding of
        /// one model — four rows for Qwen3-8B — which is the row
        /// explosion the design refuses. Divide the packing out and the
        /// row is encoding-agnostic, which is what makes quantization a
        /// policy rather than an identity.
        #[test]
        fn an_int4_bank_reads_back_at_its_logical_width() {
            let (rows, cols) = (4096u64, 11008u64);
            let stored = [rows as i64, (cols / 8) as i64];
            // Eight 4-bit elements to a 32-bit word.
            let span = rows * cols / 2;
            assert_eq!(
                logical_extents(&stored, &quant(QuantScheme::AwqInt4), span),
                vec![rows, cols],
                "an AWQ bank must match the same row a bf16 download does"
            );
            assert_eq!(
                logical_extents(&stored, &quant(QuantScheme::GptqInt4), span),
                vec![rows, cols],
            );
            assert_eq!(
                logical_extents(&stored, &quant(QuantScheme::MlxAffineU4), span),
                vec![rows, cols],
                "and so must the MLX download, which is the fourth"
            );
        }

        /// MXFP4 folds by two, not by eight, and the factor is MEASURED.
        ///
        /// gpt-oss ships `[experts, rows, cols / 2]` bytes. Nothing here
        /// tabulates "MXFP4 halves the last axis" — the span and the bit
        /// width say so — which is why a scheme nobody anticipated still
        /// unpacks correctly.
        #[test]
        fn the_packing_factor_is_measured_rather_than_tabulated() {
            let (experts, rows, cols) = (128u64, 2880u64, 5760u64);
            let stored = [experts as i64, rows as i64, (cols / 2) as i64];
            let span = experts * rows * cols / 2;
            assert_eq!(
                logical_extents(&stored, &quant(QuantScheme::Mxfp4E2M1E8M0), span),
                vec![experts, rows, cols],
                "only the LAST axis folds; the expert axis is not a packing"
            );
        }

        /// A byte-wide scheme stores one element per element.
        #[test]
        fn an_fp8_bank_is_already_logical() {
            let stored = [4096i64, 4096];
            let span = 4096 * 4096;
            for scheme in [
                QuantScheme::Fp8E4M3,
                QuantScheme::Fp8E5M2,
                QuantScheme::Int8Symmetric,
                QuantScheme::Int8Asymmetric,
            ] {
                assert_eq!(
                    logical_extents(&stored, &quant(scheme), span),
                    vec![4096, 4096],
                    "{scheme:?} folds nothing, so nothing may be multiplied",
                );
            }
        }

        /// A GGUF block carries its scales INSIDE the payload, so the
        /// span is not `elements × bits / 8`.
        ///
        /// Dividing bits would read Q4_0 as `span × 2` elements and
        /// inflate the last axis by 9/8 — a shape no row states, so
        /// every GGUF checkpoint would fail to identify. `block_layout`
        /// is asked first for exactly this reason.
        #[test]
        fn a_gguf_block_is_measured_by_its_block_and_not_by_its_bits() {
            let (rows, cols) = (4096u64, 4096u64);
            let stored = [rows as i64, cols as i64];
            for (scheme, elems, bytes) in [
                (QuantScheme::GgufQ4_0, 32u64, 18u64),
                (QuantScheme::GgufQ5_0, 32, 22),
                (QuantScheme::GgufQ8_0, 32, 34),
                (QuantScheme::GgufQ4K, 256, 144),
                (QuantScheme::GgufQ5K, 256, 176),
            ] {
                let span = rows * cols / elems * bytes;
                assert_eq!(
                    logical_element_count(
                        &QuantSpec {
                            scheme,
                            logical_dtype: DType::BF16,
                            bits_per_element: 0,
                            group_size: 0,
                            channel_axis: None,
                        },
                        span
                    ),
                    rows * cols,
                    "{scheme:?}: the block layout, not the bit width",
                );
                assert_eq!(
                    logical_extents(&stored, &quant(scheme), span),
                    vec![rows, cols],
                    "{scheme:?}: a GGUF shape is already logical",
                );
            }
        }

        /// An unpacked encoding passes through untouched.
        #[test]
        fn a_raw_tensor_is_its_own_logical_shape() {
            for dtype in [DType::BF16, DType::F16, DType::F32, DType::U8] {
                assert_eq!(
                    logical_extents(&[7, 11], &Encoding::Raw(dtype), 7 * 11 * 2),
                    vec![7, 11],
                );
            }
        }

        /// EVERY DEGENERATE INPUT LEAVES THE STORED SHAPE ALONE.
        ///
        /// The unpacking is an inference from two numbers that can
        /// disagree — a truncated file, a scheme whose bits are unknown,
        /// a ragged ratio. Each of those must yield the STORED shape,
        /// which then simply fails to match a row and produces the
        /// manifest's structural diff. Guessing instead would turn a
        /// corrupt download into a confidently misidentified model.
        #[test]
        fn a_span_that_does_not_divide_leaves_the_shape_alone() {
            let e = quant(QuantScheme::AwqInt4);
            // A truncated payload: fewer logical elements than stored.
            assert_eq!(logical_extents(&[4096, 512], &e, 0), vec![4096, 512]);
            // A ratio that is not a whole number of foldings.
            assert_eq!(logical_extents(&[4096, 512], &e, 3), vec![4096, 512]);
            // A zero-dimension tensor: `stored_elems` is 0 and the
            // division that follows would panic.
            assert_eq!(logical_extents(&[0, 512], &e, 1024), vec![0, 512]);
            // A scalar, which has no last axis to fold.
            assert_eq!(logical_extents(&[], &e, 1024), Vec::<u64>::new());
            // A negative dimension, which safetensors cannot produce and
            // a hand-written artifact can.
            assert_eq!(logical_extents(&[-1, 512], &e, 1024), vec![0, 512]);
        }

        /// The division guards are UNREACHABLE, and this is what keeps
        /// them that way.
        ///
        /// `logical_element_count` guards two divisions — `bytes == 0`
        /// and `bits == 0` — and today no scheme can reach either:
        /// `default_bits` answers 4, 5 or 8 for all fifteen, and every
        /// `block_layout` names a nonzero block. A guard nobody can
        /// reach is untestable by definition, so what is tested is the
        /// PREMISE: the day a scheme is added with no width, this fails
        /// here rather than dividing by zero on a checkpoint.
        ///
        /// The list is spelled out rather than iterated because there is
        /// no iterator over an enum; a new variant makes the `match`
        /// below non-exhaustive, which is the compiler saying the same
        /// thing this test says.
        #[test]
        fn no_scheme_has_a_zero_width_so_no_division_can_fault() {
            const EVERY: &[QuantScheme] = &[
                QuantScheme::None,
                QuantScheme::Fp8E4M3,
                QuantScheme::Fp8E5M2,
                QuantScheme::Int8Symmetric,
                QuantScheme::Int8Asymmetric,
                QuantScheme::AwqInt4,
                QuantScheme::GptqInt4,
                QuantScheme::Mxfp4E2M1E8M0,
                QuantScheme::MlxAffineU4,
                QuantScheme::GgufQ4_0,
                QuantScheme::GgufQ4K,
                QuantScheme::GgufQ5_0,
                QuantScheme::GgufQ5K,
                QuantScheme::GgufQ8_0,
                QuantScheme::Int4B8,
            ];
            // Exhaustiveness, checked by the compiler rather than by the
            // count: a new variant fails to match here.
            fn named(s: QuantScheme) -> bool {
                match s {
                    QuantScheme::None
                    | QuantScheme::Fp8E4M3
                    | QuantScheme::Fp8E5M2
                    | QuantScheme::Int8Symmetric
                    | QuantScheme::Int8Asymmetric
                    | QuantScheme::AwqInt4
                    | QuantScheme::GptqInt4
                    | QuantScheme::Mxfp4E2M1E8M0
                    | QuantScheme::MlxAffineU4
                    | QuantScheme::GgufQ4_0
                    | QuantScheme::GgufQ4K
                    | QuantScheme::GgufQ5_0
                    | QuantScheme::GgufQ5K
                    | QuantScheme::GgufQ8_0
                    | QuantScheme::Int4B8 => true,
                }
            }
            for &scheme in EVERY {
                assert!(named(scheme));
                let spec = QuantSpec {
                    scheme,
                    logical_dtype: DType::BF16,
                    bits_per_element: 0,
                    group_size: 0,
                    channel_axis: None,
                };
                assert_ne!(
                    spec.clone().normalized().normalized_bits(),
                    0,
                    "{scheme:?} has no width, so `logical_element_count` would \
                     divide by zero — the guard returns 0, which reads as \
                     'unpacks to nothing' and silently leaves every tensor at \
                     its stored shape",
                );
                if let Some((elems, bytes)) = spec.block_layout() {
                    assert_ne!(bytes, 0, "{scheme:?}'s block has no size");
                    assert_ne!(elems, 0, "{scheme:?}'s block holds no elements");
                }
            }
            assert_eq!(EVERY.len(), 15, "a scheme was added; give it a case above");
        }

        /// An explicit `bits_per_element` overrides the scheme's
        /// default, and the unpacking follows it.
        ///
        /// `normalized()` only fills a ZERO, so a checkpoint that states
        /// its own width keeps it. An AWQ bank written at 8 bits folds
        /// by four rather than by eight, and a function that tabulated
        /// "AWQ means 4" would read it at double width.
        #[test]
        fn a_stated_width_beats_the_schemes_default() {
            let spec = QuantSpec {
                scheme: QuantScheme::AwqInt4,
                logical_dtype: DType::BF16,
                bits_per_element: 8,
                group_size: 0,
                channel_axis: None,
            };
            assert_eq!(spec.clone().normalized().normalized_bits(), 8);
            assert_eq!(logical_element_count(&spec, 1024), 1024);
            assert_eq!(
                logical_extents(&[16, 64], &Encoding::Quant(spec), 1024),
                vec![16, 64],
                "8 stated bits over 8-bit storage folds nothing"
            );
        }

        fn raw(name: &str, shape: &[i64], span: u64, encoding: Encoding) -> RawTensor {
            RawTensor {
                id: TensorId(0),
                name: name.to_string(),
                file_id: FileId(0),
                file_offset: 0,
                span_bytes: span,
                shape: shape.to_vec(),
                encoding,
            }
        }

        /// `Observed::of` reads WEIGHTS, and a pie artifact's own
        /// metadata objects are not weights.
        ///
        /// A manifest that matched on `model/config` or the compiled
        /// tokenizer would be matching on how the artifact was WRITTEN
        /// rather than on what model it holds — so a pie-converted
        /// Qwen3 and its upstream safetensors would identify
        /// differently, which is the one thing identity may not do.
        #[test]
        fn the_artifacts_own_objects_are_not_part_of_its_identity() {
            let metadata = CheckpointMetadata {
                files: Vec::new(),
                tensors: vec![
                    raw("model.embed_tokens.weight", &[151_936, 1024], 0, Encoding::Raw(DType::BF16)),
                    raw(&meta_name(crate::encoding::CONFIG_OBJECT), &[64], 64, Encoding::Raw(DType::U8)),
                    raw(&meta_name("tokenizer/vocab_bytes"), &[9], 9, Encoding::Raw(DType::U8)),
                    raw(
                        "model.layers.0.self_attn.q_proj.weight",
                        &[2048, 128],
                        1024 * 2048 / 2,
                        quant(QuantScheme::AwqInt4),
                    ),
                ],
            };
            let observed = Observed::of(&metadata);
            assert!(observed.has("embed_tokens"), "a weight is seen");
            assert!(
                !observed.names().any(|n| n.contains("config") || n.contains("tokenizer")),
                "an artifact object is not; saw {:?}",
                observed.names().collect::<Vec<_>>()
            );
            assert_eq!(
                observed.extents("layer.{}.self_attn.q_proj").map(<[u64]>::to_vec),
                Some(vec![2048, 1024]),
                "and the weight that IS seen arrives unpacked"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seen(pairs: &[(&str, &[u64])]) -> Observed {
        Observed::from_pairs(pairs.iter().map(|(n, s)| (*n, *s)))
    }

    /// The vocabulary rule, which is what lets one spec row read a
    /// checkpoint however its converter nested the text tower.
    #[test]
    fn one_logical_name_serves_every_spelling_of_it() {
        for raw in [
            "model.layers.7.self_attn.q_proj.weight",
            "language_model.model.layers.7.self_attn.q_proj.weight",
            // The order gemma-3's and gemma-4's own converters write,
            // and the one a single stripping pass got wrong: it takes
            // `model.` off and leaves `language_model.` standing.
            "model.language_model.layers.7.self_attn.q_proj.weight",
            "layers.7.self_attn.q_proj",
            // pie's OWN artifacts, which name a tensor the way a spec
            // does — singular, with the index already resolved. An
            // artifact this crate wrote could not be identified by the
            // row it was written from until this spelling collapsed too.
            "layer.7.self_attn.q_proj",
            "model.layer.7.self_attn.q_proj.weight",
        ] {
            assert_eq!(Observed::logical(raw), "layer.{}.self_attn.q_proj", "{raw}");
        }
        // Idempotent, because a `Manifest`'s own rows are looked up
        // through this function and they already carry `{}`.
        assert_eq!(
            Observed::logical("layer.{}.self_attn.q_proj"),
            "layer.{}.self_attn.q_proj",
            "collapsing a collapsed name must not change it"
        );
        // A word that merely CONTAINS `layer` is not an index.
        assert_eq!(
            Observed::logical("model.embed_tokens_per_layer.weight"),
            "embed_tokens_per_layer"
        );
        assert_eq!(
            Observed::logical("model.layers.3.per_layer_input_gate.weight"),
            "layer.{}.per_layer_input_gate"
        );
    }

    /// A per-layer row matches a checkpoint's stack.
    ///
    /// The defect this covers: `rows()` expanded `{}` to `0` while
    /// [`Observed::logical`] collapsed `layers.7.` to `layer.{}.`, so
    /// the two halves of every comparison spelled the same tensor
    /// differently and EVERY per-layer row of every generation reported
    /// `Missing` against a real checkpoint. Nothing caught it because
    /// the round-trip tests built their `Observed` by expanding `{}` the
    /// same wrong way.
    #[test]
    fn a_per_layer_row_matches_a_stack_whatever_index_it_saw() {
        let spec = Manifest::new(8)
            .with(TensorSpec::required("layer.{}.q_proj", [64u64, 32]))
            .with(TensorSpec::absent("layer.{}.q_norm"));
        let stack = seen(&[
            ("model.layers.0.q_proj.weight", &[64, 32]),
            ("model.layers.5.q_proj.weight", &[64, 32]),
            ("model.layers.7.q_proj.weight", &[64, 32]),
        ]);
        assert!(spec.check(&stack).is_ok(), "{:?}", spec.check(&stack));

        let normed = seen(&[
            ("model.layers.0.q_proj.weight", &[64, 32]),
            ("model.layers.0.q_norm.weight", &[32]),
        ]);
        assert!(
            spec.check(&normed).is_err(),
            "a forbidden per-layer tensor must be found wherever in the stack it sits"
        );
    }

    /// A tensor's absence is a fact a spec may state, and it is what
    /// tells a tied variant from an untied one that agrees on every
    /// extent.
    #[test]
    fn absence_discriminates_where_extents_cannot() {
        let tied = Manifest::new(1).with(TensorSpec::absent("lm_head"));
        let untied = Manifest::new(1).with(TensorSpec::required("lm_head", [32u64, 16]));

        let without = seen(&[("model.embed_tokens.weight", &[32, 16])]);
        assert!(tied.check(&without).is_ok());
        assert!(untied.check(&without).is_err());

        let with = seen(&[("model.embed_tokens.weight", &[32, 16]), ("lm_head.weight", &[32, 16])]);
        assert!(tied.check(&with).is_err());
        assert!(untied.check(&with).is_ok());
    }

    /// The q-norm question the old derivation asked by dividing a byte
    /// count: per-head ships `[head_dim]`, global ships
    /// `[heads * head_dim]`. As extents, the two specs simply differ.
    #[test]
    fn qk_norm_width_is_an_extent_rather_than_a_byte_count() {
        let per_head = Manifest::new(1).with(TensorSpec::required("layer.{}.q_norm", [128u64]));
        let global = Manifest::new(1).with(TensorSpec::required("layer.{}.q_norm", [2048u64]));
        let olmo = seen(&[("model.layers.0.q_norm.weight", &[2048])]);
        assert!(per_head.check(&olmo).is_err());
        assert!(global.check(&olmo).is_ok());
    }

    /// A mismatch reports every disagreement, because "which variant is
    /// this" is answered by HOW a checkpoint differs.
    #[test]
    fn a_refusal_carries_the_whole_diff_rather_than_the_first_fault() {
        let spec = Manifest::new(1)
            .with(TensorSpec::required("embed_tokens", [151_936u64, 1024]))
            .with(TensorSpec::required("layer.{}.q_proj", [2048u64, 1024]))
            .with(TensorSpec::absent("lm_head"));
        let wrong = seen(&[
            ("model.layers.0.q_proj.weight", &[4096, 1024]),
            ("lm_head.weight", &[151_936, 1024]),
        ]);
        let err = spec.check(&wrong).expect_err("three faults");
        assert_eq!(err.faults.len(), 3, "{err}");
    }

    /// A `[n]` gamma and an `[n, 1]` one are the same vector; which a
    /// converter wrote is not a fact about the model.
    #[test]
    fn a_degenerate_axis_is_not_a_disagreement() {
        let spec = Manifest::new(1).with(TensorSpec::required("norm", [1024u64]));
        assert!(spec.check(&seen(&[("model.norm.weight", &[1024, 1])])).is_ok());
        assert!(spec.check(&seen(&[("model.norm.weight", &[1024])])).is_ok());
    }
}
