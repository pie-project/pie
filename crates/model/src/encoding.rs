//! What a checkpoint's FILES say about how its numbers are stored.
//!
//! The third decision of the catalog design, and the one that keeps the
//! table small enough to be a table: **quantization is policy, not
//! identity**. Qwen3-8B ships as bf16, as FP8, as AWQ-int4 and as
//! MLX-int4. Four rows for one model would quadruple the catalog and
//! would be four statements of one geometry, which is the exact failure
//! the catalog exists to end.
//!
//! So a row is the LOGICAL model — the shapes — and the manifest
//! matches modulo encoding, undoing the packing on the last axis before
//! it compares. What is observed instead lands here and flows to the
//! authoring pass as a [`Policy`](crate::shared::policy::Policy) input, which is
//! where it always belonged: `push_mlx_affine_declared` needs to know a
//! width, and a width is a property of a file.
//!
//! These three fields are the ones `ModelFacts` carried that were never
//! shape. Everything else that struct held is either the dispatch key
//! (gone — the row IS the dispatch) or is
//! [`LoadShape`](crate::catalog::LoadShape).

/// Where a `.zt` artifact carries the checkpoint's own `config.json`.
///
/// It used to be `model/descriptor` and it used to hold a `pie.model/1`
/// document — ~40 fields resolved out of a 136-field schema by an
/// 845-line normalizer, written so that a driver would not have to know
/// HuggingFace's spelling variations. All but three of those fields
/// were facts about the MODEL, and a model is a catalog row now.
///
/// The name changed with the content deliberately. A driver reading an
/// old artifact finds no `model/config` and refuses at the door, rather
/// than parsing a descriptor as a config and finding no
/// `quantization_config` in it — which would look exactly like an
/// unquantized checkpoint and would author an AWQ model as bf16.
/// Re-import the artifact.
pub const CONFIG_OBJECT: &str = "model/config";

/// The quantization a checkpoint DECLARES.
///
/// Declared rather than measured: this is what the `config.json`'s
/// `quantization_config` (or `mlx_lm`'s bare `quantization` block) says
/// about itself. The tensors are the authority on geometry — that is
/// what the manifest is for — but only the config states a group size,
/// because a group size is not an extent of anything.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Encoding {
    /// `quantization_config.quant_method`: `"fp8"`, `"awq"`, `"mxfp4"`,
    /// `"compressed-tensors"`, empty for an unquantized checkpoint.
    ///
    /// A `String` and not an enum on purpose. The set is open — it grows
    /// whenever a quantizer ships — and unlike `model_type` nothing
    /// DISPATCHES on it: the authors ask "is this empty" and "is this
    /// mxfp4", and an unknown method is refused by the scheme check
    /// rather than silently routed. A string is the honest shape for a
    /// value whose only closed question is whether it is one particular
    /// word.
    pub method: String,
    /// The declared width in bits, 0 when the config declares none.
    ///
    /// Zero is EXPENSIVE rather than merely absent, which is why it is
    /// stated: `push_mlx_affine_declared` answers an undeclared width
    /// with 4 bits, so an 8-bit checkpoint arriving here as 0 is
    /// authored with twice the logical columns and no error anywhere.
    pub bits: u32,
    /// The declared group size beside it, 0 when undeclared.
    pub group_size: u32,
}

impl Encoding {
    /// An unquantized checkpoint.
    #[must_use]
    pub fn dense() -> Self {
        Self::default()
    }

    /// The checkpoint declares no quantization.
    #[must_use]
    pub fn is_none(&self) -> bool {
        self.method.is_empty()
    }

    /// MXFP4, the one method the authors branch on by name.
    ///
    /// A method and a predicate rather than a comparison at each site:
    /// four places asked `quant_method == "mxfp4"` and one asked
    /// `.contains("mxfp4")`, which are not the same question.
    #[must_use]
    pub fn is_mxfp4(&self) -> bool {
        self.method.eq_ignore_ascii_case("mxfp4")
    }

    /// Read the declared encoding out of a checkpoint's `config.json`.
    ///
    /// # What this replaced
    ///
    /// An 845-line normalizer, a 136-field schema, and a `pie.model/1`
    /// descriptor written on one side of a process boundary and parsed
    /// on the other. Those existed to carry ~40 numbers across; 34 of
    /// them are the row's now, and these three are what is left —
    /// because they are the only ones that are NOT a property of the
    /// model.
    ///
    /// # Why a config and not the tensors
    ///
    /// The tensors are the authority on geometry and the manifest asks
    /// them. They cannot answer this: a group size is not an extent of
    /// anything, and a checkpoint quantized at 128 and one quantized at
    /// 64 have identically-shaped scale tensors when the row count
    /// happens to divide both ways. Declared is the only available
    /// truth, so declared is what this reads.
    ///
    /// # The four spellings
    ///
    /// `quantization_config` under the text view, `quantization_config`
    /// at the root (where multimodal configs put it), and `quantization`
    /// in either place — `mlx_lm`'s spelling of the same block. All four
    /// are read because the last one used to be unreachable from here:
    /// the C++ normalizer served CUDA and MLX checkpoints reach Metal,
    /// so an MLX 8-bit checkpoint arrived with an undeclared width, and
    /// `push_mlx_affine_declared` answers an undeclared width with 4
    /// bits. Twice the logical columns, authored silently.
    ///
    /// # Errors
    ///
    /// The document is not JSON. A document with no quantization block
    /// is not an error — it is an unquantized checkpoint, which is most
    /// of them.
    #[cfg(feature = "contract")]
    pub fn from_config_json(text: &str) -> Result<Self, serde_json::Error> {
        let root: serde_json::Value = serde_json::from_str(text)?;
        Ok(Self::from_config_value(&root))
    }

    /// As [`Self::from_config_json`], for a caller that already parsed.
    ///
    /// Gated with it on `contract` — reading an encoding is something
    /// only the LOAD path does, and a build that links this crate for
    /// its chat templates alone should not carry a JSON parser to do it.
    #[cfg(feature = "contract")]
    #[must_use]
    pub fn from_config_value(root: &serde_json::Value) -> Self {
        let text = root.get("text_config");
        let block = [
            text.and_then(|t| t.get("quantization_config")),
            root.get("quantization_config"),
            text.and_then(|t| t.get("quantization")),
            root.get("quantization"),
        ]
        .into_iter()
        .flatten()
        .find(|v| v.is_object());
        let Some(q) = block else {
            return Self::dense();
        };
        let u32_of = |key: &str| {
            q.get(key)
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())
                .unwrap_or(0)
        };
        Self {
            method: q
                .get("quant_method")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            bits: u32_of("bits"),
            group_size: u32_of("group_size"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unquantized_checkpoint_declares_nothing() {
        let e = Encoding::dense();
        assert!(e.is_none());
        assert!(!e.is_mxfp4());
        assert_eq!(e.bits, 0);
        assert_eq!(e.group_size, 0);
    }

    /// The five call sites asked this two different ways — four by
    /// equality and one by substring — so it is asked once here.
    #[test]
    fn mxfp4_is_one_question_with_one_answer() {
        let e = Encoding { method: "mxfp4".into(), bits: 4, group_size: 32 };
        assert!(e.is_mxfp4());
        assert!(!e.is_none());
        let upper = Encoding { method: "MXFP4".into(), bits: 4, group_size: 32 };
        assert!(upper.is_mxfp4(), "a method name is not case-significant");
        let awq = Encoding { method: "awq".into(), bits: 4, group_size: 128 };
        assert!(!awq.is_mxfp4());
        assert!(!awq.is_none());
    }

    /// Zero bits is a value a reader must not treat as "4".
    #[test]
    fn an_undeclared_width_is_zero_and_says_so() {
        let e = Encoding { method: "awq".into(), bits: 0, group_size: 0 };
        assert!(!e.is_none(), "a method with no width is still quantized");
        assert_eq!(e.bits, 0);
    }

    /// Most checkpoints declare nothing, and that is not a failure.
    #[cfg(feature = "contract")]
    #[test]
    fn a_config_with_no_quantization_block_is_an_unquantized_checkpoint() {
        let e = Encoding::from_config_json(r#"{"model_type":"qwen3"}"#).expect("valid json");
        assert!(e.is_none());
        assert_eq!(e.bits, 0);
        assert_eq!(e.group_size, 0);
    }

    /// The plain spelling, under the text view.
    #[cfg(feature = "contract")]
    #[test]
    fn the_text_views_block_is_read() {
        let e = Encoding::from_config_json(
            r#"{"text_config":{"quantization_config":
                {"quant_method":"awq","bits":4,"group_size":128}}}"#,
        )
        .expect("valid json");
        assert_eq!(e.method, "awq");
        assert_eq!(e.bits, 4);
        assert_eq!(e.group_size, 128);
    }

    /// Multimodal configs put it at the root instead.
    #[cfg(feature = "contract")]
    #[test]
    fn the_root_block_is_read_for_a_multimodal_config() {
        let e = Encoding::from_config_json(
            r#"{"text_config":{"hidden_size":2048},
                "quantization_config":{"quant_method":"fp8","bits":8}}"#,
        )
        .expect("valid json");
        assert_eq!(e.method, "fp8");
        assert_eq!(e.bits, 8);
        assert_eq!(e.group_size, 0, "fp8 declares no group size and says so");
    }

    /// `mlx_lm`'s bare `quantization` key, which the C++ normalizer
    /// never read.
    ///
    /// The defect that made this worth a branch: MLX checkpoints reach
    /// Metal, the normalizer served CUDA, so an MLX 8-bit checkpoint
    /// arrived here with `bits: 0` — and `push_mlx_affine_declared`
    /// answers an undeclared width with 4. Twice the logical columns,
    /// no error, a model that loads and produces noise.
    #[cfg(feature = "contract")]
    #[test]
    fn the_mlx_spelling_is_read_so_an_eight_bit_checkpoint_is_not_authored_as_four() {
        let e = Encoding::from_config_json(
            r#"{"quantization":{"group_size":64,"bits":8}}"#,
        )
        .expect("valid json");
        assert_eq!(e.bits, 8, "not 4, which is what an undeclared width becomes");
        assert_eq!(e.group_size, 64);
        assert!(
            e.method.is_empty(),
            "mlx declares a width without naming a method, and inventing one              would route the authoring pass somewhere it was not asked to go"
        );
    }

    /// The text view wins over the root when both carry a block.
    ///
    /// Stated as a test rather than left to argument order: a
    /// multimodal checkpoint whose towers are unquantized and whose text
    /// stack is not would otherwise author the towers' encoding onto the
    /// stack.
    #[cfg(feature = "contract")]
    #[test]
    fn the_text_view_is_preferred_when_both_declare() {
        let e = Encoding::from_config_json(
            r#"{"text_config":{"quantization_config":{"quant_method":"awq","bits":4}},
                "quantization_config":{"quant_method":"fp8","bits":8}}"#,
        )
        .expect("valid json");
        assert_eq!(e.method, "awq");
        assert_eq!(e.bits, 4);
    }

    /// A block that is not an object is not a block.
    ///
    /// `"quantization_config": null` appears in real configs as a way of
    /// saying "none", and reading it as a present-but-empty block would
    /// have made the next lookup fall through to the root's.
    #[cfg(feature = "contract")]
    #[test]
    fn a_null_block_falls_through_rather_than_matching() {
        let e = Encoding::from_config_json(
            r#"{"text_config":{"quantization_config":null},
                "quantization_config":{"quant_method":"mxfp4","bits":4,"group_size":32}}"#,
        )
        .expect("valid json");
        assert!(e.is_mxfp4(), "the root's block answered");
        assert_eq!(e.group_size, 32);
    }

    /// A width that is not a number, or is negative, reads as
    /// undeclared rather than as a panic or a wrapped value.
    #[cfg(feature = "contract")]
    #[test]
    fn a_nonsense_width_is_undeclared_rather_than_wrapped() {
        let e = Encoding::from_config_json(
            r#"{"quantization_config":{"quant_method":"awq","bits":-4,"group_size":"128"}}"#,
        )
        .expect("valid json");
        assert_eq!(e.method, "awq");
        assert_eq!(e.bits, 0);
        assert_eq!(e.group_size, 0);
    }

    /// A document that is not JSON is an error, not a default.
    ///
    /// The distinction the old path lost: the normalizer swallowed a
    /// missing field with a default and the descriptor refused, so one
    /// document had two failure policies depending on which reader got
    /// there first.
    #[cfg(feature = "contract")]
    #[test]
    fn an_unparseable_config_refuses_rather_than_defaulting() {
        assert!(Encoding::from_config_json("not json at all").is_err());
    }

    /// Both entry points answer the same.
    #[cfg(feature = "contract")]
    #[test]
    fn parsing_once_and_parsing_twice_agree() {
        let text = r#"{"quantization_config":{"quant_method":"fp8","bits":8,"group_size":0}}"#;
        let value: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(
            Encoding::from_config_json(text).unwrap(),
            Encoding::from_config_value(&value)
        );
    }
}
