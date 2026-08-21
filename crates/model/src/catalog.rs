use crate::manifest::Manifest;

#[cfg(feature = "chat")]
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadShape {

    pub layers: u32,

    pub head_dim: u32,

    pub n_experts: u32,

    pub mamba_groups: u32,

    pub kv_shared_layers: u32,

    pub tied_embeddings: bool,
}

impl LoadShape {

    #[must_use]
    pub const fn dense(layers: u32, head_dim: u32, tied_embeddings: bool) -> Self {
        Self {
            layers,
            head_dim,
            n_experts: 0,
            mamba_groups: 0,
            kv_shared_layers: 0,
            tied_embeddings,
        }
    }

    #[must_use]
    pub const fn mixture(
        layers: u32,
        head_dim: u32,
        n_experts: u32,
        tied_embeddings: bool,
    ) -> Self {
        Self {
            layers,
            head_dim,
            n_experts,
            mamba_groups: 0,
            kv_shared_layers: 0,
            tied_embeddings,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalBinding {

    pub qmm_tile: Option<(u32, u32)>,

    pub qmm_partial_rows: bool,

    pub qmm_fp16_precast: bool,

    pub quant_group: u32,

    pub quant_bits: u32,

    pub router_quant_group: u32,

    pub router_quant_bits: u32,

    pub moe_mxfp4: bool,

    pub fuse_residual_gemv: bool,

    pub paged_multi_batch: bool,

    pub qmm_multi_batch: bool,

    pub add_bias: bool,

    pub fused_qk_rope: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Backend<'a> {

    #[default]
    Cuda,

    Metal(&'a MetalBinding),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Deployed<'a> {

    pub backend: Backend<'a>,

    pub tp_size: u32,

    pub layer_scalars: &'a [f32],
}

impl<'a> Deployed<'a> {

    #[must_use]
    pub fn single() -> Self {
        Self {
            backend: Backend::Cuda,
            tp_size: 1,
            layer_scalars: &[],
        }
    }

    #[must_use]
    pub fn metal(binding: &'a MetalBinding) -> Self {
        Self {
            backend: Backend::Metal(binding),
            tp_size: 1,
            layer_scalars: &[],
        }
    }
}

pub trait Variant: Sync + Send + 'static {

    fn id(&self) -> &'static str;

    fn manifest(&self) -> Manifest;

    fn load_shape(&self) -> LoadShape;

    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal>;

    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error>;

    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, crate::deployment::Refusal>;

    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct>;
}

#[must_use]
pub fn catalog() -> &'static [&'static dyn Variant] {
    static CATALOG: std::sync::OnceLock<Vec<&'static dyn Variant>> = std::sync::OnceLock::new();
    CATALOG.get_or_init(|| {
        let mut rows: Vec<&'static dyn Variant> = Vec::new();
        for generation in GENERATIONS {
            rows.extend_from_slice(generation());
        }
        rows
    })
}

type Rows = fn() -> &'static [&'static dyn Variant];

#[macro_export]
macro_rules! rows_of {
    ($row:ty) => {

        #[must_use]
        pub fn rows() -> &'static [&'static dyn $crate::catalog::Variant] {
            static ROWS: std::sync::OnceLock<Vec<&'static dyn $crate::catalog::Variant>> =
                std::sync::OnceLock::new();
            ROWS.get_or_init(|| {
                VARIANTS
                    .iter()
                    .map(|v| v as &'static dyn $crate::catalog::Variant)
                    .collect()
            })
        }
    };
}

const GENERATIONS: &[Rows] = &[
    crate::llama_3::rows,
    crate::qwen_2::rows,
    crate::qwen_3::rows,
    crate::qwen_3_5::rows,
    crate::gemma_2::rows,
    crate::gemma_3::rows,
    crate::gemma_3n::rows,
    crate::gemma_4::rows,
    crate::glm_5::rows,
    crate::gpt_oss::rows,
    crate::kimi_k2::rows,
    crate::kimi_k3::rows,
    crate::deepseek_v4::rows,
    crate::nemotron_h::rows,
    crate::olmo_2::rows,
    crate::olmo_3::rows,
    crate::phi_3::rows,
    crate::mistral_3::rows,
    crate::csm::rows,

    #[cfg(feature = "test-rows")]
    crate::test_rows::rows,
];

#[must_use]
pub fn find(id: &str) -> Option<&'static dyn Variant> {
    catalog().iter().copied().find(|row| row.id() == id)
}

#[must_use]
pub fn ids() -> Vec<&'static str> {
    catalog().iter().map(|row| row.id()).collect()
}

#[must_use]
pub fn arches() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = catalog()
        .iter()
        .filter_map(|row| row.deployment(Deployed::single()).ok())
        .map(|d| d.advertised.arch)
        .filter(|a| !a.is_empty())
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

#[must_use]
pub fn nearest_ids(id: &str, take: usize) -> Vec<&'static str> {
    let mut scored: Vec<(usize, &'static str)> = ids()
        .into_iter()
        .map(|k| (edit_distance(id, k), k))
        .collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored.into_iter().take(take).map(|(_, k)| k).collect()
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unmatched {

    NoRow {
        nearest: Vec<(&'static str, String)>,
    },

    Ambiguous { ids: Vec<&'static str> },

    NoSuchId {
        id: String,
        nearest: Vec<&'static str>,
    },
}

impl std::fmt::Display for Unmatched {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoRow { nearest } => {
                write!(f, "this checkpoint matches no model this build serves")?;
                for (id, why) in nearest {
                    write!(f, "\n  {id}: {why}")?;
                }
                Ok(())
            }
            Self::Ambiguous { ids } => write!(
                f,
                "this checkpoint matches {} models equally well ({}); two rows \
                 no checkpoint can tell apart are one row",
                ids.len(),
                ids.join(", "),
            ),
            Self::NoSuchId { id, nearest } if nearest.is_empty() => write!(
                f,
                "no model named '{id}' in this build; `pie model list` prints the \
                 ids this binary serves",
            ),
            Self::NoSuchId { id, nearest } => write!(
                f,
                "no model named '{id}' in this build; did you mean {}?",
                nearest.join(", "),
            ),
        }
    }
}

impl std::error::Error for Unmatched {}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Override {

    #[default]
    None,

    Id(String),
}

pub const GEOMETRIC_TWINS: &[&[&str]] = &[&["llama-3.1-70b", "llama-3.3-70b"]];

#[must_use]
pub fn are_declared_twins(ids: &[&str]) -> bool {
    GEOMETRIC_TWINS
        .iter()
        .any(|set| set.len() == ids.len() && set.iter().all(|id| ids.contains(id)))
}

#[cfg(feature = "contract")]
mod identify {
    use super::{Override, Unmatched, Variant, catalog, find, nearest_ids};
    use crate::manifest::Observed;
    use model_loader::checkpoint::Attributes;
    use model_loader::checkpoint::CheckpointMetadata;
    use model_loader::checkpoint::meta;

    pub fn identify_artifact(
        attributes: &Attributes,
        metadata: &CheckpointMetadata,
        chosen: &Override,
    ) -> Result<&'static dyn Variant, Unmatched> {
        let Some(stated) = stated_row(attributes) else {
            return identify(metadata, chosen);
        };
        let row = find(stated).ok_or_else(|| Unmatched::NoSuchId {
            id: stated.to_string(),
            nearest: nearest_ids(stated, 3),
        })?;
        if let Override::Id(id) = chosen
            && id != row.id()
        {
            return Err(Unmatched::NoRow {
                nearest: vec![(
                    row.id(),
                    format!(
                        "`pie model build` wrote this artifact as this row; `{id}` asks for another, and a built artifact is not evidence either way"
                    ),
                )],
            });
        }
        Ok(row)
    }

    fn stated_row(attributes: &Attributes) -> Option<&str> {
        let contract = attributes.text(meta::CONTRACT_KEY)?;
        if contract != meta::CONTRACT_REVISION.to_string() {
            return None;
        }
        attributes.text(meta::MODEL_ID_KEY)
    }

    pub fn identify(
        metadata: &CheckpointMetadata,
        chosen: &Override,
    ) -> Result<&'static dyn Variant, Unmatched> {
        identify_observed(&Observed::of(metadata), chosen)
    }

    pub fn identify_observed(
        observed: &Observed,
        chosen: &Override,
    ) -> Result<&'static dyn Variant, Unmatched> {
        if let Override::Id(id) = chosen {
            let row = find(id).ok_or_else(|| Unmatched::NoSuchId {
                id: id.clone(),
                nearest: nearest_ids(id, 3),
            })?;
            return row
                .manifest()
                .check(observed)
                .map(|()| row)
                .map_err(|why| Unmatched::NoRow {
                    nearest: vec![(row.id(), why.to_string())],
                });
        }

        let mut matched: Vec<&'static dyn Variant> = Vec::new();
        let mut misses: Vec<(&'static str, usize, String)> = Vec::new();
        for row in catalog() {
            match row.manifest().check(observed) {
                Ok(()) => matched.push(*row),
                Err(why) => misses.push((row.id(), why.faults.len(), why.to_string())),
            }
        }
        match matched.len() {
            1 => Ok(matched[0]),
            0 => {
                misses.sort_by_key(|(_, faults, _)| *faults);
                misses.truncate(3);
                Err(Unmatched::NoRow {
                    nearest: misses.into_iter().map(|(id, _, why)| (id, why)).collect(),
                })
            }
            _ => Err(Unmatched::Ambiguous {
                ids: matched.iter().map(|row| row.id()).collect(),
            }),
        }
    }
}

#[cfg(feature = "contract")]
pub use identify::{identify, identify_artifact, identify_observed};

#[cfg(all(test, feature = "contract"))]
pub(crate) mod identify_tests {
    use super::*;
    use crate::manifest::Presence;
    use model_loader::checkpoint::meta;
    use model_loader::checkpoint::{Attribute, Attributes, CheckpointMetadata, RawTensor};
    use model_loader::types::{DType, Encoding, FileId, TensorId};

    pub(crate) fn checkpoint_of(row: &dyn Variant) -> CheckpointMetadata {
        let manifest = row.manifest();
        let tensors = manifest
            .tensors
            .iter()
            .filter(|t| t.presence != Presence::Absent)
            .enumerate()
            .map(|(i, t)| {
                let extents = if t.extents.is_empty() {
                    vec![1]
                } else {
                    t.extents.clone()
                };
                let elems: u64 = extents.iter().product();
                RawTensor {
                    id: TensorId(u32::try_from(i).unwrap_or(0)),

                    name: t.name.clone(),
                    file_id: FileId(0),
                    file_offset: 0,
                    span_bytes: elems * 2,
                    shape: extents
                        .iter()
                        .map(|&e| i64::try_from(e).unwrap_or(0))
                        .collect(),
                    encoding: Encoding::Raw(DType::BF16),
                }
            })
            .collect();
        CheckpointMetadata {
            files: Vec::new(),
            tensors,
        }
    }

    #[test]
    fn every_row_is_identified_as_itself_and_not_as_a_sibling() {
        let mut collisions: Vec<String> = Vec::new();
        let mut twinned: Vec<&str> = Vec::new();
        for row in catalog() {
            let metadata = checkpoint_of(*row);
            match identify(&metadata, &Override::None) {
                Ok(found) if found.id() == row.id() => {}
                Ok(found) => collisions.push(format!("{} identified as {}", row.id(), found.id())),
                Err(Unmatched::Ambiguous { ids }) if are_declared_twins(&ids) => {
                    twinned.push(row.id());
                }
                Err(Unmatched::Ambiguous { ids }) => {
                    collisions.push(format!("{} is ambiguous with {ids:?}", row.id()));
                }
                Err(e) => collisions.push(format!("{} did not identify: {e}", row.id())),
            }
        }
        assert!(
            collisions.is_empty(),
            "identification is not one-to-one, so a checkpoint can load as a \
             model it is not — which is the one thing the manifest exists to \
             make impossible. If a pair here is genuinely one geometry under \
             two release names, declare it in `GEOMETRIC_TWINS`; otherwise a \
             manifest is wrong:\n  {}",
            collisions.join("\n  ")
        );
        for set in GEOMETRIC_TWINS {
            for id in *set {
                assert!(
                    twinned.contains(id),
                    "{id} is declared a geometric twin and identifies cleanly \
                     anyway, so the declaration is stale — drop it, or the \
                     next real collision hides behind it",
                );
            }
        }
    }

    #[test]
    fn a_tied_row_still_identifies_itself_when_the_export_publishes_the_head() {
        let mut checked = 0usize;
        let mut wrong: Vec<String> = Vec::new();
        for row in catalog() {
            let copies: Vec<(String, Vec<u64>)> = row
                .manifest()
                .tensors
                .iter()
                .filter(|t| !t.tied_copy.is_empty())
                .map(|t| (t.name.clone(), t.tied_copy.clone()))
                .collect();
            if copies.is_empty() {
                continue;
            }
            checked += 1;
            let mut metadata = checkpoint_of(*row);
            for (name, extents) in copies {
                let elems: u64 = extents.iter().product();
                metadata.tensors.push(RawTensor {
                    id: TensorId(u32::try_from(metadata.tensors.len()).unwrap_or(0)),
                    name,
                    file_id: FileId(0),
                    file_offset: 0,
                    span_bytes: elems * 2,
                    shape: extents
                        .iter()
                        .map(|&e| i64::try_from(e).unwrap_or(0))
                        .collect(),
                    encoding: Encoding::Raw(DType::BF16),
                });
            }
            match identify(&metadata, &Override::None) {
                Ok(found) if found.id() == row.id() => {}
                Err(Unmatched::Ambiguous { ids }) if are_declared_twins(&ids) => {}
                Ok(found) => {
                    wrong.push(format!("{} identified as {}", row.id(), found.id()));
                }
                Err(e) => wrong.push(format!("{} no longer identifies: {e}", row.id())),
            }
        }
        assert!(
            checked > 0,
            "no row in the catalog states a tie, so this test measured \
             nothing — `TensorSpec::tied` is unreached and the refusal it \
             was written for is back",
        );
        assert!(
            wrong.is_empty(),
            "tolerating the redundant copy an HF export writes broke \
             one-to-one identification, which is worse than the refusal it \
             replaced:\n  {}",
            wrong.join("\n  ")
        );
    }

    #[test]
    fn a_declared_twin_still_loads_when_the_caller_names_it() {
        for set in GEOMETRIC_TWINS {
            for id in *set {
                let row =
                    find(id).unwrap_or_else(|| panic!("{id} is declared but not in the catalog"));
                let metadata = checkpoint_of(row);
                let found = identify(&metadata, &Override::Id((*id).to_string()))
                    .unwrap_or_else(|e| panic!("{id} named explicitly and still refused: {e}"));
                assert_eq!(found.id(), *id);
            }
        }
    }

    #[test]
    fn a_built_artifact_is_taken_at_its_word() {
        let row = catalog()[0];
        let metadata = post_transform_metadata();
        assert!(
            identify(&metadata, &Override::None).is_err(),
            "the control is broken: these tensors were supposed to match no row, \
             so accepting them proves nothing",
        );
        let found = identify_artifact(&built_as(row.id()), &metadata, &Override::None)
            .expect("a build states its row and the row is in this catalog");
        assert_eq!(found.id(), row.id());
    }

    #[test]
    fn a_build_from_another_contract_revision_is_not_believed() {
        let row = catalog()[0];
        let stale = Attributes::from_pairs(vec![
            (
                meta::CONTRACT_KEY.to_string(),
                Attribute::Text((meta::CONTRACT_REVISION + 1).to_string()),
            ),
            (
                meta::MODEL_ID_KEY.to_string(),
                Attribute::Text(row.id().to_string()),
            ),
        ]);
        assert!(
            identify_artifact(&stale, &post_transform_metadata(), &Override::None).is_err(),
            "a statement about a layout this build does not read was believed",
        );
    }

    #[test]
    fn a_stated_row_this_build_does_not_have_is_refused_rather_than_guessed() {
        let Err(err) = identify_artifact(
            &built_as("qwen3-0.6b-but-not-really"),
            &post_transform_metadata(),
            &Override::None,
        ) else {
            panic!("an unknown row was resolved to something");
        };
        assert!(
            matches!(err, Unmatched::NoSuchId { .. }),
            "the refusal should name the id that is missing, not diff the \
             tensors against rows that were never in question: {err}",
        );
    }

    #[test]
    fn an_override_may_confirm_the_record_but_not_contradict_it() {
        let row = catalog()[0];
        let other = catalog()
            .iter()
            .find(|candidate| candidate.id() != row.id())
            .expect("the catalog has more than one row");
        let stated = built_as(row.id());
        let confirmed = identify_artifact(
            &stated,
            &post_transform_metadata(),
            &Override::Id(row.id().to_string()),
        )
        .expect("an override naming the row the artifact states is not a disagreement");
        assert_eq!(confirmed.id(), row.id());
        assert!(
            identify_artifact(
                &stated,
                &post_transform_metadata(),
                &Override::Id(other.id().to_string()),
            )
            .is_err(),
            "an override contradicting the artifact's own record was allowed",
        );
    }

    #[test]
    fn an_artifact_that_states_nothing_is_identified_from_its_tensors() {
        for row in catalog() {
            let metadata = checkpoint_of(*row);
            let bare = identify(&metadata, &Override::None);
            let through = identify_artifact(&Attributes::default(), &metadata, &Override::None);
            assert_eq!(
                bare.map(Variant::id).map_err(|e| e.to_string()),
                through.map(Variant::id).map_err(|e| e.to_string()),
                "{} identifies differently through the artifact path",
                row.id(),
            );
        }
    }

    fn built_as(id: &str) -> Attributes {
        Attributes::from_pairs(vec![
            (
                meta::CONTRACT_KEY.to_string(),
                Attribute::Text(meta::CONTRACT_REVISION.to_string()),
            ),
            (
                meta::MODEL_ID_KEY.to_string(),
                Attribute::Text(id.to_string()),
            ),
        ])
    }

    fn post_transform_metadata() -> CheckpointMetadata {
        CheckpointMetadata {
            files: Vec::new(),
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "model.layers.0.self_attn.qkv_proj.fused.weight".to_string(),
                file_id: FileId(0),
                file_offset: 0,
                span_bytes: 8,
                shape: vec![2, 2],
                encoding: Encoding::Raw(DType::BF16),
            }],
        }
    }

    #[test]
    fn a_checkpoint_no_row_describes_is_refused_with_its_near_misses() {
        let metadata = CheckpointMetadata {
            files: Vec::new(),
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "embed_tokens".to_string(),
                file_id: FileId(0),
                file_offset: 0,
                span_bytes: 8,
                shape: vec![2, 2],
                encoding: Encoding::Raw(DType::BF16),
            }],
        };
        let Err(Unmatched::NoRow { nearest }) = identify(&metadata, &Override::None) else {
            panic!("a two-by-two embedding is no model in this catalog");
        };
        assert!(
            !nearest.is_empty(),
            "a refusal with no near miss is not a diagnosis"
        );
        assert!(nearest.len() <= 3, "the three closest, not the whole table");
        for (id, why) in &nearest {
            assert!(!id.is_empty() && !why.is_empty(), "{id}: {why}");
        }
    }

    #[test]
    fn an_override_names_a_row_and_still_holds_it_to_the_manifest() {
        let row = *catalog().first().expect("the catalog is not empty");
        let chosen = Override::Id(row.id().to_string());

        let matching = checkpoint_of(row);
        assert_eq!(
            identify(&matching, &chosen)
                .map(|r| r.id())
                .unwrap_or("<refused>"),
            row.id(),
            "the named row accepts the checkpoint it describes",
        );

        let empty = CheckpointMetadata {
            files: Vec::new(),
            tensors: Vec::new(),
        };
        let refused = identify(&empty, &chosen);
        assert!(
            matches!(refused, Err(Unmatched::NoRow { .. })),
            "an override must not turn a mismatch into a load; got {:?}",
            refused.map(|r| r.id()),
        );
    }

    #[test]
    fn an_override_with_an_unknown_id_suggests_rather_than_guesses() {
        let real = catalog().first().expect("the catalog is not empty").id();
        let typo = format!("{real}x");
        let metadata = CheckpointMetadata {
            files: Vec::new(),
            tensors: Vec::new(),
        };
        let Err(Unmatched::NoSuchId { id, nearest }) =
            identify(&metadata, &Override::Id(typo.clone()))
        else {
            panic!("'{typo}' names no row");
        };
        assert_eq!(id, typo);
        assert!(
            nearest.contains(&real),
            "the nearest ids must include the one a single character away: {nearest:?}",
        );
    }

    #[test]
    fn no_override_is_the_default() {
        assert_eq!(Override::default(), Override::None);
    }
}
