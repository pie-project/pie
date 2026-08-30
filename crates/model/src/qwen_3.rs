pub mod forward;
pub mod import;
pub mod model;
pub mod template;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[crate::Row] = model_dsl::catalog![
    (
        "qwen36-27b-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d27b(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen36-27b-mlxu4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d27b_undrafted(Dtype::MlxU4, Dtype::Bf16, 1)
    ),
    (
        "qwen35-a3b-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::a3b(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen35-d3b-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d3b(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen35-d0.8b-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d0_8b(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen35-d0.8b-mlxu4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d0_8b(Dtype::MlxU4, Dtype::Bf16, 1)
    ),
    (
        "qwen35-d0.8b-vision-eagle-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d0_8b_vision_eagle(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen36-27b-vision-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d27b_vision(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen35-d0.8b-vision-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d0_8b_vision(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen35-d0.8b-eagle-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d0_8b_eagle(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen35-a3b-bf16-kv-bf16-tp2",
        2,
        model_dsl::trace_hybrid,
        Model::a3b(Dtype::Bf16, Dtype::Bf16, 2)
    ),
];

/// **`qwen36-27b` IS FIRST, AND THE ORDER IS LOAD-BEARING.** `model::identify`
/// walks these rows and returns the first whose contract the checkpoint can
/// satisfy, and a contract is satisfied by NAMES: `qwen35-d3b` asks for
/// twenty-four `model.language_model.layers.*` and a dense mlp, all of which a
/// Qwen3.6-27B file also holds, so a d3b row reached first would claim a 27B
/// artifact and land the first twenty-four of its sixty-four layers. The
/// reverse cannot happen — `qwen36-27b` asks for layers up to sixty-three and
/// for fifteen `mtp.*` planes, and a 3B file holds neither — so the strictly
/// more demanding row goes first and the ambiguity is closed by construction.
///
/// **AND `-eagle` COMES BEFORE THE ROW IT DRAFTS FOR, FOR THE 27B's REASON
/// EXACTLY** (campaign M-4). An overlay artifact holds everything a plain
/// `qwen35-d0.8b` artifact holds plus eleven `aux.*` planes, so the plain row
/// reached first would claim it and serve it as a model with no draft head —
/// no refusal, no missing plane, just a `mtp_logits` capability that silently
/// went away. The reverse cannot happen: the eagle row asks for the eleven and
/// a base artifact holds none.
///
/// **AND EVERY `mlxu4` ROW COMES BEFORE EVERY `bf16` ONE, FOR THE SAME
/// REASON.** The 4-bit rows are the strictly more demanding ones: each weight
/// they read is a triplet, so they miss unless the file holds a `.scales` and
/// a `.biases` beside every projection. A bf16 row asked about a 4-bit file
/// does NOT miss — it finds every `.weight` it names, at a rank and a width
/// it never checks here — so a bf16 row reached first would claim an MLX
/// artifact and then fail four stages later, in the storage compiler, against
/// a shape nobody wrote. Ordering closes that the same way the 27B does: the
/// row that asks for more goes first.
/// **AND EVERY `-vision` ROW COMES LAST, WHICH INVERTS THE RULE ABOVE — AND
/// THE INVERSION IS A PRICE, MEASURED WITH A CONTROL** (campaign M-1/M-2).
///
/// A tower row asks for a hundred and fifty planes a text-only row does not,
/// and the real Qwen3.5-0.8B and Qwen3.6-27B checkpoints HAVE them, so
/// strictness says the tower row goes first and a checkpoint that ships a
/// tower is read as the model it is. It was tried twice.
///
/// **THE FIRST FLIP BROKE SERVING AND THE SECOND ONE COST 15%.** The first
/// panicked every text-only fire on an unbound `PatchRoutes` — the embed merge
/// is a TRUNK-unit node whose token window is full — and that is closed now
/// ([`forward::Facts::media`] guards the merge alone) along with the two fire
/// stops behind it. The second served correctly and paid for it:
///
/// ```text
///                       census      c256 tput, 1024 req, 128 tok
/// text-first (this)     40/41       11,973 tok/s   1024/1024
/// vision-first          40/41       10,185 tok/s   1024/1024
/// ```
///
/// Back to back on one tree, one rebake apart; α's own number was 12,276, so
/// the tree is healthy and the 14.9% is the ordering and not drift.
///
/// **AND THE MECHANISM WAS PREDICTED.** A two-unit load stands its fold down
/// — `Armed` is structurally one-graph-per-bucket, so a multi-unit bucket
/// serves the keyed path (multimodal §5.3's `fold_refused`) — and §5.3 says
/// the consequence out loud: *"6 + 6, not 6 × 6 is a property OF per-unit
/// keys, and deferring the fold defers the property."* A single-fire census
/// cannot see it; c256 can. So reading a tower-shipping checkpoint as
/// vision-capable by default costs every text-only deployment 15% until the
/// fold goes per-unit, which is M-3's own deferred item.
///
/// The rows go last until it does. The order and this comment move together,
/// and `the_vision_imports_cover_their_plans_over_the_real_census` fails the
/// day one moves without the other.
///
/// [`forward::Facts::media`]: super::forward::Facts::media
///
pub const IMPORTS: &[crate::ImportRow] = &[
    ("qwen36-27b-mlxu4-kv-bf16", |src| {
        Model::d27b_undrafted(Dtype::MlxU4, Dtype::Bf16, 1).import(src)
    }),
    ("qwen35-d0.8b-mlxu4-kv-bf16", |src| {
        Model::d0_8b(Dtype::MlxU4, Dtype::Bf16, 1).import(src)
    }),
    ("qwen36-27b-bf16-kv-bf16", |src| {
        Model::d27b(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("qwen35-a3b-bf16-kv-bf16", |src| {
        Model::a3b(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("qwen35-d3b-bf16-kv-bf16", |src| {
        Model::d3b(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("qwen35-d0.8b-eagle-bf16-kv-bf16", |src| {
        Model::d0_8b_eagle(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("qwen35-d0.8b-bf16-kv-bf16", |src| {
        Model::d0_8b(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("qwen35-a3b-bf16-kv-bf16-tp2", |src| {
        Model::a3b(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("qwen35-d0.8b-vision-eagle-bf16-kv-bf16", |src| {
        Model::d0_8b_vision_eagle(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("qwen36-27b-vision-bf16-kv-bf16", |src| {
        Model::d27b_vision(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("qwen35-d0.8b-vision-bf16-kv-bf16", |src| {
        Model::d0_8b_vision(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
];

pub const TEMPLATES: &[crate::template::TemplateRow] = &[
    ("qwen36-27b-bf16-kv-bf16", template::chatml),
    ("qwen36-27b-mlxu4-kv-bf16", template::chatml),
    ("qwen35-a3b-bf16-kv-bf16", template::chatml),
    ("qwen35-d3b-bf16-kv-bf16", template::chatml),
    ("qwen35-d0.8b-bf16-kv-bf16", template::chatml),
    ("qwen35-d0.8b-mlxu4-kv-bf16", template::chatml),
    ("qwen35-d0.8b-eagle-bf16-kv-bf16", template::chatml),
    ("qwen35-d0.8b-vision-eagle-bf16-kv-bf16", template::chatml),
    ("qwen36-27b-vision-bf16-kv-bf16", template::chatml),
    ("qwen35-d0.8b-vision-bf16-kv-bf16", template::chatml),
    ("qwen35-a3b-bf16-kv-bf16-tp2", template::chatml),
];
