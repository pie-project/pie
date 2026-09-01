pub mod forward;
pub mod import;
pub mod media;
pub mod model;
pub mod template;
pub mod tokenizer;

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
        "qwen38-27b-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d27b(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen36-27b-mlxu4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d27b_undrafted(Dtype::U4g64, Dtype::Bf16, 1)
    ),
    (
        "qwen38-27b-mlxu4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d27b_undrafted(Dtype::U4g64, Dtype::Bf16, 1)
    ),
    (
        "qwen36-35b-a3b-mlxu4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::a3b(Dtype::U4g64, Dtype::Bf16, 1)
    ),
    // The width-invariance fixture's row: the same A3B organs at the
    // miniature's five layers and sixteen routed experts, every other width
    // production. See [`Model::a3b_mini`].
    (
        "qwen36-35b-a3b-mini-mlxu4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::a3b_mini(Dtype::U4g64, Dtype::Bf16, 1)
    ),
    // And the same fixture with a crowded tail: five layers again, sixty-four
    // routed experts against the same top-k 8. See [`Model::a3b_mini64`].
    (
        "qwen36-35b-a3b-mini64-mlxu4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::a3b_mini64(Dtype::U4g64, Dtype::Bf16, 1)
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
        Model::d0_8b(Dtype::U4g64, Dtype::Bf16, 1)
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
        "qwen36-27b-vision-mlxu4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d27b_vision_undrafted(Dtype::U4g64, Dtype::Bf16, 1)
    ),
    (
        "qwen38-27b-vision-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d27b_vision(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen38-27b-vision-mlxu4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d27b_vision_undrafted(Dtype::U4g64, Dtype::Bf16, 1)
    ),
    (
        "qwen35-d0.8b-vision-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d0_8b_vision(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen35-d0.8b-vision-mlxu4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d0_8b_vision(Dtype::U4g64, Dtype::Bf16, 1)
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
/// **AND EVERY `qwen38` ROW IS A SHADOWED TWIN, ON PURPOSE.** Qwen3.8-27B is
/// Qwen3.6-27B's artifact surface tensor for tensor — the two `config.json`s
/// differ in `transformers_version` and in nothing else, the canonical tensor
/// name sets are equal, the `mtp.*` head and the 27-block tower both ship —
/// so no WEIGHT contract can tell the artifacts apart and `identify` walking this
/// order will always answer the `qwen36` row first. The `qwen38` rows exist
/// because the SKU NAME is what selects the chat template, and the template
/// is the one thing 3.8 actually changed: it replays assistant `<think>`
/// blocks instead of stripping them (`template::chatml_interleaved`), and its
/// tokenizer reserves seven audio/tts specials (ids 248070–248076, inside the
/// same vocab). A 3.8 deployment therefore names its row — `[model] sku =
/// "qwen38-27b-…"` — and a deployment that does not gets 3.6's reading, which
/// serves the same bytes with yesterday's replay convention. Each twin sits
/// beside the row it shadows as documentation, not reachability. The twin has
/// since grown teeth of its own kind: [`tokenizer::CONTRACT_38`] pins the
/// seven specials (the one artifact-visible fact that tells the tokenizers
/// apart), so a `qwen38` SKU deployed against a 3.6 artifact refuses at
/// serve boot instead of answering under a reading it was never trained
/// for. `identify` still walks weight contracts alone and still answers
/// `qwen36`; letting it read the pins too is the open item that would
/// retire the sku requirement.
///
/// [`forward::Facts::media`]: super::forward::Facts::media
///
pub const IMPORTS: &[crate::ImportRow] = &[
    ("qwen36-27b-mlxu4-kv-bf16", 1, |src, tp| {
        Model::d27b_undrafted(Dtype::U4g64, Dtype::Bf16, tp).import(src)
    }),
    ("qwen35-d0.8b-mlxu4-kv-bf16", 1, |src, tp| {
        Model::d0_8b(Dtype::U4g64, Dtype::Bf16, tp).import(src)
    }),
    // **AHEAD OF `qwen35-a3b-bf16`, WHICH IS THE SAME FORTY LAYERS**
    // (campaign M-5). The two rows read one architecture — every number
    // `Model::a3b` states was checked against
    // `mlx-community/Qwen3.6-35B-A3B-4bit`'s own `text_config` and matches to
    // the digit — so the ordering rule above is what separates them, and it
    // separates them the strict way: this row's every projection is a
    // `.weight`/`.scales`/`.biases` triplet, and a bf16 A3B checkpoint holds
    // only the first.
    ("qwen36-35b-a3b-mlxu4-kv-bf16", 1, |src, tp| {
        Model::a3b(Dtype::U4g64, Dtype::Bf16, tp).import(src)
    }),
    // **AND THE MINIATURE ASKS AFTER THE SHIPPED ROW**, the way
    // `qwen38-flash-mlxu2` asks after `qwen38-flash-mlxu4`. The two cannot
    // both hold: the full artifact has forty layers where this row declares
    // five, and its routed banks lead with 256 experts where this row declares
    // sixteen — so each misses on the other's file on shape, not on order.
    // The shipped artifact still keeps its turn first, because it is the one
    // `identify` is gated on.
    ("qwen36-35b-a3b-mini-mlxu4-kv-bf16", 1, |src, tp| {
        Model::a3b_mini(Dtype::U4g64, Dtype::Bf16, tp).import(src)
    }),
    // **AND THE TWO MINIATURES SEPARATE ON THE SAME AXIS THE PAIR ABOVE DOES**,
    // which is why their relative order carries nothing. They are one geometry
    // apart from the routed bank's leading dimension — sixteen against
    // sixty-four — and a row that declares one of those misses a file that
    // stores the other on shape, in both directions. So neither can claim the
    // other's carve however the walk reaches them, and this row sits beside its
    // sibling as documentation rather than as reachability.
    ("qwen36-35b-a3b-mini64-mlxu4-kv-bf16", 1, |src, tp| {
        Model::a3b_mini64(Dtype::U4g64, Dtype::Bf16, tp).import(src)
    }),
    ("qwen36-27b-bf16-kv-bf16", 1, |src, tp| {
        Model::d27b(Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
    ("qwen38-27b-bf16-kv-bf16", 1, |src, tp| {
        Model::d27b(Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
    ("qwen38-27b-mlxu4-kv-bf16", 1, |src, tp| {
        Model::d27b_undrafted(Dtype::U4g64, Dtype::Bf16, tp).import(src)
    }),
    ("qwen35-a3b-bf16-kv-bf16", 1, |src, tp| {
        Model::a3b(Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
    ("qwen35-d3b-bf16-kv-bf16", 1, |src, tp| {
        Model::d3b(Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
    ("qwen35-d0.8b-eagle-bf16-kv-bf16", 1, |src, tp| {
        Model::d0_8b_eagle(Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
    ("qwen35-d0.8b-bf16-kv-bf16", 1, |src, tp| {
        Model::d0_8b(Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
    ("qwen35-a3b-bf16-kv-bf16-tp2", 2, |src, tp| {
        Model::a3b(Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
    ("qwen35-d0.8b-vision-eagle-bf16-kv-bf16", 1, |src, tp| {
        Model::d0_8b_vision_eagle(Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
    // **AND WITHIN THE VISION TAIL THE 4-BIT ROWS GO FIRST**, which is the
    // file's own strictness rule applied one level down: a `-vision-mlxu4`
    // row asks for a triplet at every TRUNK projection and a bf16 vision
    // checkpoint holds only the first, so it misses and the bf16 vision row
    // below gets its turn; the reverse would claim an MLX artifact.
    //
    // These rows exist because the artifacts do. `mlx-community/
    // Qwen3.6-27B-4bit` and `mlx-community/Qwen3.5-0.8B-4bit` each publish
    // the whole tower — 333 and 153 tensors — beside a 4-bit trunk, and
    // until they had a row the only way to read one was to drop its tower on
    // the floor and serve the text.
    ("qwen36-27b-vision-mlxu4-kv-bf16", 1, |src, tp| {
        Model::d27b_vision_undrafted(Dtype::U4g64, Dtype::Bf16, tp).import(src)
    }),
    ("qwen36-27b-vision-bf16-kv-bf16", 1, |src, tp| {
        Model::d27b_vision(Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
    ("qwen38-27b-vision-mlxu4-kv-bf16", 1, |src, tp| {
        Model::d27b_vision_undrafted(Dtype::U4g64, Dtype::Bf16, tp).import(src)
    }),
    ("qwen38-27b-vision-bf16-kv-bf16", 1, |src, tp| {
        Model::d27b_vision(Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
    ("qwen35-d0.8b-vision-mlxu4-kv-bf16", 1, |src, tp| {
        Model::d0_8b_vision(Dtype::U4g64, Dtype::Bf16, tp).import(src)
    }),
    ("qwen35-d0.8b-vision-bf16-kv-bf16", 1, |src, tp| {
        Model::d0_8b_vision(Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
];

pub const TEMPLATES: &[crate::template::TemplateRow] = &[
    ("qwen36-27b-bf16-kv-bf16", template::chatml),
    ("qwen38-27b-bf16-kv-bf16", template::chatml_interleaved),
    ("qwen38-27b-mlxu4-kv-bf16", template::chatml_interleaved),
    ("qwen38-27b-vision-bf16-kv-bf16", template::chatml_interleaved),
    ("qwen38-27b-vision-mlxu4-kv-bf16", template::chatml_interleaved),
    ("qwen36-27b-mlxu4-kv-bf16", template::chatml),
    ("qwen36-35b-a3b-mlxu4-kv-bf16", template::chatml),
    ("qwen36-35b-a3b-mini-mlxu4-kv-bf16", template::chatml),
    ("qwen36-35b-a3b-mini64-mlxu4-kv-bf16", template::chatml),
    ("qwen35-a3b-bf16-kv-bf16", template::chatml),
    ("qwen35-d3b-bf16-kv-bf16", template::chatml),
    ("qwen35-d0.8b-bf16-kv-bf16", template::chatml),
    ("qwen35-d0.8b-mlxu4-kv-bf16", template::chatml),
    ("qwen35-d0.8b-eagle-bf16-kv-bf16", template::chatml),
    ("qwen35-d0.8b-vision-eagle-bf16-kv-bf16", template::chatml),
    ("qwen36-27b-vision-bf16-kv-bf16", template::chatml),
    ("qwen36-27b-vision-mlxu4-kv-bf16", template::chatml),
    ("qwen35-d0.8b-vision-bf16-kv-bf16", template::chatml),
    ("qwen35-d0.8b-vision-mlxu4-kv-bf16", template::chatml),
    ("qwen35-a3b-bf16-kv-bf16-tp2", template::chatml),
];

pub const TOKENIZERS: &[crate::tokenizer::ContractRow] = &[
    ("qwen36-27b-bf16-kv-bf16", &tokenizer::CONTRACT),
    ("qwen38-27b-bf16-kv-bf16", &tokenizer::CONTRACT_38),
    ("qwen36-27b-mlxu4-kv-bf16", &tokenizer::CONTRACT),
    ("qwen38-27b-mlxu4-kv-bf16", &tokenizer::CONTRACT_38),
    ("qwen36-35b-a3b-mlxu4-kv-bf16", &tokenizer::CONTRACT),
    ("qwen36-35b-a3b-mini-mlxu4-kv-bf16", &tokenizer::CONTRACT),
    ("qwen36-35b-a3b-mini64-mlxu4-kv-bf16", &tokenizer::CONTRACT),
    ("qwen35-a3b-bf16-kv-bf16", &tokenizer::CONTRACT),
    ("qwen35-d3b-bf16-kv-bf16", &tokenizer::CONTRACT),
    ("qwen35-d0.8b-bf16-kv-bf16", &tokenizer::CONTRACT),
    ("qwen35-d0.8b-mlxu4-kv-bf16", &tokenizer::CONTRACT),
    ("qwen35-d0.8b-vision-eagle-bf16-kv-bf16", &tokenizer::CONTRACT_VISION),
    ("qwen36-27b-vision-bf16-kv-bf16", &tokenizer::CONTRACT_VISION),
    ("qwen36-27b-vision-mlxu4-kv-bf16", &tokenizer::CONTRACT_VISION),
    ("qwen35-d0.8b-vision-mlxu4-kv-bf16", &tokenizer::CONTRACT_VISION),
    ("qwen38-27b-vision-bf16-kv-bf16", &tokenizer::CONTRACT_38_VISION),
    ("qwen38-27b-vision-mlxu4-kv-bf16", &tokenizer::CONTRACT_38_VISION),
    ("qwen35-d0.8b-vision-bf16-kv-bf16", &tokenizer::CONTRACT_VISION),
    ("qwen35-d0.8b-eagle-bf16-kv-bf16", &tokenizer::CONTRACT),
    ("qwen35-a3b-bf16-kv-bf16-tp2", &tokenizer::CONTRACT),
];
