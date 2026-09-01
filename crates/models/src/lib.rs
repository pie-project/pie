pub mod adapter;
pub mod deepseek_v4;
pub mod gemma_4;
pub mod glm_5;
pub mod gpt_oss;
pub mod kimi_k3;
pub mod media;
pub mod qwen_3;
pub mod qwen_4;
pub mod template;
pub mod tokenizer;

use checkpoint::contract::ModelContract;
use model_dsl::Dtype;

/// The vocabulary a caller needs to USE the catalog's fourth column, through
/// the same door the column comes out of. A party that holds a [`ClassifyFn`]
/// has to build the [`Request`] it takes, and making it name `model_dsl` for
/// that would put the authoring eDSL in the dependency graph of everyone who
/// wants a lane's word — the runtime's fire path, which authors nothing.
/// **AND THE SAME ARGUMENT PUTS `placing_for` ON THIS DOOR** (§J4c). A
/// caller that holds one of [`imports`]'s function pointers is stating a load
/// contract, and a family's text may declare a `Dtype` PLACEMENT — an
/// arrangement of a bank's bytes that some platforms' kernels read and others
/// answer nonsense off. `model_dsl::place` resolves one against the platform
/// the declaration is read under; the trace column gets that from `catalog!`,
/// and an [`ImportRow`] — `fn(&Source)`, the checkpoint's shape being the only
/// argument a contract has ever taken — gets it from a caller that wraps the
/// call. `runtime::engine::load` is the shipping one and it does not depend on
/// `model_dsl` on purpose (its `Cargo.toml` says why: nothing there authors a
/// forward pass). So the word comes out the door the column comes out of.
/// **AND THE TWO FUNCTIONS THAT MINT A COMPANION'S NAME**, out of the same
/// door and for the same reason.
///
/// `scales_name` and `biases_name` are what a model text calls to declare
/// `<w>.scales` and `<w>.biases`, so asking them of a trace's param names is
/// the minting function run FORWARDS — not the suffix match this tree refuses
/// by name, which is what pairs a plane with a weight it never belonged to.
/// `runtime::engine::load::sequence` needs that pairing to rank an artifact's
/// planes at import, where there is no compiled load plan to read it off,
/// and it does not depend on `model_dsl` for the reason above.
pub use model_dsl::{ClassifyFn, Request, biases_name, place, placing_for, scales_name};

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

/// A family's reading of a checkpoint, at a stated world width.
///
/// **THE WIDTH IS AN ARGUMENT AND NOT A LITERAL, WHICH IS THE WHOLE OF WHY
/// THE ROW BELOW CAN BE CHECKED.** It used to be written inside the closure —
/// `Model::k3(.., 1)` — where nothing outside the family module could read it,
/// and six rows said 1 while their catalog row said 2. A column beside the
/// closure would only have moved the problem: two numbers in one row, and
/// nothing to say which one the model was built at. Taking it as a parameter
/// leaves ONE number per table, and then the two tables can be compared.
pub type ImportFn = fn(&ztensor::Source, u32) -> Result<ModelContract, checkpoint_dsl::Error>;

/// One row of the import table: a SKU's name, the tensor-parallel width its
/// contract is built at, and how it reads a checkpoint.
///
/// **THE SECOND COLUMN IS [`Row`]'S SECOND COLUMN, AND [`imports`] HOLDS THEM
/// EQUAL.** The catalog's says what the SKU is TRACED for — how many ranks its
/// bands are cut for, and what §M's artifact stamp will say the file is. This
/// one says what the contract that LANDS those bands is built at. They are one
/// deployment fact written twice, and until §M-4g nothing compared them.
///
/// **WHAT THE DISAGREEMENT COSTS TODAY, SAID EXACTLY.** A `read_own` landing
/// is tp-INVARIANT as `checkpoint_dsl::claim` is written — a weight's declared
/// shape is `per-rank × tp` and an `Expr::Shard` carries an axis and no world
/// — so the contract a tp2 row states at tp 1 happens to equal the one it
/// states at tp 2, and `contract_for` handing a two-rank trace a one-rank
/// model was survivable by ACCIDENT rather than by anything either table said.
/// What the number does change is the door: the foreign verbs refuse `tp > 1`
/// (`Builder::whole_checkpoint` — a sharded deployment does not import a
/// checkpoint nothing has banded), so a tp2 row that claimed tp 1 could take a
/// whole HuggingFace file down an arm written for a one-rank world. The
/// invariance is a property of one function in another crate; the agreement is
/// a property of this table, and it is the one worth stating.
pub type ImportRow = (&'static str, u32, ImportFn);

#[must_use]
pub fn catalog() -> Vec<Row> {
    [
        deepseek_v4::CATALOG,
        gemma_4::CATALOG,
        glm_5::CATALOG,
        gpt_oss::CATALOG,
        kimi_k3::CATALOG,
        qwen_3::CATALOG,
        qwen_4::CATALOG,
    ]
    .concat()
}

/// **THE ONE DOOR THE IMPORT TABLE COMES OUT OF, AND IT REFUSES A BUILD WHOSE
/// TWO TABLES DISAGREE.**
///
/// `identify`, [`import_of`], `runtime::engine::load::{identify,
/// conversion_contract}` and every test that walks the rows all draw them from
/// here, so this is the place a disagreement can be caught once and caught on
/// every path. It is an `assert!` and not a `debug_assert!`: the fact is about
/// two `const` tables compiled into the same binary, so it is either wrong in
/// every build or wrong in none, and the release build is the one that writes
/// the artifact. See [`tp_disagreements`] for what it refuses and why.
#[must_use]
pub fn imports() -> Vec<ImportRow> {
    let faults = tp_disagreements();
    assert!(
        faults.is_empty(),
        "the import table and the catalog state different worlds for {} row(s):\n{}",
        faults.len(),
        faults.join("\n"),
    );
    [
        deepseek_v4::IMPORTS,
        gemma_4::IMPORTS,
        glm_5::IMPORTS,
        gpt_oss::IMPORTS,
        kimi_k3::IMPORTS,
        qwen_3::IMPORTS,
        qwen_4::IMPORTS,
    ]
    .concat()
}

/// **THE SERVED NUMERIC FORM OF EVERY CATALOG ROW**, stated once.
///
/// `checkpoint::serving::Stamp` compares `precision` field by field and
/// `serving::Name` puts it in the artifact's FILENAME, so it is the field that
/// makes *one model at two quantizations* two files. Nothing else in this
/// crate carries it: [`Row`] is `(sku, tp, TraceFn, ClassifyFn)`.
///
/// # It is written down because it cannot be read off the name
///
/// The obvious move is to slice the segment before `-kv-`, and the catalog
/// refutes it:
///
/// ```text
/// gptoss-20b-bf16-mxfp4-kv-bf16     bf16 dense, mxfp4 experts — TWO
/// gptoss-20b-mlxu4-mxfp4-kv-bf16    mlxu4 dense, mxfp4 experts
/// glm5-a12b-bf16-bf16-kv-bf16       two bf16 segments
/// qwen35-d0.8b-mlxu4-kv-bf16        one
/// ```
///
/// A parse taking one segment is wrong for every MoE row and wrong SILENTLY:
/// the stamp would say `bf16` for a file whose experts are mxfp4, and
/// `Stamp::check` would pass it against a deployment that computed `bf16` the
/// same wrong way. Two halves agreeing on one mistake is the worst shape a
/// checked field can have.
///
/// # And it is not derived from the trace either
///
/// That would be a classification table from a dtype multiset to a canonical
/// string — the shape this tree keeps being bitten by. Five such tables were
/// missing their `U2` rows when 2-bit opened and `QuantSpec::term`'s `4 | 8`
/// whitelist was a sixth. What the trace IS good for is checking this table,
/// which is what `every_row_states_a_precision_its_trace_agrees_with` does:
/// each token here must be witnessed by a param's dtype and every quantized
/// param must be named by a token, in both directions.
///
/// The spelling is dense-then-experts, deduplicated: `glm5-a12b-bf16-bf16` is
/// `bf16` because both halves are, and `gptoss-20b-bf16-mxfp4` is
/// `bf16-mxfp4` because they differ.
///
/// # The two DQ rows are why this table beats the SKU name, and I got them
/// # wrong on the first pass
///
/// `dsv4-flash-mlxu2-kv-bf16` and `qwen38-flash-mlxu2-kv-bf16` are NAMED for
/// their two-bit expert banks and their traces are mostly FOUR-bit:
///
/// ```text
/// dsv4-flash-mlxu2    15 params U2g32/U2g64 (experts), 51 U4g64, 175 bf16
/// qwen38-flash-mlxu2   8 params U2g128     (experts), 50 U4g64/U4g32, 147 bf16
/// ```
///
/// A DQ checkpoint spends its bits per tensor, so the SKU name states the
/// HEADLINE and this table states the fact. Both are `mlxu4-mlxu2`, and they
/// went in as `mlxu2` read off the name — which is exactly the mistake this
/// table exists to make impossible, and which
/// `every_row_states_a_precision_its_trace_agrees_with` caught on its first
/// run. It is also the clearest answer to "why is `precision` a field when
/// `sku` already implies it": here, it does not.
const PRECISIONS: &[(&str, &str)] = &[
    ("dsv4-base-bf16-kv-bf16", "bf16"),
    ("dsv4-base-bf16-kv-bf16-tp2", "bf16"),
    ("dsv4-flash-bf16-kv-bf16", "bf16"),
    ("dsv4-flash-mlxu2-kv-bf16", "mlxu4-mlxu2"),
    ("gemma4-e4b-eagle-bf16-kv-bf16", "bf16"),
    ("gemma4-e4b-vision-bf16-kv-bf16", "bf16"),
    ("gemma4-e4b-bf16-kv-bf16", "bf16"),
    ("gemma4-26b-a4b-mlxu4-kv-bf16", "mlxu4"),
    ("gemma4-26b-a4b-vision-mlxu4-kv-bf16", "mlxu4"),
    ("gemma4-31b-bf16-kv-bf16", "bf16"),
    ("gemma4-31b-mlxu4-kv-bf16", "mlxu4"),
    ("gemma4-31b-vision-mlxu4-kv-bf16", "mlxu4"),
    ("gemma4-31b-bf16-kv-bf16-tp2", "bf16"),
    ("glm5-a12b-bf16-bf16-kv-bf16", "bf16"),
    ("glm5-a12b-bf16-bf16-kv-bf16-tp2", "bf16"),
    ("gptoss-20b-mlxu4-mxfp4-kv-bf16", "mlxu4-mxfp4"),
    ("gptoss-20b-bf16-mxfp4-kv-bf16", "bf16-mxfp4"),
    ("gptoss-120b-bf16-mxfp4-kv-bf16", "bf16-mxfp4"),
    ("gptoss-120b-bf16-mxfp4-kv-bf16-tp2", "bf16-mxfp4"),
    ("kimik3-bf16-mxfp4-kv-bf16", "bf16-mxfp4"),
    ("kimik3-bf16-mxfp4-kv-bf16-tp2", "bf16-mxfp4"),
    ("qwen36-27b-bf16-kv-bf16", "bf16"),
    ("qwen38-27b-bf16-kv-bf16", "bf16"),
    ("qwen36-27b-mlxu4-kv-bf16", "mlxu4"),
    ("qwen38-27b-mlxu4-kv-bf16", "mlxu4"),
    ("qwen36-35b-a3b-mlxu4-kv-bf16", "mlxu4"),
    ("qwen35-a3b-bf16-kv-bf16", "bf16"),
    ("qwen35-d3b-bf16-kv-bf16", "bf16"),
    ("qwen35-d0.8b-bf16-kv-bf16", "bf16"),
    ("qwen35-d0.8b-mlxu4-kv-bf16", "mlxu4"),
    ("qwen35-d0.8b-vision-eagle-bf16-kv-bf16", "bf16"),
    ("qwen36-27b-vision-bf16-kv-bf16", "bf16"),
    ("qwen36-27b-vision-mlxu4-kv-bf16", "mlxu4"),
    ("qwen38-27b-vision-bf16-kv-bf16", "bf16"),
    ("qwen38-27b-vision-mlxu4-kv-bf16", "mlxu4"),
    ("qwen35-d0.8b-vision-bf16-kv-bf16", "bf16"),
    ("qwen35-d0.8b-vision-mlxu4-kv-bf16", "mlxu4"),
    ("qwen35-d0.8b-eagle-bf16-kv-bf16", "bf16"),
    ("qwen35-a3b-bf16-kv-bf16-tp2", "bf16"),
    ("qwen38-flash-mlxu4-kv-bf16", "mlxu4"),
    ("qwen38-flash-bf16-kv-bf16", "bf16"),
    ("qwen38-flash-mlxu2-kv-bf16", "mlxu4-mlxu2"),
];

/// The served numeric form [`PRECISIONS`] states for `sku`.
///
/// `None` for a row this build does not ship, which is the same answer
/// [`catalog`] gives it.
#[must_use]
pub fn precision_of(sku: &str) -> Option<&'static str> {
    PRECISIONS
        .iter()
        .find(|(row, _)| *row == sku)
        .map(|(_, precision)| *precision)
}

/// Every row whose precision is missing from [`PRECISIONS`], and every entry
/// there that names no catalog row — one sentence each, in BOTH directions.
///
/// [`tp_disagreements`]'s shape and its reason: a table beside a table is only
/// as good as the thing that holds the two together, and a completeness check
/// that ran one way would let a deleted row keep its entry forever.
#[must_use]
pub fn precision_disagreements() -> Vec<String> {
    let rows = catalog();
    let mut out = Vec::new();
    for (sku, _, _, _) in &rows {
        if precision_of(sku).is_none() {
            out.push(format!(
                "`{sku}` is in the catalog and states no precision; add it to \
                 `PRECISIONS` — the stamp and the artifact's own filename both \
                 carry this field"
            ));
        }
    }
    for (sku, _) in PRECISIONS {
        if !rows.iter().any(|(row, _, _, _)| row == sku) {
            out.push(format!(
                "`PRECISIONS` states a precision for `{sku}`, which no catalog row \
                 names any more"
            ));
        }
    }
    out
}

/// Every row whose import table and catalog row name a different number of
/// ranks, one sentence each, NAMING BOTH NUMBERS AND THE ROW.
///
/// **WHY BOTH NUMBERS AND NOT JUST "THEY DISAGREE".** The two are not
/// interchangeable readings of one fact: the catalog's is what the SKU is
/// TRACED for — how many ranks the plan cuts its bands for, and what §M's
/// stamp will say the artifact is — and the import table's is what the
/// contract that LANDS those bands is built at. An operator reading a refusal
/// has to know which one is the lie, and only the pair says it.
///
/// Read off the family `const`s and not off [`imports`], because [`imports`]
/// asserts on this and a walk that went through it could never report a second
/// fault. That is also what lets the gate in `the_zt_contract_states_the_cut`
/// report the FULL list.
#[must_use]
pub fn tp_disagreements() -> Vec<String> {
    let traced = catalog();
    let mut faults = Vec::new();
    for (sku, imported, _) in [
        deepseek_v4::IMPORTS,
        gemma_4::IMPORTS,
        glm_5::IMPORTS,
        gpt_oss::IMPORTS,
        kimi_k3::IMPORTS,
        qwen_3::IMPORTS,
        qwen_4::IMPORTS,
    ]
    .concat()
    {
        let Some((_, traced, ..)) = traced.iter().find(|(name, ..)| *name == sku) else {
            // A row naming no SKU is `every_sku_ships_whole`'s fault to
            // report, and reporting it twice under a worse name would send an
            // operator looking for a rank count that is not the problem.
            continue;
        };
        if imported != *traced {
            faults.push(format!(
                "`{sku}` is imported at tp {imported} and traced at tp {traced}: one \
                 deployment, two widths. `models::import_of(\"{sku}\")` — which is \
                 what a serving load reaches through \
                 `runtime::engine::load::contract_for` — would state this SKU's \
                 contract over a world of {imported} rank(s) while its plan, its \
                 bands and §M's artifact stamp all say {traced}. Whichever number \
                 is the lie, one of the two tables has to change: the catalog row \
                 is the traced fact, so ordinarily it is the import row that \
                 follows it."
            ));
        }
    }
    faults
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
) -> Option<impl Fn(&ztensor::Source) -> Result<ModelContract, checkpoint_dsl::Error>> {
    let (_, tp, make) = imports().into_iter().find(|(n, ..)| *n == sku)?;
    // The row's own width, closed over here rather than asked of the caller:
    // every call site in the tree wants "read this checkpoint as this SKU",
    // and a `u32` in that signature would be a fifty-first place for the
    // number to be written down wrong.
    Some(move |src: &ztensor::Source| make(src, tp))
}

/// The dtype the planes BESIDE a bank of `banks` are stated in.
///
/// **A NORM IS NOT A BANK, AND A QUANTIZED SKU IS NOT A SECOND FAMILY TEXT.**
/// A family's `new` takes one weight representation and stamps it on every
/// plane it declares, which was right while every representation stored itself
/// verbatim. `Mxfp4` and `U4g64` do not: they are a bank's codes, they come
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


pub fn identify(src: &ztensor::Source) -> Result<&'static str, Unmatched> {
    let mut misses: Vec<(&'static str, String)> = Vec::new();
    let rows = catalog();

    for (sku, tp, import) in imports() {
        if rows.iter().any(|row| row.0 == sku && row.1 > 1) {
            continue;
        }
        match import(src, tp) {
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
