pub mod deployment;
pub mod deepseek_v4;
pub mod gemma_4;
pub mod glm_5;
pub mod gpt_oss;
pub mod kimi_k3;
pub mod produce;
pub mod qwen_3_5;
/// A SECOND `Instruct` TRAIT STOOD BESIDE THIS ONE, and R3 deleted it.
///
/// `crate::instruct` was the thin `push(u32) -> Option<String>` sketch that
/// arrived with the family takeover, together with `crate::decoders` and six
/// per-family `template.rs` files that implemented it — 1 216 lines forming a
/// closed island with no consumer inside this crate or outside it. R1 flagged
/// collapsing the two as a decision about the GUEST CONTRACT and left the
/// sketch standing rather than choose.
///
/// It is not that decision. The contract-shaped question is which trait a
/// guest is written against, and only one of the two was ever an answer:
/// [`serve::instruct`] is what the WIT surface says (`feed(&[u32]) ->
/// ChatEvent`, `reset`, a tool grammar) and what `engine` links. The sketch
/// could not answer that surface as written and nothing called it. Deleting
/// code with no consumer is not a contract decision, so it is deleted.
///
/// The serving half — chat templates, multimodal decode, the per-SKU
/// `(layers, vocab)` a sampler is sized from, and the artifact metadata an
/// engine is handed. Moved here from `model-legacy`'s `chat` feature by the
/// R1 cutover; see its module doc.
///
/// NOT feature-gated as a whole any more. R3 made `driver-cuda` read
/// [`serve::ROWS`] for the architecture label and the context ceiling it
/// advertises, and a driver must not link twenty image crates to get a
/// `&'static str`. The `serve` feature now gates exactly what needs a codec:
/// [`serve::multimodal`].
pub mod serve;
pub mod snapshot;

pub fn catalog() -> Vec<(&'static str, fn(model_dsl::Plane) -> model_dsl::Plan)> {
    [
        deepseek_v4::CATALOG,
        gemma_4::CATALOG,
        glm_5::CATALOG,
        gpt_oss::CATALOG,
        kimi_k3::CATALOG,
        qwen_3_5::CATALOG,
    ]
    .concat()
}

/// Every shipping import point, across all families.
///
/// Shorter than [`catalog`] and always will be: a tensor-parallel row is the
/// same bytes cut a different way at load, so it names no import of its own.
pub fn imports() -> Vec<model_dsl::load::ImportRow> {
    [
        deepseek_v4::IMPORTS,
        gemma_4::IMPORTS,
        glm_5::IMPORTS,
        gpt_oss::IMPORTS,
        kimi_k3::IMPORTS,
        qwen_3_5::IMPORTS,
    ]
    .concat()
}

/// The trace fn for `sku`, or `None` if no row ships under that name.
pub fn trace_of(sku: &str) -> Option<fn(model_dsl::Plane) -> model_dsl::Plan> {
    catalog().into_iter().find(|(n, _)| *n == sku).map(|(_, f)| f)
}

/// The import table that builds `sku` from a `base`-flavored checkpoint.
///
/// Both halves of the key are required. Asking for a SKU alone would pick
/// whichever flavor was filed first, which for Gemma is a coin flip between
/// its safetensors release and its GGUF one.
pub fn import_of(sku: &str, base: &str) -> Option<model_dsl::load::Import> {
    imports()
        .into_iter()
        .find(|r| r.sku == sku && r.base == base)
        .map(|r| (r.make)())
}

/// Every checkpoint flavor `sku` can be built from, in table order.
pub fn bases_for(sku: &str) -> Vec<&'static str> {
    imports()
        .into_iter()
        .filter(|r| r.sku == sku)
        .map(|r| r.base)
        .collect()
}

/// Which SKU a checkpoint IS, asked of its tensors.
///
/// # Why the import table is the manifest
///
/// `model-legacy` answered this from a hand-written `Manifest` per catalog
/// row — a second list of tensor names and shapes, kept in step with the
/// load contract that read them. There is no second list here: an
/// [`Import`](model_dsl::load::Import) already names every tensor a SKU is
/// built from, because that is what production reads, and a checkpoint that
/// is missing one cannot be produced at all. So identification asks exactly
/// the question the load is about to ask, one step earlier and without
/// reading a byte of payload.
///
/// Two things are checked, and the second is what makes the answer unique.
/// The NAMES separate the families and the depths — a 40-layer row reads
/// `layer.39.*` and a 24-layer one does not. The `embed` row's leading
/// extent separates two SKUs of the same shape and different vocabularies,
/// which is the whole difference between `qwen35-d0.8b` (248 320) and
/// `qwen35-d3b` (151 936).
///
/// `shape_of` is the caller's reader: it answers a canonical tensor name
/// with that tensor's extents, or `None` for a name the checkpoint does not
/// hold. Whoever opens the file owns the spelling, which is the same rule
/// [`crate::snapshot`] states.
///
/// A tensor-parallel row is never an answer: `-tp2` SKUs name no import of
/// their own (the same bytes, cut at load), so they are not candidates and a
/// TP deployment states its SKU.
///
/// # Errors
///
/// No row matched, or more than one did — both name what they found, because
/// "this checkpoint is not one of these" and "two rows no checkpoint can tell
/// apart" are different problems with different fixes.
pub fn identify(shape_of: &dyn Fn(&str) -> Option<Vec<u64>>) -> Result<&'static str, Unmatched> {
    let mut matched: Vec<&'static str> = Vec::new();
    let mut misses: Vec<(&'static str, String)> = Vec::new();
    for row in imports() {
        let Some(serving) = serve::row(row.sku) else {
            // A catalog row with no serving row cannot be reported as loaded,
            // so it is not an identification this build can act on.
            continue;
        };
        let import = (row.make)();
        match matches(&import, serving.vocab, shape_of) {
            Ok(()) => {
                if !matched.contains(&row.sku) {
                    matched.push(row.sku);
                }
            }
            Err(why) => misses.push((row.sku, why)),
        }
    }
    match matched.len() {
        1 => Ok(matched[0]),
        0 => {
            misses.sort_by(|a, b| a.0.cmp(b.0));
            misses.dedup_by(|a, b| a.0 == b.0);
            Err(Unmatched::NoRow { misses })
        }
        _ => Err(Unmatched::Ambiguous { skus: matched }),
    }
}

/// Whether `import` can be produced from a checkpoint whose tensors
/// `shape_of` answers for, and whose `embed` is `vocab` rows deep.
fn matches(
    import: &model_dsl::load::Import,
    vocab: u32,
    shape_of: &dyn Fn(&str) -> Option<Vec<u64>>,
) -> Result<(), String> {
    let mut sources = Vec::new();
    for row in &import.rows {
        leaves(&row.source, &mut sources);
    }
    let mut absent: Vec<&str> = sources
        .iter()
        .copied()
        .filter(|name| shape_of(name).is_none())
        .collect();
    if !absent.is_empty() {
        absent.sort_unstable();
        let total = absent.len();
        absent.truncate(3);
        return Err(format!(
            "{total} of its {} source tensors are not in this checkpoint ({})",
            sources.len(),
            absent.join(", "),
        ));
    }
    let embed = import
        .rows
        .iter()
        .find(|r| r.target == "embed")
        .ok_or_else(|| "this import writes no `embed`, so it names no vocabulary".to_string())?;
    let mut named = Vec::new();
    leaves(&embed.source, &mut named);
    let [source] = named.as_slice() else {
        return Err("its `embed` is built from more than one tensor".to_string());
    };
    let shape = shape_of(source).expect("every source was just found");
    match shape.first() {
        Some(&rows) if rows == u64::from(vocab) => Ok(()),
        Some(&rows) => Err(format!(
            "its `embed` is {rows} rows deep and this row states {vocab}"
        )),
        None => Err("its `embed` source is a scalar".to_string()),
    }
}

/// Every checkpoint tensor name a production source reads, in tree order.
fn leaves<'a>(source: &'a model_dsl::load::Source, into: &mut Vec<&'a str>) {
    use model_dsl::load::Source;
    match source {
        Source::Copy(n)
        | Source::PlusOne(n)
        | Source::ScalarOf(n)
        | Source::Deinterleave(n, _, _)
        | Source::Squeeze(n, _) => into.push(n),
        Source::Pack(each) | Source::Stack(each) => {
            for one in each {
                leaves(one, into);
            }
        }
    }
}

/// Why no single SKU answers for a checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unmatched {
    /// Nothing matched, with each row's own reason.
    NoRow { misses: Vec<(&'static str, String)> },
    /// More than one row matched. Two rows no checkpoint can tell apart are
    /// one row, so this is a catalog defect and says so.
    Ambiguous { skus: Vec<&'static str> },
}

impl std::fmt::Display for Unmatched {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoRow { misses } => {
                write!(f, "this checkpoint matches no SKU this build ships")?;
                for (sku, why) in misses {
                    write!(f, "\n  {sku}: {why}")?;
                }
                Ok(())
            }
            Self::Ambiguous { skus } => write!(
                f,
                "this checkpoint matches {} SKUs equally well ({}); two rows \
                 no checkpoint can tell apart are one row",
                skus.len(),
                skus.join(", "),
            ),
        }
    }
}

impl std::error::Error for Unmatched {}
