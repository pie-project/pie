//! A warm load takes each fire's program from its traced-fire record,
//! checking the first against a fresh trace: a record that parts from its
//! fire's trace (here every record made stale, each naming another fire's
//! program) is caught, the load traces every fire in full, serving the
//! logits a traced load gives, and rewrites the records, which the next
//! load takes again. Skipped without a PJRT plugin or a disk cache.

use crate::common_llm::{Model, Refs, Seq, tokens};

/// The loads' context.
const CONTEXT: u32 = 320;

/// A prompt and three decodes: each step's logits, the load's records
/// (fires taken, held) and where they lie.
pub(crate) fn served(
    context: u32,
) -> (Vec<Vec<f32>>, (u64, bool, u64), Option<std::path::PathBuf>) {
    // a context no other test loads at: a scope of the test's own, whose
    // records no other test's load reads while this one makes them stale
    let mut model = Model::load_with("qwen3-micro-text", None, context, 7, None);
    let vocab = model.vocab;
    let mut refs = Refs::of(&model);
    let mut seq = Seq::new(0);
    let mut out = Vec::new();
    let lane = seq.lane(&model, &mut refs, tokens(3, 12, vocab));
    out.extend(model.fire(vec![lane]));
    seq.held += 12;
    for t in 0..3u32 {
        let lane = seq.lane(&model, &mut refs, vec![5 + t]);
        out.extend(model.fire(vec![lane]));
        seq.held += 1;
    }
    let shell = model.engine.shell().expect("a loaded model has its shell");
    (out, shell.trace_records(), shell.trace_records_dir())
}

/// Makes every record in `dir` stale: each names the program the next
/// distinct one does (its text's hash, file and length).
fn make_stale(dir: &std::path::Path) -> usize {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("the records' folder reads")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_none())
        .collect();
    files.sort();
    let mut records: Vec<serde_json::Value> = files
        .iter()
        .map(|f| {
            serde_json::from_slice(&std::fs::read(f).expect("a record reads"))
                .expect("a record parses")
        })
        .collect();
    let program = |r: &serde_json::Value| (r["hash"].clone(), r["name"].clone(), r["len"].clone());
    let mut distinct: Vec<_> = records.iter().map(program).collect();
    distinct.sort_by_key(|p| p.1.to_string());
    distinct.dedup();
    assert!(distinct.len() > 1, "a ladder traces more than one program");
    for r in &mut records {
        let at = distinct
            .iter()
            .position(|p| *p == program(r))
            .expect("its own program");
        let (hash, name, len) = distinct[(at + 1) % distinct.len()].clone();
        r["hash"] = hash;
        r["name"] = name;
        r["len"] = len;
    }
    for (f, r) in files.iter().zip(&records) {
        std::fs::write(f, serde_json::to_vec(r).expect("a record writes"))
            .expect("a record is kept");
    }
    files.len()
}

pub(crate) fn argmax(row: &[f32]) -> usize {
    (0..row.len())
        .max_by(|&a, &b| row[a].total_cmp(&row[b]))
        .unwrap_or(0)
}

#[test]
fn a_stale_trace_record_is_traced_again() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    // a load that keeps records (traced, or taken from earlier runs' ones)
    let (want, _, dir) = served(CONTEXT);
    let Some(dir) = dir.filter(|d| d.is_dir()) else {
        return;
    };
    let made = make_stale(&dir);
    assert!(made > 0, "the load kept its fires' records");

    let (got, (used, hold, _), _) = served(CONTEXT);
    assert!(
        !hold,
        "a stale record parts from its fire's trace and is caught"
    );
    assert_eq!(
        used, 0,
        "a load whose record parted takes none: it traces every fire"
    );
    assert_eq!(got.len(), want.len());
    for (step, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(argmax(g), argmax(w), "step {step}: the traced load's token");
        assert_eq!(g, w, "step {step}: the traced load's logits");
    }

    // the parted load rewrote the records: the next takes them again
    let (again, (used, hold, _), _) = served(CONTEXT);
    assert!(hold, "the rewritten records hold");
    assert!(used > 0, "the rewritten records are taken");
    for (step, (g, w)) in again.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "step {step}: a record load's logits");
    }
}
