//! A traced-fire record whose program file in the disk cache is not as it
//! was kept is mended before ready, never built while serving. A file of
//! another length (here a few bytes no program deserializes from) is found
//! as the ladder reads the records: its fire is traced and built there, the
//! load serving the logits a sound cache gives. A file corrupt within its
//! length is found only as it loads behind serving: its record is dropped
//! with the file (a fire of its shape would be refused by name, nothing
//! built after ready), and the next load builds it before ready. The load
//! after takes every record. Skipped without a PJRT plugin or a disk cache.

use crate::a_stale_trace_record_is_traced_again::{argmax, served};
use crate::common_llm::Model;

/// The loads' context: one no other test loads at, so the records' scope
/// is this test's own.
const CONTEXT: u32 = 352;

/// The program file a record of `dir` names, and that record's path.
fn a_recorded_file(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let mut records: Vec<_> = std::fs::read_dir(dir)
        .expect("the records' folder reads")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_none())
        .collect();
    records.sort();
    let at = records
        .first()
        .expect("the load kept its fires' records")
        .clone();
    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&at).expect("a record reads"))
            .expect("a record parses");
    let name = record["name"].as_str().expect("a record names its program");
    let cache = dir
        .parent()
        .and_then(std::path::Path::parent)
        .expect("records lie under the disk cache");
    (cache.join(name), at)
}

/// A load that fires nothing, its programs loaded behind serving: the
/// records it mended.
fn loaded_only() -> u64 {
    let mut model = Model::load_with("qwen3-micro-text", None, CONTEXT, 7, None);
    let shell = model
        .engine
        .shell_mut()
        .expect("a loaded model has its shell");
    shell.load_parked_programs();
    shell.trace_records().2
}

#[test]
fn a_corrupt_recorded_program_is_built_again() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let (want, _, dir) = served(CONTEXT);
    let Some(dir) = dir.filter(|d| d.is_dir()) else {
        return;
    };
    let same = |got: &[Vec<f32>], what: &str| {
        assert_eq!(got.len(), want.len());
        for (step, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(argmax(g), argmax(w), "step {step}: {what}'s token");
            assert_eq!(g, w, "step {step}: {what}'s logits");
        }
    };

    // another length: mended before ready, in the load that finds it
    let (file, _) = a_recorded_file(&dir);
    let short = b"no program deserializes from these bytes".to_vec();
    std::fs::write(&file, &short).expect("the program file is overwritten");
    let (got, (_, hold, mended), _) = served(CONTEXT);
    assert!(hold, "a record whose file changed parts from nothing");
    assert!(
        mended > 0,
        "the record whose file changed is mended before ready"
    );
    let kept = std::fs::read(&file).expect("the program is built and kept again");
    assert_ne!(
        kept, short,
        "the changed file is replaced by the built program"
    );
    same(&got, "a load mending before ready");

    // its length, corrupt within: dropped behind serving, or built again
    // there where its text is held, before ready
    let (file, _) = a_recorded_file(&dir);
    let len = std::fs::metadata(&file)
        .expect("the program file is there")
        .len();
    let corrupt = vec![0x5a; len as usize];
    std::fs::write(&file, &corrupt).expect("the program file is overwritten");
    assert!(
        loaded_only() > 0,
        "loading behind serving finds the file corrupt"
    );
    // the record the loader took it by is dropped with the file, or, where
    // the loader holds the program's text, the program is built again
    // before ready; records naming the file find it gone or sound at the
    // next load's ladder
    assert!(
        std::fs::read(&file).map_or(true, |kept| kept != corrupt),
        "the corrupt file is dropped or built again"
    );
    // (every record naming the dropped file, mended before ready)
    let (again, (_, hold, _), _) = served(CONTEXT);
    assert!(hold, "the records hold");
    same(&again, "a load building the dropped record's program");

    let (last, (used, hold, mended), _) = served(CONTEXT);
    assert!(hold && used > 0, "every record is taken again");
    assert_eq!(mended, 0, "nothing left to mend");
    same(&last, "a sound cache");
}
