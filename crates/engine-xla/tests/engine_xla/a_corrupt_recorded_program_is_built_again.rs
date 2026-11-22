//! A traced-fire record whose program file in the disk cache does not load
//! (here overwritten with bytes no program deserializes from) is not the
//! load's failure: the record is dropped with the file, its fire traced
//! again, and its program built and kept again, the load serving the logits
//! a sound cache gives; the load after takes every record again. Skipped
//! without a PJRT plugin or a disk cache.

use crate::a_stale_trace_record_is_traced_again::{argmax, served};

/// The loads' context: one no other test loads at, so the records' scope
/// is this test's own.
const CONTEXT: u32 = 352;

#[test]
fn a_corrupt_recorded_program_is_built_again() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let (want, _, dir) = served(CONTEXT);
    let Some(dir) = dir.filter(|d| d.is_dir()) else {
        return;
    };
    // a record's program file, overwritten
    let mut records: Vec<_> = std::fs::read_dir(&dir)
        .expect("the records' folder reads")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_none())
        .collect();
    records.sort();
    let record: serde_json::Value = serde_json::from_slice(
        &std::fs::read(records.first().expect("the load kept its fires' records"))
            .expect("a record reads"),
    )
    .expect("a record parses");
    let name = record["name"].as_str().expect("a record names its program");
    let cache = dir
        .parent()
        .and_then(std::path::Path::parent)
        .expect("records lie under the disk cache");
    let file = cache.join(name);
    let corrupt = b"no program deserializes from these bytes".to_vec();
    std::fs::write(&file, &corrupt).expect("the program file is overwritten");

    let (got, (_, hold, mended), _) = served(CONTEXT);
    assert!(
        hold,
        "a record whose program does not load parts from nothing"
    );
    assert!(
        mended > 0,
        "the record whose program does not load is mended"
    );
    let kept = std::fs::read(&file).expect("the program is kept again");
    assert_ne!(
        kept, corrupt,
        "the corrupt file is replaced by the built program"
    );
    assert_eq!(got.len(), want.len());
    for (step, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(argmax(g), argmax(w), "step {step}: a sound cache's token");
        assert_eq!(g, w, "step {step}: a sound cache's logits");
    }

    let (again, (used, hold, mended), _) = served(CONTEXT);
    assert!(hold && used > 0, "the mended records are taken");
    assert_eq!(mended, 0, "nothing left to mend");
    for (step, (g, w)) in again.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "step {step}: a mended cache's logits");
    }
}
