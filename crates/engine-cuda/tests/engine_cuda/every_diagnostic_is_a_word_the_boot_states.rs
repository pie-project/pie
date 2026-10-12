use engine_cuda::{Diagnostics, Knobs};

const WORDS: &[(&str, fn(&Diagnostics) -> bool)] = &[
    ("golden-probe", |d| d.golden_probe),
    ("arm-trace", |d| d.arm_trace),
    ("capture-serial", |d| d.capture_serial),
    ("boundary-trace", |d| d.boundary_trace),
    ("reap-trace", |d| d.reap_trace),
    ("trace-census", |d| d.trace_census),
    ("nan-check", |d| d.nan_check),
    ("ptr-trace=decode", |d| {
        d.ptr_trace.as_deref() == Some("decode")
    }),
    ("grid-trace=decode", |d| {
        d.grid_trace.as_deref() == Some("decode")
    }),
    ("plan-trace=all", |d| d.plan_trace.as_deref() == Some("all")),
    ("graph-dot=/tmp/dots", |d| {
        d.graph_dot.as_deref() == Some(std::path::Path::new("/tmp/dots"))
    }),
    ("fuse-chains=off", |d| !d.fuse_chains),
    ("gumbel-direct=off", |d| !d.gumbel_direct),
];

#[test]
fn every_diagnostic_is_a_word_the_boot_states() {
    let silent = Knobs::default().diagnostics;
    assert_eq!(silent, Diagnostics::default());
    assert!(!silent.any(), "silence is silence");
    assert_eq!(silent.words(), None);

    for (word, reads) in WORDS {
        let one: Diagnostics = word
            .parse()
            .unwrap_or_else(|why| panic!("`{word}` is a word this shell speaks: {why}"));
        assert!(reads(&one), "`{word}` did not set the field it names");
        assert!(one.any(), "`{word}` is not silence");
        let mut moved = 0;
        for (_, other) in WORDS {
            if other(&one) != other(&Diagnostics::default()) {
                moved += 1;
            }
        }
        assert_eq!(moved, 1, "`{word}` moved {moved} fields, not one");
    }

    let all = WORDS
        .iter()
        .map(|(word, _)| *word)
        .collect::<Vec<_>>()
        .join(",");
    let every: Diagnostics = all.parse().expect("the vocabulary parses");
    for (word, reads) in WORDS {
        assert!(reads(&every), "`{word}` was lost in the full list");
    }
    let again: Diagnostics = every
        .words()
        .expect("something is on")
        .parse()
        .expect("the record's own spelling parses");
    assert_eq!(every, again, "the record round-trips through its words");

    let why = "goldenprobe"
        .parse::<Diagnostics>()
        .expect_err("a misspelling is not silence");
    assert!(why.contains("goldenprobe"), "the refusal names it: {why}");
    assert!(
        why.contains("golden-probe`"),
        "and lists the vocabulary: {why}"
    );
}
