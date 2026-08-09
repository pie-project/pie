//! Building a batch of pipelines, and getting them back from disk.

use std::path::{Path, PathBuf};

use driver_metal::Request;
use driver_metal::device::{Archives, Context};
use driver_metal::program::{Archived, Compiler, Math};

/// A scratch cache directory of this test's own.
///
/// Never the developer's real cache: a test that wrote there would make the
/// next run of itself a hit and stop testing the miss.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "pie-pso-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

/// Write `text` into `dir` and hand back the path.
fn kernel(dir: &Path, name: &str, text: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, text).expect("write kernel");
    path
}

fn source(function: &str) -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void {function}(device uint* out [[buffer(0)]], \
         uint gid [[thread_position_in_grid]]) {{ out[gid] = gid; }}\n"
    )
}

#[test]
fn a_second_run_of_the_same_batch_comes_out_of_the_archive() {
    let dir = scratch("hit");
    let context = Context::new().expect("context");
    let path = kernel(
        &dir,
        "k.metal",
        &format!("{}{}", source("one"), source("two")),
    );
    let requests = vec![Request::new(&path, "one"), Request::new(&path, "two")];

    let archives = Archives::new(Some(dir.join("cache")));

    let first = Compiler::with_archives(&context, archives.clone())
        .expect("compiler")
        .compile_batch(&context, &requests);
    assert!(
        matches!(first.archive, Archived::Written),
        "a cold cache is a miss that writes: {:?}",
        first.archive
    );
    assert_eq!(first.pipelines.len(), 2);
    assert!(first.all().is_ok());

    // A NEW compiler, because a serializer that already holds the binaries
    // would not prove they came off the disk.
    let second = Compiler::with_archives(&context, archives.clone())
        .expect("compiler")
        .compile_batch(&context, &requests);
    assert!(
        second.archive.is_hit(),
        "the same batch against a warm cache must be served, not rebuilt: {:?}",
        second.archive
    );
    assert!(second.all().is_ok());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn editing_the_source_misses_the_archive_it_had_written() {
    let dir = scratch("stale");
    let context = Context::new().expect("context");
    let path = kernel(&dir, "k.metal", &source("one"));
    let requests = vec![Request::new(&path, "one")];
    let archives = Archives::new(Some(dir.join("cache")));

    let first = Compiler::with_archives(&context, archives.clone())
        .expect("compiler")
        .compile_batch(&context, &requests);
    assert!(matches!(first.archive, Archived::Written));

    // Same path, same entry point, different body. Serving the old binary
    // here would be a silent miscompile rather than a slow start.
    std::fs::write(
        &path,
        source("one").replace("out[gid] = gid;", "out[gid] = gid + 1u;"),
    )
    .expect("edit");

    let second = Compiler::with_archives(&context, archives)
        .expect("compiler")
        .compile_batch(&context, &requests);
    assert!(
        matches!(second.archive, Archived::Written),
        "an edited source is a different key, so a miss: {:?}",
        second.archive
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The math mode is part of what a source compiles to, so it must be part of
/// the key.
///
/// Without it the precise batch is served the fast batch's binaries and
/// reports a hit. That is the worst shape of failure this cache can have: not
/// a slow start, but the wrong arithmetic arriving quickly and looking
/// correct. A transcode kernel served a reassociated build produces
/// quantisation codes that are off by a step.
#[test]
fn the_two_math_modes_do_not_share_an_archive() {
    let dir = scratch("math");
    let context = Context::new().expect("context");
    let path = kernel(&dir, "k.metal", &source("one"));
    let requests = vec![Request::new(&path, "one")];
    let archives = Archives::new(Some(dir.join("cache")));

    let fast = Compiler::with_archives(&context, archives.clone())
        .expect("compiler")
        .compile_batch_with(&context, &requests, Math::Fast);
    assert!(matches!(fast.archive, Archived::Written));

    let precise = Compiler::with_archives(&context, archives.clone())
        .expect("compiler")
        .compile_batch_with(&context, &requests, Math::Precise);
    assert!(
        matches!(precise.archive, Archived::Written),
        "the precise batch was served the fast batch's binaries: {:?}",
        precise.archive
    );

    // And each mode still hits its OWN archive -- a key that merely differed
    // every time would also pass the assertion above while caching nothing.
    let again = Compiler::with_archives(&context, archives)
        .expect("compiler")
        .compile_batch_with(&context, &requests, Math::Precise);
    assert!(
        again.archive.is_hit(),
        "the precise batch did not find the archive it had just written, so \
         the mode is being mixed into the key unstably: {:?}",
        again.archive
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn one_broken_kernel_fails_alone_and_writes_nothing() {
    let dir = scratch("partial");
    let context = Context::new().expect("context");
    let good = kernel(&dir, "good.metal", &source("good"));
    let bad = kernel(&dir, "bad.metal", "kernel void bad(");
    let requests = vec![
        Request::new(&good, "good"),
        Request::new(&bad, "bad"),
        Request::new(&good, "nosuch"),
    ];
    let cache = dir.join("cache");
    let compiled = Compiler::with_archives(&context, Archives::new(Some(cache.clone())))
        .expect("compiler")
        .compile_batch(&context, &requests);

    assert!(compiled.pipelines[0].is_ok(), "the good one still builds");
    assert!(compiled.pipelines[1].is_err(), "the unparseable one fails");
    assert!(
        compiled.pipelines[2].is_err(),
        "a missing entry point fails even though its library compiled"
    );
    assert!(
        matches!(compiled.archive, Archived::Skipped),
        "a partial batch must not leave an archive that reads as complete: {:?}",
        compiled.archive
    );
    let written = std::fs::read_dir(&cache)
        .map(|entries| entries.count())
        .unwrap_or(0);
    assert_eq!(written, 0, "nothing was written");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_file_names_itself_and_leaves_the_rest_alone() {
    let dir = scratch("missing");
    let context = Context::new().expect("context");
    let good = kernel(&dir, "good.metal", &source("good"));
    let compiled = Compiler::with_archives(&context, Archives::new(None))
        .expect("compiler")
        .compile_batch(
            &context,
            &[
                Request::new(&good, "good"),
                Request::new(dir.join("absent.metal"), "gone"),
            ],
        );
    assert!(compiled.pipelines[0].is_ok());
    let message = compiled.pipelines[1]
        .as_ref()
        .expect_err("absent")
        .to_string();
    assert!(
        message.contains("absent.metal"),
        "the failure has to name the file that is not there: {message}"
    );
    assert!(
        matches!(compiled.archive, Archived::Disabled),
        "no cache directory is not a failed write: {:?}",
        compiled.archive
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_empty_batch_builds_nothing_and_touches_no_disk() {
    let context = Context::new().expect("context");
    let compiled = Compiler::new(&context)
        .expect("compiler")
        .compile_batch(&context, &[]);
    assert!(compiled.pipelines.is_empty());
    assert!(matches!(compiled.archive, Archived::Skipped));
}
