//! **The warm-boot artifact, promoted** (alto streaming §0 and build-order
//! item 2) — the plane-group index, the mmap resolver, and the version
//! refusal.
//!
//! The stored weights need no load-time conversion, and that format already
//! existed in this tree: the warm-boot artifact is a snapshot of the DEVICE
//! STORE ITSELF, offsets identical to store offsets. So SSD streaming is not a
//! format design; it is the PROMOTION of that file from boot accelerator to
//! serving-time source. This gate asserts the three claims that promotion
//! rests on:
//!
//! ```text
//!  1. index   -> a plane group resolves to the store's own (offset, bytes),
//!                and the bytes at that offset are the bytes that were written
//!  2. no copy -> the resolution is a window on a mapping, not a read
//!  3. version -> an artifact from another build is REFUSED BY NAME on the
//!                serving door, not parsed under this build's rules
//! ```
//!
//! **No GPU, no model, no checkpoint.** Every artifact here is synthesized
//! from host bytes through [`weight_cache::seed`], which is the `probe` twin of
//! the device-side writer, so the whole file runs on a plain `cargo test`.
//!
//! ```bash
//! cargo test -p engine-cuda --test the_artifact_says_where_every_plane_lives
//! ```

use std::path::PathBuf;

use engine_cuda::weight_cache::{self, Artifact, Group, Refused};

/// A temporary directory that removes itself, however the test leaves.
///
/// Not a crate: this is four lines and the alternative is a dev-dependency
/// whose whole job is `Drop`. Borrowed verbatim from
/// `a_warm_boot_reads_the_weights_it_wrote`.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch(what: &str) -> Scratch {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    let dir = std::env::temp_dir().join(format!("pie-{what}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temporary directory");
    Scratch(dir)
}

/// **A store of four planes under three plane groups**, laid out the way
/// `weights.rs`' `places` lays one out: offsets ascending, every span aligned
/// up, group 2 split across two planes the way an mxfp4 bank is.
///
/// The blob's bytes are a function of the group they belong to, so a resolver
/// that returned the wrong span would return the wrong number and not merely a
/// short one.
fn synthetic() -> (Vec<Group>, Vec<u8>) {
    const ALIGN: u64 = 256;
    let shapes: [(u32, u32, u64); 4] = [
        // (id, plane, published bytes)
        (0, 0, 300),
        (1, 0, 512),
        (2, 0, 100),
        (2, 1, 100),
    ];
    let mut groups = Vec::new();
    let mut blob = Vec::new();
    let mut at = 0u64;
    for (id, plane, bytes) in shapes {
        let reserved = bytes.next_multiple_of(ALIGN);
        groups.push(Group {
            id,
            plane,
            offset: at,
            bytes,
            reserved,
        });
        // A byte that names its group, so a mis-resolved span is a wrong
        // value rather than a wrong length.
        let mark = u8::try_from(id * 16 + plane + 1).unwrap_or(0xff);
        blob.extend(std::iter::repeat_n(mark, usize::try_from(bytes).unwrap_or(0)));
        // The padding between planes is what the store gives and the
        // checkpoint does not publish. It is deliberately NOT the mark, so a
        // resolver that handed back `reserved` instead of `bytes` would be
        // caught.
        blob.extend(std::iter::repeat_n(
            0u8,
            usize::try_from(reserved - bytes).unwrap_or(0),
        ));
        at += reserved;
    }
    assert_eq!(blob.len() as u64, at, "the blob is exactly what it reserved");
    (groups, blob)
}

/// ── 1 & 2. THE INDEX RESOLVES, AND IT RESOLVES ONTO THE MAPPING.
#[test]
fn every_plane_group_resolves_to_the_bytes_that_were_written() {
    let dir = scratch("artifact-index");
    let path = dir.0.join("00000000deadbeef.weights");
    let (groups, blob) = synthetic();
    weight_cache::seed(&path, 0xdead_beef, &groups, &blob).expect("a synthetic artifact");

    let artifact = Artifact::open(&path).expect("the artifact this test just wrote");

    assert_eq!(artifact.key(), 0xdead_beef, "the key round-trips");
    assert_eq!(
        artifact.total(),
        blob.len() as u64,
        "the header states the blob it holds"
    );
    assert_eq!(
        artifact.groups().len(),
        groups.len(),
        "every plane group survived the round trip"
    );
    assert_eq!(artifact.groups(), &groups[..], "and each one, field for field");

    // **THE RESOLUTION ITSELF.** Each group answers with the store's own
    // three numbers, and the bytes behind them are that group's mark —
    // `bytes` of it, not `reserved` of it.
    for group in &groups {
        let found = artifact
            .resolve(group.id, group.plane)
            .expect("a group the index carries");
        assert_eq!(&found, group, "plane group {}/{}", group.id, group.plane);

        let seen = artifact
            .plane(group.id, group.plane)
            .expect("bytes for a group the index carries");
        assert_eq!(
            seen.len() as u64,
            group.bytes,
            "plane group {}/{} is handed its PUBLISHED bytes and not its reserved span",
            group.id,
            group.plane
        );
        let mark = u8::try_from(group.id * 16 + group.plane + 1).unwrap_or(0xff);
        assert!(
            seen.iter().all(|&byte| byte == mark),
            "plane group {}/{} resolved onto somebody else's bytes",
            group.id,
            group.plane
        );
    }

    // The split-plane bank is two entries under one id, which is the shape
    // `WeightRow::Planes` needs and the reason the index is keyed by a pair.
    assert_ne!(
        artifact.resolve(2, 0),
        artifact.resolve(2, 1),
        "one weight id, two device planes"
    );

    // A group nothing published is a `None`, not a guess.
    assert_eq!(artifact.resolve(9, 0), None, "an id the index does not carry");
    assert_eq!(artifact.plane(2, 7), None, "a plane the group does not have");

    // **NO COPY.** The slices point into the mapping, which is the whole
    // property the T2 pointer class is built on: two resolutions of the same
    // group are the same address.
    let once = artifact.plane(1, 0).expect("bytes").as_ptr();
    let twice = artifact.plane(1, 0).expect("bytes").as_ptr();
    assert_eq!(once, twice, "resolving twice did not copy twice");
    let blob_base = artifact.blob().as_ptr();
    // SAFETY: both pointers are into the same mapping — that is the claim.
    let offset = (once as usize) - (blob_base as usize);
    assert_eq!(
        offset as u64,
        groups[1].offset,
        "a plane's address is the blob's base plus the STORE's own offset"
    );

    // And the always-verified discipline is available on this side too.
    artifact.verify().expect("the digest the seed wrote");
}

/// ── 3. THE VERSION REFUSAL, BY NAME.
///
/// The magic and the format are separate fields on purpose: an artifact from
/// another build is RECOGNIZED and then refused for its version, rather than
/// mistaken for somebody else's file. The serving door refuses; the boot path
/// treats the same file as a miss and recomputes, which is
/// `a_warm_boot_reads_the_weights_it_wrote`'s business.
#[test]
fn an_artifact_from_another_build_is_refused_by_its_version() {
    let dir = scratch("artifact-version");
    let path = dir.0.join("0000000000000001.weights");
    let (groups, blob) = synthetic();
    weight_cache::seed(&path, 1, &groups, &blob).expect("a synthetic artifact");

    // It opens today.
    Artifact::open(&path).expect("the current format");

    // Now it is a file this build does not read. Only the four version bytes
    // move, so everything else about it stays true — which is what makes the
    // refusal a refusal about the VERSION and not about a corruption.
    weight_cache::restate_format(&path, 1).expect("restating the format");

    let refused = Artifact::open(&path).expect_err("a stale format cannot be served from");
    assert!(
        matches!(refused, Refused::StaleFormat { states: 1, .. }),
        "a stale artifact is refused by name, not as a parse failure: {refused:?}"
    );
    let said = refused.to_string();
    assert!(
        said.contains("states format 1") && said.contains("regenerate"),
        "the refusal names both versions and what to do: {said}"
    );

    // A version from the FUTURE is the same refusal — this reader does not
    // guess at fields it has never seen.
    weight_cache::restate_format(&path, u32::MAX).expect("restating the format");
    assert!(
        matches!(
            Artifact::open(&path),
            Err(Refused::StaleFormat {
                states: u32::MAX,
                ..
            })
        ),
        "a format from ahead of this build is refused the same way"
    );
}

/// Everything that is not an artifact says so under its own name, so that a
/// caller can tell "there is no cache here" from "the cache is broken".
#[test]
fn what_is_not_an_artifact_is_refused_under_its_own_name() {
    let dir = scratch("artifact-refusals");

    let missing = dir.0.join("nothing.weights");
    assert!(
        matches!(Artifact::open(&missing), Err(Refused::Unreadable { .. })),
        "a path with no file behind it"
    );

    let junk = dir.0.join("junk.weights");
    std::fs::write(&junk, b"this is not a weight artifact at all, not even close")
        .expect("a junk file");
    assert!(
        matches!(Artifact::open(&junk), Err(Refused::NotAnArtifact)),
        "a file whose first eight bytes are not the magic"
    );

    let tiny = dir.0.join("tiny.weights");
    std::fs::write(&tiny, b"PIEWCAC1").expect("a file with the magic and nothing else");
    assert!(
        matches!(Artifact::open(&tiny), Err(Refused::NotAnArtifact)),
        "a file too short to hold a header has not made a claim to be short of"
    );

    // A real artifact, cut off inside its blob.
    let cut = dir.0.join("cut.weights");
    let (groups, blob) = synthetic();
    weight_cache::seed(&cut, 7, &groups, &blob).expect("a synthetic artifact");
    let whole = std::fs::metadata(&cut).expect("its size").len();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&cut)
        .expect("reopening it");
    file.set_len(whole - 64).expect("truncating it");
    drop(file);
    assert!(
        matches!(Artifact::open(&cut), Err(Refused::Truncated { .. })),
        "a file shorter than its own header claims"
    );
}

/// **AN ARTIFACT WITH NO INDEX IS STILL AN ARTIFACT.**
///
/// The loader forms the key from its layout and has never had to hand that
/// layout any further, so today's cold load writes zero groups. The format
/// carries the index either way; a reader gets an honest empty answer rather
/// than a refusal, which is what lets the promotion land before the wiring
/// does.
#[test]
fn an_artifact_with_no_index_carries_its_blob_and_says_so() {
    let dir = scratch("artifact-groupless");
    let path = dir.0.join("0000000000000002.weights");
    let blob = vec![0xabu8; 4096];
    weight_cache::seed(&path, 2, &[], &blob).expect("a synthetic artifact");

    let artifact = Artifact::open(&path).expect("an indexless artifact still opens");
    assert!(artifact.groups().is_empty(), "no groups were declared");
    assert_eq!(artifact.resolve(0, 0), None, "and none resolve");
    assert_eq!(artifact.blob(), &blob[..], "the blob is whole");
    artifact.verify().expect("and verified");
}

/// **THE W-4 MEASUREMENT'S SWITCH IS A FUNCTION, NOT AN ENVIRONMENT VARIABLE.**
///
/// Article 9: shells read no environment. The gate that measures the staged
/// pump against the blocking loop it replaced needs both arms in one process,
/// so the selection is a `probe` hook — dropped by a serving binary with
/// `default-features = false`, exactly like every other gate-only door in this
/// crate. Serving always takes the pump.
#[test]
fn the_restore_path_is_selected_in_code_and_defaults_to_the_pump() {
    assert!(
        weight_cache::restore_is_pumped(),
        "a process that has said nothing restores through the pump"
    );
    weight_cache::restore_through_the_pump(false);
    assert!(
        !weight_cache::restore_is_pumped(),
        "the blocking arm is selectable for the measurement"
    );
    weight_cache::restore_through_the_pump(true);
    assert!(weight_cache::restore_is_pumped(), "and selectable back");
}
