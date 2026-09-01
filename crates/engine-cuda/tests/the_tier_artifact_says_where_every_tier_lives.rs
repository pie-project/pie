//! **The serving artifact** (§K phase T-1, rewritten for §M) — one image per
//! plane, one index, per-image block digests, and a key with no budget in it.
//!
//! A streamed load materializes three images: what the device store holds,
//! what the pinned tier holds, and what T2 serves out of. Formats 1 and 2
//! carried exactly that — three sections — and paid for it by making the file
//! a function of the BUDGETS: a different `device_weight_budget` was a
//! different key, a different file, and a hundred gigabytes rewritten to say
//! the same thing about the same weights.
//!
//! Format 3 carries the model instead. A plane's bytes are identical on all
//! three rungs (§M.3's measured fact — `weights::ALIGN` is what makes the same
//! address arithmetic valid out of the store, out of page-locked memory and
//! out of a mapping), so the file holds each image ONCE, in a budget-free
//! priority order, and a boot cuts it with the budget it has. This gate
//! asserts that format, before any load path reads or writes one:
//!
//! ```text
//!  1. one image  -> an entry resolves to the bytes that were written, and to
//!                   its PUBLISHED length, never its padded span
//!  2. aligned    -> the payload starts on TIER_ALIGN, whatever the index and
//!                   the block table cost
//!  3. tiled      -> the images are a CONCATENATION; a gap, an overlap or an
//!                   image that overruns its span is named with that image
//!  4. refused    -> truncated, stale and misnamed files are refused by name,
//!                   the stale one carrying both format numbers (§K.5)
//!  5. keyed      -> every part of the identity moves the key — and there is
//!                   no budget and no rung in the identity at all, which is
//!                   what ONE FILE, ANY BUDGET rests on
//!  6. checked    -> one flipped byte is a named block of a named image, it
//!                   moves the corruption counter, and EVERY OTHER IMAGE
//!                   STILL VERIFIES — the property a cut needs
//!  7. blocked    -> the blocks of an image tile it exactly, at TIER_BLOCK,
//!                   so any contiguous prefix of the index is a prefix of the
//!                   table
//!  8. rewritten  -> a format-2 file is refused with both numbers, LEFT ON
//!                   THE DISK (§M.4), and replaced only by the one writer
//!                   there is — `tier::store`, which since §M-3 only a
//!                   `pie model import` ever reaches
//! ```
//!
//! **No GPU, no model, no checkpoint.** Every artifact here is synthesized
//! from host bytes through [`tier::seed`], the `probe` twin of the device-side
//! writer, so the whole file runs on a plain `cargo test`.
//!
//! ```bash
//! cargo test -p engine-cuda --test the_tier_artifact_says_where_every_tier_lives
//! ```

use std::path::{Path, PathBuf};

use engine_cuda::weight_cache::{self, Group, Refused, tier};
use tier::Identity;

/// A temporary directory that removes itself, however the test leaves.
///
/// Not a crate: this is four lines and the alternative is a dev-dependency
/// whose whole job is `Drop`. Borrowed verbatim from
/// `the_artifact_says_where_every_plane_lives`.
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

/// An index and the payload it describes, owned, so that the borrowed slices a
/// writer takes can be handed out of one place.
struct Images {
    entries: Vec<Group>,
    payload: Vec<u8>,
    flags: u32,
}

impl Images {
    fn entry(&self, param: u32) -> Group {
        *self
            .entries
            .iter()
            .find(|group| group.id == param)
            .expect("an image this test wrote")
    }
}

/// The byte one image is filled with, so that a mis-resolved span is a WRONG
/// value and not merely a short one.
fn mark(param: u32) -> u8 {
    u8::try_from(param * 16 + 1).unwrap_or(0xff)
}

/// **A payload laid out the way a ranking lays one out**: one entry per plane,
/// offsets consecutive from zero, every span the published bytes rounded up to
/// the store's own 256 — a split-plane bank's two planes as two entries under
/// two param ids, because a plane is a `Trace::params` row and this index is
/// keyed by that row.
///
/// The seven images are the shape a cut walks: the first few are what a device
/// budget takes, the middle is what a host budget takes, the tail is what
/// neither does — and **nothing in the file says which is which**, which is
/// the whole of §M.3.
fn synthetic() -> Images {
    const ALIGN: u64 = 256;
    // (param, published bytes)
    let shapes: [(u32, u64); 7] = [
        (0, 300),
        (1, 512),
        (2, 100),
        (3, 100),
        (4, 700),
        (5, 64),
        (6, 128),
    ];
    let mut entries = Vec::new();
    let mut payload = Vec::new();
    let mut at = 0u64;
    for (param, published) in shapes {
        let reserved = published.next_multiple_of(ALIGN);
        entries.push(Group {
            id: param,
            plane: 0,
            offset: at,
            bytes: published,
            reserved,
        });
        payload.extend(std::iter::repeat_n(
            mark(param),
            usize::try_from(published).unwrap_or(0),
        ));
        // The padding a span gives and a checkpoint does not publish. It is
        // deliberately NOT the mark, so a resolver that handed back `reserved`
        // instead of `bytes` would be caught.
        payload.extend(std::iter::repeat_n(
            0u8,
            usize::try_from(reserved - published).unwrap_or(0),
        ));
        at += reserved;
    }
    assert_eq!(payload.len() as u64, at, "the payload is exactly what it reserved");
    Images {
        entries,
        payload,
        flags: tier::FLAG_ADAPTERS_ZEROED,
    }
}

/// Seed one at the path this key would be written under.
fn seeded(dir: &Path, key: u64, images: &Images) -> PathBuf {
    let path = tier::path(dir, key);
    tier::seed(&path, key, &images.entries, images.flags, &images.payload)
        .expect("a synthetic serving artifact");
    path
}

/// ── 1. EVERY IMAGE ROUND-TRIPS, AND EACH RESOLVES INTO ITS OWN BYTES.
#[test]
fn every_image_resolves_to_the_bytes_that_were_written() {
    let dir = scratch("tier-images");
    let images = synthetic();
    let path = seeded(&dir.0, 0xdead_beef, &images);

    let artifact = tier::Artifact::open(&path).expect("the artifact this test just wrote");
    let head = artifact.head();

    assert_eq!(artifact.key(), 0xdead_beef, "the key round-trips");
    assert_eq!(head.format, tier::TIER_FORMAT, "and so does the version");
    assert_eq!(head.format, 3, "which is 3, where §L left it at 2");
    assert_eq!(
        head.flags,
        tier::FLAG_ADAPTERS_ZEROED,
        "and what the file states about its own contents"
    );
    assert_eq!(head.index_at, 96, "the index follows the header");
    assert_eq!(head.block_bytes, tier::TIER_BLOCK, "the file states its own block size");
    assert_eq!(
        head.entries as usize,
        images.entries.len(),
        "and how many images it holds"
    );
    assert_eq!(
        artifact.entries(),
        &images.entries[..],
        "the index came back field for field, IN ORDER — the order IS the ranking"
    );
    assert_eq!(
        head.payload_total,
        images.payload.len() as u64,
        "the header states the payload it holds"
    );

    // The file, read as bytes, so that "where the payload is" is checked
    // against the file and not only against the mapping that reports it.
    let raw = std::fs::read(&path).expect("the file itself");
    let at = usize::try_from(head.payload_at).expect("an offset");
    assert_eq!(
        &raw[at..at + images.payload.len()],
        &images.payload[..],
        "the payload's byte range is where its bytes actually are"
    );
    assert_eq!(artifact.payload(), &images.payload[..], "and the mapping agrees");

    // **THE RESOLUTION ITSELF.** Each image answers with its own three
    // numbers, and the bytes behind them are that image's mark — `bytes` of
    // it, not `reserved` of it.
    for group in &images.entries {
        let found = artifact.resolve(group.id).expect("an image the index carries");
        assert_eq!(&found, group, "image for param {}", group.id);

        let seen = artifact.plane(group.id).expect("bytes for an image it carries");
        assert_eq!(
            seen.len() as u64,
            group.bytes,
            "param {}'s image is handed its PUBLISHED bytes",
            group.id
        );
        let wanted = mark(group.id);
        assert!(
            seen.iter().all(|&byte| byte == wanted),
            "param {} resolved onto somebody else's bytes",
            group.id
        );

        // And the block table answers for it: an image's digests are a
        // contiguous run, which is what lets a cut check any subset.
        let (again, first) = artifact.locate(group.id).expect("its blocks");
        assert_eq!(again, found, "`locate` and `resolve` are one answer");
        assert!(
            (first + head.blocks_of(group.reserved)) <= head.blocks,
            "param {}'s blocks are inside the table",
            group.id
        );
    }

    assert_eq!(artifact.resolve(99), None, "a param nobody carries");
    assert_eq!(artifact.plane(99), None, "and it has no bytes either");

    // **NO COPY.** The slices point into the mapping, which is the property
    // the T2 pointer class is built on: two resolutions are one address.
    let once = artifact.plane(1).expect("bytes").as_ptr();
    let twice = artifact.plane(1).expect("bytes").as_ptr();
    assert_eq!(once, twice, "resolving twice did not copy twice");
    let base = artifact.payload().as_ptr();
    assert_eq!(
        (once as usize - base as usize) as u64,
        images.entry(1).offset,
        "an image's address is the payload's base plus that image's own offset"
    );

    // And the always-verified discipline, over the whole table and its fold.
    artifact.verify().expect("the digests the seed wrote");
}

/// ── 2. THE PAYLOAD STARTS ON TIER_ALIGN, WHATEVER THE INDEX COSTS.
///
/// The alignment is not decoration: images are pumped straight to the device
/// and read straight into page-locked memory the allocator hands out on
/// huge-page boundaries, and an index long enough to push past a boundary must
/// move the payload rather than the payload's alignment. 70,000 entries is
/// 2.24 MiB of index — past the first boundary — which is the case a smaller
/// model would never reach and a large MoE would.
#[test]
fn the_payload_starts_on_a_two_mebibyte_boundary() {
    let dir = scratch("tier-align");

    let empty = Images {
        entries: Vec::new(),
        payload: Vec::new(),
        flags: 0,
    };
    // A long index over an empty payload: what moves is where the payload
    // starts. Every entry reserves nothing, so they all sit at offset zero and
    // still tile it.
    let long = Images {
        entries: (0..70_000u32)
            .map(|id| Group {
                id,
                plane: 0,
                offset: 0,
                bytes: 0,
                reserved: 0,
            })
            .collect(),
        payload: Vec::new(),
        flags: 0,
    };
    let cases = [(&empty, 0u64), (&synthetic(), 1), (&long, 2)];

    for (images, key) in cases {
        let path = seeded(&dir.0, key, images);
        let head = tier::Artifact::open(&path).expect("it opens").head();
        assert_eq!(
            head.payload_at % tier::TIER_ALIGN,
            0,
            "key {key}: the payload starts at {}",
            head.payload_at
        );
        let floor = 96 + (images.entries.len() * 32) as u64 + head.blocks * 8;
        assert!(
            head.payload_at >= floor,
            "key {key}: the payload at {} is inside the index and table before it \
             ({floor})",
            head.payload_at
        );
    }

    // And the long index really did push the payload past the first boundary,
    // so the case is exercised rather than merely stated.
    let head = tier::Artifact::open(&tier::path(&dir.0, 2)).expect("it opens").head();
    assert!(
        head.payload_at > tier::TIER_ALIGN,
        "70,000 entries did not outgrow one boundary; the case is no longer covered"
    );
}

/// ── 3. THE IMAGES ARE A CONCATENATION, AND A BREAK IS NAMED WITH ITS IMAGE.
///
/// This replaces format 2's "an entry past the end of its OWN section", and
/// the claim got stronger rather than weaker. There is one payload now, so
/// "inside the file" is not enough: an index with a GAP describes bytes no
/// digest covers, an index with an OVERLAP describes one image twice, and
/// either would be cut onto a rung by a boot that believed it. So the reader
/// checks the whole tiling, not just the last byte.
#[test]
fn an_index_that_does_not_tile_its_payload_is_named_with_its_image() {
    let dir = scratch("tier-tiling");
    let images = synthetic();

    // ── A GAP: image 3 starts one span too far along.
    let path = seeded(&dir.0, 3, &images);
    tier::Artifact::open(&path).expect("it opens before the entry is restated");
    let third = images.entries[3];
    tier::restate_group(
        &path,
        3,
        Group {
            offset: third.offset + 256,
            ..third
        },
    )
    .expect("restating one entry");
    let refused = tier::Artifact::open(&path).expect_err("a hole in the payload");
    let said = refused.to_string();
    assert!(
        matches!(refused, Refused::IndexCorrupt { .. }),
        "an index that does not tile is an index fault: {refused:?}"
    );
    assert!(said.contains("param 3"), "and it names the image: {said}");
    assert!(said.contains("concatenation"), "and what it broke: {said}");

    // ── MORE PUBLISHED THAN RESERVED: an image that overruns its own span
    //    would hand a reader a window over the next image's bytes.
    let over = seeded(&dir.0, 0x3d, &images);
    tier::restate_group(
        &over,
        2,
        Group {
            bytes: images.entries[2].reserved + 1,
            ..images.entries[2]
        },
    )
    .expect("restating one entry");
    let said = tier::Artifact::open(&over)
        .expect_err("an image longer than its span")
        .to_string();
    assert!(said.contains("param 2"), "the image is named: {said}");

    // ── AND A SPAN THAT DOES NOT ADD UP TO THE PAYLOAD. The last image is
    //    shortened, so every offset still agrees and the total does not.
    let short = seeded(&dir.0, 0x3e, &images);
    let last = images.entries[6];
    tier::restate_group(&short, 6, Group { reserved: 128, ..last })
        .expect("restating one entry");
    let said = tier::Artifact::open(&short)
        .expect_err("an index that tiles less than the payload")
        .to_string();
    assert!(
        said.contains(&format!("{}", images.payload.len())),
        "the payload's own length is named: {said}"
    );
}

/// ── 4a. A FILE SHORTER THAN ITS OWN HEADER CLAIMS.
#[test]
fn a_truncated_tier_artifact_is_refused_by_name() {
    let dir = scratch("tier-truncated");
    let images = synthetic();
    let path = seeded(&dir.0, 4, &images);
    let whole = std::fs::metadata(&path).expect("its size").len();

    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("reopening it");
    file.set_len(whole - 64).expect("truncating it");
    drop(file);

    let refused = tier::Artifact::open(&path).expect_err("a file cut off inside its payload");
    assert!(
        matches!(refused, Refused::Truncated { .. }),
        "a short file is refused as truncated: {refused:?}"
    );
    let Refused::Truncated { states, holds } = refused else {
        unreachable!("matched above")
    };
    assert_eq!(states, whole, "the refusal names what the header accounts for");
    assert_eq!(holds, whole - 64, "and what the file actually holds");

    // And what is not a serving artifact at all says so under its own name.
    let junk = dir.0.join("junk.tiers");
    std::fs::write(&junk, b"this is not a serving artifact at all, not even close")
        .expect("junk");
    assert!(
        matches!(tier::Artifact::open(&junk), Err(Refused::NotAnArtifact)),
        "a file whose first eight bytes are not the magic"
    );
    let missing = dir.0.join("nothing.tiers");
    assert!(
        matches!(tier::Artifact::open(&missing), Err(Refused::Unreadable { .. })),
        "a path with no file behind it"
    );
}

/// ── 4b. A FILE FROM ANOTHER BUILD IS REFUSED WITH BOTH NUMBERS (§K.5).
///
/// The magic and the format are separate fields on purpose: a file from
/// another build is RECOGNIZED and then refused for its version, rather than
/// mistaken for somebody else's. The boot path turns the same refusal into a
/// MISS said out loud with `pie model import` beside it; the serving door
/// cannot, having no recipe to fall back to.
#[test]
fn a_tier_artifact_from_another_build_is_refused_with_both_formats() {
    let dir = scratch("tier-version");
    let images = synthetic();
    let path = seeded(&dir.0, 5, &images);
    tier::Artifact::open(&path).expect("the current format");

    // The two formats share a field offset, so the resident cache's gate-only
    // rewriter is the one that states this one's version too — which is itself
    // the claim that the two headers begin the same way.
    weight_cache::restate_format(&path, 0).expect("restating the format");

    let refused = tier::Artifact::open(&path).expect_err("a stale format cannot be served from");
    assert!(
        matches!(refused, Refused::StaleFormat { states: 0, reads: 3 }),
        "a format-0 file is refused by name, carrying both numbers: {refused:?}"
    );
    let said = refused.to_string();
    assert!(
        said.contains("states format 0") && said.contains("reads 3"),
        "the refusal says both out loud: {said}"
    );

    // **AND THE ONES THAT ARE ACTUALLY ON THE DISKS** (§M): formats 1 and 2,
    // which every seat booted before the sections went away left behind.
    // `states` is the FILE's number and `reads` is this BUILD's, in that
    // order.
    for stale in [1u32, 2] {
        weight_cache::restate_format(&path, stale).expect("restating the format");
        assert!(
            matches!(
                tier::Artifact::open(&path),
                Err(Refused::StaleFormat { states, reads: 3 }) if states == stale
            ),
            "a format-{stale} file is the §M transition's own case"
        );
        assert!(
            path.exists(),
            "and the door that refused it did not delete it (§M.4)"
        );
    }

    // A version from the FUTURE is the same refusal — this reader does not
    // guess at fields it has never seen.
    weight_cache::restate_format(&path, u32::MAX).expect("restating the format");
    assert!(
        matches!(
            tier::Artifact::open(&path),
            Err(Refused::StaleFormat { states: u32::MAX, .. })
        ),
        "a format from ahead of this build is refused the same way"
    );
}

/// ── 4c. A FILE WHOSE NAME AND HEADER DISAGREE ABOUT THE KEY.
///
/// The filename carries the key, so the key is stated twice about every file.
/// A file under one key's name holding another key's images is the case that
/// would serve the wrong deployment's weights with every digest agreeing,
/// which is why it is a refusal and not a preference for one of the two.
#[test]
fn a_tier_artifact_whose_name_and_header_disagree_is_refused() {
    let dir = scratch("tier-misnamed");
    let images = synthetic();

    // Written under one key's name, stating another's.
    let named = tier::path(&dir.0, 0x0000_0000_0000_0aaa);
    tier::seed(
        &named,
        0x0000_0000_0000_0bbb,
        &images.entries,
        images.flags,
        &images.payload,
    )
    .expect("a misnamed artifact");

    let refused = tier::Artifact::open(&named).expect_err("a name that lies");
    assert!(
        matches!(
            refused,
            Refused::WrongKey {
                states: 0xbbb,
                names: 0xaaa
            }
        ),
        "the disagreement is refused by name: {refused:?}"
    );
    let said = refused.to_string();
    assert!(
        said.contains("0000000000000aaa") && said.contains("0000000000000bbb"),
        "and both keys are in it: {said}"
    );

    // A path this module would never have written carries no claim, so there
    // is nothing to disagree with: a caller may name its own file.
    let anonymous = dir.0.join("mine.tiers");
    tier::seed(&anonymous, 0xbbb, &images.entries, images.flags, &images.payload)
        .expect("an unnamed artifact");
    tier::Artifact::open(&anonymous).expect("a file whose name makes no claim");
}

/// ── 5. EVERY PART OF THE IDENTITY MOVES THE KEY — AND THERE IS NO BUDGET IN
/// IT.
///
/// A false miss costs one re-import; a false hit puts silently wrong weights
/// on the device and in the pinned tier. So each field is checked to MOVE the
/// key, which is the only direction that matters.
///
/// **And the fields that are GONE are the point of the wave.** Format 2's
/// identity carried `device_layout`, `host_layout` and every param's rung, and
/// this gate used to assert that moving one rung moved the key — that a
/// changed budget was a different file. §M.3 says that was the bug: the three
/// rungs hold the same bytes, so the split belongs to the boot and not to the
/// file. What replaces the three is `images`, which is what the payload
/// physically holds and is a pure function of the trace and the recipe. The
/// budget-dependence assertion is therefore not weakened here — it is
/// INVERTED, and its positive form is proved where it can be: two boots at two
/// different budgets, one artifact, in
/// `a_second_streamed_boot_maps_the_tiers_it_wrote`.
#[test]
fn every_part_of_the_tier_identity_moves_the_key() {
    let layout = [(0u64, 100u64, 256u64), (256, 40, 256)];
    // (param, published bytes, span) — the payload's own contents, in order.
    let images = [(7u64, 300u64, 512u64), (8, 100, 256), (9, 64, 256)];
    let base = Identity {
        checkpoint: Path::new("/models/qwen"),
        trace_name: "qwen35-d0.8b",
        plan_json: b"{\"steps\":[]}",
        total: 512,
        layout: &layout,
        images: &images,
    };
    let key = base.key();

    let other_layout = [(0u64, 100u64, 256u64), (256, 44, 256)];
    // ONE SPAN MOVED: the same planes, one of them four bytes longer.
    let wider = [(7u64, 304u64, 512u64), (8, 100, 256), (9, 64, 256)];
    // THE SAME SPANS, REORDERED: the ranking IS the file, so two orders are
    // two files even when they hold the same bytes — a boot that cut one with
    // the other's index would route every image to the wrong rung.
    let reordered = [(8u64, 100u64, 256u64), (7, 300, 512), (9, 64, 256)];
    // ONE IMAGE FEWER.
    let shorter = [(7u64, 300u64, 512u64), (8, 100, 256)];

    for (what, moved) in [
        (
            "the checkpoint",
            Identity {
                checkpoint: Path::new("/models/other"),
                ..base
            },
        ),
        (
            "the trace name",
            Identity {
                trace_name: "qwen35-d1.7b",
                ..base
            },
        ),
        (
            "the plan",
            Identity {
                plan_json: b"{\"steps\":[1]}",
                ..base
            },
        ),
        ("the total", Identity { total: 768, ..base }),
        (
            "the resident layout",
            Identity {
                layout: &other_layout,
                ..base
            },
        ),
        (
            "one image's span",
            Identity {
                images: &wider,
                ..base
            },
        ),
        (
            "the order of the images",
            Identity {
                images: &reordered,
                ..base
            },
        ),
        (
            "how many images there are",
            Identity {
                images: &shorter,
                ..base
            },
        ),
    ] {
        assert_ne!(key, moved.key(), "{what} changed and the key did not");
    }

    // And the same deployment is the same key, which is the whole point.
    assert_eq!(key, base.key());

    // **THE TWO CACHES CANNOT SHARE A FILE.** The resident identity over the
    // fields the two have in common forms a different number — this key mixes
    // TIER_FORMAT and the image list — so the same deployment never names one
    // cache's artifact with the other's key.
    let resident = weight_cache::Identity {
        checkpoint: base.checkpoint,
        trace_name: base.trace_name,
        plan_json: base.plan_json,
        layout: &layout,
        total: base.total,
    };
    assert_ne!(key, resident.key(), "two formats, two keys");
}

/// ── 6. ONE FLIPPED BYTE IS A NAMED BLOCK OF A NAMED IMAGE, IT IS COUNTED,
/// AND EVERY OTHER IMAGE STILL VERIFIES.
///
/// §K.5: *a silently-corrupt weight artifact produces garbage tokens with no
/// error*, which is not a trade an operator should be offered for a few
/// seconds of load time. There is no `verify = false`.
///
/// The last clause is §M's addition and it is what the cut rests on. A boot
/// verifies the images ITS budget puts on each rung — the pinned ones as they
/// are read, the mapped ones up front — and a granularity coarser than the
/// image would have made "check what you serve" mean "check the whole file".
#[test]
fn a_single_flipped_byte_names_its_block_and_counts_the_corruption() {
    use std::os::unix::fs::FileExt;

    let dir = scratch("tier-corrupt");
    let images = synthetic();
    let path = seeded(&dir.0, 6, &images);

    let head = tier::Artifact::open(&path).expect("it opens").head();
    tier::Artifact::open(&path)
        .expect("it opens")
        .verify()
        .expect("and verifies before anything rots");

    // One byte of param 4's image, which is neither the first nor the last: a
    // reader that hashed the file instead of the image would still notice, and
    // one that hashed the wrong image would not.
    let at = head.payload_at + images.entry(4).offset + 17;
    let mut byte = [0u8; 1];
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .expect("reopening it");
    file.read_exact_at(&mut byte, at).expect("the byte");
    file.write_at(&[byte[0] ^ 0x01], at).expect("flipping it");
    file.sync_all().expect("and landing it");
    drop(file);

    let before = tier::observed();
    let artifact = tier::Artifact::open(&path).expect("the index is untouched, so it still opens");

    // **EVERY OTHER IMAGE IS WHOLE**, one at a time and all together — which
    // is what lets a boot check the prefix it pumps without hashing the tail
    // it maps.
    let others: Vec<u32> = images
        .entries
        .iter()
        .map(|group| group.id)
        .filter(|id| *id != 4)
        .collect();
    artifact
        .verify_entries(&others)
        .expect("the other six images did not move");
    for id in &others {
        artifact
            .verify_entries(&[*id])
            .unwrap_or_else(|why| panic!("param {id} on its own: {why}"));
    }

    let refused = artifact
        .verify_entries(&[4])
        .expect_err("a flipped byte in param 4's image");
    let said = refused.to_string();
    assert!(
        matches!(refused, Refused::IndexCorrupt { .. }),
        "a digest that does not match its bytes: {refused:?}"
    );
    assert!(said.contains("param 4"), "the image is named: {said}");
    assert!(said.contains("block "), "and the block: {said}");
    assert!(
        said.contains("payload bytes "),
        "and the block's byte range within the payload: {said}"
    );

    let after = tier::observed();
    assert!(
        after.corrupt > before.corrupt,
        "a corruption is counted where a gate can see it: {before:?} -> {after:?}"
    );

    // `verify` over the whole payload stops at the same place.
    assert!(artifact.verify().is_err(), "the whole file fails on that image");
}

/// **THE REGISTER MOVES WHERE A TEST CAN REACH IT.**
///
/// Its own counters (`experts::observed`'s precedent) and not a widening of
/// [`weight_cache::observed`]: the two files answer different questions and a
/// boot can hit one and miss the other.
///
/// Three of the four are reachable without a device. The fourth — a write
/// declined for want of DISK SPACE — is not: [`tier::write_out`] asks
/// `statvfs` for the real free space of the real filesystem, and a test cannot
/// shape one that reports under 256 MiB free without a loop device and root.
/// What it can shape is the other half of the same counter, which is what the
/// counter's own documentation says it covers: a write that failed outright.
#[test]
fn the_tier_register_counts_what_it_says_it_counts() {
    let dir = scratch("tier-register");
    let images = synthetic();
    let fill = |param: u32, at: u64, into: &mut [u8]| {
        let group = images.entry(param);
        let from = usize::try_from(group.offset + at).expect("an offset");
        into.copy_from_slice(&images.payload[from..from + into.len()]);
        Ok(())
    };

    // ── stored: the counted wrapper writes through `write_out`, temp file,
    //    rename and all, and the result is a file the reader opens.
    let before = tier::observed();
    tier::store(Some(&dir.0), 0x51, &images.entries, images.flags, fill);
    let after = tier::observed();
    assert!(after.stored > before.stored, "a written artifact is counted");
    let written = tier::path(&dir.0, 0x51);
    let artifact = tier::Artifact::open(&written).expect("what `store` published opens");
    artifact.verify().expect("and verifies");
    assert_eq!(artifact.key(), 0x51, "under the key it was asked for");
    assert!(
        std::fs::read_dir(&dir.0)
            .expect("the directory")
            .flatten()
            .all(|entry| !entry.file_name().to_string_lossy().ends_with(".part")),
        "the temp file was renamed, not left behind"
    );

    // ── skipped: the same key, asked for twice. The key is a function of
    //    everything the payload is a function of, so a readable file under it
    //    already holds what a second write would produce — and since §M that
    //    is the ORDINARY case, because `pie model import` wrote it and every
    //    boot after that finds it.
    let before = tier::observed();
    tier::store(Some(&dir.0), 0x51, &images.entries, images.flags, fill);
    let after = tier::observed();
    assert!(
        after.skipped > before.skipped,
        "a key already on the disk is not written a second time: {before:?} -> {after:?}"
    );
    assert_eq!(after.stored, before.stored, "and the skip is not a write");
    assert_eq!(after.declined, before.declined, "nor a decline");

    // ── declined: a directory that cannot exist, because a file is in the
    //    way. The load is not broken by it — `store` answers nothing at all.
    let blocked = dir.0.join("blocked");
    std::fs::write(&blocked, b"in the way").expect("a file where a directory would go");
    let before = tier::observed();
    tier::store(
        Some(&blocked.join("under")),
        0x52,
        &images.entries,
        images.flags,
        fill,
    );
    let after = tier::observed();
    assert!(after.declined > before.declined, "a failed write is counted");
    assert!(after.stored == before.stored, "and is not also counted as stored");

    // ── nothing at all: a deployment that named no directory did not decline
    //    a file it was never offered.
    let before = tier::observed();
    tier::store(None, 0x53, &images.entries, images.flags, fill);
    assert_eq!(tier::observed(), before, "no directory, no write, no counter");

    // ── restored: the boot's door, moved by a path with a device on it, so
    //    the counter it moves is checked here rather than left to a gate that
    //    needs one.
    let before = tier::observed();
    tier::count_restored();
    assert!(
        tier::observed().restored > before.restored,
        "the restore counter is reachable"
    );
}

/// A write refuses an index it cannot honour, rather than publishing a file
/// whose reader will refuse it — the writer and the reader check the same
/// claim, in that order.
#[test]
fn a_write_refuses_an_index_that_does_not_tile_its_payload() {
    let dir = scratch("tier-writefault");
    let images = synthetic();
    let mut entries = images.entries.clone();
    // Image 5 starts a span too far along: a hole nothing would ever fill.
    entries[5].offset += 256;

    let why = tier::write_out(&dir.0, 0x61, &entries, images.flags, |_, _, into| {
        into.fill(0);
        Ok(())
    })
    .expect_err("an index that does not tile");
    assert!(why.contains("param 5"), "the image is named: {why}");
    assert!(!tier::path(&dir.0, 0x61).exists(), "and nothing was published");
}

/// The block a flipped byte at `at` lands in, and the byte range the refusal
/// states for it — read out of the refusal itself, because the refusal is the
/// only place the format publishes a block's span.
///
/// The file is left exactly as it was found: the byte is flipped, the image is
/// verified, and the byte is flipped back.
fn block_at(path: &Path, param: u32, at: u64) -> (u64, u64, u64) {
    use std::os::unix::fs::FileExt;

    let artifact = tier::Artifact::open(path).expect("it opens");
    let head = artifact.head();
    let group = artifact.resolve(param).expect("an image it carries");
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("reopening it");
    let byte_at = head.payload_at + group.offset + at;
    let mut byte = [0u8; 1];
    file.read_exact_at(&mut byte, byte_at).expect("the byte");
    file.write_at(&[byte[0] ^ 0x01], byte_at).expect("flipping it");
    file.sync_all().expect("and landing it");

    let said = tier::Artifact::open(path)
        .expect("the index is untouched")
        .verify_entries(&[param])
        .expect_err("a flipped byte")
        .to_string();

    file.write_at(&byte, byte_at).expect("flipping it back");
    file.sync_all().expect("and landing that");
    tier::Artifact::open(path)
        .expect("it opens")
        .verify_entries(&[param])
        .expect("the file is as it was found");

    let rest = said
        .split("for block ")
        .nth(1)
        .unwrap_or_else(|| panic!("the refusal names no block: {said}"));
    let (block, rest) = rest.split_once(' ').expect("a block number");
    let span = rest
        .split_once("(payload bytes ")
        .expect("a byte range")
        .1
        .split_once(')')
        .expect("a closed byte range")
        .0;
    let (from, to) = span.split_once("..").expect("a range");
    (
        block.parse().expect("a block number"),
        from.parse::<u64>().expect("a first byte") - group.offset,
        to.parse::<u64>().expect("a last byte") - group.offset,
    )
}

/// ── 7. THE BLOCKS OF AN IMAGE TILE IT EXACTLY, AT TIER_BLOCK.
///
/// The granularity argument, whole. §L.3 measured the verify that hides under
/// a warm boot's device pump and picked eight concurrent chains; format 2 spent
/// a HEADER FIELD on that number, which meant a re-measurement invalidated
/// every artifact on the disk. §M's blocks are a property of the FORMAT and
/// the readers are a property of the machine, so [`tier::TIER_READERS`] can
/// move on a new measurement and no file cares.
///
/// What the format owes instead is the tiling: an image's blocks are
/// contiguous, cover it exactly, and the last one carries the remainder — and
/// since the images tile the payload, the blocks of a prefix of images are a
/// prefix of the table. That is the property the cut needs, and it is asserted
/// two ways: through the arithmetic the reader uses, at every length an image
/// can have, and through the refusals themselves, which is the only place the
/// format publishes a block's span to an operator.
#[test]
fn the_blocks_of_an_image_tile_it_exactly() {
    let dir = scratch("tier-blocks");
    assert_eq!(tier::TIER_BLOCK, 64 << 20, "the block is the writer's own chunk");

    // ── THE ARITHMETIC, at every shape an image can be, without writing one:
    //    the small ones a real index is mostly made of, the boundaries, and
    //    the remainder.
    let head = tier::Artifact::open(&seeded(&dir.0, 7, &synthetic()))
        .expect("it opens")
        .head();
    for span in [
        0,
        1,
        255,
        4096,
        tier::TIER_BLOCK - 1,
        tier::TIER_BLOCK,
        tier::TIER_BLOCK + 1,
        tier::TIER_BLOCK * 8,
        tier::TIER_BLOCK * 8 + 4097,
        60 * (1 << 30),
    ] {
        let count = head.blocks_of(span);
        assert_eq!(count, span.div_ceil(tier::TIER_BLOCK), "{span} bytes");
        let mut floor = 0u64;
        for block in 0..count {
            let (from, len) = head.block_span(1 << 21, span, block);
            assert_eq!(
                from,
                (1 << 21) + floor,
                "{span} bytes: block {block} starts at {from} and {floor} was next"
            );
            assert!(
                len > 0 && len <= tier::TIER_BLOCK,
                "{span} bytes: block {block} spans {len}"
            );
            floor += len;
        }
        assert_eq!(floor, span, "{span} bytes: the blocks end where the image does");
    }

    // ── AND THE REFUSALS SAY THE SAME THING ABOUT A REAL FILE. Two images,
    //    one of two whole blocks plus a remainder and one of a single short
    //    block, so a multi-block image and the ordinary case are both walked.
    let big = 2 * tier::TIER_BLOCK + 4096;
    let small = 40_009u64.next_multiple_of(256);
    let entries = vec![
        Group { id: 0, plane: 0, offset: 0, bytes: big, reserved: big },
        Group { id: 1, plane: 0, offset: big, bytes: small, reserved: small },
    ];
    let payload: Vec<u8> = (0..big + small).map(|at| (at % 251) as u8).collect();
    let path = tier::path(&dir.0, 0x77);
    tier::seed(&path, 0x77, &entries, tier::FLAG_ADAPTERS_ZEROED, &payload)
        .expect("a two-image artifact");

    for group in &entries {
        let mut floor = 0u64;
        for block in 0..head.blocks_of(group.reserved) {
            let (named, from, to) = block_at(&path, group.id, floor);
            assert_eq!(
                named, block,
                "param {}: the byte at {floor} is in block {named}, not {block}",
                group.id
            );
            assert_eq!(
                from, floor,
                "param {}: block {block} states it starts at {from} and {floor} was next",
                group.id
            );
            // The far end of the same block is the same block: the range is a
            // claim about bytes, not a number the message carries.
            assert_eq!(
                block_at(&path, group.id, to - 1),
                (block, from, to),
                "param {}: the last byte of block {block} is somebody else's",
                group.id
            );
            floor = to;
        }
        assert_eq!(
            floor, group.reserved,
            "param {}'s blocks end at {floor} and its image ends at {}",
            group.id, group.reserved
        );
    }
}

/// ── 8. A FORMAT-2 FILE IS SAID OUT LOUD, LEFT ALONE, AND REPLACED ONLY BY A
/// WRITE.
///
/// The §M transition, whole, without a device: every seat that booted before
/// the sections went away has a format-2 file under its key, and what happens
/// to it is three separate claims. It is REFUSED with both numbers (§K.5) —
/// the boot path turns that into a sentence naming
/// `pie model import --prepare-only`, and then STOPS, because since §M-3 a
/// streamed serving load has no cold path to go to. **It is NOT DELETED** —
/// which used to be the distinction between a version refusal and a
/// corruption, and since §M.4 is true of BOTH: no reader on this path deletes
/// anything, because the file is how this machine holds the model and the boot
/// that finds it wrong is not the boot that can rebuild it. And the PREPARE's
/// own write REPLACES it, at this format, atomically — which is the one door
/// that ever does, and since §M-3 the only one an import can reach.
///
/// The last of the three is the one a reader would doubt, because [`store`]
/// SKIPS a key already on the disk: it skips a key whose file `read_head`
/// accepts, and a stale format is not accepted.
///
/// [`store`]: tier::store
#[test]
fn a_format_two_file_is_refused_left_alone_and_replaced_at_format_three() {
    let dir = scratch("tier-transition");
    let images = synthetic();
    let fill = |param: u32, at: u64, into: &mut [u8]| {
        let group = images.entry(param);
        let from = usize::try_from(group.offset + at).expect("an offset");
        into.copy_from_slice(&images.payload[from..from + into.len()]);
        Ok(())
    };
    let path = seeded(&dir.0, 8, &images);
    let before_bytes = std::fs::read(&path).expect("the file this build writes");

    // ── THE FILE THE PREVIOUS BUILD LEFT. Its header was 320 bytes where this
    //    build's is 96, and it carried three section digests where this one
    //    carries a block table — so every field after the format word is at
    //    another offset, which is exactly why the version is checked before
    //    any of them is read.
    weight_cache::restate_format(&path, 2).expect("restating the format");
    let refused = tier::Artifact::open(&path).expect_err("a file from the three-section build");
    assert!(
        matches!(refused, Refused::StaleFormat { states: 2, reads: 3 }),
        "the file's number first, this build's second: {refused:?}"
    );
    assert!(
        std::fs::metadata(&path).is_ok(),
        "and the reader that refused it did not delete the file (§M.4)"
    );
    // Nor does the door an operator's message comes out of — which since §M-3
    // BUILDS the sentence rather than printing it, because a serving load puts
    // it in a `Fault::Residency` and only a prepare prints one.
    let said = tier::refuse(
        &path,
        Some(std::path::Path::new("/models/some.zt")),
        "states a format this build does not read",
    );
    assert!(path.exists(), "`refuse` names the remedy and removes nothing");
    assert!(
        said.contains("pie model import --prepare-only /models/some.zt"),
        "and the remedy is the command that reaches the writer below: {said}"
    );

    // ── AND THE PREPARE THAT FOLLOWS REPLACES IT. Not a skip: `store` skips
    //    a key whose file this build can READ, and it cannot read this one.
    //    §M-3 narrowed WHO calls this — only `Weights::resident` under
    //    `Intent::Prepare`, so only `pie model import` — and changed nothing
    //    about what it does when it is called, which is what this asserts.
    let before = tier::observed();
    tier::store(Some(&dir.0), 8, &images.entries, images.flags, fill);
    let after = tier::observed();
    assert!(
        after.stored > before.stored,
        "the stale file was replaced, not skipped: {before:?} -> {after:?}"
    );

    let artifact = tier::Artifact::open(&path).expect("the rewritten file opens");
    assert_eq!(
        artifact.head().format,
        tier::TIER_FORMAT,
        "and the rewrite landed at this build's format"
    );
    artifact.verify().expect("with digests over its own blocks");
    assert_eq!(
        std::fs::read(&path).expect("the rewritten file"),
        before_bytes,
        "and the rewrite is byte-for-byte what this build writes from these images"
    );

    // ── THE SECOND BOOT AFTER THE UPGRADE IS THE CHEAP ONE AGAIN.
    let before = tier::observed();
    tier::store(Some(&dir.0), 8, &images.entries, images.flags, fill);
    assert!(
        tier::observed().skipped > before.skipped,
        "a file this build can read is not written twice"
    );
}
