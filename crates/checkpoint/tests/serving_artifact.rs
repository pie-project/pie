//! The `pie.serving/1` pair, against real files: `file/emit.rs` writes one
//! and `file/serve.rs` reads it back.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use checkpoint::error::Error;
use checkpoint::file::emit::{self, Object, Part, Payload};
use checkpoint::file::serve::{self, Artifact};
use checkpoint::serving::{self, BlockAlgorithm, Field, PROFILE, Stamp};
use ztensor::DType as ZDType;
use ztensor::format::cbor::{self, Value};

/// The profile's floor: a few KB of fixture spans several blocks,
/// exercising tiling.
const BLOCK: u64 = serving::MIN_BLOCK_BYTES;

fn tmpdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pie_serving_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn stamp() -> Stamp {
    Stamp {
        serving: serving::PROFILE.to_string(),
        backend: "cuda".to_string(),
        sku: "qwen_3".to_string(),
        layout_revision: 1,
        block_bytes: BLOCK,
        block_algorithm: BlockAlgorithm::Xxh3,
        adapters_zeroed: true,
    }
}

fn plane(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|at| seed.wrapping_mul(31).wrapping_add(at as u8))
        .collect()
}

fn dense<'a>(name: &'a str, bytes: &'a [u8]) -> Object<'a> {
    Object {
        name,
        shape: vec![bytes.len() as u64],
        layout: "dense",
        attributes: None,
        parts: vec![Part {
            name: "data",
            dtype: ZDType::U8,
            logical: None,
            payload: Payload::Whole(bytes),
        }],
    }
}

/// A two-part object under the layout `file/write.rs` gives an mxfp4 plane.
fn banked<'a>(name: &'a str, data: &'a [u8], scales: &'a [u8]) -> Object<'a> {
    Object {
        name,
        shape: vec![data.len() as u64 * 2],
        layout: "zt.mx/1",
        attributes: Some(cbor::map([("axis", 1u64), ("block_size", 32u64)])),
        parts: vec![
            Part {
                name: "data",
                dtype: ZDType::U8,
                logical: Some("f4_e2m1"),
                payload: Payload::Whole(data),
            },
            Part {
                name: "scales",
                dtype: ZDType::U8,
                logical: Some("f8_e8m0"),
                payload: Payload::Whole(scales),
            },
        ],
    }
}

/// The fixture every test below writes: three planes in hand-picked order
/// (not name order), one two-part, one aliased.
struct Fixture {
    dir: PathBuf,
    path: PathBuf,
    embed: Vec<u8>,
    codes: Vec<u8>,
    scales: Vec<u8>,
    norm: Vec<u8>,
}

impl Fixture {
    fn write(tag: &str, align: u64) -> Fixture {
        let dir = tmpdir(tag);
        let path = dir.join("qwen--qwen3-30b-a3b.qwen_3.cuda-tp1.mxfp4.zt");
        let fixture = Fixture {
            dir,
            path,
            embed: plane(1, 3 * BLOCK as usize + 777),
            codes: plane(2, 2 * BLOCK as usize + 13),
            scales: plane(3, BLOCK as usize / 4),
            norm: plane(4, 2048),
        };
        fixture.publish(align, &fixture.path);
        fixture
    }

    /// Writes the fixture again at `align`, so a test can hold two files
    /// differing only in placement policy.
    fn publish(&self, align: u64, path: &Path) {
        let objects = [
            dense("embed", &self.embed),
            banked("layer.0.expert_down_bank", &self.codes, &self.scales),
            dense("layer.0.norm", &self.norm),
            // Aliased: `head` is `embed`'s bytes (a blessed tying case).
            dense("head", &self.embed),
            dense("__meta__/model/descriptor", b"{\"sku\":\"qwen_3\"}"),
        ];
        emit::write(path, &stamp(), &provenance(), align, &objects, |object, part, _| {
            panic!("{object}/{part} asked to be filled and this fixture hands its bytes in")
        })
        .unwrap();
    }
}

fn provenance() -> BTreeMap<String, String> {
    BTreeMap::from([
        (checkpoint::file::meta::VERSION_KEY.to_string(), "0.4.0".to_string()),
        (
            checkpoint::file::meta::SOURCE_KEY.to_string(),
            "qwen/qwen3-30b-a3b".to_string(),
        ),
    ])
}

/// Flips one byte of a part, addressed by the manifest's own offset rather
/// than a hard-coded number, so it corrupts only the region meant to be tested.
fn flip(path: &Path, object: &str, part: &str, at: u64) {
    use std::io::{Seek, SeekFrom, Write};
    let manifest = ztensor::read::manifest_of(path).unwrap().unwrap();
    let blob = &manifest.objects[object].parts[part].blob;
    assert!(at < blob.length, "byte {at} is past {object}/{part}");
    let mut file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(blob.offset + at)).unwrap();
    file.write_all(&[0xa5]).unwrap();
    file.sync_all().unwrap();
}

// ── the round trip ──────────────────────────────────────────────────────────

/// Every tensor's bytes come back exactly, in the order the writer was handed.
#[test]
fn every_plane_reads_back_as_the_bytes_that_were_written() {
    let fixture = Fixture::write("roundtrip", emit::SERVING_ALIGN);
    let artifact = Artifact::open(&fixture.path).unwrap();

    // `head` shares `embed`'s offset; the two sit adjacent and the name
    // breaks the tie.
    assert_eq!(
        artifact.sequence(),
        vec!["embed", "head", "layer.0.expert_down_bank", "layer.0.norm"],
    );
    assert_eq!(artifact.part("embed", "data").unwrap(), fixture.embed);
    assert_eq!(
        artifact.part("layer.0.expert_down_bank", "data").unwrap(),
        fixture.codes,
    );
    assert_eq!(
        artifact.part("layer.0.expert_down_bank", "scales").unwrap(),
        fixture.scales,
    );
    assert_eq!(artifact.part("layer.0.norm", "data").unwrap(), fixture.norm);
    assert_eq!(artifact.part("head", "data").unwrap(), fixture.embed);

    // The derivations are the profile layer's, exposed rather than re-derived.
    let spans = artifact.spans();
    assert_eq!(spans.len(), 5, "four planes, one of which has two parts");
    assert_eq!(artifact.alignment(), emit::SERVING_ALIGN);
    assert!(serving::tiling_fault(&spans, artifact.alignment()).is_none());
    assert_eq!(artifact.padded_spans().len(), spans.len());
    let run = artifact.payload();
    assert!(run.start > 0 && run.end > run.start);
    artifact.verify_all().unwrap();
}

/// An exact-equal alias round-trips: two names, one span; the tiling
/// check waves it through rather than reading an overlap.
#[test]
fn a_tied_plane_is_one_span_under_two_names() {
    let fixture = Fixture::write("alias", 4096);
    let artifact = Artifact::open(&fixture.path).unwrap();
    let blob = |name: &str| {
        let part = &artifact.manifest().objects[name].parts["data"];
        (part.blob.offset, part.blob.length)
    };
    assert_eq!(blob("embed"), blob("head"));
    assert_eq!(artifact.part("head", "data").unwrap(), fixture.embed);
    // Both names verify, from the one stretch of bytes both claim.
    artifact.verify(&["embed", "head"]).unwrap();
}

// ── the stamp ───────────────────────────────────────────────────────────────

/// The stamp survives the write; a deployment that differs is refused by
/// the field that differs.
#[test]
fn the_stamp_comes_back_whole_and_a_mismatch_names_the_field() {
    let fixture = Fixture::write("stamp", 4096);
    let artifact = Artifact::open(&fixture.path).unwrap();
    assert_eq!(artifact.stamp(), &stamp());
    artifact.check(&stamp()).unwrap();

    // One field at a time, so the refusal's subject is never in doubt.
    let wants = |mutate: fn(&mut Stamp)| {
        let mut deployment = stamp();
        mutate(&mut deployment);
        artifact.check(&deployment).unwrap_err()
    };
    for (field, mismatch) in [
        (Field::Backend, wants(|it| it.backend = "metal".to_string())),
        (Field::Sku, wants(|it| it.sku = "qwen_3-bf16".to_string())),
        (Field::LayoutRevision, wants(|it| it.layout_revision = 2)),
    ] {
        assert_eq!(mismatch.field, field);
        let sentence = mismatch.refuse(&fixture.path.to_string_lossy());
        assert!(sentence.contains(field.key()), "{sentence}");
        assert!(sentence.contains(&mismatch.artifact), "{sentence}");
        assert!(sentence.contains(&mismatch.deployment), "{sentence}");
        // The middle part: nothing was rewritten and nothing was deleted.
        assert!(sentence.contains("nothing here deletes it"), "{sentence}");
        assert!(sentence.contains("pie model import --force"), "{sentence}");
    }
    assert!(fixture.path.exists(), "a refusal deleted the artifact");
}

/// The version is refused before any other field is believed, as
/// `Unsupported` (re-import) rather than "malformed".
#[test]
fn a_profile_version_this_build_does_not_implement_is_not_a_broken_file() {
    let fixture = Fixture::write("version", 4096);
    let future = fixture.dir.join("future.zt");
    // Only the key moves; the block is otherwise a valid v1 block, which is
    // what putting the version in the key guards against.
    restate(&fixture.path, &future, |attributes| {
        rename(attributes, PROFILE, "pie.serving/2");
    });
    let why = Artifact::open(&future).unwrap_err();
    assert!(matches!(why, Error::Unsupported(_)), "{why}");
    assert!(format!("{why}").contains("re-import"), "{why}");
    assert!(format!("{why}").contains("pie.serving/2"), "{why}");
    // Refusing well is not compatibility: the parts are all there and the
    // command names this file.
    let why = format!("{why}");
    assert!(why.contains("nothing here deletes it"), "{why}");
    assert!(
        why.contains(&format!(
            "pie model import --force {}",
            future.to_string_lossy()
        )),
        "{why}",
    );
    assert!(future.exists(), "a refusal deleted the artifact");

    // An ordinary checkpoint (no serving key) is a different sentence:
    // nothing to re-import, nothing broken.
    let plain = fixture.dir.join("plain.zt");
    strip_every_serving_key(&fixture.path, &plain);
    let why = Artifact::open(&plain).unwrap_err();
    assert!(matches!(why, Error::Checkpoint(_)), "{why}");
    assert!(format!("{why}").contains("ordinary checkpoint"), "{why}");
    assert!(format!("{why}").contains("nothing here deletes it"), "{why}");
}

// ── an open is not a verification ───────────────────────────────────────────

/// A file whose payload has rotted opens exactly as cleanly as one that
/// has not — no byte of the payload is hashed at open. The rot is found
/// by asking.
#[test]
fn an_open_hashes_nothing_and_a_verify_is_what_finds_the_rot() {
    let fixture = Fixture::write("split", 4096);
    flip(&fixture.path, "layer.0.norm", "data", 3);

    let artifact = Artifact::open(&fixture.path).expect("a rotted file still opens");
    assert_eq!(artifact.stamp(), &stamp());
    assert_eq!(artifact.sequence().len(), 4);
    assert_eq!(artifact.alignment(), 4096);
    artifact.identity().unwrap();

    let why = artifact.verify(&["layer.0.norm"]).unwrap_err();
    assert!(format!("{why}").contains("layer.0.norm"), "{why}");
}

/// A flipped byte names the part and the block it's in, ordinal taken in
/// its own part, not the concatenated table.
#[test]
fn a_flipped_byte_is_named_by_its_part_and_its_block() {
    let fixture = Fixture::write("flip", 4096);
    // The second block of the scales part, so the refusal must distinguish
    // part and block.
    flip(&fixture.path, "layer.0.expert_down_bank", "data", BLOCK + 5);

    let artifact = Artifact::open(&fixture.path).unwrap();
    let why = format!(
        "{}",
        artifact.verify(&["layer.0.expert_down_bank"]).unwrap_err()
    );
    assert!(why.contains("layer.0.expert_down_bank"), "{why}");
    assert!(why.contains("\"data\""), "{why}");
    assert!(why.contains("block 1 "), "{why}");
    assert!(why.contains(&format!("{}..{}", BLOCK, 2 * BLOCK)), "{why}");
    assert!(why.contains("xxh3"), "{why}");

    // And the sibling part is untouched, which is what part-local blocks buy.
    let blocks = artifact
        .blocks("layer.0.expert_down_bank", "scales")
        .unwrap();
    assert_eq!(blocks.size(), fixture.scales.len() as u64);
    assert_eq!(blocks.count(), 1);
}

/// A subset verify hashes that subset and nothing else: a rotted plane
/// outside the subset still passes the subset and fails the whole file.
#[test]
fn an_entry_subset_verify_reaches_only_its_own_blocks() {
    let fixture = Fixture::write("subset", 4096);
    flip(&fixture.path, "layer.0.norm", "data", 1);
    let artifact = Artifact::open(&fixture.path).unwrap();

    artifact
        .verify(&["embed", "layer.0.expert_down_bank"])
        .expect("a subset that excludes the rot is intact");
    artifact
        .verify(&["layer.0.norm"])
        .expect_err("the subset that includes it is not");
    artifact
        .verify_all()
        .expect_err("and the whole file is not");
}

/// A prefix of the sequence verifies without touching the rest — the
/// property block granularity exists for.
#[test]
fn a_prefix_of_the_sequence_verifies_without_reading_the_rest() {
    let fixture = Fixture::write("prefix", 4096);
    let sequence = Artifact::open(&fixture.path).unwrap().sequence().len();
    assert_eq!(sequence, 4);
    // The rot is in the last plane, so every proper prefix is clean and
    // the whole is not.
    flip(&fixture.path, "layer.0.norm", "data", 9);
    let artifact = Artifact::open(&fixture.path).unwrap();
    assert_eq!(*artifact.sequence().last().unwrap(), "layer.0.norm");

    for upto in 0..sequence {
        artifact
            .verify_prefix(upto)
            .unwrap_or_else(|why| panic!("prefix {upto} refused: {why}"));
    }
    artifact.verify_prefix(sequence).unwrap_err();
    // A prefix longer than the sequence is the sequence (saturates, not refuses).
    artifact.verify_prefix(sequence + 40).unwrap_err();
}

// ── alignment is the writer's policy ────────────────────────────────────────

/// A file written at one alignment verifies against tables computed at
/// another: blocks tile each part's decoded size, so placement can't
/// reach them, and two files at different alignments carry
/// byte-identical `pie_blocks`.
#[test]
fn a_file_written_at_one_alignment_verifies_under_another() {
    let fixture = Fixture::write("align", 4096);
    let coarse = fixture.dir.join("coarse.zt");
    fixture.publish(emit::SERVING_ALIGN, &coarse);

    let (fine, coarse) = (
        Artifact::open(&fixture.path).unwrap(),
        Artifact::open(&coarse).unwrap(),
    );
    assert_eq!(fine.alignment(), 4096);
    assert_eq!(coarse.alignment(), emit::SERVING_ALIGN);
    assert_ne!(
        fine.payload().start,
        coarse.payload().start,
        "the two placements are genuinely different",
    );

    for (object, part) in [
        ("embed", "data"),
        ("layer.0.expert_down_bank", "data"),
        ("layer.0.expert_down_bank", "scales"),
        ("layer.0.norm", "data"),
    ] {
        assert_eq!(
            fine.blocks(object, part).unwrap().as_bytes(),
            coarse.blocks(object, part).unwrap().as_bytes(),
            "{object}/{part}'s table moved with the alignment",
        );
    }
    fine.verify_all().unwrap();
    coarse.verify_all().unwrap();

    // The artifact key is the same: the reduction is blind to offsets,
    // lengths, alignment and padding.
    assert_eq!(fine.identity().unwrap(), coarse.identity().unwrap());
}

// ── the two doors ───────────────────────────────────────────────────────────

/// The header door opens no mapping and answers the same stamp.
#[test]
fn the_header_door_and_the_mapping_door_say_the_same_thing() {
    let fixture = Fixture::write("doors", 4096);
    let (stated, manifest) = serve::read_head(&fixture.path).unwrap();
    let artifact = Artifact::open(&fixture.path).unwrap();
    assert_eq!(&stated, artifact.stamp());
    assert_eq!(serving::sequence(&manifest), artifact.sequence());
    assert_eq!(
        serving::identity(&stated, &manifest).unwrap(),
        artifact.identity().unwrap(),
    );
}

/// The fill door reads straight into a caller's destination, verifies
/// what landed there, and reports a rot as a claim about the file.
#[test]
fn a_fill_lands_the_bytes_it_verified_in_the_destination() {
    let fixture = Fixture::write("fill", 4096);
    let artifact = Artifact::open(&fixture.path).unwrap();

    let mut embed = vec![0u8; fixture.embed.len()];
    let mut scales = vec![0u8; fixture.scales.len()];
    let fills = [
        serve::Fill {
            object: "embed",
            part: "data",
            into: embed.as_mut_ptr(),
        },
        serve::Fill {
            object: "layer.0.expert_down_bank",
            part: "scales",
            into: scales.as_mut_ptr(),
        },
    ];
    // SAFETY: two distinct local buffers, each valid for its part's whole
    // length, neither aliased and neither read by anything else here.
    unsafe { serve::read_spans_into(&artifact, &fills) }.unwrap();
    assert_eq!(embed, fixture.embed);
    assert_eq!(scales, fixture.scales);

    drop(artifact);
    flip(&fixture.path, "embed", "data", 2 * BLOCK + 1);
    let artifact = Artifact::open(&fixture.path).unwrap();
    let fills = [serve::Fill {
        object: "embed",
        part: "data",
        into: embed.as_mut_ptr(),
    }];
    // SAFETY: as above.
    let why = unsafe { serve::read_spans_into(&artifact, &fills) }.unwrap_err();
    assert!(format!("{why}").contains("block 2 of \"embed\""), "{why}");
}

/// One constructor, so the two sides cannot disagree about a policy: four
/// of the nine required fields (`layout_revision`, `block_bytes`,
/// `block_algorithm`, `adapters_zeroed`) are build facts, not deployment
/// facts. `Stamp::of` takes only the five that differ and fills the rest,
/// so a deployment built from an artifact's own five facts always accepts it.
#[test]
fn a_deployment_built_the_same_way_accepts_the_artifact() {
    let mine = Stamp::of("cuda", "qwen_3");
    let deployment = Stamp::of("cuda", "qwen_3");
    mine.check(&deployment)
        .expect("a deployment built from the same five facts accepts it");
}

/// The cross-recipe artifact is refused by name: a repack moves no value,
/// so two artifacts for different backends can share names, shapes, spans
/// and digests — the stamp is the only thing that can tell them apart.
/// The refusal names the field and the remediation, not just "different".
#[test]
fn an_artifact_for_another_shell_is_refused_naming_the_field() {
    let artifact = Stamp::of("cuda", "qwen_3");
    let deployment = Stamp::of("metal", "qwen_3");
    let why = artifact
        .check(&deployment)
        .expect_err("a cuda artifact is not servable on metal");
    assert_eq!(why.field, serving::Field::Backend);
    let said = why.refuse("/srv/pie/q3.zt");
    for wanted in ["backend", "\"cuda\"", "\"metal\"", "pie model import --force"] {
        assert!(said.contains(wanted), "the refusal does not say {wanted:?}: {said}");
    }
    // Another recipe of the same family is caught the same way.
    let sharded = Stamp::of("cuda", "qwen_3-tp2");
    assert_eq!(
        artifact.check(&sharded).unwrap_err().field,
        serving::Field::Sku,
    );
}

/// A plane's padded extent is real, readable, and zero: `Artifact::part`
/// answers a part's published length, but a tier seats each plane at
/// `reserved` (rounded up to alignment). Padding between blobs is
/// guaranteed `0x00`, and an over-ask is refused rather than answered
/// with a neighbouring plane's bytes.
#[test]
fn a_padded_span_reads_the_file_s_own_zeros_and_refuses_past_them() {
    let fixture = Fixture::write("span", 4096);
    let artifact = Artifact::open(&fixture.path).unwrap();

    // The plane is chosen, not assumed: an aliased blob's neighbour in the
    // sequence need not be its neighbour in the file, so the test finds
    // whichever plane actually has padding.
    let spans = artifact.spans();
    let padded = serving::padded_spans(&spans);
    let (at, span) = spans
        .iter()
        .enumerate()
        .find(|(at, span)| padded[*at] > span.length)
        .expect("some plane of this fixture has padding after it");
    let (object, part, published, room) = (span.object, span.part, span.length, padded[at]);

    // At and below the published length it is `part`, exactly.
    let whole_part = artifact.part(object, part).unwrap().to_vec();
    assert_eq!(artifact.span(object, part, published).unwrap(), &whole_part[..]);
    assert_eq!(artifact.span(object, part, 16).unwrap(), &whole_part[..16]);

    // Past it, the bytes are the writer's padding and they are zero.
    let whole = artifact.span(object, part, room).unwrap();
    assert_eq!(&whole[..published as usize], &whole_part[..]);
    assert!(
        whole[published as usize..].iter().all(|byte| *byte == 0),
        "§2.4 makes inter-blob padding 0x00 and {object:?}/{part:?} does not hold to it",
    );

    // And one byte past what the writer left is a refusal, not a neighbour.
    let why = artifact
        .span(object, part, room + 1)
        .expect_err("asking for padding the file does not have");
    let said = format!("{why}");
    assert!(said.contains("padding this file does not have"), "{said}");

    // An absent name is refused too, rather than answered by position.
    assert!(artifact.span("layer.0.norm", "scales", 8).is_err());
    assert!(artifact.span("no.such.plane", "data", 8).is_err());
    let _ = std::fs::remove_dir_all(&fixture.dir);
}

// ── the owner's rule, made executable ───────────────────────────────────────

/// A tp=1 serving artifact, stripped of its serving key, is still an
/// ordinary checkpoint of the same weights (the owner's rule: the format
/// may not be ztensor while the content is something else). Stripping
/// deletes one key at file level and one per object; nothing else moves.
///
/// Scoped to tp=1: every weight object then carries a layout zTensor
/// itself defines, so stripping pie's additive attributes leaves a file a
/// generic reader interprets in full. At `tp_size > 1`, `pie.banded/1`
/// objects would need this test re-scoped.
#[test]
fn a_tp1_artifact_stripped_of_its_serving_key_is_an_ordinary_checkpoint() {
    let fixture = Fixture::write("strip", 4096);
    let before = ztensor::read::manifest_of(&fixture.path).unwrap().unwrap();
    assert!(
        !serving_keys(before.attributes.as_ref().unwrap()).is_empty(),
        "the fixture had nothing to strip",
    );
    let stripped = fixture.dir.join("stripped.zt");
    strip_every_serving_key(&fixture.path, &stripped);

    // A stock reader, opened with no pie vocabulary of any kind.
    let source = ztensor::Source::open(&stripped).unwrap();
    let after = source.provenance().as_root().unwrap().clone();
    assert_eq!(
        after.objects.keys().collect::<Vec<_>>(),
        before.objects.keys().collect::<Vec<_>>(),
    );
    for (name, object) in &after.objects {
        let was = &before.objects[name];
        assert_eq!(object.shape, was.shape, "{name}'s shape moved");
        assert_eq!(object.layout, was.layout, "{name}'s layout moved");
        for (part, held) in &object.parts {
            assert_eq!(held.blob, was.parts[part].blob, "{name}/{part} moved");
            assert_eq!(held.digest, was.parts[part].digest, "{name}/{part} changed");
        }
    }
    let servable = Artifact::open(&fixture.path).unwrap();
    for (name, object) in &after.objects {
        for part in object.parts.keys() {
            let tensor = source.tensor(name).unwrap();
            assert_eq!(
                &*tensor.part(part).unwrap().bytes().unwrap(),
                servable.part(name, part).unwrap(),
                "{name}/{part}'s bytes changed",
            );
            // The container's own digest still answers for them.
            tensor.part(part).unwrap().verify().unwrap();
        }
    }
    drop(servable);
    assert!(
        after.attributes.is_none()
            || serving_keys(after.attributes.as_ref().unwrap()).is_empty(),
    );
    for object in after.objects.values() {
        if let Some(attributes) = &object.attributes {
            assert!(serving_keys(attributes).is_empty());
        }
    }
    // Provenance keys aren't this profile's and stay: they say where the
    // weights came from.
    let attributes = after.attributes.as_ref().unwrap();
    assert!(attributes.get(checkpoint::file::meta::SOURCE_KEY).is_some());
    // Object attributes the layout needs survive untouched, making the
    // stripped file decodable.
    assert_eq!(
        after.objects["layer.0.expert_down_bank"]
            .attributes
            .as_ref()
            .and_then(|it| it.get("block_size")),
        Some(&Value::Uint(32)),
    );

    // It has stopped being servable, the other half of the rule.
    let why = Artifact::open(&stripped).unwrap_err();
    assert!(format!("{why}").contains(PROFILE), "{why}");
    assert!(format!("{why}").contains("ordinary checkpoint"), "{why}");
}

/// The cause, asserted beside the effect: every weight object this writer
/// emits carries a layout from zTensor's own vocabulary, never a `pie.*`
/// one — what makes the strip above pass. An unknown layout must be
/// refused by a generic reader; an unknown attribute key must be ignored,
/// which is why the two are held to different standards here.
#[test]
fn every_weight_carries_a_layout_ztensor_itself_defines() {
    let fixture = Fixture::write("layouts", 4096);
    let manifest = ztensor::read::manifest_of(&fixture.path).unwrap().unwrap();
    for (name, object) in &manifest.objects {
        assert!(
            !object.layout.starts_with("pie."),
            "{name} carries the pie layout {:?}; the file profile adds attributes and \
             not layouts, and a stripped object under a pie layout is one a generic \
             reader must refuse to interpret",
            object.layout,
        );
        assert!(
            object.layout == "dense"
                || object.layout.starts_with("zt.")
                || object.layout.starts_with("gguf."),
            "{name} carries the unregistered layout {:?}",
            object.layout,
        );
    }
}

// ── the surgery the strip test needs ────────────────────────────────────────

/// Copies `from` to `to` with the serving key deleted: one key at file
/// level, one key per object, nothing else touched.
///
/// Deliberately not a re-write: the payload region is copied byte for
/// byte and only the manifest blob is rebuilt, so every offset and length
/// is the one the serving writer chose.
fn strip_every_serving_key(from: &Path, to: &Path) {
    restate(from, to, |attributes| strip(attributes, PROFILE));
}

/// Copies `from` to `to`, letting `edit` rewrite the file-level
/// attributes and deleting the serving key from every object's
/// attributes. The manifest blob is decoded as CBOR, edited, re-encoded,
/// and a fresh footer written over the new offsets.
fn restate(from: &Path, to: &Path, edit: impl FnOnce(&mut Value)) {
    const FOOTER: usize = 40;
    let raw = std::fs::read(from).unwrap();
    let footer = &raw[raw.len() - FOOTER..];
    let at = u64::from_le_bytes(footer[0..8].try_into().unwrap()) as usize;
    let len = u64::from_le_bytes(footer[8..16].try_into().unwrap()) as usize;

    let mut manifest = cbor::decode(&raw[at..at + len]).unwrap();
    let Value::Map(entries) = &mut manifest else {
        panic!("a manifest that is not a map");
    };
    let mut edit = Some(edit);
    for (key, value) in entries.iter_mut() {
        match key {
            Value::Text(key) if key == "attributes" => {
                if let Some(edit) = edit.take() {
                    edit(value);
                }
            }
            Value::Text(key) if key == "objects" => {
                let Value::Map(objects) = value else { continue };
                for (_, object) in objects.iter_mut() {
                    let Value::Map(fields) = object else { continue };
                    for (field, attributes) in fields.iter_mut() {
                        if matches!(field, Value::Text(it) if it == "attributes") {
                            // An object whose only attribute was serving now
                            // has an empty map; permitted, ignored.
                            strip(attributes, PROFILE);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let encoded = cbor::encode(&manifest).unwrap();
    let mut out = raw[..at].to_vec();
    out.extend_from_slice(&encoded);
    let digest = ztensor::DigestAlgorithm::Xxh3.digest(&encoded);
    let digest = u64::from_str_radix(digest.split_once(':').unwrap().1, 16).unwrap();
    let mut footer = [0u8; FOOTER];
    footer[0..8].copy_from_slice(&(at as u64).to_le_bytes());
    footer[8..16].copy_from_slice(&(encoded.len() as u64).to_le_bytes());
    footer[16..24].copy_from_slice(&digest.to_le_bytes());
    footer[24..28].copy_from_slice(&2u32.to_le_bytes());
    footer[32..40].copy_from_slice(&[0x89, b'Z', b'T', b'2', 0x0d, 0x0a, 0x1a, 0x0a]);
    out.extend_from_slice(&footer);
    std::fs::write(to, out).unwrap();
}

/// Every key of `attributes` that belongs to this profile, at any version.
fn serving_keys(attributes: &Value) -> Vec<String> {
    let Value::Map(entries) = attributes else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|(key, _)| match key {
            Value::Text(key) if key.starts_with("pie.serving/") => Some(key.clone()),
            _ => None,
        })
        .collect()
}

fn strip(attributes: &mut Value, key: &str) {
    if let Value::Map(entries) = attributes {
        entries.retain(|(spelled, _)| !matches!(spelled, Value::Text(it) if it == key));
    }
}

/// Moves an attribute's VALUE to a different key, leaving the value untouched.
fn rename(attributes: &mut Value, from: &str, to: &str) {
    if let Value::Map(entries) = attributes {
        for (key, _) in entries.iter_mut() {
            if matches!(key, Value::Text(it) if it == from) {
                *key = Value::Text(to.to_string());
            }
        }
    }
}
