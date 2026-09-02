//! The `pie.serving/1` pair, against real files: `file/emit.rs` writes one
//! and `file/serve.rs` reads it back.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use checkpoint::error::Error;
use checkpoint::file::emit::{self, Object, Payload};
use checkpoint::file::serve::{self, Artifact};
use checkpoint::serving::{self, Field, PROFILE, Stamp};
use ztensor::format::cbor::{self, Value};
use ztensor::{Leaf, Term};

/// The block every blob's digests tile; only the block-ordinal test pays for
/// a fixture that spans one.
const BLOCK: u64 = serving::BLOCK_BYTES;

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
        adapters_zeroed: true,
    }
}

fn plane(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|at| seed.wrapping_mul(31).wrapping_add(at as u8))
        .collect()
}

fn leaf<'a>(name: &'a str, bytes: &'a [u8]) -> Object<'a> {
    Object::leaf(name, vec![bytes.len() as u64], Leaf::U8, bytes)
}

/// The two-plane object an mxfp4 weight is: two codes a byte, one scale per
/// group of 32, both in one blob.
fn banked<'a>(name: &'a str, codes: &'a [u8], scales: &'a [u8]) -> Object<'a> {
    Object {
        name,
        shape: vec![codes.len() as u64 * 2],
        term: Some(Term::parse("g32_e2m1_e8m0_n").unwrap()),
        layout: None,
        attributes: None,
        planes: vec![Payload::Whole(codes), Payload::Whole(scales)],
    }
}

/// The fixture every test below writes: three weights in hand-picked order
/// (not name order), one of two planes, one aliased.
struct Fixture {
    dir: PathBuf,
    path: PathBuf,
    embed: Vec<u8>,
    codes: Vec<u8>,
    scales: Vec<u8>,
    norm: Vec<u8>,
}

impl Fixture {
    /// A few KB: placement and the stamp, at any alignment.
    fn write(tag: &str, align: u64) -> Fixture {
        Fixture::sized(tag, align, 4096)
    }

    /// Past one block, so `embed` and the bank each tile into two.
    fn blocked(tag: &str, align: u64) -> Fixture {
        Fixture::sized(tag, align, BLOCK / 2)
    }

    fn sized(tag: &str, align: u64, unit: u64) -> Fixture {
        let dir = tmpdir(tag);
        let path = dir.join("qwen--qwen3-30b-a3b.qwen_3.cuda-tp1.mxfp4.zt");
        let unit = unit as usize;
        // Sizes chosen so the objects land an odd number of pages apart.
        // The bank's byte count is a multiple of 16: two codes a byte, 32 a
        // group, one scale each.
        let codes = plane(2, 2 * unit + 16);
        let scales = plane(3, codes.len() / 16);
        let fixture = Fixture {
            dir,
            path,
            embed: plane(1, 3 * unit + 777),
            codes,
            scales,
            norm: plane(4, 2048),
        };
        fixture.publish(align, &fixture.path);
        fixture
    }

    /// Writes the fixture again at `align`, so a test can hold two files
    /// differing only in placement policy.
    fn publish(&self, align: u64, path: &Path) {
        let objects = [
            leaf("embed", &self.embed),
            banked("layer.0.expert_down_bank", &self.codes, &self.scales),
            leaf("layer.0.norm", &self.norm),
            // Aliased: `head` is `embed`'s bytes (a blessed tying case).
            leaf("head", &self.embed),
            leaf("__meta__/model/descriptor", b"{\"sku\":\"qwen_3\"}"),
        ];
        emit::write(path, &stamp(), &provenance(), align, &objects, |object, plane, _| {
            panic!("{object}/{plane} asked to be filled and this fixture hands its bytes in")
        })
        .unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
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

/// Flips one byte of an object's blob, addressed by the manifest's own
/// offset rather than a hard-coded number, so it corrupts only the region
/// meant to be tested.
fn flip(path: &Path, object: &str, at: u64) {
    use std::io::{Seek, SeekFrom, Write};
    let manifest = ztensor::read::manifest_of(path).unwrap().unwrap();
    let blob = &manifest.objects[object].blob;
    assert!(at < blob.length, "byte {at} is past {object}");
    let mut file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(blob.offset + at)).unwrap();
    file.write_all(&[0xa5]).unwrap();
    file.sync_all().unwrap();
}

/// Every plane's bytes come back exactly, by the name a trace gives it, in
/// the order the writer was handed.
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
    assert_eq!(artifact.plane("embed").unwrap(), fixture.embed);
    assert_eq!(
        artifact.plane("layer.0.expert_down_bank").unwrap(),
        fixture.codes,
    );
    assert_eq!(
        artifact.plane("layer.0.expert_down_bank.scales").unwrap(),
        fixture.scales,
    );
    assert_eq!(artifact.plane("layer.0.norm").unwrap(), fixture.norm);
    assert_eq!(artifact.plane("head").unwrap(), fixture.embed);
    // The bank's planes lie in one blob, the scales at the first plane
    // boundary past the codes.
    let bank = artifact.object("layer.0.expert_down_bank").unwrap();
    let align = ztensor::format::PLANE_ALIGN as usize;
    let scales_at = fixture.codes.len().div_ceil(align) * align;
    assert_eq!(&bank[..fixture.codes.len()], &fixture.codes[..]);
    assert_eq!(&bank[scales_at..], &fixture.scales[..]);

    // The derivations are the profile layer's, exposed rather than re-derived.
    let spans = artifact.spans();
    assert_eq!(spans.len(), 4, "four weights, one of which has two planes");
    assert_eq!(artifact.alignment(), emit::SERVING_ALIGN);
    assert!(serving::tiling_fault(&spans, artifact.alignment()).is_none());
    assert!(spans[0].offset > 0 && spans[0].end() > spans[0].offset);
    artifact.verify_all().unwrap();
}

/// An exact-equal alias round-trips: two names, one span; the tiling
/// check waves it through rather than reading an overlap.
#[test]
fn a_tied_plane_is_one_span_under_two_names() {
    let fixture = Fixture::write("alias", 4096);
    let artifact = Artifact::open(&fixture.path).unwrap();
    let blob = |name: &str| {
        let blob = &artifact.manifest().objects[name].blob;
        (blob.offset, blob.length)
    };
    assert_eq!(blob("embed"), blob("head"));
    assert_eq!(artifact.plane("head").unwrap(), fixture.embed);
    // Both names verify, from the one stretch of bytes both claim.
    artifact.verify(&["embed", "head"]).unwrap();
}

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

/// A file whose payload has rotted opens exactly as cleanly as one that
/// has not — no byte of the payload is hashed at open. The rot is found
/// by asking.
#[test]
fn an_open_hashes_nothing_and_a_verify_is_what_finds_the_rot() {
    let fixture = Fixture::write("split", 4096);
    flip(&fixture.path, "layer.0.norm", 3);

    let artifact = Artifact::open(&fixture.path).expect("a rotted file still opens");
    assert_eq!(artifact.stamp(), &stamp());
    assert_eq!(artifact.sequence().len(), 4);
    assert_eq!(artifact.alignment(), 4096);

    let why = artifact.verify(&["layer.0.norm"]).unwrap_err();
    assert!(format!("{why}").contains("layer.0.norm"), "{why}");
}

/// A flipped byte names the object and the block it's in, ordinal taken
/// within the object's own blob. The blocks tile the blob whole — both
/// planes and the padding between — and reach no neighbour.
#[test]
fn a_flipped_byte_is_named_by_its_object_and_its_block() {
    let fixture = Fixture::blocked("flip", 4096);
    // The second block of the bank's blob, so the refusal must count
    // blocks within the object.
    flip(&fixture.path, "layer.0.expert_down_bank", BLOCK + 5);

    let artifact = Artifact::open(&fixture.path).unwrap();
    let blocks = artifact.blocks("layer.0.expert_down_bank").unwrap();
    let blob = artifact.object("layer.0.expert_down_bank").unwrap().len() as u64;
    assert_eq!(blocks.size(), blob);
    assert_eq!(blocks.block_bytes(), BLOCK);
    assert_eq!(blocks.count(), blob.div_ceil(BLOCK));
    let second = blocks.span(1).unwrap();
    assert_eq!(second.start, BLOCK);

    let why = format!(
        "{}",
        artifact.verify(&["layer.0.expert_down_bank"]).unwrap_err()
    );
    assert!(why.contains("layer.0.expert_down_bank"), "{why}");
    assert!(why.contains("block 1 "), "{why}");
    assert!(why.contains(&format!("{}..{}", second.start, second.end)), "{why}");
    assert!(why.contains("xxh3"), "{why}");

    // Blocks are blob-local: the neighbouring object is untouched, and so
    // is the bank's other plane.
    artifact.verify(&["layer.0.norm", "embed"]).unwrap();
    assert_eq!(
        artifact.plane("layer.0.expert_down_bank.scales").unwrap(),
        fixture.scales,
    );
}

/// A subset verify hashes that subset and nothing else: a rotted plane
/// outside the subset still passes the subset and fails the whole file.
#[test]
fn an_entry_subset_verify_reaches_only_its_own_blocks() {
    let fixture = Fixture::write("subset", 4096);
    flip(&fixture.path, "layer.0.norm", 1);
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
    flip(&fixture.path, "layer.0.norm", 9);
    let artifact = Artifact::open(&fixture.path).unwrap();
    assert_eq!(*artifact.sequence().last().unwrap(), "layer.0.norm");

    let sequence = artifact.sequence();
    for upto in 0..sequence.len() {
        artifact
            .verify(&sequence[..upto])
            .unwrap_or_else(|why| panic!("prefix {upto} refused: {why}"));
    }
    artifact.verify(&sequence).unwrap_err();
}

/// A file written at one alignment verifies against digests computed at
/// another: blocks tile each blob's decoded size, so placement can't reach
/// them, and two files at different alignments carry identical blocks.
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
        fine.spans()[0].offset,
        coarse.spans()[0].offset,
        "the two placements are genuinely different",
    );

    for object in ["embed", "layer.0.expert_down_bank", "layer.0.norm"] {
        assert_eq!(
            fine.blocks(object).unwrap().iter().collect::<Vec<_>>(),
            coarse.blocks(object).unwrap().iter().collect::<Vec<_>>(),
            "{object}'s blocks moved with the alignment",
        );
    }
    fine.verify_all().unwrap();
    coarse.verify_all().unwrap();
}

/// The header door opens no mapping and answers the same stamp.
#[test]
fn the_header_door_and_the_mapping_door_say_the_same_thing() {
    let fixture = Fixture::write("doors", 4096);
    let (stated, manifest) = serve::read_head(&fixture.path).unwrap();
    let artifact = Artifact::open(&fixture.path).unwrap();
    assert_eq!(&stated, artifact.stamp());
    assert_eq!(serving::sequence(&manifest), artifact.sequence());
}

/// `Stamp::of` fills the build facts (`layout_revision`, `adapters_zeroed`)
/// itself, so a deployment built from an artifact's own facts always
/// accepts it.
#[test]
fn a_deployment_built_the_same_way_accepts_the_artifact() {
    let mine = Stamp::of("cuda", "qwen_3");
    let deployment = Stamp::of("cuda", "qwen_3");
    mine.check(&deployment)
        .expect("a deployment built from the same facts accepts it");
}

/// Two artifacts for different backends can share names, shapes, spans and
/// digests, so the stamp alone tells them apart; the refusal names the field
/// and the remedy.
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

/// A tp=1 serving artifact with its serving key deleted is an ordinary
/// checkpoint of the same weights: at tp=1 every weight lies canonically or
/// under a layout zTensor itself defines. A repacked (`pie.mma_tiled/1`)
/// bank would need this re-scoped.
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
        assert_eq!(object.term, was.term, "{name}'s type moved");
        assert_eq!(object.layout, was.layout, "{name}'s layout moved");
        assert_eq!(object.blob, was.blob, "{name} moved");
    }
    let servable = Artifact::open(&fixture.path).unwrap();
    for name in after.objects.keys() {
        let tensor = source.tensor(name).unwrap();
        assert_eq!(
            tensor.map().unwrap(),
            servable.object(name).unwrap(),
            "{name}'s bytes changed",
        );
        // The container's own digest still answers for them.
        tensor.verify().unwrap();
    }
    drop(servable);
    assert!(
        after.attributes.is_none()
            || serving_keys(after.attributes.as_ref().unwrap()).is_empty(),
    );
    // Provenance keys aren't this profile's and stay: they say where the
    // weights came from.
    let attributes = after.attributes.as_ref().unwrap();
    assert!(attributes.get(checkpoint::file::meta::SOURCE_KEY).is_some());
    // The type a generic reader decodes the bank by survives untouched.
    assert_eq!(
        after.objects["layer.0.expert_down_bank"]
            .term
            .as_ref()
            .map(ToString::to_string)
            .as_deref(),
        Some("g32_e2m1_e8m0_n"),
    );

    // It has stopped being servable, the other half of the rule.
    let why = Artifact::open(&stripped).unwrap_err();
    assert!(format!("{why}").contains(PROFILE), "{why}");
    assert!(format!("{why}").contains("ordinary checkpoint"), "{why}");
}

/// Every weight object this fixture writes lies canonically or under a
/// layout from zTensor's own vocabulary, never a `pie.*` one — what lets the
/// strip above pass (an unknown layout is refused by a generic reader; an
/// unknown attribute key is ignored).
#[test]
fn every_weight_carries_a_layout_ztensor_itself_defines() {
    let fixture = Fixture::write("layouts", 4096);
    let manifest = ztensor::read::manifest_of(&fixture.path).unwrap().unwrap();
    for (name, object) in &manifest.objects {
        let layout = object.layout.as_deref();
        assert!(
            !layout.is_some_and(|it| it.starts_with("pie.")),
            "{name} carries the pie layout {layout:?}; the file profile adds attributes and \
             not layouts, and a stripped object under a pie layout is one a generic \
             reader must refuse to interpret",
        );
        assert!(
            layout.is_none_or(|it| it.starts_with("zt.") || it.starts_with("gguf.")),
            "{name} carries the unregistered layout {layout:?}",
        );
    }
}

/// Copies `from` to `to` with the serving key deleted. The payload region is
/// copied byte for byte and only the manifest blob is rebuilt, so every
/// offset and length stays the serving writer's.
fn strip_every_serving_key(from: &Path, to: &Path) {
    restate(from, to, |attributes| strip(attributes, PROFILE));
}

/// Copies `from` to `to`, letting `edit` rewrite the file-level
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
        if matches!(key, Value::Text(key) if key == "attributes") {
            if let Some(edit) = edit.take() {
                edit(value);
            }
        }
    }

    let encoded = cbor::encode(&manifest).unwrap();
    let mut out = raw[..at].to_vec();
    out.extend_from_slice(&encoded);
    let digest = xxhash_rust::xxh3::xxh3_64(&encoded);
    let mut footer = [0u8; FOOTER];
    footer[0..8].copy_from_slice(&(at as u64).to_le_bytes());
    footer[8..16].copy_from_slice(&(encoded.len() as u64).to_le_bytes());
    footer[16..24].copy_from_slice(&digest.to_le_bytes());
    footer[24..28].copy_from_slice(&3u32.to_le_bytes());
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
