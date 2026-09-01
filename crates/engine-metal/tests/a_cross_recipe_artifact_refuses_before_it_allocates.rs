//! **A SERVING ARTIFACT WRITTEN FOR ANOTHER DEPLOYMENT IS REFUSED, AND THE
//! REFUSAL TAKES NOTHING** (§M-4c, the Metal call site).
//!
//! The failure this closes was measured on this shell: a `.zt` converted for
//! the CUDA engine and served here loaded in 0.1 s and answered
//! `"productiveeldahar打造成…"`. A repack moves no value, so the cross-shell
//! artifact and the right one have the same object names, the same shapes, the
//! same spans and the same part digests — `weights::readable_plane_orders`'s
//! header states exactly why nothing about the BYTES can tell them apart. The
//! `pie.serving/1` stamp is the only thing that can, and
//! `weights::serves_this_deployment` is where it is asked.
//!
//! # What this file adds that the sibling shell's unit gates do not
//!
//! `engine-cuda`'s three gates assert the refusal and its wording. They assert
//! *"before a plane lands"* STRUCTURALLY — `serve::read_head` does positioned
//! reads, and the call sits above the settle. This one MEASURES it. Two
//! process-wide counters exist for that and for nothing else:
//!
//! * [`engine_metal::device::reservations`] — every `MTLBuffer` this process
//!   has asked the device for, counted at all three doors that can mint one.
//! * [`engine_metal::host_source::descriptors`] — every backing file the
//!   streamed tier's source door has opened.
//!
//! Read around the refusing call, both must be unmoved. That covers the whole
//! of what a Metal load takes before it can answer: the weight store
//! (`Buffer::zeroed`), the host band table (`HostSource::open`), the arena and
//! the scratch plane — all four are reservations at one of those two doors, and
//! all four are below the check in `Weights::resident`.
//!
//! **AND THE COUNTER IS NOT VACUOUS**, which is the arm that makes the claim
//! worth anything: the ACCEPTING load in
//! [`a_stamp_that_agrees_passes_and_that_load_does_reserve`] moves the same
//! counter, off the same call, on the next line. A gate that only ever saw zero
//! would pass just as well against a counter nobody increments.
//!
//! # The fixture, and why it is written rather than imported
//!
//! `pie model import` on this box stamps `backend` `"metal"` —
//! `runtime::engine::load::this_box()` keys on the linked engine feature — so
//! the interesting artifact, a cuda-stamped one, is not a thing this machine's
//! importer will produce. It is written directly instead:
//! `checkpoint::file::write::Writer::create_serving` takes a `Stamp` and one
//! plane of bytes, which is a whole, valid, readable serving artifact and
//! exactly as much of one as a stamp check ever looks at. The two fixtures here
//! differ in ONE field of one attribute and are byte-identical otherwise, which
//! is the property the negative and positive arms rest on.
//!
//! The trace is likewise hand-built and empty. That is not a shortcut: the
//! check reads `Trace::name` and `Trace::platform` and nothing else, and an
//! empty parameter list is what lets the ACCEPTING arm run to completion for a
//! single one-byte reservation instead of standing up half a gigabyte of
//! weights to prove that a text field matched.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use checkpoint::file::write::Writer;
use checkpoint::serving::Stamp;
use checkpoint::types::{DType, Encoding, TensorDecl, TensorId, Visibility};
use engine_metal::device::{self, Context};
use engine_metal::{Fault, Handles, ResidencyPlan, Weights};
use model_ir::{Platform, Trace};

/// The row this gate speaks for: the four-bit 0.8B, which is
/// `four_bit_first_light`'s vehicle and the one artifact shape both shells are
/// pairing on.
const SKU: &str = "qwen35-d0.8b-mlxu4-kv-bf16";

/// **THE PRECISION IS THE CATALOG'S ANSWER AND NEVER THE NAME'S.** Read here
/// rather than spelled, for the reason `models::PRECISIONS` gives at length:
/// the two DQ rows are NAMED for their two-bit experts and are mostly
/// four-bit, so `qwen38-flash-mlxu2-kv-bf16` is precision `mlxu4-mlxu2`. A
/// gate that spelled its own would be asserting against a second table.
fn precision() -> &'static str {
    models::precision_of(SKU).expect("the catalog states this row's precision")
}

/// A directory of this test's own, emptied first.
fn tmp(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("metal_stamp_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// **ONE SERVING ARTIFACT, STAMPED FOR `backend` AT DEGREE `tp_size`.**
///
/// One plane of 8 KiB under a name a serving artifact would use. Everything
/// else — the layout revision, the block size, the digest algorithm, the zeroed
/// adapters — comes from [`Stamp::of`] and is not spelled here, which is the
/// whole reason that constructor exists: a boot compares field by field, and a
/// policy constant written twice is a field that can disagree with itself.
fn artifact(dir: &Path, backend: &str, tp_size: u64, precision: &str) -> PathBuf {
    let path = dir.join(format!("{backend}-tp{tp_size}.zt"));
    let stamp = Stamp::of(backend, tp_size, SKU, precision, None);
    let mut writer =
        Writer::create_serving(&path, &BTreeMap::new(), stamp).expect("the artifact opens");
    writer
        .add_tensor(&plane("embed"), &vec![7u8; 8192])
        .expect("its one plane writes");
    writer.finish().expect("and it publishes");
    path
}

/// An ordinary checkpoint of the same bytes — no `pie.serving/1` key at all,
/// which is every snapshot this tree has ever landed.
fn plain(dir: &Path) -> PathBuf {
    let path = dir.join("plain.zt");
    let mut writer = Writer::create(&path, &BTreeMap::new()).expect("the checkpoint opens");
    writer
        .add_tensor(&plane("embed"), &vec![7u8; 8192])
        .expect("its one plane writes");
    writer.finish().expect("and it publishes");
    path
}

fn plane(name: &str) -> TensorDecl {
    TensorDecl {
        id: TensorId(0),
        name: name.to_string(),
        shape: vec![8192],
        encoding: Encoding::Raw(DType::U8),
        alignment: 64,
        visibility: Visibility::default(),
    }
}

/// **THE TRACE THE CHECK READS, AND ONLY WHAT IT READS.** `name` is the SKU the
/// stamp is compared against and `platform` is the backend; the parameter list
/// is empty so that a load which gets PAST the check finishes for one empty
/// reservation rather than for a model.
fn trace() -> Trace {
    Trace {
        name: SKU.to_string(),
        platform: Platform::Metal,
        params: Vec::new(),
        caches: Vec::new(),
        values: Vec::new(),
        nodes: Vec::new(),
        seams: Vec::new(),
    }
}

/// A contract with nothing in it. The landing has no plane to bind because the
/// trace declares none; what is under test is the field comparison above it.
fn contract() -> ModelContract {
    ModelContract {
        alignment: 64,
        tensors: Vec::new(),
        groups: Vec::new(),
    }
}

/// **THE COUNTERS ARE THE PROCESS'S AND THE TESTS ARE THREADS**, so every
/// measured load takes this first.
///
/// [`device::reservations`] counts what this PROCESS asked the device for —
/// there is no per-load counter and there should not be, since the claim is
/// about a shell that has not been built yet at the moment of the refusal. Two
/// gates in this binary running at once would each read the other's
/// reservations into its own delta, and the accepting arm exists precisely to
/// make sure that number is not always zero. So the four are serialized, and
/// the lock is taken OUTSIDE the pair of reads rather than between them.
static MEASURING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// One load, and what the two counters did across it.
fn land(
    device: &Context,
    path: &Path,
    tp_size: u64,
    precision: &str,
) -> (engine_metal::Result<Weights>, u64, u64) {
    let _measuring = MEASURING.lock().unwrap_or_else(|it| it.into_inner());
    let handles = Handles::new();
    let (trace, contract) = (trace(), contract());
    let (buffers, files) = (device::reservations(), engine_metal::host_source::descriptors());
    let landed = Weights::resident(
        device,
        &handles,
        &trace,
        &contract,
        path,
        &ResidencyPlan::default(),
        tp_size,
        precision,
    );
    (
        landed,
        device::reservations() - buffers,
        engine_metal::host_source::descriptors() - files,
    )
}

/// The refusal, as text, or a panic naming what landed instead.
fn refusal(landed: engine_metal::Result<Weights>, what: &str) -> String {
    match landed {
        Ok(_) => panic!("{what} was landed rather than refused"),
        Err(fault) => {
            assert!(
                matches!(fault, Fault::Recipe(_)),
                "{what} was refused, but not by the stamp gate: {fault}"
            );
            fault.to_string()
        }
    }
}

/// **THE GATE.** A cuda-stamped artifact fed to this shell is refused by the
/// field that disagrees — and the device and the source door are untouched when
/// it is.
#[test]
fn a_cross_recipe_artifact_refuses_before_it_allocates() {
    if !device::present() {
        println!("SKIP: this machine publishes no Metal device");
        return;
    }
    let device = Context::bind().expect("the system default device binds");
    let dir = tmp("cross");
    let foreign = artifact(&dir, "cuda", 1, precision());

    let (landed, buffers, files) = land(&device, &foreign, 1, precision());
    let said = refusal(landed, "a cuda artifact");

    // (1) THE FIELD, AND BOTH SIDES OF IT. The whole reason the stamp is typed
    //     fields rather than a fold: "different" is an error code, and this is
    //     a sentence an operator can act on.
    for wanted in [
        "backend",
        "\"cuda\"",
        "\"metal\"",
        "pie model import --force",
        "nothing here deletes it",
    ] {
        assert!(
            said.contains(wanted),
            "the refusal does not say {wanted:?}: {said}"
        );
    }

    // (2) AND IT TOOK NOTHING. Every byte a Metal load reserves — the weight
    //     store, the host band table, the arena, the scratch plane — is minted
    //     at one of these two doors, and both are below the check.
    assert_eq!(
        buffers, 0,
        "the refusing load reserved {buffers} device buffer(s); the stamp gate is \
         supposed to answer before `Buffer::zeroed`, the arena and the scratch plane"
    );
    assert_eq!(
        files, 0,
        "the refusing load opened {files} backing file(s); the stamp gate is supposed \
         to answer before `HostSource::open` takes a descriptor"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// **THE DEGREE IS THE OTHER HALF OF A RECIPE**, and it is caught the same way
/// and just as early. Mirrors `engine-cuda`'s `an_artifact_cut_for_another_degree_is_refused`.
#[test]
fn an_artifact_cut_for_another_degree_is_refused_just_as_early() {
    if !device::present() {
        println!("SKIP: this machine publishes no Metal device");
        return;
    }
    let device = Context::bind().expect("the system default device binds");
    let dir = tmp("degree");
    let one = artifact(&dir, "metal", 1, precision());

    let (landed, buffers, files) = land(&device, &one, 2, precision());
    let said = refusal(landed, "a tp1 artifact under a tp2 deployment");
    assert!(said.contains("tp_size"), "{said}");
    assert_eq!((buffers, files), (0, 0), "{said}");

    // And the precision is the third field, read off the catalog rather than
    // off the SKU's name — a deployment serving the same row at another
    // numeric form is a different artifact.
    let (landed, buffers, files) = land(&device, &one, 1, "bf16");
    let said = refusal(landed, "an mlxu4 artifact under a bf16 deployment");
    assert!(said.contains("precision"), "{said}");
    assert_eq!((buffers, files), (0, 0), "{said}");

    std::fs::remove_dir_all(&dir).ok();
}

/// **THE POSITIVE ARM, AND THE ONE THAT MAKES THE COUNTER MEAN SOMETHING.**
///
/// The same fixture stamped for THIS deployment passes the check — and the load
/// behind it goes on to reserve, off the same call, through the same counter.
/// Byte for byte the artifact above with one field of one attribute changed.
#[test]
fn a_stamp_that_agrees_passes_and_that_load_does_reserve() {
    if !device::present() {
        println!("SKIP: this machine publishes no Metal device");
        return;
    }
    let device = Context::bind().expect("the system default device binds");
    let dir = tmp("agrees");
    let ours = artifact(&dir, "metal", 1, precision());

    let (landed, buffers, _) = land(&device, &ours, 1, precision());
    landed.expect("a metal-stamped artifact of this row serves this deployment");
    assert!(
        buffers > 0,
        "the accepting load reserved nothing at all, so the refusing arm's `== 0` is \
         a claim about a counter nobody moves"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A `pie.serving` CLAIM whose stamp does not read back is a refusal, not the
/// ordinary checkpoint the file also is — a stamped artifact losing its stamp
/// to decay and then passing the very check the stamp feeds would be the
/// cross-recipe boot by the quiet door. The fixture is a WELL-FORMED container
/// whose serving block states `tp_size: 0`, which `Stamp::decode` refuses by
/// name; byte-editing a good file's member was tried on the CUDA side first
/// and trips `ManifestHash` instead — an earlier, different refusal that
/// tests nothing about this split.
#[test]
fn a_claimed_profile_that_does_not_read_back_is_refused_not_served_unstamped() {
    if !device::present() {
        println!("SKIP: this machine publishes no Metal device");
        return;
    }
    let device = Context::bind().expect("the system default device binds");
    let dir = tmp("rotted");
    let rotted = artifact(&dir, "metal", 0, precision());

    let (landed, buffers, files) = land(&device, &rotted, 1, precision());
    let said = refusal(landed, "a serving claim that does not read back");
    assert!(said.contains("tp_size"), "{said}");
    // And nothing was reserved deciding it: the rot is answered off the same
    // positioned manifest read as agreement, before any door opens.
    assert_eq!((buffers, files), (0, 0), "{said}");

    std::fs::remove_dir_all(&dir).ok();
}

/// **THE REAL-ARTIFACT ARM.** The four arms above build their fixtures with
/// the writer; this one feeds the shell an artifact `pie model import`
/// actually produced — a cuda-stamped `.zt` that exists on this box because
/// `this_box()` keys on the linked engine feature and the importing binary
/// was built without `engine-metal` (the oddity the pending `--backend`
/// ruling would make explicit). A synthetic fixture proves the gate reads a
/// stamp; a real import product proves the WRITER and the gate agree on what
/// a stamp is. Skips with a named reason when no such artifact is on disk —
/// a fixture this test cannot mint is a fixture it must not require.
#[test]
fn a_real_cuda_import_product_is_refused_the_same_way() {
    if !device::present() {
        println!("SKIP: this machine publishes no Metal device");
        return;
    }
    let Some(home) = std::env::var_os("HOME") else {
        println!("SKIP: no HOME to look under");
        return;
    };
    let real = Path::new(&home)
        .join(".pie/models/mini-l5-e16-k8")
        .join("mini-l5-e16-k8.qwen36-35b-a3b-mini-mlxu4-kv-bf16.cuda-tp1.mlxu4.zt");
    if !real.is_file() {
        println!("SKIP: no cuda-stamped import product at {}", real.display());
        return;
    }
    let device = Context::bind().expect("the system default device binds");

    let (landed, buffers, files) = land(&device, &real, 1, precision());
    let said = refusal(landed, "a real cuda-stamped import product");
    assert!(said.contains("backend"), "{said}");
    assert!(said.contains("\"cuda\"") && said.contains("\"metal\""), "{said}");
    assert_eq!((buffers, files), (0, 0), "{said}");
}

/// **TWO ABSENCES, TWO MEANINGS** — the cut both shells implement, asserted on
/// this one. An ORDINARY checkpoint states no stamp and proceeds, because that
/// is every load this tree ran before the profile existed; a REQUEST that
/// states no precision is a runtime that could not assemble the comparison, and
/// that one refuses even though the file is innocent.
#[test]
fn an_ordinary_checkpoint_passes_and_a_factless_load_does_not() {
    if !device::present() {
        println!("SKIP: this machine publishes no Metal device");
        return;
    }
    let device = Context::bind().expect("the system default device binds");
    let dir = tmp("plain");
    let plain = plain(&dir);

    let (landed, _, _) = land(&device, &plain, 1, precision());
    landed.expect("an ordinary checkpoint is not a cross-recipe artifact");

    let (landed, buffers, files) = land(&device, &plain, 1, "");
    let said = refusal(landed, "a load that states no precision");
    assert!(said.contains("states no precision"), "{said}");
    // And it refuses before the file is so much as opened, which is why the
    // check sits above the read: the fact is about the LOAD, not the file.
    assert_eq!((buffers, files), (0, 0), "{said}");

    std::fs::remove_dir_all(&dir).ok();
}
