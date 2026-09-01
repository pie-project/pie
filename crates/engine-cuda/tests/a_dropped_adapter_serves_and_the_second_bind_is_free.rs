//! **A-3's DEVICE HALF AND A-4, ON REAL HARDWARE** (alto adapter §3.3, §6.1).
//!
//! # What this file adds to the host gates next door
//!
//! `a_shared_adapter_is_one_slot_and_one_load.rs` judges the whole residency
//! table with no GPU in the machine — which slot, whose slot, what is
//! reclaimed, what is refused — by handing `Adapters::bind` a recorder in
//! place of the landing. That is the right shape for every claim that is a
//! host decision, and it leaves exactly two claims it cannot make, because
//! both are about bytes that really arrived on a device:
//!
//! ```text
//! (a) ONE DEVICE COPY  — A-3. Two binds naming one mounted adapter move the
//!     weight store's digest ONCE. The second bind answers the first one's
//!     slot, writes nothing, and the digest is the same number afterwards.
//! (b) ITS OWN SLOT     — a byte-seeded instance shares nothing: a different
//!     slot, and the digest moves again.
//! (c) A FILE DROP SERVES — A-4. An adapter directory written into the mount
//!     AFTER the shell loaded binds and lands with no restart, no re-mount
//!     and no registration verb. `register_operator_adapters` demotes to
//!     cache warming, exactly as §3.3 says.
//! ```
//!
//! The digest is the observable in every one of them, and it is the right one
//! for the reason `a_warm_boot_reads_the_weights_it_wrote.rs` gives about the
//! same number: "the same slot was answered" is a claim about a host table,
//! and "one device copy" is a claim about the device — so the device is what
//! gets asked.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 \
//!   --test a_dropped_adapter_serves_and_the_second_bind_is_free -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_cuda::{AdapterSource, BankSeat, Boot, Graphs, Shell};
use model_compiler::Budget;
use model_dsl::Platform;

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// One shell at a time per process — `kernels-cuda`'s scratch slabs are
/// process-global and keyed by name (`serve_smoke.rs` argues it whole).
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

// ── the mount ────────────────────────────────────────────────────────────

/// This test's own directory, unique per process and per nanosecond — the
/// convention the rest of this crate's gates use.
fn scratch(what: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_nanos())
        .unwrap_or(0);
    let at = std::env::temp_dir().join(format!("pie-hotadd-{what}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&at).expect("a scratch directory");
    at
}

/// f32 to bf16, round-to-nearest-even — the loader's own conversion, stated
/// here for the reason `adapter_banks.rs` states it: a truncating fixture
/// would be writing a slightly different adapter than the one it describes.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// What the load's banks say one role's `[layers, rank, hidden]` source is.
fn geometry(seats: &[BankSeat], role: &str) -> (u64, u64, u64) {
    let banks: Vec<&BankSeat> = seats
        .iter()
        .filter(|seat| engine_cuda::role_of(&seat.name) == role)
        .collect();
    let seat = banks.first().expect("the SKU declares this role's banks");
    (
        banks.len() as u64,
        seat.rows.min(seat.cols),
        seat.rows.max(seat.cols),
    )
}

/// **WRITE ONE ADAPTER DIRECTORY INTO THE MOUNT.**
///
/// A manifest and two plane files, in the orientations §6.3's statute fixes:
/// `A` rank-major `[layers, rank, hidden]`, `B` out-major
/// `[layers, hidden, rank]`. `amplitude` scales `B`'s contents, so a zero one
/// writes a correction that is exactly zero and every adapter written here is
/// still a DIFFERENT identity from every other — the numbers are what the
/// digest reads, and the stamp is what the slot table keys on.
fn write_adapter(mount: &Path, name: &str, seats: &[BankSeat], amplitude: f32) {
    let dir = mount.join(name);
    std::fs::create_dir_all(&dir).expect("an adapter directory");
    let (layers, rank, hidden) = geometry(seats, "lora_a");
    let plane = |salt: u32, amp: f32, count: u64| -> Vec<u8> {
        (0..count)
            .flat_map(|at| {
                let mixed = (at as u32).wrapping_mul(2_654_435_761).wrapping_add(salt);
                let value = ((mixed % 2_000) as f32 / 1_000.0 - 1.0) * amp;
                bf16_bits(value).to_le_bytes()
            })
            .collect()
    };
    std::fs::write(
        dir.join("lora_a.bin"),
        plane(0x0a0a_a0a0, 0.05, layers * rank * hidden),
    )
    .expect("the A plane writes");
    std::fs::write(
        dir.join("lora_b.bin"),
        plane(0x0b0b_b0b0, amplitude, layers * hidden * rank),
    )
    .expect("the B plane writes");
    std::fs::write(
        dir.join("adapter.toml"),
        format!(
            "rank = {rank}\n\n\
             [[plane]]\nrole = \"lora_a\"\nfile = \"lora_a.bin\"\nlayout = \"rank_major\"\n\n\
             [[plane]]\nrole = \"lora_b\"\nfile = \"lora_b.bin\"\nlayout = \"out_major\"\n"
        ),
    )
    .expect("the manifest writes");
}

/// One instance's full-capacity planes, built by hand — the private-adapter
/// path, and the shape `register_adapter` has always taken.
fn own_planes(seats: &[BankSeat]) -> Vec<(String, Vec<u8>)> {
    seats
        .iter()
        .map(|seat| {
            let count = usize::try_from(seat.slot).expect("a slot fits this host") / 2;
            let bytes = (0..count)
                .flat_map(|at| {
                    let value = ((at % 97) as f32 / 97.0 - 0.5) * 0.25;
                    bf16_bits(value).to_le_bytes()
                })
                .collect();
            (seat.name.clone(), bytes)
        })
        .collect()
}

// ── the claims ───────────────────────────────────────────────────────────

/// **(a) + (b) + (c), IN ONE SHELL.**
///
/// One load, because a load is thirty seconds and every claim below is about
/// the same weight store — and because A-4's whole sentence is "without
/// restart", which is only sayable inside one process's lifetime.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn one_mount_two_binds_one_copy_and_a_dropped_file_serves() {
    let _guard = serialized();
    let Some(mut shell) = ready("one_mount_two_binds_one_copy_and_a_dropped_file_serves") else {
        return;
    };
    let seats = shell.bank_seats();
    assert!(
        !seats.is_empty(),
        "this SKU declares adapter banks; a load with none has nowhere to put one"
    );

    // The mount, stated AFTER the load — §3.3's whole posture: where the
    // shared adapters live is the deployment's, not the bake's.
    let mount = scratch("mount");
    write_adapter(&mount, "alice-v2", &seats, 0.5);
    shell.mount_adapters(Some(mount.clone()));

    let base = shell.weight_digest().expect("the store reads back");

    // ── (a) ONE DEVICE COPY. The first bind lands; the second joins it.
    let first = shell
        .bind_adapter(AdapterSource::Shared { name: "/alice-v2" })
        .expect("the mount serves the adapter it holds");
    assert!(first.shared, "a name in the mount is a shared source");
    assert!(first.landed, "the first bind is the one that pays");
    let after_first = shell.weight_digest().expect("the store reads back");
    assert_ne!(
        after_first, base,
        "the first bind landed nothing: the digest of the whole weight store is \
         unchanged, so the bank was never written and every claim below would be \
         about an absence"
    );

    let second = shell
        .bind_adapter(AdapterSource::Shared { name: "/alice-v2" })
        .expect("a second instance names the same adapter");
    assert_eq!(
        second.slot, first.slot,
        "two instances naming ONE blob must land on ONE slot (alto adapter §3.3): \
         the whole point of keying residency by blob identity is that the second \
         tenant of an adapter costs the device nothing"
    );
    assert!(
        !second.landed,
        "the second bind paid a landing; it should have joined the one already \
         resident"
    );
    assert_eq!(
        shell.weight_digest().expect("the store reads back"),
        after_first,
        "the second bind MOVED THE DEVICE BYTES. It answered the first bind's slot, \
         so anything it wrote was a second copy of what was already there — which is \
         exactly the H2D §3.3's keying exists to not pay twice"
    );
    assert_eq!(
        shell.adapters().slots().resident().len(),
        1,
        "one adapter, one resident slot"
    );
    assert_eq!(
        shell.adapters().slots().refs(first.slot),
        2,
        "two live binds hold it, and the slot is pinned until both give it back"
    );

    // ── (b) A BYTE-SEEDED INSTANCE IS NOBODY'S NEIGHBOUR.
    let built = own_planes(&seats);
    let planes: Vec<engine_cuda::AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| engine_cuda::AdapterPlane {
            bank: bank.as_str(),
            bytes,
        })
        .collect();
    let private = shell
        .bind_adapter(AdapterSource::Own {
            instance: 77,
            planes: &planes,
        })
        .expect("a private adapter lands from the caller's own bytes");
    assert!(!private.shared, "bytes are not a name in the mount");
    assert!(private.landed, "a private adapter always pays its own landing");
    assert_ne!(
        private.slot, first.slot,
        "a byte-seeded instance gets a slot of its OWN (§3.3): content-hash dedup \
         across private adapters is a later optimization, and sharing one here \
         would put one tenant's fine-tune under another tenant's rows"
    );
    let after_private = shell.weight_digest().expect("the store reads back");
    assert_ne!(
        after_private, after_first,
        "the private landing wrote nothing to the device"
    );
    assert_eq!(
        shell.adapters().slots().resident().len(),
        2,
        "two identities, two slots"
    );

    // ── (c) A-4: HOT-ADD IS A FILE DROP. The shell has been up and serving
    //    since before this directory existed; nothing is restarted, nothing is
    //    re-mounted, and no verb is called to announce it.
    write_adapter(&mount, "bob-v1", &seats, 0.25);
    let dropped = shell
        .bind_adapter(AdapterSource::Shared { name: "/bob-v1" })
        .expect(
            "an adapter written into the mount while the box serves must bind: alto \
             adapter §3.3 makes adding a LoRA a file drop, so a refusal here would \
             mean the catalog was snapshotted at boot after all",
        );
    assert!(dropped.landed, "a name nobody has bound before pays a landing");
    assert_ne!(
        dropped.slot, first.slot,
        "a different adapter is a different identity"
    );
    assert_ne!(
        dropped.slot, private.slot,
        "and it is not the private one's either"
    );
    assert_ne!(
        shell.weight_digest().expect("the store reads back"),
        after_private,
        "the hot-added adapter's bytes never reached the device"
    );
    assert_eq!(
        shell.adapters().slots().resident().len(),
        3,
        "three identities, three slots — and the mount was never re-stated"
    );

    // ── AND A NAME THAT IS NOT THERE IS STILL A REFUSAL. The same directory
    //    that just grew an adapter does not grow one that was never written:
    //    a hot-add is a file, so an absence is an absence.
    let why = shell
        .bind_adapter(AdapterSource::Shared {
            name: "/carol-v9",
        })
        .expect_err("a name nobody wrote is not in the mount");
    let said = why.to_string();
    assert!(
        said.contains("carol-v9"),
        "the refusal names the adapter: {said}"
    );

    // Give the binds back, so the slots this test pinned are reclaimable and
    // the next gate in the process starts from a table nobody holds.
    shell.release_adapter(first);
    shell.release_adapter(second);
    shell.release_adapter(private);
    shell.release_adapter(dropped);
    assert_eq!(
        shell.adapters().slots().refs(first.slot),
        0,
        "a released bind holds nothing"
    );
    let _ = std::fs::remove_dir_all(&mount);
}

// ── the load ─────────────────────────────────────────────────────────────

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .find(|path| path.join("tokenizer.json").exists())
}

fn container(snapshot: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

/// A loaded shell, or `None` and a sentence saying what was missing —
/// `adapter_banks.rs`'s fixture, with the tokenizer dropped because nothing
/// here decodes anything.
fn ready(what: &str) -> Option<Shell> {
    if !engine_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no Qwen3.5-0.8B snapshot in the hugging face cache \
             (set PIE_SMOKE_SNAPSHOT)"
        );
        return None;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };
    let trace = models::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let seats = trace
        .params
        .iter()
        .filter(|param| param.source == model_ir::ParamSource::Registered)
        .map(|param| param.shape.first().copied().unwrap_or(0))
        .min()
        .expect("the SKU declares adapter banks");
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = models::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget {
            max_adapters: u32::try_from(seats).expect("a capacity fits a u32"),
            ..Budget::new(4, 256)
        },
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: None,
    })
    .expect("the shell loads");
    eprintln!("{SKU} loaded — {} banks, {seats} slots", shell.banks().len());
    Some(shell)
}
