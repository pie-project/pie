//! **The warm-read primitive (M-1) on a real Apple GPU**: an artifact mapped
//! off disk and handed to Metal through `newBufferWithBytesNoCopy` answers a
//! kernel byte-for-byte what the same kernel answers over an eagerly-copied
//! buffer of the same file — and the mapping outlives the buffer, refuses to
//! be written, and does not care whether the payload ends on a page.
//!
//! # What this file is FOR
//!
//! [`engine_metal::mapping`] and
//! [`Buffer::mapped`](engine_metal::device::Buffer::mapped) can be
//! type-checked anywhere; three of their four claims cannot be checked
//! anywhere but here. That `newBufferWithBytesNoCopy` accepts a
//! `PROT_READ`/`MAP_PRIVATE` mapping at a page-rounded length at all. That a
//! kernel reading through the resulting `MTLBuffer` sees the FILE's bytes
//! rather than a copy of them or garbage. And that a payload whose length is
//! not a page multiple — where Metal is told about a zero-filled tail no
//! caller may address — still reads its true last byte.
//!
//! The fourth claim is measured rather than asserted. `.wiki/alto/
//! streaming.md`'s residency measurement says a `StorageModeShared` page
//! WIRES on GPU touch and the pager takes none of it back, mapped or not
//! (+4.03 GiB against a 4 GiB span, free down to 0.066 GiB). This file
//! reproduces the shape of that at 1 GiB and PRINTS the `Pages wired down`
//! delta rather than asserting it: the number is global, this box runs other
//! work, and a pinned threshold would measure the box rather than the
//! primitive. The convention is the previous lanes' — observe loudly, pin
//! nothing that another process can move.
//!
//! # Gating
//!
//! Apple at compile time, `device::present()` at run time, and a printed
//! skip on a machine that publishes no GPU.
//!
//! ```text
//! cargo test -p engine-metal --release \
//!     --test a_mapped_artifact_is_the_bytes_without_the_copy -- --nocapture
//! ```
//!
//! The 1 GiB run's size is `PIE_MAP_GATE_BYTES` if it is set (rounded down
//! to a whole row), so a busy box can be told to do the semantics at 256 MiB
//! — but the wired observation is only worth reading at the default.

#![cfg(target_vendor = "apple")]

use std::io::Write;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use engine_metal::device::{self, Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use engine_metal::mapping::{self, Mapping};
use kernels_metal::Tensor;
use model_ir::Dtype;

/// **ONE DEVICE AT A TIME**, and here for a second reason beyond
/// `device_floor`'s: the wired-page reading is a GLOBAL number, and two of
/// these touching a gigabyte each at once would each report the other's.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The device, or a printed skip and `None`.
fn device_or_skip(what: &str) -> Option<Context> {
    if !device::present() {
        println!("SKIP {what}: this machine publishes no Metal device");
        return None;
    }
    Some(Context::bind().expect("the device binds"))
}

/// The row of the table this gate gathers: eight bf16 lanes, 16 bytes.
const HIDDEN: u32 = 8;
const ROW: u64 = HIDDEN as u64 * 2;

/// A temp artifact of `bytes` bytes, deleted by [`Scratch`] on drop.
///
/// The pattern is the one thing the byte comparisons rest on, and it is
/// chosen so that no two-byte element is a bf16 NaN or infinity: element `e`
/// holds `bf16((e % 128) as f32)`, every one of which is a small exact
/// non-negative float. A gather that moves bits verbatim therefore lands
/// bits this test can compare with `==` — a canonicalising NaN would be the
/// one way a byte-level assertion could lie, and this pattern has none.
struct Scratch(std::path::PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl Scratch {
    fn of(name: &str, bytes: u64) -> Scratch {
        let path = std::env::temp_dir()
            .join(format!("pie-map-gate-{}-{name}", std::process::id()));
        let scratch = Scratch(path);
        // The pattern repeats every 128 elements, so it is built ONCE as a
        // 256-byte unit and tiled — a gigabyte artifact written a megabyte
        // at a time, never a gigabyte of `Vec` beside the mapping it is
        // about to become, and never a per-byte division either. A megabyte
        // is a whole number of units, so every chunk starts in phase and the
        // short last one is a prefix of the same chunk.
        const CHUNK: u64 = 1 << 20;
        let unit: Vec<u8> = (0..128u32).flat_map(|at| bf16(at as f32)).collect();
        let chunk: Vec<u8> = unit
            .iter()
            .copied()
            .cycle()
            .take(CHUNK as usize)
            .collect();
        let mut file = std::fs::File::create(&scratch.0).expect("a scratch artifact");
        let mut done = 0u64;
        while done < bytes {
            let want = CHUNK.min(bytes - done);
            file.write_all(&chunk[..want as usize])
                .expect("the pattern lands");
            done += want;
        }
        file.sync_all().expect("the artifact is on disk");
        scratch
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

/// One f32 as the two bytes of its bf16 truncation, little-endian.
fn bf16(value: f32) -> [u8; 2] {
    ((value.to_bits() >> 16) as u16).to_le_bytes()
}

fn as_bytes_i32(ids: &[i32]) -> Vec<u8> {
    ids.iter().flat_map(|it| it.to_le_bytes()).collect()
}

/// Global `Pages wired down`, in bytes, off `vm_stat`.
///
/// The authority signal the residency measurement used, and for its reason:
/// mapped wired pages are attributed to the kernel rather than to this
/// process, so a process-footprint reading would MISS the whole effect.
/// `None` when `vm_stat` is not there or does not parse, which is a reason
/// to print nothing rather than to fail a correctness gate.
fn wired() -> Option<u64> {
    let said = std::process::Command::new("vm_stat").output().ok()?;
    let said = String::from_utf8_lossy(&said.stdout);
    let page: u64 = said
        .lines()
        .next()?
        .split("page size of ")
        .nth(1)?
        .split(' ')
        .next()?
        .parse()
        .ok()?;
    let pages: u64 = said
        .lines()
        .find(|line| line.starts_with("Pages wired down:"))?
        .split(':')
        .nth(1)?
        .trim()
        .trim_end_matches('.')
        .parse()
        .ok()?;
    Some(pages * page)
}

fn gib(bytes: i128) -> String {
    format!("{:+.3} GiB", bytes as f64 / (1u64 << 30) as f64)
}

/// Gather `ids` out of `table` with `layout.embed`, and answer the output's
/// raw bytes.
///
/// **WHY THIS ENTRY.** It is `device_floor`'s own end-to-end point, so it is
/// already proved to compile, to bind its five arguments where the shader
/// says, and to compute a permutation — which means anything this file finds
/// is about the BUFFER and not about the kernel. And it is a gather, so a
/// list of ids one row apart per page is a kernel that touches every page of
/// the artifact, which is exactly the shape the wiring measurement needs.
fn gather(
    device: &Context,
    pipelines: &Pipelines,
    table: &Buffer,
    rows: u32,
    ids: &[i32],
) -> Vec<u8> {
    let handles = Handles::new();
    let count = u32::try_from(ids.len()).expect("an id list a fire can name");

    let mut id_store =
        Buffer::zeroed(device, 4 * u64::from(count)).expect("the ids reserve");
    id_store
        .write(0, &as_bytes_i32(ids))
        .expect("the ids land");
    let out_bytes = u64::from(count) * ROW;
    let out = Buffer::zeroed(device, out_bytes).expect("the output reserves");

    let ids_h = handles
        .bind(&id_store, 0, 4 * u64::from(count))
        .expect("ids");
    // The table handle is minted over the artifact's TRUE length — the whole
    // point of `Buffer::mapped` reporting the file's size rather than the
    // page-rounded span it told Metal about.
    let table_h = handles
        .bind(table, 0, u64::from(rows) * ROW)
        .expect("table");
    let out_h = handles.bind(&out, 0, out_bytes).expect("out");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, &handles);
        kernels_metal::layout::embed(
            &sink,
            Tensor::new(ids_h, count, 1, Dtype::I32),
            Tensor::new(table_h, rows, HIDDEN, Dtype::Bf16),
            rows,
            Tensor::new(out_h, count, HIDDEN, Dtype::Bf16),
        )
        .expect("the embed encodes");
    }
    frame.commit().expect("the fire completes");

    let mut got = vec![0u8; out_bytes as usize];
    out.read(0, &mut got).expect("the output reads back");
    got
}

/// **THE GATE.** One kernel over an artifact's own mapped pages answers
/// exactly what the same kernel answers over an eager copy of the same file,
/// and the wired-page cost of the mapped touch is printed.
#[test]
fn a_kernel_over_a_mapped_artifact_answers_what_a_kernel_over_a_copy_answers() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the zero-copy bind") else {
        return;
    };
    let pipelines = Pipelines::new();

    let bytes = std::env::var("PIE_MAP_GATE_BYTES")
        .ok()
        .and_then(|it| it.parse::<u64>().ok())
        .unwrap_or(1 << 30)
        / ROW
        * ROW;
    let rows = u32::try_from(bytes / ROW).expect("a table a fire can name");

    // One id per page of the artifact, so the fire faults EVERY page — the
    // touch the residency measurement is about.
    let page = mapping::page() as u64;
    let ids: Vec<i32> = (0..bytes / page)
        .map(|at| i32::try_from(at * page / ROW).expect("a row a gather can name"))
        .collect();

    let artifact = Scratch::of("whole", bytes);
    let map = Mapping::of(artifact.path()).expect("the artifact maps");
    assert_eq!(map.len(), bytes, "the mapping is the file, to the byte");
    assert_eq!(
        map.backing(),
        Some(bytes),
        "and the file behind it is still that size"
    );

    let before = wired();
    let table = Buffer::mapped(&device, Arc::clone(&map)).expect("the artifact binds");
    assert!(table.is_mapped(), "the reservation knows what it is");
    assert_eq!(
        table.bytes(),
        bytes,
        "the reservation reports the TRUE length, not the page-rounded span"
    );
    let bound = wired();

    let from_map = gather(&device, &pipelines, &table, rows, &ids);
    let touched = wired();

    // The honest half of the claim, printed and not pinned: what a GPU touch
    // of a mapped span costs in pages nothing will reclaim.
    if let (Some(before), Some(bound), Some(touched)) = (before, bound, touched) {
        println!(
            "OBSERVED wired: bind {} over {:.3} GiB mapped, GPU touch of every page {} \
             (global `Pages wired down`, this box, not an assertion)",
            gib(i128::from(bound) - i128::from(before)),
            bytes as f64 / (1u64 << 30) as f64,
            gib(i128::from(touched) - i128::from(bound)),
        );
    }

    // The control: the identical bytes, eagerly copied into a buffer Metal
    // allocated, through the identical kernel.
    let mut copy = Buffer::zeroed(&device, bytes).expect("the eager control reserves");
    copy.write(0, &map).expect("the artifact copies in");
    let from_copy = gather(&device, &pipelines, &copy, rows, &ids);

    assert_eq!(
        from_map.len(),
        from_copy.len(),
        "the two gathers are the same shape"
    );
    assert!(
        from_map == from_copy,
        "the mapped artifact and its eager copy answer the same kernel differently"
    );

    // And neither is vacuously right: every gathered row IS the artifact's
    // bytes at the row the id names, read back off the mapping on the CPU.
    for (at, id) in ids.iter().enumerate() {
        let want = usize::try_from(*id as u64 * ROW).expect("an offset in the artifact");
        let got = at * ROW as usize;
        assert_eq!(
            &from_map[got..got + ROW as usize],
            &map[want..want + ROW as usize],
            "row {at} (id {id}) is not the artifact's own bytes"
        );
    }
    println!(
        "{} rows gathered off {:.3} GiB of mapped artifact, byte-identical to the eager copy",
        ids.len(),
        bytes as f64 / (1u64 << 30) as f64
    );
}

/// **The alignment edge**: a payload whose length is not a page multiple
/// binds, and the kernel reads its true last row.
///
/// Metal is told about `round_up(len, page)` bytes — a tail of zero-fill
/// that belongs to no file — and everything above the bind is held to `len`.
/// The proof is two-sided: the LAST row of the true payload gathers
/// correctly (so the tail did not displace anything), and a handle over one
/// byte past the payload is refused (so the tail is not addressable).
#[test]
fn a_payload_that_does_not_end_on_a_page_still_reads_its_true_length() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the alignment edge") else {
        return;
    };
    let pipelines = Pipelines::new();

    // Two pages and a bit, cut to a whole row: deliberately not a page
    // multiple, and deliberately more than one page so the tail is a partial
    // LAST page rather than the whole mapping.
    let page = mapping::page() as u64;
    let bytes = (page * 2 + page / 3) / ROW * ROW;
    assert_ne!(bytes % page, 0, "this gate is about a payload that is not aligned");
    let rows = u32::try_from(bytes / ROW).expect("a table a fire can name");

    let artifact = Scratch::of("odd", bytes);
    let map = Mapping::of(artifact.path()).expect("the artifact maps");
    assert_eq!(map.span() as u64, bytes.next_multiple_of(page));
    assert!(map.span() as u64 > map.len(), "this artifact needs the tail");

    let table = Buffer::mapped(&device, Arc::clone(&map)).expect("the artifact binds");
    assert_eq!(
        table.bytes(),
        bytes,
        "Metal was told about the tail and no caller above the bind was"
    );

    // The last row of the true payload, the first, and one in the partial
    // page — the three places an over-stated length could go wrong.
    let last = i32::try_from(rows - 1).expect("a row");
    let in_the_tail_page = i32::try_from(u64::from(rows) - (bytes % page) / ROW / 2 - 1)
        .expect("a row inside the partial last page");
    let ids = [last, 0, in_the_tail_page, last];
    let got = gather(&device, &pipelines, &table, rows, &ids);
    for (at, id) in ids.iter().enumerate() {
        let want = usize::try_from(*id as u64 * ROW).expect("an offset in the artifact");
        let seen = at * ROW as usize;
        assert_eq!(
            &got[seen..seen + ROW as usize],
            &map[want..want + ROW as usize],
            "row {at} (id {id}) of an unaligned payload is not the artifact's bytes"
        );
    }

    // The other side: the tail is real memory to Metal and unreachable here.
    let handles = Handles::new();
    handles
        .bind(&table, 0, bytes)
        .expect("the whole true payload binds");
    let fault = handles
        .bind(&table, 0, bytes + 1)
        .expect_err("one byte past the payload is the zero-fill, and is not the artifact's");
    println!("the tail is not addressable: {fault}");
}

/// A mapped reservation is `PROT_READ`, and says so instead of faulting the
/// process.
#[test]
fn a_mapped_reservation_refuses_to_be_written() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the read-only refusal") else {
        return;
    };
    let bytes = mapping::page() as u64;
    let artifact = Scratch::of("readonly", bytes);
    let map = Mapping::of(artifact.path()).expect("the artifact maps");
    let mut table = Buffer::mapped(&device, map).expect("the artifact binds");

    let said = table
        .write(0, &[0u8; 16])
        .expect_err("a write through PROT_READ pages is refused, not attempted")
        .to_string();
    assert!(said.contains("read-only"), "the refusal says why: {said}");
    let said = table
        .zero_span(0, 16)
        .expect_err("and so is a zero")
        .to_string();
    assert!(said.contains("read-only"), "the refusal says why: {said}");

    // A READ is the whole point and still works.
    let mut got = [0u8; 16];
    table.read(0, &mut got).expect("the artifact reads");
    assert_eq!(got, map_first_16(artifact.path()));
}

fn map_first_16(path: &std::path::Path) -> [u8; 16] {
    let mut got = [0u8; 16];
    got.copy_from_slice(&std::fs::read(path).expect("the artifact reads")[..16]);
    got
}

/// **Lifetime**: the mapping goes when the LAST buffer over it goes, and not
/// before — and there is no way to spell one that has already gone.
///
/// The `Arc` strong count is the observable. A `Buffer::mapped` holds one; a
/// clone of that buffer holds a second (the same sentence the type makes
/// about its retain); dropping either leaves the mapping alive for the
/// other, and dropping the last returns the count to the one this test
/// holds — at which point dropping THAT is the `munmap`. Bind-after-drop
/// needs no test because it needs no refusal: the only door onto a zero-copy
/// reservation takes an `Arc<Mapping>` and keeps it, so a buffer whose
/// mapping is gone cannot be constructed.
#[test]
fn the_mapping_outlives_every_buffer_over_it_and_no_longer() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the mapping's lifetime") else {
        return;
    };
    let bytes = mapping::page() as u64 * 4;
    let artifact = Scratch::of("lifetime", bytes);
    let map = Mapping::of(artifact.path()).expect("the artifact maps");
    assert_eq!(Arc::strong_count(&map), 1, "this test holds the only one");

    let table = Buffer::mapped(&device, Arc::clone(&map)).expect("the artifact binds");
    assert_eq!(Arc::strong_count(&map), 2, "the reservation holds the mapping");

    let second = table.clone();
    assert_eq!(
        Arc::strong_count(&map),
        3,
        "a clone of the reservation is a second owner of the mapping, not a copy of it"
    );

    drop(table);
    assert_eq!(
        Arc::strong_count(&map),
        2,
        "the mapping is still alive for the clone that still names it"
    );
    // And the clone still READS, which is the claim the count is standing in
    // for: the pages are there after the first owner went.
    let mut got = [0u8; 16];
    second.read(0, &mut got).expect("the survivor still reads");

    drop(second);
    assert_eq!(
        Arc::strong_count(&map),
        1,
        "the last reservation released the mapping back to this test"
    );

    // The unmap itself: after this statement nothing in the process holds
    // the span, and the only honest observation of that is that the drop
    // runs cleanly under the harness rather than a count nobody publishes.
    drop(map);
}
