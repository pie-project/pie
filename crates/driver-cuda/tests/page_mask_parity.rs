//! Behavioural parity with the C++ `FirePageMask` / `prepare_page_mask_capture`.
//!
//! The oracle in `tests/oracle/page_mask/` compiles the real
//! `model/attn_page_mask.cu` against the real `model/hook_sideband_arena.cpp`,
//! replaces the four CUDA entry points and the compaction kernel, and drives
//! them over eleven fire geometries and four scripted scenarios. This test
//! replays the same scripts against the port and requires equal transcripts.
//!
//! Run `tests/oracle/page_mask/run.sh` to regenerate [`GOLDEN_FNV1A64`]. The
//! pinned value is the **C++'s**; a golden regenerated from the port would only
//! prove the port agrees with itself.
//!
//! # What the transcript is really checking
//!
//! Two independent call paths compute the same carve, and one of them bakes
//! its answer into a captured CUDA graph. A one-byte disagreement means the
//! replayed graph writes the compacted page table where the attention does not
//! read it — no error, no crash, just attention over stale pages. So every
//! shape reports both carves and an `agree` row, and the compaction reports
//! which carved buffer landed in which kernel parameter, because a buffer
//! carved correctly and passed in the wrong slot is invisible to a layout-only
//! check.
//!
//! Offsets, not addresses: the property is the layout. A golden full of
//! malloc's return values would be a golden about malloc.

use std::ffi::c_void;
use std::fmt::Write as _;

use driver_cuda::fire::page_mask::{
    FireGeometry, FirePageMask, MaskError, MaskOps, MaskSlotLayout, PageMaskCapturePlan,
    prepare_page_mask_capture,
};
use driver_cuda::fire::sideband_arena::{DeviceMemory, Region, SidebandArena};

/// FNV-1a 64 of the C++ oracle's transcript.
const GOLDEN_FNV1A64: u64 = 0xa084_1dc3_318a_057b;

/// Rows the transcript must contain, so a truncated sweep cannot pass.
const GOLDEN_ROWS: usize = 58;

const SEP: char = '\u{1f}';

/// The oracle's slab allocator, reproduced. Bump-allocates out of one buffer
/// so every pointer can be reported as an offset from a known base.
struct Slab {
    bytes: Vec<u8>,
    next: usize,
    allocs: u32,
    events: Vec<String>,
    recording: bool,
}

impl Slab {
    fn new() -> Self {
        Self {
            bytes: vec![0u8; 64 << 20],
            next: 0,
            allocs: 0,
            events: Vec::new(),
            recording: false,
        }
    }

    fn record(&mut self, e: String) {
        if self.recording {
            self.events.push(e);
        }
    }

    fn drain(&mut self) -> String {
        if self.events.is_empty() {
            return "-".to_owned();
        }
        let joined = self.events.join(",");
        self.events.clear();
        joined
    }
}

impl DeviceMemory for Slab {
    fn alloc(&mut self, bytes: usize) -> Option<*mut c_void> {
        let start = self.next.next_multiple_of(256);
        if start + bytes > self.bytes.len() {
            return None;
        }
        self.next = start + bytes;
        self.allocs += 1;
        self.record(format!("malloc:{bytes}"));
        Some(unsafe { self.bytes.as_mut_ptr().add(start) }.cast::<c_void>())
    }

    fn free(&mut self, ptr: *mut c_void) {
        if !ptr.is_null() {
            self.record("free".to_owned());
        }
    }

    fn synchronize(&mut self) -> bool {
        self.record("sync".to_owned());
        true
    }
}

/// The compaction kernel and the memset, replaced by recorders.
#[derive(Default)]
struct Ops {
    memset: Option<(usize, u8, usize)>,
    compact: Option<Compact>,
}

struct Compact {
    keep: usize,
    counts: usize,
    out_indices: usize,
    out_indptr: usize,
    out_last_lens: usize,
    keep_stride: u32,
    num_requests: i32,
    inputs: (usize, usize, usize),
}

impl MaskOps for Ops {
    fn memset_async(&mut self, dst: *mut u8, value: u8, bytes: usize) {
        self.memset = Some((dst as usize, value, bytes));
    }

    fn compact_page_csr(
        &mut self,
        page_indices_in: *const u32,
        page_indptr_in: *const u32,
        last_page_lens_in: *const u32,
        keep: *const u8,
        scratch_counts: *mut u32,
        keep_stride: u32,
        num_requests: i32,
        page_indices_out: *mut u32,
        page_indptr_out: *mut u32,
        last_page_lens_out: *mut u32,
    ) {
        self.compact = Some(Compact {
            keep: keep as usize,
            counts: scratch_counts as usize,
            out_indices: page_indices_out as usize,
            out_indptr: page_indptr_out as usize,
            out_last_lens: page_last(last_page_lens_out),
            keep_stride,
            num_requests,
            inputs: (
                page_indices_in as usize,
                page_indptr_in as usize,
                last_page_lens_in as usize,
            ),
        });
    }
}

fn page_last(p: *mut u32) -> usize {
    p as usize
}

/// The oracle's `at()`: a pointer as `+N` from the slot base, or `null`.
fn at(base: Option<usize>, p: usize) -> String {
    if p == 0 {
        return "null".to_owned();
    }
    match base {
        None => "?".to_owned(),
        Some(b) if p < b => "before_base".to_owned(),
        Some(b) => format!("+{}", p - b),
    }
}

/// The C++ `what()` for an error, so refusals compare as the C++ reports them.
fn outcome(e: MaskError) -> String {
    e.cpp_message()
        .map_or_else(|| "throw:?".to_owned(), |m| format!("throw:{m}"))
}

/// The oracle's eleven fire geometries: realistic decode/prefill shapes plus
/// every way the CSR validation can reject one.
fn shapes() -> Vec<(&'static str, Vec<u32>)> {
    vec![
        ("single_page", vec![0, 1]),
        ("single_request_deep", vec![0, 129]),
        ("uniform_decode_4", vec![0, 8, 16, 24, 32]),
        ("ragged_decode_5", vec![0, 3, 3, 40, 41, 97]),
        ("leading_empty", vec![0, 0, 7]),
        ("trailing_empty", vec![0, 7, 7]),
        (
            "wide_batch_16",
            vec![0, 2, 5, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 90, 104, 119, 135],
        ),
        ("all_empty", vec![0, 0, 0]),
        ("zero_total", vec![0]),
        ("non_monotonic", vec![0, 9, 4, 12]),
        ("end_past_total", vec![0, 5, 99, 12]),
    ]
}

/// Script 1 — the carve, from both call paths, plus their agreement.
fn script_carve(out: &mut String) {
    for (name, csr) in shapes() {
        let mut mem = Slab::new();
        let mut arena = SidebandArena::new();
        let geometry = FireGeometry::new(&csr);

        let plan: Option<PageMaskCapturePlan> = geometry.ok().and_then(|g| {
            prepare_page_mask_capture(&mut arena, &mut mem, g).ok()
        });
        let mut base = plan.map(|p| p.out_indices as usize);

        let (p_ok, p_requests, p_stride, p_idx, p_indptr, p_lens, p_keep) = plan.map_or_else(
            || {
                (
                    0,
                    0,
                    0,
                    "null".to_owned(),
                    "null".to_owned(),
                    "null".to_owned(),
                    "null".to_owned(),
                )
            },
            |p| {
                (
                    1,
                    p.num_requests,
                    p.stride,
                    at(base, p.out_indices as usize),
                    at(base, p.out_indptr as usize),
                    at(base, p.out_last_lens as usize),
                    at(base, p.keep as usize),
                )
            },
        );
        writeln!(
            out,
            "carve{SEP}{name}{SEP}plan{SEP}{p_ok}{SEP}{p_requests}{SEP}{p_stride}\
             {SEP}{p_idx}{SEP}{p_indptr}{SEP}{p_lens}{SEP}{p_keep}"
        )
        .unwrap();

        let built = geometry
            .and_then(|g| FirePageMask::new(true, Some(g), Some(&mut arena), &mut mem));
        let (active, requests, stride, idx, indptr, lens, keep, result) = match built {
            Ok(mut mask) => {
                if base.is_none() {
                    base = Some(mask.page_indices() as usize);
                }
                let idx = at(base, mask.page_indices() as usize);
                let indptr = at(base, mask.page_indptr() as usize);
                let lens = at(base, mask.last_page_lens() as usize);
                let active = i32::from(mask.active());
                let (keep, requests, stride) = mask.sink().map_or_else(
                    || ("null".to_owned(), 0, 0),
                    |s| (at(base, s.keep as usize), s.num_requests, s.stride),
                );
                mask.release(&mut arena);
                (
                    active,
                    requests,
                    stride,
                    idx,
                    indptr,
                    lens,
                    keep,
                    "ok".to_owned(),
                )
            }
            Err(e) => (
                0,
                0,
                0,
                "null".to_owned(),
                "null".to_owned(),
                "null".to_owned(),
                "null".to_owned(),
                outcome(e),
            ),
        };
        writeln!(
            out,
            "carve{SEP}{name}{SEP}fire{SEP}{active}{SEP}{requests}{SEP}{stride}\
             {SEP}{idx}{SEP}{indptr}{SEP}{lens}{SEP}{keep}{SEP}{result}"
        )
        .unwrap();

        let agree = p_ok == active
            && p_requests == requests
            && p_stride == stride
            && p_idx == idx
            && p_indptr == indptr
            && p_lens == lens
            && p_keep == keep;
        writeln!(out, "carve{SEP}{name}{SEP}agree{SEP}{}", i32::from(agree)).unwrap();

        arena.destroy(&mut mem);
    }
}

/// Script 2 — the layer loop.
fn script_layer_loop(out: &mut String) {
    let csr = vec![0u32, 3, 3, 40, 41, 97];
    let g = FireGeometry::new(&csr).unwrap();
    let mut mem = Slab::new();
    let mut arena = SidebandArena::new();
    let mut ops = Ops::default();
    let mut mask = FirePageMask::new(true, Some(g), Some(&mut arena), &mut mem).unwrap();
    let base = Some(mask.page_indices() as usize);

    // The oracle passes real (stack) arrays as the fire's device CSR; the port
    // only forwards them, so any three distinct non-null pointers do.
    let in_idx = [0u32; 128];
    let in_indptr = [0u32; 8];
    let in_lens = [0u32; 8];

    for layer in 0..3u32 {
        ops.memset = None;
        mask.begin_layer(&mut ops);
        let (dst, value, bytes) = ops.memset.unwrap();
        writeln!(
            out,
            "loop{SEP}layer{layer}{SEP}begin{SEP}{}{SEP}{value}{SEP}{bytes}{SEP}{}{SEP}{}",
            at(base, dst),
            i32::from(mask.written_for(layer)),
            i32::from(mask.written_for(layer + 1)),
        )
        .unwrap();

        if layer == 1 {
            mask.sink().unwrap().written_layer = Some(layer);
        }

        writeln!(
            out,
            "loop{SEP}layer{layer}{SEP}written{SEP}{}{SEP}{}{SEP}{}",
            i32::from(mask.written_for(0)),
            i32::from(mask.written_for(1)),
            i32::from(mask.written_for(2)),
        )
        .unwrap();

        if mask.written_for(layer) {
            ops.compact = None;
            mask.compact(
                &mut ops,
                in_idx.as_ptr(),
                in_indptr.as_ptr(),
                in_lens.as_ptr(),
                5,
            )
            .unwrap();
            let c = ops.compact.as_ref().unwrap();
            let inputs_forwarded = c.inputs
                == (
                    in_idx.as_ptr() as usize,
                    in_indptr.as_ptr() as usize,
                    in_lens.as_ptr() as usize,
                );
            writeln!(
                out,
                "loop{SEP}layer{layer}{SEP}compact{SEP}1{SEP}{}{SEP}{}{SEP}{}{SEP}{}\
                 {SEP}{}{SEP}{}{SEP}{}{SEP}{}",
                at(base, c.keep),
                at(base, c.counts),
                at(base, c.out_indices),
                at(base, c.out_indptr),
                at(base, c.out_last_lens),
                c.keep_stride,
                c.num_requests,
                i32::from(inputs_forwarded),
            )
            .unwrap();
        }
    }

    ops.compact = None;
    let mismatch = mask
        .compact(
            &mut ops,
            in_idx.as_ptr(),
            in_indptr.as_ptr(),
            in_lens.as_ptr(),
            4,
        )
        .map_or_else(outcome, |()| "ok".to_owned());
    writeln!(
        out,
        "loop{SEP}mismatch{SEP}-{SEP}{}{SEP}{mismatch}",
        i32::from(ops.compact.is_some()),
    )
    .unwrap();

    mask.release(&mut arena);
    arena.destroy(&mut mem);
}

/// Script 3 — the fire that carries no mask, and the fires that cannot.
fn script_inactive(out: &mut String) {
    let csr = vec![0u32, 8, 16];
    let g = FireGeometry::new(&csr).unwrap();
    let mut mem = Slab::new();
    let mut arena = SidebandArena::new();

    // (name, wants, with_obs, with_arena). The oracle's `null_hooks` case is a
    // null `StageHooks*`, which the port models as `wants_page_mask == false`:
    // there is no hooks pointer to be null.
    let cases = [
        ("null_hooks", false, false, false),
        ("wants_false", false, true, true),
        ("wants_false_no_obs", false, false, false),
        ("no_observation", true, false, true),
        ("no_arena", true, true, false),
    ];

    for (name, wants, with_obs, with_arena) in cases {
        let mut ops = Ops::default();
        let geometry = with_obs.then_some(g);
        let built = if with_arena {
            FirePageMask::new(wants, geometry, Some(&mut arena), &mut mem)
        } else {
            FirePageMask::new(wants, geometry, None, &mut mem)
        };
        let (active, sink_null, wrote, result) = match built {
            Ok(mut mask) => {
                let active = i32::from(mask.active());
                let sink_null = i32::from(mask.sink().is_none());
                mask.begin_layer(&mut ops);
                let wrote = i32::from(ops.memset.is_some());
                mask.compact(
                    &mut ops,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    99,
                )
                .unwrap();
                mask.release(&mut arena);
                (active, sink_null, wrote, "ok".to_owned())
            }
            Err(e) => (-1, -1, -1, outcome(e)),
        };
        writeln!(
            out,
            "inactive{SEP}{name}{SEP}{active}{SEP}{sink_null}{SEP}{wrote}{SEP}{}{SEP}{result}",
            i32::from(ops.compact.is_some()),
        )
        .unwrap();
    }
    arena.destroy(&mut mem);
}

/// Whether the mask slot is still held, observed as a caller observes it.
fn slot_is_held(arena: &mut SidebandArena, mem: &mut Slab) -> bool {
    match arena.acquire(mem, Region::Mask, 1) {
        Ok(_) => {
            arena.release(Region::Mask);
            false
        }
        Err(_) => true,
    }
}

/// Script 4 — across fires: the graph-capture precondition, end to end.
fn script_across_fires(out: &mut String) {
    let mut mem = Slab::new();
    let mut arena = SidebandArena::new();
    let fires: Vec<Vec<u32>> = vec![
        vec![0, 8, 16, 24, 32],
        vec![0, 4, 8],
        vec![0, 1],
        vec![0, 8, 16, 24, 32],
        vec![0, 8000, 16000, 24000],
        vec![0, 8, 16, 24, 32],
    ];

    let mut prev_base: Option<usize> = None;
    for (i, csr) in fires.iter().enumerate() {
        let g = FireGeometry::new(csr).unwrap();
        mem.recording = true;
        let mut mask = FirePageMask::new(true, Some(g), Some(&mut arena), &mut mem).unwrap();
        let base = mask.page_indices() as usize;
        let moved = prev_base.is_some_and(|p| p != base);
        prev_base = Some(base);
        let sink = mask.sink().unwrap();
        let (requests, stride, keep) = (sink.num_requests, sink.stride, sink.keep as usize);
        let events = mem.drain();
        mem.recording = false;
        writeln!(
            out,
            "fires{SEP}{i}{SEP}{requests}{SEP}{stride}{SEP}{}{SEP}{}{SEP}{events}",
            at(Some(base), keep),
            if moved { "base_moved" } else { "base_same" },
        )
        .unwrap();
        mask.release(&mut arena);

        writeln!(
            out,
            "fires{SEP}{i}{SEP}held_after{SEP}{}",
            i32::from(slot_is_held(&mut arena, &mut mem)),
        )
        .unwrap();
    }
    arena.destroy(&mut mem);
}

fn transcript() -> String {
    let mut out = String::new();
    script_carve(&mut out);
    script_layer_loop(&mut out);
    script_inactive(&mut out);
    script_across_fires(&mut out);
    out
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn the_rust_page_mask_reproduces_the_cpp_transcript() {
    let t = transcript();
    assert_eq!(
        t.lines().count(),
        GOLDEN_ROWS,
        "transcript row count drifted from the C++ oracle"
    );
    assert_eq!(
        fnv1a64(t.as_bytes()),
        GOLDEN_FNV1A64,
        "transcript differs from the C++ oracle; \
         run tests/oracle/page_mask/run.sh with PM_ORACLE_OUT set to diff them"
    );
}

/// Stated directly rather than left implicit in a hash, because it is the one
/// property a captured hook graph depends on.
#[test]
fn the_two_carve_paths_agree_on_every_geometry_the_oracle_drives() {
    for (name, csr) in shapes() {
        let Ok(g) = FireGeometry::new(&csr) else {
            continue;
        };
        let Ok(layout) = MaskSlotLayout::plan(g) else {
            continue;
        };
        let mut mem = Slab::new();
        let mut arena = SidebandArena::new();
        let plan = prepare_page_mask_capture(&mut arena, &mut mem, g).unwrap();
        let mut mask = FirePageMask::new(true, Some(g), Some(&mut arena), &mut mem).unwrap();
        let base = mask.page_indices() as usize;

        assert_eq!(plan.out_indices as usize, base, "{name}: indices");
        assert_eq!(
            plan.out_indptr as usize,
            base + layout.indptr_offset,
            "{name}: indptr"
        );
        assert_eq!(
            plan.out_last_lens as usize,
            base + layout.last_lens_offset,
            "{name}: last_lens"
        );
        let sink = mask.sink().unwrap();
        assert_eq!(sink.keep as usize, base + layout.keep_offset, "{name}: keep");
        assert_eq!(plan.keep, sink.keep, "{name}: the two paths disagree");
        assert_eq!(plan.stride, sink.stride, "{name}: stride");
        assert_eq!(plan.num_requests, sink.num_requests, "{name}: requests");

        mask.release(&mut arena);
        arena.destroy(&mut mem);
    }
}

/// The sizing claim the fixed stride rests on: a host CSR is a *bound*, so
/// every keep row must cover the page list it governs even when the device
/// resolves a shorter one.
#[test]
fn every_keep_row_covers_its_request_and_the_slot_covers_every_buffer() {
    for (name, csr) in shapes() {
        let Ok(g) = FireGeometry::new(&csr) else {
            continue;
        };
        let Ok(layout) = MaskSlotLayout::plan(g) else {
            continue;
        };
        for r in 0..layout.num_requests as usize {
            let pages = csr[r + 1] - csr[r];
            assert!(
                pages <= layout.stride,
                "{name}: request {r} spans {pages} pages, the row holds {}",
                layout.stride
            );
        }
        let requests = layout.num_requests as usize;
        let total_pages = csr[requests] as usize;
        assert!(layout.indptr_offset >= total_pages * 4, "{name}: indices");
        assert!(
            layout.counts_offset - layout.indptr_offset >= (requests + 1) * 4,
            "{name}: indptr"
        );
        assert!(
            layout.last_lens_offset - layout.counts_offset >= requests * 4,
            "{name}: counts"
        );
        assert!(
            layout.keep_offset - layout.last_lens_offset >= requests * 4,
            "{name}: last_lens"
        );
        assert_eq!(
            layout.total - layout.keep_offset,
            layout.keep_bytes(),
            "{name}: keep"
        );
    }
}

/// Escape hatch for regenerating/diffing: `PM_RUST_OUT=/tmp/rust.txt cargo test
/// -p driver-cuda --features cuda-13 --test page_mask_parity -- --nocapture
/// dump_transcript --ignored`.
#[test]
#[ignore = "diagnostic"]
fn dump_transcript() {
    if let Ok(path) = std::env::var("PM_RUST_OUT") {
        std::fs::write(path, transcript()).unwrap();
    } else {
        print!("{}", transcript());
    }
}
