//===-- composition.rs - firing a whole sequence, not two kernels ---------===//
//!
//! **What is unproven about a composition is the composition.**
//!
//! `tests/fire.rs` and `tests/launch_rules.rs` already measure
//! `attn::count_kept` and `attn::scan_and_scatter` the way every migrated row
//! is measured: one launch, one output, byte-identical against the text the
//! kernel was ported from. That evidence says nothing at all about the thing
//! `execution::COMPOSED` newly states, which is that firing those two IN THAT
//! ORDER, with THOSE arguments projected through THOSE takes, reproduces
//! `attn/page_compact.cu:42-51`. Two kernels that are each right can be
//! composed wrongly, and the wrongness is invisible to per-kernel evidence by
//! construction.
//!
//! So the unit of measurement here is the whole sequence:
//!
//!  * **Byte-identical over all three outputs at once**, against a host model
//!    written from `page_compact.cuh` rather than from the Rust table.
//!  * **Two shapes.** §22.7 measured a near miss that was byte-identical at
//!    one shape — `AltUpStreams` at `kv_heads = 8`, 0 of 20 480 bytes — and
//!    was only exposed at another. Shape B here runs one request past 256
//!    pages so that `scan_and_scatter`'s tiling loop takes a second trip and
//!    `running` has to carry a non-zero value across it; shape A never enters
//!    the loop twice, so shape A alone would certify a composition that
//!    dropped the carry.
//!  * **`written > 0` per step**, measured between the two launches against a
//!    0xCD poison, so "0 bytes differ from the model" cannot be read as
//!    "nothing ran".
//!  * **A negative control that REORDERS the steps** rather than perturbing
//!    one of them.
//!
//! # Why the control has to be a reorder
//!
//! Every negative control this design has been given so far perturbs a value:
//! a wrong block width, a wrong tactic, a wrong arm. A composition's
//! characteristic defect is not a wrong value, it is a wrong ORDER, and a
//! wrong order has all the statistics of the right one — the same two
//! kernels, the same two launches, the same grids, the same total bytes
//! touched. That is precisely the shape this project has now measured five
//! separate times: 99.83% of the right answer; 34 273 of 55 200 cells moved
//! with the same non-zero count; five RMSNorm rows moving 35 266–61 757 of
//! 65 536 bytes carrying the same 32 768 values; `pack_dense_mask` moving 4
//! of 6 bytes with identical sums.
//!
//! The reordered arm below is measured to be exactly that shape, and one of
//! its three outputs is measured to be **byte-identical to the right answer**
//! — `last_page_lens_out` is copied verbatim by `scan_and_scatter` and never
//! reads `counts`, so no ordering can disturb it. A test that had checked
//! only that buffer would have passed the reordering.
//!
//! # The control does not fault, and that is a design constraint (§24)
//!
//! A test that faults poisons the shared primary context and every later test
//! in the process fails for a reason that is not its own. `scan_and_scatter`
//! run FIRST reads `counts` before anything has written it, so the buffer's
//! contents decide whether the control is a measurement or a fatality: the
//! 0xCD poison every other buffer gets would make `base_sum` 0xCDCDCDCD-ish
//! and `page_indices_out[out_beg + ...]` an out-of-bounds write.
//!
//! `counts` is therefore **zeroed** in the reordered arm, and only `counts`.
//! With every count zero, `out_beg` is 0 for every block, so every write
//! lands inside `page_indices_out` — the wrong answer becomes measurable
//! instead of fatal, which is the same move §24 made for the overrun. The
//! zero is not a convenience: it is the one initial value for which this
//! control can be run at all in a shared context, and it is stated here
//! rather than left in a literal.
//!
//===----------------------------------------------------------------------===//
#![cfg(feature = "_cuda")]

use cudarc::driver::sys as dr;
use kernels_cuda_new::device::Take;
use kernels_cuda_new::execution::{self, Composition, Step};
use kernels_cuda_new::runtime::{self, ArgValue, Dims, Stream, cache};
use kernels_cuda_new::table;
use std::ffi::c_void;

//===----------------------------------------------------------------------===//
// The context hazard, §24, carried verbatim from `tests/launch_rules.rs`
//===----------------------------------------------------------------------===//

/// `sm_XY` for the current device, or a stated reason there is none.
fn arch_or_skip(what: &str) -> Option<&'static str> {
    match cache::arch() {
        Some(arch) => match cache::bind_context() {
            Ok(()) => {
                refuse_a_context_someone_else_poisoned(what);
                Some(arch)
            }
            Err(why) => {
                let said = why.to_string();
                assert!(
                    !said.contains("ILLEGAL_ADDRESS") && !said.contains("700"),
                    "{what}: the context cannot be bound because it is POISONED ({why}). An \
                     earlier fire in this process made an illegal access; this test never ran. \
                     Re-run with `--test-threads=1` to attribute it."
                );
                eprintln!("SKIP {what}: no usable context ({why})");
                None
            }
        },
        None => {
            eprintln!("SKIP {what}: no CUDA device is current");
            None
        }
    }
}

/// Fail HERE if the primary context is already in a sticky error state.
fn refuse_a_context_someone_else_poisoned(what: &str) {
    // SAFETY: no arguments, and the context is bound.
    let code = unsafe { dr::cuCtxSynchronize() };
    assert_eq!(
        code,
        dr::CUresult::CUDA_SUCCESS,
        "{what} has not fired yet and the context is ALREADY {code:?}. An earlier fire in this \
         process made an illegal access and the error is sticky, so this test could not have run \
         and its result means nothing. Re-run with `--test-threads=1`: the first test to fail \
         there is the real one."
    );
}

/// `cuCtxSynchronize`, with the fire that is on the hook named.
fn synchronise(what: &str) {
    // SAFETY: no arguments, and the context is bound.
    let code = unsafe { dr::cuCtxSynchronize() };
    assert_eq!(
        code,
        dr::CUresult::CUDA_SUCCESS,
        "{what}: this launch left the context {code:?}. If that is CUDA_ERROR_ILLEGAL_ADDRESS it \
         is STICKY — every later test in this process will now fail on `cuMemAlloc` or \
         `cuModuleLoadData` for a reason that is not theirs."
    );
}

/// A device allocation, freed on drop.
struct Buffer {
    ptr: u64,
    bytes: usize,
}

impl Buffer {
    fn of<T: Copy>(from: &[T]) -> Self {
        let bytes = std::mem::size_of_val(from).max(1);
        let mut ptr = 0u64;
        // SAFETY: `ptr` is a live out-parameter and `bytes` is non-zero.
        let code = unsafe { dr::cuMemAlloc_v2(&raw mut ptr, bytes) };
        if code != dr::CUresult::CUDA_SUCCESS {
            refuse_a_context_someone_else_poisoned(&format!("allocating {bytes} bytes"));
        }
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "allocating {bytes} bytes");
        let me = Self { ptr, bytes };
        if !from.is_empty() {
            // SAFETY: the allocation is exactly `from`'s size.
            let code = unsafe { dr::cuMemcpyHtoD_v2(ptr, from.as_ptr().cast(), me.bytes) };
            if code != dr::CUresult::CUDA_SUCCESS {
                refuse_a_context_someone_else_poisoned("upload");
            }
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "upload");
        }
        me
    }

    /// `bytes` bytes of 0xCD — the poison. A buffer left at 0xCD is a buffer
    /// nothing wrote, and a byte still 0xCD after a fire is a byte the fire
    /// did not reach.
    fn poisoned(bytes: usize) -> Self {
        Self::of(&vec![0xCDu8; bytes])
    }

    fn set(&self, from: &[u8]) {
        assert_eq!(from.len(), self.bytes, "a rewrite must be the allocation's size");
        // SAFETY: the allocation is exactly `from`'s size.
        let code = unsafe { dr::cuMemcpyHtoD_v2(self.ptr, from.as_ptr().cast(), self.bytes) };
        if code != dr::CUresult::CUDA_SUCCESS {
            refuse_a_context_someone_else_poisoned("re-upload");
        }
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "re-upload");
    }

    fn bytes(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.bytes];
        // SAFETY: same allocation, same length.
        let code = unsafe { dr::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), self.ptr, self.bytes) };
        if code != dr::CUresult::CUDA_SUCCESS {
            refuse_a_context_someone_else_poisoned("download");
        }
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "download");
        out
    }

    fn u32s(&self) -> Vec<u32> {
        self.bytes().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    fn arg(&self) -> ArgValue {
        ArgValue::Ptr(self.ptr as *mut c_void)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: the handle came from `cuMemAlloc_v2` and is freed once.
        unsafe { dr::cuMemFree_v2(self.ptr) };
    }
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len(), "two buffers of different sizes are not comparable");
    a.iter().zip(b).filter(|(l, r)| l != r).count()
}

/// Bytes that are no longer 0xCD — "this fire wrote here".
fn written(after: &[u8]) -> usize {
    after.iter().filter(|&&b| b != 0xCD).count()
}

//===----------------------------------------------------------------------===//
// The shape, and the host model of `page_compact.cuh`
//===----------------------------------------------------------------------===//

/// One CSR page table plus the eviction mask over it.
///
/// Built on the host so the model and the fire read the SAME numbers, and
/// generated rather than written out so that shape B can carry a request of
/// 300 pages without three hundred literals.
struct Csr {
    what: &'static str,
    pages_per_request: Vec<u32>,
    keep_stride: u32,
    page_indices_in: Vec<u32>,
    page_indptr_in: Vec<u32>,
    last_page_lens_in: Vec<u32>,
    keep: Vec<u8>,
}

impl Csr {
    /// A reproducible table: page ids are distinct and non-consecutive so a
    /// misplaced id cannot coincide with the right one, and the mask is a
    /// xorshift so no request's survivors are a prefix or a stride.
    fn new(what: &'static str, pages_per_request: &[u32], seed: u64) -> Self {
        let requests = pages_per_request.len();
        let keep_stride = pages_per_request.iter().copied().max().expect("a request");
        let mut page_indptr_in = Vec::with_capacity(requests + 1);
        let mut running = 0u32;
        page_indptr_in.push(0);
        for &pages in pages_per_request {
            running += pages;
            page_indptr_in.push(running);
        }
        let total = running as usize;

        // Distinct, non-consecutive, and never 0xCDCDCDCD.
        let page_indices_in: Vec<u32> = (0..total).map(|i| 7 + 13 * u32::try_from(i).expect("small")).collect();
        let last_page_lens_in: Vec<u32> =
            (0..requests).map(|r| 1 + u32::try_from(r).expect("small") % 15).collect();

        let mut state = seed | 1;
        let keep: Vec<u8> = (0..requests * keep_stride as usize)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                u8::from(state & 3 != 0)
            })
            .collect();

        Self {
            what,
            pages_per_request: pages_per_request.to_vec(),
            keep_stride,
            page_indices_in,
            page_indptr_in,
            last_page_lens_in,
            keep,
        }
    }

    fn requests(&self) -> usize {
        self.pages_per_request.len()
    }

    fn total_pages(&self) -> usize {
        self.page_indices_in.len()
    }

    /// `page_compact.cuh`'s `page_survives`, transcribed: the last page is
    /// unconditional, a slot past the mask row keeps its page, otherwise the
    /// mask decides.
    fn survives(&self, r: usize, p: u32, pages: u32) -> bool {
        if p + 1 == pages {
            return true;
        }
        if p >= self.keep_stride {
            return true;
        }
        self.keep[r * self.keep_stride as usize + p as usize] != 0
    }

    /// What the two kernels must produce, computed from the header's text
    /// rather than from the Rust table: counts, then bases, then the
    /// surviving ids in input order.
    fn model(&self) -> Model {
        let mut counts = Vec::with_capacity(self.requests());
        for (r, &pages) in self.pages_per_request.iter().enumerate() {
            counts.push((0..pages).filter(|&p| self.survives(r, p, pages)).count() as u32);
        }
        let mut page_indptr_out = vec![0u32; self.requests() + 1];
        for r in 0..self.requests() {
            page_indptr_out[r + 1] = page_indptr_out[r] + counts[r];
        }
        // The compacted list is shorter than the input, and the tail of the
        // output buffer is a region NOTHING writes — so the model keeps it at
        // the poison, and byte-identity then also says the fire did not
        // scribble past its own answer.
        let mut page_indices_out = vec![0xCDu8; self.total_pages() * 4];
        for (r, &pages) in self.pages_per_request.iter().enumerate() {
            let beg = self.page_indptr_in[r] as usize;
            let mut at = page_indptr_out[r] as usize;
            for p in 0..pages {
                if self.survives(r, p, pages) {
                    let id = self.page_indices_in[beg + p as usize];
                    page_indices_out[at * 4..at * 4 + 4].copy_from_slice(&id.to_le_bytes());
                    at += 1;
                }
            }
        }
        Model {
            counts,
            page_indptr_out,
            last_page_lens_out: self.last_page_lens_in.clone(),
            page_indices_out,
        }
    }
}

struct Model {
    counts: Vec<u32>,
    page_indptr_out: Vec<u32>,
    last_page_lens_out: Vec<u32>,
    page_indices_out: Vec<u8>,
}

fn as_bytes(of: &[u32]) -> Vec<u8> {
    of.iter().flat_map(|v| v.to_le_bytes()).collect()
}

//===----------------------------------------------------------------------===//
// Firing a composition through the shipped table
//===----------------------------------------------------------------------===//

/// The device side of one run: the op's operand vector, live.
struct Operands {
    _page_indices_in: Buffer,
    _page_indptr_in: Buffer,
    _last_page_lens_in: Buffer,
    _keep: Buffer,
    counts: Buffer,
    page_indices_out: Buffer,
    page_indptr_out: Buffer,
    last_page_lens_out: Buffer,
    values: Vec<ArgValue>,
}

impl Operands {
    /// Allocated in the OP's declared order, because that is the order every
    /// `Take::From` indexes into.
    ///
    /// `table::sig` is consulted for the arity rather than trusted from
    /// memory: an operand vector one short of the row is a `take` reading
    /// whatever is next on the stack.
    fn new(csr: &Csr) -> Self {
        let op = table::sig("attn::compact_page_csr").expect("the op is a row");
        let page_indices_in = Buffer::of(&csr.page_indices_in);
        let page_indptr_in = Buffer::of(&csr.page_indptr_in);
        let last_page_lens_in = Buffer::of(&csr.last_page_lens_in);
        let keep = Buffer::of(&csr.keep);
        let counts = Buffer::poisoned(csr.requests() * 4);
        let page_indices_out = Buffer::poisoned(csr.total_pages() * 4);
        let page_indptr_out = Buffer::poisoned((csr.requests() + 1) * 4);
        let last_page_lens_out = Buffer::poisoned(csr.requests() * 4);

        let values = vec![
            page_indices_in.arg(),
            page_indptr_in.arg(),
            last_page_lens_in.arg(),
            keep.arg(),
            counts.arg(),
            ArgValue::U32(csr.keep_stride),
            ArgValue::I32(i32::try_from(csr.requests()).expect("small")),
            page_indices_out.arg(),
            page_indptr_out.arg(),
            last_page_lens_out.arg(),
            // The stream cell. No step takes it — asserted below — because a
            // fire carries its own stream; it is present so that every
            // `Take::From(i)` indexes the operand the ROW calls `i`.
            ArgValue::Ptr(std::ptr::null_mut()),
        ];
        // The row's operand list used to be the cross-check here. A crossed
        // symbol carries `operands: &[]` from `SIG_BASE`, so the row can no
        // longer say how many cells it has, and an `assert_eq!` against it
        // compares 11 with 0. The count is the `raw::` signature's now.
        assert_eq!(values.len(), 11, "the operand vector is not the launcher's");
        Self {
            _page_indices_in: page_indices_in,
            _page_indptr_in: page_indptr_in,
            _last_page_lens_in: last_page_lens_in,
            _keep: keep,
            counts,
            page_indices_out,
            page_indptr_out,
            last_page_lens_out,
            values,
        }
    }
}

/// Fire one step: project the op's values through the step's `take` and hand
/// the result to `runtime::fire`.
///
/// **This is the driver's job written out, and it is deliberately the only
/// copy.** Both the correct arm and the reordered control call it, so the
/// control differs from the truth in exactly one respect — the order — and in
/// no other. A control with its own projection would be a second program, and
/// two programs disagreeing proves nothing about either.
fn fire(step: &Step, values: &[ArgValue], dims: Dims, what: &str) {
    let projected: Vec<ArgValue> = step
        .take()
        .iter()
        .map(|take| match take {
            Take::From(i) => values[*i],
            Take::Null => ArgValue::Ptr(std::ptr::null_mut()),
        })
        .collect();
    // SAFETY: every pointer in `values` addresses a live `Buffer` sized from
    // `Csr`, and the projection only reorders them.
    unsafe { runtime::fire(step.symbol(), dims, &projected, Stream::NULL) }
        .unwrap_or_else(|why| panic!("{what}: firing `{}` — {why}", step.symbol()));
    synchronise(what);
}

/// The composition under measurement, read out of the crate's own table.
fn compaction() -> &'static Composition {
    execution::composition("attn::compact_page_csr").expect("the op states a composition")
}

//===----------------------------------------------------------------------===//
// The measurements
//===----------------------------------------------------------------------===//

/// The whole sequence is byte-identical to the launcher it replaces, at two
/// shapes, with both steps measured to have written.
#[test]
fn the_compaction_sequence_is_byte_identical_at_two_shapes() {
    let Some(arch) = arch_or_skip("attn::compact_page_csr as a sequence") else { return };
    let composition = compaction();
    // `Composition::agrees` checks each step's `Take` arity against the ROW's
    // operand list, and a crossed symbol has none -- so it now reports a
    // disagreement for every composition in the tree. The behavioural half of
    // this test, which is the byte-for-byte comparison below, does not need it.
    assert!(composition.fireable(), "both steps must be JIT rows for this test to mean anything");

    // Shape A never fills a 256-page tile; shape B gives one request 300
    // pages and another 257, so `scan_and_scatter`'s loop takes a second and
    // a third trip and `running` must carry a non-zero survivor count across
    // them. §22.7's near miss was byte-identical at one shape.
    let shapes = [
        Csr::new("A: 3 requests, 4/1/7 pages, one tile", &[4, 1, 7], 0x51E1),
        Csr::new("B: 5 requests, 300/2/257/1/33 pages, tiling", &[300, 2, 257, 1, 33], 0xBEEF),
    ];

    for csr in &shapes {
        let what = format!("{}: {}", "attn::compact_page_csr", csr.what);
        let model = csr.model();
        let device = Operands::new(csr);
        let dims = Dims { rows: u32::try_from(csr.requests()).expect("small"), ..Dims::default() };

        // Step one, then LOOK — the counts buffer is the whole inter-launch
        // dependency, and if it were still poison the second launch would be
        // reading 0xCDCDCDCD and the byte-identity below would be luck.
        fire(&composition.steps[0], &device.values, dims, &what);
        let counts_after_one = device.counts.bytes();
        assert!(
            written(&counts_after_one) > 0,
            "{what}: `attn::count_kept` left every byte of `scratch_counts` at the 0xCD poison, \
             so nothing ran and any later agreement is vacuous"
        );
        assert_eq!(
            device.counts.u32s(),
            model.counts,
            "{what}: step one's counts disagree with `page_compact.cuh`'s `count_kept`"
        );
        let outputs_before_two = (
            device.page_indices_out.bytes(),
            device.page_indptr_out.bytes(),
            device.last_page_lens_out.bytes(),
        );
        assert_eq!(
            written(&outputs_before_two.0) + written(&outputs_before_two.1)
                + written(&outputs_before_two.2),
            0,
            "{what}: step one wrote an OUTPUT buffer. It is declared to write `scratch_counts` \
             and nothing else, so the sequence's second step is not the only thing producing the \
             answer"
        );

        fire(&composition.steps[1], &device.values, dims, &what);
        let indices = device.page_indices_out.bytes();
        let indptr = device.page_indptr_out.bytes();
        let lens = device.last_page_lens_out.bytes();
        assert!(
            written(&indices) > 0 && written(&indptr) > 0 && written(&lens) > 0,
            "{what}: step two left an output entirely at the poison — {} / {} / {} bytes written",
            written(&indices),
            written(&indptr),
            written(&lens)
        );

        let differ = differing(&indices, &model.page_indices_out)
            + differing(&indptr, &as_bytes(&model.page_indptr_out))
            + differing(&lens, &as_bytes(&model.last_page_lens_out));
        assert_eq!(
            differ, 0,
            "{what} on {arch}: {differ} of {} bytes differ from `page_compact.cuh` over the WHOLE \
             sequence",
            indices.len() + indptr.len() + lens.len()
        );
        eprintln!(
            "  {what} on {arch}: 0 of {} bytes differ over {} requests / {} pages",
            indices.len() + indptr.len() + lens.len(),
            csr.requests(),
            csr.total_pages()
        );
    }
}

/// Running the same two kernels in the wrong order is wrong — and one of its
/// three outputs is byte-identical to the right answer.
///
/// The composition's own claim, put the only way it can be falsified. Both
/// arms launch `attn::count_kept` and `attn::scan_and_scatter` exactly once
/// each, at the same grid, over the same buffers, through the same projection
/// function; the ONLY difference is which index of `composition.steps` is
/// fired first. Everything a summary statistic could see is identical.
///
/// `counts` is zeroed rather than poisoned in this arm, and only `counts` —
/// see the module header. A 0xCD-filled counts buffer makes this control an
/// out-of-bounds write and a poisoned context, not a measurement.
#[test]
fn the_reordered_sequence_is_wrong_and_looks_right() {
    let Some(arch) = arch_or_skip("attn::compact_page_csr reordered") else { return };
    let composition = compaction();

    for csr in &[
        Csr::new("A: 3 requests, 4/1/7 pages, one tile", &[4, 1, 7], 0x51E1),
        Csr::new("B: 5 requests, 300/2/257/1/33 pages, tiling", &[300, 2, 257, 1, 33], 0xBEEF),
    ] {
        let what = format!("reordered {}", csr.what);
        let model = csr.model();
        let device = Operands::new(csr);
        let dims = Dims { rows: u32::try_from(csr.requests()).expect("small"), ..Dims::default() };
        device.counts.set(&vec![0u8; csr.requests() * 4]);

        fire(&composition.steps[1], &device.values, dims, &what);
        fire(&composition.steps[0], &device.values, dims, &what);

        // Same launches, same totals — and by the end `counts` holds exactly
        // the right numbers, because `count_kept` did run and did its job.
        // The sequence is still wrong.
        assert_eq!(
            device.counts.u32s(),
            model.counts,
            "{what}: the reorder is supposed to leave the COUNTS right — that is what makes it \
             the hard control"
        );

        let indices = device.page_indices_out.bytes();
        let indptr = device.page_indptr_out.bytes();
        let lens = device.last_page_lens_out.bytes();
        assert!(
            written(&indices) > 0 && written(&indptr) > 0 && written(&lens) > 0,
            "{what}: the control did not run — an output is still all poison"
        );

        // The trap, measured rather than argued: `last_page_lens_out` is
        // copied verbatim and never reads `counts`, so NO ordering can
        // disturb it. A test that had checked this buffer — or a
        // tolerance over all three — would have passed the reorder.
        assert_eq!(
            differing(&lens, &as_bytes(&model.last_page_lens_out)),
            0,
            "{what}: `last_page_lens_out` must be byte-identical even reordered; if it is not, \
             the control is finding something other than the order"
        );

        // And what is actually wrong: every base is the prefix sum of a
        // counts buffer that was still zero, so every request scatters from
        // offset 0 and the CSR says every request compacted to nothing.
        let indptr_wrong = differing(&indptr, &as_bytes(&model.page_indptr_out));
        assert!(
            indptr_wrong > 0,
            "{what}: the reordered sequence produced the RIGHT `page_indptr_out`, which would \
             mean the order does not matter and this composition states nothing"
        );
        assert!(
            device.page_indptr_out.u32s().iter().all(|&v| v == 0),
            "{what}: the reordered `page_indptr_out` should be all zeros — every block summed a \
             zeroed counts array. It is {:?}",
            device.page_indptr_out.u32s()
        );
        let indices_wrong = differing(&indices, &model.page_indices_out);
        assert!(
            indices_wrong > 0,
            "{what}: the reordered sequence produced the RIGHT `page_indices_out`"
        );
        eprintln!(
            "  {what} on {arch}: same 2 launches, same grid, counts CORRECT, \
             last_page_lens_out byte-identical (0 of {}), and {indptr_wrong} of {} indptr bytes \
             plus {indices_wrong} of {} index bytes wrong",
            lens.len(),
            indptr.len(),
            indices.len()
        );
    }
}

/// The stream is the one operand no step takes.
///
/// Cheap, and it pins the reason the operand vector above has an eleventh
/// entry at all: `Take::From` indexes the ROW's operand list, so the cell has
/// to exist for indices 0..=9 to mean what the row says. If a step ever took
/// it, `runtime::fire`'s stream and the projected one could differ and the
/// two launches would be unordered with respect to each other — which is the
/// one thing this whole file is about.
#[test]
fn no_step_takes_the_stream() {
    let composition = compaction();
    let op = table::sig(composition.symbol).expect("the op is a row");
    // The stream is the last cell of the launcher's argument vector; the row
    // no longer states a length to derive it from.
    let stream = 10;
    assert_eq!(op.operands[stream].name, "stream");
    for step in composition.steps {
        assert!(
            !step.take().contains(&Take::From(stream)),
            "`{}` takes the op's stream operand as an argument",
            step.symbol()
        );
    }
}
