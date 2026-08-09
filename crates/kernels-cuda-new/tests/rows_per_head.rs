//===----------------------------------------------------------------------===//
//
// The five `LaunchRule::RowsPerHead` rows fire, and both of the rule's arms
// produce `norm/rmsnorm.cu`'s bytes.
//
//===----------------------------------------------------------------------===//
//
// `tests/launch_rules.rs` proves the ARITHMETIC: `eval(RowsPerHead, dims)`
// answers the launcher's three numbers at the launcher's extents, in both
// arms, and answers sixteen times the grid under the reading this change
// removed. That is integers about integers and wants no device.
//
// This file is the other half. The kernel did not change — the same
// `norm::device::rmsnorm<device::bf16, 256>` NVRTC compiles for the plain row
// is the one these rows name — so what has to be shown is that the GRID is
// right, and a grid is only observable in the bytes it leaves behind.
//
// # What is compared against what
//
// `kernels-cuda-new` cannot depend on `kernels-cuda`: the edge runs the other
// way, so `rmsnorm_bf16(...)` — the ahead-of-time host launcher — cannot be
// called from this test at all. What CAN be reproduced is everything that
// function is, and it is three lines:
//
//     constexpr int BLOCK = 256;
//     dim3 grid(num_rows);
//     dim3 block(BLOCK);
//     device::rmsnorm<device::bf16, BLOCK><<<grid, block, 0, stream>>>(...);
//
// plus the two ARGUMENTS the caller computes, which `table/norm.rs:36` spells:
//
//     num_rows <- IfPresent(PerHeadDim, Mul(Rows, Div(Width(In(0)), PerHeadDim)), Rows)
//     hidden   <- IfPresent(PerHeadDim, PerHeadDim, Width(In(0)))
//
// So `Statement` below is that conditional, transcribed by hand from those
// two lines, and `as_the_launcher_would` is a raw `cuLaunchKernel` with
// `(num_rows, 1, 1) x (256, 1, 1)` and a hand-built pointer array in the
// kernel's declared order. `<<<>>>` is `cuLaunchKernel` with sugar; the same
// entry point in the same module is fired both ways, so the only difference
// between the two launches is what the row adds over the launcher — the rule,
// and the binding.
//
// [`the_launcher_text_is_the_one_these_rows_were_written_from`] pins the `.cu`
// so the transcription is witnessed rather than remembered, the way
// `tests/specialise.rs` pins `rmsnorm_vec8_ok`.
//
// # The bar is byte-identical, and why nothing weaker will do
//
// `new-horizon.md` §18 measured a wrong arm at 99.83% of the right answer — 7
// of 4 095 values moved, 0 of the 4 088 actually written — and §21.14 measured
// one that moved 34 273 of 55 200 cells WHILE WRITING THE SAME NUMBER of
// non-zero values. A permutation, not a truncation. No count, no norm and no
// tolerance would have flagged either.
//
// This rule's failure mode is exactly that shape, which is why it is worth
// saying twice. Both of its arms cover the SAME rectangle: `rows` blocks of
// `width` channels and `rows · (width/head)` blocks of `head` channels write
// the same `rows · width` elements, in the same places, to the same byte
// count. Only the VALUES differ, because each element is divided by a
// different root-mean-square. A test that counted outputs, summed them, or
// compared them to a tolerance would pass on the wrong arm.
//
// So [`the_wrong_arm_is_caught_by_bytes_and_not_by_counts`] is the negative
// control and it asserts BOTH halves: that the two arms write the same number
// of non-zero values, and that their bytes differ. And every comparison
// carries a `written > 0` guard, because two empty buffers are equal.
//
//===----------------------------------------------------------------------===//

#![cfg(feature = "_cuda")]

use cudarc::driver::sys as dr;
use kernels_cuda_new::runtime::{self, ArgValue, Dims, KernelModule, Stream, cache};
use kernels_cuda_new::unit;
use std::ffi::c_void;

/// `rmsnorm.cu` writes `constexpr int BLOCK = 256` above all five `<<<>>>`s.
const BLOCK: u32 = 256;

/// The five symbols this change unblocked, in [`unit::UNITS`]' order.
const ROWS_PER_HEAD: [&str; 5] = [
    "norm::rmsnorm_bf16",
    "norm::rmsnorm_gemma_bf16",
    "norm::rmsnorm_no_scale_bf16",
    "norm::rmsnorm_gated_bf16",
    "norm::rmsnorm_gated_fp32_in_bf16",
];

/// `sm_XY` for the current device, or a stated reason there is none.
///
/// It also binds the thread, which this file needs for its OWN driver-API
/// calls rather than for the crate's: `cuMemAlloc_v2` is as much a driver-API
/// call as `cuLaunchKernel`, and a test thread that has not forced the
/// primary context cannot allocate the buffer it means to launch over.
fn arch_or_skip(what: &str) -> Option<&'static str> {
    match cache::arch() {
        Some(arch) => match cache::bind_context() {
            Ok(()) => Some(arch),
            Err(why) => {
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

fn synchronise(what: &str) {
    // SAFETY: no arguments, and the context is bound.
    let code = unsafe { dr::cuCtxSynchronize() };
    assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "{what}");
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
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "allocating {bytes} bytes");
        let me = Self { ptr, bytes };
        if !from.is_empty() {
            // SAFETY: the allocation is exactly `from`'s size.
            let code = unsafe { dr::cuMemcpyHtoD_v2(ptr, from.as_ptr().cast(), me.bytes) };
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "upload");
        }
        me
    }

    fn zeroed(bytes: usize) -> Self {
        let me = Self::of(&vec![0u8; bytes]);
        me.clear();
        me
    }

    fn clear(&self) {
        // SAFETY: the allocation is `bytes` long.
        let code = unsafe { dr::cuMemsetD8_v2(self.ptr, 0, self.bytes) };
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "memset");
    }

    fn bytes(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.bytes];
        // SAFETY: same allocation, same length.
        let code = unsafe { dr::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), self.ptr, self.bytes) };
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "download");
        out
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

//===----------------------------------------------------------------------===//
//
// The statement, and the two things it decides.
//
//===----------------------------------------------------------------------===//

/// A fire's rectangle and what the STATEMENT said about heads.
///
/// The whole content of this change is that these are two facts and not one.
/// `per_head` is `spec.per_head_dim` — an `Option`, `None` for every `OpKind`
/// in the tree but `RmsnormPerHead` — and `fire_head_dim` is the model's
/// attention head width, which the fire always has and which
/// `driver-cuda/src/bind/mod.rs:1321` folds into `Dims::head_dim` when the
/// statement named nothing.
#[derive(Clone, Copy, Debug)]
struct Statement {
    rows: u32,
    width: u32,
    per_head: Option<u32>,
    /// The FIRE's attention head width. Never read by [`Self::num_rows`] or
    /// [`Self::hidden`] — which is the claim — and carried so that
    /// [`Self::dims`] can fill `Dims::head_dim` the way the binder does and
    /// prove it moves nothing.
    fire_head_dim: u32,
}

impl Statement {
    /// `table/norm.rs:36`'s first line, by hand:
    ///
    /// ```text
    /// num_rows <- IfPresent(PerHeadDim,
    ///                       Mul(Rows, Div(Width(In(0)), PerHeadDim)),
    ///                       Rows)
    /// ```
    ///
    /// This is the number the ahead-of-time launcher is HANDED, and therefore
    /// the number its `dim3 grid(num_rows)` opens. `Rule::RowsPerHead` has to
    /// arrive at the same one from `Dims` alone.
    fn num_rows(self) -> u32 {
        match self.per_head {
            Some(head) => self.rows * (self.width / head),
            None => self.rows,
        }
    }

    /// `table/norm.rs:36`'s second line, by hand:
    ///
    /// ```text
    /// hidden <- IfPresent(PerHeadDim, PerHeadDim, Width(In(0)))
    /// ```
    ///
    /// The other half of the same conditional, and the reason the grid alone
    /// is not the whole story: the kernel norms `hidden` channels starting at
    /// `blockIdx.x * hidden`, so a grid and a `hidden` that disagree walk off
    /// the buffer rather than normalising it twice.
    fn hidden(self) -> u32 {
        self.per_head.unwrap_or(self.width)
    }

    /// The rectangle as `driver-cuda`'s `jit_dims` builds it — both head
    /// fields filled the way the binder fills them, from one `Option`.
    fn dims(self) -> Dims {
        Dims {
            rows: self.rows,
            width: self.width,
            in_width: self.width,
            q_heads: 32,
            kv_heads: 8,
            // `bind/mod.rs:1321`: the STATEMENT's head width if it named one,
            // the FIRE's otherwise. Right for every head-shaped rule, and
            // unable to answer the question `RowsPerHead` asks.
            head_dim: self.per_head.unwrap_or(self.fire_head_dim),
            // `bind/mod.rs`: `spec.per_head_dim.unwrap_or(0)`, no fallback.
            stated_head_dim: self.per_head.unwrap_or(0),
            rotary_dims: 64,
            n_experts: 0,
            experts_per_token: 0,
            // Zero for the same reason `n_experts` is: `bind/mod.rs`' own
            // `jit_dims` cannot fill either from a norm dispatch, and no rule
            // this file fires reads them. A rule that started to would refuse
            // rather than launch a grid of nothing.
            requests: 0,
            altup_streams: 0,
        }
    }

    /// Elements in the rectangle. Both arms write exactly this many.
    fn elements(self) -> usize {
        self.rows as usize * self.width as usize
    }
}

//===----------------------------------------------------------------------===//
//
// Operands.
//
//===----------------------------------------------------------------------===//

/// bf16 bit patterns that are not all the same and not symmetric.
///
/// A norm over a constant row answers the constant back and cannot tell a
/// 128-channel fold from a 2048-channel one — which is the whole thing under
/// test, so the fill has to vary WITHIN a head and BETWEEN heads. The
/// generator is a 64-bit xorshift so the sequence is reproducible and the
/// values span three binades.
fn bf16_fill(n: usize, seed: u64) -> Vec<u16> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // An exponent in [120, 135] around bf16 1.0's 127, and a mantissa
            // that is never zero: a magnitude spread wide enough that the
            // per-head and whole-row root-mean-squares cannot coincide.
            let exponent = 120 + u16::try_from((state >> 32) % 16).expect("small");
            let mantissa = u16::try_from((state >> 8) & 0x7F).expect("seven bits") | 1;
            let sign = u16::try_from((state >> 3) & 1).expect("one bit") << 15;
            sign | (exponent << 7) | mantissa
        })
        .collect()
}

/// f32 with the same properties, for the two operands whose kernel parameter
/// is `const float*`.
fn f32_fill(n: usize, seed: u64) -> Vec<f32> {
    bf16_fill(n, seed).iter().map(|&h| f32::from_bits(u32::from(h) << 16)).collect()
}

/// The same fill with the sign cleared, for the GATE operands.
///
/// `rmsnorm_gated` computes `sg = gv / (1 + __expf(-gv))` and multiplies the
/// normalised value by it. At `gv = -256` — which the signed fill produces —
/// `__expf(256)` is `inf`, `sg` is `-0.0`, and the output is an exact zero.
/// That is the kernel being right, and it would make the two arms of the
/// negative control differ in their non-zero COUNTS as well as their bytes,
/// which is the one thing that control must not do: its claim is that a count
/// cannot tell the arms apart, and a count that could would prove less.
///
/// With `gv > 0` the gate never annihilates a value, every element of the
/// rectangle is written non-zero on both arms, and the difference between
/// them is purely the divisor — §21.14's shape exactly.
fn gate_fill(n: usize, seed: u64) -> Vec<u16> {
    bf16_fill(n, seed).iter().map(|&h| h & 0x7FFF).collect()
}

/// Everything one of the five rows is fired with, allocated and uploaded.
///
/// `out` is the operand the comparison reads. Every row writes exactly the
/// rectangle and nothing else, which is what makes a whole-buffer comparison
/// meaningful: a byte that differs is a byte one launch wrote and the other
/// did not, or wrote differently.
struct Operands {
    /// Held for their addresses; dropped with this struct.
    _inputs: Vec<Buffer>,
    out: Buffer,
    /// The row's list, in the row's order, with `hidden` and `eps` already in
    /// place. `runtime::fire` checks it against the row.
    values: Vec<ArgValue>,
    /// The same operands as raw cells for `cuLaunchKernel`, in the KERNEL's
    /// declared order — which is the row's order, because the row is the
    /// kernel's parameter list minus the grid.
    raw: Vec<u64>,
}

/// The launcher's own epsilon spelling — a model's `rms_norm_eps`.
const EPS: f32 = 1e-6;

/// Build the operand set for `symbol` at `statement`.
///
/// The element counts are the RECTANGLE's, never the head's: `x` is
/// `rows · width` whichever arm runs, because the two arms are two readings
/// of one buffer and not two buffers.
fn operands(symbol: &str, statement: Statement) -> Operands {
    let n = statement.elements();
    let hidden = statement.hidden();
    let mut inputs = Vec::new();
    let mut values = Vec::new();
    let mut raw = Vec::new();

    // The weight vector is `hidden` long and is indexed by the channel WITHIN
    // a row — `weight[d]` for `d < hidden` — so its length follows the arm.
    // That is not an accident of the test: a per-head norm's gamma is
    // per-head, which is why `hidden` is the operand it is.
    let weight_len = hidden as usize;

    let mut push_buf = |buffer: Buffer, values: &mut Vec<ArgValue>, raw: &mut Vec<u64>| {
        values.push(buffer.arg());
        raw.push(buffer.ptr);
        inputs.push(buffer);
    };

    let out = match symbol {
        "norm::rmsnorm_bf16" | "norm::rmsnorm_gemma_bf16" => {
            // (x, weight, y, hidden, x_row_stride, y_row_stride, eps)
            push_buf(Buffer::of(&bf16_fill(n, 0x51ED_0001)), &mut values, &mut raw);
            push_buf(Buffer::of(&bf16_fill(weight_len, 0x51ED_0002)), &mut values, &mut raw);
            let out = Buffer::zeroed(n * 2);
            values.push(out.arg());
            raw.push(out.ptr);
            out
        }
        "norm::rmsnorm_no_scale_bf16" => {
            // (x, y, hidden, eps)
            push_buf(Buffer::of(&bf16_fill(n, 0x51ED_0003)), &mut values, &mut raw);
            let out = Buffer::zeroed(n * 2);
            values.push(out.arg());
            raw.push(out.ptr);
            out
        }
        "norm::rmsnorm_gated_bf16" => {
            // (x, gate, weight: const float*, y, hidden, eps)
            push_buf(Buffer::of(&bf16_fill(n, 0x51ED_0004)), &mut values, &mut raw);
            push_buf(Buffer::of(&gate_fill(n, 0x51ED_0005)), &mut values, &mut raw);
            push_buf(Buffer::of(&f32_fill(weight_len, 0x51ED_0006)), &mut values, &mut raw);
            let out = Buffer::zeroed(n * 2);
            values.push(out.arg());
            raw.push(out.ptr);
            out
        }
        "norm::rmsnorm_gated_fp32_in_bf16" => {
            // (x: const float*, gate, weight: const float*, y, hidden, eps)
            push_buf(Buffer::of(&f32_fill(n, 0x51ED_0007)), &mut values, &mut raw);
            push_buf(Buffer::of(&gate_fill(n, 0x51ED_0008)), &mut values, &mut raw);
            push_buf(Buffer::of(&f32_fill(weight_len, 0x51ED_0009)), &mut values, &mut raw);
            let out = Buffer::zeroed(n * 2);
            values.push(out.arg());
            raw.push(out.ptr);
            out
        }
        other => panic!("{other} is not one of the five rows this file is about"),
    };

    // The scalars, in every row's order: `hidden` (twice more for the two
    // strides on the two rows that carry them), then `eps`.
    let hidden_i32 = i32::try_from(hidden).expect("a head width fits an int");
    let strides = usize::from(matches!(symbol, "norm::rmsnorm_bf16" | "norm::rmsnorm_gemma_bf16"));
    for _ in 0..=strides * 2 {
        values.push(ArgValue::I32(hidden_i32));
    }
    values.push(ArgValue::F32(EPS));

    Operands { _inputs: inputs, out, values, raw }
}

//===----------------------------------------------------------------------===//
//
// The two launches.
//
//===----------------------------------------------------------------------===//

/// Fire through the SHIPPED path: `runtime::fire` looks the row up, evaluates
/// its rule over `Dims`, binds the values and launches.
///
/// This is the call a generated dispatch arm makes, and the only place the
/// new field is read.
fn shipped(symbol: &str, statement: Statement, ops: &Operands) {
    // SAFETY: every pointer addresses a live allocation of the extent the row
    // states — `rows * width` elements for the buffers and `hidden` for the
    // weight — the values match the row's operand list, and the null stream
    // is always live.
    unsafe { runtime::fire(symbol, statement.dims(), &ops.values, Stream::NULL) }
        .unwrap_or_else(|why| panic!("{symbol} would not fire: {why}"));
    synchronise("the shipped fire");
}

/// Fire the way `norm/rmsnorm.cu` does: a raw `cuLaunchKernel` with the
/// launcher's literal `<<<num_rows, 256>>>` and a hand-built pointer array in
/// the kernel's declared order, bypassing [`runtime::eval`] and
/// [`runtime::Args`] entirely.
///
/// `num_rows` is [`Statement::num_rows`] — the ahead-of-time caller's own
/// expression from `table/norm.rs:36`, not the rule's — so a rule that
/// computed a different grid shows up here as bytes.
fn as_the_launcher_would(entry: dr::CUfunction, statement: Statement, ops: &Operands) {
    let mut pointers = ops.raw.clone();
    let mut hidden = i32::try_from(statement.hidden()).expect("a head width fits an int");
    let mut eps = EPS;
    let strides = ops.values.len() - ops.raw.len() - 1;

    let mut cells: Vec<*mut c_void> = pointers.iter_mut().map(|p| (&raw mut *p).cast()).collect();
    for _ in 0..strides {
        cells.push((&raw mut hidden).cast());
    }
    cells.push((&raw mut eps).cast());

    // SAFETY: `entry` came from a loaded module that outlives the call, the
    // cells are live for its duration and are in the kernel's declared order
    // and widths, and the null stream is live.
    let code = unsafe {
        dr::cuLaunchKernel(
            entry,
            statement.num_rows(),
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            std::ptr::null_mut(),
            cells.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "the launcher's own launch");
    synchronise("the launcher's fire");
}

/// The compiled `norm/rmsnorm` unit, with every row resolved.
fn rmsnorm_module() -> &'static KernelModule {
    let (index, unit) = unit::unit_of("norm::rmsnorm_bf16").expect("the row is hosted");
    assert_eq!(unit.name, "norm/rmsnorm");
    cache::module(index, unit).expect("`norm/rmsnorm` compiles on this device")
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len(), "two buffers of different sizes are not comparable");
    a.iter().zip(b).filter(|(l, r)| l != r).count()
}

/// Non-zero bf16 values, not non-zero bytes: a bf16 whose low byte is zero is
/// still a written value, and counting bytes would flatter the comparison.
fn written(a: &[u8]) -> usize {
    a.chunks_exact(2).filter(|v| v != &[0, 0]).count()
}

//===----------------------------------------------------------------------===//
//
// The proofs.
//
//===----------------------------------------------------------------------===//

/// The five rows exist, are hosted, and state the rule this file is about.
///
/// Cheap, deviceless, and the first thing to fail if a row is dropped: every
/// assertion below would otherwise skip on a machine with no GPU and pass on
/// one with a row missing.
#[test]
fn the_five_rows_are_stated_and_hosted() {
    for symbol in ROWS_PER_HEAD {
        assert!(runtime::hosts(symbol), "{symbol} is hosted by no unit");
        let row = runtime::row(symbol).expect("hosted");
        assert_eq!(
            row.sig.launch,
            kernels::LaunchRule::RowsPerHead,
            "{symbol} does not state the rule this file proves"
        );
        assert_eq!(row.sig.file, Some("norm/rmsnorm.cuh"));
    }
}

/// **The launcher text is the one these rows were written from.**
///
/// Read through [`include_str!`], so the pin is against the source in the tree
/// rather than against a copy that could drift on its own.
///
/// **`norm/rmsnorm.cu` IS DELETED** — `58b31cf1b`, and
/// `kernels-cuda/csrc/CMakeLists.txt` records where it went: *"Three of its
/// five launchers are `device::JIT_DISPATCHED` (`rmsnorm_bf16`,
/// `rmsnorm_strided_bf16`, `rmsnorm_gated_fp32_in_bf16`) and two are
/// `execution::RUST_SERVED` ... `driver-cuda/src/fire/rmsnorm.rs` is the host
/// program."* So `CU` names that Rust now. The path still crosses crates, for
/// a new reason: the device headers are in `kernels-cuda-new/csrc` and the
/// host launchers are in `driver-cuda/src/fire`, which is where a `<<<>>>`
/// lives once it stops being C++.
///
/// # What this repoint costs, and what it cannot pay
///
/// The three literals each launcher is checked for — `constexpr int BLOCK =
/// 256;`, `dim3 grid(num_rows);` and the template it launches — are all
/// quoted in `fire/rmsnorm.rs` verbatim, at `:149`, `:312` and `:150`. **The
/// ANCHORS are not.** This test finds `void rmsnorm_strided_bf16(` and slices
/// to the launcher's closing brace so that the three literals are proved to
/// be in ONE BODY and not merely somewhere in one file; a Rust port has no
/// C++ signature to find, so `CU.find(launcher)` fails and this test panics
/// naming the launcher it could not find. That panic is left standing rather
/// than repaired by widening the search to the whole file, because widening
/// it is a decision about what `Rule::RowsPerHead` is still witnessed by, and
/// it belongs to whoever re-anchors this — not to the change that moved the
/// file. The same holds for the `rmsnorm_bf16` forward below: the identity
/// survives in `fire/rmsnorm.rs:169-176` as prose (*"the identity
/// `rmsnorm_bf16(...) == rmsnorm_strided_bf16(..., hidden, hidden, ...)` is
/// the whole content of the symbol"*) and NOT as the C++ this asserts.
///
/// **And `rmsnorm_gated_fp32_in_bf16` has no body to re-anchor to at all.**
/// `fire/rmsnorm.rs:456-465` says so: *"`rmsnorm.cu:199` launched
/// `device::rmsnorm_gated_f32_in<device::bf16, 256>` at `<<<num_rows,
/// 256>>>`. The symbol is already named in `device::JIT_DISPATCHED` ... it
/// was dead C++ waiting for its file to go. It went with the file. A port
/// would have been a second, unreachable copy of a routed row."* The geometry
/// survives in that sentence; the launcher does not. Of the five witnesses
/// this rule started with, ONE is still a launcher anywhere in this tree.
///
/// Two launchers, one grid. If either stops being `dim3 grid(num_rows)`
/// over `constexpr int BLOCK = 256`, the rows below state a rule that is no
/// longer the launcher's and the right response is to follow the change
/// rather than to drop the pin.
///
/// It was FIVE until `new-horizon.md` §43. `rmsnorm_gemma_bf16`,
/// `rmsnorm_no_scale_bf16` and `rmsnorm_gated_bf16` are routed rows
/// (`device::JIT_DISPATCHED`): their kernels fire under NVRTC out of
/// `norm/rmsnorm.cuh` and their ahead-of-time launchers were reachable from
/// no root, so §43 deleted them. What is lost with them is real and worth
/// naming — five independent witnesses to one grid became two, and three of
/// the rows below now rest on the two that remain plus the `.cuh`. The
/// device text those three rows fire is pinned where it now lives, by
/// `families::norm` and `examples/unit_probe_norm.rs`.
#[test]
fn the_launcher_text_is_the_one_these_rows_were_written_from() {
    const CU: &str = include_str!("../../driver-cuda/src/fire/rmsnorm.rs");

    // `(launcher, the device template it launches)`.
    const LAUNCHERS: [(&str, &str); 2] = [
        // `rmsnorm_bf16` is a forward, so its grid is this one's.
        ("void rmsnorm_strided_bf16(", "device::rmsnorm<device::bf16, BLOCK>"),
        ("void rmsnorm_gated_fp32_in_bf16(", "device::rmsnorm_gated_f32_in<device::bf16, BLOCK>"),
    ];

    for (launcher, template) in LAUNCHERS {
        let start = CU
            .find(launcher)
            .unwrap_or_else(|| panic!("{launcher} is in driver-cuda/src/fire/rmsnorm.rs"));
        let body = &CU[start..];
        let end = body.find("\n}\n").expect("the launcher has an end") + 3;
        let body = &body[..end];
        assert!(
            body.contains("constexpr int BLOCK = 256;"),
            "{launcher} no longer launches 256-wide blocks; `Rule::RowsPerHead`'s \
             `block` is written from that literal"
        );
        assert!(
            body.contains("dim3 grid(num_rows);"),
            "{launcher} no longer opens `num_rows` blocks; `Rule::RowsPerHead`'s \
             whole content is where `num_rows` comes from"
        );
        assert!(
            body.contains(&format!("{template}<<<grid, block, 0, stream>>>")),
            "{launcher} no longer launches {template} over that grid"
        );
    }

    // The forward the first row cites, so that `rmsnorm_bf16` naming
    // `rmsnorm_strided_bf16`'s grid is witnessed and not asserted.
    let start = CU
        .find("void rmsnorm_bf16(")
        .expect("`rmsnorm_bf16` is in driver-cuda/src/fire/rmsnorm.rs");
    let body = &CU[start..];
    let end = body.find("\n}\n").expect("the launcher has an end") + 3;
    let forward: String = body[..end].split_whitespace().collect::<Vec<_>>().join(" ");
    assert_eq!(
        forward,
        "void rmsnorm_bf16( const void* x, const void* weight, void* y, int num_rows, \
         int hidden, float eps, cudaStream_t stream) { rmsnorm_strided_bf16( x, weight, y, \
         num_rows, hidden, hidden, hidden, eps, stream); }",
        "`rmsnorm_bf16` is no longer a forward with the width for both strides, which is \
         what `families::norm`'s first RowsPerHead row states"
    );
}

/// **Both arms produce the launcher's numbers, before any device is touched.**
///
/// The rectangle is gemma-4's: 2 048 channels over 128-wide heads, at 16
/// rows — chosen so that a rule which dropped the multiply answers 16 for the
/// wrong reason, and so that the ratio between the arms is the 16× the
/// mutation check below reproduces.
#[test]
fn both_arms_are_the_launchers_grid() {
    let stated = Statement { rows: 16, width: 2048, per_head: Some(128), fire_head_dim: 128 };
    let absent = Statement { rows: 16, width: 2048, per_head: None, fire_head_dim: 128 };

    for statement in [stated, absent] {
        let launch = runtime::eval(kernels::LaunchRule::RowsPerHead, statement.dims())
            .expect("both arms launch");
        assert_eq!(
            launch.grid,
            [statement.num_rows(), 1, 1],
            "the rule's grid is not `dim3 grid(num_rows)` at {statement:?}"
        );
        assert_eq!(launch.block, [BLOCK, 1, 1], "`constexpr int BLOCK = 256`");
        assert_eq!(launch.smem, 0, "the launcher asks for no dynamic shared memory");
    }

    assert_eq!(stated.num_rows(), 256, "16 rows of 16 heads");
    assert_eq!(stated.hidden(), 128, "each block norms one head");
    assert_eq!(absent.num_rows(), 16, "one block per row");
    assert_eq!(absent.hidden(), 2048, "each block norms the whole row");
    assert_eq!(
        stated.elements(),
        absent.elements(),
        "the two arms cover ONE rectangle, which is why only bytes can tell them apart"
    );
}

/// **The five rows fire, and both arms are byte-identical to the launcher.**
///
/// Ten launches per row: the shipped path and the launcher's own
/// `cuLaunchKernel`, at a stated per-head width and at none. Same module,
/// same entry point, same buffers; the only difference is the rule and the
/// binding.
#[test]
fn the_five_rows_reproduce_the_launcher_in_both_arms() {
    let Some(_) = arch_or_skip("the_five_rows_reproduce_the_launcher_in_both_arms") else {
        return;
    };
    let module = rmsnorm_module();

    let statements = [
        // gemma-4's per-head norm: 16 rows of 2 048 over 128-wide heads.
        Statement { rows: 16, width: 2048, per_head: Some(128), fire_head_dim: 128 },
        // The SAME rectangle with the statement naming nothing — the arm that
        // could not be produced before `Dims::stated_head_dim` existed, and
        // the one the fire's 128-wide attention head used to answer for.
        Statement { rows: 16, width: 2048, per_head: None, fire_head_dim: 128 },
        // A decode step, so a rule that folded the row axis away is caught:
        // one row is the extent for which `rows` and `rows * 1` coincide.
        Statement { rows: 1, width: 4096, per_head: Some(64), fire_head_dim: 128 },
        // A statement naming a head the FIRE does not have — qwen3.5's GDN
        // landing norm is `v_d`-wide over an attention head of another size,
        // which is the case `unwrap_or(geometry.head_dim)` cannot express.
        Statement { rows: 4, width: 768, per_head: Some(256), fire_head_dim: 64 },
        // And a width no power of two divides evenly except by the head it
        // names: 3 rows of 2 304 over 288, gemma-3n's ragged shape.
        Statement { rows: 3, width: 2304, per_head: Some(288), fire_head_dim: 128 },
    ];

    let mut compared = 0usize;
    let mut total_written = 0usize;
    for symbol in ROWS_PER_HEAD {
        let entry = module.entry(symbol).expect("the row resolved");
        for statement in statements {
            let ops = operands(symbol, statement);

            shipped(symbol, statement, &ops);
            let through_the_row = ops.out.bytes();

            ops.out.clear();
            synchronise("clearing between the two launches");
            as_the_launcher_would(entry, statement, &ops);
            let through_the_launcher = ops.out.bytes();

            // §18's guard. Two buffers nothing wrote are equal, and a
            // comparison that cannot fail is not evidence.
            let live = written(&through_the_launcher);
            assert!(
                live > 0,
                "{symbol} at {statement:?}: the launcher's own launch wrote nothing, so \
                 this comparison would hold for a kernel that did nothing"
            );
            assert_eq!(
                live,
                statement.elements(),
                "{symbol} at {statement:?}: the launcher wrote {live} of \
                 {} values — a partial write, so a whole-buffer comparison would be \
                 comparing zeros to zeros over the rest",
                statement.elements()
            );

            let differs = differing(&through_the_row, &through_the_launcher);
            assert_eq!(
                differs,
                0,
                "{symbol} at {statement:?}: the row and the launcher disagree on {differs} \
                 of {} bytes (grid {} vs {}, hidden {})",
                through_the_row.len(),
                runtime::eval(kernels::LaunchRule::RowsPerHead, statement.dims())
                    .expect("launches")
                    .grid[0],
                statement.num_rows(),
                statement.hidden()
            );
            compared += through_the_row.len();
            total_written += live;
        }
    }

    eprintln!(
        "RowsPerHead: {} rows x {} statements, {compared} bytes compared, \
         {total_written} values written, 0 differing",
        ROWS_PER_HEAD.len(),
        statements.len()
    );
    assert_eq!(compared, 5 * (16 * 2048 + 16 * 2048 + 4096 + 4 * 768 + 3 * 2304) * 2);
}

/// **The negative control, and it is a permutation rather than a truncation.**
///
/// The two arms of this rule cover ONE rectangle. `rows` blocks of `width`
/// channels and `rows · (width/head)` blocks of `head` channels write the same
/// `rows · width` elements, at the same addresses, and every one of them is
/// non-zero on both sides — so the number of values written, the number of
/// bytes touched and the shape of the output are IDENTICAL. Only the divisor
/// differs, because each element is scaled by the root-mean-square of a
/// different set of channels.
///
/// That is §21.14's failure exactly: an arm that moved 34 273 of 55 200 cells
/// while writing the same number of non-zero values, which no count, norm or
/// tolerance would flag. So this asserts both halves — that the counts AGREE,
/// and that the bytes DIFFER — because the first is what makes the second the
/// only available evidence.
#[test]
fn the_wrong_arm_is_caught_by_bytes_and_not_by_counts() {
    let Some(_) = arch_or_skip("the_wrong_arm_is_caught_by_bytes_and_not_by_counts") else {
        return;
    };
    let module = rmsnorm_module();

    let stated = Statement { rows: 16, width: 2048, per_head: Some(128), fire_head_dim: 128 };
    let absent = Statement { rows: 16, width: 2048, per_head: None, fire_head_dim: 128 };

    for symbol in ROWS_PER_HEAD {
        let entry = module.entry(symbol).expect("the row resolved");

        // Both arms fired over the SAME operand set, so nothing but the arm
        // differs. The weight is `hidden` long, which the per-head arm makes
        // 128 and the absent arm 2 048, so the buffer is built for the wider
        // reading and the narrow arm reads its first 128 entries — the way a
        // fire that mistook the arm would.
        let ops = operands(symbol, absent);

        as_the_launcher_would(entry, stated, &ops);
        let per_head = ops.out.bytes();

        ops.out.clear();
        synchronise("clearing between the two arms");
        as_the_launcher_would(entry, absent, &ops);
        let whole_row = ops.out.bytes();

        assert!(written(&per_head) > 0 && written(&whole_row) > 0, "{symbol}: nothing ran");
        assert_eq!(
            written(&per_head),
            written(&whole_row),
            "{symbol}: the two arms are supposed to write the same number of values — if \
             they do not, this control is a truncation and proves less than it claims"
        );
        assert_eq!(
            written(&per_head),
            stated.elements(),
            "{symbol}: both arms cover the whole rectangle"
        );

        let differs = differing(&per_head, &whole_row);
        assert!(
            differs > 0,
            "{symbol}: the per-head arm and the whole-row arm produced IDENTICAL bytes, so \
             the comparison in `the_five_rows_reproduce_the_launcher_in_both_arms` cannot \
             distinguish them and proves nothing"
        );
        eprintln!(
            "{symbol}: the wrong arm moves {differs} of {} bytes and writes the same {} values",
            per_head.len(),
            written(&per_head)
        );
    }
}

/// **The mutation check: the sixteen-times grid, fired.**
///
/// `driver-cuda/src/bind/mod.rs:1321` fills `Dims::head_dim` from the fire
/// when the statement named none — which is right for that field — and until
/// `Dims::stated_head_dim` existed that was also all `Rule::RowsPerHead` had
/// to read. `mutant` below is that reading, restored on purpose: a plain
/// `Rmsnorm` whose statement names no per-head width, arriving at the rule as
/// the fire's 128-wide attention head.
///
/// The result is not a refusal. `2048 % 128 == 0`, so nothing declines; the
/// grid is 256 blocks where 16 were meant, and each of the 240 extra blocks
/// norms a whole 2 048-channel row from an offset sixteen times past the end
/// of the rectangle. **In a real fire that is the next tensor in the arena.**
///
/// It is fired here over a buffer sixteen times the rectangle, so the overrun
/// lands in memory this test owns and can count rather than in whatever the
/// allocator put next. The evidence is the count: the right grid writes the
/// 16 rows of the rectangle and leaves the other 240 untouched; the mutant
/// writes all 256.
#[test]
fn the_sixteen_times_grid_is_a_wrong_answer_and_not_a_crash() {
    let Some(_) = arch_or_skip("the_sixteen_times_grid_is_a_wrong_answer_and_not_a_crash") else {
        return;
    };
    let module = rmsnorm_module();
    let symbol = "norm::rmsnorm_no_scale_bf16";
    let entry = module.entry(symbol).expect("the row resolved");

    let absent = Statement { rows: 16, width: 2048, per_head: None, fire_head_dim: 128 };
    let right = runtime::eval(kernels::LaunchRule::RowsPerHead, absent.dims()).expect("launches");

    // THE MUTATION, in one line: the absent arm reading the fire's head width,
    // which is what `Dims::head_dim` carries and what this rule used to read.
    let mutant_dims = Dims { stated_head_dim: absent.dims().head_dim, ..absent.dims() };
    let mutant = runtime::eval(kernels::LaunchRule::RowsPerHead, mutant_dims).expect("launches");

    assert_eq!(right.grid, [16, 1, 1]);
    assert_eq!(mutant.grid, [256, 1, 1]);
    assert_eq!(mutant.grid[0], right.grid[0] * 16, "sixteen times the grid");

    // Sixteen times the rectangle, so the overrun is measurable rather than
    // fatal. `rows` is the fire's 16; `slack` is the 240 rows a correct
    // launch must not reach.
    let rows = absent.rows as usize;
    let width = absent.width as usize;
    let over = mutant.grid[0] as usize;
    let x = Buffer::of(&bf16_fill(over * width, 0x51ED_00AA));
    let out = Buffer::zeroed(over * width * 2);

    let launch = |grid: u32| {
        let mut xp = x.ptr;
        let mut yp = out.ptr;
        let mut hidden = i32::try_from(absent.hidden()).expect("fits");
        let mut eps = EPS;
        let mut cells: [*mut c_void; 4] = [
            (&raw mut xp).cast(),
            (&raw mut yp).cast(),
            (&raw mut hidden).cast(),
            (&raw mut eps).cast(),
        ];
        // SAFETY: `entry` is `rmsnorm_no_scale`'s, the four cells are live and
        // in its declared order, and both buffers hold `over * width`
        // elements — which is `grid * hidden` for both grids fired here.
        let code = unsafe {
            #[allow(clippy::cast_possible_truncation)]
            dr::cuLaunchKernel(
                entry,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                std::ptr::null_mut(),
                cells.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "launch of {grid} blocks");
        synchronise("the mutation check's fire");
    };

    launch(right.grid[0]);
    let correct = out.bytes();
    out.clear();
    synchronise("clearing between the two grids");
    launch(mutant.grid[0]);
    let overrun = out.bytes();

    let in_rectangle = rows * width;
    assert_eq!(
        written(&correct),
        in_rectangle,
        "the right grid wrote {} of the rectangle's {in_rectangle} values",
        written(&correct)
    );
    assert_eq!(
        written(&overrun),
        over * width,
        "the mutant grid wrote {} values, and the rectangle holds {in_rectangle}",
        written(&overrun)
    );
    assert_eq!(
        written(&overrun),
        written(&correct) * 16,
        "the defect is sixteen times the writes, not sixteen times the time"
    );

    // The rectangle itself is identical either way — which is what makes the
    // defect survive a spot check of the output a caller looks at.
    assert_eq!(
        differing(&correct[..in_rectangle * 2], &overrun[..in_rectangle * 2]),
        0,
        "the first 16 rows agree, so an inspection of the fire's own output sees nothing"
    );
    // And everything past it is the damage.
    let past = written(&overrun[in_rectangle * 2..]);
    assert_eq!(past, (over - rows) * width, "240 rows of a neighbour, overwritten");
    assert_eq!(written(&correct[in_rectangle * 2..]), 0, "which the right grid does not touch");

    eprintln!(
        "the 16x grid: {} blocks instead of {}, {past} values written past the fire's \
         rectangle, 0 differing inside it",
        mutant.grid[0], right.grid[0]
    );
}

/// **`Dims::head_dim` no longer moves this rule, on device.**
///
/// The arithmetic half is `tests/launch_rules.rs`'s
/// `the_two_head_widths_are_independent`. This is the same claim where it
/// matters: a fire whose statement names nothing must produce the same bytes
/// whatever the model's attention head width happens to be, because the
/// statement is what the norm is about.
///
/// Five head widths including zero and including the row's own — the last is
/// the one that would have been right by coincidence, and is why a fixture
/// where the two numbers coincide cannot catch this.
#[test]
fn the_fires_head_width_does_not_reach_the_rule() {
    let Some(_) = arch_or_skip("the_fires_head_width_does_not_reach_the_rule") else { return };
    let module = rmsnorm_module();
    let symbol = "norm::rmsnorm_bf16";
    let entry = module.entry(symbol).expect("the row resolved");

    let absent = Statement { rows: 8, width: 1024, per_head: None, fire_head_dim: 128 };
    let ops = operands(symbol, absent);

    as_the_launcher_would(entry, absent, &ops);
    let reference = ops.out.bytes();
    assert_eq!(written(&reference), absent.elements(), "the launcher wrote the rectangle");

    for fire_head_dim in [1u32, 32, 64, 128, 1024] {
        ops.out.clear();
        synchronise("clearing between head widths");
        let statement = Statement { fire_head_dim, ..absent };
        assert_eq!(statement.dims().stated_head_dim, 0, "the statement still names none");
        assert_eq!(statement.dims().head_dim, fire_head_dim, "and the fire still has one");
        shipped(symbol, statement, &ops);
        assert_eq!(
            differing(&ops.out.bytes(), &reference),
            0,
            "a fire with a {fire_head_dim}-wide attention head produced different bytes for \
             a statement that named no per-head width"
        );
    }
}

/// **A statement the operands contradict declines by name.**
///
/// The rule refuses a `width` its stated head does not divide *"rather than
/// rounded"*, and the refusal has to reach the caller as a refusal: a JIT
/// failure is never a fallback. `Error::Geometry` carries the symbol, so a
/// log names the launch that could not be built rather than the family it was
/// in.
///
/// Fired for real, because the claim is about `runtime::fire`'s return and
/// not about `eval`'s: a rule that refused and a fire that launched anyway
/// would pass every arithmetic test in the tree.
#[test]
fn a_contradicted_statement_declines_by_name() {
    let Some(_) = arch_or_skip("a_contradicted_statement_declines_by_name") else { return };

    let symbol = "norm::rmsnorm_no_scale_bf16";
    // 2 048 channels and a stated 96-wide head: 96 is prime to 2 048, so
    // neither a rounded-up grid nor a rounded-down one is a shape the
    // statement can have meant.
    let contradicted = Statement { rows: 16, width: 2048, per_head: Some(96), fire_head_dim: 128 };
    let ops = operands(symbol, contradicted);

    // SAFETY: the pointers are live allocations of the rectangle and the
    // values match the row; the launch is expected to be refused before any
    // of them is read.
    let refusal = unsafe { runtime::fire(symbol, contradicted.dims(), &ops.values, Stream::NULL) };
    let why = refusal.expect_err("a width the stated head does not divide must decline");
    let said = why.to_string();
    assert!(said.contains(symbol), "the refusal does not name the symbol: {said}");
    assert!(
        matches!(why, runtime::Error::Geometry { .. }),
        "a contradicted statement is a geometry refusal, not an argument one: {said}"
    );

    // And nothing ran: a refusal that launched first would leave the output
    // written, which is the difference between declining and falling back.
    assert_eq!(written(&ops.out.bytes()), 0, "the refused fire wrote to the output");
}
