//! THE BAKER PATH, ON A REAL ADAPTER: one claimed point fired end to end and
//! checked against a host reference.
//!
//! # What this asks that nothing else does
//!
//! `tests/the_walk_is_the_program.rs` asks whether the walk visits the right
//! steps and binds the right regions, with a recorder standing where a device
//! would. `tests/device.rs` asks whether this crate's device half works. Neither
//! asks the question in between, which is the one that matters most:
//!
//! > does a statement, taken through `BoundOp` and the GENERATED dispatch and a
//! > `#[claims]` body and `Encode::fire`, produce a WGSL dispatch that computes
//! > the right numbers?
//!
//! Every link in that chain is real here. The plan is a `model_ir::plan::Plan`;
//! the statement is bound by `driver_wgpu::baker::bound::Bound`; the arm is the
//! one `kernels_wgpu::points_dispatch` generated from the claim table; the body
//! is `kernels_wgpu::norm`'s `rmsnorm`, which picks the entrypoint and computes
//! the grid; the dispatch is what `baker::encode::Encoder` planned; and the
//! bind groups, the uniform block and the workgroup division are what this file
//! builds out of that plan and hands to `device::Device::run_all`.
//!
//! The reference is computed on the host in `f64` from the SAME bf16 inputs the
//! shader reads, so a disagreement is the kernel's or the binding's and not a
//! rounding argument.
//!
//! # Why `norm.rmsnorm`, and why only one point
//!
//! Because it is the shortest claimed point that is not trivial: it reduces
//! across a row, which means the workgroup barrier and the lane split are
//! exercised, and it reads a `Const` bank, which means the weight arena is too.
//! And because ONE point is what this plane can honestly demonstrate — no
//! catalog row's lane binds here (see `baker::mod`'s own tests for the
//! measurement), so an end-to-end decode is not available to fire and will not
//! be until a dense matmul exists on this plane.
//!
//! A single kernel fired correctly through the whole baker path is the largest
//! true statement available today, and it is a real one: it is the first time
//! any shader plane has computed a number through the points path.

#![cfg(feature = "native")]

use std::cell::RefCell;
use std::collections::BTreeMap;

use driver_wgpu::baker::dispatch::Dispatch;
use driver_wgpu::baker::marks::{BufferId, Slice};
use driver_wgpu::baker::stage::{FireTable, KvGeometry, Pools, Slab};
use driver_wgpu::baker::walk::{Extent, Fire};
use driver_wgpu::baker::{Bank, encode::Encoder};
use driver_wgpu::binding::Bound;
use driver_wgpu::device::{Buffer, Device, Pipelines, Recorded};
use driver_wgpu::serve::{Embedded, Modules};
use kernels_wgpu::Capability;
use model_compiler::program::{Call, Dt, Program, Rows, Slot, Step};
use model_ir::plan::{Cond, Op, Param, Plan, Shard, ValueDef};

// ── the fixture's shape ────────────────────────────────────────────────

/// Rows of the fire. THREE, not one: a single row would let a kernel that
/// ignored `row_base` pass, and the value-major arena arithmetic
/// (`offset * fire_rows`) is only distinguishable from a row-major reading when
/// there is more than one row.
const ROWS: i32 = 3;

/// Elements per row. Even, so every bf16 pair fills a whole `u32` word and the
/// odd-width edge path in `rms.wgsl` — a word straddling two rows, written
/// through a `atomicCompareExchangeWeak` loop — is not what is under test here.
///
/// 96 rather than a power of two on purpose: `rms.wgsl` reduces in chunks of
/// `PIE_LANES * N_READS` = 256*4 = 1024, so a 96-wide row is entirely inside
/// one ragged tail and the per-element bound check is the thing being relied on.
const WIDTH: u64 = 96;

const EPS: f32 = 1e-5;

/// The two allocations this fire addresses.
const ARENA_BUF: BufferId = BufferId(0);
const WEIGHTS_BUF: BufferId = BufferId(1);

// ── bf16, both ways ────────────────────────────────────────────────────

/// `f32` to bf16, round-to-nearest-even — the rounding a checkpoint's producer
/// applies, restated so the host reference reads the SAME values the shader
/// does.
fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    // Ties-to-even on the 16 bits being dropped.
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(round)) >> 16) as u16
}

/// bf16 back to `f32` — exact, since bf16 is the top half of an `f32`.
fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

/// A row-major bf16 rectangle, packed two halves per little-endian `u32`.
fn pack(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&to_bf16(*v).to_le_bytes());
    }
    out
}

/// The inverse of [`pack`].
fn unpack(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| from_bf16(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

/// Reproducible pseudo-random values in roughly `[-2, 2)`.
///
/// A fixed LCG rather than a random source: a numeric gate that fails on one
/// run in fifty is a gate nobody trusts, and a seed makes a failure something
/// that can be re-run.
fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64) / f64::from(1u32 << 31);
            (u * 4.0 - 2.0) as f32
        })
        .collect()
}

// ── the host reference ─────────────────────────────────────────────────

/// `norm.rmsnorm`, on the host, in `f64`.
///
/// THE FORMULA IS READ OFF THE SHADER, not off a memory of what an RMSNorm is,
/// because the two differ in exactly the places that matter. `rms.wgsl` computes
/// `inv = inverseSqrt(sum(x*x)/axis + eps)` (`common/reduce.inc.wgsl`'s
/// `pie_inv_rms`) and then `y[i] = gain * w[i] * (x[i] * inv)` — the epsilon is
/// INSIDE the square root and applied to the MEAN, and the claim body states
/// `gain = 1.0` and `plus_one = 0`.
///
/// The inputs are the bf16-rounded ones, since that is what the shader loads.
fn reference(x: &[f32], w: &[f32], rows: usize, width: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * width];
    for r in 0..rows {
        let row = &x[r * width..(r + 1) * width];
        let sum: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let inv = 1.0 / (sum / width as f64 + f64::from(eps)).sqrt();
        for i in 0..width {
            out[r * width + i] = (f64::from(w[i]) * (f64::from(row[i]) * inv)) as f32;
        }
    }
    out
}

// ── the staging ────────────────────────────────────────────────────────

/// This fire stages no pool and no runtime plane: `norm.rmsnorm` reads neither.
///
/// Stated as refusals rather than as zero regions, which is the rule
/// `baker::stage` gives: a layer with no pool is ABSENT and not empty, so a
/// statement that reached for one would refuse by name instead of binding
/// nothing.
struct NoPools;

impl Pools for NoPools {
    fn kv(&self, _layer: u32, _values: bool) -> Option<Slice> {
        None
    }
    fn slab(&self, _layer: u32, _which: Slab) -> Option<Slice> {
        None
    }
    fn kv_geometry(&self) -> KvGeometry {
        KvGeometry::default()
    }
    fn table(&self, _which: FireTable) -> Option<Slice> {
        None
    }
}

// ── the plan and the program ───────────────────────────────────────────

/// One statement: `y = rmsnorm(x, weight, eps)`.
fn plan() -> Plan {
    Plan {
        name: "a-device-fire".into(),
        plane: model_ir::kernels::Backend::Wgpu,
        facts: vec!["qo_one".into()],
        params: vec![Param {
            name: "norm.weight".into(),
            shape: vec![WIDTH],
            shard: Shard::Replicated,
            repr: "dense".into(),
        }],
        caches: Vec::new(),
        values: vec![ValueDef::Runtime("token_ids".into()), ValueDef::Stmt(0)],
        ops: vec![Op {
            kernel: "norm.rmsnorm".to_string(),
            inputs: vec![0],
            outputs: vec![1],
            weights: vec!["norm.weight".to_string()],
            params: vec![f32::to_bits(EPS).into()],
            cache: None,
            layer: Some(0),
            cond: Cond::Always,
        }],
        seams: vec![model_ir::plan::Seam {
            seam: model_ir::seam::OUT.name.to_string(),
            values: vec![1],
            layer: None,
        }],
    }
}

/// The program, stated by hand for the reason `tests/the_walk_is_the_program.rs`
/// gives at length in its header: no point this plane claims can SEED a tower,
/// so `model_compiler::program::bound` can build nothing here. The measurement
/// itself is `no_claimed_point_can_seed_a_tower`, which walks all twenty-one
/// claims. Two value-major rectangles, `x` then `y`.
fn program() -> Program {
    let row = WIDTH * 2;
    let arena = |i: u64| Slot::Arena {
        offset: i * row,
        rows: Rows::Fire,
        width: WIDTH,
        dtype: Dt::Bf16,
    };
    Program {
        words: vec![0, 1],
        steps: vec![Step {
            op: 0,
            call: Call::Point("norm.rmsnorm".into()),
        }],
        slots: vec![arena(0), arena(1)],
        row_pitch: row * 2,
    }
}

// ── the fire ───────────────────────────────────────────────────────────

/// THE FIRST REAL SHADER-PLANE KERNEL FIRE.
///
/// A statement goes through the whole baker path onto the adapter, and the
/// bytes that come back are compared against a host reference.
#[test]
fn norm_rmsnorm_fires_through_the_baker_path_and_matches_a_host_reference() {
    let Ok(device) = Device::open() else {
        driver_wgpu::skip::skipped("no adapter answered `Device::open`");
        return;
    };
    println!("adapter: {} ({:?})", device.name(), device.backend());

    // ── the host data ──────────────────────────────────────────────────
    let rows = ROWS as usize;
    let width = WIDTH as usize;
    // Rounded to bf16 and back BEFORE the reference is computed, so the host
    // and the shader read the same numbers.
    let x: Vec<f32> = noise(0x5eed, rows * width)
        .into_iter()
        .map(|v| from_bf16(to_bf16(v)))
        .collect();
    let w: Vec<f32> = noise(0xc0ffee, width)
        .into_iter()
        .map(|v| from_bf16(to_bf16(v)))
        .collect();
    let want = reference(&x, &w, rows, width, EPS);

    // ── the device data ────────────────────────────────────────────────
    let row_bytes = WIDTH * 2;
    let arena_bytes = row_bytes * 2 * u64::from(ROWS.unsigned_abs());
    let arena_buf = device
        .zeroed(arena_bytes)
        .expect("an activation arena for two rectangles");
    device
        .write(&arena_buf, 0, &pack(&x))
        .expect("x goes at the first value's region, which starts the arena");
    let weight_buf = device.buffer(&pack(&w)).expect("the weight arena");

    // ── the walk ───────────────────────────────────────────────────────
    let banks: BTreeMap<String, Bank> = [(
        "norm.weight".to_string(),
        Bank {
            slice: Slice::whole(WEIGHTS_BUF, weight_buf.size()),
            shape: vec![WIDTH],
            dtype: model::produce::Dtype::Bf16,
            repr: "dense".to_string(),
        },
    )]
    .into_iter()
    .collect();
    let plan = plan();
    let program = program();
    let pools = NoPools;
    let fire = Fire::over(
        &plan,
        &program,
        Extent {
            arena: Slice::whole(ARENA_BUF, arena_bytes),
            rows: ROWS,
            requests: 1,
            layers: 1,
        },
        &banks,
        &pools,
    );
    let planned = {
        let encoder = Encoder::over(&fire.bindings, &fire.cursor);
        fire.walk(&encoder)
            .unwrap_or_else(|why| panic!("the walk refused: {why}"));
        encoder.finish()
    };
    assert_eq!(planned.len(), 1, "one statement plans one dispatch");
    let d = &planned[0];
    println!(
        "planned: {} in {} over {:?} lanes, {} bindings, {} uniform bytes",
        d.symbol,
        d.file,
        d.lanes,
        d.args.len(),
        d.uniform().len(),
    );
    assert_eq!(
        d.symbol, "rms_single_row_bfloat16",
        "the claim body picks the entrypoint, and this is the one it picks",
    );
    assert!(
        fire.blits.borrow().is_empty(),
        "`norm.rmsnorm` states no `InOut`, so nothing is copied first",
    );

    // ── the encode ─────────────────────────────────────────────────────
    //
    // What a device half would do with a planned fire, done here for one
    // dispatch: resolve each region to a `Bound`, build the pipeline the
    // dispatch names, and divide the lanes by the module's own workgroup size.
    let mut pipelines = Pipelines::new();
    let buffers = [&arena_buf, &weight_buf];
    let bound = bindings_of(d, &buffers, device.min_storage_offset());
    let source = Embedded
        .at(d.file, d.symbol, Capability::Baseline)
        .expect("the shader tree carries the entrypoint the body named");
    let pipeline = pipelines
        .get(&device, d.symbol, Capability::Baseline, &source)
        .expect("the module builds a pipeline on this adapter");
    let local = pipeline.module().local;
    let groups = [
        d.lanes[0].div_ceil(local.at(0)),
        d.lanes[1].div_ceil(local.at(1)),
        d.lanes[2].div_ceil(local.at(2)),
    ];
    println!(
        "workgroup {:?} -> {groups:?} groups; bindings {}",
        [local.at(0), local.at(1), local.at(2)],
        pipeline.bindings(),
    );
    assert_eq!(
        bound.len(),
        pipeline.bindings(),
        "the body's buffer list must fill every `@group(0)` binding the module \
         declares — a declared-and-unfilled slot is a bind group wgpu refuses",
    );

    let uniform = d.uniform();
    device
        .run_all(&[Recorded {
            pipeline,
            buffers: &bound,
            uniform: &uniform,
            groups,
        }])
        .unwrap_or_else(|(stage, why)| panic!("the dispatch failed at {stage:?}: {why}"));

    // ── the answer ─────────────────────────────────────────────────────
    //
    // The result is the SECOND value, so it starts one value-major run in.
    let at = row_bytes * u64::from(ROWS.unsigned_abs());
    let got = unpack(
        &device
            .read_at(&arena_buf, at, row_bytes * u64::from(ROWS.unsigned_abs()))
            .expect("the result reads back"),
    );
    assert_eq!(got.len(), want.len());

    // bf16 carries eight bits of mantissa, so one ulp is about 0.4% and the
    // shader's f32 reduction may land a ulp either side of the host's f64 one.
    // 2% is loose enough that neither is a failure and tight enough that a
    // wrong kernel, a wrong binding or a wrong row cannot pass: every one of
    // those is off by a factor, not by a ulp.
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let scale = w.abs().max(1e-3);
        let rel = (g - w).abs() / scale;
        worst = worst.max(rel);
        assert!(
            rel < 2e-2,
            "element {i} (row {}, col {}): device {g}, host {w}, relative {rel}",
            i / width,
            i % width,
        );
    }
    println!(
        "MATCHED {} elements, worst relative error {worst:e}",
        got.len()
    );

    // The input must be untouched: `rmsnorm` is out of place, and a kernel
    // that wrote through its own operand would still have matched above.
    let back = unpack(
        &device
            .read_at(&arena_buf, 0, row_bytes * u64::from(ROWS.unsigned_abs()))
            .expect("the operand reads back"),
    );
    assert_eq!(
        back, x,
        "`norm.rmsnorm` states an `In`, and an `In` is read only"
    );
}

/// Turn a planned dispatch's regions into the bind-group entries for it.
///
/// THE SEAM BETWEEN THE TWO HALVES, and it is four lines because
/// `baker::marks::Slice` was designed to be exactly this: a buffer, an offset
/// and an extent, which is a `wgpu::BufferBinding`'s three fields. Metal's
/// executor hands its device half an ADDRESS and keeps an address→buffer map
/// beside it; there is no such map here and nothing to keep in step.
///
/// The alignment is the DEVICE's rather than the specification's guaranteed
/// 256. A driver laying out an arena for a browser would want the guarantee —
/// `binding::Bound::within` takes the number for exactly that reason — but what
/// this test measures is a fire on THIS adapter, and asking for a stricter
/// alignment than the adapter has would refuse a binding the adapter accepts.
fn bindings_of<'a>(
    d: &Dispatch,
    buffers: &[&'a Buffer; 2],
    alignment: u64,
) -> Vec<Bound<'a, Buffer>> {
    d.args
        .iter()
        .map(|a| {
            let buffer = buffers[a.slice.buffer.0 as usize];
            Bound::within(buffer, a.slice.at, a.slice.bytes, alignment).unwrap_or_else(|why| {
                panic!(
                    "a region the walk bound is not addressable: {:?} of {} bytes at {}: {why:?}",
                    a.slice.buffer, a.slice.bytes, a.slice.at,
                )
            })
        })
        .collect()
}

/// THE UNIFORM BLOCK THE WALK PLANNED IS THE ONE THE MODULE DECLARES.
///
/// Checked against `naga` rather than against the shader source, and with no
/// adapter needed for the comparison itself: `reflect::entrypoint` reads the
/// `@group(1) @binding(0)` struct's member offsets out of the module, and the
/// dispatch's `param_slots` say where the executor put each scalar. A
/// disagreement is a scalar read as its neighbour, which is numbers rather than
/// errors — the failure mode `driver-vulkan` measured on a twenty-byte block
/// packed against a twenty-four byte range.
#[test]
fn the_planned_uniform_block_agrees_with_what_the_module_declares() {
    let plan = plan();
    let program = program();
    let banks: BTreeMap<String, Bank> = [(
        "norm.weight".to_string(),
        Bank {
            slice: Slice::whole(WEIGHTS_BUF, WIDTH * 2),
            shape: vec![WIDTH],
            dtype: model::produce::Dtype::Bf16,
            repr: "dense".to_string(),
        },
    )]
    .into_iter()
    .collect();
    let pools = NoPools;
    let fire = Fire::over(
        &plan,
        &program,
        Extent {
            arena: Slice::whole(ARENA_BUF, WIDTH * 2 * 2 * u64::from(ROWS.unsigned_abs())),
            rows: ROWS,
            requests: 1,
            layers: 1,
        },
        &banks,
        &pools,
    );
    let planned = {
        let encoder = Encoder::over(&fire.bindings, &fire.cursor);
        fire.walk(&encoder).expect("the fixture walks");
        encoder.finish()
    };
    let d = &planned[0];

    let declared = driver_wgpu::reflect::entrypoint(d.symbol, Capability::Baseline)
        .expect("the entrypoint reflects");
    let planned_offsets: Vec<u32> = d.param_slots.iter().map(|p| p.at).collect();
    assert_eq!(
        planned_offsets, declared.uniform_offsets,
        "the executor placed this statement's scalars at {planned_offsets:?} and \
         `{}` declares its uniform members at {:?}",
        d.symbol, declared.uniform_offsets,
    );
    assert_eq!(
        d.args.len(),
        declared.bindings as usize,
        "the body bound {} buffers and `{}` declares {} `@group(0)` bindings",
        d.args.len(),
        d.symbol,
        declared.bindings,
    );

    // And the block the walk would write is the size the module's struct is.
    let uniform = RefCell::new(d.uniform());
    assert_eq!(
        uniform.borrow().len(),
        declared
            .uniform_offsets
            .last()
            .map_or(0, |last| *last as usize + 4),
        "the planned block is exactly the declared members and their padding",
    );
}
