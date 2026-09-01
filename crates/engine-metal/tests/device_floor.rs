//! The device floor, on a real Apple GPU: the shaders compile, the handle
//! table resolves, and one kernel entry fired through the sink computes what
//! it says it computes.
//!
//! **WHAT THIS IS FOR.** Everything above this file — the resolver, the six
//! dispatch impls, the walk — has been type-checked on Linux since the
//! crate existed and has never touched a GPU. Three facts stand between
//! "it compiles" and "it runs", and none of them is checkable off the
//! device: that the shipped `.metal` sources compile at all (they have never
//! been through a Metal compiler in this tree), that a `Fire`'s positional
//! argument list lands where the shader's `[[buffer(n)]]` declarations say,
//! and that a `Tensor`'s `u32` resolves to the bytes the shell carved. This
//! file asks all three.
//!
//! # Gating
//!
//! An Apple target is not a machine with a GPU — a headless build box
//! publishes no device — so the file is `cfg`'d to Apple at compile time and
//! SKIPS at run time when `device::present()` says no, saying so. An
//! `#[ignore]`d test on the one box that could run it is a test nobody runs.
//!
//! ```text
//! cargo test -p engine-metal --release --test device_floor -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_metal::device::{self, Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::{Fire, Tensor};
use model_ir::Dtype;

/// **ONE DEVICE AT A TIME.** `cargo test` runs a file's tests on several
/// threads, and each of these binds a device and reserves buffers; two of
/// them compiling the whole shader set at once is a way to meet the Metal
/// compiler's own concurrency and learn nothing. Serialized, not because
/// the shell is unsafe across threads, but because the MEASUREMENTS are
/// only readable one at a time.
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

/// The third-party headers are included BY the sources and are not
/// translation units of their own — `mlx_steel_mma.metal` has no
/// `#include <metal_stdlib>` and does not compile alone, by design.
fn is_a_translation_unit(file: &str) -> bool {
    !file.starts_with("third_party/")
}

/// **TWO SHIPPED SOURCES DO NOT COMPILE, AND NEITHER IS REACHABLE.**
/// `attn/sdpa_sliding.metal` and `attn/sdpa_vector.metal` both call
/// `SDPA_ONLINE_FINISH()`, a macro no file in the tree defines — the
/// online-softmax epilogue of an sdpa family whose shipped half is
/// `attn/sdpa_paged.metal`. No `Fire::at` names either file (the live decode
/// and prefill points are all `sdpa_paged`'s), so nothing in this shell can
/// reach them, and nothing above this line has ever compiled them: they are
/// M15 residue that the first Metal compiler in this tree found.
///
/// They are recorded here rather than deleted or repaired, because a shader
/// is not this wave's to write (and deleting one would remove the evidence).
/// The pin is two-directional on purpose: the census proves the pair still
/// fails FOR THIS REASON, so the day the macro arrives this test goes red
/// and the entry comes out of the list — the failing-loudly-when-fixed
/// idiom, not a suppression.
const ORPHANS: &[(&str, &str)] = &[
    ("attn/sdpa_sliding.metal", "SDPA_ONLINE_FINISH"),
    ("attn/sdpa_vector.metal", "SDPA_ONLINE_FINISH"),
];

fn orphan(file: &str) -> Option<&'static str> {
    ORPHANS
        .iter()
        .find(|(name, _)| *name == file)
        .map(|(_, marker)| *marker)
}

#[test]
fn the_device_binds_and_says_what_it_is() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the device binds") else {
        return;
    };
    println!(
        "device: {} | working set {:.1} GiB | max buffer {:.1} GiB | cores(stated) {}",
        device.name(),
        device.working_set() as f64 / (1 << 30) as f64,
        device.max_buffer() as f64 / (1 << 30) as f64,
        device.cores()
    );
    assert!(device.working_set() > 0, "a device holds something");
    // The unified-memory assertion is inside `bind` — reaching here at all
    // is the check passing, and this states why it matters.
    assert!(
        device.max_buffer() >= 1 << 30,
        "this shell reserves the whole checkpoint as one buffer"
    );
}

#[test]
fn every_shipped_shader_compiles_and_every_entrypoint_instantiates() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the shaders compile") else {
        return;
    };
    let pipelines = Pipelines::new();
    let mut files = 0;
    let mut points = 0;
    let mut refused = Vec::new();
    let mut orphans_still_broken = 0;
    for (file, _) in kernels_metal::SOURCES {
        if !is_a_translation_unit(file) {
            continue;
        }
        if let Some(marker) = orphan(file) {
            let said = pipelines
                .entrypoints(&device, file)
                .expect_err("an orphan does not compile")
                .to_string();
            assert!(
                said.contains(marker),
                "`{file}` no longer fails on `{marker}` — take it out of ORPHANS: {said}"
            );
            orphans_still_broken += 1;
            continue;
        }
        files += 1;
        let names = match pipelines.entrypoints(&device, file) {
            Ok(names) => names,
            Err(fault) => {
                refused.push(format!("{file}: {fault}"));
                continue;
            }
        };
        for name in names {
            // `Fire::at` takes `&'static str`; the library's own names are
            // owned, so they are leaked into the census. A test that
            // compiles every point in the tree once is allowed one leak.
            let entry: &'static str = Box::leak(name.into_boxed_str());
            match pipelines.warm(&device, Fire::at(file, entry)) {
                Ok(()) => points += 1,
                Err(fault) => refused.push(format!("{file}:{entry}: {fault}")),
            }
        }
    }
    println!(
        "compiled {points} entrypoints across {files} sources; {orphans_still_broken} orphaned \
         sources still refuse, by name"
    );
    assert_eq!(orphans_still_broken, ORPHANS.len());
    assert!(
        refused.is_empty(),
        "{} points refused:\n{}",
        refused.len(),
        refused.join("\n")
    );
    assert!(points > 0, "the tree ships shaders");
    assert_eq!(
        pipelines.compiled(),
        points as u64,
        "every point compiled exactly once"
    );
}

#[test]
fn a_second_sighting_of_a_point_compiles_nothing() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the pipeline cache holds") else {
        return;
    };
    let pipelines = Pipelines::new();
    let point = Fire::at("layout/embed.metal", "embed_bfloat16");
    pipelines.warm(&device, point).expect("the point compiles");
    assert_eq!(pipelines.compiled(), 1);
    for _ in 0..8 {
        pipelines.warm(&device, point).expect("and is held");
    }
    assert_eq!(
        pipelines.compiled(),
        1,
        "a warm point compiles nothing — the absence is only observable through the counter"
    );
}

#[test]
fn a_jit_stamp_mints_the_entrypoint_the_shipped_source_does_not_hold() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the stamp path") else {
        return;
    };
    let pipelines = Pipelines::new();
    // One affine qmm point, composed the way `linear::quant` composes the one
    // it selects. The shipped source declares `PIE_STAMP_qmm_t` and
    // instantiates nothing with it, so the entrypoint exists only because the
    // driver appended the stamp.
    let point = kernels_metal::linear::quant::qmm_point(
        "quant.qmm_t",
        "",
        "PIE_STAMP_qmm_t",
        64,
        4,
        32,
        32,
    )
    .expect("an axis point");
    let stamped = Fire::at("linear/quant_qmm_t.metal", point.entry).stamp(point.stamp);
    assert!(
        !pipelines
            .entrypoints(&device, "linear/quant_qmm_t.metal")
            .expect("the source compiles")
            .iter()
            .any(|name| name == point.entry),
        "`{}` is instantiated in source, and this test is about the one that is not",
        point.entry
    );
    pipelines
        .warm(&device, stamped)
        .expect("the stamp mints it");
}

#[test]
fn a_stamp_that_mints_no_such_entrypoint_is_refused_by_name() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the stamp refusal") else {
        return;
    };
    let pipelines = Pipelines::new();
    // A well-formed invocation of the file's own macro, minting a symbol
    // nobody asked for — so the source compiles and the LOOKUP is what fails.
    let stamped = Fire::at(
        "linear/quant_qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32",
    )
    .stamp("PIE_STAMP_qmm_t(\"somebody_elses_symbol\", 64, 4, 32, 32, 32)");
    let fault = pipelines
        .warm(&device, stamped)
        .expect_err("the stamp mints a name the fire does not ask for");
    let said = fault.to_string();
    assert!(
        said.contains("somebody_elses_symbol"),
        "the refusal names the stamp: {said}"
    );
}

/// The one end-to-end statement of the floor: `layout.embed` gathers rows of
/// a table by id, fired through the real sink, over handles the shell
/// minted, into bytes the shell reads back.
///
/// **WHY THIS ENTRY.** It is the first launch of every fire, its shader
/// signature interleaves three buffers and two scalars (`ids[0]`,
/// `table[1]`, `y[2]`, `hidden[3]`, `vocab[4]`), and its answer is a
/// permutation — so a mis-indexed argument, a handle resolved to the wrong
/// offset, or a threadgroup the shell sized wrong all show up as the wrong
/// rows rather than as a crash.
#[test]
fn one_entry_fired_through_the_sink_computes_what_it_says() {
    let _serial = serialized();
    let Some(device) = device_or_skip("a real dispatch") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    let vocab = 6u32;
    let hidden = 4u32;
    let rows = 3u32;

    // The table: row v holds `[v*10 + 0 .. v*10 + 3]`, in bf16.
    let mut table_bytes = Vec::new();
    for v in 0..vocab {
        for d in 0..hidden {
            table_bytes.extend_from_slice(&bf16(v as f32 * 10.0 + d as f32));
        }
    }
    let mut table = Buffer::zeroed(&device, table_bytes.len() as u64).expect("the table reserves");
    table.write(0, &table_bytes).expect("the table lands");

    // The ids, i32, and the output rectangle.
    let ids: [i32; 3] = [4, 0, 2];
    let mut id_store = Buffer::zeroed(&device, 4 * rows as u64).expect("the ids reserve");
    id_store
        .write(0, bytemuck_i32(&ids))
        .expect("the ids land");
    let out_bytes = u64::from(rows) * u64::from(hidden) * 2;
    let out = Buffer::zeroed(&device, out_bytes).expect("the output reserves");

    let ids_h = handles.bind(&id_store, 0, 4 * u64::from(rows)).expect("ids");
    let table_h = handles
        .bind(&table, 0, table_bytes.len() as u64)
        .expect("table");
    let out_h = handles.bind(&out, 0, out_bytes).expect("out");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::layout::embed(
            &sink,
            Tensor::new(ids_h, rows, 1, Dtype::I32),
            Tensor::new(table_h, vocab, hidden, Dtype::Bf16),
            vocab,
            Tensor::new(out_h, rows, hidden, Dtype::Bf16),
        )
        .expect("the embed encodes");
    }
    frame.commit().expect("the fire completes");

    let mut got = vec![0u8; out_bytes as usize];
    out.read(0, &mut got).expect("the output reads back");
    let got: Vec<f32> = got
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect();

    let want: Vec<f32> = ids
        .iter()
        .flat_map(|&id| (0..hidden).map(move |d| id as f32 * 10.0 + d as f32))
        .collect();
    assert_eq!(got, want, "the gather is the permutation the ids name");
    println!("layout.embed: {got:?}");
}

/// **THE UNADDRESSABLE ID, AT BOTH DENSE POINTS, IN ONE FIRE.**
///
/// `layout.embed` and `layout.embed_concat` are one shader body under two
/// stamps, and the stamp is exactly this: an id outside `[0, vocab)` reads
/// **row zero** at the plain point and writes **zero** at the concatenating
/// one. Neither answer is this plane's invention — they are what
/// `kernels-cuda`'s `::pie::layout::embed` (`layout.cuh`, `tid_raw >= 0 &&
/// tid_raw < vocab ? tid_raw : 0`) and `::pie::layout::embed_concat`
/// (`embed_concat.cuh`, `y[at] = (id < 0 || id >= vocab) ? 0 : ...`) do, and
/// the point of the gate is that the two PLANES agree per op, not that either
/// op is right in the abstract.
///
/// The concat's answer is the one that has to differ: its sixteen ids per
/// token are a HASH's output, and a head that hashed out of the table
/// contributing row zero sixteen times would be a made-up embedding rather
/// than an absent one.
///
/// Both landings are poisoned before the fire, so "zero" is a write and not an
/// untouched buffer.
#[test]
fn an_unaddressable_id_clamps_at_one_dense_point_and_zeros_at_the_other() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the dense embed bounds") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    let vocab = 6u32;
    let hidden = 4u32;

    // Row v holds `[v*10 + 0 .. v*10 + 3]` — the same table the permutation
    // test above reads, so a wrong row is a wrong number.
    let mut table_bytes = Vec::new();
    for v in 0..vocab {
        for d in 0..hidden {
            table_bytes.extend_from_slice(&bf16(v as f32 * 10.0 + d as f32));
        }
    }
    let mut table = Buffer::zeroed(&device, table_bytes.len() as u64).expect("the table reserves");
    table.write(0, &table_bytes).expect("the table lands");
    let table_h = handles
        .bind(&table, 0, table_bytes.len() as u64)
        .expect("table");

    // Four shapes of wrong beside two in-range ids: negative, one past the
    // last row, far past it, and `i32::MIN`, whose negation overflows.
    let ids: [i32; 6] = [4, -1, vocab as i32, 2, 1 << 20, i32::MIN];
    let mut id_store = Buffer::zeroed(&device, 4 * ids.len() as u64).expect("the ids reserve");
    id_store.write(0, bytemuck_i32(&ids)).expect("the ids land");

    // The plain point reads them as six rows of one id; the concatenating one
    // as three rows of two, landing an eight-wide row. Same ids, same table,
    // same total elements — only the stamp differs.
    let out_bytes = ids.len() as u64 * u64::from(hidden) * 2;
    let poison: Vec<u8> = (0..ids.len() * hidden as usize)
        .flat_map(|_| bf16(-1.0))
        .collect();

    let mut plain = Buffer::zeroed(&device, out_bytes).expect("the plain landing reserves");
    plain.write(0, &poison).expect("the poison lands");
    let mut concat = Buffer::zeroed(&device, out_bytes).expect("the concat landing reserves");
    concat.write(0, &poison).expect("the poison lands");

    let ids_h = handles
        .bind(&id_store, 0, 4 * ids.len() as u64)
        .expect("ids");
    let plain_h = handles.bind(&plain, 0, out_bytes).expect("plain out");
    let concat_h = handles.bind(&concat, 0, out_bytes).expect("concat out");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        let table_t = Tensor::new(table_h, vocab, hidden, Dtype::Bf16);
        kernels_metal::layout::embed(
            &sink,
            Tensor::new(ids_h, ids.len() as u32, 1, Dtype::I32),
            table_t,
            vocab,
            Tensor::new(plain_h, ids.len() as u32, hidden, Dtype::Bf16),
        )
        .expect("the plain embed encodes");
        kernels_metal::layout::embed_concat(
            &sink,
            Tensor::new(ids_h, ids.len() as u32 / 2, 2, Dtype::I32),
            table_t,
            vocab,
            Tensor::new(concat_h, ids.len() as u32 / 2, hidden * 2, Dtype::Bf16),
        )
        .expect("the concatenating embed encodes");
    }
    frame.commit().expect("the fire completes");

    let read = |buffer: &Buffer| -> Vec<f32> {
        let mut got = vec![0u8; out_bytes as usize];
        buffer.read(0, &mut got).expect("the output reads back");
        got.chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect()
    };
    let got_plain = read(&plain);
    let got_concat = read(&concat);

    // The concat's slices are the plain point's rows at another stride, so one
    // expectation walks both landings — and differs only in what an
    // unaddressable id is worth.
    let addressable = |id: i32| id >= 0 && (id as u32) < vocab;
    let want_plain: Vec<f32> = ids
        .iter()
        .flat_map(|&id| {
            let row = if addressable(id) { id } else { 0 };
            (0..hidden).map(move |d| row as f32 * 10.0 + d as f32)
        })
        .collect();
    let want_concat: Vec<f32> = ids
        .iter()
        .flat_map(|&id| {
            (0..hidden).map(move |d| {
                if addressable(id) {
                    id as f32 * 10.0 + d as f32
                } else {
                    0.0
                }
            })
        })
        .collect();

    assert_eq!(
        got_plain, want_plain,
        "`layout.embed` clamps an unaddressable id to row zero, as `layout.cuh` does"
    );
    assert_eq!(
        got_concat, want_concat,
        "`layout.embed_concat` writes zero for one, as `embed_concat.cuh` does"
    );
    assert!(
        got_plain.iter().chain(&got_concat).all(|v| *v != -1.0),
        "both landings were written, so a zero is an answer and not a leftover"
    );
    println!("dense embed bounds: ids {ids:?} clamp at one point and zero at the other");
}

/// The handle table's own two claims: a view past its buffer is refused, and
/// a fire's rows do not survive the fire.
#[test]
fn the_handle_table_refuses_a_view_past_its_buffer_and_rewinds_to_the_seal() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the handle table") else {
        return;
    };
    let buffer = Buffer::zeroed(&device, 1024).expect("a reservation");
    let handles = Handles::new();

    let load_lived = handles.bind(&buffer, 0, 1024).expect("the whole reservation");
    handles.seal();
    assert_eq!(handles.sealed(), 1);

    let fault = handles
        .bind(&buffer, 512, 1024)
        .expect_err("a view that leaves its buffer");
    assert!(
        fault.to_string().contains("1536"),
        "the refusal names the span it wanted: {fault}"
    );

    let fire_lived = handles.bind(&buffer, 256, 64).expect("a fire's row");
    assert_eq!(handles.len(), 2);
    assert_eq!(handles.get(fire_lived).map(|b| b.offset()), Some(256));

    handles.rewind();
    assert_eq!(handles.len(), 1, "the fire's rows are gone");
    assert!(handles.get(load_lived).is_some(), "the load's rows are not");
    assert!(handles.get(fire_lived).is_none());
    assert!(
        handles.get(u32::MAX).is_none(),
        "and `absent` resolves to nothing"
    );
}

/// **THE CORRECTION, AGAINST A HOST REFERENCE AND AGAINST ITSELF** (lane J,
/// alto adapter §6.3).
///
/// `linear.lora_correct` is the one entry on this plane whose two halves live
/// inside a single threadgroup — the `rank`-wide waist never leaves it
/// (`kernels_metal::linear::lora`'s own header) — so nothing between the
/// projection and the accumulate is observable from outside the fire. That
/// makes a numerical gate the only way to know the halves agree, and it is
/// why this one fires DETERMINISTIC PSEUDO-RANDOM `A` and `B` rather than a
/// pattern the kernel could get right by symmetry: an `A` read down the wrong
/// axis, a `B` read at the wrong stride and a waist accumulated in the wrong
/// order all produce numbers, and only one arrangement produces THESE.
///
/// Three claims in one fire:
///
/// * **the routed rows compute `y += B[a]·(A[a]·x)`** against a host
///   reference in f32, at bf16 tolerance;
/// * **a `-1` row is the base model, exactly.** The op is IN PLACE, so an
///   unrouted row must read back the value the trunk wrote, bit for bit —
///   which is the identity-without-a-merge the whole guarded-window design
///   rests on;
/// * **a zero `B` corrects by nothing, exactly**, which is the arithmetic
///   under `lora-probe`'s `adapter_scale = 0.0` parity claim.
#[test]
fn the_correction_adds_what_the_host_says_and_leaves_an_unrouted_row_alone() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the adapter correction") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    // Small enough to reference by hand, wide enough that a row/column
    // transposition of either bank changes the answer: `in` and `out` differ,
    // and so does `rank`.
    let rows = 4u32;
    let in_width = 6u32;
    let out_width = 5u32;
    let rank = 3u32;
    let adapters = 2u32;

    // Deterministic pseudo-random, and the same splitmix the guest surface
    // uses — so a failure here reads against `lora-probe`'s own fixture.
    let pattern = |i: u32, salt: u32, amp: f32| -> f32 {
        let mut x = (i ^ salt).wrapping_add(0x9e37_79b9);
        x ^= x >> 16;
        x = x.wrapping_mul(0x85eb_ca6b);
        x ^= x >> 13;
        x = x.wrapping_mul(0xc2b2_ae35);
        x ^= x >> 16;
        ((x % 10_000) as f32 / 10_000.0 - 0.5) * 2.0 * amp
    };
    // Everything is round-tripped through bf16 BEFORE the reference is taken,
    // so the host and the device multiply the same numbers and the only
    // tolerance left is the accumulation order.
    let round = |v: f32| f32::from_bits(u32::from(u16::from_le_bytes(bf16(v))) << 16);

    // `x` is `[rows, in]`; `y` starts as the trunk's own value.
    let x: Vec<f32> = (0..rows * in_width).map(|i| round(pattern(i, 0x1111, 1.0))).collect();
    let y0: Vec<f32> = (0..rows * out_width).map(|i| round(pattern(i, 0x2222, 2.0))).collect();
    // Adapter 0 is live; adapter 1's `B` is all zeros — the identity.
    let bank_a: Vec<f32> = (0..adapters * rank * in_width)
        .map(|i| round(pattern(i, 0x3333, 0.5)))
        .collect();
    let bank_b: Vec<f32> = (0..adapters * out_width * rank)
        .map(|i| match i >= out_width * rank {
            true => 0.0,
            false => round(pattern(i, 0x4444, 0.5)),
        })
        .collect();
    // Row 0 takes the live adapter, row 1 the zero-`B` one, row 2 is the base
    // model and row 3 takes the live one again.
    let routes: [i32; 4] = [0, 1, -1, 0];

    let stage = |values: &[f32]| -> Vec<u8> {
        values.iter().flat_map(|&v| bf16(v)).collect()
    };
    let x_bytes = stage(&x);
    let y_bytes = stage(&y0);
    let a_bytes = stage(&bank_a);
    let b_bytes = stage(&bank_b);

    let mut x_buf = Buffer::zeroed(&device, x_bytes.len() as u64).expect("x reserves");
    x_buf.write(0, &x_bytes).expect("x lands");
    let mut a_buf = Buffer::zeroed(&device, a_bytes.len() as u64).expect("A reserves");
    a_buf.write(0, &a_bytes).expect("A lands");
    let mut b_buf = Buffer::zeroed(&device, b_bytes.len() as u64).expect("B reserves");
    b_buf.write(0, &b_bytes).expect("B lands");
    let mut r_buf = Buffer::zeroed(&device, 4 * u64::from(rows)).expect("routes reserve");
    r_buf.write(0, bytemuck_i32(&routes)).expect("routes land");
    // **IN PLACE**: `y` carries the trunk's value in and the corrected value
    // out, which is the aliasing the dispatch arm relies on.
    let mut y_buf = Buffer::zeroed(&device, y_bytes.len() as u64).expect("y reserves");
    y_buf.write(0, &y_bytes).expect("y lands");

    let x_h = handles.bind(&x_buf, 0, x_bytes.len() as u64).expect("x");
    let a_h = handles.bind(&a_buf, 0, a_bytes.len() as u64).expect("A");
    let b_h = handles.bind(&b_buf, 0, b_bytes.len() as u64).expect("B");
    let r_h = handles.bind(&r_buf, 0, 4 * u64::from(rows)).expect("routes");
    let y_h = handles.bind(&y_buf, 0, y_bytes.len() as u64).expect("y");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::linear::lora::correct(
            &sink,
            Tensor::new(x_h, rows, in_width, Dtype::Bf16),
            // The weight table's `rows x width` reading of the bank shapes:
            // `[adapters, rank·in]` and `[adapters, out·rank]`.
            Tensor::new(a_h, adapters, rank * in_width, Dtype::Bf16),
            Tensor::new(b_h, adapters, out_width * rank, Dtype::Bf16),
            Tensor::new(r_h, rows, 1, Dtype::I32),
            Tensor::new(y_h, rows, out_width, Dtype::Bf16),
        )
        .expect("the correction encodes");
    }
    frame.commit().expect("the fire completes");

    let mut got = vec![0u8; y_bytes.len()];
    y_buf.read(0, &mut got).expect("y reads back");
    let got: Vec<f32> = got
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect();

    // ── THE HOST REFERENCE ──────────────────────────────────────────────
    for row in 0..rows as usize {
        let adapter = routes[row];
        for out in 0..out_width as usize {
            let base = y0[row * out_width as usize + out];
            let want = match adapter {
                // **BIT FOR BIT**, and the assertion says so: an unrouted row
                // is returned on before a bank is read, so its bytes are the
                // ones the trunk wrote and no tolerance applies.
                -1 => {
                    assert_eq!(
                        got[row * out_width as usize + out].to_bits(),
                        base.to_bits(),
                        "row {row} routes to the base model and column {out} moved"
                    );
                    continue;
                }
                a => {
                    let a = a as usize;
                    let waist: Vec<f32> = (0..rank as usize)
                        .map(|r| {
                            (0..in_width as usize)
                                .map(|k| {
                                    let at = a * (rank * in_width) as usize
                                        + r * in_width as usize
                                        + k;
                                    bank_a[at] * x[row * in_width as usize + k]
                                })
                                .sum::<f32>()
                        })
                        .collect();
                    let delta: f32 = (0..rank as usize)
                        .map(|r| {
                            let at = a * (out_width * rank) as usize + out * rank as usize + r;
                            bank_b[at] * waist[r]
                        })
                        .sum();
                    base + delta
                }
            };
            let saw = got[row * out_width as usize + out];
            // bf16 carries 8 mantissa bits; the store rounds once and the
            // f32 accumulation inside the threadgroup may order its adds
            // differently from this loop.
            let tol = 0.02 * want.abs().max(1.0);
            assert!(
                (saw - want).abs() <= tol,
                "row {row} (adapter {adapter}) column {out}: device {saw}, host {want}"
            );
        }
        // The zero-`B` adapter is the identity, and exactly so: every term of
        // the sum is a product with a positive zero.
        if adapter == 1 {
            for out in 0..out_width as usize {
                assert_eq!(
                    got[row * out_width as usize + out].to_bits(),
                    y0[row * out_width as usize + out].to_bits(),
                    "row {row} carries a zero-`B` adapter and column {out} moved; the \
                     correction is exactly zero, which is what `lora-probe`'s \
                     `adapter_scale = 0.0` parity rests on"
                );
            }
        }
    }
}

/// f32 → the two bytes of its bf16 truncation, little-endian.
fn bf16(v: f32) -> [u8; 2] {
    ((v.to_bits() >> 16) as u16).to_le_bytes()
}

/// An `i32` slice as the bytes the shell would stage.
fn bytemuck_i32(values: &[i32]) -> &[u8] {
    // SAFETY: `i32` has no padding and no invalid bit patterns, and the
    // slice's lifetime is the borrow's.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}
