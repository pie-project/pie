//! Whether this backend can reach the matrix unit, and what it is worth.
//!
//! # The claim this file exists to settle
//!
//! `serving.rs`'s llama.cpp comparison ends on a sentence that decides what is
//! worth working on next: *"the prefill's 4.2x is in an instruction this
//! backend cannot emit"*. llama.cpp's Metal backend announces
//! `simdgroup matrix mul. = true` and builds its GEMM on it; `quant/qmm_t.wgsl`
//! is a scalar-FMA tile. If that sentence is right, pp512 is closed by a
//! feature nobody has, and the honest thing to do is stop tuning tiles.
//!
//! It is wrong, and this file is the proof. Every layer of the path exists in
//! the versions already in this tree's lock file:
//!
//! | layer | what it offers | how this file knows |
//! | --- | --- | --- |
//! | the M4 Pro's Metal adapter | `EXPERIMENTAL_COOPERATIVE_MATRIX` | [`whether_this_adapter_has_a_matrix_unit_this_tree_can_reach`] |
//! | `naga 30`'s WGSL front end | `enable wgpu_cooperative_matrix`, `coop_mat8x8<T, Role>`, `coopLoad`/`coopLoadT`/`coopStoreT`/`coopMultiplyAdd` | it parses, below |
//! | `naga 30`'s MSL back end | `NagaCooperativeLoad` / `NagaCooperativeMultiplyAdd`, i.e. `simdgroup_matrix` | it runs, below |
//!
//! No browser, no Dawn, no `chromium_experimental_subgroup_matrix`, no
//! upstream patch. The one thing standing between this backend and the matrix
//! unit is `device.rs` asking for `wgpu::ExperimentalFeatures::disabled()`.
//!
//! # What the adapter actually offers, which is not what the Vulkan note said
//!
//! `serving.rs` recorded six shapes "including 16x16x16 F16 in, F32
//! accumulate", from a `VK_KHR_cooperative_matrix` adapter. **This machine
//! offers three, and all of them are 8x8x8**: F32xF32->F32, F16xF16->F16, and
//! F16xF16->F32. That is Metal's `simdgroup_matrix<T, 8, 8>` and nothing
//! wider, so a kernel written to the Vulkan note's 16x16x16 would not compile
//! here. The shape is the backend's, not the standard's, and a portable
//! cooperative kernel has to ask.
//!
//! # And then the same file on Vulkan, which found two more things
//!
//! Everything above was written and measured against Metal on an M4 Pro. Run
//! it against Vulkan on an RTX 4090 -- which advertises the same feature bit
//! and passes the first test unchanged -- and neither measurement worked, for
//! two unrelated reasons that both had to be fixed before a number appeared.
//!
//! 1. **`naga` 30.0.0 does not emit a cooperative store's operands.** The
//!    `coopStore`/`coopStoreT` arm of `src/front/wgsl/lower/mod.rs` pushes its
//!    statement without the `emitter.finish` / `emitter.start` pair that every
//!    other statement-producing builtin around it has, so an expression used
//!    ONLY by a store never lands in an `Emit` range and the SPIR-V back end
//!    aborts the PROCESS with `Expression [n] is not cached!`. That is a panic
//!    and not a `Result`, so no caller can catch it or skip past it.
//!    [`coop_wgsl`]'s doc carries the citation and the shader-side repair.
//!
//! 2. **The kernels hard-coded a shape this adapter does not have.** The
//!    section above says in as many words that *"the shape is the backend's,
//!    not the standard's, and a portable cooperative kernel has to ask"* --
//!    and then both kernels below asked for `coop_mat8x8` because that is all
//!    an M4 Pro has. The 4090 offers `16x16x16`, `16x8x16` and `16x8x8` and NO
//!    `8x8` at all, and an unimplemented `coopMultiplyAdd` on that driver is
//!    not a compile error, not a validation error and not a device loss: it
//!    leaves the accumulator alone and says nothing. Both kernels ran, stored,
//!    and wrote 1,572,864 zeroes. [`square_tile`] asks now, and both
//!    generators take the tile as a parameter.
//!
//! # What the matrix unit is worth on Vulkan, which is not what it is on Metal
//!
//! At the same `[m 512, n 3072, k 1024]`, at `coop_mat16x16`, every
//! spot-checked output bit-exact against an f32 CPU dot over all 1024 terms:
//!
//! | kernel | ms | TFLOP/s |
//! | --- | --- | --- |
//! | shipped `affine_qmm_t_..._bm_32_bn_64` | ~1.25 | 2.58 |
//! | f16 weights, 2x4 or 4x4 tiles of 16 | **0.072** | **44.7** |
//! | 4-bit affine, 4 simdgroups | **0.100** | **32.1** |
//!
//! **17x and 12x**, against 2.4x and 2.3x on the M4 Pro. That is not a better
//! kernel, it is a different machine: a 4090's tensor cores are most of its
//! arithmetic and a scalar-FMA tile reaches almost none of it, where an M4
//! Pro's simdgroup matmul is a smaller fraction of a smaller number. It also
//! moves the register-blocking finding: 2x4 and 4x4 tiles of 16 tie at the
//! top within run-to-run noise -- 32x64 and 64x64 outputs -- while 8x4 falls
//! off a cliff to ~8 TFLOP/s. That is the same register-file wall the Metal
//! sweep found, hit at half the tile count because each tile is now four
//! times the accumulator.
//!
//! # These are `#[ignore]`d and the rest of this crate's GPU tests are not
//!
//! `device.rs`'s header argues that an ignored test is one nobody runs, and it
//! is right. The exception here is not about hardware: enabling the feature
//! costs an `unsafe` token whose contract is *"there may be UB-containing bugs
//! in these apis"*, and a suite that opens an experimental device on every
//! `cargo test` spends the whole crate's reliability on a measurement. So the
//! correctness test and the benchmark are both `#[ignore = "measurement"]` and
//! this file's job is to be re-runnable, not to be a gate.

#![cfg(feature = "native")]

/// The three things a cooperative-matrix kernel needs, asked of the adapter in
/// front of this suite.
///
/// Prints rather than asserts the shapes: this is a machine-dependent fact and
/// a test that pinned 8x8x8 would fail on the next adapter for being right.
/// What it DOES assert is the feature bit, because the whole of this file's
/// argument is that the bit is present and unclaimed.
///
/// M4 Pro, `wgpu 30`, Metal:
///
/// ```text
/// Metal / Apple M4 Pro / IntegratedGpu  coop_matrix=true
///     m 8 n 8 k 8  ab F32 cr F32
///     m 8 n 8 k 8  ab F16 cr F16
///     m 8 n 8 k 8  ab F16 cr F32
/// ```
///
/// RTX 4090, `wgpu 30`, Vulkan -- six shapes, no `f32` inputs, and NOTHING
/// square below 16, which is the fact [`square_tile`] exists to carry:
///
/// ```text
/// Vulkan / NVIDIA GeForce RTX 4090 / DiscreteGpu  coop_matrix=true
///     m 16 n 16 k 16  ab F16 cr F16
///     m 16 n 8 k 16  ab F16 cr F16
///     m 16 n 8 k 8  ab F16 cr F16
///     m 16 n 16 k 16  ab F16 cr F32
///     m 16 n 8 k 16  ab F16 cr F32
///     m 16 n 8 k 8  ab F16 cr F32
/// ```
#[test]
#[ignore = "measurement"]
fn whether_this_adapter_has_a_matrix_unit_this_tree_can_reach() {
    let Some(adapter) = adapter() else {
        return;
    };
    let info = adapter.get_info();
    let has = adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
    println!(
        "{:?} / {} / {:?}  coop_matrix={has}",
        info.backend, info.name, info.device_type
    );
    for c in adapter.cooperative_matrix_properties() {
        println!(
            "    m {} n {} k {}  ab {:?} cr {:?}",
            c.m_size, c.n_size, c.k_size, c.ab_type, c.cr_type
        );
    }
    assert!(
        has,
        "the argument in this file's header is about an adapter that offers this"
    );
    assert!(
        adapter.features().contains(wgpu::Features::SHADER_F16),
        "the only mixed-precision shape this machine offers takes f16 in"
    );
}

/// The square `f16` in, `f32` accumulate tile this adapter offers, widest
/// first, or `None` if it offers no square one at all.
///
/// # Why asking is not optional, and what happens when a kernel does not
///
/// This file was written on an M4 Pro, where the only shapes are `8x8x8`, and
/// it hard-coded `coop_mat8x8` in both of its kernels. Run the same shaders
/// against Vulkan on an RTX 4090 and they compile, bind, dispatch, and store
/// -- and every one of the 1,572,864 outputs is ZERO. The 4090 offers
/// `16x16x16`, `16x8x16` and `16x8x8` and NO `8x8` at all, and an unsupported
/// `coopMultiplyAdd` on that driver is not a compile error, not a validation
/// error and not a device loss. It yields a zero accumulator and says nothing.
///
/// It was pinned down by filling the destination with `1.0` before the
/// dispatch: the sentinel comes back as `0.0` everywhere, so the cooperative
/// STORE is working perfectly and writing an accumulator that the multiply-add
/// never touched. Regenerating the identical kernel at `coop_mat16x16` fills
/// the same buffer with real products on the same adapter and the same driver.
///
/// So the header's warning -- *"the shape is the backend's, not the
/// standard's, and a portable cooperative kernel has to ask"* -- was written
/// one paragraph above two kernels that did not ask. They ask now, and the
/// generators below take the tile as a parameter rather than a constant.
///
/// Widest first because the accumulator is what the register blocking is
/// spending, and a 16-wide tile does four times the arithmetic of an 8-wide
/// one for the same two loads.
fn square_tile(adapter: &wgpu::Adapter) -> Option<u32> {
    let mut best = None;
    for c in adapter.cooperative_matrix_properties() {
        if c.m_size != c.n_size || c.n_size != c.k_size {
            continue;
        }
        if c.ab_type != wgpu::CooperativeScalarType::F16
            || c.cr_type != wgpu::CooperativeScalarType::F32
        {
            continue;
        }
        if best.is_none_or(|b| c.m_size > b) {
            best = Some(c.m_size);
        }
    }
    best
}

/// A cooperative-matrix GEMM at a projection's exact shape, checked against a
/// scalar dot product computed on the CPU, and timed.
///
/// # The shape
///
/// `[m 512, n 3072, k 1024]` is qwen3-0.6B's gate and up projection at the
/// prompt length `serving.rs` benchmarks -- hidden 1024, ffn 3072, 512 rows.
/// It is chosen so the number here divides into a number that file already
/// measured, rather than being a GEMM benchmark in the abstract.
///
/// # The answer
///
/// | kernel | ms | TFLOP/s |
/// | --- | --- | --- |
/// | shipped `affine_qmm_t_..._bm_32_bn_64` | ~1.25 | 2.58 |
/// | `coop_mat8x8` f16->f32, this file | **0.527** | **6.11** |
///
/// **2.4x, with every spot-checked output bit-exact** against an f32 CPU dot
/// over all 1024 terms. The shipped figure is rectangles 46 and 47 of
/// `where_a_prefills_time_goes_across_its_plan`, which are this exact GEMM.
///
/// That 2.4x is an UPPER BOUND and the reason is the honest caveat here: this
/// kernel reads f16 weights and the shipped one reads 4-bit affine. The
/// weights are 6 MB here against 1.5 MB there, so a real cooperative kernel
/// has to dequantise into the matrix unit's operand and will pay for it. At
/// m=512 the GEMM is compute-bound enough that the comparison means something;
/// at decode it would not.
///
/// # Two findings about writing these, both of which cost more than the tile
///
/// **Accumulators must be NAMED, never indexed.** `array<coop_mat8x8<f32, C>,
/// 8>` subscripted by a loop variable reads 0.76 TFLOP/s; the identical
/// arithmetic in eight separate `var`s reads 4.86. **6.4x**, and the cause is
/// that an indexed array of matrices cannot live in registers, so every
/// `coopMultiplyAdd` round-trips a simdgroup matrix through stack. This is why
/// the shader below is generated with a nested loop in the test rather than
/// written with one in WGSL.
///
/// **Register blocking is worth 4x on its own.** One 8x8 tile per simdgroup
/// issues two loads for one multiply-accumulate and reads 1.46 TFLOP/s. The
/// sweep, all bit-exact:
///
/// | tiles down x across | output block | ms | TFLOP/s |
/// | --- | --- | --- | --- |
/// | 1 x 1 | 8x8 | 2.210 | 1.46 |
/// | 1 x 8 | 8x64 | 0.847 | 3.80 |
/// | 2 x 4 | 16x32 | 0.663 | 4.86 |
/// | 2 x 8 | 16x64 | 0.564 | 5.72 |
/// | 8 x 4 | 64x32 | 0.607 | 5.31 |
/// | 4 x 8 | 32x64 | 0.639 | 5.04 |
/// | **4 x 4** | **32x32** | **0.527** | **6.11** |
///
/// Square and 16 accumulators, turning over on both sides -- fewer and the
/// loads dominate, more and the register file does. No workgroup staging at
/// all: this is a naive tile by the standards of `qmm_t.wgsl`, which stages
/// both operands and vectorises them, and it is still 2.4x ahead.
///
/// # What it means for the gap
///
/// A layer at the shipped tile is ~12.4 ms: attention 5.5, seven projections
/// ~6.4. Giving the projections 2.4x takes the layer to ~8.7 and pp512 from
/// 1436 to ~2050 tok/s. That alone does NOT reach llama.cpp's 6076 -- but the
/// attention is the other half of the same instruction, which is what
/// FlashAttention on a tensor core is, and 5.5 ms is the single largest
/// rectangle in the layer. Both halves through the matrix unit is the shape of
/// a 3x, not of a 1.3x.
///
/// So `serving.rs`'s "an instruction this backend cannot emit" is retired.
/// It can emit it, today, on the pinned version. What it cannot do is emit it
/// without `device.rs` signing wgpu's `unsafe` experimental token, and THAT is
/// the decision this file hands over -- a judgement about shipping an
/// experimental path, not a fact about the standard.
///
/// Run with `--ignored --nocapture --release`.
#[test]
#[ignore = "measurement"]
fn what_the_matrix_unit_is_worth_at_a_projections_shape() {
    if cfg!(debug_assertions) {
        panic!("a matrix unit timed in debug measures the profile");
    }
    let Some(adapter) = adapter() else {
        return;
    };
    let want = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16;
    if !adapter.features().contains(want) {
        driver_wgpu::skip::inapplicable(
            "this adapter offers no cooperative matrix, so nothing here is measured",
        );
        return;
    }
    let Some(tile) = square_tile(&adapter) else {
        driver_wgpu::skip::inapplicable(
            "this adapter has a matrix unit but no SQUARE f16 -> f32 shape, and \
             both kernels here are written on a square tile. See `square_tile`",
        );
        return;
    };
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: want,
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
        // The token this whole file is about. Sound here because the device
        // is opened, measured and dropped inside one ignored test.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
    }))
    .expect("a device opens with the feature the adapter advertised");

    const M: u32 = 512;
    const N: u32 = 3072;
    const K: u32 = 1024;
    // Small integers, exactly representable in f16 and summed exactly in f32
    // over 1024 terms, so `worst abs err 0` is a real claim about the matrix
    // unit's arithmetic rather than a tolerance nobody chose.
    let a: Vec<f32> = (0..M * K).map(|i| ((i % 7) as f32) - 3.0).collect();
    let b: Vec<f32> = (0..N * K).map(|i| ((i % 5) as f32) - 2.0).collect();

    println!("\n  [m {M}, n {N}, k {K}] -- qwen3-0.6b's gate/up projection at pp512");
    println!("  the adapter's widest square f16 -> f32 tile is {tile}x{tile}");
    let mut best_of_all = (f64::INFINITY, 0u32, 0u32);
    for (rm, rn) in [(1u32, 1u32), (1, 8), (2, 4), (2, 8), (8, 4), (4, 8), (4, 4)] {
        if !M.is_multiple_of(rm * tile) || !N.is_multiple_of(rn * tile) {
            continue;
        }
        let Some(ms) = run_one(&device, &queue, rm, rn, tile, M, N, K, &a, &b) else {
            driver_wgpu::skip::inapplicable(
                "this adapter enumerated a square shape its driver then \
                 declined: the destination comes back entirely zero, which is \
                 what an unimplemented `coopMultiplyAdd` looks like. See \
                 `square_tile`. Nothing here is measured on such an adapter",
            );
            return;
        };
        let flop = 2.0 * f64::from(M) * f64::from(N) * f64::from(K);
        println!(
            "    {rm} x {rn} tiles  {:>3}x{:<3} block  {ms:7.3} ms  {:5.2} TFLOP/s",
            rm * tile,
            rn * tile,
            flop / (ms / 1000.0) / 1e12
        );
        if ms < best_of_all.0 {
            best_of_all = (ms, rm, rn);
        }
    }
    let (ms, rm, rn) = best_of_all;
    let flop = 2.0 * f64::from(M) * f64::from(N) * f64::from(K);
    println!(
        "\n  best {rm}x{rn}: {ms:.3} ms, {:.2} TFLOP/s, against the shipped kernel's ~1.25 ms",
        flop / (ms / 1000.0) / 1e12
    );
}

/// Builds, verifies and times one blocking shape. Returns the fastest
/// milliseconds of five rounds of fifty dispatches.
///
/// Verification is not optional and not sampled loosely: a matrix unit that
/// transposed an operand would still produce a plausible TFLOP/s, and the
/// whole point of the number is that it is the same answer. Twenty-five
/// outputs spread across the corners and interior are recomputed in f32 on the
/// CPU over all `k` terms and required to match EXACTLY.
///
/// # `None` when the accumulator comes back untouched
///
/// Returns `None`, rather than failing, when the whole destination reads back
/// as zero. That is not a near-miss and it is not a tolerance -- it is the
/// signature of a `coopMultiplyAdd` at a shape the driver does not implement,
/// which [`square_tile`]'s doc describes: no compile error, no validation
/// error, no device loss, just an accumulator nobody wrote. `tile` is asked of
/// the adapter precisely so this cannot happen, so reaching here means the
/// adapter answered a shape its driver then declined, and the honest report is
/// a skip rather than a very fast `TFLOP/s` for a kernel that computed
/// nothing.
#[allow(clippy::too_many_arguments)]
fn run_one(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rm: u32,
    rn: u32,
    tile: u32,
    m: u32,
    n: u32,
    k: u32,
    a: &[f32],
    b: &[f32],
) -> Option<f64> {
    use wgpu::util::DeviceExt;

    let ab = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("a"),
        contents: &halved(a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("b"),
        contents: &halved(b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let cb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("c"),
        size: u64::from(m) * u64::from(n) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let pb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: &[m, n, k, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: u64::from(m) * u64::from(n) * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("coop"),
        source: wgpu::ShaderSource::Wgsl(coop_wgsl(rm, rn, tile).into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("coop"),
        layout: None,
        module: &module,
        entry_point: Some("coop"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ab.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bb.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cb.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: pb.as_entire_binding(),
            },
        ],
    });

    let fire = |reps: u32| {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            for _ in 0..reps {
                pass.dispatch_workgroups(n / (rn * tile), m / (rm * tile), 1);
            }
        }
        queue.submit([enc.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the queue drains");
    };

    fire(1);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(&cb, 0, &read, 0, u64::from(m) * u64::from(n) * 4);
    queue.submit([enc.finish()]);
    read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the map resolves");
    let got: Vec<f32> = {
        let view = read.slice(..).get_mapped_range().expect("it mapped");
        view.chunks_exact(4)
            .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    };
    read.unmap();

    if got.iter().all(|v| *v == 0.0) {
        return None;
    }

    for r in [0usize, 1, 17, 255, m as usize - 1] {
        for c in [0usize, 1, 33, 1000, n as usize - 1] {
            let mut want = 0.0f32;
            for kk in 0..k as usize {
                want += a[r * k as usize + kk] * b[c * k as usize + kk];
            }
            let is = got[r * n as usize + c];
            assert_eq!(
                is, want,
                "the matrix unit's answer at ({r}, {c}) at {rm}x{rn}; a wrong \
                 transpose still produces a plausible TFLOP/s"
            );
        }
    }

    const REPS: u32 = 50;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        fire(REPS);
        let ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(REPS);
        if ms < best {
            best = ms;
        }
    }
    Some(best)
}

/// The same GEMM on THIS TREE'S ACTUAL WEIGHTS: 4-bit affine, group 64,
/// dequantised into workgroup memory and fed to the matrix unit.
///
/// # Why this test decides whether any of it is worth building
///
/// [`what_the_matrix_unit_is_worth_at_a_projections_shape`] reads 2.4x and
/// says so under a caveat that could have eaten the whole result: it feeds the
/// unit f16 weights, and this backend does not have f16 weights. It has 4-bit
/// affine, 6 MB against 1.5 at this shape, and a real kernel has to turn one
/// into the other before the matrix unit can see it. If dequantisation costs
/// what the matrix unit saves then the 2.4x is a benchmark artefact and the
/// honest move is to leave `qmm_t.wgsl` alone.
///
/// It does not. The staged tile is `BN 32 x BK 64` f16 -- 4 KB, and BK is one
/// whole quantisation group so no group is ever split across two stages --
/// filled by unpacking eight nibbles per word and applying `n * scale + zero`.
///
/// | | ms | TFLOP/s |
/// | --- | --- | --- |
/// | shipped `affine_qmm_t_..._bm_32_bn_64` | ~1.25 | 2.58 |
/// | coop, 4-bit, 1 simdgroup a workgroup | 0.719 | 4.48 |
/// | coop, 4-bit, 2 simdgroups | 0.601 | 5.36 |
/// | coop, 4-bit, 4 simdgroups | 0.565 | 5.70 |
/// | **coop, 4-bit, 8 simdgroups** | **0.550** | **5.86** |
/// | coop, f16 weights, no dequant at all | 0.524 | 6.14 |
///
/// **2.3x against the shipped kernel, on the same encoding, every
/// spot-checked output exact.** The f16 row is the ceiling and the 8-simdgroup
/// row is within 5% of it, so at that width dequantisation is FREE.
///
/// # And the reason it becomes free is the finding
///
/// Dequantising costs `BN * BK` work and produces `BM * BN` outputs, so its
/// share of a result is `BK / BM` -- it does not depend on `BN` at all, and
/// the only lever is how many ROWS share one staged tile. One simdgroup owns
/// 32 rows and pays 27% for the unpacking. Widening the accumulator block to
/// get more rows does not work; the sweep in the sibling test already shows
/// 8x4 tiles losing to 4x4 because the register file runs out.
///
/// So the rows come from putting more SIMDGROUPS in the workgroup instead.
/// Each keeps the 4x4 accumulator block that measured best, owns its own 32
/// rows, and reads the same `ws` the others do -- eight of them amortise one
/// unpacking over 256 rows and the cost vanishes. That is a workgroup-shape
/// change with no cost in registers, which is exactly the axis a
/// register-blocked kernel has left.
///
/// This also answers the sibling test's open question in the other direction:
/// the quantised kernel wants a WIDE workgroup where the f16 one did not care,
/// because only the quantised one has a fixed per-tile cost to amortise.
///
/// Run with `--ignored --nocapture --release`.
#[test]
#[ignore = "measurement"]
fn what_the_matrix_unit_is_worth_on_this_trees_actual_weights() {
    if cfg!(debug_assertions) {
        panic!("a matrix unit timed in debug measures the profile");
    }
    let Some(adapter) = adapter() else {
        return;
    };
    let want = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16;
    if !adapter.features().contains(want) {
        driver_wgpu::skip::inapplicable(
            "this adapter offers no cooperative matrix, so nothing here is measured",
        );
        return;
    }
    let Some(tile) = square_tile(&adapter) else {
        driver_wgpu::skip::inapplicable(
            "this adapter has a matrix unit but no SQUARE f16 -> f32 shape, and \
             both kernels here are written on a square tile. See `square_tile`",
        );
        return;
    };
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: want,
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
    }))
    .expect("a device opens with the feature the adapter advertised");

    const M: u32 = 512;
    const N: u32 = 3072;
    const K: u32 = 1024;
    const GS: u32 = 64;

    let a: Vec<f32> = (0..M * K).map(|i| ((i % 7) as f32) - 3.0).collect();
    let nib: Vec<u32> = (0..N * K).map(|i| i % 13 % 16).collect();
    let groups = (K / GS) as usize;
    let scale: Vec<f32> = (0..N as usize * groups)
        .map(|g| 0.25 + ((g % 3) as f32) * 0.25)
        .collect();
    let zero: Vec<f32> = (0..N as usize * groups)
        .map(|g| -1.0 - ((g % 2) as f32))
        .collect();

    println!("\n  [m {M}, n {N}, k {K}] affine 4-bit, group {GS} -- the shipped encoding");
    println!("  the adapter's widest square f16 -> f32 tile is {tile}x{tile}");
    for sgs in [1u32, 2, 4, 8] {
        if !M.is_multiple_of(sgs * 4 * tile) || !N.is_multiple_of(4 * tile) {
            continue;
        }
        let Some(ms) = run_quantised(
            &device, &queue, sgs, tile, M, N, K, GS, &a, &nib, &scale, &zero,
        ) else {
            driver_wgpu::skip::inapplicable(
                "this adapter enumerated a square shape its driver then \
                 declined: the destination comes back entirely zero, which is \
                 what an unimplemented `coopMultiplyAdd` looks like. See \
                 `square_tile`. Nothing here is measured on such an adapter",
            );
            return;
        };
        let flop = 2.0 * f64::from(M) * f64::from(N) * f64::from(K);
        println!(
            "    {sgs} simdgroup(s), {:>3} rows a workgroup  {ms:7.3} ms  {:5.2} TFLOP/s",
            sgs * 4 * tile,
            flop / (ms / 1000.0) / 1e12
        );
    }
    println!("  against the shipped quantised kernel's ~1.25 ms at this shape");
}

/// Builds, verifies and times the quantised kernel at `sgs` simdgroups.
///
/// The CPU reference dequantises THROUGH f16, exactly as the shader does, so
/// the equality asserted below is about the matrix unit and the staging rather
/// than about rounding. A reference that stayed in f32 would force a tolerance
/// and a tolerance would hide a transposed operand.
///
/// `None` for the reason [`run_one`]'s doc sets out at length: an all-zero
/// destination means the cooperative store landed nowhere, and timing a kernel
/// that stores nothing is timing nothing.
#[allow(clippy::too_many_arguments)]
fn run_quantised(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sgs: u32,
    tile: u32,
    m: u32,
    n: u32,
    k: u32,
    gs: u32,
    a: &[f32],
    nib: &[u32],
    scale: &[f32],
    zero: &[f32],
) -> Option<f64> {
    use wgpu::util::DeviceExt;

    let groups = (k / gs) as usize;
    let mut packed: Vec<u8> = Vec::with_capacity((n * k / 2) as usize);
    for row in 0..n as usize {
        for w in 0..(k / 8) as usize {
            let mut v = 0u32;
            for t in 0..8 {
                v |= nib[row * k as usize + w * 8 + t] << (t * 4);
            }
            packed.extend_from_slice(&v.to_le_bytes());
        }
    }
    let storage = wgpu::BufferUsages::STORAGE;
    let mk = |data: &[u8], usage| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage,
        })
    };
    let ab = mk(&halved(a), storage);
    let bq = mk(&packed, storage);
    let bs = mk(&halved(scale), storage);
    let bz = mk(&halved(zero), storage);
    let pb = mk(
        &[m, n, k, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
        wgpu::BufferUsages::UNIFORM,
    );
    let cb = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(m) * u64::from(n) * 4,
        usage: storage | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(m) * u64::from(n) * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("coop_q"),
        source: wgpu::ShaderSource::Wgsl(quantised_wgsl(sgs, tile).into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("coop_q"),
        layout: None,
        module: &module,
        entry_point: Some("coop"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ab.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bq.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cb.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: pb.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: bs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: bz.as_entire_binding(),
            },
        ],
    });

    let fire = |reps: u32| {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            for _ in 0..reps {
                pass.dispatch_workgroups(n / (4 * tile), m / (sgs * 4 * tile), 1);
            }
        }
        queue.submit([enc.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the queue drains");
    };

    fire(1);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(&cb, 0, &read, 0, u64::from(m) * u64::from(n) * 4);
    queue.submit([enc.finish()]);
    read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the map resolves");
    let got: Vec<f32> = {
        let view = read.slice(..).get_mapped_range().expect("it mapped");
        view.chunks_exact(4)
            .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    };
    read.unmap();

    if got.iter().all(|v| *v == 0.0) {
        return None;
    }

    for r in [0usize, 1, 17, 255, m as usize - 1] {
        for c in [0usize, 1, 33, 1000, n as usize - 1] {
            let mut want = 0.0f32;
            for kk in 0..k as usize {
                let g = c * groups + kk / gs as usize;
                let w = nib[c * k as usize + kk] as f32 * scale[g] + zero[g];
                want += a[r * k as usize + kk] * through_f16(w);
            }
            assert_eq!(
                got[r * n as usize + c],
                want,
                "the dequantising kernel's answer at ({r}, {c}) with {sgs} simdgroups"
            );
        }
    }

    const REPS: u32 = 50;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        fire(REPS);
        let ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(REPS);
        if ms < best {
            best = ms;
        }
    }
    Some(best)
}

/// The dequantising kernel, at `sgs` simdgroups sharing one staged tile.
///
/// `BK` is 64 and that is not a tuning choice: it is the quantisation group,
/// so a staged tile holds whole groups and the unpacking loop reads one scale
/// and one zero per column instead of testing for a boundary.
///
/// Written out rather than looped for the reason the sibling generator gives:
/// an indexed array of cooperative matrices spills to stack and costs 6.4x.
///
/// The store's stride and destinations are named before the `k0` loop for the
/// reason [`coop_wgsl`]'s doc gives at length: `naga` 30.0.0 does not emit the
/// operands of a cooperative store, and the panic that follows is not one a
/// caller can catch.
fn quantised_wgsl(sgs: u32, tile: u32) -> String {
    let lanes = sgs * 32;
    // Four tiles down per simdgroup, which is the accumulator block the sweep
    // in the sibling test settled on. `BN` is four tiles across for the same
    // reason, so the staged tile is exactly what one simdgroup consumes.
    let rows = sgs * 4 * tile;
    let bn = 4 * tile;
    let mat = format!("coop_mat{tile}x{tile}");
    let mut s = format!(
        "enable f16;\n\
         enable wgpu_cooperative_matrix;\n\
         struct P {{ m: u32, n: u32, k: u32, pad: u32 }};\n\
         @group(0) @binding(0) var<storage, read> a: array<f16>;\n\
         @group(0) @binding(1) var<storage, read> bq: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> c: array<f32>;\n\
         @group(0) @binding(3) var<uniform> p: P;\n\
         @group(0) @binding(4) var<storage, read> bs: array<f16>;\n\
         @group(0) @binding(5) var<storage, read> bz: array<f16>;\n\
         const BN = {bn}u;\n\
         const BK = 64u;\n\
         var<workgroup> ws: array<f16, BN * BK>;\n\
         @compute @workgroup_size({lanes})\n\
         fn coop(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{\n\
         \x20   let sg = li / 32u;\n\
         \x20   let row0 = wg.y * {rows}u + sg * {sg_rows}u;\n\
         \x20   let col0 = wg.x * BN;\n",
        sg_rows = 4 * tile
    );
    s.push_str("    let ldc = p.n;\n");
    for i in 0..4 {
        for j in 0..4 {
            s.push_str(&format!(
                "    let at{i}_{j} = (row0 + {}u) * ldc + col0 + {}u;\n",
                i * tile,
                j * tile
            ));
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            s.push_str(&format!("    var acc{i}_{j}: {mat}<f32, C>;\n"));
        }
    }
    s.push_str(&format!(
        "    let words_per_row = p.k / 8u;\n\
         \x20   let groups_per_row = p.k / 64u;\n\
         \x20   for (var k0 = 0u; k0 < p.k; k0 = k0 + BK) {{\n\
         \x20       workgroupBarrier();\n\
         \x20       for (var w = li; w < BN * BK / 8u; w = w + {lanes}u) {{\n\
         \x20           let col = w / (BK / 8u);\n\
         \x20           let wi = w % (BK / 8u);\n\
         \x20           let n_g = col0 + col;\n\
         \x20           let packed = bq[n_g * words_per_row + k0 / 8u + wi];\n\
         \x20           let gi = n_g * groups_per_row + k0 / 64u;\n\
         \x20           let scale = bs[gi];\n\
         \x20           let zero = bz[gi];\n\
         \x20           let base = col * BK + wi * 8u;\n\
         \x20           for (var t = 0u; t < 8u; t = t + 1u) {{\n\
         \x20               let nib = (packed >> (t * 4u)) & 15u;\n\
         \x20               ws[base + t] = f16(f32(nib)) * scale + zero;\n\
         \x20           }}\n\
         \x20       }}\n\
         \x20       workgroupBarrier();\n\
         \x20       for (var kk = 0u; kk < BK; kk = kk + {tile}u) {{\n"
    ));
    for i in 0..4 {
        s.push_str(&format!(
            "            let av{i} = coopLoadT<{mat}<f16, A>>(&a[(row0 + {}u) * p.k + k0 + kk], p.k);\n",
            i * tile
        ));
    }
    for j in 0..4 {
        s.push_str(&format!(
            "            let bv{j} = coopLoad<{mat}<f16, B>>(&ws[{}u * BK + kk], BK);\n",
            j * tile
        ));
        for i in 0..4 {
            s.push_str(&format!(
                "            acc{i}_{j} = coopMultiplyAdd(av{i}, bv{j}, acc{i}_{j});\n"
            ));
        }
    }
    s.push_str("        }\n    }\n");
    for i in 0..4 {
        for j in 0..4 {
            s.push_str(&format!(
                "    coopStoreT(acc{i}_{j}, &c[at{i}_{j}], ldc);\n"
            ));
        }
    }
    s.push_str("}\n");
    s
}

/// A round trip through IEEE binary16, which is what the shader's `ws` does to
/// every dequantised weight.
fn through_f16(x: f32) -> f32 {
    let b = halved(&[x]);
    let h = u16::from_le_bytes([b[0], b[1]]);
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    if exp == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

/// The kernel, generated at `rm` x `rn` 8x8 tiles.
///
/// GENERATED and not written because of the finding in
/// [`what_the_matrix_unit_is_worth_at_a_projections_shape`]'s doc: the
/// accumulators have to be separate named `var`s, so the unrolling is the
/// source rather than something a compiler is asked to do.
///
/// `coopLoadT` for A and `coopLoad` for B because the weights are stored
/// `[n, k]` -- a projection's matrix is transposed in this tree, which is what
/// `affine_qmm_T` means -- so B's 8x8 block at `(col, kk)` is already in the
/// unit's canonical order and A's is not.
///
/// # Why the store's stride and destinations are named `let`s before the loop
///
/// They read like a stylistic hoist and they are not. `naga` 30.0.0's WGSL
/// front end lowers `coopStore`/`coopStoreT` WITHOUT flushing its expression
/// emitter first, which every other statement-producing builtin in that file
/// does -- `textureStore` at `src/front/wgsl/lower/mod.rs:3436` opens with the
/// three lines of `emitter.finish` / `emitter.start` that the `coopStore` arm
/// at 3789 does not have. The consequence is that any expression whose ONLY
/// use is a cooperative store never lands in an `Emit` range, so the SPIR-V
/// back end finds no id for it and aborts:
///
/// ```text
/// internal error: entered unreachable code: Expression [62] is not cached!
///   naga-30.0.0/src/back/spv/block.rs:4160     <- the stride operand
///   naga-30.0.0/src/back/spv/index.rs:550      <- the destination's index
/// ```
///
/// That is a panic and not a `Result`, so it is not something a caller can
/// catch, degrade on, or `skip` past: written the obvious way this file kills
/// the test process on every adapter that HAS a matrix unit, which is the only
/// kind of adapter it is for. Naming every operand of every store above the
/// `k` loop is the repair that lives on our side of the line -- a loop is a
/// statement, the front end DOES flush its emitter when it opens one, so the
/// named values are emitted and have ids by the time the stores below ask.
///
/// Delete all of this the day that upstream arm grows its two flush lines.
/// The shader is strictly worse for the workaround and nothing else wants it.
fn coop_wgsl(rm: u32, rn: u32, tile: u32) -> String {
    let mut s = String::from(
        "enable f16;\n\
         enable wgpu_cooperative_matrix;\n\
         struct P { m: u32, n: u32, k: u32, pad: u32 };\n\
         @group(0) @binding(0) var<storage, read> a: array<f16>;\n\
         @group(0) @binding(1) var<storage, read> b: array<f16>;\n\
         @group(0) @binding(2) var<storage, read_write> c: array<f32>;\n\
         @group(0) @binding(3) var<uniform> p: P;\n\
         @compute @workgroup_size(32)\n\
         fn coop(@builtin(workgroup_id) wg: vec3<u32>) {\n",
    );
    let mat = format!("coop_mat{tile}x{tile}");
    s.push_str(&format!("    let row0 = wg.y * {}u;\n", rm * tile));
    s.push_str(&format!("    let col0 = wg.x * {}u;\n", rn * tile));
    // Every operand of every store below, named here so that the `for` that
    // follows flushes them into an `Emit` range. See this function's doc.
    s.push_str("    let ldc = p.n;\n");
    for i in 0..rm {
        for j in 0..rn {
            s.push_str(&format!(
                "    let at{i}_{j} = (row0 + {}u) * ldc + col0 + {}u;\n",
                i * tile,
                j * tile
            ));
        }
    }
    for i in 0..rm {
        for j in 0..rn {
            s.push_str(&format!("    var acc{i}_{j}: {mat}<f32, C>;\n"));
        }
    }
    s.push_str(&format!(
        "    for (var kk = 0u; kk < p.k; kk = kk + {tile}u) {{\n"
    ));
    for i in 0..rm {
        s.push_str(&format!(
            "        let av{i} = coopLoadT<{mat}<f16, A>>(&a[(row0 + {}u) * p.k + kk], p.k);\n",
            i * tile
        ));
    }
    for j in 0..rn {
        s.push_str(&format!(
            "        let bv{j} = coopLoad<{mat}<f16, B>>(&b[(col0 + {}u) * p.k + kk], p.k);\n",
            j * tile
        ));
        for i in 0..rm {
            s.push_str(&format!(
                "        acc{i}_{j} = coopMultiplyAdd(av{i}, bv{j}, acc{i}_{j});\n"
            ));
        }
    }
    s.push_str("    }\n");
    for i in 0..rm {
        for j in 0..rn {
            s.push_str(&format!(
                "    coopStoreT(acc{i}_{j}, &c[at{i}_{j}], ldc);\n"
            ));
        }
    }
    s.push_str("}\n");
    s
}

/// `f32` to IEEE binary16 bytes.
///
/// Hand-rolled rather than taken from `half` because this crate has no such
/// edge and one measurement does not justify adding one. Correct for the
/// normal range only, which is all this test produces -- it feeds the kernel
/// small integers, and `run_one` asserts EXACT equality afterwards, so a
/// conversion that was wrong would fail the test rather than quietly widen it.
fn halved(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127;
        let mant = bits & 0x007f_ffff;
        assert!(
            x == 0.0 || (-14..=15).contains(&exp),
            "this converter is the normal range's only"
        );
        let half = if x == 0.0 {
            sign
        } else {
            sign | (((exp + 15) as u16) << 10) | ((mant >> 13) as u16)
        };
        out.extend_from_slice(&half.to_le_bytes());
    }
    out
}

/// The adapter this suite runs against, or `None` with a printed reason.
///
/// Opened here rather than through `driver_wgpu::device::Device` on purpose:
/// that constructor asks for `ExperimentalFeatures::disabled()`, which is the
/// exact thing this file exists to measure around.
fn adapter() -> Option<wgpu::Adapter> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle().with_env());
    match block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    })) {
        Ok(a) => Some(a),
        Err(why) => {
            driver_wgpu::skip::skipped(&format!(
                "no adapter answered ({why}), so nothing here is measured. \
                 On a Linux runner `PIE_WGPU_FALLBACK=1` takes the software \
                 adapter, which is a real implementation of the same WGSL \
                 and not a way of passing"
            ));
            None
        }
    }
}

/// `wgpu`'s three async entry points all resolve on the spot on a native
/// adapter. `device.rs` says the same thing at more length and this file
/// cannot reach it: that helper is behind the crate's own module tree, and a
/// test binary linking `wgpu` directly needs its own.
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    // `Waker::noop` is the same three lines the standard library already
    // wrote; a hand-rolled `Wake` impl here is one more thing to read.
    let waker = std::task::Waker::noop();
    let mut cx = std::task::Context::from_waker(waker);
    let mut f = std::pin::pin!(f);
    loop {
        if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
    }
}
