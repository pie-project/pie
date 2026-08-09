//! Does every `vision` unit compile under NVRTC, and does every row it states
//! resolve to a mangled symbol?
//!
//! # The question this answers
//!
//! A row is three strings — a template path, an element type, and a symbol —
//! and nothing in the type system connects them to the `.cuh` that has to hold
//! the template. A row can name a kernel that was renamed, a template that
//! takes two arguments where the row supplies one, or a header that stopped
//! compiling under NVRTC the day someone reached for `<cstdint>`. Every one of
//! those is a clean `cargo build` and a failure at run time, on a machine with
//! a GPU, in a process serving tokens.
//!
//! So this compiles each of `vision`'s units the way `runtime::cache` will,
//! against the header set carried in the binary, with one
//! `nvrtcAddNameExpression` per row, and refuses to be satisfied by anything
//! less than a lowered name for EVERY row and a non-empty cubin.
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example unit_probe_vision
//! ```
//!
//! # Why this family needs the probe more than the others did
//!
//! The vision towers came off `driver_internal`, whose rows are three
//! whole-tower bridges and not thirty-two kernels, so nothing in the tree ever
//! named these `__global__`s by symbol — they were anonymous-namespace text
//! inside three translation units, reachable only from a `<<<>>>` in the same
//! file. Twenty-eight of the thirty-nine now have rows. The remaining eleven
//! have geometry no ported rule states: three want three independent extents,
//! one wants a block width `PerHead` fixes differently, one puts a tile count
//! on its leading grid axis, five are dead and have no launcher to check
//! anything against, and the sixth dead one is in another header.
//! `families::vision` catalogues each with the `<<<>>>` it was judged on.
//!
//! NVRTC PARSES an uninstantiated template and stops there. A template that
//! only fails when instantiated — a `Elem<T>` specialisation that does not
//! exist, an intrinsic NVRTC does not declare, a `__shared__` array sized from
//! a dependent expression — is exactly the failure a parse does not find. So
//! the second table below instantiates all eleven anyway, through rows this
//! probe owns and nothing fires.
//!
//! They are carried HERE rather than in [`kernels_cuda_new::families::vision`]
//! because a `Unit` with rows in it is a claim that the rows can be LAUNCHED,
//! and these cannot. `LaunchRule::Unstated` is the honest spelling: the row has
//! not said. `tests/units.rs` enforces the other half of it — every unit in
//! `families::ALL` must declare at least one row — which is why a header with
//! nothing rowable would have no home in the family at all.
//!
//! Nothing here is left out. All thirty-nine `__global__`s in the five vision
//! headers are instantiated by one table or the other, and the sum is checked
//! at the end rather than trusted.
//!
//! # Why this file carries a `cfg` fence
//!
//! Every example in this crate that touches `cudarc` is declared in
//! `Cargo.toml` with `required-features = ["_cuda"]`, because `cargo test`
//! with no features builds every example and an example naming
//! [`kernels_cuda_new::runtime`] does not exist in a feature-free build. This
//! one has no such entry: it was written by a migration that owns four files
//! and `Cargo.toml` is not one of them, and two agents editing the manifest at
//! once is a merge conflict in the one file that has to parse for anything to
//! build at all. So the fence below is the workaround and the manifest entry
//! is the fix — delete the fence when the entry lands.

#[cfg(not(feature = "_cuda"))]
fn main() {
    eprintln!(
        "unit_probe_vision asks NVRTC to compile things and needs a CUDA backend:\n  \
         cargo run -p kernels-cuda-new --features cuda-13 --example unit_probe_vision"
    );
}

#[cfg(feature = "_cuda")]
fn main() {
    probe::main();
}

#[cfg(feature = "_cuda")]
mod probe {
    use std::time::Duration;

    use kernels::KernelSig;
    use kernels::kernel;
    use kernels::operands;
    use kernels_cuda_new::device::DeviceKernel;
    use kernels_cuda_new::runtime::nvrtc;
    use kernels_cuda_new::source;
    use kernels_cuda_new::unit::Unit;

    /// `vision/gemma4_vision`'s one: `k_gelu_mul`, which is dead.
    const GEMMA4_VISION_UNROWED: Unit = Unit {
        name: "vision/gemma4_vision",
        root: include_str!("../csrc/src/vision/gemma4_vision.cuh"),
        rows: GEMMA4_VISION_UNROWED_ROWS,
        options: &[],
    };

    /// `vision/gemma4_audio`'s four — the conv trio, whose `[C, T, F]`
    /// rectangles are three independent extents where `Dims` carries two and
    /// whose tiled pair are transposes of each other; and `k_local_attn`,
    /// whose leading grid axis counts TILES at a width chosen for a 1 KiB
    /// per-thread local array.
    /// `vision/qwen3_vl_tower`'s six — five of them DEAD, which is the
    /// largest single finding of the migration.
    const QWEN3_VL_UNROWED: Unit = Unit {
        name: "vision/qwen3_vl_tower",
        root: include_str!("../csrc/src/vision/qwen3_vl_tower.cuh"),
        rows: QWEN3_VL_UNROWED_ROWS,
        options: &[],
    };

    static GEMMA4_VISION_UNROWED_ROWS: &[DeviceKernel] = &[DeviceKernel {
        sig: &UNROWED_SIGS[0],
        template_path: "vision::device::k_gelu_mul",
        elem: "device::bf16",
    }];

    // `GEMMA4_AUDIO_UNROWED_ROWS` STOOD HERE with four entries, and it is
    // empty rather than shortened: `k_conv2d_s2`, `k_chlast`, `k_chfirst` and
    // `k_local_attn` are rows of `families::vision` now. The paragraph above
    // said their grids were "three independent extents where `Dims` carries
    // two" and "a leading grid axis that counts TILES", and both are still
    // true -- what changed is that a row whose grid no rule states is a row
    // with `LaunchRule::Unstated`, fired by a Rust caller that computes the
    // grid itself. `driver-cuda/src/tower/gemma4_audio.rs` is that caller.
    // The unit `vision/gemma4_audio` no longer appears in this probe at all.

    static QWEN3_VL_UNROWED_ROWS: &[DeviceKernel] = &[
        DeviceKernel { sig: &UNROWED_SIGS[1], template_path: "vision::device::k_add_inplace", elem: "device::bf16" },
        DeviceKernel { sig: &UNROWED_SIGS[2], template_path: "vision::device::k_split_qkv", elem: "device::bf16" },
        DeviceKernel { sig: &UNROWED_SIGS[3], template_path: "vision::device::k_split_qkv_bias", elem: "device::bf16" },
        DeviceKernel { sig: &UNROWED_SIGS[4], template_path: "vision::device::k_rope_vis", elem: "device::bf16" },
        DeviceKernel { sig: &UNROWED_SIGS[5], template_path: "vision::device::k_rope_qk", elem: "device::bf16" },
    ];

    // `k_split_rope_qkv` WAS the sixth entry here and is now
    // `families::vision::QWEN3_VL_ROWS`' sixth row, `LaunchRule::Unstated`.
    // It was the only one of the six that was ever alive -- the other five
    // are superseded kernels nothing launches -- and the only one whose
    // refusal was about geometry rather than deadness. The tower's Rust
    // states `[NH, N, 1] x [HEAD/2, 1, 1]` off `qwen3_vl_tower.cu:249`, so
    // the block-width decision the refusal was protecting is still the
    // tower owner's and is no longer paid for with an absent row.
    //
    // Everything below this line is dead device text and nothing but this
    // probe compiles it.

    /// The contracts these probe rows are written against.
    ///
    /// Operands in the kernels' order and `LaunchRule::Unstated` throughout.
    /// What is being asked is whether NVRTC can INSTANTIATE the template; an
    /// operand list is what a `DeviceKernel` needs to exist, not what this
    /// probe checks. The symbols are spelled the way a real row would spell
    /// them so that promoting one is a move between two files and not a
    /// rename.
    ///
    /// **Fourteen have made that move.** `Tile16`, `AxialRope` and
    /// `PerRowNarrow` (`new-horizon.md` §21.13) state the eleven 16x16 tiles,
    /// the axial rope's three-axis grid and the SSCP layernorm's 128-wide
    /// block, so `k_matmul`, `k_matmul_bias`, `k_addpos_grid2d`, `k_qk`,
    /// `k_av`, `k_pool`, `k_glu`, `k_sscp_flatten`, `k_qkv_scale`,
    /// `k_rel_pos_enc`, `k_merge_gather`, `k_rope_axial2d` and
    /// `k_layernorm_relu` are rows of [`kernels_cuda_new::families::vision`]
    /// now and are instantiated by the first table above rather than this one.
    /// `vision/tower_naive_kernels` lost its last unrowed kernel with them and
    /// no longer appears here at all.
    ///
    /// The fourteenth is `k_split_rope_qkv`, and it moved on a different
    /// argument: no rule states its launch and none was added. It is a row
    /// with `LaunchRule::Unstated`, fired by a Rust tower walk that states
    /// the grid itself. What is left below is dead device text only.
    #[rustfmt::skip]
    static UNROWED_SIGS: [KernelSig; 6] = [
        // DEAD. No `<<<>>>` -- `mlp::geglu_tanh_bf16` took its call site.
        kernel!(k_gelu_mul "vision::k_gelu_mul_bf16",
            file = Some("vision/gemma4_vision.cuh"),
            operands = operands![g: Buf, u: Buf, o: BufMut, t: Usize]),
        // FOUR STOOD HERE -- `k_conv2d_s2`, `k_chlast`, `k_chfirst` and
        // `k_local_attn`. They are contracts in `families::vision` now, with
        // the same operands in the same order and the same `dim3` quoted on
        // each row, because that is what promoting one was always supposed to
        // be: "a move between two files and not a rename".
        // DEAD, and byte-identical to `k_add`, `k_add_pe` and
        // `norm::device::residual_add`.
        kernel!(k_add_inplace "vision::k_add_inplace_bf16",
            file = Some("vision/qwen3_vl_tower.cuh"),
            operands = operands![h: BufMut, x: Buf, t: Usize]),
        // DEAD. Superseded by `k_split_qkv_bias`, itself superseded.
        kernel!(k_split_qkv "vision::k_split_qkv_bf16",
            file = Some("vision/qwen3_vl_tower.cuh"),
            operands = operands![qkv: Buf, q: BufMut, k: BufMut, v: BufMut,
                                 n: I32, h: I32]),
        // DEAD. Superseded by `k_split_rope_qkv`.
        kernel!(k_split_qkv_bias "vision::k_split_qkv_bias_bf16",
            file = Some("vision/qwen3_vl_tower.cuh"),
            operands = operands![qkv: Buf, b: Buf | null, q: BufMut, k: BufMut, v: BufMut,
                                 n: I32, h: I32]),
        // DEAD. Superseded by `k_rope_qk`, itself superseded.
        kernel!(k_rope_vis "vision::k_rope_vis_bf16",
            file = Some("vision/qwen3_vl_tower.cuh"),
            operands = operands![q: BufMut, pos: F32s, n: I32, nh: I32, head: I32,
                                 theta: F32]),
        // DEAD. Superseded by `k_split_rope_qkv`.
        kernel!(k_rope_qk "vision::k_rope_qk_bf16",
            file = Some("vision/qwen3_vl_tower.cuh"),
            operands = operands![q: BufMut, k: BufMut, pos: F32s, n: I32, nh: I32,
                                 head: I32, theta: F32]),
    ];

    /// Every `__global__` in the five vision headers, counted by hand off the
    /// device text and asserted against what the two tables instantiate.
    ///
    /// A constant rather than a comment because the failure it catches is
    /// silent: a kernel added to a `.cuh` and named by neither table PARSES on
    /// every run of this probe and is never instantiated, which is the whole
    /// thing the second table exists to prevent. Six of the thirty-nine are
    /// already dead and nothing but this probe compiles them.
    const KERNELS_IN_THE_HEADERS: usize = 39;

    /// What one unit's compile came to.
    struct Report {
        unit: &'static str,
        rows: usize,
        lowered: usize,
        millis: f64,
        cubin: usize,
        verdict: Result<(), String>,
    }

    pub fn main() {
        let arch = kernels_cuda_new::runtime::cache::arch().unwrap_or("sm_89");
        println!("NVRTC version: {}", version());
        println!("architecture:  {arch}");
        println!("header set:    {} headers carried in the binary", source::DEVICE_HEADERS.len());

        println!("\nthe units, and the rows they state:\n");
        let mut stated: Vec<Report> = Vec::new();
        for unit in kernels_cuda_new::families::vision::UNITS {
            stated.push(probe(unit, arch));
        }
        table(&stated);

        println!("\nkernels no row names, instantiated anyway:\n");
        let unrowed: Vec<Report> = [&GEMMA4_VISION_UNROWED, &QWEN3_VL_UNROWED]
            .into_iter()
            .map(|unit| probe(unit, arch))
            .collect();
        table(&unrowed);

        let all: Vec<&Report> = stated.iter().chain(unrowed.iter()).collect();
        let rows: usize = all.iter().map(|row| row.rows).sum();
        let failed: Vec<&&Report> = all.iter().filter(|row| row.verdict.is_err()).collect();
        println!();
        if rows != KERNELS_IN_THE_HEADERS {
            println!(
                "the two tables instantiate {rows} kernels and the headers hold \
                 {KERNELS_IN_THE_HEADERS}.\nA kernel named by neither table is parsed and never \
                 instantiated, which is\nthe failure this probe exists to find."
            );
            std::process::exit(1);
        }
        if failed.is_empty() {
            let bytes: usize = all.iter().map(|row| row.cubin).sum();
            let millis: f64 = all.iter().map(|row| row.millis).sum();
            let named: usize = stated.iter().map(|row| row.rows).sum();
            println!(
                "{} compiles, {rows} instantiations ({named} with a rule, {} without),\n\
                 {rows} lowered names, {bytes} bytes of cubin, {millis:.0} ms. Every template a\n\
                 row names exists in the unit its `file` claims and instantiates at the element\n\
                 type the row states, and every `__global__` in the five headers is one of them.",
                all.len(),
                rows - named
            );
        } else {
            for row in &failed {
                println!("{}: {}", row.unit, row.verdict.as_ref().unwrap_err());
            }
            std::process::exit(1);
        }
    }

    fn table(rows: &[Report]) {
        println!("  {:<30} {:>4} {:>8} {:>9} {:>11}", "unit", "rows", "lowered", "ms", "cubin");
        for row in rows {
            let mark = if row.verdict.is_ok() { "OK" } else { "FAILED" };
            println!(
                "  {:<30} {:>4} {:>8} {:>9.1} {:>11}  {mark}",
                row.unit, row.rows, row.lowered, row.millis, row.cubin
            );
        }
    }

    /// Compile one unit and check the two things a fire depends on: a lowered
    /// name per row, and an image to load.
    fn probe(unit: &Unit, arch: &str) -> Report {
        match nvrtc::compile(unit, arch) {
            Ok(compiled) => {
                let mut verdict = Ok(());
                if compiled.lowered.len() != unit.rows.len() {
                    verdict = Err(format!(
                        "{} rows, {} lowered names",
                        unit.rows.len(),
                        compiled.lowered.len()
                    ));
                } else if let Some((symbol, _)) =
                    compiled.lowered.iter().find(|(_, mangled)| mangled.is_empty())
                {
                    verdict = Err(format!("`{symbol}` lowered to the empty string"));
                } else if compiled.cubin.is_empty() {
                    verdict = Err("the compile succeeded and produced no cubin".into());
                }
                if !compiled.log.trim().is_empty() {
                    println!("  {} said:\n{}", unit.name, compiled.log.trim());
                }
                Report {
                    unit: unit.name,
                    rows: unit.rows.len(),
                    lowered: compiled.lowered.len(),
                    millis: duration_ms(compiled.elapsed),
                    cubin: compiled.cubin.len(),
                    verdict,
                }
            }
            Err(why) => Report {
                unit: unit.name,
                rows: unit.rows.len(),
                lowered: 0,
                millis: 0.0,
                cubin: 0,
                verdict: Err(why.to_string()),
            },
        }
    }

    fn duration_ms(elapsed: Duration) -> f64 {
        elapsed.as_secs_f64() * 1e3
    }

    /// `libnvrtc`'s own version, so a compile that behaves differently on
    /// another machine can be told apart from one that behaves differently on
    /// this one.
    fn version() -> String {
        use cudarc::nvrtc::sys as nv;
        let (mut major, mut minor) = (0, 0);
        // SAFETY: both are live out-parameters for the call's duration.
        let code = unsafe { nv::nvrtcVersion(&raw mut major, &raw mut minor) };
        if code == nv::nvrtcResult::NVRTC_SUCCESS {
            format!("{major}.{minor}")
        } else {
            format!("unavailable ({code:?})")
        }
    }
}
