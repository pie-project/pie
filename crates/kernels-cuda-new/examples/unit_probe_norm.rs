//! Does every `norm` unit compile under NVRTC, and does every row it states
//! resolve to a mangled symbol?
//!
//! # The question this answers
//!
//! A row is three strings — a template path, an element type, and the symbol a
//! trace names — and nothing in the type system connects them to the `.cuh`
//! that has to hold the template. A row can name a kernel that was renamed, a
//! template that takes two arguments where the row supplies one, or a header
//! that stopped compiling under NVRTC the day someone added `<cstdint>` to it.
//! Every one of those is a clean `cargo build` and a fire that fails at run
//! time, on a machine with a GPU, in a process serving tokens.
//!
//! So this compiles each of `norm`'s units the way `runtime::cache` will,
//! against the header set carried in the binary, with one
//! `nvrtcAddNameExpression` per row — and refuses to be satisfied by anything
//! less than a lowered name for EVERY row and a non-empty cubin. That is the
//! whole of what layer 3 needs to be true before a fire can be trusted.
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example unit_probe_norm
//! ```
//!
//! # The second table: kernels no row names
//!
//! Twelve templates in `norm`'s headers are reachable only through an
//! ahead-of-time launcher, because their geometry fits no rule
//! [`kernels_cuda_new::runtime::launch`] evaluates — a three-axis grid, a
//! `gridDim.y` read for a head count, a grid sized on the INPUT width, a block
//! width of 512, or a choice made at run time from pointer alignment. Each
//! `.cuh` says which and why beside the kernel.
//!
//! NVRTC PARSES those templates as part of the unit that holds them and does
//! not instantiate them, and a template that only fails when instantiated is
//! exactly the failure a parse does not find. So the second table instantiates
//! them anyway, through rows this probe owns and nothing fires — carried here
//! rather than in [`kernels_cuda_new::families::norm`] because a `Unit` with
//! rows in it is a claim that the rows can be LAUNCHED, and these cannot.
//! `LaunchRule::Unstated` is the honest spelling of that: the row has not said.
//!
//! Three are left out and stay parse-only under NVRTC: `rmsnorm_vec8`,
//! `residual_add_rmsnorm_vec8` and `rmsnorm_rasr_vec8` take a block width as
//! their FIRST template argument, and [`DeviceKernel::instantiation`] spells
//! exactly one, a type. nvcc instantiates all three from `rmsnorm.cu` on every
//! ahead-of-time build, which is the gate that covers them.

//! # Why this file carries a `cfg` fence
//!
//! Every example in this crate that touches `cudarc` is declared in
//! `Cargo.toml` with `required-features = ["_cuda"]`, because `cargo test`
//! with no features builds every example and an example naming
//! [`kernels_cuda_new::runtime`] does not exist in a feature-free build. This
//! one has no such entry: it was written by a migration that owns three files
//! and `Cargo.toml` is not one of them, and two agents editing the manifest at
//! once is a merge conflict in the one file that has to parse for anything to
//! build at all. So the fence below is the workaround and the manifest entry
//! is the fix -- delete the fence when the entry lands.

#[cfg(not(feature = "_cuda"))]
fn main() {
    eprintln!(
        "unit_probe_norm asks NVRTC to compile things and needs a CUDA backend:\n  \
         cargo run -p kernels-cuda-new --features cuda-13 --example unit_probe_norm"
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

    /// `norm/altup`'s two templates. Both launchers build a
    /// `dim3(T, K, ceil(H/128))` grid and no ported rule produces a `gridDim.y`,
    /// so the file is a unit nowhere and would otherwise be compiled by nothing.
    const ALTUP: Unit = Unit {
        name: "norm/altup",
        root: include_str!("../csrc/src/norm/altup.cuh"),
        rows: ALTUP_ROWS,
        options: &[],
    };

    /// The four hyper-connection kernels `norm/dsv4_hc`'s unit leaves out: two
    /// whose grid covers the input width rather than the output's, and two that
    /// launch `dim3(N, heads)`.
    const DSV4_HC_UNROWED: Unit = Unit {
        name: "norm/dsv4_hc",
        root: include_str!("../csrc/src/norm/dsv4_hc.cuh"),
        rows: DSV4_HC_UNROWED_ROWS,
        options: &[],
    };

    /// The five scalar RMSNorm templates `norm/rmsnorm`'s unit leaves out: four
    /// named by symbols with a second, per-head reading, and one whose scalar
    /// launcher is 512 threads wide.
    const RMSNORM_UNROWED: Unit = Unit {
        name: "norm/rmsnorm",
        root: include_str!("../csrc/src/norm/rmsnorm.cuh"),
        rows: RMSNORM_UNROWED_ROWS,
        options: &[],
    };

    static ALTUP_ROWS: &[DeviceKernel] = &[
        DeviceKernel {
            sig: &UNROWED_SIGS[0],
            template_path: "norm::device::altup_predict",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[1],
            template_path: "norm::device::altup_correct",
            elem: "device::bf16",
        },
    ];

    static DSV4_HC_UNROWED_ROWS: &[DeviceKernel] = &[
        DeviceKernel { sig: &UNROWED_SIGS[2], template_path: "norm::device::hc_post", elem: "device::bf16" },
        DeviceKernel { sig: &UNROWED_SIGS[3], template_path: "norm::device::hc_expand", elem: "device::bf16" },
        DeviceKernel {
            sig: &UNROWED_SIGS[4],
            template_path: "norm::device::attn_sink_correction",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[5],
            template_path: "norm::device::per_head_rmsnorm",
            elem: "device::bf16",
        },
    ];

    static RMSNORM_UNROWED_ROWS: &[DeviceKernel] = &[
        DeviceKernel {
            sig: &UNROWED_SIGS[6],
            template_path: "norm::device::rmsnorm_gemma",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[7],
            template_path: "norm::device::rmsnorm_no_scale",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[8],
            template_path: "norm::device::rmsnorm_gated",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[9],
            template_path: "norm::device::rmsnorm_gated_f32_in",
            elem: "device::bf16",
        },
        // The one place a second template argument is spelled, and the reason it
        // has to be: this kernel's `BLOCK` has no default, because its scalar
        // launcher is 512 threads wide and `LaunchRule::Rms` fixes 256. A row
        // could not say this -- `instantiation()` builds one argument -- which is
        // the same fact as "no rule states it", read from the other end.
        DeviceKernel {
            sig: &UNROWED_SIGS[10],
            template_path: "norm::device::rmsnorm_residual_add_scale_rmsnorm",
            elem: "device::bf16, 512",
        },
    ];

    /// The contracts these probe rows are written against.
    ///
    /// Operands in the kernels' order and `LaunchRule::Unstated` throughout: what
    /// is being asked is whether NVRTC can instantiate the template, and an
    /// operand list is what a `DeviceKernel` needs to exist, not what this probe
    /// checks.
    #[rustfmt::skip]
    static UNROWED_SIGS: [KernelSig; 11] = [
        kernel!(altup_predict "norm::altup_predict_bf16",
            file = Some("norm/altup.cuh"),
            operands = operands![streams: Buf, coefs: F32s, predictions: BufMut,
                                 k: I32, t: I32, h: I32]),
        kernel!(altup_correct "norm::altup_correct_bf16",
            file = Some("norm/altup.cuh"),
            operands = operands![predictions: Buf, activated: Buf,
                                 correction_coefs_plus_one: F32s, corrected: BufMut,
                                 k: I32, t: I32, h: I32, active_idx: I32]),
        kernel!(hc_post "norm::hc_post_bf16",
            file = Some("norm/dsv4_hc.cuh"),
            operands = operands![x: Buf, residual: Buf, post_mix: F32s, comb_mix: F32s,
                                 out_residual: BufMut, n: I32, hc_mult: I32, hidden_size: I32]),
        kernel!(hc_expand "norm::hc_expand_bf16",
            file = Some("norm/dsv4_hc.cuh"),
            operands = operands![input: Buf, output: BufMut,
                                 n: I32, hc_mult: I32, hidden_size: I32]),
        kernel!(attn_sink_correction "norm::attn_sink_correction_bf16",
            file = Some("norm/dsv4_hc.cuh"),
            operands = operands![out: BufMut, lse: F32s, sink: F32s,
                                 num_heads: I32, head_dim: I32]),
        kernel!(per_head_rmsnorm "norm::per_head_rmsnorm_bf16",
            file = Some("norm/dsv4_hc.cuh"),
            operands = operands![q: BufMut, head_dim: I32, eps: F32]),
        kernel!(rmsnorm_gemma "norm::rmsnorm_gemma_bf16",
            file = Some("norm/rmsnorm.cuh"),
            operands = operands![x: Buf, weight: Buf, y: BufMut, hidden: I32,
                                 x_row_stride: I32, y_row_stride: I32, eps: F32]),
        kernel!(rmsnorm_no_scale "norm::rmsnorm_no_scale_bf16",
            file = Some("norm/rmsnorm.cuh"),
            operands = operands![x: Buf, y: BufMut, hidden: I32, eps: F32]),
        kernel!(rmsnorm_gated "norm::rmsnorm_gated_bf16",
            file = Some("norm/rmsnorm.cuh"),
            operands = operands![x: Buf, gate: Buf, weight: F32s, y: BufMut,
                                 hidden: I32, eps: F32]),
        kernel!(rmsnorm_gated_f32_in "norm::rmsnorm_gated_fp32_in_bf16",
            file = Some("norm/rmsnorm.cuh"),
            operands = operands![x: F32s, gate: Buf, weight: F32s, y: BufMut,
                                 hidden: I32, eps: F32]),
        kernel!(norm_residual_scale_norm "norm::rmsnorm_residual_add_scale_rmsnorm_bf16",
            file = Some("norm/rmsnorm.cuh"),
            operands = operands![x: Buf, weight: Buf, hidden: BufMut, scale: F32,
                                 next_weight: Buf, norm_out: BufMut,
                                 hidden_size: I32, eps: F32]),
    ];

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
        for unit in kernels_cuda_new::families::norm::UNITS {
            stated.push(probe(unit, arch));
        }
        table(&stated);

        println!("\nkernels no row names, instantiated anyway:\n");
        let unrowed: Vec<Report> = [&ALTUP, &DSV4_HC_UNROWED, &RMSNORM_UNROWED]
            .into_iter()
            .map(|unit| probe(unit, arch))
            .collect();
        table(&unrowed);

        let all: Vec<&Report> = stated.iter().chain(unrowed.iter()).collect();
        let failed: Vec<&&Report> = all.iter().filter(|row| row.verdict.is_err()).collect();
        println!();
        if failed.is_empty() {
            let rows: usize = all.iter().map(|row| row.rows).sum();
            let bytes: usize = all.iter().map(|row| row.cubin).sum();
            let millis: f64 = all.iter().map(|row| row.millis).sum();
            println!(
                "{} compiles, {rows} instantiations, {rows} lowered names, {bytes} bytes of\n\
                 cubin, {millis:.0} ms. Every template a row names exists in the unit its `file`\n\
                 claims and instantiates at the element type the row states.",
                all.len()
            );
        } else {
            for row in &failed {
                println!("{}: {}", row.unit, row.verdict.as_ref().unwrap_err());
            }
            std::process::exit(1);
        }
    }

    fn table(rows: &[Report]) {
        println!("  {:<20} {:>4} {:>8} {:>9} {:>11}", "unit", "rows", "lowered", "ms", "cubin");
        for row in rows {
            let mark = if row.verdict.is_ok() { "OK" } else { "FAILED" };
            println!(
                "  {:<20} {:>4} {:>8} {:>9.1} {:>11}  {mark}",
                row.unit, row.rows, row.lowered, row.millis, row.cubin
            );
        }
    }

    /// Compile one unit and check the two things a fire depends on: a lowered name
    /// per row, and an image to load.
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

    /// `libnvrtc`'s own version, so a compile that behaves differently on another
    /// machine can be told apart from one that behaves differently on this one.
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
