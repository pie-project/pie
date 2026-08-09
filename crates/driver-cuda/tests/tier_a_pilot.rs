//! Tier A end to end, on a real device.
//!
//! The chain this exercises is the whole experiment, and no link in it is
//! stubbed — nor is any link a file a human wrote:
//!
//! ```text
//! kernels_cuda_new::device::ALTUP_AUX   the rows
//!   -> bind::nvrtc::SOURCE             the templates, from the BINARY
//!   -> nvrtcAddNameExpression          the instantiations the rows name
//!   -> nvrtcGetLoweredName             what C++ calls each one
//!   -> bind::device::KernelModule      loaded, every row resolved by mangled name
//!   -> bind::launch::eval              the grid the ROW states
//!   -> bind::device::Args::bind        the operands the ROW declares
//!   -> cuLaunchKernel                  fired from Rust
//! ```
//!
//! There is no `.cu` in that list and no nvcc. The only C++ is
//! `csrc/src/norm/altup_aux.cuh` — six `__global__` templates — and it is
//! compiled from a string held in this binary.
//!
//! `cuda-progress.md`'s rule is why it is written this way: *"Non-GPU green is
//! weak evidence. Every dispatch defect found on 2026-08-10 compiled cleanly
//! and passed the full non-GPU battery."* A launch rule is arithmetic, and
//! arithmetic is exactly the class of thing that is right in isolation and
//! wrong against a driver.
//!
//! Skipped without a device, like every other `gpu_*` binary here.

use driver_cuda::bind::device::{ArgValue, Args, KernelModule};
use driver_cuda::bind::headers::Header;
use driver_cuda::bind::launch::{Dims, eval};
use driver_cuda::bind::nvrtc::{Family, NORM_ELEMENTWISE, compile_unit};
use driver_cuda::device::{Allocator, OwnedStream};
use kernels::KernelSig;
use kernels_cuda_new::device::{ALTUP_AUX as ENTRIES, DeviceKernel, ELEMENTWISE};

mod common;
use common::{device_or_skip, gpu_guard};

/// gemma-3n's AltUp shape, small enough to check by hand and wide enough
/// that the reduction spans several warps.
const T: usize = 16;
const H: usize = 2048;
const K: usize = 4;

/// `f32 -> bf16`, round-to-nearest-even, as the hardware converts.
///
/// Written out rather than pulled from a crate because the reference has to
/// round the way `__float2bfloat16` does or the comparison measures the
/// rounding rather than the kernel.
fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return (bits >> 16) as u16 | 0x0040; // a NaN stays a NaN
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// `bf16 -> f32`, which is exact: the low sixteen bits are zero.
fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

fn bytes_of_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u16s_of_bytes(v: &[u8]) -> Vec<u16> {
    v.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn f32s_of_bytes(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A deterministic input with a wide dynamic range, so a wrong stride shows
/// up as a wrong value rather than as a similar one.
fn sample(i: usize) -> f32 {
    let x = (i % 97) as f32 / 97.0;
    (x - 0.5) * (1.0 + (i % 7) as f32)
}

fn row(symbol: &str) -> &'static KernelSig {
    ENTRIES
        .iter()
        .find(|k| k.sig.symbol == symbol)
        .expect("the pilot states this row")
        .sig
}

/// The six kernels, fired from Rust, against a host reference.
///
/// One test rather than six because the steps share a module, a stream and an
/// allocator, and because a failure in any of them has the same cause to
/// investigate: splitting them would report six failures for one defect.
#[test]
#[allow(clippy::too_many_lines)]
fn the_tier_a_entries_run_and_answer() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("Tier A launch") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let arch = format!("sm_{major}{minor}");

    // The compile IS a check: NVRTC has to find every template a row names,
    // instantiate it at the tag the row names, and hand back a lowered name
    // for it. A row that drifted from the header fails here, before a single
    // launch.
    let compiled = Family::compile(&arch).unwrap_or_else(|e| {
        panic!("the rows and the templates disagree:\n{e}");
    });
    eprintln!(
        "tier A: {} bytes of cubin for {arch}, {} instantiations, compiled in {:.1} ms",
        compiled.cubin.len(),
        compiled.lowered.len(),
        compiled.elapsed.as_secs_f64() * 1e3,
    );
    for (symbol, mangled) in &compiled.lowered {
        eprintln!("  {symbol:38} -> {mangled}");
    }

    let sigs: Vec<&'static KernelSig> = ENTRIES.iter().map(|k| k.sig).collect();
    let module = KernelModule::load_mangled(&compiled.cubin, &sigs, &compiled.lowered)
        .expect("every row's instantiation resolves");
    assert_eq!(module.len(), ENTRIES.len());

    let alloc = Allocator::new();
    let stream = OwnedStream::new(0).expect("stream");
    let s = stream.as_ref();

    // ---- the inputs, and their host-side truth ---------------------------
    let x_host: Vec<u16> = (0..T * H).map(|i| to_bf16(sample(i))).collect();
    let streams_host: Vec<u16> = (0..K * T * H).map(|i| to_bf16(sample(i * 3 + 1))).collect();
    let predict_host: Vec<u16> = (0..T * K * K).map(|i| to_bf16(sample(i * 5 + 2))).collect();
    let correct_host: Vec<u16> = (0..T * K).map(|i| to_bf16(sample(i * 11 + 3))).collect();

    let mut x = alloc.alloc(x_host.len() * 2).expect("x");
    x.copy_from_host(&bytes_of_u16(&x_host), s)
        .expect("upload x");
    let mut rms = alloc.alloc(T * 4).expect("rms");
    rms.memset(0, s).expect("clear rms");

    // ---- compute_rms ----------------------------------------------------
    let sig = row("norm::compute_rms_bf16");
    let geometry = eval(
        sig.launch,
        Dims {
            rows: T as u32,
            width: H as u32,
            in_width: H as u32,
        },
    )
    .expect("the row's rule evaluates");
    let mut args = Args::bind(
        sig,
        &[
            ArgValue::Ptr(x.ptr_at(0, x_host.len() * 2).expect("x is live")),
            ArgValue::Ptr(rms.ptr_at(0, T * 4).expect("rms is live")),
            ArgValue::I32(H as i32),
            ArgValue::F32(1e-5),
        ],
    )
    .expect("the values match the row");
    module
        .fire(sig, geometry, &mut args, s)
        .expect("compute_rms fires");
    s.synchronize().expect("sync");

    let mut got = vec![0u8; T * 4];
    rms.copy_to_host(&mut got, s).expect("download rms");
    let got_rms = f32s_of_bytes(&got);
    for t in 0..T {
        let mean_sq: f32 = (0..H)
            .map(|h| from_bf16(x_host[t * H + h]).powi(2))
            .sum::<f32>()
            / H as f32;
        let want = mean_sq.max(1e-5).sqrt();
        assert!(
            (got_rms[t] - want).abs() <= want * 1e-4,
            "compute_rms row {t}: {} vs {want}",
            got_rms[t]
        );
    }

    // ---- magnitude_rescale, which reads what compute_rms wrote ----------
    // Rescaling x to the RMS it already has is the identity, so the target
    // is scaled: a kernel that ignored `target_rms` would pass the identity
    // check and fail this one.
    let target: Vec<f32> = got_rms.iter().map(|v| v * 2.0).collect();
    let mut targets = alloc.alloc(T * 4).expect("target");
    targets
        .copy_from_host(
            &target
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            s,
        )
        .expect("upload target");

    let sig = row("norm::magnitude_rescale_bf16");
    let geometry = eval(
        sig.launch,
        Dims {
            rows: T as u32,
            width: H as u32,
            in_width: H as u32,
        },
    )
    .expect("the row's rule evaluates");
    let mut args = Args::bind(
        sig,
        &[
            ArgValue::Ptr(x.ptr_at(0, x_host.len() * 2).expect("x is live")),
            ArgValue::Ptr(targets.ptr_at(0, T * 4).expect("target is live")),
            ArgValue::I32(H as i32),
            ArgValue::F32(1e-5),
        ],
    )
    .expect("the values match the row");
    module
        .fire(sig, geometry, &mut args, s)
        .expect("magnitude_rescale fires");
    s.synchronize().expect("sync");

    let mut got = vec![0u8; x_host.len() * 2];
    x.copy_to_host(&mut got, s).expect("download x");
    let rescaled = u16s_of_bytes(&got);
    for t in 0..T {
        let want = 2.0 * from_bf16(x_host[t * H]);
        let have = from_bf16(rescaled[t * H]);
        assert!(
            (have - want).abs() <= want.abs().mul_add(1e-2, 1e-3),
            "magnitude_rescale row {t}: {have} vs {want}"
        );
    }

    // ---- mean_streams ---------------------------------------------------
    let mut streams = alloc.alloc(streams_host.len() * 2).expect("streams");
    streams
        .copy_from_host(&bytes_of_u16(&streams_host), s)
        .expect("upload streams");
    let mut mean = alloc.alloc(T * H * 2).expect("mean");
    mean.memset(0, s).expect("clear mean");

    let sig = row("norm::mean_streams_bf16");
    let geometry = eval(
        sig.launch,
        Dims {
            rows: T as u32,
            width: H as u32,
            in_width: H as u32,
        },
    )
    .expect("the row's rule evaluates");
    let mut args = Args::bind(
        sig,
        &[
            ArgValue::Ptr(
                streams
                    .ptr_at(0, streams_host.len() * 2)
                    .expect("streams is live"),
            ),
            ArgValue::Ptr(mean.ptr_at(0, T * H * 2).expect("mean is live")),
            ArgValue::I32(K as i32),
            ArgValue::I32(T as i32),
            ArgValue::I32(H as i32),
        ],
    )
    .expect("the values match the row");
    module
        .fire(sig, geometry, &mut args, s)
        .expect("mean_streams fires");
    s.synchronize().expect("sync");

    let mut got = vec![0u8; T * H * 2];
    mean.copy_to_host(&mut got, s).expect("download mean");
    let got_mean = u16s_of_bytes(&got);
    // Every channel, not a sample: `grid.y` is the axis the rule computes
    // and a rounding bug there leaves the TAIL of each row untouched.
    for t in 0..T {
        for h in 0..H {
            let want = to_bf16(
                (0..K)
                    .map(|k| from_bf16(streams_host[k * T * H + t * H + h]))
                    .sum::<f32>()
                    / K as f32,
            );
            assert_eq!(got_mean[t * H + h], want, "mean_streams at ({t}, {h})");
        }
    }

    // ---- the two coefficient unpacks ------------------------------------
    let mut packed = alloc.alloc(predict_host.len() * 2).expect("predict in");
    packed
        .copy_from_host(&bytes_of_u16(&predict_host), s)
        .expect("upload predict");
    let mut coefs = alloc.alloc(T * K * K * 4).expect("predict out");
    coefs.memset(0, s).expect("clear predict out");

    let sig = row("norm::altup_unpack_predict_coefs");
    let geometry = eval(
        sig.launch,
        Dims {
            rows: T as u32,
            width: (K * K) as u32,
            in_width: (K * K) as u32,
        },
    )
    .expect("the row's rule evaluates");
    let mut args = Args::bind(
        sig,
        &[
            ArgValue::Ptr(packed.ptr_at(0, predict_host.len() * 2).expect("live")),
            ArgValue::Ptr(coefs.ptr_at(0, T * K * K * 4).expect("live")),
            ArgValue::I32(K as i32),
        ],
    )
    .expect("the values match the row");
    module
        .fire(sig, geometry, &mut args, s)
        .expect("unpack_predict fires");
    s.synchronize().expect("sync");

    let mut got = vec![0u8; T * K * K * 4];
    coefs
        .copy_to_host(&mut got, s)
        .expect("download predict out");
    let got_coefs = f32s_of_bytes(&got);
    for t in 0..T {
        for k in 0..K {
            for j in 0..K {
                // The permute is the whole point: `out[t, j, k] = in[t, k*K + j]`.
                assert_eq!(
                    got_coefs[t * K * K + j * K + k],
                    from_bf16(predict_host[t * K * K + k * K + j]),
                    "unpack_predict at ({t}, {j}, {k})"
                );
            }
        }
    }

    let mut packed = alloc.alloc(correct_host.len() * 2).expect("correct in");
    packed
        .copy_from_host(&bytes_of_u16(&correct_host), s)
        .expect("upload correct");
    let mut coefs = alloc.alloc(T * K * 4).expect("correct out");
    coefs.memset(0, s).expect("clear correct out");

    let sig = row("norm::altup_unpack_correct_coefs");
    let geometry = eval(
        sig.launch,
        Dims {
            rows: T as u32,
            width: K as u32,
            in_width: K as u32,
        },
    )
    .expect("the row's rule evaluates");
    let mut args = Args::bind(
        sig,
        &[
            ArgValue::Ptr(packed.ptr_at(0, correct_host.len() * 2).expect("live")),
            ArgValue::Ptr(coefs.ptr_at(0, T * K * 4).expect("live")),
            ArgValue::I32(K as i32),
        ],
    )
    .expect("the values match the row");
    module
        .fire(sig, geometry, &mut args, s)
        .expect("unpack_correct fires");
    s.synchronize().expect("sync");

    let mut got = vec![0u8; T * K * 4];
    coefs
        .copy_to_host(&mut got, s)
        .expect("download correct out");
    let got_coefs = f32s_of_bytes(&got);
    for i in 0..T * K {
        assert_eq!(
            got_coefs[i],
            from_bf16(correct_host[i]) + 1.0,
            "unpack_correct at {i} -- the `+ 1.0` is HF's and belongs to the kernel"
        );
    }

    // ---- tanh, the flat one ---------------------------------------------
    let mut y = alloc.alloc(x_host.len() * 2).expect("y");
    y.copy_from_host(&bytes_of_u16(&x_host), s)
        .expect("upload y");

    let sig = row("norm::tanh_bf16");
    let geometry = eval(
        sig.launch,
        Dims {
            rows: T as u32,
            width: H as u32,
            in_width: H as u32,
        },
    )
    .expect("the row's rule evaluates");
    let mut args = Args::bind(
        sig,
        &[
            ArgValue::Ptr(y.ptr_at(0, x_host.len() * 2).expect("live")),
            ArgValue::I32((T * H) as i32),
        ],
    )
    .expect("the values match the row");
    module
        .fire(sig, geometry, &mut args, s)
        .expect("tanh fires");
    s.synchronize().expect("sync");

    let mut got = vec![0u8; x_host.len() * 2];
    y.copy_to_host(&mut got, s).expect("download y");
    let got_tanh = u16s_of_bytes(&got);
    // The LAST element as well as the first: a grid that rounded down
    // instead of up leaves the tail of the extent untouched, and the tail
    // is the only place that shows it.
    for i in [0, 1, (T * H) / 2, T * H - 1] {
        let want = from_bf16(x_host[i]).tanh();
        let have = from_bf16(got_tanh[i]);
        assert!((have - want).abs() <= 8e-3, "tanh at {i}: {have} vs {want}");
    }
}

/// What one launch costs the host to issue.
///
/// Reported rather than asserted: an issue cost is a property of the machine
/// as much as of the code, and a threshold here would be a flake. The number
/// it prints is the one the pilot is judged on, against the same measurement
/// taken for `<<<>>>` in `scripts/` -- see the pilot write-up.
#[test]
fn the_issue_cost_of_a_stated_launch() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("Tier A issue cost") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let (module, _) = Family::load(&format!("sm_{major}{minor}")).expect("the family loads");

    let alloc = Allocator::new();
    let stream = OwnedStream::new(0).expect("stream");
    let s = stream.as_ref();
    let mut y = alloc.alloc(T * H * 2).expect("y");
    y.memset(0, s).expect("clear");

    let sig = row("norm::tanh_bf16");
    let dims = Dims {
        rows: T as u32,
        width: H as u32,
        in_width: H as u32,
    };
    const N: usize = 2000;

    // Warm the driver: the first launch of a kernel pays for its module's
    // lazy load, and averaging that over N would report it as per-launch
    // cost.
    for _ in 0..50 {
        let geometry = eval(sig.launch, dims).expect("rule");
        let mut args = Args::bind(
            sig,
            &[
                ArgValue::Ptr(y.ptr_at(0, T * H * 2).expect("live")),
                ArgValue::I32((T * H) as i32),
            ],
        )
        .expect("bind");
        module.fire(sig, geometry, &mut args, s).expect("fire");
    }
    s.synchronize().expect("sync");

    let start = std::time::Instant::now();
    for _ in 0..N {
        // The FULL per-launch cost, not just the driver call: evaluating the
        // rule and marshalling the operands is work the C++ launcher also
        // did, so leaving it out of the loop would flatter this path.
        let geometry = eval(sig.launch, dims).expect("rule");
        let mut args = Args::bind(
            sig,
            &[
                ArgValue::Ptr(y.ptr_at(0, T * H * 2).expect("live")),
                ArgValue::I32((T * H) as i32),
            ],
        )
        .expect("bind");
        module.fire(sig, geometry, &mut args, s).expect("fire");
    }
    let issued = start.elapsed();
    s.synchronize().expect("sync");
    let retired = start.elapsed();

    eprintln!(
        "tier A issue cost: {:.3} us/launch (issue), {:.3} us/launch (to retire), n={N}",
        issued.as_secs_f64() * 1e6 / N as f64,
        retired.as_secs_f64() * 1e6 / N as f64,
    );
}

/// The fp16 kernel, which cost one row and no C++ at all.
///
/// Fired for real rather than merely resolved: a lowered name proves the
/// instantiation exists, and only a launch proves it is the RIGHT one. The
/// two `tanh` rows share a template and differ in their element type, so an
/// instantiation that picked the wrong one would still load, still fire, and
/// answer with the bf16 kernel's reading of fp16 bits — finite, plausible,
/// wrong.
#[test]
fn a_second_numeric_format_costs_one_row() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("Tier A fp16") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let (module, _) = Family::load(&format!("sm_{major}{minor}")).expect("the family loads");

    let alloc = Allocator::new();
    let stream = OwnedStream::new(0).expect("stream");
    let s = stream.as_ref();

    // fp16 bit patterns for a handful of values, and their tanh.
    let host: Vec<u16> = vec![
        0x0000, // +0
        0x3c00, // 1.0
        0xbc00, // -1.0
        0x4000, // 2.0
        0x3800, // 0.5
        0xb800, // -0.5
        0x4900, // 10.0  -- saturates
        0x2e66, // ~0.1
    ];
    let want: Vec<f32> = vec![
        0.0,
        1.0f32.tanh(),
        (-1.0f32).tanh(),
        2.0f32.tanh(),
        0.5f32.tanh(),
        (-0.5f32).tanh(),
        10.0f32.tanh(),
        0.1f32.tanh(),
    ];

    let mut x = alloc.alloc(host.len() * 2).expect("x");
    x.copy_from_host(&bytes_of_u16(&host), s).expect("upload");

    let sig = row("norm::tanh_f16");
    let geometry = eval(
        sig.launch,
        Dims {
            rows: 1,
            width: host.len() as u32,
            in_width: host.len() as u32,
        },
    )
    .expect("the row's rule evaluates");
    let mut args = Args::bind(
        sig,
        &[
            ArgValue::Ptr(x.ptr_at(0, host.len() * 2).expect("live")),
            ArgValue::I32(host.len() as i32),
        ],
    )
    .expect("the values match the row");
    module
        .fire(sig, geometry, &mut args, s)
        .expect("tanh_f16 fires");
    s.synchronize().expect("sync");

    let mut got = vec![0u8; host.len() * 2];
    x.copy_to_host(&mut got, s).expect("download");
    for (i, (bits, want)) in u16s_of_bytes(&got).iter().zip(&want).enumerate() {
        // fp16 -> f32 for the comparison, by the same shifts the kernel uses.
        let e = (u32::from(*bits) >> 10) & 0x1f;
        let m = u32::from(*bits) & 0x3ff;
        let sign = (u32::from(*bits) & 0x8000) << 16;
        let have = if e == 0 {
            f32::from_bits(sign)
                + if m == 0 {
                    0.0
                } else {
                    f32::from_bits(0x3380_0000) * m as f32
                }
        } else {
            f32::from_bits(sign | ((e + 112) << 23) | (m << 13))
        };
        assert!(
            (have - *want).abs() <= 1e-2,
            "tanh_f16 at {i}: {have} vs {want} -- a wrong TAG answers plausibly"
        );
    }
}

/// A row that drifted from the header does not compile.
///
/// This is the property the generated C shim used to buy, asked of the
/// runtime path: NVRTC has to find the template, instantiate it at the tag
/// the row names, and produce a lowered name. Each of the three can be wrong
/// on its own, so each is wrong here on its own.
///
/// The shim's check was at BUILD time and this one is at load time, which is
/// the trade Tier A makes and the reason `abi::emit_device_typecheck` still
/// exists: it makes the same three mistakes fail `cargo build`.
#[test]
fn a_row_that_names_nothing_is_refused() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("Tier A drift") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let arch = format!("sm_{major}{minor}");

    let good = &ENTRIES[0];
    let sig = good.sig;

    // (what is wrong, the row)
    let drifted = [
        (
            "a template the header does not define",
            DeviceKernel {
                sig,
                template_path: "norm::device::compute_rms_v2",
                elem: good.elem,
            },
        ),
        (
            "an element type the header does not define",
            DeviceKernel {
                sig,
                template_path: good.template_path,
                elem: "norm::device::fp8",
            },
        ),
        (
            "the right names in the wrong namespace",
            DeviceKernel {
                sig,
                template_path: "norm::compute_rms",
                elem: good.elem,
            },
        ),
    ];

    for (what, row) in drifted {
        match Family::compile_rows(&arch, std::slice::from_ref(&row)) {
            Ok(_) => panic!("a row naming {what} compiled, which means nothing checks it"),
            Err(why) => eprintln!("refused ({what}): {why}"),
        }
    }

    // And the correct row still compiles, so the test above is not passing
    // because everything fails.
    assert!(Family::compile_rows(&arch, std::slice::from_ref(good)).is_ok());
}

/// **Stage B's gate, as a negative control.**
///
/// `altup_aux.cuh` no longer defines the element types it instantiates over;
/// `#include "norm/pie_device.cuh"` does, and NVRTC resolves that name
/// against the header array `bind::headers::DEVICE_HEADERS` hands it — never
/// against a path, so the compile is the same on a machine with a CUDA
/// toolkit and on one without.
///
/// Every other test here shows the compile SUCCEEDING, which is consistent
/// with the header array being ignored and the definitions having never
/// moved. This one takes the array away and requires the failure, which is
/// the only way to show the array is what resolved the include.
///
/// The refusal must also name the file NVRTC could not find, because the
/// mistake this guards against — adding a `#include` and forgetting the entry
/// in `DEVICE_HEADERS` — is one whose diagnosis is the missing name.
#[test]
fn the_header_set_is_what_resolves_the_include() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("Stage B header set") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let arch = format!("sm_{major}{minor}");

    // The set the driver ships: the include resolves and the family compiles.
    assert!(
        Family::compile(&arch).is_ok(),
        "the shipped header set must compile"
    );

    // The same source and the same rows, with nothing to resolve against.
    let refusal = Family::compile_with(&arch, Family::rows(), &[])
        .err()
        .expect("an unresolvable include cannot compile");
    let text = refusal.to_string();
    assert!(
        text.contains("pie_device.cuh"),
        "the refusal must name the header NVRTC could not find, since that is \
         the whole diagnosis of a missing entry: {text}"
    );
}

/// **Stage D's gate: a diamond include resolves once.**
///
/// `new-horizon.md` §3.2 gives three reasons the NVRTC header array beats the
/// Metal splicer. Two were settled by Stage B — the source stays
/// nvcc-compilable, and nothing is read from disk. The third was a claim:
///
/// > include guards work, so a diamond dependency is not a double definition.
///
/// It is the one that cannot be reasoned to, because the guard is evaluated
/// by NVRTC's preprocessor against its own virtual filesystem, and whether
/// `#pragma once` identifies a header by NAME there is NVRTC's business. If
/// it did not, every shared header would need include guards written by hand
/// — or worse, the splicer's `HashSet`-of-paths approximation — and Stage D
/// would be a different design.
///
/// So it is measured. `base` defines a function; `left` and `right` both
/// include it; the root includes both and calls it. Without a working guard
/// the definition arrives twice and NVRTC rejects the program.
///
/// The negative half matters as much: the same diamond with the guard removed
/// **must** fail, or the test is passing because NVRTC deduplicated by path,
/// or because it never included anything at all.
#[test]
fn a_diamond_include_resolves_once() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("Stage D diamond") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let arch = format!("sm_{major}{minor}");

    const LEFT: &str = "#pragma once\n#include \"probe/base.cuh\"\n";
    const RIGHT: &str = "#pragma once\n#include \"probe/base.cuh\"\n";
    // Includes BOTH arms, and uses what only `base` defines -- so a compile
    // that succeeded without resolving the includes would not link the call.
    const ROOT: &str = "#include \"probe/left.cuh\"\n#include \"probe/right.cuh\"\n\
                        __global__ void probe_use(float* o) { *o = pie_probe_base(*o); }\n";

    let diamond = |base: &'static str| {
        vec![
            Header {
                name: "probe/base.cuh",
                text: base,
            },
            Header {
                name: "probe/left.cuh",
                text: LEFT,
            },
            Header {
                name: "probe/right.cuh",
                text: RIGHT,
            },
        ]
    };

    // Guarded: one definition, however many paths reach it.
    let guarded = diamond(
        "#pragma once\n__device__ __forceinline__ float pie_probe_base(float v) { return v + 1.f; }\n",
    );
    compile_unit(&arch, "pie_diamond.cu", ROOT, &[], &guarded)
        .expect("a guarded diamond is one definition, not two");

    // Unguarded: the same two paths, and now they both arrive.
    let unguarded =
        diamond("__device__ __forceinline__ float pie_probe_base(float v) { return v + 1.f; }\n");
    let refusal = compile_unit(&arch, "pie_diamond.cu", ROOT, &[], &unguarded)
        .err()
        .expect(
            "without a guard the definition arrives twice -- if this compiles, \
             NVRTC is deduplicating headers itself and the guard above proved \
             nothing",
        );
    assert!(
        // NVRTC 13.0 words it "has already been defined"; older cicc says
        // "redefinition". Both are accepted because what is under test is
        // that the second definition ARRIVES, not how the front end phrases
        // its objection.
        refusal.to_string().contains("already been defined")
            || refusal.to_string().contains("redefinition"),
        "the second definition is what must be refused: {refusal}"
    );
    assert!(
        refusal.to_string().contains("pie_probe_base"),
        "and the refusal must name what was defined twice: {refusal}"
    );
}

/// **Stage D's content, compiled.** The two shared device headers resolve,
/// and they resolve as a diamond over the prelude.
///
/// `a_diamond_include_resolves_once` proves the MECHANISM with three headers
/// invented for it. This proves the mechanism carries the tree's real ones:
/// `rope_device.cuh` and `kv_paged_addr.cuh` both include `pie_device.cuh`,
/// so a source taking both takes the prelude twice — the diamond that used to
/// be hypothetical.
///
/// Neither header included anything of ours before Stage B. Both opened with
/// `#include <cuda_bf16.h>` and `<cstdint>`, which NVRTC does not have, so
/// neither could be compiled at run time at all. What replaced those is the
/// prelude, and what this test asserts is that the replacement is complete —
/// a leftover `__nv_bfloat16` or `std::uint32_t` is a compile error here.
#[test]
fn the_shared_device_headers_compile_as_a_diamond() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("Stage D content") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let arch = format!("sm_{major}{minor}");

    // Takes both arms, and calls through each — so a header that resolved but
    // did not define what it claims to would still fail.
    const ROOT: &str = r#"
#include "rope_device.cuh"
#include "kv_paged_addr.cuh"

namespace k = ::pie_cuda_driver::kernels;

__global__ void probe_shared(k::device::bf16* h, float c, float s) {
    k::rotate_pair(h, 4, 0, c, s);
    k::rotate_pair_interleaved(h, 0, c, s);
}
"#;

    compile_unit(
        &arch,
        "pie_shared_probe.cu",
        ROOT,
        &[],
        Header::device_headers(),
    )
    .expect("both shared headers resolve, over one prelude");
}

/// **The second unit, fired.** `residual_add` and `scalar_mul` compiled at run
/// time, launched from Rust, and answering.
///
/// What this is evidence of is not that two pointwise kernels work — they
/// always did — but that adding a family is now a LINE. `NORM_ELEMENTWISE` is
/// a `Unit` beside `NORM`; nothing about the compile, the load, the binding or
/// the launch is written twice, and the C++ that was deleted to get here was
/// two four-line launchers whose whole content is `LaunchRule::Elementwise`.
///
/// The fp16 `residual_add` is the same claim `a_second_numeric_format_costs_
/// one_row` makes for `tanh_inplace`, one file over: the AOT build never had
/// it, and it cost a row.
#[test]
fn the_elementwise_unit_runs_and_answers() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("elementwise unit") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let arch = format!("sm_{major}{minor}");

    let (module, elapsed) = NORM_ELEMENTWISE
        .load(&arch)
        .unwrap_or_else(|e| panic!("the elementwise unit does not load:\n{e}"));
    eprintln!(
        "elementwise: {} instantiations in {:.1} ms",
        module.len(),
        elapsed.as_secs_f64() * 1e3
    );
    assert_eq!(module.len(), ELEMENTWISE.len());

    let alloc = Allocator::new();
    let stream = OwnedStream::new(0).expect("stream");
    let s = stream.as_ref();

    const N: usize = 4096;
    let y_host: Vec<u16> = (0..N).map(|i| to_bf16(sample(i))).collect();
    let x_host: Vec<u16> = (0..N).map(|i| to_bf16(sample(i * 7 + 1))).collect();

    let bind = |sig: &'static KernelSig, values: Vec<ArgValue>| {
        let geometry = eval(
            sig.launch,
            Dims {
                rows: 1,
                width: N as u32,
                in_width: N as u32,
            },
        )
        .expect("the row's rule evaluates");
        (
            geometry,
            Args::bind(sig, &values).expect("the values match the row"),
        )
    };

    // ---- residual_add: y += x -------------------------------------------
    let mut y = alloc.alloc(N * 2).expect("y");
    y.copy_from_host(&bytes_of_u16(&y_host), s)
        .expect("upload y");
    let mut x = alloc.alloc(N * 2).expect("x");
    x.copy_from_host(&bytes_of_u16(&x_host), s)
        .expect("upload x");

    let sig = ELEMENTWISE[0].sig;
    let (geometry, mut args) = bind(
        sig,
        vec![
            ArgValue::Ptr(y.ptr_at(0, N * 2).expect("y is live")),
            ArgValue::Ptr(x.ptr_at(0, N * 2).expect("x is live")),
            ArgValue::Usize(N),
        ],
    );
    module
        .fire(sig, geometry, &mut args, s)
        .expect("residual_add fires");
    s.synchronize().expect("sync");

    let mut got = vec![0u8; N * 2];
    y.copy_to_host(&mut got, s).expect("download y");
    let got = u16s_of_bytes(&got);
    for i in 0..N {
        // The kernel widens both, adds in fp32 and narrows once, so the host
        // truth is the same three steps and the comparison is exact.
        let want = to_bf16(from_bf16(y_host[i]) + from_bf16(x_host[i]));
        assert_eq!(got[i], want, "residual_add[{i}]");
    }

    // ---- scalar_mul: x *= bf16(s) ---------------------------------------
    // The scalar is rounded to bf16 BEFORE the multiply, which is the whole
    // reason this kernel exists rather than a generic scale. 1.1 is not
    // representable in bf16, so a kernel that skipped the rounding would
    // disagree here and nowhere else.
    const SCALE: f32 = 1.1;
    let mut z = alloc.alloc(N * 2).expect("z");
    z.copy_from_host(&bytes_of_u16(&y_host), s)
        .expect("upload z");

    let sig = ELEMENTWISE[1].sig;
    let (geometry, mut args) = bind(
        sig,
        vec![
            ArgValue::Ptr(z.ptr_at(0, N * 2).expect("z is live")),
            ArgValue::F32(SCALE),
            ArgValue::Usize(N),
        ],
    );
    module
        .fire(sig, geometry, &mut args, s)
        .expect("scalar_mul fires");
    s.synchronize().expect("sync");

    let mut got = vec![0u8; N * 2];
    z.copy_to_host(&mut got, s).expect("download z");
    let got = u16s_of_bytes(&got);
    let rounded = from_bf16(to_bf16(SCALE));
    for i in 0..N {
        let want = to_bf16(from_bf16(y_host[i]) * rounded);
        assert_eq!(got[i], want, "scalar_mul[{i}]");
    }
}

/// **The switch, thrown.** `norm::scalar_mul_bf16` has no `.cu` launcher and
/// no shim entry, and a fire of it still works.
///
/// This is the property the whole migration is for, and it is the one that
/// could not be tested until something was actually deleted. Every other test
/// here shows the JIT path WORKING; this one shows the AOT path is GONE:
///
/// * `csrc/src/norm/scalar_mul.cu` and its `.hpp` are deleted;
/// * `emit_c_shim` skips the row, so `pie_k_norm_scalar_mul_bf16` is not
///   emitted and nothing declares it;
/// * `emit_rust_dispatch` emits `bind::jit::fire` for it instead.
///
/// So the only way this fire reaches a GPU is through the module NVRTC built,
/// and the numbers below are the evidence that it did.
#[test]
fn a_row_with_no_launcher_still_fires() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("dispatcher switch") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let arch = format!("sm_{major}{minor}");

    // The row the dispatcher now routes to the JIT path.
    assert!(
        kernels_cuda_new::device::JIT_DISPATCHED.contains(&"norm::scalar_mul_bf16"),
        "the switch is what this test is about"
    );

    let (module, _) = NORM_ELEMENTWISE.load(&arch).expect("the unit loads");
    let sig = ELEMENTWISE[1].sig;
    assert_eq!(sig.symbol, "norm::scalar_mul_bf16");

    let alloc = Allocator::new();
    let stream = OwnedStream::new(0).expect("stream");
    let s = stream.as_ref();

    const N: usize = 1024;
    const SCALE: f32 = 1.1;
    let host: Vec<u16> = (0..N).map(|i| to_bf16(sample(i))).collect();
    let mut x = alloc.alloc(N * 2).expect("x");
    x.copy_from_host(&bytes_of_u16(&host), s).expect("upload");

    let geometry = eval(
        sig.launch,
        Dims {
            rows: 1,
            width: N as u32,
            in_width: N as u32,
        },
    )
    .expect("the row's rule evaluates");
    let mut args = Args::bind(
        sig,
        &[
            ArgValue::Ptr(x.ptr_at(0, N * 2).expect("x is live")),
            ArgValue::F32(SCALE),
            ArgValue::Usize(N),
        ],
    )
    .expect("the values match the row");
    module
        .fire(sig, geometry, &mut args, s)
        .expect("it fires with no launcher");
    s.synchronize().expect("sync");

    let mut got = vec![0u8; N * 2];
    x.copy_to_host(&mut got, s).expect("download");
    let got = u16s_of_bytes(&got);
    let rounded = from_bf16(to_bf16(SCALE));
    for i in 0..N {
        assert_eq!(
            got[i],
            to_bf16(from_bf16(host[i]) * rounded),
            "scalar_mul[{i}]"
        );
    }
}
