//! Tier A end to end, on a real device.
//!
//! The chain this exercises is the whole experiment, and no link in it is
//! stubbed:
//!
//! ```text
//! kernels_cuda::norm_device::ENTRIES   the rows
//!   -> abi::emit_device_typecheck      the generated TU (and the proof)
//!   -> nvcc -fatbin                    the device image, no host launcher
//!   -> bind::device::KernelModule      loaded, every row's entry resolved
//!   -> bind::launch::eval              the grid the ROW states
//!   -> bind::device::Args::bind        the operands the ROW declares
//!   -> cuLaunchKernel                  fired from Rust
//! ```
//!
//! `cuda-progress.md`'s rule is why it is written this way: *"Non-GPU green is
//! weak evidence. Every dispatch defect found on 2026-08-10 compiled cleanly
//! and passed the full non-GPU battery."* A launch rule is arithmetic, and
//! arithmetic is exactly the class of thing that is right in isolation and
//! wrong against a driver.
//!
//! Skipped without a device or without nvcc, like every other `gpu_*` binary
//! here.

use std::path::PathBuf;
use std::process::Command;

use driver_cuda::bind::device::{ArgValue, Args, KernelModule};
use driver_cuda::bind::launch::{Dims, eval};
use driver_cuda::device::{Allocator, OwnedStream};
use kernels::KernelSig;
use kernels_cuda::norm_device::ENTRIES;

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
        .find(|k| k.symbol == symbol)
        .expect("the pilot states this row")
}

/// `nvcc`, wherever this machine keeps it.
fn nvcc() -> Option<PathBuf> {
    let from_env = std::env::var_os("CUDA_HOME")
        .or_else(|| std::env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"))
        .join("bin/nvcc");
    if from_env.is_file() {
        return Some(from_env);
    }
    let on_path = Command::new("nvcc").arg("--version").output().ok()?;
    on_path.status.success().then(|| PathBuf::from("nvcc"))
}

/// Emit the typecheck TU from the rows and compile it for this device.
///
/// Compiling it IS the proof that every row matches its entry point, so a
/// failure here is a table defect and the test says so rather than skipping.
fn build_fatbin(arch: &str) -> Vec<u8> {
    let text =
        kernels_cuda::abi::emit_device_typecheck(&[ENTRIES]).expect("the rows emit a typecheck TU");
    let dir = std::env::temp_dir().join(format!("pie-tier-a-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let source = dir.join("device_typecheck.cu");
    let fatbin = dir.join("tier_a.fatbin");
    std::fs::write(&source, text).expect("write the generated TU");

    let include = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../kernels-cuda/csrc/src")
        .canonicalize()
        .expect("csrc/src is where the entry points live");

    let out = Command::new(nvcc().expect("checked by the caller"))
        .args(["-std=c++20", &format!("-arch={arch}"), "-fatbin"])
        .arg("-I")
        .arg(&include)
        .arg(&source)
        .arg("-o")
        .arg(&fatbin)
        .output()
        .expect("nvcc runs");
    assert!(
        out.status.success(),
        "the generated typecheck TU did not compile, which means a ROW does not \
         match its entry point:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let image = std::fs::read(&fatbin).expect("read the fatbin");
    let _ = std::fs::remove_dir_all(&dir);
    image
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
    let Some(_) = nvcc() else {
        eprintln!("skipping Tier A launch: no nvcc to build the device image with");
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let image = build_fatbin(&format!("sm_{major}{minor}"));
    eprintln!(
        "tier A: {} bytes of fatbin for sm_{major}{minor}",
        image.len()
    );

    let module = KernelModule::load(&image, ENTRIES).expect("every row's entry resolves");
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
    let Some(_) = nvcc() else {
        eprintln!("skipping Tier A issue cost: no nvcc");
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let image = build_fatbin(&format!("sm_{major}{minor}"));
    let module = KernelModule::load(&image, ENTRIES).expect("entries resolve");

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
