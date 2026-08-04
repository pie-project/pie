//! What the zTensor reader costs, against the reader it replaces.
//!
//! Header parsing is not on the hot path — a load spends its time moving
//! bytes, not reading a manifest — but it is on the *latency* path: nothing
//! can be planned until the checkpoint's facts are known, and a 32k-tensor
//! checkpoint made that measurable once before (see `Sources`' comment about
//! the quadratic name lookup).
//!
//! So this is a guard, not a benchmark: it asserts the zTensor path stays
//! within a wide multiple of the native one on the same file. Run with
//! `--nocapture` to see the numbers.
//!
//! `cargo test -p pie-loader --release --test ztensor_perf -- --nocapture`

use std::path::PathBuf;
use std::time::{Duration, Instant};

use pie_loader::checkpoint::read::parse_checkpoint_metadata;
use pie_loader::checkpoint::zt::parse_checkpoint;
use pie_loader::checkpoint::zt::parse_checkpoint_files;

fn tmpdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("zt_perf_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A safetensors file with `count` small tensors — the shape that stresses
/// header parsing rather than byte movement.
fn write_safetensors(dir: &std::path::Path, count: usize) -> PathBuf {
    let per = 64usize; // bytes per tensor
    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        let begin = i * per;
        entries.push(format!(
            r#""blk.{i:05}.attn.weight":{{"dtype":"F32","shape":[16],"data_offsets":[{begin},{}]}}"#,
            begin + per
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend(std::iter::repeat_n(0u8, count * per));
    let path = dir.join("model.safetensors");
    std::fs::write(&path, &out).unwrap();
    path
}

fn median(mut xs: Vec<Duration>) -> Duration {
    xs.sort();
    xs[xs.len() / 2]
}

fn time(runs: usize, mut f: impl FnMut()) -> Duration {
    // One warmup: the first open faults the file in.
    f();
    median(
        (0..runs)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed()
            })
            .collect(),
    )
}

#[test]
fn reading_a_checkpoint_through_ztensor_stays_within_budget() {
    const TENSORS: usize = 8192;
    const RUNS: usize = 5;

    let dir = tmpdir("parse");
    let path = write_safetensors(&dir, TENSORS);
    let files = vec![path.clone()];

    let native = time(RUNS, || {
        let m = parse_checkpoint_files(&files).unwrap();
        std::hint::black_box(m.tensors.len());
    });
    let through_zt = time(RUNS, || {
        let m = parse_checkpoint(&path).unwrap();
        std::hint::black_box(m.tensors.len());
    });

    println!(
        "parse {TENSORS} tensors: native {:.2} ms, zTensor {:.2} ms ({:.2}x)",
        native.as_secs_f64() * 1e3,
        through_zt.as_secs_f64() * 1e3,
        through_zt.as_secs_f64() / native.as_secs_f64().max(f64::MIN_POSITIVE),
    );

    // Both readers parse the same JSON header; zTensor then builds an object
    // model and validates it, so it is expected to be somewhat slower. The
    // budget is deliberately loose — this catches an order-of-magnitude
    // regression (a quadratic lookup, a per-tensor allocation storm), not a
    // constant factor.
    assert!(
        through_zt < native * 8 + Duration::from_millis(50),
        "zTensor parse {:.2} ms against native {:.2} ms — more than the 8x budget",
        through_zt.as_secs_f64() * 1e3,
        native.as_secs_f64() * 1e3,
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Reading the `.zt` artifact `convert` writes, against reading the
/// safetensors it replaced. The manifest is CBOR rather than JSON and the
/// reader validates every blob reference, so this is the number that decides
/// whether the artifact format costs anything at load.
#[test]
fn reading_a_zt_artifact_is_not_slower_than_safetensors() {
    const TENSORS: usize = 8192;
    const RUNS: usize = 5;

    let dir = tmpdir("artifact");
    let st_path = write_safetensors(&dir, TENSORS);

    // The same tensors as a .zt artifact.
    let zt_path = dir.join("model.zt");
    {
        let mut w = ztensor::Writer::create(&zt_path).unwrap();
        let payload = vec![0u8; 64];
        for i in 0..TENSORS {
            w.add(
                &format!("blk.{i:05}.attn.weight"),
                &[16],
                ztensor::DType::F32,
                &payload,
            )
            .unwrap();
        }
        w.finish().unwrap();
    }

    let st_files = vec![st_path];
    let safetensors = time(RUNS, || {
        let m = parse_checkpoint_files(&st_files).unwrap();
        std::hint::black_box(m.tensors.len());
    });
    let zt = time(RUNS, || {
        let m = parse_checkpoint_metadata(&zt_path).unwrap();
        std::hint::black_box(m.tensors.len());
    });

    println!(
        "open {TENSORS}-tensor artifact: safetensors {:.2} ms, .zt {:.2} ms ({:.2}x)",
        safetensors.as_secs_f64() * 1e3,
        zt.as_secs_f64() * 1e3,
        zt.as_secs_f64() / safetensors.as_secs_f64().max(f64::MIN_POSITIVE),
    );

    assert!(
        zt < safetensors * 8 + Duration::from_millis(50),
        ".zt open {:.2} ms against safetensors {:.2} ms — more than the 8x budget",
        zt.as_secs_f64() * 1e3,
        safetensors.as_secs_f64() * 1e3,
    );

    std::fs::remove_dir_all(&dir).ok();
}
