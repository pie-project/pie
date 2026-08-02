//! `convert`, end to end: the CLI quantizes a BF16 checkpoint to MXFP4 on the
//! host and writes a new checkpoint this crate's own parser reads back.
//!
//! The arithmetic itself is pinned by `testkit::host_executor`'s unit tests;
//! what this exercises is everything around it — argument plumbing, the
//! conversion target, plan-ordered output, the safetensors writer's header,
//! and the refusal to overwrite. The expected bytes are the unit test's,
//! restated rather than imported, so a change to either is a conscious change
//! to both.

use std::path::{Path, PathBuf};
use std::process::Command;

use half::bf16;

use pie_loader::checkpoint::read::parse_checkpoint_metadata;

fn write_snapshot(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    #[rustfmt::skip]
    let row0: [f32; 32] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        0.2, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
        0.4, 0.6, 1.1, 1.6, 2.2, 3.2, 4.5, 5.5,
    ];
    let mut data = Vec::new();
    for value in row0 {
        data.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    for _ in 0..32 {
        data.extend_from_slice(&bf16::from_f32(12.0).to_bits().to_le_bytes());
    }
    let header = r#"{"raw":{"dtype":"BF16","shape":[2,32],"data_offsets":[0,128]}}"#;
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), &file).unwrap();
}

fn contract_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/contracts/convert_bf16_to_mxfp4.json")
}

/// What `convert` writes, end to end. `.zt` is the only output: an MXFP4
/// payload spans half its element count, and the scheme is named by the file
/// rather than inferred from a dtype tag the way the interchange format
/// required.
#[test]
fn convert_writes_a_zt_checkpoint() {
    let root = std::env::temp_dir().join(format!("pie_convert_zt_{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let snapshot = root.join("snapshot");
    let out = root.join("converted");
    write_snapshot(&snapshot);

    let run = Command::new(env!("CARGO_BIN_EXE_pie-loader"))
        .args(["convert", snapshot.to_str().unwrap()])
        .arg(contract_path())
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "convert failed:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );

    let artifact = out.join("model.zt");
    assert!(
        artifact.is_file(),
        "convert did not write {}",
        artifact.display()
    );

    let parsed = parse_checkpoint_metadata(&artifact).unwrap();
    let w = parsed.tensor_by_name("w").expect("payload is in the file");
    assert_eq!(w.shape, vec![2, 32]);
    assert_eq!(w.span_bytes, 32);
    let s = parsed
        .tensor_by_name("w_scale")
        .expect("scales are in the file");
    assert_eq!(s.span_bytes, 2);

    // The bytes the host executor produced, at the offsets this file declares.
    let file = std::fs::read(&artifact).unwrap();
    let payload = &file[w.file_offset as usize..(w.file_offset + w.span_bytes) as usize];
    #[rustfmt::skip]
    let expected_row0: [u8; 16] = [
        0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE,
        0x10, 0x32, 0x54, 0x76, 0x11, 0x32, 0x54, 0x76,
    ];
    assert_eq!(&payload[..16], expected_row0);
    assert_eq!(&payload[16..], [0x77u8; 16]);
    let scales = &file[s.file_offset as usize..(s.file_offset + s.span_bytes) as usize];
    assert_eq!(scales, [127, 128]);

    // And every tensor verifies against its own digest.
    let reader = ztensor::Source::open(&artifact).unwrap();
    for name in ["w", "w_scale"] {
        assert!(
            reader.tensor(name).unwrap().verify().unwrap().is_checked(),
            "{name} carries no digest"
        );
    }

    // Provenance landed in the file's own attributes rather than a sidecar.
    let attributes = format!(
        "{:?}",
        reader
            .attributes()
            .as_ref()
            .expect("provenance is recorded")
    );
    for key in [
        "pie_convert_compiler",
        "pie_convert_contract",
        "pie_convert_source",
    ] {
        assert!(attributes.contains(key), "missing {key}");
    }

    // A destination that exists is refused, not overwritten.
    let rerun = Command::new(env!("CARGO_BIN_EXE_pie-loader"))
        .args(["convert", snapshot.to_str().unwrap()])
        .arg(contract_path())
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        !rerun.status.success(),
        "an existing destination was overwritten"
    );

    std::fs::remove_dir_all(&root).ok();
}
