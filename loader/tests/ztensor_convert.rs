//! The convert path, end to end: a checkpoint in, a `.zt` artifact out, and
//! the artifact read back as a checkpoint.
//!
//! `pie model convert` writes what the engine will later load, so the two
//! halves have to agree about more than shapes: the bytes a plan addresses in
//! the artifact must be the bytes the executor produced. These tests run the
//! write and the read against each other and compare payloads, not just
//! metadata.

use std::collections::BTreeMap;
use std::path::PathBuf;

use pie_loader::checkpoint::write::WriteTensor;
use pie_loader::checkpoint::write::write_zt;
use pie_loader::checkpoint::zt::parse_checkpoint;
use pie_loader::types::{
    DType, Encoding, QuantScheme, QuantSpec, TensorDecl, TensorId, Visibility,
};

fn tmpdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("zt_convert_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn decl(name: &str, shape: Vec<i64>, encoding: Encoding) -> TensorDecl {
    TensorDecl {
        id: TensorId(0),
        name: name.to_string(),
        shape,
        encoding,
        alignment: 64,
        visibility: Visibility::default(),
    }
}

/// Reads back the bytes a tensor's plan coordinates point at.
fn bytes_at(metadata: &pie_loader::checkpoint::CheckpointMetadata, name: &str) -> Vec<u8> {
    use std::io::{Read, Seek, SeekFrom};
    let tensor = metadata.tensor_by_name(name).expect("tensor present");
    let file = metadata
        .files
        .iter()
        .find(|f| f.id == tensor.file_id)
        .expect("file present");
    let mut handle = std::fs::File::open(&file.path).unwrap();
    handle.seek(SeekFrom::Start(tensor.file_offset)).unwrap();
    let mut out = vec![0u8; tensor.span_bytes as usize];
    handle.read_exact(&mut out).unwrap();
    out
}

/// A model of the shapes `convert` actually produces — plain dtypes decoded
/// from a blocked checkpoint, several tensors, mixed widths — written and read
/// back with every payload compared.
#[test]
fn a_converted_artifact_reads_back_byte_for_byte() {
    let dir = tmpdir("artifact");
    let path = dir.join("model.zt");

    let embed: Vec<u8> = (0..4096u32)
        .flat_map(|i| ((i % 97) as f32).to_le_bytes())
        .collect();
    let norm: Vec<u8> = (0..256u32)
        .flat_map(|i| ((i % 13) as f32).to_le_bytes())
        .collect();
    let bias = vec![3u8; 64];

    let d_embed = decl(
        "model.embed.weight",
        vec![1024, 4],
        Encoding::Raw(DType::F32),
    );
    let d_norm = decl("model.norm.weight", vec![256], Encoding::Raw(DType::F32));
    let d_bias = decl("model.bias", vec![64], Encoding::Raw(DType::U8));

    let mut provenance = BTreeMap::new();
    provenance.insert("pie_convert".to_string(), "normalize".to_string());
    provenance.insert("pie_convert_source".to_string(), "deadbeef".to_string());

    write_zt(
        &path,
        &provenance,
        &[
            WriteTensor {
                decl: &d_embed,
                bytes: &embed,
            },
            WriteTensor {
                decl: &d_norm,
                bytes: &norm,
            },
            WriteTensor {
                decl: &d_bias,
                bytes: &bias,
            },
        ],
    )
    .unwrap();

    let metadata = parse_checkpoint(&path).unwrap();
    assert_eq!(metadata.tensors.len(), 3);

    for (name, expected) in [
        ("model.embed.weight", &embed),
        ("model.norm.weight", &norm),
        ("model.bias", &bias),
    ] {
        assert_eq!(&bytes_at(&metadata, name), expected, "{name}");
    }

    // Every tensor is page-placed, which is what makes the artifact
    // streamable without the align rewrite.
    for tensor in &metadata.tensors {
        assert_eq!(
            tensor.file_offset % 65536,
            0,
            "{} is not page-placed",
            tensor.name
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// Corruption in an artifact is an error, not a wrong answer — the property
/// the safetensors output could not provide.
#[test]
fn a_corrupt_artifact_is_caught_by_its_digest() {
    let dir = tmpdir("corrupt");
    let path = dir.join("model.zt");
    let bytes = vec![42u8; 4096];
    let d = decl("w", vec![4096], Encoding::Raw(DType::U8));
    write_zt(
        &path,
        &BTreeMap::new(),
        &[WriteTensor {
            decl: &d,
            bytes: &bytes,
        }],
    )
    .unwrap();

    // The manifest is untouched, so the file still opens; the tensor's own
    // digest is what fails.
    let mut raw = std::fs::read(&path).unwrap();
    raw[65536] ^= 0xff;
    std::fs::write(&path, &raw).unwrap();

    let reader = ztensor::Source::open(&path).expect("still opens: the manifest is intact");
    let err = reader.tensor("w").unwrap().verify().unwrap_err();
    assert!(
        format!("{err}").contains("digest mismatch"),
        "expected a digest mismatch, got {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The quantized schemes safetensors has no tag for, and the property that
/// makes the profile parametric: each one comes back as *itself*.
///
/// AWQ and GPTQ differ only in packing order, and MLX-affine only in how its
/// zero points are stored. If the reader recovered the scheme from a name the
/// writer had left behind, this test would pass while telling us nothing. It
/// passes because the parameters are in the file.
#[test]
fn each_affine_group_scheme_round_trips_as_itself() {
    let dir = tmpdir("schemes");
    for (scheme, group, bits) in [
        (QuantScheme::AwqInt4, 128u32, 4u8),
        (QuantScheme::GptqInt4, 128, 4),
        (QuantScheme::MlxAffineU4, 64, 4),
        (QuantScheme::Int4B8, 32, 4),
        (QuantScheme::Int8Symmetric, 0, 8),
        (QuantScheme::Int8Asymmetric, 0, 8),
    ] {
        let path = dir.join(format!("{scheme:?}.zt"));
        let payload = vec![0x5au8; 512];
        let spec = QuantSpec {
            scheme,
            logical_dtype: DType::BF16,
            bits_per_element: bits,
            group_size: group,
            channel_axis: None,
        };
        let d = decl("w", vec![1024], Encoding::Quant(spec));
        write_zt(
            &path,
            &BTreeMap::new(),
            &[WriteTensor {
                decl: &d,
                bytes: &payload,
            }],
        )
        .unwrap_or_else(|err| panic!("{scheme:?} could not be written: {err}"));

        let metadata = parse_checkpoint(&path).unwrap_or_else(|err| {
            panic!("{scheme:?} was written but could not be read back: {err}")
        });
        let w = metadata.tensor_by_name("w").unwrap();
        match &w.encoding {
            Encoding::Quant(got) => {
                assert_eq!(
                    got.scheme, scheme,
                    "{scheme:?} came back as {:?} — the parameters did not identify it",
                    got.scheme
                );
                assert_eq!(got.bits_per_element, bits, "{scheme:?}: bits");
            }
            other => panic!("{scheme:?} read back as {other:?}"),
        }
        assert_eq!(bytes_at(&metadata, "w"), payload, "{scheme:?}: payload");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// No name is carried. The file describes the scheme by its parameters, so a
/// reader that never heard of pie's enum can still decode it — and a reader
/// that has one recovers it without being told.
#[test]
fn the_artifact_names_parameters_not_schemes() {
    let dir = tmpdir("parametric");
    let path = dir.join("model.zt");
    let spec = QuantSpec {
        scheme: QuantScheme::GptqInt4,
        logical_dtype: DType::BF16,
        bits_per_element: 4,
        group_size: 128,
        channel_axis: None,
    };
    let d = decl("w", vec![1024], Encoding::Quant(spec));
    write_zt(
        &path,
        &BTreeMap::new(),
        &[WriteTensor {
            decl: &d,
            bytes: &vec![0u8; 512],
        }],
    )
    .unwrap();

    let reader = ztensor::Source::open(&path).unwrap();
    let object = reader.get("w").unwrap();
    assert_eq!(object.layout(), "zt.quant_group/1");

    let attributes = object.attributes().expect("parameters are recorded");
    let rendered = format!("{attributes:?}");
    for parameter in ["bits", "group_size", "packing", "scale_form", "zero_point"] {
        assert!(rendered.contains(parameter), "missing {parameter}");
    }
    // The scheme's own name appears nowhere: that is the point.
    assert!(
        !rendered.contains("GptqInt4"),
        "the file carries the scheme's name: {rendered}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
