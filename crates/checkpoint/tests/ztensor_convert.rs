use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use checkpoint::contract::materialize::materialize_contract;
use checkpoint::file::read::parse_metadata;
use checkpoint::file::write::{WriteTensor, write_zt, write_zt_grouped};
use checkpoint::file::zt;
use checkpoint::serving::plane_name;

use checkpoint::types::{
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

fn bytes_at(metadata: &checkpoint::file::Metadata, name: &str) -> Vec<u8> {
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

fn planes(codes: &TensorDecl, seed: u8) -> (Vec<String>, Vec<(TensorDecl, Vec<u8>)>) {
    let Encoding::Quant(spec) = &codes.encoding else {
        panic!("{} is not quantized", codes.name);
    };
    let term =
        ztensor::Term::parse(spec.term().expect("a scheme with a term").mangle().as_str()).unwrap();
    let shape: Vec<u64> = codes.shape.iter().map(|&d| d as u64).collect();
    let mut out = Vec::new();
    for (at, plane) in term.planes(&shape).unwrap().into_iter().enumerate() {
        let name = plane_name(&codes.name, &plane.path);
        let decl = match at {
            0 => codes.clone(),
            _ => {
                let leaf = match plane.leaf {
                    ztensor::Leaf::BF16 => DType::Bf16,
                    ztensor::Leaf::F16 => DType::F16,
                    ztensor::Leaf::F32 => DType::F32,
                    ztensor::Leaf::E8M0 => DType::E8m0,
                    other => panic!("{name} holds `{other}`, which no scheme here declares"),
                };
                let shape = plane.shape.iter().map(|&d| d as i64).collect();
                decl(&name, shape, Encoding::Raw(leaf))
            }
        };
        out.push((
            decl,
            pattern(plane.len as usize, seed.wrapping_add(at as u8)),
        ));
    }
    let names = out.iter().map(|(decl, _)| decl.name.clone()).collect();
    (names, out)
}

fn write_grouped(
    path: &Path,
    object: &str,
    planes: &[(TensorDecl, Vec<u8>)],
) -> Result<(), checkpoint::error::Error> {
    let tensors: Vec<WriteTensor<'_>> = planes
        .iter()
        .map(|(decl, bytes)| WriteTensor { decl, bytes })
        .collect();
    let names = planes.iter().map(|(decl, _)| decl.name.clone()).collect();
    write_zt_grouped(
        path,
        &BTreeMap::new(),
        &tensors,
        &[(object.to_string(), names)],
    )
}

#[test]
fn ztensor_convert_every_case() {
    a_converted_artifact_reads_back_byte_for_byte();
    a_corrupt_artifact_is_caught_by_its_digest();
    each_affine_group_scheme_round_trips_as_itself();
    the_artifact_names_parameters_not_schemes();
    a_gguf_block_reaches_the_artifact_as_stored_and_says_what_it_means();
    every_scheme_the_bridge_names_is_stamped_and_every_one_it_refuses_is_not();
    a_plain_tensor_carries_its_own_leaf();
}

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

    let metadata = zt::parse(&path).unwrap();
    assert_eq!(metadata.tensors.len(), 3);

    for (name, expected) in [
        ("model.embed.weight", &embed),
        ("model.norm.weight", &norm),
        ("model.bias", &bias),
    ] {
        assert_eq!(&bytes_at(&metadata, name), expected, "{name}");
    }

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

fn each_affine_group_scheme_round_trips_as_itself() {
    enum Back {
        Scheme(QuantScheme),
        NoType,
    }
    let dir = tmpdir("schemes");
    for (scheme, group, bits, back) in [
        (QuantScheme::AwqInt4, 128u32, 4u8, Back::NoType),
        (QuantScheme::GptqInt4, 128, 4, Back::NoType),
        (
            QuantScheme::MlxAffineU4,
            64,
            4,
            Back::Scheme(QuantScheme::MlxAffineU4),
        ),
        (
            QuantScheme::Int4B8,
            32,
            4,
            Back::Scheme(QuantScheme::Int4B8),
        ),
        (
            QuantScheme::Int8Symmetric,
            0,
            8,
            Back::Scheme(QuantScheme::Int8Symmetric),
        ),
        (QuantScheme::Int8Asymmetric, 0, 8, Back::NoType),
    ] {
        let path = dir.join(format!("{scheme:?}.zt"));
        let spec = QuantSpec {
            scheme,
            logical_dtype: DType::Bf16,
            bits_per_element: bits,
            group_size: group,
            channel_axis: None,
        };
        let d = decl("w", vec![1024], Encoding::Quant(spec.clone()));
        let Back::Scheme(back) = back else {
            let err = write_zt(
                &path,
                &BTreeMap::new(),
                &[WriteTensor {
                    decl: &d,
                    bytes: &[0u8; 1024],
                }],
            )
            .expect_err("a scheme with no term cannot be stated");
            assert!(err.to_string().contains("no type"), "{scheme:?}: {err}");
            continue;
        };
        let (_, planes) = planes(&d, 0x5a);
        write_grouped(&path, "w", &planes)
            .unwrap_or_else(|err| panic!("{scheme:?} could not be written: {err}"));
        let manifest = ztensor::read::manifest_of(&path).unwrap().unwrap();
        assert_eq!(
            manifest.objects["w"]
                .term
                .as_ref()
                .map(ToString::to_string)
                .as_deref(),
            Some(spec.term().unwrap().mangle().as_str()),
            "{scheme:?}: the file states another type"
        );

        let metadata = zt::parse(&path).unwrap_or_else(|err| {
            panic!("{scheme:?} was written but could not be read back: {err}")
        });
        let w = metadata.tensor_by_name("w").unwrap();
        match &w.encoding {
            Encoding::Quant(got) => {
                assert_eq!(
                    got.scheme, back,
                    "{scheme:?} came back as {:?} — the type did not identify it",
                    got.scheme
                );
                assert_eq!(got.bits_per_element, bits, "{scheme:?}: bits");
            }
            other => panic!("{scheme:?} read back as {other:?}"),
        }
        for (decl, bytes) in &planes {
            assert_eq!(
                &bytes_at(&metadata, &decl.name),
                bytes,
                "{scheme:?}: {}",
                decl.name
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

fn the_artifact_names_parameters_not_schemes() {
    let dir = tmpdir("parametric");
    let path = dir.join("model.zt");
    let spec = QuantSpec {
        scheme: QuantScheme::MlxAffineU4,
        logical_dtype: DType::Bf16,
        bits_per_element: 4,
        group_size: 128,
        channel_axis: None,
    };
    let d = decl("w", vec![1024], Encoding::Quant(spec.clone()));
    let (_, planes) = planes(&d, 0x11);
    write_grouped(&path, "w", &planes).unwrap();

    let manifest = ztensor::read::manifest_of(&path).unwrap().unwrap();
    let object = &manifest.objects["w"];
    assert_eq!(object.layout, None, "the planes lie canonically");
    assert_eq!(
        object.attributes, None,
        "nothing beside the type describes them"
    );
    let stated = object
        .term
        .as_ref()
        .expect("the type is recorded")
        .to_string();
    assert_eq!(stated, spec.term().unwrap().mangle().as_str());
    for parameter in ["g128", "u4", "bf16"] {
        assert!(
            stated.contains(parameter),
            "{stated} does not state {parameter}"
        );
    }
    let rendered = format!("{manifest:?}").to_ascii_lowercase();
    assert!(
        !rendered.contains("mlx") && !rendered.contains("affine"),
        "the file carries the scheme's name: {rendered}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

fn gguf(tensors: &[(&str, Vec<u64>, u32, Vec<u8>)]) -> Vec<u8> {
    const ALIGN: usize = 32;
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&0u64.to_le_bytes());
    let mut data = Vec::new();
    for (name, shape, type_id, payload) in tensors {
        head.extend_from_slice(&(name.len() as u64).to_le_bytes());
        head.extend_from_slice(name.as_bytes());
        head.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dim in shape.iter().rev() {
            head.extend_from_slice(&dim.to_le_bytes());
        }
        head.extend_from_slice(&type_id.to_le_bytes());
        head.extend_from_slice(&(data.len() as u64).to_le_bytes());
        data.extend_from_slice(payload);
        while !data.len().is_multiple_of(ALIGN) {
            data.push(0);
        }
    }
    while !head.len().is_multiple_of(ALIGN) {
        head.push(0);
    }
    head.extend_from_slice(&data);
    head
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(seed))
        .collect()
}

fn type_of(source: &ztensor::Source, name: &str) -> Option<String> {
    source
        .get(name)
        .expect("the artifact holds the tensor")
        .term()
        .map(ToString::to_string)
}

fn convert(source_dir: &std::path::Path, metadata: &checkpoint::file::Metadata, out: &Path) {
    use checkpoint::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};

    let materialization = materialize_contract(metadata).unwrap();
    let decoded = if materialization.contract.tensors.is_empty() {
        Default::default()
    } else {
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };
        let plan = checkpoint::plan::compile(metadata, &materialization.contract, target).unwrap();
        let storage = checkpoint::executor::Execution::new(&plan, source_dir)
            .run()
            .unwrap();
        plan.tensors
            .iter()
            .filter(|decl| decl.visibility.is_public())
            .map(|decl| {
                (
                    decl.name.clone(),
                    (decl.clone(), storage.tensors[&decl.name].clone()),
                )
            })
            .collect::<BTreeMap<_, _>>()
    };

    let mut entries: Vec<&str> = materialization
        .decoded
        .iter()
        .chain(materialization.passthrough.iter())
        .map(String::as_str)
        .collect();
    entries.sort_unstable();

    let mut writer = checkpoint::file::write::Writer::create(out, &BTreeMap::new()).unwrap();
    for name in entries {
        match decoded.get(name) {
            Some((decl, bytes)) => writer.add_tensor(decl, bytes).unwrap(),
            None => {
                let raw = metadata.tensor_by_name(name).unwrap();
                let file = metadata
                    .files
                    .iter()
                    .find(|file| file.id == raw.file_id)
                    .unwrap();
                let mut bytes = std::fs::read(source_dir.join(&file.path)).unwrap();
                let start = raw.file_offset as usize;
                bytes = bytes[start..start + raw.span_bytes as usize].to_vec();
                let decl = TensorDecl {
                    id: raw.id,
                    name: raw.name.clone(),
                    shape: raw.shape.clone(),
                    encoding: raw.encoding.clone(),
                    alignment: 1,
                    visibility: Visibility::default(),
                };
                writer.begin_tensor(&decl, raw.span_bytes).unwrap();
                writer.write(&bytes).unwrap();
                writer.end_tensor().unwrap();
            }
        }
    }
    writer.finish().unwrap();
}

fn a_gguf_block_reaches_the_artifact_as_stored_and_says_what_it_means() {
    let dir = tmpdir("stored");
    let source = dir.join("model.gguf");

    let q4_k = pattern(144, 0x11);
    let iq2 = pattern(66, 0x77);
    let f32s: Vec<u8> = [1.0f32, -2.0, 0.5, 384.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let bf16s: Vec<u8> = [1.0f32, -2.0, 0.5, 384.0]
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect();
    std::fs::write(
        &source,
        gguf(&[
            ("block.q4_k", vec![256], 12, q4_k.clone()),
            ("lattice.iq2_xxs", vec![256], 16, iq2.clone()),
            ("plain.bf16", vec![4], 30, bf16s.clone()),
            ("plain.f32", vec![4], 0, f32s.clone()),
        ]),
    )
    .unwrap();

    let metadata = parse_metadata(&source).unwrap();
    let materialization = materialize_contract(&metadata).unwrap();
    assert_eq!(materialization.decoded, ["plain.f32"]);
    assert_eq!(
        materialization.passthrough,
        ["block.q4_k", "lattice.iq2_xxs", "plain.bf16"]
    );

    let out = dir.join("model.zt");
    convert(&dir, &metadata, &out);

    let reader = ztensor::Source::open(&out).unwrap();
    assert_eq!(
        reader.get("block.q4_k").unwrap().layout(),
        Some("gguf.q4_k/2")
    );
    assert_eq!(
        reader.get("lattice.iq2_xxs").unwrap().layout(),
        Some("gguf.iq2_xxs/2")
    );
    assert_eq!(reader.get("plain.bf16").unwrap().layout(), None);
    assert_eq!(reader.get("plain.f32").unwrap().layout(), None);

    let artifact = parse_metadata(&out).unwrap();
    assert_eq!(bytes_at(&artifact, "block.q4_k"), q4_k);
    assert_eq!(bytes_at(&artifact, "lattice.iq2_xxs"), iq2);
    assert_eq!(bytes_at(&artifact, "plain.bf16"), bf16s);
    assert_eq!(bytes_at(&artifact, "plain.f32"), bf16s);

    match &artifact.tensor_by_name("block.q4_k").unwrap().encoding {
        Encoding::Quant(spec) => assert_eq!(spec.scheme, QuantScheme::GgufQ4K),
        other => panic!("the block read back as {other:?}"),
    }

    let q4_k = QuantSpec {
        scheme: QuantScheme::GgufQ4K,
        logical_dtype: DType::Bf16,
        bits_per_element: 0,
        group_size: 0,
        channel_axis: None,
    }
    .term()
    .expect("Q4_K has a term");
    assert_eq!(q4_k.mangle().as_str(), "g32_u4_g8_u6_f16_n_b_g8_u6_f16_n");
    assert_eq!(
        type_of(&reader, "block.q4_k").as_deref(),
        Some(q4_k.mangle().as_str())
    );

    assert_eq!(type_of(&reader, "plain.bf16").as_deref(), Some("bf16"));
    assert_eq!(type_of(&reader, "plain.f32").as_deref(), Some("bf16"));

    assert_eq!(
        QuantSpec {
            scheme: QuantScheme::GgufIq2Xxs,
            logical_dtype: DType::Bf16,
            bits_per_element: 0,
            group_size: 0,
            channel_axis: None,
        }
        .term(),
        None,
        "the bridge would have to name a lattice to stamp one"
    );
    assert_eq!(type_of(&reader, "lattice.iq2_xxs"), None);
    let rendered = format!(
        "{:?}",
        reader.get("lattice.iq2_xxs").unwrap().attributes().unwrap()
    );
    assert!(rendered.contains("elems_per_block"), "{rendered}");
    assert!(rendered.contains("block_bytes"), "{rendered}");

    std::fs::remove_dir_all(&dir).ok();
}

fn every_scheme_the_bridge_names_is_stamped_and_every_one_it_refuses_is_not() {
    let dir = tmpdir("stamp");
    for (scheme, logical) in [
        (QuantScheme::GgufQ4_0, DType::Bf16),
        (QuantScheme::GgufQ4_1, DType::Bf16),
        (QuantScheme::GgufQ2K, DType::Bf16),
        (QuantScheme::GgufQ3K, DType::Bf16),
        (QuantScheme::GgufQ4K, DType::Bf16),
        (QuantScheme::GgufQ5_0, DType::Bf16),
        (QuantScheme::GgufQ5_1, DType::Bf16),
        (QuantScheme::GgufQ5K, DType::Bf16),
        (QuantScheme::GgufQ6K, DType::Bf16),
        (QuantScheme::GgufQ8_0, DType::Bf16),
        (QuantScheme::GgufMxfp4, DType::Bf16),
        (QuantScheme::GgufIq4Nl, DType::Bf16),
        (QuantScheme::GgufIq4Xs, DType::Bf16),
        (QuantScheme::GgufIq2Xxs, DType::Bf16),
        (QuantScheme::GgufIq2Xs, DType::Bf16),
        (QuantScheme::GgufIq2S, DType::Bf16),
        (QuantScheme::GgufIq3Xxs, DType::Bf16),
        (QuantScheme::GgufIq3S, DType::Bf16),
        (QuantScheme::AwqInt4, DType::Bf16),
        (QuantScheme::GptqInt4, DType::Bf16),
        (QuantScheme::MlxAffineU4, DType::Bf16),
        (QuantScheme::Int4B8, DType::Bf16),
        (QuantScheme::Int8Symmetric, DType::Bf16),
        (QuantScheme::Int8Asymmetric, DType::Bf16),
        (QuantScheme::Mxfp4E2M1E8M0, DType::Bf16),
        (QuantScheme::Fp8E4M3, DType::E4m3),
        (QuantScheme::Fp8E5M2, DType::E5m2),
    ] {
        let spec = QuantSpec {
            scheme,
            logical_dtype: logical,
            bits_per_element: 0,
            group_size: 0,
            channel_axis: None,
        }
        .normalized();
        let path = dir.join(format!("{scheme:?}.zt"));
        let expected =
            checkpoint::term_of(&Encoding::Quant(spec.clone())).map(|term| term.to_string());
        let d = decl("w", vec![256], Encoding::Quant(spec.clone()));
        if let Some((elems, bytes)) = spec.block_layout() {
            let payload = pattern(256 / elems as usize * bytes as usize, 0x2b);
            write_zt(
                &path,
                &BTreeMap::new(),
                &[WriteTensor {
                    decl: &d,
                    bytes: &payload,
                }],
            )
            .unwrap_or_else(|err| panic!("{scheme:?} could not be written: {err}"));
        } else if expected.is_some() {
            let (_, planes) = planes(&d, 0x2b);
            write_grouped(&path, "w", &planes)
                .unwrap_or_else(|err| panic!("{scheme:?} could not be written: {err}"));
        } else {
            let err = write_zt(
                &path,
                &BTreeMap::new(),
                &[WriteTensor {
                    decl: &d,
                    bytes: &[0u8; 256],
                }],
            )
            .expect_err("a scheme the bridge refuses has no type to write");
            assert!(err.to_string().contains("no type"), "{scheme:?}: {err}");
            continue;
        }

        let reader = ztensor::Source::open(&path).unwrap();
        assert_eq!(
            type_of(&reader, "w"),
            expected,
            "{scheme:?}: the file disagrees with the bridge"
        );
        zt::parse(&path)
            .unwrap_or_else(|err| panic!("{scheme:?} was typed into unreadability: {err}"));
    }
    std::fs::remove_dir_all(&dir).ok();
}

fn a_plain_tensor_carries_its_own_leaf() {
    let dir = tmpdir("plain");
    for (dtype, spelling, width) in [
        (DType::Bf16, "bf16", 2usize),
        (DType::F32, "f32", 4),
        (DType::F16, "f16", 2),
        (DType::I8, "i8", 1),
        (DType::U8, "u8", 1),
    ] {
        let path = dir.join(format!("{dtype:?}.zt"));
        let payload = pattern(16 * width, 0x5c);
        let d = decl("w", vec![16], Encoding::Raw(dtype));
        write_zt(
            &path,
            &BTreeMap::new(),
            &[WriteTensor {
                decl: &d,
                bytes: &payload,
            }],
        )
        .unwrap();
        let reader = ztensor::Source::open(&path).unwrap();
        assert_eq!(reader.get("w").unwrap().layout(), None);
        assert_eq!(type_of(&reader, "w").as_deref(), Some(spelling));
        assert_eq!(dtype.repr().mangle().as_str(), spelling);
    }
    std::fs::remove_dir_all(&dir).ok();
}
