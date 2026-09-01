//! The convert path, end to end: a checkpoint in, a `.zt` artifact out, and
//! the artifact read back as a checkpoint.
//!
//! `pie model import` writes what the runtime will later load, so the two
//! halves have to agree about more than shapes: the bytes a plan addresses in
//! the artifact must be the bytes the executor produced. These tests run the
//! write and the read against each other and compare payloads, not just
//! metadata.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use checkpoint::contract::materialize::materialize_contract;
use checkpoint::file::read::parse_metadata;
use checkpoint::file::write::WriteTensor;
use checkpoint::file::write::write_zt;
use checkpoint::file::zt;

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

/// Reads back the bytes a tensor's plan coordinates point at.
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

    let metadata = zt::parse(&path).unwrap();
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
            logical_dtype: DType::Bf16,
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

        let metadata = zt::parse(&path).unwrap_or_else(|err| {
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
        logical_dtype: DType::Bf16,
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

// ─────────────────────────────────────────────────────────────────────────
// SERVE-AS-STORED, END TO END (campaign J5, step 1)
//
// A GGUF in, an artifact out, and the two questions the serving wave will
// ask of it: are the block's bytes still the block's bytes, and does the
// artifact say what they MEAN. Everything above writes a hand-built
// declaration; the first below writes a real GGUF header and lets
// `materialize_contract` decide the split, because the split is the claim.
// ─────────────────────────────────────────────────────────────────────────

/// A GGUF v3 file: header, tensor infos, then the data section on a 32-byte
/// boundary.
///
/// Built here rather than fetched because the fixture has to hold schemes no
/// small public checkpoint carries together — a K-quant, an IQ lattice and two
/// plain widths in one file — and because a downloaded file would make these
/// tests depend on the network. The layout is the one `ztensor-compat`'s
/// `gguf` projection reads: magic, `u32` version, `u64` tensor count, `u64` KV
/// count, the KVs, then per tensor a name, a dimension count, the dimensions
/// **fastest-first**, a ggml type id and an offset into the data section.
///
/// No metadata KVs, so `general.alignment` takes its default of 32.
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
        // ggml stores dimensions fastest-first; the projection reverses them
        // back to row-major, so the caller writes row-major and this reverses.
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

/// A payload nothing could produce by accident, so a copy that lost or
/// reordered a byte reads differently.
fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(seed))
        .collect()
}

/// The `qnf` attribute a written object carries, or `None` when it carries
/// none.
fn qnf_of(source: &ztensor::Source, name: &str) -> Option<String> {
    let object = source.get(name).expect("the artifact holds the tensor");
    let attributes = object.attributes()?;
    match attributes.get("qnf")? {
        ztensor::format::cbor::Value::Text(spelling) => Some(spelling.clone()),
        other => panic!("{name}: qnf is {other:?} and not text"),
    }
}

/// Writes what `pie model import` would write for `metadata`: the decoded set
/// through the plan, the passthrough set copied, both in ascending name order.
///
/// The same two calls the command makes — `Writer::add_tensor` for a decoded
/// tensor and `begin_tensor`/`write`/`end_tensor` for a copy — so a test that
/// passes here is a statement about the command and not about a shortcut.
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
        // `compile` and not `compile_streaming`, which is what the command
        // itself uses (`ops::model::import::compile_decode`): the two differ
        // only in the SCHEDULE, and these tests read payloads. The schedule is
        // pinned where it matters, beside the call site.
        let plan =
            checkpoint::plan::compile(metadata, &materialization.contract, target).unwrap();
        let storage = checkpoint::executor::Execution::new(&plan, source_dir)
            .run()
            .unwrap();
        plan.tensors
            .iter()
            .filter(|decl| decl.visibility.is_public())
            .map(|decl| (decl.name.clone(), (decl.clone(), storage.tensors[&decl.name].clone())))
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

/// A GGUF's blocks reach the artifact byte for byte, under `.zt`'s own name
/// for the scheme, with the QNF spelling of what they mean beside it.
///
/// **The passthrough half is a PIN, not a change.** A self-contained block has
/// been kept as stored since the decode moved to the point that needs it
/// unpacked, and this states that in the terms the serving wave will use: a
/// Q4_K tensor's 144 bytes are the source's 144 bytes, the profile is
/// `gguf.q4_k/1`, and nothing about the file says bf16. Without this, the
/// property is asserted only against a hand-built `Metadata` in
/// `contract::materialize`'s own tests and never against a real GGUF header.
///
/// **The `qnf` half is new.** A layout profile says how bytes are ADDRESSED;
/// two schemes can share one and mean different arithmetic. The attribute says
/// what they mean, in the one spelling a kernel table can be keyed on.
#[test]
fn a_gguf_block_reaches_the_artifact_as_stored_and_says_what_it_means() {
    let dir = tmpdir("stored");
    let source = dir.join("model.gguf");

    // 256 elements each: a Q4_K super-block is 144 bytes and an IQ2_XXS one is
    // 66. Four plain f32 and four plain bf16 behind them.
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
    // The whole of the split, stated: only the width no kernel reads is
    // rewritten. Both blocks stay stored — the IQ one as much as the K one,
    // because keeping bytes needs no decoder, only a name to keep them under.
    assert_eq!(materialization.decoded, ["plain.f32"]);
    assert_eq!(
        materialization.passthrough,
        ["block.q4_k", "lattice.iq2_xxs", "plain.bf16"]
    );

    let out = dir.join("model.zt");
    convert(&dir, &metadata, &out);

    let reader = ztensor::Source::open(&out).unwrap();
    assert_eq!(reader.get("block.q4_k").unwrap().layout(), "gguf.q4_k/1");
    assert_eq!(
        reader.get("lattice.iq2_xxs").unwrap().layout(),
        "gguf.iq2_xxs/1"
    );
    assert_eq!(reader.get("plain.bf16").unwrap().layout(), "dense");
    assert_eq!(reader.get("plain.f32").unwrap().layout(), "dense");

    // Byte for byte, through the reader's own coordinates.
    let artifact = parse_metadata(&out).unwrap();
    assert_eq!(bytes_at(&artifact, "block.q4_k"), q4_k);
    assert_eq!(bytes_at(&artifact, "lattice.iq2_xxs"), iq2);
    assert_eq!(bytes_at(&artifact, "plain.bf16"), bf16s);
    // The one rewrite: f32 narrowed to the width every kernel reads.
    assert_eq!(bytes_at(&artifact, "plain.f32"), bf16s);

    // And the scheme survives a full round trip, so the artifact can be
    // re-read as the same quantized tensor rather than as opaque bytes.
    match &artifact.tensor_by_name("block.q4_k").unwrap().encoding {
        Encoding::Quant(spec) => assert_eq!(spec.scheme, QuantScheme::GgufQ4K),
        other => panic!("the block read back as {other:?}"),
    }

    // What the bytes MEAN, in QNF. Read off the bridge rather than typed out,
    // so a row that moves moves here too; the literal is asserted beside it
    // because a spelling is a wire fact once a kernel table keys on it.
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
    assert_eq!(qnf_of(&reader, "block.q4_k").as_deref(), Some(q4_k.mangle().as_str()));

    // A decoded tensor and a copied one that hold the same width say the same
    // word: the stamp follows the encoding, not the route.
    assert_eq!(qnf_of(&reader, "plain.bf16").as_deref(), Some("bf16"));
    assert_eq!(qnf_of(&reader, "plain.f32").as_deref(), Some("bf16"));

    // The refusal, and it is silence rather than a guess. An IQ lattice's
    // points are compiled into llama.cpp, so no group width and no code leaf
    // describes its bytes — the tensor keeps its profile and carries no
    // spelling at all.
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
    assert_eq!(qnf_of(&reader, "lattice.iq2_xxs"), None);
    // And the profile's own attributes are still there: the stamp ADDS a key
    // or adds nothing, and a scheme with no signature must not lose the
    // constants a reader sizes its blocks from.
    let rendered = format!(
        "{:?}",
        reader.get("lattice.iq2_xxs").unwrap().attributes().unwrap()
    );
    assert!(rendered.contains("elems_per_block"), "{rendered}");
    assert!(rendered.contains("block_bytes"), "{rendered}");

    std::fs::remove_dir_all(&dir).ok();
}

/// Every scheme the bridge names gets its spelling into the file, and every
/// one it refuses gets none — checked over the whole enum rather than the four
/// a fixture happens to hold.
///
/// The exhaustive half is what makes this worth writing: `stamp_qnf` reads an
/// `Option` and the compiler cannot tell a refused row from a forgotten one,
/// so the only thing standing between "no signature yet" and "the stamp
/// silently stopped working" is asking every row and comparing against the
/// bridge's own answer.
#[test]
fn every_scheme_the_bridge_names_is_stamped_and_every_one_it_refuses_is_not() {
    let dir = tmpdir("stamp");
    // 256 elements of each scheme, and the bytes that takes. A blocked scheme
    // answers through `block_layout`; the rest store a plain array, and the
    // two FP8 rows are the ones whose LOGICAL type is the element itself —
    // `dense` plus that type is how the writer says "plain f8 values".
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
        // 256 elements is a whole super-block for every K-quant and a whole
        // number of blocks for the rest, so the payload is a legal extent
        // whatever the scheme.
        let stored = match spec.block_layout() {
            Some((elems, bytes)) => 256 / elems as usize * bytes as usize,
            // OCP MXFP4's element is half a byte and the writer says so with
            // the `f4_e2m1` logical type; everything else here is a byte.
            None if scheme == QuantScheme::Mxfp4E2M1E8M0 => 128,
            None => 256,
        };
        let payload = pattern(stored, 0x2b);
        let d = decl("w", vec![256], Encoding::Quant(spec.clone()));
        write_zt(
            &path,
            &BTreeMap::new(),
            &[WriteTensor {
                decl: &d,
                bytes: &payload,
            }],
        )
        .unwrap_or_else(|err| panic!("{scheme:?} could not be written: {err}"));

        let reader = ztensor::Source::open(&path).unwrap();
        let expected = spec.term().map(|term| term.mangle().as_str().to_string());
        assert_eq!(
            qnf_of(&reader, "w"),
            expected,
            "{scheme:?}: the file disagrees with the bridge"
        );
        // Whatever the stamp did, the file still reads back — the check that
        // would have caught the first draft of `stamp_qnf`, which dropped a
        // profile's own attributes for every scheme the bridge refused.
        zt::parse(&path)
            .unwrap_or_else(|err| panic!("{scheme:?} was stamped into unreadability: {err}"));
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// A plain dtype is stamped too, and with the leaf that names it.
///
/// `Dtype::repr` is total, so `dense` is the case where an absent attribute
/// can only mean the stamp did not run.
#[test]
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
        assert_eq!(reader.get("w").unwrap().layout(), "dense");
        assert_eq!(qnf_of(&reader, "w").as_deref(), Some(spelling));
        assert_eq!(dtype.repr().mangle().as_str(), spelling);
    }
    std::fs::remove_dir_all(&dir).ok();
}
