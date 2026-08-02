//! Reading real files through the zTensor checkpoint reader.
//!
//! The unit tests in `checkpoint::zt` translate a manifest built in memory.
//! These write actual files and read them back, which is what catches an
//! offset convention that is right on paper and wrong on disk: every tensor's
//! `(file_offset, span_bytes)` is checked by seeking there and comparing the
//! bytes against what was written.

use std::path::{Path, PathBuf};

use pie_loader::checkpoint::CheckpointMetadata;
use pie_loader::checkpoint::read::parse_checkpoint_metadata;
use pie_loader::checkpoint::zt::parse_checkpoint;
use pie_loader::types::{DType, Encoding};

fn tmpdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("zt_roundtrip_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Reads the bytes a tensor's plan coordinates point at.
fn bytes_at(metadata: &CheckpointMetadata, name: &str) -> Vec<u8> {
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

fn write_zt(path: &Path, tensors: &[(&str, Vec<u64>, ztensor::DType, Vec<u8>)]) {
    let mut writer = ztensor::Writer::create(path).unwrap();
    for (name, shape, dtype, data) in tensors {
        writer.add(*name, shape.to_vec(), *dtype, data).unwrap();
    }
    writer.finish().unwrap();
}

#[test]
fn zt_offsets_address_the_right_bytes() {
    let dir = tmpdir("offsets");
    let path = dir.join("model.zt");
    let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let b = f32_bytes(&[9.0; 8]);
    write_zt(
        &path,
        &[
            ("a.weight", vec![2, 2], ztensor::DType::F32, a.clone()),
            ("b.weight", vec![8], ztensor::DType::F32, b.clone()),
        ],
    );

    let metadata = parse_checkpoint(&path).unwrap();
    assert_eq!(metadata.tensors.len(), 2);
    assert_eq!(bytes_at(&metadata, "a.weight"), a);
    assert_eq!(bytes_at(&metadata, "b.weight"), b);

    let tensor = metadata.tensor_by_name("a.weight").unwrap();
    assert_eq!(tensor.shape, vec![2, 2]);
    assert_eq!(tensor.encoding, Encoding::Raw(DType::F32));
    // Canonical placement: every tensor starts on a 64 KiB boundary.
    assert_eq!(tensor.file_offset % 65536, 0);

    std::fs::remove_dir_all(&dir).ok();
}

/// The same tensors, written as safetensors and as `.zt`, must describe the
/// same model: same names, same shapes, same encodings, same bytes. Only the
/// offsets differ, which is the point of the exercise.
#[test]
fn safetensors_and_zt_agree_on_the_model() {
    let dir = tmpdir("agree");
    let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b: Vec<u8> = (0..16).collect();

    // safetensors, written by hand: header length, JSON, then the payloads.
    let header = format!(
        r#"{{"a.weight":{{"dtype":"F32","shape":[2,3],"data_offsets":[0,{}]}},"b.weight":{{"dtype":"U8","shape":[16],"data_offsets":[{},{}]}}}}"#,
        a.len(),
        a.len(),
        a.len() + b.len()
    );
    let mut st = (header.len() as u64).to_le_bytes().to_vec();
    st.extend_from_slice(header.as_bytes());
    st.extend_from_slice(&a);
    st.extend_from_slice(&b);
    std::fs::write(dir.join("model.safetensors"), &st).unwrap();

    let zt_path = dir.join("model.zt");
    write_zt(
        &zt_path,
        &[
            ("a.weight", vec![2, 3], ztensor::DType::F32, a.clone()),
            ("b.weight", vec![16], ztensor::DType::U8, b.clone()),
        ],
    );

    let from_st = parse_checkpoint_metadata(&dir).unwrap();
    let from_zt = parse_checkpoint(&zt_path).unwrap();

    let mut st_names: Vec<&str> = from_st.tensors.iter().map(|t| t.name.as_str()).collect();
    let mut zt_names: Vec<&str> = from_zt.tensors.iter().map(|t| t.name.as_str()).collect();
    st_names.sort();
    zt_names.sort();
    assert_eq!(st_names, zt_names);

    for name in ["a.weight", "b.weight"] {
        let s = from_st.tensor_by_name(name).unwrap();
        let z = from_zt.tensor_by_name(name).unwrap();
        assert_eq!(s.shape, z.shape, "{name}: shape");
        assert_eq!(s.encoding, z.encoding, "{name}: encoding");
        assert_eq!(s.span_bytes, z.span_bytes, "{name}: span");
        assert_eq!(
            bytes_at(&from_st, name),
            bytes_at(&from_zt, name),
            "{name}: bytes"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// The zTensor reader accepts the same safetensors file the native reader
/// does, and describes it identically — the property that lets one replace
/// the other.
#[test]
fn zt_reader_matches_the_native_safetensors_reader() {
    let dir = tmpdir("parity");
    let a = f32_bytes(&[0.5; 12]);
    let b: Vec<u8> = (0..64).map(|i| (i % 251) as u8).collect();
    let header = format!(
        r#"{{"x":{{"dtype":"F32","shape":[3,4],"data_offsets":[0,{}]}},"y":{{"dtype":"U8","shape":[64],"data_offsets":[{},{}]}}}}"#,
        a.len(),
        a.len(),
        a.len() + b.len()
    );
    let mut st = (header.len() as u64).to_le_bytes().to_vec();
    st.extend_from_slice(header.as_bytes());
    st.extend_from_slice(&a);
    st.extend_from_slice(&b);
    let path = dir.join("model.safetensors");
    std::fs::write(&path, &st).unwrap();

    let native = parse_checkpoint_metadata(&dir).unwrap();
    let through_zt = parse_checkpoint(&path).unwrap();

    assert_eq!(native.tensors.len(), through_zt.tensors.len());
    for tensor in &native.tensors {
        let other = through_zt
            .tensor_by_name(&tensor.name)
            .unwrap_or_else(|| panic!("{} missing from the zTensor read", tensor.name));
        assert_eq!(tensor.shape, other.shape, "{}: shape", tensor.name);
        assert_eq!(tensor.encoding, other.encoding, "{}: encoding", tensor.name);
        // The same file: the offsets must agree exactly, not merely address
        // equal bytes.
        assert_eq!(
            tensor.file_offset, other.file_offset,
            "{}: offset",
            tensor.name
        );
        assert_eq!(tensor.span_bytes, other.span_bytes, "{}: span", tensor.name);
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// A sharded `.zt` root: the manifest names shards, the tensors live in other
/// files, and the loader must end up with coordinates that address the right
/// bytes in the right file.
///
/// `parse_checkpoint` claims a root "brings its shards along". This is that
/// claim, checked: nothing else in the loader's tests opens a root whose
/// tensors are somewhere else, so the whole shard-resolution path — name to
/// file, store id to `FileId` — would otherwise be untested here.
#[test]
fn a_sharded_root_addresses_bytes_in_its_shards() {
    let dir = tmpdir("sharded");
    let payload = f32_bytes(&[1.5, 2.5, 3.5, 4.5]);
    let local = f32_bytes(&[7.0, 8.0]);

    // The positional convention: `model.zt` finds a shard named `00001` at
    // `model-00001.zt`, with no resolver configured by the loader.
    let shard_path = dir.join("model-00001.zt");
    write_zt(
        &shard_path,
        &[(
            "embed.weight",
            vec![2, 2],
            ztensor::DType::F32,
            payload.clone(),
        )],
    );
    let identity =
        ztensor::read::shard_identity(&shard_path, ztensor::DigestAlgorithm::Xxh3).unwrap();
    let shard_object = ztensor::read::manifest_of(&shard_path)
        .unwrap()
        .expect("the shard has a manifest")
        .object("embed.weight")
        .unwrap()
        .clone();

    let root = dir.join("model.zt");
    let mut writer = ztensor::Writer::options()
        .canonical(false)
        .create(&root)
        .unwrap();
    // One tensor of its own, so the root exercises both a local part and a
    // foreign one — the two spellings of a blob reference.
    writer
        .add("norm.weight", vec![2], ztensor::DType::F32, &local)
        .unwrap();
    writer.add_shard("00001", &identity).unwrap();
    writer.link("embed.weight", &shard_object, "00001").unwrap();
    writer.finish().unwrap();

    let metadata = parse_checkpoint(&root).unwrap();
    assert_eq!(metadata.files.len(), 2, "root plus one shard");
    assert_eq!(bytes_at(&metadata, "embed.weight"), payload);
    assert_eq!(bytes_at(&metadata, "norm.weight"), local);

    // The foreign tensor must be attributed to the shard, not the root: a
    // `FileId` that pointed at the root would still read *some* bytes, and
    // `bytes_at` above would be comparing them against the wrong file.
    let embed = metadata.tensor_by_name("embed.weight").unwrap();
    let norm = metadata.tensor_by_name("norm.weight").unwrap();
    assert_ne!(embed.file_id, norm.file_id);
    let file_of = |id| {
        metadata
            .files
            .iter()
            .find(|f| f.id == id)
            .map(|f| f.path.clone())
            .unwrap()
    };
    assert_eq!(file_of(embed.file_id), shard_path.display().to_string());
    assert_eq!(file_of(norm.file_id), root.display().to_string());
}

/// A `.zt` checkpoint reports itself as `.zt`.
///
/// It used to come back `Unknown` — the enum predated the zTensor reader and
/// never grew a variant for the loader's own container. This reads the format
/// off a real file rather than the mapping table, so it also covers a shard,
/// which is a `.zt` reached a different way.
#[test]
fn a_zt_checkpoint_says_it_is_zt() {
    use pie_loader::types::CheckpointFormat;

    let dir = tmpdir("format");
    let path = dir.join("model.zt");
    write_zt(
        &path,
        &[("w", vec![2], ztensor::DType::F32, f32_bytes(&[1.0, 2.0]))],
    );

    let metadata = parse_checkpoint(&path).unwrap();
    assert_eq!(metadata.files.len(), 1);
    assert_eq!(metadata.files[0].format, CheckpointFormat::Zt);
}
