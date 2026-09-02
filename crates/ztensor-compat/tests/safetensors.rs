//! safetensors projection: strict open, honest caps, and the conversion
//! path to canonical `.zt`.

use std::fs;
use std::path::PathBuf;
use ztensor::{Error, Leaf, Term, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

/// Builds a safetensors file. Offsets are assigned in the given tensor
/// order; `meta` becomes `__metadata__`.
fn st_bytes(tensors: &[(&str, &str, &[u64], &[u8])], meta: &[(&str, &str)]) -> Vec<u8> {
    let mut entries = Vec::new();
    if !meta.is_empty() {
        let kv: Vec<String> = meta
            .iter()
            .map(|(k, v)| format!("\"{k}\":\"{v}\""))
            .collect();
        entries.push(format!("\"__metadata__\":{{{}}}", kv.join(",")));
    }
    let mut cursor = 0usize;
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let dims: Vec<String> = shape.iter().map(u64::to_string).collect();
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            dims.join(","),
            cursor,
            cursor + bytes.len()
        ));
        cursor += bytes.len();
        data.extend_from_slice(bytes);
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

fn st_file(name: &str, tensors: &[(&str, &str, &[u64], &[u8])], meta: &[(&str, &str)]) -> PathBuf {
    let path = tmp(name);
    fs::write(&path, st_bytes(tensors, meta)).unwrap();
    path
}

fn f32s(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn open_and_read() {
    let a = f32s(&[1.0, 2.0, 3.0, 4.0]);
    let b = vec![7u8; 8]; // 4 bf16 elements
    let path = st_file(
        "basic.safetensors",
        &[
            ("a.weight", "F32", &[2, 2], &a),
            ("b.weight", "BF16", &[4], &b),
        ],
        &[("format", "pt")],
    );

    let st = ztensor_compat::open(&path).unwrap();
    assert_eq!(st.len(), 2);
    let obj = st.tensor("a.weight").unwrap();
    assert_eq!(obj.shape().to_vec(), vec![2, 2]);
    assert_eq!(obj.term(), Some(&Term::Leaf(Leaf::F32)));
    assert!(st.attributes().is_some());

    assert_eq!(
        st.tensor("a.weight")
            .unwrap()
            .bytes()
            .unwrap()
            .into_owned(),
        a
    );
    assert_eq!(
        st.tensor("b.weight")
            .unwrap()
            .map()
            .unwrap(),
        &b[..]
    );

    let caps = st.tensor("a.weight").unwrap().caps();
    assert!(caps.map);
    assert!(!caps.verify);
    assert!(caps.map);
}

#[test]
fn dtype_projections() {
    let path = st_file(
        "dtypes.safetensors",
        &[
            ("mask", "BOOL", &[4], &[0, 1, 0, 1]),
            ("fp8", "F8_E4M3", &[4], &[1, 2, 3, 4]),
            ("fp4", "F4_E2M1", &[4], &[0x21, 0x43]), // packed two per byte
        ],
        &[],
    );
    let st = ztensor_compat::open(&path).unwrap();
    let leaf = |name: &str| st.tensor(name).unwrap().term().and_then(Term::leaf);
    assert_eq!(leaf("mask"), Some(Leaf::Bool));
    assert_eq!(leaf("fp8"), Some(Leaf::E4M3));
    assert_eq!(leaf("fp4"), Some(Leaf::E2M1));
    assert_eq!(st.tensor("fp4").unwrap().nbytes(), 2);
}

#[test]
fn unknown_dtype_refused() {
    let path = st_file("f4.safetensors", &[("t", "F4", &[2], &[0x21])], &[]);
    assert!(matches!(
        ztensor_compat::open(&path),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn rejects_bad_geometry() {
    // size mismatch: F32 [2,2] needs 16 bytes
    let path = st_file(
        "short.safetensors",
        &[("t", "F32", &[2, 2], &[0u8; 12])],
        &[],
    );
    assert!(ztensor_compat::open(&path).is_err());

    // overlap / hole: hand-build offsets that don't tile
    let mut bytes = st_bytes(&[("a", "U8", &[8], &[1u8; 8])], &[]);
    // corrupt data_offsets [0,8] -> [0,4]: shape mismatch aside, the data
    // section now has a trailing hole
    let needle = b"[0,8]";
    let pos = bytes.windows(5).position(|w| w == needle).unwrap();
    bytes[pos..pos + 5].copy_from_slice(b"[0,4]");
    let path = tmp("hole.safetensors");
    fs::write(&path, &bytes).unwrap();
    assert!(ztensor_compat::open(&path).is_err());

    // truncated header
    let path = st_file("trunc.safetensors", &[("t", "U8", &[4], &[9u8; 4])], &[]);
    let bytes = fs::read(&path).unwrap();
    fs::write(&path, &bytes[..9]).unwrap();
    assert!(ztensor_compat::open(&path).is_err());
}

/// The conversion path: HF checkpoint in, canonical tier-3 `.zt` out, with
/// bit-reproducibly.
#[test]
fn convert_to_canonical_zt() {
    let a = f32s(&[1.0, 2.0, 3.0, 4.0]);
    let b = vec![3u8; 8];
    let st_path = st_file(
        "convert.safetensors",
        &[
            ("b.weight", "BF16", &[4], &b),
            ("a.weight", "F32", &[2, 2], &a),
        ],
        &[("format", "pt")],
    );
    let st = ztensor_compat::open(&st_path).unwrap();

    let convert = |out: &PathBuf| {
        let mut w = Writer::create(out).unwrap();
        w.ingest(&st).unwrap();
        w.finish().unwrap();
    };
    let zt1 = tmp("converted1.zt");
    let zt2 = tmp("converted2.zt");
    convert(&zt1);
    convert(&zt2);

    // Bit-reproducible: same source, identical canonical output.
    assert_eq!(fs::read(&zt1).unwrap(), fs::read(&zt2).unwrap());

    let r = ztensor::Source::open(&zt1).unwrap();
    assert_eq!(
        r.tensor("a.weight")
            .unwrap()
            .bytes()
            .unwrap()
            .into_owned(),
        a
    );
    assert_eq!(
        r.tensor("b.weight")
            .unwrap()
            .bytes()
            .unwrap()
            .into_owned(),
        b
    );
    assert!(r.tensor("a.weight").unwrap().verify().unwrap().is_checked()); // digests added
    assert!(r.attributes().is_some()); // metadata carried over

    // Everything the projection could not offer, the conversion added: a
    // digest to verify against, and (on <=64K page hosts) pages of its own.
    let caps = r.tensor("a.weight").unwrap().caps();
    assert!(caps.verify && caps.map && caps.locate);
    if ztensor::provide::page_size() <= ztensor::format::ALIGN_CANONICAL {
        assert!(caps.evict);
    }
}
