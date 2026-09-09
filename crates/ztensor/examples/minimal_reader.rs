use std::env;
use std::fs;

use xxhash_rust::xxh3::xxh3_64;
use ztensor::format::cbor::{self, Value};
use ztensor::Term;

fn main() {
    let path = env::args().nth(1).expect("usage: minimal_reader <file.zt>");
    let buf = fs::read(&path).expect("read file");

    assert!(buf.len() >= 48, "not a .zt file: too small");

    let magic = [0x89, b'Z', b'T', b'2', 0x0d, 0x0a, 0x1a, 0x0a];
    assert_eq!(&buf[..8], &magic, "bad header magic");
    let footer = &buf[buf.len() - 40..];
    assert_eq!(&footer[32..40], &magic, "bad footer magic");
    let version = u32::from_le_bytes(footer[24..28].try_into().unwrap());
    assert_eq!(version, 3, "unsupported version");

    let m_off = u64::from_le_bytes(footer[0..8].try_into().unwrap()) as usize;
    let m_len = u64::from_le_bytes(footer[8..16].try_into().unwrap()) as usize;
    let m_hash = u64::from_le_bytes(footer[16..24].try_into().unwrap());
    if m_len == 0 {
        println!("{path}: data shard (no manifest)");
        return;
    }

    let manifest_bytes = &buf[m_off..m_off + m_len];
    assert_eq!(xxh3_64(manifest_bytes), m_hash, "manifest hash mismatch");

    let root = cbor::decode(manifest_bytes).expect("manifest CBOR");

    let objects = map_get(&root, "objects").expect("manifest missing 'objects'");
    for (key, obj) in objects.as_map().unwrap() {
        let name = key.as_text().unwrap();
        let shape: Vec<u64> = map_get(obj, "shape")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default();
        let ty = map_get(obj, "type").and_then(Value::as_text);
        let layout = map_get(obj, "layout").and_then(Value::as_text);
        let blob = map_get(obj, "blob").expect("object missing 'blob'");
        let offset = map_get(blob, "offset").and_then(Value::as_u64).unwrap();
        let length = map_get(blob, "length").and_then(Value::as_u64).unwrap();
        println!(
            "{name}: {} {shape:?} at {offset}+{length}",
            ty.unwrap_or("(no type)"),
        );

        match layout {
            Some(layout) => println!("  layout {layout}"),
            None => {
                let term = Term::parse(ty.expect("an object without a layout has a type"))
                    .expect("well-formed type");
                for p in term.planes(&shape).expect("shape fits the type") {
                    println!(
                        "  {} {} {:?} at +{} ({} bytes)",
                        p.path, p.leaf, p.shape, p.offset, p.len
                    );
                }
            }
        }
    }
}

fn map_get<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.as_map()?
        .iter()
        .find(|(k, _)| k.as_text() == Some(key))
        .map(|(_, val)| val)
}
