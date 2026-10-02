use std::path::Path;
use std::sync::Arc;

use ztensor::{Leaf, Source, Writer};

fn artifact_bytes(tag: &str) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("ztensor_memfs_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.zt");
    let mut writer = Writer::create(&path).unwrap();
    let a: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    writer.add("a.weight", vec![2, 2], Leaf::F32, &a).unwrap();
    writer
        .add(
            "b.weight",
            vec![16],
            Leaf::U8,
            &(0..16u8).collect::<Vec<_>>(),
        )
        .unwrap();
    writer.finish().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

#[test]
fn memfs_every_case() {
    a_mounted_path_opens_like_a_file();
    a_chunked_mount_opens_like_a_file();
    a_source_can_be_built_straight_from_bytes();
}

fn a_chunked_mount_opens_like_a_file() {
    let bytes = artifact_bytes("chunked");
    let fake = Path::new("/nowhere/on/this/machine/chunked.zt");
    let parts: Vec<Arc<[u8]>> = bytes.chunks(7).map(Arc::from).collect();
    ztensor::memfs::mount_chunks(fake, parts.clone());
    assert!(ztensor::memfs::get(fake).is_none(), "not one allocation");

    let source = Source::open(fake).unwrap();
    assert_eq!(source.len(), 2);
    let store = source.store(ztensor::StoreId(0));
    assert!(store.is_memory() && store.is_mapped());
    assert!(store.bytes().is_none());
    assert_eq!(store.len(), bytes.len() as u64);
    assert_eq!(store.read(0, store.len()).unwrap(), bytes);

    let b = source.tensor("b.weight").unwrap();
    assert!(!b.caps().map, "16 bytes across seven-byte pieces");
    assert!(b.caps().locate && b.caps().verify);
    let err = b.map().unwrap_err().to_string();
    assert!(err.contains("two chunks"), "{err}");
    assert_eq!(&*b.bytes().unwrap(), &(0..16u8).collect::<Vec<_>>()[..]);
    assert_eq!(b.verify().unwrap(), ztensor::Verified::Digest);
    assert!(source.tensor("a.weight").unwrap().verify().is_ok());
    assert!(ztensor::read::manifest_of(fake).unwrap().is_some());
    assert!(ztensor::read::canonical_violations(fake)
        .unwrap()
        .is_empty());

    let direct = Source::from_memory(fake, parts).unwrap();
    assert_eq!(
        &*direct.tensor("b.weight").unwrap().bytes().unwrap(),
        &(0..16u8).collect::<Vec<_>>()[..]
    );
    ztensor::memfs::unmount(fake);
}

fn a_mounted_path_opens_like_a_file() {
    let bytes = artifact_bytes("mounted");
    let fake = Path::new("/nowhere/on/this/machine/mounted.zt");
    assert!(
        Source::open(fake).is_err(),
        "nothing stands behind the name yet"
    );

    ztensor::memfs::mount(fake, Arc::from(bytes.as_slice()));
    let source = Source::open(fake).unwrap();
    assert!(source.provenance().as_root().is_some());
    assert_eq!(source.len(), 2);
    let store = source.store(ztensor::StoreId(0));
    assert!(store.is_memory());
    assert_eq!(store.path(), fake);
    assert_eq!(store.len(), bytes.len() as u64);

    let b = source.tensor("b.weight").unwrap();
    let caps = b.caps();
    assert!(caps.map && caps.locate && caps.verify);
    assert!(!caps.evict, "there is no page cache behind a mount");
    assert_eq!(b.map().unwrap(), &(0..16u8).collect::<Vec<_>>()[..]);
    assert_eq!(b.bytes().unwrap().as_ref(), b.map().unwrap());
    let at = b.locate().unwrap();
    assert_eq!(
        &bytes[at.offset as usize..(at.offset + at.len) as usize],
        b.map().unwrap(),
        "the address is an offset into the mounted bytes"
    );
    assert_eq!(b.verify().unwrap(), ztensor::Verified::Digest);
    assert!(b.prefetch().is_ok());
    assert!(b.evict().unwrap_err().to_string().contains("mount"));

    let indexed = Source::options().map(false).open(fake).unwrap();
    assert!(
        indexed.tensor("a.weight").unwrap().map().is_ok(),
        "memory is always mapped"
    );

    assert!(ztensor::read::manifest_of(fake).unwrap().is_some());
    assert!(ztensor::read::canonical_violations(fake)
        .unwrap()
        .is_empty());

    ztensor::memfs::unmount(fake);
    assert!(Source::open(fake).is_err());
    assert_eq!(source.tensor("b.weight").unwrap().map().unwrap().len(), 16);
}

fn a_source_can_be_built_straight_from_bytes() {
    let bytes = artifact_bytes("direct");
    let name = Path::new("/nowhere/on/this/machine/direct.zt");
    let source = Source::from_memory(name, Arc::from(bytes.as_slice())).unwrap();
    assert!(
        !ztensor::memfs::is_mounted(name),
        "from_memory mounts nothing"
    );
    assert_eq!(source.store(ztensor::StoreId(0)).path(), name);
    let a = source.tensor("a.weight").unwrap();
    assert_eq!(a.shape(), &[2, 2]);
    assert_eq!(a.map().unwrap().len(), 16);
    assert_eq!(a.verify().unwrap(), ztensor::Verified::Digest);

    let truncated = Source::from_memory(name, Arc::from(&bytes[..bytes.len() - 1]));
    assert!(truncated.is_err(), "the frame is checked like a file's");
}
