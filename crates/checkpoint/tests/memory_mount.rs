use std::path::{Path, PathBuf};
use std::sync::Arc;

use checkpoint::contract::{Expr, ModelContract, TensorContract};
use checkpoint::executor::Execution;
use checkpoint::file::read::{parse_metadata, read_meta, verify_declared_files};
use checkpoint::file::zt;
use checkpoint::plan::StorageTarget;
use checkpoint::types::{DType, Encoding};

fn tmpdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pie_memory_mount_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

const DESCRIPTOR: &[u8] = b"{\"sku\":\"probe\"}";

fn write_zt(path: &Path, wide: usize) -> (Vec<u8>, Vec<u8>) {
    let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f32_bytes(&[9.0; 8]);
    let mut writer = ztensor::Writer::create(path).unwrap();
    writer
        .add(
            "__meta__/model/descriptor",
            vec![DESCRIPTOR.len() as u64],
            ztensor::Leaf::U8,
            DESCRIPTOR,
        )
        .unwrap();
    writer
        .add("a.weight", vec![2, 3], ztensor::Leaf::F32, &a)
        .unwrap();
    writer
        .add("b.weight", vec![8], ztensor::Leaf::F32, &b)
        .unwrap();
    for index in 0..wide {
        let values: Vec<f32> = (0..3000).map(|at| (index * 3000 + at) as f32).collect();
        writer
            .add(
                wide_name(index),
                vec![3000],
                ztensor::Leaf::F32,
                &f32_bytes(&values),
            )
            .unwrap();
    }
    writer.finish().unwrap();
    (a, b)
}

fn wide_name(index: usize) -> String {
    format!("layers.{index:02}.weight")
}

fn contract(wide: usize) -> ModelContract {
    let mut tensors = vec![
        TensorContract::new(
            "a.weight",
            Expr::src("a.weight"),
            vec![2, 3],
            Encoding::Raw(DType::F32),
        ),
        TensorContract::new(
            "b.weight",
            Expr::src("b.weight"),
            vec![8],
            Encoding::Raw(DType::F32),
        ),
    ];
    for index in 0..wide {
        let name = wide_name(index);
        tensors.push(TensorContract::new(
            &name,
            Expr::src(&name),
            vec![3000],
            Encoding::Raw(DType::F32),
        ));
    }
    ModelContract {
        groups: Vec::new(),
        alignment: 1,
        tensors,
    }
}

#[test]
fn memory_mount_every_case() {
    a_single_chunk_mount_reads_like_the_file();
    a_chunked_mount_reads_like_the_file();
    a_lazy_mount_reads_like_the_file();
}

fn a_lazy_mount_reads_like_the_file() {
    const WIDE: usize = 12;
    let dir = tmpdir("lazy");
    let real = dir.join("model.zt");
    let (a, b) = write_zt(&real, WIDE);
    let bytes = std::fs::read(&real).unwrap();
    let fake = Path::new("/nowhere/on/this/machine/lazy.zt");

    let from_file = parse_metadata(&real).unwrap();
    let plan_file =
        checkpoint::plan::compile(&from_file, &contract(WIDE), StorageTarget::default()).unwrap();
    let on_disk = Execution::new(&plan_file, &dir).run().unwrap();
    assert_eq!(on_disk.tensors.len(), 2 + WIDE);

    for window in [ztensor::memfs::LAZY_WINDOW, 4096] {
        let requests = Arc::new(std::sync::Mutex::new(Vec::<(u64, u64)>::new()));
        let log = Arc::clone(&requests);
        let source = real.clone();
        let fetch = Box::new(move |offset: u64, into: &mut [u8]| {
            use std::io::{Read, Seek, SeekFrom};
            log.lock().unwrap().push((offset, into.len() as u64));
            let mut file = std::fs::File::open(&source).map_err(|e| e.to_string())?;
            file.seek(SeekFrom::Start(offset))
                .map_err(|e| e.to_string())?;
            file.read_exact(into).map_err(|e| e.to_string())
        });
        ztensor::memfs::mount_lazy_windowed(fake, bytes.len() as u64, window, fetch);
        assert!(ztensor::memfs::get(fake).is_none(), "nothing is held");
        assert_eq!(ztensor::memfs::len(fake), Some(bytes.len() as u64));

        let source = ztensor::Source::open(fake).unwrap();
        assert!(source.store(ztensor::StoreId(0)).is_memory());
        assert!(source.store(ztensor::StoreId(0)).bytes().is_none());
        for tensor in source.tensors() {
            assert!(!tensor.caps().map, "{}", tensor.name());
            let at = tensor.locate().unwrap();
            let expect = &bytes[at.offset as usize..(at.offset + at.len) as usize];
            assert_eq!(&*tensor.bytes().unwrap(), expect, "{}", tensor.name());
            assert!(tensor.map().is_err());
            assert_eq!(tensor.verify().unwrap(), ztensor::Verified::Digest);
        }

        let from_mount = parse_metadata(fake).unwrap();
        assert_eq!(from_mount.files.len(), 1);
        assert_eq!(from_mount.files[0].path, fake.display().to_string());
        assert_eq!(from_mount.files[0].size_bytes, bytes.len() as u64);
        assert_eq!(from_mount.tensors, from_file.tensors);
        assert_eq!(
            read_meta(&from_mount, "model/descriptor").unwrap(),
            read_meta(&from_file, "model/descriptor").unwrap()
        );
        assert_eq!(
            read_meta(&from_mount, "model/descriptor")
                .unwrap()
                .as_deref(),
            Some(DESCRIPTOR)
        );
        assert_eq!(zt::verify(fake).unwrap(), 3 + WIDE);
        assert_eq!(
            zt::artifact_identity(fake).unwrap(),
            zt::artifact_identity(&real).unwrap()
        );
        assert_eq!(
            zt::read_attributes(fake).unwrap(),
            zt::read_attributes(&real).unwrap()
        );

        let plan =
            checkpoint::plan::compile(&from_mount, &contract(WIDE), StorageTarget::default())
                .unwrap();
        verify_declared_files(&plan, Path::new("/")).unwrap();
        let before = ztensor::memfs::lazy_stats(fake).unwrap();
        let mounted = Execution::new(&plan, Path::new("/")).run().unwrap();
        let after = ztensor::memfs::lazy_stats(fake).unwrap();
        assert_eq!(mounted.tensors["a.weight"], a);
        assert_eq!(mounted.tensors["b.weight"], b);
        assert_eq!(on_disk.tensors, mounted.tensors);
        assert_eq!(on_disk.arena, mounted.arena);

        let served = requests.lock().unwrap().clone();
        assert_eq!(served.len() as u64, after.requests);
        assert_eq!(served.iter().map(|(_, n)| n).sum::<u64>(), after.bytes);
        for (offset, len) in &served {
            assert_eq!(offset % window, 0, "fetches are window-aligned");
            assert!(offset + len <= bytes.len() as u64);
        }
        if window == ztensor::memfs::LAZY_WINDOW {
            assert_eq!(
                served,
                vec![(0, bytes.len() as u64)],
                "one window holds the whole artifact"
            );
            assert_eq!(after, before, "the execution read only the cache");
        } else {
            assert!(after.requests > before.requests);
            assert!(served.iter().any(|(_, n)| *n > window), "{served:?}");
            let held = ztensor::memfs::chunks(fake).unwrap().cached_bytes();
            let most = (ztensor::memfs::LAZY_KEEP as u64 + ztensor::memfs::LAZY_PIN_TAIL) * window;
            assert!(held <= most, "{held} bytes cached");
        }

        ztensor::memfs::mount_lazy_windowed(
            fake,
            bytes.len() as u64,
            window,
            Box::new(|offset: u64, _: &mut [u8]| Err(format!("the page is gone at {offset}"))),
        );
        let err = Execution::new(&plan, Path::new("/")).run().unwrap_err();
        assert!(err.to_string().contains("the page is gone at"), "{err}");
        assert!(parse_metadata(fake).is_err());
        ztensor::memfs::unmount(fake);
    }
    assert!(parse_metadata(fake).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

fn a_single_chunk_mount_reads_like_the_file() {
    let dir = tmpdir("all");
    let real = dir.join("model.zt");
    let (a, b) = write_zt(&real, 0);
    let bytes: Arc<[u8]> = Arc::from(std::fs::read(&real).unwrap());
    let fake = Path::new("/nowhere/on/this/machine/model.zt");
    ztensor::memfs::mount(fake, bytes.clone());

    let from_file = parse_metadata(&real).unwrap();
    let from_mount = parse_metadata(fake).unwrap();
    assert_eq!(from_mount.files.len(), 1);
    assert_eq!(from_mount.files[0].path, fake.display().to_string());
    assert_eq!(from_mount.files[0].size_bytes, bytes.len() as u64);
    assert_eq!(from_mount.tensors, from_file.tensors);
    assert_eq!(
        read_meta(&from_mount, "model/descriptor")
            .unwrap()
            .as_deref(),
        Some(DESCRIPTOR)
    );
    assert_eq!(zt::verify(fake).unwrap(), 3);
    assert_eq!(
        zt::artifact_identity(fake).unwrap(),
        zt::artifact_identity(&real).unwrap()
    );
    assert_eq!(
        zt::read_attributes(fake).unwrap(),
        zt::read_attributes(&real).unwrap()
    );

    let plan =
        checkpoint::plan::compile(&from_mount, &contract(0), StorageTarget::default()).unwrap();
    verify_declared_files(&plan, Path::new("/")).unwrap();
    let mounted = Execution::new(&plan, Path::new("/")).run().unwrap();
    assert_eq!(mounted.tensors["a.weight"], a);
    assert_eq!(mounted.tensors["b.weight"], b);

    let plan_file =
        checkpoint::plan::compile(&from_file, &contract(0), StorageTarget::default()).unwrap();
    let on_disk = Execution::new(&plan_file, &dir).run().unwrap();
    assert_eq!(on_disk.tensors, mounted.tensors);
    assert_eq!(on_disk.arena, mounted.arena);

    ztensor::memfs::mount(fake, Arc::from(&bytes[..bytes.len() - 8]));
    assert!(verify_declared_files(&plan, Path::new("/")).is_err());

    ztensor::memfs::unmount(fake);
    assert!(parse_metadata(fake).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

fn a_chunked_mount_reads_like_the_file() {
    const WIDE: usize = 12;
    const CHUNK: usize = 4096;
    let dir = tmpdir("chunked");
    let real = dir.join("model.zt");
    let (a, b) = write_zt(&real, WIDE);
    let bytes = std::fs::read(&real).unwrap();
    let parts: Vec<Arc<[u8]>> = bytes.chunks(CHUNK).map(Arc::from).collect();
    assert!(parts.len() > WIDE, "{} chunks", parts.len());
    let fake = Path::new("/nowhere/on/this/machine/chunked.zt");
    ztensor::memfs::mount_chunks(fake, parts);
    assert!(ztensor::memfs::get(fake).is_none());
    assert_eq!(ztensor::memfs::len(fake), Some(bytes.len() as u64));

    let source = ztensor::Source::open(fake).unwrap();
    let straddling = source.tensors().filter(|tensor| !tensor.caps().map).count();
    assert!(
        straddling >= WIDE / 2,
        "{straddling} tensors straddle a seam"
    );
    for tensor in source.tensors() {
        let at = tensor.locate().unwrap();
        let expect = &bytes[at.offset as usize..(at.offset + at.len) as usize];
        assert_eq!(&*tensor.bytes().unwrap(), expect, "{}", tensor.name());
        assert_eq!(tensor.map().is_ok(), tensor.caps().map, "{}", tensor.name());
        assert_eq!(tensor.verify().unwrap(), ztensor::Verified::Digest);
    }

    let from_file = parse_metadata(&real).unwrap();
    let from_mount = parse_metadata(fake).unwrap();
    assert_eq!(from_mount.files.len(), 1);
    assert_eq!(from_mount.files[0].path, fake.display().to_string());
    assert_eq!(from_mount.files[0].size_bytes, bytes.len() as u64);
    assert_eq!(from_mount.tensors, from_file.tensors);
    assert_eq!(
        read_meta(&from_mount, "model/descriptor").unwrap(),
        read_meta(&from_file, "model/descriptor").unwrap()
    );
    assert_eq!(
        read_meta(&from_mount, "model/descriptor")
            .unwrap()
            .as_deref(),
        Some(DESCRIPTOR)
    );
    assert_eq!(zt::verify(fake).unwrap(), zt::verify(&real).unwrap());
    assert_eq!(zt::verify(fake).unwrap(), 3 + WIDE);
    assert_eq!(
        zt::artifact_identity(fake).unwrap(),
        zt::artifact_identity(&real).unwrap()
    );
    assert_eq!(
        zt::read_attributes(fake).unwrap(),
        zt::read_attributes(&real).unwrap()
    );

    let plan =
        checkpoint::plan::compile(&from_mount, &contract(WIDE), StorageTarget::default()).unwrap();
    verify_declared_files(&plan, Path::new("/")).unwrap();
    let view = checkpoint::verify::PlanView {
        files: vec![checkpoint::verify::FileView {
            id: 0,
            path: &from_mount.files[0].path,
            size_bytes: bytes.len() as u64,
        }],
        sources: Vec::new(),
        tensors: Vec::new(),
        instr_count: 0,
        schedule: Vec::new(),
        finalized: Vec::new(),
        reads: Vec::new(),
    };
    assert!(checkpoint::verify::verify(&view, None).is_ok());
    let mounted = Execution::new(&plan, Path::new("/")).run().unwrap();
    assert_eq!(mounted.tensors["a.weight"], a);
    assert_eq!(mounted.tensors["b.weight"], b);
    let plan_file =
        checkpoint::plan::compile(&from_file, &contract(WIDE), StorageTarget::default()).unwrap();
    let on_disk = Execution::new(&plan_file, &dir).run().unwrap();
    assert_eq!(on_disk.tensors.len(), 2 + WIDE);
    assert_eq!(on_disk.tensors, mounted.tensors);
    assert_eq!(on_disk.arena, mounted.arena);

    let short: Vec<Arc<[u8]>> = bytes[..bytes.len() - 8]
        .chunks(CHUNK)
        .map(Arc::from)
        .collect();
    ztensor::memfs::mount_chunks(fake, short);
    assert!(verify_declared_files(&plan, Path::new("/")).is_err());
    let tail = bytes[..bytes.len() / 2]
        .chunks(CHUNK)
        .map(Arc::from)
        .collect();
    ztensor::memfs::mount_chunks(fake, tail);
    let err = Execution::new(&plan, Path::new("/")).run().unwrap_err();
    assert!(err.to_string().contains("mounted bytes"), "{err}");

    ztensor::memfs::unmount(fake);
    assert!(parse_metadata(fake).is_err());
    std::fs::remove_dir_all(&dir).ok();
}
