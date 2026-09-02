//! Reading one source from several threads.
//!
//! A `Source` is `Send + Sync`, which is what lets a loader fan a checkpoint
//! out across threads, the ordinary case for anything feeding a GPU. The part
//! that has to be right for that claim is the opaque readers: a deflated zip
//! entry has no address, so producing it means holding a lock over an archive
//! and, for torch, over a cache of inflated storages.
//!
//! These tests are about contention on exactly those locks. What they check is
//! that concurrent readers get the same bytes a single reader would, and that
//! they finish. A deadlock here would look like a hung loader, which is the
//! failure mode nobody can debug from a stack trace.

#![cfg(feature = "npz")]

use std::borrow::Cow;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn npy(shape: &str, data: &[u8]) -> Vec<u8> {
    let dict = format!("{{'descr': '|u1', 'fortran_order': False, 'shape': {shape}, }}");
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend((dict.len() as u16).to_le_bytes());
    out.extend(dict.as_bytes());
    out.extend(data);
    out
}

/// An `.npz` whose entries are deflated, so every read goes through the
/// archive lock rather than the mapping.
fn deflated_npz(name: &str, tensors: &[(&str, u8, usize)]) -> PathBuf {
    let path = tmp(name);
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (entry, byte, len) in tensors {
        zip.start_file(format!("{entry}.npy"), options).unwrap();
        zip.write_all(&npy(&format!("({len},)"), &vec![*byte; *len]))
            .unwrap();
    }
    zip.finish().unwrap();
    path
}

/// Every thread reads every tensor, starting together.
///
/// The barrier is the point: without it the threads would arrive at the lock
/// one after another and never contend.
#[test]
fn threads_contending_on_an_opaque_reader_all_get_the_right_bytes() {
    let tensors: Vec<(&str, u8, usize)> = vec![
        ("a", 0xA1, 4096),
        ("b", 0xB2, 8192),
        ("c", 0xC3, 1024),
        ("d", 0xD4, 16384),
    ];
    let path = deflated_npz("threads-npz.zt", &tensors);
    let src = Arc::new(ztensor_compat::open(&path).unwrap());

    // Nothing here is mappable, so this exercises the lock.
    for (name, _, _) in &tensors {
        let caps = src.tensor(name).unwrap().caps();
        assert!(!caps.map && !caps.locate, "{name} should be opaque");
    }

    const THREADS: usize = 8;
    const ROUNDS: usize = 12;
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();
    for _ in 0..THREADS {
        let src = Arc::clone(&src);
        let barrier = Arc::clone(&barrier);
        let expected = tensors.clone();
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            for _ in 0..ROUNDS {
                for (name, byte, len) in &expected {
                    let bytes = src.tensor(name).unwrap().bytes().unwrap();
                    assert_eq!(bytes.len(), *len, "{name}: wrong length");
                    assert!(bytes.iter().all(|b| b == byte), "{name}: wrong content");
                    assert!(
                        matches!(bytes, Cow::Owned(_)),
                        "{name}: a deflated entry cannot be mapped"
                    );
                }
            }
        }));
    }
    for handle in handles {
        handle.join().expect("a reader thread panicked or hung");
    }
}

/// The same tensor from every thread at once: one entry, one lock, maximum
/// contention on it.
#[test]
fn one_hot_tensor_read_by_everyone_at_once() {
    let path = deflated_npz("threads-hot.zt", &[("hot", 0x5A, 65536)]);
    let src = Arc::new(ztensor_compat::open(&path).unwrap());

    const THREADS: usize = 16;
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();
    for _ in 0..THREADS {
        let src = Arc::clone(&src);
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            let bytes = src.tensor("hot").unwrap().bytes().unwrap();
            assert_eq!(bytes.len(), 65536);
            assert!(bytes.iter().all(|&b| b == 0x5A));
        }));
    }
    for handle in handles {
        handle.join().expect("a reader thread panicked or hung");
    }
}

/// Mapped and opaque tensors read side by side.
///
/// A mixed file is the realistic shape, a checkpoint where some entries were
/// stored and some deflated. The two paths share nothing but the source,
/// so this is where a mistake in that sharing would show.
#[test]
fn mapped_and_opaque_tensors_are_read_side_by_side() {
    let path = tmp("threads-mixed.zt");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    let deflated = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zip.start_file("plain.npy", stored).unwrap();
    zip.write_all(&npy("(2048,)", &vec![0x11; 2048])).unwrap();
    zip.start_file("packed.npy", deflated).unwrap();
    zip.write_all(&npy("(2048,)", &vec![0x22; 2048])).unwrap();
    zip.finish().unwrap();

    let src = Arc::new(ztensor_compat::open(&path).unwrap());
    assert!(src.tensor("plain").unwrap().caps().map);
    assert!(!src.tensor("packed").unwrap().caps().map);

    const THREADS: usize = 8;
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();
    for i in 0..THREADS {
        let src = Arc::clone(&src);
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            for _ in 0..16 {
                if i % 2 == 0 {
                    let mapped = src.tensor("plain").unwrap().map().unwrap();
                    assert!(mapped.iter().all(|&b| b == 0x11));
                } else {
                    let bytes = src
                        .tensor("packed")
                        .unwrap()
                        .bytes()
                        .unwrap();
                    assert!(bytes.iter().all(|&b| b == 0x22));
                }
            }
        }));
    }
    for handle in handles {
        handle.join().expect("a reader thread panicked or hung");
    }
}

/// A source built on one thread and used on another, with no `Arc` at all.
/// the `Send` half of the claim, which sharing alone does not exercise.
#[test]
fn a_source_can_be_moved_to_another_thread() {
    let path = deflated_npz("threads-moved.zt", &[("w", 0x77, 4096)]);
    let src = ztensor_compat::open(&path).unwrap();
    let read = std::thread::spawn(move || {
        let bytes = src
            .tensor("w")
            .unwrap()
            .bytes()
            .unwrap()
            .into_owned();
        (bytes.len(), bytes.iter().all(|&b| b == 0x77))
    })
    .join()
    .unwrap();
    assert_eq!(read, (4096, true));
}
