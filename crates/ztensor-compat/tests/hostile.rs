//! Regression tests for the adversarial-robustness review.
//!
//! Every case here was a verified panic, hang, or silent-wrong-data path.
//! The contract these lock in: hostile input yields `Err`, never a crash,
//! never an unbounded allocation, never a fabricated tensor.

use std::fs;
use std::path::PathBuf;

fn tmp(name: &str) -> PathBuf {
    let p = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_file(&p);
    p
}

// =======================================================================
// hdf5
// =======================================================================

#[cfg(feature = "hdf5")]
mod hdf5 {
    use super::*;

    /// Superblock v0 with caller-chosen root B-tree and heap addresses.
    fn superblock(btree: u64, heap: u64, len: usize) -> Vec<u8> {
        let mut b = vec![0u8; len.max(96)];
        b[0..8].copy_from_slice(b"\x89HDF\r\n\x1a\n");
        b[13] = 8; // offset size
        b[14] = 8; // length size
        b[16..18].copy_from_slice(&4u16.to_le_bytes());
        b[18..20].copy_from_slice(&16u16.to_le_bytes());
        let eof = b.len() as u64;
        b[40..48].copy_from_slice(&eof.to_le_bytes());
        b[72..76].copy_from_slice(&1u32.to_le_bytes()); // root cache type: group
        b[80..88].copy_from_slice(&btree.to_le_bytes());
        b[88..96].copy_from_slice(&heap.to_le_bytes());
        b
    }

    /// C1: heap address `u64::MAX`, where `pos + 4` wrapped past the guard.
    #[test]
    fn heap_address_wraparound() {
        let path = tmp("c1.h5");
        fs::write(&path, superblock(96, u64::MAX, 96)).unwrap();
        assert!(ztensor_compat::open(&path).is_err());
    }

    /// C2: near-`u64::MAX` addresses in every signature check.
    #[test]
    fn btree_address_wraparound() {
        for addr in [u64::MAX - 8, u64::MAX - 1, 1 << 62] {
            let path = tmp("c2.h5");
            fs::write(&path, superblock(addr, 96, 256)).unwrap();
            assert!(ztensor_compat::open(&path).is_err(), "addr {addr}");
        }
    }

    /// C3: heap data segment / link-name offsets far past EOF.
    #[test]
    fn heap_data_offset_out_of_range() {
        let mut b = superblock(96, 144, 416);
        // group B-tree @96 with one SNOD @192
        b[96..100].copy_from_slice(b"TREE");
        b[102..104].copy_from_slice(&1u16.to_le_bytes());
        b[104..112].copy_from_slice(&[0xff; 8]);
        b[112..120].copy_from_slice(&[0xff; 8]);
        b[128..136].copy_from_slice(&192u64.to_le_bytes());
        // local heap @144 whose data segment address is 2^40
        b[144..148].copy_from_slice(b"HEAP");
        b[152..160].copy_from_slice(&16u64.to_le_bytes());
        b[168..176].copy_from_slice(&(1u64 << 40).to_le_bytes());
        // SNOD @192, one symbol
        b[192..196].copy_from_slice(b"SNOD");
        b[196] = 1;
        b[198..200].copy_from_slice(&1u16.to_le_bytes());
        b[200..208].copy_from_slice(&8u64.to_le_bytes());
        b[208..216].copy_from_slice(&248u64.to_le_bytes());
        let path = tmp("c3.h5");
        fs::write(&path, &b).unwrap();
        assert!(ztensor_compat::open(&path).is_err());
    }
}

// =======================================================================
// gguf
// =======================================================================

#[cfg(feature = "gguf")]
mod gguf {
    use super::*;

    fn gstr(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    /// C8: alignment rounds the data section past EOF; a zero-length
    /// tensor then produced an out-of-bounds blob that panicked in view().
    #[test]
    fn data_section_past_eof() {
        let mut b = Vec::new();
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend(1u64.to_le_bytes()); // tensors
        b.extend(0u64.to_le_bytes()); // kvs
        gstr(&mut b, "t");
        b.extend(1u32.to_le_bytes()); // ndims
        b.extend(0u64.to_le_bytes()); // shape [0]
        b.extend(0u32.to_le_bytes()); // F32
        b.extend(0u64.to_le_bytes()); // offset
                                      // file ends here: 32-byte alignment rounds data_start past EOF
        let path = tmp("c8.gguf");
        fs::write(&path, &b).unwrap();
        match ztensor_compat::open(&path) {
            Err(_) => {}
            Ok(g) => {
                // If it opens, the blob must at least be in bounds.
                let _ = g
                    .tensor("t")
                    .unwrap()
                    .bytes()
                    .expect("in-bounds read");
            }
        }
    }

    /// LOW: a huge declared tensor/KV count must not drive allocation.
    #[test]
    fn lying_counts_do_not_allocate() {
        let mut b = Vec::new();
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend(100_000u64.to_le_bytes()); // tensor count, file is ~30 B
        b.extend(100_000u64.to_le_bytes()); // kv count
        let path = tmp("counts.gguf");
        fs::write(&path, &b).unwrap();
        assert!(ztensor_compat::open(&path).is_err());
    }
}

// =======================================================================
// npz
// =======================================================================

#[cfg(feature = "npz")]
mod npz {
    use super::*;
    use std::io::Write;

    fn npy(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
        let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend((dict.len() as u16).to_le_bytes());
        out.extend(dict.as_bytes());
        out.extend(data);
        out
    }

    fn write_npz(name: &str, entries: &[(&str, Vec<u8>, bool)]) -> PathBuf {
        let path = tmp(name);
        let mut z = zip::ZipWriter::new(fs::File::create(&path).unwrap());
        for (entry, bytes, compress) in entries {
            let method = if *compress {
                zip::CompressionMethod::Deflated
            } else {
                zip::CompressionMethod::Stored
            };
            let opts = zip::write::SimpleFileOptions::default().compression_method(method);
            z.start_file(format!("{entry}.npy"), opts).unwrap();
            z.write_all(bytes).unwrap();
        }
        z.finish().unwrap();
        path
    }

    /// C9: `')'` before `'('` in the shape tuple sliced backwards.
    #[test]
    fn reversed_shape_parens() {
        let path = write_npz("c9.npz", &[("t", npy("<f4", ")junk(", &[]), false)]);
        assert!(ztensor_compat::open(&path).is_err());
    }

    /// H1: a shape declaring gigabytes must not reserve them.
    #[test]
    fn huge_declared_shape_rejected() {
        let path = write_npz(
            "h1.npz",
            &[("t", npy("<f8", "(536870864,)", &[0u8; 8]), true)],
        );
        // Either the size equation rejects it at open, or the read is
        // bounded, never an unbounded reservation.
        if let Ok(n) = ztensor_compat::open(&path) {
            assert!(n.tensor("t").unwrap().bytes().is_err());
        }
    }

    /// M5: an archive whose entries collide on a tensor name must never
    /// yield two different meanings for that name. (The ZIP writer refuses
    /// literal duplicates, so the second entry is renamed in the bytes:
    /// what a hostile producer would do. The ZIP *reader* then collapses
    /// them, so the projection sees one entry; the guard in `open` covers
    /// the case where it does not.)
    #[test]
    fn duplicate_names_are_unambiguous() {
        let path = write_npz(
            "dup.npz",
            &[
                ("ta", npy("|u1", "(1,)", &[1]), false),
                ("tb", npy("|u1", "(1,)", &[2]), false),
            ],
        );
        let mut bytes = fs::read(&path).unwrap();
        for i in 0..bytes.len().saturating_sub(6) {
            if &bytes[i..i + 6] == b"tb.npy" {
                bytes[i + 1] = b'a';
            }
        }
        let dup = tmp("dup2.npz");
        fs::write(&dup, &bytes).unwrap();

        match ztensor_compat::open(&dup) {
            Err(_) => {}
            Ok(n) => {
                assert_eq!(n.len(), 1);
                // Whatever it resolved to, reading it must agree with the
                // manifest's declared size.
                let declared = n.tensor("ta").unwrap().nbytes();
                assert_eq!(
                    n.tensor("ta")
                        .unwrap()
                        .bytes()
                        .unwrap()
                        .into_owned()
                        .len() as u64,
                    declared
                );
            }
        }
    }
}

// =======================================================================
// pt
// =======================================================================

#[cfg(feature = "pickle")]
mod pt {
    use super::*;
    use std::io::Write;

    fn write_pt(name: &str, pickle: &[u8]) -> PathBuf {
        let path = tmp(name);
        let mut z = zip::ZipWriter::new(fs::File::create(&path).unwrap());
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        z.start_file("archive/data.pkl", opts).unwrap();
        z.write_all(pickle).unwrap();
        z.start_file("archive/data/0", opts).unwrap();
        z.write_all(&[0u8; 16]).unwrap();
        z.finish().unwrap();
        path
    }

    /// H3: `T <- (T, T)` doubled the heap every few opcodes because the
    /// memo deep-cloned tuples. 24 rounds took ~17 s before the fix.
    #[test]
    fn memo_self_doubling_is_bounded() {
        let mut p = vec![0x80, 0x02]; // PROTO 2
        p.extend([0x8c, 0x01, b'x']); // SHORT_BINUNICODE "x"
        p.push(0x85); // TUPLE1
        p.push(0x94); // MEMOIZE
        for _ in 0..30 {
            p.extend([0x68, 0x00]); // BINGET 0
            p.extend([0x68, 0x00]); // BINGET 0
            p.push(0x86); // TUPLE2
            p.push(0x94); // MEMOIZE
        }
        p.push(0x2e); // STOP

        let path = write_pt("h3.pt", &p);
        let start = std::time::Instant::now();
        let _ = ztensor_compat::open(&path); // errs: no tensors; the point is it returns
        assert!(
            start.elapsed().as_secs() < 5,
            "pickle memo blow-up: {:?}",
            start.elapsed()
        );
    }

    /// H7: mark-less POP_MARKs rescanned the whole stack each time.
    #[test]
    fn markless_pop_is_linear() {
        let mut p = vec![0x80, 0x02];
        p.extend(std::iter::repeat_n(0x4eu8, 200_000)); // NONE
        p.extend(std::iter::repeat_n(0x31u8, 200_000)); // POP_MARK, no mark
        p.push(0x2e);
        let path = write_pt("h7.pt", &p);
        let start = std::time::Instant::now();
        let _ = ztensor_compat::open(&path);
        assert!(
            start.elapsed().as_secs() < 5,
            "quadratic pop_to_mark: {:?}",
            start.elapsed()
        );
    }
}
