use std::fs;
use std::path::PathBuf;

fn tmp(name: &str) -> PathBuf {
    let p = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_file(&p);
    p
}

#[cfg(feature = "hdf5")]
mod hdf5 {
    use super::*;

    fn superblock(btree: u64, heap: u64, len: usize) -> Vec<u8> {
        let mut b = vec![0u8; len.max(96)];
        b[0..8].copy_from_slice(b"\x89HDF\r\n\x1a\n");
        b[13] = 8;
        b[14] = 8;
        b[16..18].copy_from_slice(&4u16.to_le_bytes());
        b[18..20].copy_from_slice(&16u16.to_le_bytes());
        let eof = b.len() as u64;
        b[40..48].copy_from_slice(&eof.to_le_bytes());
        b[72..76].copy_from_slice(&1u32.to_le_bytes());
        b[80..88].copy_from_slice(&btree.to_le_bytes());
        b[88..96].copy_from_slice(&heap.to_le_bytes());
        b
    }

    #[test]
    fn hostile_every_case() {
        heap_address_wraparound();
        btree_address_wraparound();
        heap_data_offset_out_of_range();
    }

    fn heap_address_wraparound() {
        let path = tmp("c1.h5");
        fs::write(&path, superblock(96, u64::MAX, 96)).unwrap();
        assert!(ztensor_compat::open(&path).is_err());
    }

    fn btree_address_wraparound() {
        for addr in [u64::MAX - 8, u64::MAX - 1, 1 << 62] {
            let path = tmp("c2.h5");
            fs::write(&path, superblock(addr, 96, 256)).unwrap();
            assert!(ztensor_compat::open(&path).is_err(), "addr {addr}");
        }
    }

    fn heap_data_offset_out_of_range() {
        let mut b = superblock(96, 144, 416);
        b[96..100].copy_from_slice(b"TREE");
        b[102..104].copy_from_slice(&1u16.to_le_bytes());
        b[104..112].copy_from_slice(&[0xff; 8]);
        b[112..120].copy_from_slice(&[0xff; 8]);
        b[128..136].copy_from_slice(&192u64.to_le_bytes());
        b[144..148].copy_from_slice(b"HEAP");
        b[152..160].copy_from_slice(&16u64.to_le_bytes());
        b[168..176].copy_from_slice(&(1u64 << 40).to_le_bytes());
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

#[cfg(feature = "gguf")]
mod gguf {
    use super::*;

    fn gstr(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    #[test]
    fn hostile_1_every_case() {
        data_section_past_eof();
        lying_counts_do_not_allocate();
    }

    fn data_section_past_eof() {
        let mut b = Vec::new();
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        gstr(&mut b, "t");
        b.extend(1u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend(0u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        let path = tmp("c8.gguf");
        fs::write(&path, &b).unwrap();
        match ztensor_compat::open(&path) {
            Err(_) => {}
            Ok(g) => {
                let _ = g.tensor("t").unwrap().bytes().expect("in-bounds read");
            }
        }
    }

    fn lying_counts_do_not_allocate() {
        let mut b = Vec::new();
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend(100_000u64.to_le_bytes());
        b.extend(100_000u64.to_le_bytes());
        let path = tmp("counts.gguf");
        fs::write(&path, &b).unwrap();
        assert!(ztensor_compat::open(&path).is_err());
    }
}

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

    #[test]
    fn hostile_2_every_case() {
        reversed_shape_parens();
        huge_declared_shape_rejected();
        duplicate_names_are_unambiguous();
    }

    fn reversed_shape_parens() {
        let path = write_npz("c9.npz", &[("t", npy("<f4", ")junk(", &[]), false)]);
        assert!(ztensor_compat::open(&path).is_err());
    }

    fn huge_declared_shape_rejected() {
        let path = write_npz(
            "h1.npz",
            &[("t", npy("<f8", "(536870864,)", &[0u8; 8]), true)],
        );
        if let Ok(n) = ztensor_compat::open(&path) {
            assert!(n.tensor("t").unwrap().bytes().is_err());
        }
    }

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
                let declared = n.tensor("ta").unwrap().nbytes();
                assert_eq!(
                    n.tensor("ta").unwrap().bytes().unwrap().into_owned().len() as u64,
                    declared
                );
            }
        }
    }
}

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

    #[test]
    fn hostile_3_every_case() {
        memo_self_doubling_is_bounded();
        markless_pop_is_linear();
    }

    fn memo_self_doubling_is_bounded() {
        let mut p = vec![0x80, 0x02];
        p.extend([0x8c, 0x01, b'x']); // SHORT_BINUNICODE "x"
        p.push(0x85);
        p.push(0x94);
        for _ in 0..30 {
            p.extend([0x68, 0x00]);
            p.extend([0x68, 0x00]);
            p.push(0x86);
            p.push(0x94);
        }
        p.push(0x2e);

        let path = write_pt("h3.pt", &p);
        let start = std::time::Instant::now();
        let _ = ztensor_compat::open(&path);
        assert!(
            start.elapsed().as_secs() < 5,
            "pickle memo blow-up: {:?}",
            start.elapsed()
        );
    }

    fn markless_pop_is_linear() {
        let mut p = vec![0x80, 0x02];
        p.extend(std::iter::repeat_n(0x4eu8, 200_000));
        p.extend(std::iter::repeat_n(0x31u8, 200_000));
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
