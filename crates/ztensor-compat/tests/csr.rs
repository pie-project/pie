//! Assembling `zt.sparse_csr/2`, which is a layout profile and so lives here
//! rather than in the core crate.
//!
//! The core writer still validates the layout's *metadata* rules and its size
//! equation, because those are vocabulary and the vocabulary is core's. What
//! is here is the data-level half: reading the planes back out of one blob and
//! checking the invariants that only the bytes can violate.

use std::path::PathBuf;

use ztensor::{Error, Leaf, Source, Term, Writer};
use ztensor_compat::csr;

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn le_u64s(vals: &[u64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32s(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `indptr`, `indices`, then the values, each at the next 256-byte boundary.
fn csr_blob(indptr: &[u8], indices: &[u8], values: &[u8]) -> Vec<u8> {
    let mut blob = indptr.to_vec();
    blob.resize(blob.len().div_ceil(256) * 256, 0);
    blob.extend_from_slice(indices);
    blob.resize(blob.len().div_ceil(256) * 256, 0);
    blob.extend_from_slice(values);
    blob
}

#[test]
fn csr_roundtrip() {
    // [[1.0, 0, 2.0], [0, 3.0, 0]] as CSR
    let path = tmp("csr.zt");
    let values = f32s(&[1.0, 2.0, 3.0]);
    let indices = le_u64s(&[0, 2, 1]);
    let indptr = le_u64s(&[0, 2, 3]);
    let blob = csr_blob(&indptr, &indices, &values);

    let mut w = Writer::create(&path).unwrap();
    w.object("m", |o| {
        o.shape([2u64, 3])
            .layout("zt.sparse_csr/2")
            .term(Leaf::F32)
            .attr("nnz", 3u64)
            .attr("index", "u64")
            .bytes(&blob)
    })
    .unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let tensor = src.tensor("m").unwrap();
    let csr = csr::read(&tensor).unwrap();
    assert_eq!((csr.rows, csr.cols), (2, 3));
    assert_eq!(csr.indices, vec![0, 2, 1]);
    assert_eq!(csr.indptr, vec![0, 2, 3]);
    assert_eq!(csr.values, values);
    assert_eq!(csr.term, Term::Leaf(Leaf::F32));
    // The blob is an ordinary mappable range; the values sit at 512.
    assert_eq!(&tensor.map().unwrap()[512..], &values[..]);
}

/// A data-level violation the metadata cannot see: an index past `cols`.
#[test]
fn index_past_cols_is_rejected() {
    let path = tmp("csr-bad.zt");
    let blob = csr_blob(
        &le_u64s(&[0, 2, 3]),
        &le_u64s(&[0, 3, 1]),
        &f32s(&[1.0, 2.0, 3.0]),
    );
    let mut w = Writer::create(&path).unwrap();
    w.object("m", |o| {
        o.shape([2u64, 3])
            .layout("zt.sparse_csr/2")
            .term(Leaf::F32)
            .attr("nnz", 3u64)
            .attr("index", "u64")
            .bytes(&blob)
    })
    .unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let err = csr::read(&src.tensor("m").unwrap()).unwrap_err();
    assert!(
        matches!(err, Error::Reject { rule: ztensor::Rule::LayoutData, .. }),
        "{err:?}"
    );
}

/// A tensor that is not CSR at all is refused rather than misread. The core
/// crate has the matching test that such a layout still *reads* structurally.
#[test]
fn a_foreign_layout_is_not_assembled_as_csr() {
    let path = tmp("csr-not.zt");
    let mut w = Writer::create(&path).unwrap();
    w.object("q", |o| {
        o.shape([64u64])
            .layout("pie.custom/1")
            .term(Leaf::U8)
            .bytes(&[1u8; 96])
    })
    .unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let tensor = src.tensor("q").unwrap();
    assert!(matches!(csr::read(&tensor), Err(Error::Unsupported(_))));
}
