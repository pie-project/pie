use std::fs;
use std::path::PathBuf;

use xxhash_rust::xxh3::xxh3_64;
use ztensor::format::cbor;
use ztensor::format::cbor::Value;
use ztensor::format::{MAGIC, VERSION};
use ztensor::vocab::{gguf, CsrPlan, Layout};
use ztensor::{Error, Leaf, Object, Rule, Source, Term, Vocabulary, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn le_u32s(vals: &[u32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32s(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn padded(planes: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, p) in planes.iter().enumerate() {
        if i > 0 {
            out.resize(out.len().div_ceil(256) * 256, 0);
        }
        out.extend_from_slice(p);
    }
    out
}

fn text(s: &str) -> Value {
    Value::Text(s.to_string())
}

fn assemble(path: &PathBuf, object: Vec<(Value, Value)>) {
    let manifest = Value::Map(vec![(
        text("objects"),
        Value::Map(vec![(text("t"), Value::Map(object))]),
    )]);
    let manifest_bytes = cbor::encode(&manifest).unwrap();
    let m_off = 8192u64;
    let mut bytes = vec![0xabu8; m_off as usize];
    bytes[..8].copy_from_slice(&MAGIC);
    bytes.extend_from_slice(&manifest_bytes);
    let mut footer = [0u8; 40];
    footer[0..8].copy_from_slice(&m_off.to_le_bytes());
    footer[8..16].copy_from_slice(&(manifest_bytes.len() as u64).to_le_bytes());
    footer[16..24].copy_from_slice(&xxh3_64(&manifest_bytes).to_le_bytes());
    footer[24..28].copy_from_slice(&VERSION.to_le_bytes());
    footer[32..40].copy_from_slice(&MAGIC);
    bytes.extend_from_slice(&footer);
    fs::write(path, &bytes).unwrap();
}

fn blob(offset: u64, length: u64) -> (Value, Value) {
    (
        text("blob"),
        Value::Map(vec![
            (text("offset"), Value::Uint(offset)),
            (text("length"), Value::Uint(length)),
        ]),
    )
}

fn shape(dims: &[u64]) -> (Value, Value) {
    (
        text("shape"),
        Value::Array(dims.iter().map(|&d| Value::Uint(d)).collect()),
    )
}

#[test]
fn profiles_every_case() {
    a_group_type_round_trips_plane_by_plane();
    planes_and_padded_bytes_are_the_same_object();
    the_writer_checks_planes_against_the_type();
    a_malformed_type_is_rejected();
    a_blob_of_the_wrong_size_is_rejected();
    an_object_with_no_type_and_no_layout_is_rejected();
    leaf_content_rules_are_checked_on_verify();
    block_digests_verify_one_window_at_a_time();
    block_digests_agree_across_every_way_of_writing();
    csr_round_trips_through_its_plan();
    writer_rejects_invalid_csr_metadata();
    a_csr_blob_of_the_wrong_size_is_rejected_by_a_reader_that_knows();
    a_gguf_block_type_is_kept_byte_for_byte();
    a_gguf_object_must_match_its_row();
    a_codebook_gguf_type_takes_no_term();
    every_gguf_row_is_a_registered_layout();
    an_unregistered_layout_is_written_and_stays_structural();
    a_registered_layout_is_checked_like_a_built_in();
    a_registered_layout_rejects_a_file_that_violates_it();
}

fn a_group_type_round_trips_plane_by_plane() {
    let path = tmp("planes.zt");
    let term = Term::parse("g32_u4_bf16_b_bf16").unwrap();
    let codes: Vec<u8> = (0..64).collect();
    let scales = [0x3fu8; 8];
    let biases = [0x40u8; 8];

    let mut w = Writer::create(&path).unwrap();
    w.object("q", |o| {
        o.shape([2u64, 64])
            .term(term.clone())
            .planes([&codes[..], &scales[..], &biases[..]])
    })
    .unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let q = src.tensor("q").unwrap();
    assert_eq!(q.term(), Some(&term));
    assert_eq!(q.term().unwrap().to_string(), "g32_u4_bf16_b_bf16");
    let bytes = q.map().unwrap();
    let planes = q.planes().unwrap();
    let paths: Vec<&str> = planes.iter().map(|p| p.path.as_str()).collect();
    assert_eq!(paths, ["code", "gain", "offset"]);
    assert_eq!(planes[1].shape, [2, 2]);
    assert_eq!(planes[1].leaf, Leaf::BF16);
    assert_eq!(&bytes[planes[0].range()], &codes[..]);
    assert_eq!(&bytes[planes[1].range()], &scales[..]);
    assert_eq!(&bytes[planes[2].range()], &biases[..]);
    assert_eq!(q.nbytes(), term.canonical_size(&[2, 64]).unwrap());
    assert!(q.verify().unwrap().is_checked());
}

fn planes_and_padded_bytes_are_the_same_object() {
    let path = tmp("planes-vs-bytes.zt");
    let term = Term::parse("g16_e2m1_gt_e4m3_f32_n_n").unwrap();
    let codes = vec![0x21u8; 24];
    let scales = vec![0x44u8; 3];
    let global = f32s(&[2.0]);
    let whole = padded(&[&codes, &scales, &global]);
    assert_eq!(whole.len() as u64, term.canonical_size(&[3, 16]).unwrap());

    let mut w = Writer::create(&path).unwrap();
    w.object("by_bytes", |o| {
        o.shape([3u64, 16]).term(term.clone()).bytes(&whole)
    })
    .unwrap();
    w.object("by_planes", |o| {
        o.shape([3u64, 16])
            .term(term.clone())
            .planes([&codes[..], &scales[..], &global[..]])
    })
    .unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let a = src.tensor("by_bytes").unwrap();
    let b = src.tensor("by_planes").unwrap();
    assert_eq!(a.map().unwrap(), b.map().unwrap());
    assert_eq!(a.digest(), b.digest());
    assert_eq!(a.locate().unwrap(), b.locate().unwrap(), "shared blob");
    assert_eq!(a.planes().unwrap(), b.planes().unwrap());
}

fn the_writer_checks_planes_against_the_type() {
    let path = tmp("planes-bad.zt");
    let mut w = Writer::create(&path).unwrap();
    let term = Term::parse("g32_e2m1_e8m0_n").unwrap();
    let (codes, scales) = ([0u8; 32], [0u8; 2]);

    let err = w
        .object("q", |o| {
            o.shape([64u64]).term(term.clone()).planes([&codes[..]])
        })
        .unwrap_err();
    assert!(err.to_string().contains("2 planes"), "{err}");

    let err = w
        .object("q", |o| {
            o.shape([64u64])
                .term(term.clone())
                .planes([&codes[..], &scales[..1]])
        })
        .unwrap_err();
    assert!(err.to_string().contains("\"gain\""), "{err}");

    let err = w
        .object("q", |o| o.shape([64u64]).term(term.clone()).bytes(&codes))
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err}");

    let err = w
        .object("q", |o| {
            o.shape([100u64])
                .term(term.clone())
                .planes([&codes[..], &scales[..]])
        })
        .unwrap_err();
    assert!(err.to_string().contains("g32"), "{err}");
    w.abandon();
}

fn a_malformed_type_is_rejected() {
    for bad in ["g64_u4_bf16", "f4", "u0", "bf16_n", "G64_u4_bf16_n"] {
        let err = Term::parse(bad).unwrap_err();
        assert_eq!(err.rule(), Some(Rule::Type), "{bad:?}: {err}");
    }

    let path = tmp("bad-type.zt");
    assemble(
        &path,
        vec![
            shape(&[64]),
            (text("type"), text("g64_u4_bf16")),
            blob(4096, 64),
        ],
    );
    let err = Source::open(&path).unwrap_err();
    assert_eq!(err.rule(), Some(Rule::Type), "{err}");

    let path = tmp("bad-type-shape.zt");
    assemble(
        &path,
        vec![
            shape(&[100]),
            (text("type"), text("g32_u4_bf16_n")),
            blob(4096, 64),
        ],
    );
    let err = Source::open(&path).unwrap_err();
    assert_eq!(err.rule(), Some(Rule::Type), "{err}");
}

fn a_blob_of_the_wrong_size_is_rejected() {
    let path = tmp("bad-size.zt");
    assemble(
        &path,
        vec![
            shape(&[64]),
            (text("type"), text("g32_e2m1_e8m0_n")),
            blob(4096, 34),
        ],
    );
    let err = Source::open(&path).unwrap_err();
    assert_eq!(err.rule(), Some(Rule::Size), "{err}");

    let path = tmp("good-size.zt");
    assemble(
        &path,
        vec![
            shape(&[64]),
            (text("type"), text("g32_e2m1_e8m0_n")),
            blob(4096, 258),
        ],
    );
    Source::open(&path).unwrap();
}

fn an_object_with_no_type_and_no_layout_is_rejected() {
    let path = tmp("no-type.zt");
    assemble(&path, vec![shape(&[64]), blob(4096, 64)]);
    let err = Source::open(&path).unwrap_err();
    assert_eq!(err.rule(), Some(Rule::Schema), "{err}");

    let mut w = Writer::create(tmp("no-type-w.zt")).unwrap();
    let err = w
        .object("t", |o| o.shape([64u64]).bytes(&[0u8; 64]))
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err}");
    w.abandon();
}

fn leaf_content_rules_are_checked_on_verify() {
    let path = tmp("content-rules.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add("flags", [2u64], Leaf::Bool, &[1, 2]).unwrap();
    w.add("ok", [3u64], Leaf::U(4), &[0x21, 0x03]).unwrap();
    w.add("packed", [3u64], Leaf::U(4), &[0x21, 0xf3]).unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    src.tensor("ok").unwrap().verify().unwrap();
    for name in ["flags", "packed"] {
        let err = src.tensor(name).unwrap().verify().unwrap_err();
        assert_eq!(err.rule(), Some(Rule::LayoutData), "{name}: {err}");
    }
}

fn block_digests_verify_one_window_at_a_time() {
    let path = tmp("blocks.zt");
    let data: Vec<u8> = (0..2500u32).map(|i| (i * 7 % 251) as u8).collect();
    let mut w = Writer::options()
        .canonical(false)
        .blocks(1000)
        .create(&path)
        .unwrap();
    w.add("t", [2500u64], Leaf::U8, &data).unwrap();
    w.finish().unwrap();
    let mut w = Writer::options()
        .canonical(false)
        .create(tmp("blocks-none.zt"))
        .unwrap();
    w.add("plain", [4u64], Leaf::U8, &[1, 2, 3, 4]).unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let t = src.tensor("t").unwrap();
    let blocks = t.blocks().unwrap();
    assert_eq!(blocks.size, 1000);
    assert_eq!(blocks.digests.len(), 3);
    assert_eq!(ztensor::Blocks::count(1000, 2500), 3);
    let bytes = t.map().unwrap();
    for which in 0..3 {
        let span = blocks.span(which, t.nbytes()).unwrap();
        t.verify_block(which, &bytes[span.start as usize..span.end as usize])
            .unwrap();
    }
    assert!(t.verify_block(3, &[]).is_err(), "no fourth block");
    assert!(
        matches!(
            t.verify_block(0, &bytes[..999]),
            Err(Error::InvalidInput(_))
        ),
        "a window of the wrong length is a caller error"
    );
    let mut wrong = bytes[2000..].to_vec();
    wrong[3] ^= 1;
    assert_eq!(
        t.verify_block(2, &wrong).unwrap_err().rule(),
        Some(Rule::Digest)
    );
    let src = Source::open(tmp("blocks-none.zt")).unwrap();
    assert!(src.tensor("plain").unwrap().blocks().is_none());
    assert!(matches!(
        src.tensor("plain").unwrap().verify_block(0, &[1, 2, 3, 4]),
        Err(Error::Unsupported(_))
    ));

    let err = Writer::options()
        .blocks(1000)
        .create(tmp("blocks-canonical.zt"))
        .unwrap_err();
    assert!(err.to_string().contains("canonical(false)"), "{err}");
}

fn block_digests_agree_across_every_way_of_writing() {
    let term = Term::parse("g32_e2m1_e8m0_n").unwrap();
    let codes: Vec<u8> = (0..96).collect();
    let scales = vec![0x7fu8; 6];
    let whole = padded(&[&codes, &scales]);

    let opts = || Writer::options().canonical(false).blocks(50);
    let by_bytes = tmp("blocks-bytes.zt");
    let mut w = opts().create(&by_bytes).unwrap();
    w.object("t", |o| o.shape([192u64]).term(term.clone()).bytes(&whole))
        .unwrap();
    w.finish().unwrap();

    let by_planes = tmp("blocks-planes.zt");
    let mut w = opts().create(&by_planes).unwrap();
    w.object("t", |o| {
        o.shape([192u64])
            .term(term.clone())
            .planes([&codes[..], &scales[..]])
    })
    .unwrap();
    w.finish().unwrap();

    let streamed = tmp("blocks-streamed.zt");
    let mut w = opts().create(&streamed).unwrap();
    let mut sink = w
        .stream("t", |o| {
            o.shape([192u64])
                .term(term.clone())
                .length(whole.len() as u64)
        })
        .unwrap();
    for chunk in whole.chunks(37) {
        sink.write(&mut w, chunk).unwrap();
    }
    sink.close(&mut w).unwrap();
    w.finish().unwrap();

    let reference = fs::read(&by_bytes).unwrap();
    assert_eq!(fs::read(&by_planes).unwrap(), reference);
    assert_eq!(fs::read(&streamed).unwrap(), reference);
    let src = Source::open(&by_bytes).unwrap();
    assert_eq!(src.tensor("t").unwrap().blocks().unwrap().digests.len(), 6);
}

fn csr_bytes(plan: &CsrPlan, values: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; plan.size as usize];
    let indptr = le_u32s(&[0, 1, 2]);
    let indices = le_u32s(&[2, 0]);
    buf[plan.indptr.range()].copy_from_slice(&indptr);
    buf[plan.indices.range()].copy_from_slice(&indices);
    buf[plan.values[0].range()].copy_from_slice(values);
    buf
}

fn csr_round_trips_through_its_plan() {
    let path = tmp("csr.zt");
    let object = Object {
        shape: vec![2, 3],
        term: Some(Term::Leaf(Leaf::F32)),
        layout: Some("zt.sparse_csr/2".into()),
        attributes: Some(cbor::map::<&str, Value>([
            ("index", "u32".into()),
            ("nnz", 2u64.into()),
        ])),
        blob: ztensor::Blob::local(0, 0),
    };
    let plan = CsrPlan::of(
        "m",
        &object.shape,
        object.term.as_ref(),
        object.attributes.as_ref(),
    )
    .unwrap();
    assert_eq!(
        (plan.rows, plan.cols, plan.nnz, plan.index),
        (2, 3, 2, Leaf::U32)
    );
    assert_eq!(plan.indptr.offset, 0);
    assert_eq!(plan.indices.offset, 256);
    assert_eq!(plan.values[0].offset, 512);
    assert_eq!(plan.size, 520);
    let values = f32s(&[1.5, -2.0]);
    let buf = csr_bytes(&plan, &values);

    let mut w = Writer::create(&path).unwrap();
    w.object("m", |o| {
        o.shape([2u64, 3])
            .term(Leaf::F32)
            .layout("zt.sparse_csr/2")
            .attr("index", "u32")
            .attr("nnz", 2u64)
            .bytes(&buf)
    })
    .unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let m = src.tensor("m").unwrap();
    assert_eq!(m.layout(), Some("zt.sparse_csr/2"));
    assert!(
        matches!(m.planes(), Err(Error::Unsupported(_))),
        "a named layout places its own planes"
    );
    let stored = src.provenance().as_root().unwrap().object("m").unwrap();
    let plan = CsrPlan::of(
        "m",
        &stored.shape,
        stored.term.as_ref(),
        stored.attributes.as_ref(),
    )
    .unwrap();
    let bytes = m.map().unwrap();
    assert_eq!(&bytes[plan.indices.range()], &le_u32s(&[2, 0])[..]);
    assert_eq!(&bytes[plan.values[0].range()], &values[..]);
    assert!(m.verify().unwrap().is_checked());
}

fn writer_rejects_invalid_csr_metadata() {
    let path = tmp("csr-bad.zt");
    let mut w = Writer::create(&path).unwrap();
    let buf = vec![0u8; 520];

    let err = w
        .object("m", |o| {
            o.shape([2u64, 3])
                .term(Leaf::F32)
                .layout("zt.sparse_csr/2")
                .attr("index", "u32")
                .bytes(&buf)
        })
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err}");
    assert!(err.to_string().contains("nnz"), "{err}");

    let err = w
        .object("m", |o| {
            o.shape([2u64, 3])
                .layout("zt.sparse_csr/2")
                .attr("index", "u32")
                .attr("nnz", 2u64)
                .bytes(&buf)
        })
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err}");

    let err = w
        .object("m", |o| {
            o.shape([2u64, 3])
                .term(Leaf::F32)
                .layout("zt.sparse_csr/2")
                .attr("index", "u32")
                .attr("nnz", 2u64)
                .bytes(&buf[..519])
        })
        .unwrap_err();
    assert!(err.to_string().contains("520"), "{err}");
    w.abandon();
}

fn a_csr_blob_of_the_wrong_size_is_rejected_by_a_reader_that_knows() {
    let path = tmp("csr-size.zt");
    let bare = Vocabulary::default();
    let mut w = Writer::options().vocabulary(&bare).create(&path).unwrap();
    w.object("m", |o| {
        o.shape([2u64, 3])
            .term(Leaf::F32)
            .layout("zt.sparse_csr/2")
            .attr("index", "u32")
            .attr("nnz", 2u64)
            .bytes(&[0u8; 100])
    })
    .unwrap();
    w.finish().unwrap();

    let err = Source::open(&path).unwrap_err();
    assert_eq!(err.rule(), Some(Rule::Size), "{err}");
    let src = Source::options().vocabulary(&bare).open(&path).unwrap();
    assert_eq!(src.tensor("m").unwrap().nbytes(), 100);
}

fn a_gguf_block_type_is_kept_byte_for_byte() {
    let path = tmp("gguf.zt");
    let row = gguf::row_of("q4_0").unwrap();
    assert_eq!(row.layout_id(), "gguf.q4_0/2");
    let term = row.term().unwrap();
    assert_eq!(term.to_string(), "g32_i4_f16_n");
    let blocks: Vec<u8> = (0..72).collect();

    let mut w = Writer::create(&path).unwrap();
    w.object("w", |o| {
        o.shape([2u64, 64])
            .term(term.clone())
            .layout(row.layout_id())
            .attr("elems_per_block", row.elems_per_block)
            .attr("block_bytes", row.block_bytes)
            .bytes(&blocks)
    })
    .unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let t = src.tensor("w").unwrap();
    assert_eq!(t.layout(), Some("gguf.q4_0/2"));
    assert_eq!(t.term(), Some(&term));
    assert_eq!(t.map().unwrap(), &blocks[..]);
    assert!(t.verify().unwrap().is_checked());
}

fn a_gguf_object_must_match_its_row() {
    let mut w = Writer::create(tmp("gguf-bad.zt")).unwrap();
    let row = gguf::row_of("q4_0").unwrap();
    let blocks = [0u8; 36];

    let err = w
        .object("w", |o| {
            o.shape([64u64])
                .term(Term::parse("g32_u4_f16_n").unwrap())
                .layout("gguf.q4_0/2")
                .attr("elems_per_block", 32u64)
                .attr("block_bytes", 18u64)
                .bytes(&blocks)
        })
        .unwrap_err();
    assert!(err.to_string().contains("g32_i4_f16_n"), "{err}");

    let err = w
        .object("w", |o| {
            o.shape([64u64])
                .term(row.term().unwrap())
                .layout("gguf.q4_0/2")
                .bytes(&blocks)
        })
        .unwrap_err();
    assert!(err.to_string().contains("elems_per_block"), "{err}");

    let err = w
        .object("w", |o| {
            o.shape([48u64])
                .term(row.term().unwrap())
                .layout("gguf.q4_0/2")
                .attr("elems_per_block", 32u64)
                .attr("block_bytes", 18u64)
                .bytes(&blocks[..27])
        })
        .unwrap_err();
    assert!(err.to_string().contains("multiple"), "{err}");

    let err = w
        .object("w", |o| {
            o.shape([64u64])
                .term(row.term().unwrap())
                .layout("gguf.q4_0/2")
                .attr("elems_per_block", 32u64)
                .attr("block_bytes", 18u64)
                .bytes(&blocks[..35])
        })
        .unwrap_err();
    assert!(err.to_string().contains("36"), "{err}");
    w.abandon();
}

fn a_codebook_gguf_type_takes_no_term() {
    let path = tmp("gguf-codebook.zt");
    let row = gguf::row_of("iq4_nl").unwrap();
    assert!(row.term().is_none());
    let blocks = vec![3u8; 36];

    let mut w = Writer::create(&path).unwrap();
    let err = w
        .object("w", |o| {
            o.shape([64u64])
                .term(Leaf::U8)
                .layout(row.layout_id())
                .attr("elems_per_block", row.elems_per_block)
                .attr("block_bytes", row.block_bytes)
                .bytes(&blocks)
        })
        .unwrap_err();
    assert!(err.to_string().contains("codebook"), "{err}");
    w.object("w", |o| {
        o.shape([64u64])
            .layout(row.layout_id())
            .attr("elems_per_block", row.elems_per_block)
            .attr("block_bytes", row.block_bytes)
            .bytes(&blocks)
    })
    .unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let t = src.tensor("w").unwrap();
    assert_eq!(t.term(), None);
    assert!(matches!(t.planes(), Err(Error::Unsupported(_))));
    assert_eq!(t.map().unwrap(), &blocks[..]);
    let stored = src.provenance().as_root().unwrap().object("w").unwrap();
    assert!(matches!(stored.term(), Err(Error::Unsupported(_))));
    assert!(t.verify().unwrap().is_checked());
}

fn every_gguf_row_is_a_registered_layout() {
    let vocab = Vocabulary::standard();
    for row in gguf::TABLE {
        assert!(vocab.layout(&row.layout_id()).is_some(), "{}", row.name);
        let _ = row.term();
    }
    assert!(vocab.layout("gguf.q9_9/2").is_none());
}

fn an_unregistered_layout_is_written_and_stays_structural() {
    let path = tmp("custom-layout.zt");
    let mut w = Writer::create(&path).unwrap();
    w.object("q", |o| {
        o.shape([64u64])
            .term(Leaf::U8)
            .layout("pie.custom/1")
            .bytes(&[2u8; 96])
    })
    .unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let tensor = src.tensor("q").unwrap();
    assert_eq!(tensor.layout(), Some("pie.custom/1"));
    assert_eq!(&*tensor.bytes().unwrap(), &[2u8; 96]);
    assert!(matches!(tensor.planes(), Err(Error::Unsupported(_))));
    assert!(tensor.verify().unwrap().is_checked());
}

struct Grouped;

impl Layout for Grouped {
    fn id(&self) -> &str {
        "pie.custom/1"
    }

    fn validate(&self, name: &str, obj: &Object) -> ztensor::Result<()> {
        let has = obj
            .attributes
            .as_ref()
            .and_then(|a| a.get("group_size"))
            .is_some();
        if !has {
            return Err(Error::reject(
                Rule::LayoutRule,
                format!("{name:?}: pie.custom/1 requires attribute 'group_size'"),
            ));
        }
        Ok(())
    }
}

fn a_registered_layout_is_checked_like_a_built_in() {
    let vocab = Vocabulary::standard().with_layout(Grouped);

    let path = tmp("registered-layout.zt");
    let mut w = Writer::options().vocabulary(&vocab).create(&path).unwrap();
    let err = w
        .object("q", |o| {
            o.shape([32u64])
                .term(Leaf::U8)
                .layout("pie.custom/1")
                .bytes(&[1u8; 32])
        })
        .unwrap_err();
    assert!(format!("{err}").contains("group_size"), "{err}");

    w.object("q", |o| {
        o.shape([32u64])
            .term(Leaf::U8)
            .layout("pie.custom/1")
            .attr("group_size", 16u64)
            .bytes(&[1u8; 32])
    })
    .unwrap();
    w.finish().unwrap();

    Source::options().vocabulary(&vocab).open(&path).unwrap();
    let plain = Source::open(&path).unwrap();
    assert_eq!(plain.tensor("q").unwrap().map().unwrap(), &[1u8; 32]);
}

fn a_registered_layout_rejects_a_file_that_violates_it() {
    let path = tmp("violates-registered-layout.zt");
    let mut w = Writer::create(&path).unwrap();
    w.object("q", |o| {
        o.shape([32u64])
            .term(Leaf::U8)
            .layout("pie.custom/1")
            .bytes(&[1u8; 32])
    })
    .unwrap();
    w.finish().unwrap();

    let vocab = Vocabulary::standard().with_layout(Grouped);
    let err = Source::options()
        .vocabulary(&vocab)
        .open(&path)
        .unwrap_err();
    assert_eq!(err.rule(), Some(Rule::LayoutRule), "{err}");
    assert!(Source::open(&path).is_ok());
}

#[cfg(feature = "zstd")]
mod zstd_seekable {
    use super::*;
    use std::borrow::Cow;

    const ENC: &str = "zt.zstd-seekable/1";

    fn writer(path: &PathBuf) -> Writer {
        Writer::options()
            .canonical(false)
            .align(4096)
            .create(path)
            .unwrap()
    }

    #[test]
    fn profiles_1_every_case() {
        encoded_dense_roundtrip();
        encoded_empty_blob();
        canonical_forbids_encoding();
        an_unregistered_encoding_is_refused_not_guessed();
        corrupt_stream_rejected_not_zero_filled();
    }

    fn encoded_dense_roundtrip() {
        let path = tmp("zstd.zt");
        let data: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();

        let mut w = writer(&path);
        w.object("t", |o| {
            o.shape([3_000_000u64])
                .term(Leaf::U8)
                .encoding(ENC)
                .bytes(&data)
        })
        .unwrap();
        w.finish().unwrap();

        let src = Source::open(&path).unwrap();
        let tensor = src.tensor("t").unwrap();
        let stored = src.provenance().as_root().unwrap().object("t").unwrap();
        assert_eq!(stored.blob.encoding.as_deref(), Some(ENC));
        assert_eq!(stored.blob.decoded_length, Some(data.len() as u64));
        assert!(stored.blob.length < data.len() as u64, "should compress");
        assert_eq!(tensor.nbytes(), data.len() as u64);

        assert_eq!(&*tensor.bytes().unwrap(), &data[..]);
        assert!(tensor.verify().unwrap().is_checked());

        let caps = tensor.caps();
        assert!(!caps.map && !caps.locate);
        assert!(matches!(tensor.map(), Err(Error::Unsupported(_))));
        assert!(matches!(tensor.locate(), Err(Error::Unsupported(_))));
        assert!(matches!(tensor.bytes().unwrap(), Cow::Owned(_)));
    }

    fn encoded_empty_blob() {
        let path = tmp("zstd-empty.zt");
        let mut w = writer(&path);
        w.object("e", |o| {
            o.shape([0u64]).term(Leaf::U8).encoding(ENC).bytes(&[])
        })
        .unwrap();
        w.finish().unwrap();
        let src = Source::open(&path).unwrap();
        assert_eq!(&*src.tensor("e").unwrap().bytes().unwrap(), &[] as &[u8]);
    }

    fn canonical_forbids_encoding() {
        let path = tmp("zstd-canonical.zt");
        let mut w = Writer::create(&path).unwrap();
        let err = w
            .object("t", |o| {
                o.shape([4u64])
                    .term(Leaf::U8)
                    .encoding(ENC)
                    .bytes(&[1, 2, 3, 4])
            })
            .unwrap_err();
        assert!(format!("{err}").contains("canonical(false)"), "{err}");
    }

    fn an_unregistered_encoding_is_refused_not_guessed() {
        let path = tmp("zstd-unknown-encoding.zt");
        let data = vec![9u8; 4096];
        let mut w = writer(&path);
        w.object("t", |o| {
            o.shape([4096u64]).term(Leaf::U8).encoding(ENC).bytes(&data)
        })
        .unwrap();
        w.finish().unwrap();

        let bare = Vocabulary::default();
        let src = Source::options().vocabulary(&bare).open(&path).unwrap();
        let err = src.tensor("t").unwrap().bytes().unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    }

    fn corrupt_stream_rejected_not_zero_filled() {
        let path = tmp("zstd-corrupt.zt");
        let data = vec![9u8; 100_000];
        let mut w = writer(&path);
        w.object("t", |o| {
            o.shape([100_000u64])
                .term(Leaf::U8)
                .encoding(ENC)
                .bytes(&data)
        })
        .unwrap();
        w.finish().unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[4096 + 10] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();

        let src = Source::open(&path).unwrap();
        let err = src.tensor("t").unwrap().bytes().unwrap_err();
        assert_eq!(err.rule(), Some(Rule::Encoding), "{err:?}");
    }
}
