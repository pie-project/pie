//! Whether a safetensors file may be laid out so a streamer can map it.
//!
//! Streaming a weight means the GPU reads it where it lies, and a mapping can
//! only begin on a page. Checkpoints as published pack their tensors back to
//! back, so almost no tensor begins on one -- which is why the streamed path
//! today first rewrites the bytes into a page-aligned file of its own.
//!
//! The question these tests answer is whether that rewrite has to produce a
//! *new format*, or whether it can produce a safetensors file that merely
//! chose different offsets. If it can, a streamable checkpoint needs no
//! reader, no format version and no separate tooling. It is a checkpoint.
//!
//! It can, but not the obvious way. Leaving a gap between two tensors is
//! describable -- every tensor carries explicit `data_offsets` -- and this
//! loader accepts it, since it validates each tensor's extent against the file
//! and never against its neighbour. The reference implementation does not:
//! `safetensors` 0.8 rejects the file outright with `invalid offset for tensor
//! b`, because it requires each tensor to begin where the last one ended. A
//! file only this loader can read is a new format wearing a familiar
//! extension, which is the outcome worth avoiding.
//!
//! So the padding is made explicit instead. The space before a tensor that
//! needs to start on a page is *itself a tensor* -- a `U8` filler under a
//! reserved name -- and the file stays contiguous. Every reader accepts it,
//! every real tensor begins on a page, and what a contract does about the
//! fillers is that it never claims them.

use std::io::Write;
use std::path::{Path, PathBuf};

use pie_loader::checkpoint::read::parse_checkpoint_metadata;

const PAGE: u64 = 16384; // The largest page any target here uses (Apple silicon).

/// Reserved name for the space that exists only to push the next tensor onto a
/// page. A contract never claims these.
const PAD_PREFIX: &str = "__pie_pad__.";

struct Tensor {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<i64>,
    bytes: u64,
}

fn tensor(name: &'static str, rows: i64, cols: i64) -> Tensor {
    Tensor {
        name,
        dtype: "BF16",
        shape: vec![rows, cols],
        bytes: (rows * cols * 2) as u64,
    }
}

/// Write a safetensors file, optionally putting every tensor on a boundary.
///
/// `align_to` of 1 gives the ordinary back-to-back packing. A page gives the
/// layout a streamer wants, reached by inserting a filler tensor ahead of any
/// tensor that would otherwise begin mid-page -- so the file is still
/// contiguous, which is what the reference reader requires. The header is
/// padded to the same boundary, since a tensor's position in the *file* is
/// what a mapping cares about and that is the data section's start plus the
/// tensor's offset within it.
fn write_checkpoint(dir: &Path, tensors: &[Tensor], align_to: u64) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();

    let mut entries: Vec<String> = Vec::new();
    let mut at = 0u64;
    let mut fillers = 0;
    for t in tensors {
        if align_to > 1 && at % align_to != 0 {
            let fill = align_to - (at % align_to);
            entries.push(format!(
                r#""{PAD_PREFIX}{fillers}":{{"dtype":"U8","shape":[{fill}],"data_offsets":[{},{}]}}"#,
                at,
                at + fill
            ));
            at += fill;
            fillers += 1;
        }
        entries.push(format!(
            r#""{}":{{"dtype":"{}","shape":{:?},"data_offsets":[{},{}]}}"#,
            t.name,
            t.dtype,
            t.shape,
            at,
            at + t.bytes
        ));
        at += t.bytes;
    }
    let data_bytes = at;

    // Pad the header so the data section begins on the boundary too. The pad
    // goes in `__metadata__`, which the format reserves for exactly this kind
    // of out-of-band string and which every reader already skips.
    let body = entries.join(",");
    let mut header = format!("{{{body}}}");
    if align_to > 1 {
        let mut pad = 0usize;
        loop {
            let candidate = format!(r#"{{"__metadata__":{{"pad":"{}"}},{body}}}"#, "0".repeat(pad));
            if (8 + candidate.len() as u64) % align_to == 0 {
                header = candidate;
                break;
            }
            pad += 1;
            assert!(pad < align_to as usize * 2, "no header padding lands on the boundary");
        }
    }

    let path = dir.join("model.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(&vec![0u8; data_bytes as usize]).unwrap();
    f.flush().unwrap();

    if align_to > 1 {
        assert_eq!(
            (8 + header.len() as u64) % align_to,
            0,
            "the data section itself must begin on the boundary"
        );
    }
    path
}

fn scratch(tag: &str) -> PathBuf {
    let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
    let dir = PathBuf::from(base).join(format!("pie_aligned_safetensors_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn fixture() -> Vec<Tensor> {
    // Sizes chosen so back-to-back packing lands every tensor after the first
    // off a page: 100 rows of 3 is 600 bytes, which no page divides.
    vec![
        tensor("model.layers.0.mlp.experts.0.gate_proj.weight", 100, 3),
        tensor("model.layers.0.mlp.experts.1.gate_proj.weight", 100, 3),
        tensor("model.layers.0.mlp.experts.2.gate_proj.weight", 100, 3),
    ]
}

/// The baseline: an ordinary checkpoint reads, and its tensors are not aligned.
/// Without this the aligned case below proves nothing -- it could be passing
/// because alignment is already there.
#[test]
fn an_ordinary_checkpoint_is_not_page_aligned() {
    let dir = scratch("packed");
    write_checkpoint(&dir, &fixture(), 1);
    let meta = parse_checkpoint_metadata(&dir).expect("an ordinary checkpoint reads");
    assert_eq!(meta.tensors.len(), 3);

    let aligned = meta
        .tensors
        .iter()
        .filter(|t| t.file_offset % PAGE == 0)
        .count();
    assert!(
        aligned < meta.tensors.len(),
        "the premise: published checkpoints do not put their tensors on pages"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The question. A checkpoint whose tensors each begin on a page reads, and
/// reads as the same tensors -- the gaps between them are not an error, not a
/// tensor, and not counted.
#[test]
fn a_page_aligned_checkpoint_reads_as_a_checkpoint() {
    let dir = scratch("aligned");
    write_checkpoint(&dir, &fixture(), PAGE);
    let meta = parse_checkpoint_metadata(&dir)
        .expect("a page-aligned checkpoint is still a checkpoint");

    let real: Vec<_> = meta
        .tensors
        .iter()
        .filter(|t| !t.name.starts_with(PAD_PREFIX))
        .collect();

    for t in &real {
        assert_eq!(
            t.file_offset % PAGE,
            0,
            "tensor {} begins on a page at {}",
            t.name,
            t.file_offset
        );
        assert_eq!(t.span_bytes, 600, "tensor {} keeps its size", t.name);
    }

    // The real tensors a contract would claim are unchanged, name for name:
    // alignment is a property of where the bytes are, not of what the
    // checkpoint contains. The fillers are extra, and named so nothing claims
    // them by accident.
    let mut names: Vec<&str> = real.iter().map(|t| t.name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        vec![
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            "model.layers.0.mlp.experts.2.gate_proj.weight",
        ]
    );
    assert_eq!(
        meta.tensors.len() - real.len(),
        2,
        "two of the three tensors needed pushing onto a page"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The cost of the choice, stated rather than discovered later. Padding to a
/// page is free for a 132 MB expert bank and ruinous for a bias, so a packer
/// that aligns everything is not the packer to write.
#[test]
fn alignment_costs_what_the_padding_costs() {
    let dir = scratch("cost");
    let small = vec![tensor("a", 1, 8), tensor("b", 1, 8), tensor("c", 1, 8)];
    write_checkpoint(&dir, &small, PAGE);
    let bytes = std::fs::metadata(dir.join("model.safetensors")).unwrap().len();
    assert!(
        bytes > 2 * PAGE,
        "three 16-byte tensors on their own pages cost pages, not bytes: {bytes}"
    );
    // Hence the streaming contract decides what is aligned. GPT-OSS already
    // draws this line by hand: an expert bank is 132 MB and pads for free, its
    // bias is 180 KB a layer and is read whatever routes, so the bias stays
    // packed with everything else.
    let _ = std::fs::remove_dir_all(&dir);
}

/// The property that makes the aligned layout a *checkpoint* rather than a
/// private format: no byte of the data section belongs to nothing. This
/// loader would not notice if it were violated -- it checks each tensor
/// against the file and never against its neighbour -- but `safetensors` 0.8
/// refuses the file with `invalid offset for tensor b`, so a regression here
/// would be invisible until something outside this repository tried to read
/// what we wrote.
#[test]
fn the_aligned_layout_leaves_no_gaps() {
    let dir = scratch("contiguous");
    let path = write_checkpoint(&dir, &fixture(), PAGE);
    let meta = parse_checkpoint_metadata(&dir).unwrap();

    let mut spans: Vec<(u64, u64)> = meta
        .tensors
        .iter()
        .map(|t| (t.file_offset, t.file_offset + t.span_bytes))
        .collect();
    spans.sort_unstable();

    for w in spans.windows(2) {
        assert_eq!(
            w[0].1, w[1].0,
            "a tensor must begin where the last one ended, not at {} after {}",
            w[1].0, w[0].1
        );
    }

    let file_bytes = std::fs::metadata(&path).unwrap().len();
    assert_eq!(spans.last().unwrap().1, file_bytes, "and nothing trails it");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Leave the aligned fixture on disk for the reference reader to open.
/// Ignored by default; the cross-check is `PIE_EMIT_ALIGNED_FIXTURE=<dir>
/// cargo test -p pie-loader --test aligned_safetensors -- --ignored` followed
/// by `safetensors.numpy.load_file` on the result.
#[test]
#[ignore]
fn emit_fixture_for_the_reference_reader() {
    let dir = PathBuf::from(
        std::env::var("PIE_EMIT_ALIGNED_FIXTURE").expect("set PIE_EMIT_ALIGNED_FIXTURE"),
    );
    let path = write_checkpoint(&dir, &fixture(), PAGE);
    println!("{}", path.display());
}
