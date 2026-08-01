//! Making a checkpoint streamable without making it a different checkpoint.
//!
//! Streaming a weight means the GPU reads it where it lies, through a mapping,
//! so it must begin on a page -- and on a page of its own, since two tensors
//! sharing one means faulting either drags in the other. Checkpoints as
//! published pack their tensors back to back and put almost none of them on a
//! page.
//!
//! The question these tests answer is whether the fix has to produce a *new
//! format*, or whether it can produce a safetensors file that merely chose
//! different offsets. If it can, a streamable checkpoint needs no reader, no
//! format version and no separate tooling. It is a checkpoint.
//!
//! It can, but not the obvious way. Leaving a gap in front of a tensor is
//! describable -- every tensor carries explicit `data_offsets` -- and this
//! loader accepts it, since it validates each tensor against the file and never
//! against its neighbour. The reference implementation does not: `safetensors`
//! 0.8 refuses such a file with `invalid offset for tensor b`, because it
//! requires each tensor to begin where the last one ended. A file only this
//! loader can read is a new format wearing a familiar extension, which is the
//! outcome worth avoiding.
//!
//! So `checkpoint::align` makes the padding explicit: the space ahead of a
//! tensor that needs a page is itself a tensor, under a reserved name, and the
//! file stays contiguous. What is checked below is that the result is still the
//! same checkpoint -- same tensors, same bytes -- and still one the reference
//! reader will open.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use pie_loader::checkpoint::align::{ALIGN_METADATA_KEY, PAD_PREFIX, align_checkpoint};
use pie_loader::checkpoint::read::parse_checkpoint_metadata;

/// The largest page any target here uses (Apple silicon). x86-64's 4 KiB
/// divides it, so one artifact serves both.
const PAGE: u64 = 16384;

struct Tensor {
    name: &'static str,
    dtype: &'static str,
    /// Elements, one byte each, filled with a distinct value so the test can
    /// tell whose bytes ended up where.
    len: u64,
    fill: u8,
}

fn tensor(name: &'static str, len: u64, fill: u8) -> Tensor {
    Tensor {
        name,
        dtype: "U8",
        len,
        fill,
    }
}

/// Write an ordinary checkpoint: tensors back to back, which is how they come.
fn write_packed(dir: &Path, tensors: &[Tensor]) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();

    let mut entries = Vec::new();
    let mut data = Vec::new();
    for t in tensors {
        let begin = data.len() as u64;
        data.extend(std::iter::repeat_n(t.fill, t.len as usize));
        entries.push(format!(
            r#""{}":{{"dtype":"{}","shape":[{}],"data_offsets":[{},{}]}}"#,
            t.name,
            t.dtype,
            t.len,
            begin,
            begin + t.len
        ));
    }
    // A `__metadata__` the rewrite has no business losing.
    let header = format!(
        r#"{{"__metadata__":{{"format":"pt"}},{}}}"#,
        entries.join(",")
    );

    let path = dir.join("model.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(&data).unwrap();
    f.flush().unwrap();
    path
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
        let dir = PathBuf::from(base).join(format!(
            "pie_align_{tag}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Sizes chosen so back-to-back packing lands every tensor after the first off
/// a page: 600 bytes is divisible by no page size.
fn fixture() -> Vec<Tensor> {
    vec![
        tensor("model.layers.0.mlp.experts.0.gate_proj.weight", 600, 0xA1),
        tensor("model.layers.0.mlp.experts.1.gate_proj.weight", 600, 0xB2),
        tensor("model.layers.0.mlp.experts.2.gate_proj.weight", 600, 0xC3),
        tensor("model.layers.0.mlp.experts.bias", 600, 0xD4),
    ]
}

fn experts() -> BTreeSet<String> {
    fixture()
        .iter()
        .filter(|t| t.name.ends_with("gate_proj.weight"))
        .map(|t| t.name.to_string())
        .collect()
}

/// The premise. Without this the alignment below proves nothing: it could be
/// passing because the tensors were on pages to begin with.
#[test]
fn a_published_checkpoint_is_not_page_aligned() {
    let s = Scratch::new("premise");
    let src = s.join("in");
    write_packed(&src, &fixture());

    let meta = parse_checkpoint_metadata(&src).unwrap();
    let aligned = meta
        .tensors
        .iter()
        .filter(|t| t.file_offset % PAGE == 0)
        .count();
    assert!(
        aligned < meta.tensors.len(),
        "published checkpoints do not put their tensors on pages"
    );
}

/// The claim: what the caller asked for is on a page, and what it did not ask
/// for is left where the packing put it -- because a page apiece is a real cost
/// and aligning everything is the one clearly wrong policy.
#[test]
fn alignment_moves_exactly_what_was_asked_for() {
    let s = Scratch::new("moves");
    let (src, dst) = (s.join("in"), s.join("out"));
    write_packed(&src, &fixture());

    let report = align_checkpoint(&src, &dst, &experts(), PAGE).unwrap();
    assert_eq!(report.aligned(), 3);
    assert!(report.missing.is_empty(), "{:?}", report.missing);

    let meta = parse_checkpoint_metadata(&dst).unwrap();
    for name in experts() {
        let t = meta.tensor_by_name(&name).expect("still present");
        assert_eq!(
            t.file_offset % PAGE,
            0,
            "{name} begins at {}, not on a page",
            t.file_offset
        );
    }
    let bias = meta
        .tensor_by_name("model.layers.0.mlp.experts.bias")
        .expect("still present");
    assert_ne!(
        bias.file_offset % PAGE,
        0,
        "the bias was not asked for and should not have been padded onto a page"
    );
}

/// The thing that would make all of this worthless: a rewrite that moves bytes
/// is only useful if it moves the *right* bytes. Each tensor is filled with a
/// distinct value, so a swapped or truncated tensor is visible rather than
/// merely possible.
#[test]
fn alignment_preserves_every_tensors_contents() {
    let s = Scratch::new("bytes");
    let (src, dst) = (s.join("in"), s.join("out"));
    write_packed(&src, &fixture());
    align_checkpoint(&src, &dst, &experts(), PAGE).unwrap();

    let meta = parse_checkpoint_metadata(&dst).unwrap();
    let blob = std::fs::read(dst.join("model.safetensors")).unwrap();
    for t in fixture() {
        let found = meta.tensor_by_name(t.name).expect("still present");
        assert_eq!(found.span_bytes, t.len, "{} kept its size", t.name);
        let begin = found.file_offset as usize;
        let bytes = &blob[begin..begin + t.len as usize];
        assert!(
            bytes.iter().all(|b| *b == t.fill),
            "{} holds {:#x}.. not {:#x}",
            t.name,
            bytes[0],
            t.fill
        );
    }
}

/// The property that makes the result a *checkpoint* rather than a private
/// format: no byte of the data section belongs to nothing. This loader would
/// not notice if it were violated -- it checks each tensor against the file and
/// never against its neighbour -- but `safetensors` 0.8 refuses such a file, so
/// a regression here would stay invisible until something outside this
/// repository tried to read what we wrote.
#[test]
fn the_aligned_layout_leaves_no_gaps() {
    let s = Scratch::new("contiguous");
    let (src, dst) = (s.join("in"), s.join("out"));
    write_packed(&src, &fixture());
    align_checkpoint(&src, &dst, &experts(), PAGE).unwrap();

    let meta = parse_checkpoint_metadata(&dst).unwrap();
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
    let file_bytes = std::fs::metadata(dst.join("model.safetensors"))
        .unwrap()
        .len();
    assert_eq!(spans.last().unwrap().1, file_bytes, "and nothing trails it");
}

/// The fillers are the mechanism, not part of the model. A contract names
/// tensors, so what protects it from claiming one of these is that they are
/// named something no contract would write -- and the count is asserted so a
/// rewrite that quietly stopped padding could not pass the alignment test by
/// accident of the fixture's sizes.
#[test]
fn the_padding_is_named_so_nothing_claims_it() {
    let s = Scratch::new("fillers");
    let (src, dst) = (s.join("in"), s.join("out"));
    write_packed(&src, &fixture());
    let report = align_checkpoint(&src, &dst, &experts(), PAGE).unwrap();

    let meta = parse_checkpoint_metadata(&dst).unwrap();
    let fillers: Vec<_> = meta
        .tensors
        .iter()
        .filter(|t| t.name.starts_with(PAD_PREFIX))
        .collect();
    assert_eq!(fillers.len(), report.files[0].fillers);
    assert_eq!(
        fillers.len(),
        2,
        "the second and third experts needed pushing"
    );
    assert!(
        fillers.iter().all(|t| t.span_bytes > 0),
        "a zero-length filler is a filler that should not have been written"
    );
}

/// Metadata is the checkpoint's, not ours. A rewrite that dropped it would take
/// `format: pt` with it and no test of offsets would notice.
#[test]
fn alignment_keeps_the_checkpoints_metadata_and_records_its_own() {
    let s = Scratch::new("metadata");
    let (src, dst) = (s.join("in"), s.join("out"));
    write_packed(&src, &fixture());
    align_checkpoint(&src, &dst, &experts(), PAGE).unwrap();

    let header = read_header(&dst.join("model.safetensors"));
    let metadata = header["__metadata__"]
        .as_object()
        .expect("metadata survives");
    assert_eq!(metadata["format"], "pt", "the checkpoint's own metadata");
    assert_eq!(
        metadata[ALIGN_METADATA_KEY], "16384",
        "and a record of what this file was aligned to"
    );
}

/// Aligning an aligned checkpoint would stack fillers on fillers, growing the
/// file every time and eventually leaving padding between padding. Refusing is
/// the cheap answer, and the fillers make the condition detectable.
#[test]
fn aligning_twice_is_refused() {
    let s = Scratch::new("twice");
    let (src, once, twice) = (s.join("in"), s.join("out1"), s.join("out2"));
    write_packed(&src, &fixture());
    align_checkpoint(&src, &once, &experts(), PAGE).unwrap();

    let err = align_checkpoint(&once, &twice, &experts(), PAGE).unwrap_err();
    assert!(
        err.to_string().contains("aligned already"),
        "unhelpful error: {err}"
    );
}

/// A name the caller expected and the checkpoint does not have means the two
/// disagree about what is in it. Silently aligning the rest would leave a
/// streamed tensor unaligned: correct, and slow in a way nothing would explain.
#[test]
fn a_name_the_checkpoint_lacks_is_reported() {
    let s = Scratch::new("missing");
    let (src, dst) = (s.join("in"), s.join("out"));
    write_packed(&src, &fixture());

    let mut asked = experts();
    asked.insert("model.layers.9.mlp.experts.0.gate_proj.weight".into());
    let report = align_checkpoint(&src, &dst, &asked, PAGE).unwrap();
    assert_eq!(
        report.missing,
        vec!["model.layers.9.mlp.experts.0.gate_proj.weight"]
    );
}

/// Writing into a directory that already holds something is how a half-written
/// checkpoint comes to look like a whole one.
#[test]
fn an_existing_output_is_refused() {
    let s = Scratch::new("exists");
    let (src, dst) = (s.join("in"), s.join("out"));
    write_packed(&src, &fixture());
    std::fs::create_dir_all(&dst).unwrap();

    let err = align_checkpoint(&src, &dst, &experts(), PAGE).unwrap_err();
    assert!(err.to_string().contains("already exists"), "{err}");
}

/// What alignment costs, stated rather than discovered in production. A page
/// apiece rounds off against a 132 MB expert bank and triples the size of a
/// small tensor, which is the whole reason the caller chooses the set.
#[test]
fn alignment_costs_a_page_for_each_tensor_it_moves() {
    let s = Scratch::new("cost");
    let (src, dst) = (s.join("in"), s.join("out"));
    let small = vec![tensor("a", 16, 1), tensor("b", 16, 2), tensor("c", 16, 3)];
    write_packed(&src, &small);

    let all: BTreeSet<String> = small.iter().map(|t| t.name.to_string()).collect();
    let report = align_checkpoint(&src, &dst, &all, PAGE).unwrap();
    assert!(
        report.overhead_bytes() > 2 * PAGE,
        "three 16-byte tensors on their own pages cost pages, not bytes: {}",
        report.overhead_bytes()
    );
}

/// Everything beside the shards -- config, tokenizer, the shard index -- so the
/// output is a checkpoint someone can load rather than a pile of weights that
/// needs the original directory beside it.
#[test]
fn the_rest_of_the_snapshot_comes_along() {
    let s = Scratch::new("sidecars");
    let (src, dst) = (s.join("in"), s.join("out"));
    write_packed(&src, &fixture());
    std::fs::write(src.join("config.json"), br#"{"model_type":"test"}"#).unwrap();

    align_checkpoint(&src, &dst, &experts(), PAGE).unwrap();
    assert_eq!(
        std::fs::read(dst.join("config.json")).unwrap(),
        br#"{"model_type":"test"}"#
    );
}

fn read_header(path: &Path) -> serde_json::Value {
    let bytes = std::fs::read(path).unwrap();
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    serde_json::from_slice(&bytes[8..8 + len]).unwrap()
}

/// Emit what the shipped tool needs to be run on, so the claim this design
/// rests on -- that what we write is still safetensors -- can be checked
/// against the reference implementation, which no Rust test here links.
///
/// Writes an ordinary checkpoint and a contract declaring a group over its
/// experts. `pie-loader align` turns the pair into an aligned checkpoint, and
/// `safetensors.safe_open` is what says whether the result is one.
///
/// ```text
/// PIE_EMIT_ALIGNED_FIXTURE=<dir> cargo test -p pie-loader \
///     --test aligned_safetensors -- --ignored
/// pie-loader align <dir>/in <dir>/contract.json <dir>/out
/// ```
#[test]
#[ignore]
fn emit_fixture_for_the_reference_reader() {
    use pie_loader::contract::{Expr, GroupContract, ModelContract, TensorContract};
    use pie_loader::types::{DType, Encoding};

    let dir = PathBuf::from(
        std::env::var("PIE_EMIT_ALIGNED_FIXTURE").expect("set PIE_EMIT_ALIGNED_FIXTURE"),
    );
    let _ = std::fs::remove_dir_all(&dir);
    let src = dir.join("in");
    let tensors = vec![
        tensor("experts.0.w", 600, 0x10),
        tensor("experts.1.w", 600, 0x11),
        tensor("experts.2.w", 600, 0x12),
        tensor("experts.3.w", 600, 0x13),
        tensor("norm.weight", 600, 0x20),
    ];
    write_packed(&src, &tensors);

    let contract = ModelContract {
        alignment: 256,
        tensors: Vec::new(),
        groups: vec![GroupContract {
            name: "experts".to_string(),
            arity: 4,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src_indexed("experts.{}.w"),
                vec![600],
                Encoding::Raw(DType::U8),
            )],
        }],
    };
    std::fs::write(
        dir.join("contract.json"),
        serde_json::to_vec_pretty(&contract).unwrap(),
    )
    .unwrap();
    println!("{}", dir.display());
}

/// The whole chain, on a checkpoint that is actually on disk: compile a
/// contract that declares a group, let the plan say which tensors the group
/// reads, and align exactly those.
///
/// The pieces are tested apart -- `groups.rs` checks the classification, the
/// tests above check the rewrite -- but what a caller gets wrong is the joint
/// between them, and nothing above would notice if `streamable_tensors`
/// returned names in a form `align_checkpoint` could not match.
#[test]
fn a_contracts_groups_choose_what_gets_aligned() {
    use pie_loader::checkpoint::align::streamable_tensors;
    use pie_loader::contract::{Expr, GroupContract, ModelContract, TensorContract};
    use pie_loader::plan::{CUDA_TILE_MAP_MASK, StorageTarget, compile};
    use pie_loader::types::{BackendKind, DType, Encoding};

    let s = Scratch::new("endtoend");
    let (src, dst) = (s.join("in"), s.join("out"));
    // Four experts and one tensor outside the group, so "aligned exactly the
    // group" is distinguishable from "aligned everything".
    let tensors = vec![
        tensor("experts.0.w", 512, 0x10),
        tensor("experts.1.w", 512, 0x11),
        tensor("experts.2.w", 512, 0x12),
        tensor("experts.3.w", 512, 0x13),
        tensor("norm.weight", 512, 0x20),
    ];
    write_packed(&src, &tensors);

    let metadata = parse_checkpoint_metadata(&src).unwrap();
    let contract = ModelContract {
        alignment: 256,
        tensors: Vec::new(),
        groups: vec![GroupContract {
            name: "experts".to_string(),
            arity: 4,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src_indexed("experts.{}.w"),
                vec![512],
                Encoding::Raw(DType::U8),
            )],
        }],
    };
    let plan = compile(
        &metadata,
        &contract,
        StorageTarget {
            backend: BackendKind::Cuda,
            tile_map_mask: CUDA_TILE_MAP_MASK,
            ..StorageTarget::default()
        },
    )
    .unwrap();

    let streamable = streamable_tensors(&plan, &metadata);
    assert!(streamable.banded.is_empty(), "{:?}", streamable.banded);
    assert_eq!(streamable.whole.len(), 4, "{:?}", streamable.whole);

    let report = align_checkpoint(&src, &dst, &streamable.whole, PAGE).unwrap();
    assert_eq!(report.aligned(), 4);
    assert!(report.missing.is_empty(), "{:?}", report.missing);

    let aligned = parse_checkpoint_metadata(&dst).unwrap();
    for e in 0..4 {
        let name = format!("experts.{e}.w");
        let t = aligned.tensor_by_name(&name).expect("still present");
        assert_eq!(t.file_offset % PAGE, 0, "{name} is on a page");
    }
    assert_ne!(
        aligned
            .tensor_by_name("norm.weight")
            .expect("still present")
            .file_offset
            % PAGE,
        0,
        "a tensor outside the group is not the group's business"
    );
}
