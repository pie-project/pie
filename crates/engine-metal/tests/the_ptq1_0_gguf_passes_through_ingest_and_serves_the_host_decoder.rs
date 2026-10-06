#![cfg(target_vendor = "apple")]

//! M1c: a REAL PTQ1_0 (ternary, `g128_t3_f16_n`, ggml type 143) tensor loads into
//! pie as a servable `Ptq1_0` bank by NATIVE PASS-THROUGH — the ternary block bytes
//! are kept verbatim (no decode to fp, no re-encode to MLX affine) — and the M1b
//! Metal kernel serves that ingested bank bit-exact against the M1a host decoder.
//!
//! The loop this closes is file -> ingest -> bank -> kernel, on real data:
//!   1. Build a GGUF whose tensors are ggml type 143, filled with REAL 28-byte
//!      blocks lifted from Ternary-Bonsai-2-27B-PTQ1_0.gguf (the frozen M1a
//!      fixture). This is the actual on-disk shape M0 confirmed.
//!   2. Run it through the REAL ingest: `parse_metadata` recognises type 143 as
//!      `Ptq1_0`, `materialize_contract` classifies it PASS-THROUGH (not decoded),
//!      and `convert` writes the served `.zt` artifact.
//!   3. Declare it: the same `checkpoint_dsl::Builder::read` that `qwen_3`'s GGUF
//!      import calls, over a `Dtype::Ptq1_0` weight, must produce a PURE COPY
//!      contract (a bare `Expr::Src`, encoding `Quant(Ptq1_0)`, no scales plane) —
//!      proving the single-inline-plane declaration re-encodes nothing.
//!   4. Materialise the bank through the checkpoint executor and assert its bytes
//!      are BYTE-IDENTICAL to the raw GGUF blocks (pass-through mutates nothing)
//!      and that a row is exactly `ceil(K/128) * 28` bytes.
//!   5. Serve that ingested bank through the M1b kernel (`ptq1_0_qmv`) with one-hot
//!      probes and assert every decoded weight equals the M1a host decoder
//!      (`checkpoint::codec::ptq1_0::decode_block`) bit-for-bit.
//!
//! M1c is weights-only: the served output is NOT yet the full-model answer (the
//! Hadamard-rotation undo is M3). What is proven here is decode correctness of the
//! ingested ternary bank versus the host reference — the file->kernel byte path.

use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::codec::ptq1_0::decode_block;
use checkpoint::contract::Expr;
use checkpoint::contract::materialize::materialize_contract;
use checkpoint::executor::Execution;
use checkpoint::file::read::parse_metadata;
use checkpoint::file::write::Writer;
use checkpoint::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};
use checkpoint::types::{BackendKind, Encoding, QuantScheme, TensorDecl, Visibility};

use checkpoint_dsl::Builder;
use model_dsl::{Dtype, ParamSource, Platform, Shard, Weight};
use model_ir::ParamLayout;

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::encode::{Arg, Encode, Fire, Grid};
use kernels_metal::linear::quant;
use model_ir::Dtype as IrDtype;

// The M1a oracle: `ForkBlock { tensor, block_index, bytes: [u8;28], .. }` +
// `FORK_BLOCKS`, real blocks from the GGUF plus synthetic edge cases.
include!("../../checkpoint/src/codec/ptq1_0_fixture.rs");

const PTQ1_0_TYPE_ID: u32 = 143;
const BLOCK: usize = 128;
const BLOCK_BYTES: usize = 28;
const PTQ1_0_FILE: &str = "linear/quant_ptq1_0.metal";
const QMV_GROUP: [u32; 3] = [32, 2, 1];

/// One declared tensor: a name, its logical `[rows, k]` shape, and the raw ternary
/// block bytes (rows * k/128 blocks, real fixture blocks cycled through).
struct RealTensor {
    name: &'static str,
    rows: usize,
    k: usize,
    bytes: Vec<u8>,
}

fn real_tensor(name: &'static str, rows: usize, k: usize, first_block: usize) -> RealTensor {
    assert_eq!(k % BLOCK, 0, "k must be whole PTQ1_0 blocks");
    let blocks_per_row = k / BLOCK;
    let mut bytes = Vec::with_capacity(rows * blocks_per_row * BLOCK_BYTES);
    let total = FORK_BLOCKS.len();
    for r in 0..rows {
        for c in 0..blocks_per_row {
            let at = (first_block + r * blocks_per_row + c) % total;
            bytes.extend_from_slice(&FORK_BLOCKS[at].bytes);
        }
    }
    RealTensor {
        name,
        rows,
        k,
        bytes,
    }
}

/// A minimal GGUF v3 writer: `(name, logical shape, ggml type id, payload)`. The
/// shape is written fastest-dim-first (ggml order), which the projection reverses
/// back to the logical `[rows, k]` this test states.
fn gguf(tensors: &[(&str, Vec<u64>, u32, Vec<u8>)]) -> Vec<u8> {
    const ALIGN: usize = 32;
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&0u64.to_le_bytes());
    let mut data = Vec::new();
    for (name, shape, type_id, payload) in tensors {
        head.extend_from_slice(&(name.len() as u64).to_le_bytes());
        head.extend_from_slice(name.as_bytes());
        head.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dim in shape.iter().rev() {
            head.extend_from_slice(&dim.to_le_bytes());
        }
        head.extend_from_slice(&type_id.to_le_bytes());
        head.extend_from_slice(&(data.len() as u64).to_le_bytes());
        data.extend_from_slice(payload);
        while !data.len().is_multiple_of(ALIGN) {
            data.push(0);
        }
    }
    while !head.len().is_multiple_of(ALIGN) {
        head.push(0);
    }
    head.extend_from_slice(&data);
    head
}

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("m1c_ptq1_0_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Read a tensor's stored bytes straight from a parsed artifact's file extent.
fn bytes_at(metadata: &checkpoint::file::Metadata, name: &str) -> Vec<u8> {
    use std::io::{Read, Seek, SeekFrom};
    let tensor = metadata.tensor_by_name(name).expect("tensor present");
    let file = metadata
        .files
        .iter()
        .find(|f| f.id == tensor.file_id)
        .expect("file present");
    let mut handle = std::fs::File::open(&file.path).unwrap();
    handle.seek(SeekFrom::Start(tensor.file_offset)).unwrap();
    let mut out = vec![0u8; tensor.span_bytes as usize];
    handle.read_exact(&mut out).unwrap();
    out
}

/// The `pie model import` pass-through: normalise the GGUF into a `.zt` artifact.
/// A self-contained block (PTQ1_0) is carried across as its raw bytes.
fn convert(source_dir: &Path, metadata: &checkpoint::file::Metadata, out: &Path) {
    let materialization = materialize_contract(metadata).unwrap();
    let decoded = if materialization.contract.tensors.is_empty() {
        Default::default()
    } else {
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };
        let plan = checkpoint::plan::compile(metadata, &materialization.contract, target).unwrap();
        let storage = Execution::new(&plan, source_dir).run().unwrap();
        plan.tensors
            .iter()
            .filter(|decl| decl.visibility.is_public())
            .map(|decl| {
                (
                    decl.name.clone(),
                    (decl.clone(), storage.tensors[&decl.name].clone()),
                )
            })
            .collect::<BTreeMap<_, _>>()
    };

    let mut entries: Vec<&str> = materialization
        .decoded
        .iter()
        .chain(materialization.passthrough.iter())
        .map(String::as_str)
        .collect();
    entries.sort_unstable();

    let mut writer = Writer::create(out, &BTreeMap::new()).unwrap();
    for name in entries {
        match decoded.get(name) {
            Some((decl, bytes)) => writer.add_tensor(decl, bytes).unwrap(),
            None => {
                let raw = metadata.tensor_by_name(name).unwrap();
                let file = metadata.files.iter().find(|f| f.id == raw.file_id).unwrap();
                let all = std::fs::read(source_dir.join(&file.path)).unwrap();
                let start = raw.file_offset as usize;
                let bytes = all[start..start + raw.span_bytes as usize].to_vec();
                let decl = TensorDecl {
                    id: raw.id,
                    name: raw.name.clone(),
                    shape: raw.shape.clone(),
                    encoding: raw.encoding.clone(),
                    alignment: 1,
                    visibility: Visibility::default(),
                };
                writer.begin_tensor(&decl, raw.span_bytes).unwrap();
                writer.write(&bytes).unwrap();
                writer.end_tensor().unwrap();
            }
        }
    }
    writer.finish().unwrap();
}

fn ptq1_0_weight(t: &RealTensor) -> Weight {
    Weight {
        name: t.name.to_string(),
        shape: vec![t.rows as u64, t.k as u64],
        dtype: Dtype::Ptq1_0,
        shard: Shard::Replicated,
        source: ParamSource::Checkpoint,
        layout: ParamLayout::Natural,
    }
}

/// The fixture blocks are the fork's own oracle: the host decoder reproduces the
/// fork's `dequantize_row_ptq1_0` bit-for-bit. Re-pin it so the serve check below
/// is anchored to that reference and not just to itself.
fn the_fixture_blocks_are_the_fork_oracle() {
    let mut real = 0usize;
    for b in FORK_BLOCKS {
        let got = decode_block(&b.bytes);
        for (i, want) in b.expect_bits.iter().enumerate() {
            assert_eq!(
                got[i].to_bits(),
                *want,
                "{} block {} element {i}: host decoder disagrees with the fork",
                b.tensor,
                b.block_index,
            );
        }
        if !b.tensor.starts_with("synthetic:") {
            real += 1;
        }
    }
    assert!(
        real >= 20,
        "expected the spread of real GGUF blocks, got {real}"
    );
}

#[test]
fn the_ptq1_0_gguf_passes_through_ingest_and_serves_the_host_decoder_every_case() {
    the_fixture_blocks_are_the_fork_oracle();

    // A few real tensors under their GGUF names (attn_q, ffn_down, token_embd),
    // each a whole number of 128-wide ternary blocks.
    let tensors = [
        real_tensor("blk.0.attn_q.weight", 16, 256, 0),
        real_tensor("blk.0.ffn_down.weight", 8, 384, 7),
        real_tensor("token_embd.weight", 12, 128, 3),
    ];

    let dir = tmpdir("loop");
    let gguf_path = dir.join("bonsai.gguf");
    std::fs::write(
        &gguf_path,
        gguf(
            &tensors
                .iter()
                .map(|t| {
                    (
                        t.name,
                        vec![t.rows as u64, t.k as u64],
                        PTQ1_0_TYPE_ID,
                        t.bytes.clone(),
                    )
                })
                .collect::<Vec<_>>(),
        ),
    )
    .unwrap();

    // --- Ingest recognises ggml type 143 as PTQ1_0 and keeps it as a block ------
    let metadata = parse_metadata(&gguf_path).unwrap();
    for t in &tensors {
        let stored = metadata.tensor_by_name(t.name).expect("named");
        match &stored.encoding {
            Encoding::Quant(spec) => {
                assert_eq!(spec.scheme, QuantScheme::Ptq1_0, "{}", t.name);
                assert_eq!(spec.group_size, 128, "{}", t.name);
                assert!(
                    spec.scheme.is_self_contained(),
                    "{}: PTQ1_0 is a single-inline-plane block",
                    t.name
                );
            }
            other => panic!("{}: recognised as {other:?}, not a PTQ1_0 block", t.name),
        }
        // Byte-count of the on-disk tensor is exactly rows * ceil(K/128) * 28.
        let want = (t.rows * (t.k / BLOCK) * BLOCK_BYTES) as u64;
        assert_eq!(stored.span_bytes, want, "{}: block-padded size", t.name);
    }

    // The converter classifies every ternary tensor PASS-THROUGH, never decoded.
    let materialization = materialize_contract(&metadata).unwrap();
    for t in &tensors {
        assert!(
            materialization.passthrough.iter().any(|n| n == t.name),
            "{} should pass through, not be decoded ({:?} were decoded)",
            t.name,
            materialization.decoded,
        );
    }

    // --- Convert to the served .zt and assert byte-identical pass-through -------
    let zt_path = dir.join("bonsai.zt");
    convert(&dir, &metadata, &zt_path);
    let artifact = parse_metadata(&zt_path).unwrap();
    for t in &tensors {
        assert_eq!(
            bytes_at(&artifact, t.name),
            t.bytes,
            "{}: ingest must keep the ternary blocks byte-for-byte",
            t.name
        );
    }

    // --- The declared load contract is a PURE COPY (no decode, no re-encode) ----
    // This is the very path `qwen_3::import_from_gguf` drives via `b.read`.
    let src = ztensor::Source::open(&zt_path).expect("open the served artifact");
    let mut builder = Builder::new(&src, 1, Platform::Metal);
    for t in &tensors {
        builder.read(&ptq1_0_weight(t), t.name).unwrap();
    }
    let contract = builder.build();
    for t in &tensors {
        let entry = contract
            .tensors
            .iter()
            .find(|c| c.name == t.name)
            .unwrap_or_else(|| panic!("{} declared", t.name));
        match &entry.expr {
            Expr::Src(from) => assert_eq!(from, t.name, "{}", t.name),
            other => panic!(
                "{}: pass-through must be a bare source read, got {other:?} \
                 (a decode/re-encode would wrap it)",
                t.name
            ),
        }
        match &entry.encoding {
            Encoding::Quant(spec) => assert_eq!(spec.scheme, QuantScheme::Ptq1_0, "{}", t.name),
            other => panic!("{}: declared {other:?}, not a PTQ1_0 bank", t.name),
        }
        assert!(
            entry.scales.is_none(),
            "{}: PTQ1_0 is single-inline-plane and declares no separate scales",
            t.name
        );
    }

    // --- Materialise the served bank through the executor -----------------------
    // A Metal-backed plan must COMPILE: this proves the bind gate now accepts a
    // stored PTQ1_0 block on Metal (the M1b decode-in-dot kernel reads it), where
    // it would reject every other self-contained scheme as "has to be decoded".
    checkpoint::plan::compile(
        &artifact,
        &contract,
        StorageTarget::for_backend(BackendKind::Metal, 0, 1),
    )
    .expect("a Metal target binds the PTQ1_0 block rather than demanding a decode");
    // The host executor produces the exact bank bytes the device receives.
    let plan = checkpoint::plan::compile(&artifact, &contract, StorageTarget::default()).unwrap();
    let storage = Execution::new(&plan, &dir).run().unwrap();
    for t in &tensors {
        assert_eq!(
            &storage.tensors[t.name], &t.bytes,
            "{}: the materialised bank is the raw ternary blocks, unmutated",
            t.name
        );
    }

    // --- Serve the ingested bank through the M1b kernel = the host decoder ------
    let Ok(device) = Context::bind() else {
        eprintln!("m1c: no Metal device; ingest+declare asserted, serve skipped");
        std::fs::remove_dir_all(&dir).ok();
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    eprintln!("m1c device: {}", device.name());

    let mut served = 0usize;
    for t in &tensors {
        let bank_bytes = &storage.tensors[t.name];
        served += serve_and_check(&device, &handles, &pipelines, t, bank_bytes);
    }
    eprintln!(
        "m1c: {served} elements served from the ingested banks match the host \
         decoder bit-exact, across {} real tensors",
        tensors.len()
    );
    assert!(served > 0);

    std::fs::remove_dir_all(&dir).ok();
}

/// Fire the ternary decode-in-dot kernel over an ingested bank with the KxK
/// identity activation, so `y[i*rows + r]` is the single decoded weight
/// `W[r][i]`. Compare every element to the host `decode_block` of the same
/// ingested bytes. Returns the element count checked.
fn serve_and_check(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    t: &RealTensor,
    bank_bytes: &[u8],
) -> usize {
    let n = t.rows;
    let k = t.k;
    let m = k; // one one-hot probe per column
    let row_bytes = (k / BLOCK) * BLOCK_BYTES;
    assert_eq!(bank_bytes.len(), n * row_bytes);

    // Host reference: decode each row's blocks to k f32 weights (natural order).
    let mut host = vec![0.0f32; n * k];
    for r in 0..n {
        for c in 0..(k / BLOCK) {
            let at = r * row_bytes + c * BLOCK_BYTES;
            let vals = decode_block(&bank_bytes[at..at + BLOCK_BYTES]);
            host[r * k + c * BLOCK..r * k + (c + 1) * BLOCK].copy_from_slice(&vals);
        }
    }

    // Identity activation (bf16): probe i is e_i.
    let mut x = vec![0u8; m * k * 2];
    for i in 0..m {
        let bits = 1.0f32.to_bits();
        x[(i * k + i) * 2] = (bits >> 16) as u8;
        x[(i * k + i) * 2 + 1] = (bits >> 24) as u8;
    }

    let mut codes_b = Buffer::zeroed(device, bank_bytes.len() as u64).expect("codes");
    codes_b.write(0, bank_bytes).expect("write codes");
    let mut x_b = Buffer::zeroed(device, x.len() as u64).expect("x");
    x_b.write(0, &x).expect("write x");
    let y_b = Buffer::zeroed(device, (m * n * 4) as u64).expect("y");

    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("handle");
    let (hc, hx, hy) = (bind(&codes_b), bind(&x_b), bind(&y_b));

    let (mi, ni, ki) = (m as i32, n as i32, k as i32);
    let frame = device.frame().expect("frame");
    let sink = Sink::new(device, &frame, pipelines, handles);
    sink.fire(
        Fire::at(PTQ1_0_FILE, "ptq1_0_qmv_bfloat16_f32").apply(Grid::of(
            quant::qmv_grid("m1c.readback", mi, ni).expect("grid"),
            QMV_GROUP,
        )),
        &[
            Tensor::new(hc, ni.unsigned_abs(), ki.unsigned_abs(), IrDtype::Ptq1_0).arg(),
            Tensor::new(hx, mi.unsigned_abs(), ki.unsigned_abs(), IrDtype::Bf16).arg(),
            Tensor::new(hy, mi.unsigned_abs(), ni.unsigned_abs(), IrDtype::F32).arg_mut(),
            ki.arg(),
            ni.arg(),
        ],
    )
    .expect("launch");
    frame.commit().expect("commit");

    let y = handles.read(hy, (m * n * 4) as u64).expect("read y");
    let mut checked = 0usize;
    for r in 0..n {
        for i in 0..k {
            let b = &y[(i * n + r) * 4..(i * n + r) * 4 + 4];
            let got = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            let want = host[r * k + i].to_bits();
            assert_eq!(
                got, want,
                "{} row {r} col {i}: kernel {got:#010x} != host decoder {want:#010x}",
                t.name
            );
            checked += 1;
        }
    }
    checked
}
