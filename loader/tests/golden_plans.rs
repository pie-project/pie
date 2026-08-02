//! Golden plans.
//!
//! The tests around this one check that individual pieces of the compiler do
//! what they say. This one checks the thing a user actually receives: the whole
//! plan, byte for byte, for a checkpoint that exercises a real architecture's
//! passes at a real tensor-parallel rank.
//!
//! Its value is not that any particular byte is *right* — the other tests argue
//! that. It is that a change to the plan cannot happen quietly. A refactor that
//! was meant to preserve behaviour and did not shows up here as a diff a human
//! reads, and a change that was meant to alter the plan shows up as a diff a
//! human approves.
//!
//! Regenerate after an intended change:
//!
//! ```text
//! UPDATE_GOLDEN=1 cargo test -p pie-loader --test golden_plans
//! ```
//!
//! Two fields are normalized away before comparison. `compiler_version` is a
//! hash of the compiler's own source, so it changes on every edit and would
//! make every golden stale for no reason. File paths are made relative, because
//! a golden that embedded a checkout path would only pass on one machine.

use std::path::PathBuf;

use pie_loader::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
use pie_loader::contract::ModelContract;
use pie_loader::ffi::contract::read_contract;
use pie_loader::ffi::view::verify_marshalled;
use pie_loader::plan::compile as compile_load_plan;
use pie_loader::plan::{
    CUDA_TILE_MAP_MASK, FUSION_FP8_TO_MXFP4, HOST_TILE_MAP_MASK, LoadPlan, StorageTarget,
    compiler_version,
};
use pie_loader::testkit::contract_writer::write_contract;
use pie_loader::types::{
    BackendKind, CheckpointFormat, DType, Encoding, FileId, QuantScheme, QuantSpec, TensorId,
};
use pie_loader::verify::ContractView;

// ── the checkpoints ─────────────────────────────────────────────────

/// Accumulates tensors at packed, ascending file offsets, the way a real
/// safetensors shard is laid out. Offsets matter: they are what the plan's
/// source extents are expressed against, so a fixture that placed everything at
/// zero would hide any offset arithmetic the compiler gets wrong.
struct Checkpoint {
    tensors: Vec<RawTensor>,
    offset: u64,
}

impl Checkpoint {
    fn new() -> Self {
        Self {
            tensors: Vec::new(),
            offset: 0,
        }
    }

    fn push(&mut self, name: &str, shape: &[i64], encoding: Encoding) -> &mut Self {
        let elements: i64 = shape.iter().product();
        let span_bytes = match &encoding {
            Encoding::Raw(dtype) => u64::try_from(elements).unwrap() * dtype.bytes(),
            Encoding::Quant(spec) => {
                u64::try_from(elements).unwrap() * u64::from(spec.bits_per_element) / 8
            }
        };
        self.tensors.push(RawTensor {
            id: TensorId(self.tensors.len() as u32),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: self.offset,
            span_bytes,
            shape: shape.to_vec(),
            encoding,
        });
        self.offset += span_bytes;
        self
    }

    /// Write a file of the right size and point the metadata at it.
    ///
    /// Verification checks that every source extent lands inside the file it
    /// names, so a fixture that named a file it had not written would fail for
    /// a reason that has nothing to do with the plan. Writing it also means the
    /// offsets in the golden are offsets into something real.
    fn finish(self, name: &str) -> CheckpointMetadata {
        // Several tests share a checkpoint and run in parallel, so the write
        // has to be atomic: a reader that caught a plain `fs::write` between
        // truncating and filling would see a short file and report it as a
        // plan that reads past the end.
        let path = std::env::temp_dir().join(format!(
            "pie_loader_golden_{}_{}.safetensors",
            name,
            std::process::id()
        ));
        if std::fs::metadata(&path).map(|meta| meta.len()).ok() != Some(self.offset) {
            let staging = path.with_extension(format!("{:?}.partial", std::thread::current().id()));
            std::fs::write(&staging, vec![0u8; self.offset as usize])
                .expect("write fixture checkpoint");
            std::fs::rename(&staging, &path).expect("publish fixture checkpoint");
        }
        CheckpointMetadata {
            files: vec![CheckpointFile {
                id: FileId(0),
                path: path.to_string_lossy().into_owned(),
                size_bytes: self.offset,
                format: CheckpointFormat::Safetensors,
            }],
            tensors: self.tensors,
        }
    }
}

fn bf16() -> Encoding {
    Encoding::Raw(DType::BF16)
}

fn quant(scheme: QuantScheme) -> Encoding {
    Encoding::Quant(QuantSpec {
        scheme,
        logical_dtype: DType::BF16,
        bits_per_element: scheme.default_bits(),
        group_size: scheme.default_group_size(),
        channel_axis: None,
    })
}

/// A dense decoder: the q/k/v fusion path, grouped-query attention (so the k/v
/// projections shard differently from q), a gated MLP, and a tied-shape
/// embedding and head.
fn llama_checkpoint() -> CheckpointMetadata {
    let (hidden, heads, kv_heads, head_dim, intermediate, vocab) = (256, 8, 2, 32, 704, 512);
    let mut ck = Checkpoint::new();
    ck.push("model.embed_tokens.weight", &[vocab, hidden], bf16());
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        ck.push(&format!("{p}.input_layernorm.weight"), &[hidden], bf16());
        ck.push(
            &format!("{p}.self_attn.q_proj.weight"),
            &[heads * head_dim, hidden],
            bf16(),
        );
        ck.push(
            &format!("{p}.self_attn.k_proj.weight"),
            &[kv_heads * head_dim, hidden],
            bf16(),
        );
        ck.push(
            &format!("{p}.self_attn.v_proj.weight"),
            &[kv_heads * head_dim, hidden],
            bf16(),
        );
        ck.push(
            &format!("{p}.self_attn.o_proj.weight"),
            &[hidden, heads * head_dim],
            bf16(),
        );
        ck.push(
            &format!("{p}.post_attention_layernorm.weight"),
            &[hidden],
            bf16(),
        );
        ck.push(
            &format!("{p}.mlp.gate_proj.weight"),
            &[intermediate, hidden],
            bf16(),
        );
        ck.push(
            &format!("{p}.mlp.up_proj.weight"),
            &[intermediate, hidden],
            bf16(),
        );
        ck.push(
            &format!("{p}.mlp.down_proj.weight"),
            &[hidden, intermediate],
            bf16(),
        );
    }
    ck.push("model.norm.weight", &[hidden], bf16());
    ck.push("lm_head.weight", &[vocab, hidden], bf16());
    ck.finish("llama")
}

/// A mixture of experts stored one expert per tensor, which the loader stacks
/// into a single packed weight per layer.
fn qwen3_moe_checkpoint() -> CheckpointMetadata {
    let (hidden, experts, intermediate, heads, kv_heads, head_dim) = (128, 4, 256, 4, 2, 32);
    let mut ck = Checkpoint::new();
    ck.push("model.embed_tokens.weight", &[256, hidden], bf16());
    let p = "model.layers.0";
    ck.push(&format!("{p}.input_layernorm.weight"), &[hidden], bf16());
    ck.push(
        &format!("{p}.self_attn.q_proj.weight"),
        &[heads * head_dim, hidden],
        bf16(),
    );
    ck.push(
        &format!("{p}.self_attn.k_proj.weight"),
        &[kv_heads * head_dim, hidden],
        bf16(),
    );
    ck.push(
        &format!("{p}.self_attn.v_proj.weight"),
        &[kv_heads * head_dim, hidden],
        bf16(),
    );
    ck.push(
        &format!("{p}.self_attn.o_proj.weight"),
        &[hidden, heads * head_dim],
        bf16(),
    );
    ck.push(
        &format!("{p}.post_attention_layernorm.weight"),
        &[hidden],
        bf16(),
    );
    ck.push(&format!("{p}.mlp.gate.weight"), &[experts, hidden], bf16());
    for expert in 0..experts {
        let e = format!("{p}.mlp.experts.{expert}");
        ck.push(
            &format!("{e}.gate_proj.weight"),
            &[intermediate, hidden],
            bf16(),
        );
        ck.push(
            &format!("{e}.up_proj.weight"),
            &[intermediate, hidden],
            bf16(),
        );
        ck.push(
            &format!("{e}.down_proj.weight"),
            &[hidden, intermediate],
            bf16(),
        );
    }
    ck.push("model.norm.weight", &[hidden], bf16());
    ck.push("lm_head.weight", &[256, hidden], bf16());
    ck.finish("qwen3_moe")
}

/// MXFP4 experts stored as a packed block-scaled pair, which is the input to
/// the repack tile maps.
fn gpt_oss_checkpoint() -> CheckpointMetadata {
    let (hidden, experts, intermediate) = (128, 2, 256);
    let mut ck = Checkpoint::new();
    let p = "model.layers.0";
    ck.push(&format!("{p}.input_layernorm.weight"), &[hidden], bf16());
    ck.push(
        &format!("{p}.mlp.experts.gate_up_proj_blocks"),
        &[experts, 2 * intermediate, hidden / 32, 16],
        Encoding::Raw(DType::U8),
    );
    ck.push(
        &format!("{p}.mlp.experts.gate_up_proj_scales"),
        &[experts, 2 * intermediate, hidden / 32],
        Encoding::Raw(DType::U8),
    );
    ck.push(
        &format!("{p}.mlp.experts.down_proj_blocks"),
        &[experts, hidden, intermediate / 32, 16],
        Encoding::Raw(DType::U8),
    );
    ck.push(
        &format!("{p}.mlp.experts.down_proj_scales"),
        &[experts, hidden, intermediate / 32],
        Encoding::Raw(DType::U8),
    );
    ck.push("model.norm.weight", &[hidden], bf16());
    ck.finish("gpt_oss")
}

/// GPT-OSS at the ABI where the driver repacks into Marlin's layout.
///
/// The only fixture that reaches [`Expr::Repack`]. The biases used to reach it
/// too, through a gather layout that existed because the algebra could not say
/// "every other row of this rank's band"; [`Expr::Stride`] says it, so they are
/// ordinary copies and the layout is gone. Kept separate from
/// [`gpt_oss_checkpoint`] because that one carries the decoder around the
/// experts and none of the repack path.
///
/// `intermediate` is the *global* size. The stored contracts were authored
/// against a fixed local size of 64, so the sharded one needs 128 here.
fn gpt_oss_native_checkpoint(intermediate: i64) -> CheckpointMetadata {
    let (hidden, experts) = (64, 2);
    let mut ck = Checkpoint::new();
    let p = "model.layers.0.mlp.experts";
    let u8s = Encoding::Raw(DType::U8);
    ck.push(
        &format!("{p}.gate_up_proj_blocks"),
        &[experts, 2 * intermediate, hidden / 32, 16],
        u8s.clone(),
    );
    ck.push(
        &format!("{p}.gate_up_proj_scales"),
        &[experts, 2 * intermediate, hidden / 32],
        u8s.clone(),
    );
    ck.push(
        &format!("{p}.gate_up_proj_bias"),
        &[experts, 2 * intermediate],
        bf16(),
    );
    ck.push(
        &format!("{p}.down_proj_blocks"),
        &[experts, hidden, intermediate / 32, 16],
        u8s.clone(),
    );
    ck.push(
        &format!("{p}.down_proj_scales"),
        &[experts, hidden, intermediate / 32],
        u8s,
    );
    ck.push(&format!("{p}.down_proj_bias"), &[experts, hidden], bf16());
    // The name keys the temp file, and the two sizes must not collide.
    ck.finish(&format!("gpt_oss_native_{intermediate}"))
}

/// An AWQ-quantized dense decoder: the same passes as `llama_checkpoint`, but
/// over a packed 4-bit encoding, so the plan's offsets are constrained by quant
/// group boundaries rather than element boundaries.
fn awq_checkpoint() -> CheckpointMetadata {
    let (hidden, heads, kv_heads, head_dim, intermediate) = (256, 8, 2, 32, 512);
    let mut ck = Checkpoint::new();
    ck.push("model.embed_tokens.weight", &[256, hidden], bf16());
    let p = "model.layers.0";
    ck.push(&format!("{p}.input_layernorm.weight"), &[hidden], bf16());
    for (name, shape) in [
        ("self_attn.q_proj", vec![heads * head_dim, hidden]),
        ("self_attn.k_proj", vec![kv_heads * head_dim, hidden]),
        ("self_attn.v_proj", vec![kv_heads * head_dim, hidden]),
        ("self_attn.o_proj", vec![hidden, heads * head_dim]),
        ("mlp.gate_proj", vec![intermediate, hidden]),
        ("mlp.up_proj", vec![intermediate, hidden]),
        ("mlp.down_proj", vec![hidden, intermediate]),
    ] {
        ck.push(
            &format!("{p}.{name}.weight"),
            &shape,
            quant(QuantScheme::AwqInt4),
        );
    }
    ck.push(
        &format!("{p}.post_attention_layernorm.weight"),
        &[hidden],
        bf16(),
    );
    ck.push("model.norm.weight", &[hidden], bf16());
    ck.finish("awq")
}

// ── the harness ─────────────────────────────────────────────────────

fn target(backend: BackendKind, tp_rank: u32, tp_size: u32) -> StorageTarget {
    StorageTarget {
        backend,
        tp_rank,
        tp_size,
        max_tile_bytes: 64 << 20,
        preferred_alignment: 256,
        tile_map_mask: match backend {
            BackendKind::Cuda => CUDA_TILE_MAP_MASK,
            _ => HOST_TILE_MAP_MASK,
        },
        native_mxfp4_moe: false,
        fusion_mask: if backend == BackendKind::Cuda {
            FUSION_FP8_TO_MXFP4
        } else {
            0
        },
        encode_scratch_dtype: DType::BF16,
        block_scale_rows: 128,
    }
}

/// The contract this golden compiles, as a stored fixture.
///
/// A golden is a test of the *compiler*, and the compiler's input is a
/// contract. Storing the contract instead of re-deriving it from a model type
/// is what lets the loader stop knowing what a model is: after `arch/` is gone
/// there is no function in this crate that could produce one, and there should
/// not be — authorship is the driver's job. Freezing them here keeps every
/// construct the real families use (the MXFP4 repacks, the fused QKV `Concat`, the
/// `Out` aliases into a bank, the strided GPTQ slices) under test, expressed as
/// what they actually are: programs over a checkpoint.
fn contract_fixture(name: &str) -> ModelContract {
    let path = contract_path(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "{name}: cannot read {}: {err}\n\
             A new golden needs a contract next to it. Author one the way a \
             driver does — `driver/*/src/model/*/*_contract.hpp` \
             can dump the real thing under PIE_TEST_CONTRACT_DUMP — or write the \
             expression out by hand.",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("{name}: parsing the contract: {err}"))
}

fn contract_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/contracts")
        .join(format!("{name}.json"))
}

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.json"))
}

/// The driver's boot-log and debug renderings, exercised on every golden plan.
///
/// The driver has no other test for these: it prints whatever it is handed.
/// Checking that the histogram accounts for every instruction is what catches an
/// instruction variant added without a name for it.
fn check_stats_render(name: &str, plan: &LoadPlan) {
    let summary = pie_loader::dump::describe(plan);
    assert!(
        summary.contains(&format!("tensors={}", plan.tensors.len()))
            && summary.contains(&format!("instrs={}", plan.instrs.len())),
        "{name}: the boot-log line lost a field: {summary}"
    );

    let rendered = pie_loader::dump::plan_stats_json(plan);
    let stats: serde_json::Value = serde_json::from_str(&rendered)
        .unwrap_or_else(|err| panic!("{name}: stats are not JSON: {err}"));
    let counted: u64 = stats["instruction_kinds"]
        .as_object()
        .unwrap_or_else(|| panic!("{name}: stats carry no instruction histogram"))
        .values()
        .map(|n| n.as_u64().unwrap_or_default())
        .sum();
    assert_eq!(
        counted,
        plan.instrs.len() as u64,
        "{name}: the instruction histogram does not account for every instruction"
    );
}

/// Compile, check the plan against the contract it was compiled from, and
/// compare the result to the stored one.
///
/// Verification runs here and not only in its own tests because a golden that
/// was regenerated from a broken compiler would otherwise lock the breakage in:
/// the file would faithfully record whatever came out. Requiring the plan to
/// verify before it is allowed to become golden is what stops that.
fn check(name: &str, metadata: &CheckpointMetadata, target: StorageTarget) {
    let contract = contract_fixture(name);
    check_contract_survives_the_ffi(name, &contract);
    let plan = compile_load_plan(metadata, &contract, target)
        .unwrap_or_else(|err| panic!("{name}: compiling failed: {err}"));

    if let Err(violations) = verify_marshalled(&plan, Some(&ContractView::of(&contract))) {
        let listed: Vec<String> = violations.iter().map(ToString::to_string).collect();
        panic!(
            "{name}: the plan does not honour its contract:\n  {}",
            listed.join("\n  ")
        );
    }

    replay(name, &plan, metadata);
    check_stats_render(name, &plan);

    let fresh = render(&plan);
    let path = golden_path(name);
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).expect("create golden directory");
        std::fs::write(&path, &fresh).expect("write golden");
        return;
    }
    let stored = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "{name}: cannot read {}: {err}\n\
             If this plan is new, regenerate with UPDATE_GOLDEN=1.",
            path.display()
        )
    });
    if stored == fresh {
        return;
    }
    let first = stored
        .lines()
        .zip(fresh.lines())
        .enumerate()
        .find(|(_, (a, b))| a != b)
        .map(|(line, (a, b))| format!("line {}:\n  golden: {a}\n  fresh:  {b}", line + 1))
        .unwrap_or_else(|| {
            format!(
                "golden has {} lines, fresh has {}",
                stored.lines().count(),
                fresh.lines().count()
            )
        });
    panic!(
        "{name}: the compiled plan changed.\n{first}\n\n\
         If the change is intended, regenerate with UPDATE_GOLDEN=1 and review \
         the diff to {}.",
        path.display()
    );
}

/// Round-trip the contract through the POD form the C++ builder emits.
///
/// This is the safety net for moving authorship out of `arch/` and into the
/// drivers. The migration's claim is that a family's contract can be written by
/// hand in C++ and produce the identical plan; that claim is only checkable if
/// the FFI representation is known to be lossless for every construct the real
/// families use. Asserting it here, on every golden, means each family covers
/// its own constructs — the MXFP4 repacks, the fused QKV `Concat`, the `Out`
/// aliases into a bank, the strided GPTQ slices — instead of on a synthetic
/// expression that happens to use the ones someone thought of.
///
/// Equality is on the whole `ModelContract`, not on the plan it compiles to. A
/// weaker check would pass for a lossy encoding whose loss happened not to
/// change this particular plan.
fn check_contract_survives_the_ffi(name: &str, contract: &ModelContract) {
    let owned = write_contract(contract);
    let read = unsafe { read_contract(&owned.view()) }
        .unwrap_or_else(|err| panic!("{name}: the contract does not read back: {err}"));
    assert_eq!(
        &read,
        contract,
        "{name}: the contract changed crossing the FFI ({} nodes)",
        owned.node_count()
    );
}

/// Execute the plan, when this backend's plan is one the host executor can run.
///
/// A plan can be self-consistent, honour its contract, and match its golden
/// while still being unexecutable — the golden records the plan, not what
/// happens when something tries to follow it. That is not a hypothetical
/// distinction: the sharded `ExtentWrite` path was rejected outright by
/// `host.rs` because it compared the source and destination *dimensions*
/// rather than their byte counts, and nothing here noticed until
/// `pie-loader replay` was pointed at a real checkpoint at tp 1/2.
fn replay(name: &str, plan: &LoadPlan, metadata: &CheckpointMetadata) {
    if plan.target.tile_map_mask & !HOST_TILE_MAP_MASK != 0 {
        return;
    }
    let snapshot = PathBuf::from(&metadata.files[0].path);
    let storage =
        pie_loader::testkit::host_executor::execute_plan(plan, snapshot.parent().unwrap())
            .unwrap_or_else(|err| panic!("{name}: the plan does not execute: {err}"));
    for tensor in &plan.tensors {
        let materialized = storage
            .tensors
            .get(&tensor.name)
            .unwrap_or_else(|| panic!("{name}: '{}' was never materialized", tensor.name));
        let expected = pie_loader::types::encoding_nbytes(&tensor.shape, &tensor.encoding)
            .unwrap_or_else(|| panic!("{name}: '{}' has no size", tensor.name));
        assert_eq!(
            materialized.len() as u64,
            expected,
            "{name}: '{}' materialized {} bytes, but its declared type is {} bytes",
            tensor.name,
            materialized.len(),
            expected
        );
    }
}

/// Strip the two fields that are true of the machine rather than of the plan.
fn render(plan: &LoadPlan) -> String {
    let mut plan = plan.clone();
    assert_eq!(
        plan.compiler_version,
        compiler_version(),
        "a freshly compiled plan should carry this compiler's version"
    );
    plan.compiler_version = 0;
    // The fixture checkpoint lives in a temp directory under a name carrying
    // this process's pid, so the path is not a property of the plan at all.
    // The file *table* still is: its length, ids and sizes stay.
    for file in &mut plan.files {
        file.path = format!("checkpoint-{}.safetensors", file.id.0);
    }
    let mut text = serde_json::to_string_pretty(&plan).expect("serialize plan");
    text.push('\n');
    text
}

// ── the goldens ─────────────────────────────────────────────────────

#[test]
fn llama_dense_cuda() {
    check(
        "llama_dense_cuda",
        &llama_checkpoint(),
        target(BackendKind::Cuda, 0, 1),
    );
}

/// Rank 1 of 2, not rank 0: rank 0 of a sharded load often coincides with the
/// unsharded plan for the leading slice of every tensor, so it is the weakest
/// rank to pin.
#[test]
fn llama_dense_cuda_tp1_of_2() {
    check(
        "llama_dense_cuda_tp1_of_2",
        &llama_checkpoint(),
        target(BackendKind::Cuda, 1, 2),
    );
}

/// The same model on a backend with no tile-map support, which forces the
/// transforms CUDA folds into kernels to appear as explicit instructions.
#[test]
fn llama_dense_host() {
    check(
        "llama_dense_host",
        &llama_checkpoint(),
        target(BackendKind::Unknown, 0, 1),
    );
}

/// The sharded path on the one backend that can execute it here, so the plan is
/// followed and not merely recorded.
#[test]
fn llama_dense_host_tp1_of_2() {
    check(
        "llama_dense_host_tp1_of_2",
        &llama_checkpoint(),
        target(BackendKind::Unknown, 1, 2),
    );
}

#[test]
fn qwen3_moe_host() {
    check(
        "qwen3_moe_host",
        &qwen3_moe_checkpoint(),
        target(BackendKind::Unknown, 0, 1),
    );
}

#[test]
fn qwen3_moe_host_tp1_of_2() {
    check(
        "qwen3_moe_host_tp1_of_2",
        &qwen3_moe_checkpoint(),
        target(BackendKind::Unknown, 1, 2),
    );
}

#[test]
fn awq_dense_host_tp1_of_2() {
    check(
        "awq_dense_host_tp1_of_2",
        &awq_checkpoint(),
        target(BackendKind::Unknown, 1, 2),
    );
}

#[test]
fn qwen3_moe_cuda() {
    check(
        "qwen3_moe_cuda",
        &qwen3_moe_checkpoint(),
        target(BackendKind::Cuda, 0, 1),
    );
}

#[test]
fn qwen3_moe_cuda_tp1_of_2() {
    check(
        "qwen3_moe_cuda_tp1_of_2",
        &qwen3_moe_checkpoint(),
        target(BackendKind::Cuda, 1, 2),
    );
}

#[test]
fn gpt_oss_mxfp4_cuda() {
    check(
        "gpt_oss_mxfp4_cuda",
        &gpt_oss_checkpoint(),
        target(BackendKind::Cuda, 0, 1),
    );
}

#[test]
fn gpt_oss_mxfp4_cuda_native_gemm() {
    let mut target = target(BackendKind::Cuda, 0, 1);
    target.native_mxfp4_moe = true;
    check(
        "gpt_oss_mxfp4_cuda_native_gemm",
        &gpt_oss_checkpoint(),
        target,
    );
}

/// The repack path, which no other golden reaches: both layouts the plan can
/// carry, `MarlinMxfp4Weight` and `MarlinMxfp4Scale`, appear in this contract.
#[test]
fn gpt_oss_native_mxfp4() {
    let mut target = target(BackendKind::Cuda, 0, 1);
    target.native_mxfp4_moe = true;
    check(
        "gpt_oss_native_mxfp4",
        &gpt_oss_native_checkpoint(64),
        target,
    );
}

/// The same at rank 1 of 2.
///
/// GPT-OSS is the only family that resolves the rank *inside* the escape hatch,
/// as `RepackSpec::source_row_offset`, so it is the only family whose contract
/// differs between one rank and another. Moving that split back out into an
/// `Expr::Shard` must leave these bytes untouched.
#[test]
fn gpt_oss_native_mxfp4_tp1_of_2() {
    let mut target = target(BackendKind::Cuda, 1, 2);
    target.native_mxfp4_moe = true;
    check(
        "gpt_oss_native_mxfp4_tp1_of_2",
        &gpt_oss_native_checkpoint(128),
        target,
    );
}

#[test]
fn awq_dense_cuda() {
    check(
        "awq_dense_cuda",
        &awq_checkpoint(),
        target(BackendKind::Cuda, 0, 1),
    );
}

/// Sharding a 4-bit encoding is where an element/byte confusion shows up: the
/// shard boundary has to land on a quantization group, not just on a row.
#[test]
fn awq_dense_cuda_tp1_of_2() {
    check(
        "awq_dense_cuda_tp1_of_2",
        &awq_checkpoint(),
        target(BackendKind::Cuda, 1, 2),
    );
}

/// Runtime quantization rewrites the encoding of the tensors it applies to and
/// leaves the rest alone, so the plan has to carry both.
#[test]
fn llama_dense_cuda_runtime_fp8() {
    check(
        "llama_dense_cuda_runtime_fp8",
        &llama_checkpoint(),
        target(BackendKind::Cuda, 0, 1),
    );
}
