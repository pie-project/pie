//! The request entry, end to end: a real snapshot on disk, opened through
//! the C ABI, authored and compiled through `pie_loader_compile_model`.
//!
//! This is the boundary a migrated driver boot crosses — facts and policy
//! in, plan out, no contract in sight — so the test drives it exactly as C++
//! would: through the extern entry points, PODs and all.

use std::path::Path;

use pie_loader_capi::checkpoint::PieLoaderCheckpoint;
use pie_loader_capi::entry::{
    PieLoaderStatus, PieLoaderTargetSpec, pie_loader_close_checkpoint, pie_loader_open_checkpoint,
    pie_loader_release, pie_loader_release_diagnostics,
};
use pie_loader_capi::model::{
    PieLoaderFamilyKnobs, PieLoaderModelRequest, pie_loader_compile_model, pie_loader_verify_model,
};
use pie_loader_capi::{PieLoaderBackendKind, PieLoaderBytes, PieLoaderDiagnostics, PieLoaderPlan};

fn bytes(text: &str) -> PieLoaderBytes {
    PieLoaderBytes {
        ptr: text.as_ptr(),
        len: text.len(),
    }
}

/// Write a minimal real safetensors file: a dense llama-shaped decoder, all
/// zeros. Real bytes on disk, because `pie_loader_open_checkpoint` parses
/// the file the way a boot does.
fn write_snapshot(dir: &Path) {
    let (hidden, heads, kv_heads, head_dim, intermediate, vocab) =
        (64i64, 4i64, 2i64, 16i64, 96i64, 128i64);
    let mut tensors: Vec<(String, Vec<i64>)> = vec![
        ("model.embed_tokens.weight".into(), vec![vocab, hidden]),
        ("model.norm.weight".into(), vec![hidden]),
        ("lm_head.weight".into(), vec![vocab, hidden]),
    ];
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        tensors.extend([
            (format!("{p}.input_layernorm.weight"), vec![hidden]),
            (
                format!("{p}.self_attn.q_proj.weight"),
                vec![heads * head_dim, hidden],
            ),
            (
                format!("{p}.self_attn.k_proj.weight"),
                vec![kv_heads * head_dim, hidden],
            ),
            (
                format!("{p}.self_attn.v_proj.weight"),
                vec![kv_heads * head_dim, hidden],
            ),
            (
                format!("{p}.self_attn.o_proj.weight"),
                vec![hidden, heads * head_dim],
            ),
            (format!("{p}.post_attention_layernorm.weight"), vec![hidden]),
            (
                format!("{p}.mlp.gate_proj.weight"),
                vec![intermediate, hidden],
            ),
            (
                format!("{p}.mlp.up_proj.weight"),
                vec![intermediate, hidden],
            ),
            (
                format!("{p}.mlp.down_proj.weight"),
                vec![hidden, intermediate],
            ),
        ]);
    }
    pie_loader::testkit::write_safetensors_fixture(dir, &tensors);
}

/// A `pie.model/1` descriptor for `write_snapshot`'s checkpoint.
///
/// Spelled out here rather than normalized from a `config.json` fixture: this
/// test is about the ABI, and what crosses it is the descriptor. The fields
/// are the ones `ModelFacts::from_descriptor` projects; the rest of the
/// schema is `pie-model-config`'s business and has its own tests.
///
/// A `&'static str` so `bytes()` can borrow it for the whole call, the way a
/// driver borrows the buffer it read the file into.
const LLAMA_DESCRIPTOR: &str = r#"{
    "version": "pie.model/1",
    "model_type": "llama3",
    "quant_method": "",
    "num_hidden_layers": 2,
    "num_experts": 0,
    "head_dim": 16,
    "mamba_n_groups": 0,
    "tie_word_embeddings": true,
    "quant_bits": 0,
    "quant_group_size": 0,
    "num_kv_shared_layers": 0
}"#;

/// The valid request every mutation test starts from: the `llama3` descriptor
/// for `write_snapshot`'s checkpoint, against a plain CUDA target.
fn llama_request(checkpoint: *const PieLoaderCheckpoint) -> PieLoaderModelRequest {
    PieLoaderModelRequest {
        checkpoint,
        target: PieLoaderTargetSpec {
            backend: PieLoaderBackendKind::Cuda as u32,
            tp_rank: 0,
            tp_size: 1,
            max_tile_bytes: 64 << 20,
            preferred_alignment: 256,
            tile_map_mask: pie_loader::plan::CUDA_TILE_MAP_MASK,
            native_mxfp4_moe: false,
            fusion_mask: 0,
            encode_scratch_dtype: pie_loader_capi::PieLoaderDType::BF16 as u32,
            block_scale_rows: 0,
        },
        descriptor: bytes(LLAMA_DESCRIPTOR),
        projections: 0,
        naming: 0,
        runtime_quant: 0,
        moe_request: 0,
        component: 0,
        stream_routed_experts: false,
        knobs: PieLoaderFamilyKnobs {
            qwen35_mtp_int8_lm_head: false,
            nemotron_tp_mamba_sharding: true,
        },
    }
}

#[test]
fn a_llama_request_compiles_to_a_plan_through_the_abi() {
    let dir = std::env::temp_dir().join(format!("pie_capi_model_entry_{}", std::process::id()));
    write_snapshot(&dir);
    let dir_text = dir.to_string_lossy().into_owned();

    let mut checkpoint: *mut PieLoaderCheckpoint = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status =
        unsafe { pie_loader_open_checkpoint(bytes(&dir_text), &mut checkpoint, &mut diags) };
    assert_eq!(status, PieLoaderStatus::Ok, "open failed");
    assert!(!checkpoint.is_null());

    let request = llama_request(checkpoint);

    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let mut resolved_moe = u32::MAX;
    let status =
        unsafe { pie_loader_compile_model(&request, &mut plan, &mut resolved_moe, &mut diags) };
    if status != PieLoaderStatus::Ok {
        let mut listed = String::new();
        if !diags.is_null() {
            let view = unsafe { &*diags };
            for index in 0..view.len {
                let item = unsafe { &*view.items.add(index) };
                let message =
                    unsafe { std::slice::from_raw_parts(item.message.ptr, item.message.len) };
                listed.push_str(&String::from_utf8_lossy(message));
                listed.push('\n');
            }
        }
        panic!("compile_model failed ({status:?}):\n{listed}");
    }
    assert!(!plan.is_null());
    // Auto on a device without native MXFP4 resolves to RoutedDecode (wire 0).
    assert_eq!(resolved_moe, 0, "resolved moe policy was not written back");

    // The plan is the llama contract's: the fused QKV bank exists and the
    // tensor table is the size the author declared.
    let view = unsafe { &*plan };
    assert!(view.tensors.len > 0, "plan declares no tensors");
    let names: Vec<String> = (0..view.tensors.len)
        .map(|index| {
            let tensor = unsafe { &*view.tensors.ptr.add(index) };
            let name = unsafe { std::slice::from_raw_parts(tensor.name.ptr, tensor.name.len) };
            String::from_utf8_lossy(name).into_owned()
        })
        .collect();
    assert!(
        names
            .iter()
            .any(|name| name == "model.layers.0.self_attn.qkv_proj.fused.weight"),
        "the dense join's bank is missing from the plan: {names:?}"
    );

    // The verify entry re-authors from the same request and holds the plan
    // to it — the round trip a migrated driver boot will run.
    let mut verify_diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { pie_loader_verify_model(plan, &request, &mut verify_diags) };
    assert_eq!(
        status,
        PieLoaderStatus::Ok,
        "verify_model rejected its own plan"
    );
    unsafe { pie_loader_release_diagnostics(verify_diags) };

    // An unknown model_type answers with the fallback's name, not a guess.
    const UNKNOWN_DESCRIPTOR: &str = r#"{
        "version": "pie.model/1",
        "model_type": "not_a_model",
        "num_hidden_layers": 2,
        "head_dim": 16
    }"#;
    let mut unknown = request;
    unknown.descriptor = bytes(UNKNOWN_DESCRIPTOR);
    let mut no_plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut unknown_diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe {
        pie_loader_compile_model(
            &unknown,
            &mut no_plan,
            std::ptr::null_mut(),
            &mut unknown_diags,
        )
    };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(no_plan.is_null());
    unsafe { pie_loader_release_diagnostics(unknown_diags) };

    unsafe {
        pie_loader_release(plan);
        pie_loader_release_diagnostics(diags);
        pie_loader_close_checkpoint(checkpoint);
    }
    std::fs::remove_dir_all(dir).ok();
}

/// Join every diagnostic and release the array.
fn drain(diags: *mut PieLoaderDiagnostics) -> String {
    if diags.is_null() {
        return String::new();
    }
    let view = unsafe { &*diags };
    let text = (0..view.len)
        .map(|index| {
            let item = unsafe { &*view.items.add(index) };
            let raw = unsafe { std::slice::from_raw_parts(item.message.ptr, item.message.len) };
            String::from_utf8_lossy(raw).into_owned()
        })
        .collect::<Vec<_>>()
        .join("; ");
    unsafe { pie_loader_release_diagnostics(diags) };
    text
}

/// Compile the valid request after the caller has broken it, and report what
/// came back.
///
/// A real checkpoint is opened even though most cases are expected to be
/// refused before it is read: a request that carried a null handle would be
/// rejected for *that* reason instead, and the test would pass without
/// exercising the field it names. Each test gets its own directory because
/// `write_snapshot` is not atomic and these run in parallel.
fn compile_mutated(
    name: &str,
    mutate: impl FnOnce(&mut PieLoaderModelRequest),
) -> (PieLoaderStatus, String) {
    let dir = std::env::temp_dir().join(format!("pie_capi_{name}_{}", std::process::id()));
    write_snapshot(&dir);
    let dir_text = dir.to_string_lossy().into_owned();
    let mut checkpoint: *mut PieLoaderCheckpoint = std::ptr::null_mut();
    let status = unsafe {
        pie_loader_open_checkpoint(bytes(&dir_text), &mut checkpoint, std::ptr::null_mut())
    };
    assert_eq!(status, PieLoaderStatus::Ok, "open failed");

    let mut request = llama_request(checkpoint);
    mutate(&mut request);
    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status =
        unsafe { pie_loader_compile_model(&request, &mut plan, std::ptr::null_mut(), &mut diags) };
    let message = drain(diags);
    if !plan.is_null() {
        unsafe { pie_loader_release(plan) };
    }
    unsafe { pie_loader_close_checkpoint(checkpoint) };
    std::fs::remove_dir_all(dir).ok();
    (status, message)
}

#[test]
fn out_of_range_backend_is_rejected_not_transmuted() {
    let (status, message) = compile_mutated("backend", |r| r.target.backend = 7);
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(
        message.contains("backend") && message.contains('7'),
        "unexpected diagnostic: {message}"
    );
}

#[test]
fn a_target_claiming_an_unknown_tile_map_transform_is_rejected() {
    // The driver is the authority on what its kernels implement
    // (`architecture.md` §9), so it may claim fewer transforms than the loader
    // knows how to lower. It may not claim one the loader has never heard of:
    // nothing downstream would notice until a kernel dispatch failed at load
    // time.
    let (status, message) = compile_mutated("tile_map", |r| {
        r.target.tile_map_mask = pie_loader::plan::CUDA_TILE_MAP_MASK | (1 << 30);
    });
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(
        message.contains("does not define") && message.contains("cuda"),
        "got: {message}"
    );
}

#[test]
fn a_target_claiming_fewer_transforms_than_the_loader_knows_is_accepted() {
    // The converse: a driver that implements a subset is well-formed. Only the
    // plan's *use* of a transform is checked against the claim, and a BF16
    // llama plan repacks nothing.
    let (status, message) = compile_mutated("narrow_mask", |r| {
        r.target.tile_map_mask =
            pie_loader::plan::CUDA_TILE_MAP_MASK & !pie_loader::plan::TILE_MAP_REPACK;
    });
    assert_eq!(
        status,
        PieLoaderStatus::Ok,
        "a narrower mask is not a malformed request: {message}"
    );
}

#[test]
fn a_target_that_states_no_tile_budget_is_rejected_rather_than_guessed_for() {
    // `architecture.md` §9: a device constant the loader cannot measure has no
    // safe default. The number decides how much scratch every Encode allocates,
    // so guessing it would be a performance contract nobody signed.
    let (status, message) = compile_mutated("tile_budget", |r| r.target.max_tile_bytes = 0);
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(message.contains("max_tile_bytes"), "got: {message}");
}

#[test]
fn a_bad_tp_shape_is_rejected() {
    assert_eq!(
        compile_mutated("tp_zero", |r| r.target.tp_size = 0).0,
        PieLoaderStatus::InvalidRequest
    );
    assert_eq!(
        compile_mutated("tp_rank", |r| {
            r.target.tp_rank = 4;
            r.target.tp_size = 4;
        })
        .0,
        PieLoaderStatus::InvalidRequest
    );
}

#[test]
fn null_arguments_never_dereference() {
    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();

    let status = unsafe {
        pie_loader_compile_model(
            std::ptr::null(),
            &mut plan,
            std::ptr::null_mut(),
            &mut diags,
        )
    };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(plan.is_null());
    assert!(!diags.is_null(), "a null request should still be explained");
    unsafe { pie_loader_release_diagnostics(diags) };

    let dir = std::env::temp_dir().join(format!("pie_capi_nulls_{}", std::process::id()));
    write_snapshot(&dir);
    let dir_text = dir.to_string_lossy().into_owned();
    let mut checkpoint: *mut PieLoaderCheckpoint = std::ptr::null_mut();
    let status = unsafe {
        pie_loader_open_checkpoint(bytes(&dir_text), &mut checkpoint, std::ptr::null_mut())
    };
    assert_eq!(status, PieLoaderStatus::Ok);
    let request = llama_request(checkpoint);

    let status = unsafe {
        pie_loader_compile_model(
            &request,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);

    let status =
        unsafe { pie_loader_verify_model(std::ptr::null(), &request, std::ptr::null_mut()) };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);

    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe {
        pie_loader_open_checkpoint(bytes("/nonexistent"), std::ptr::null_mut(), &mut diags)
    };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    unsafe { pie_loader_release_diagnostics(diags) };

    unsafe { pie_loader_close_checkpoint(checkpoint) };
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn verify_rejects_a_plan_compiled_with_a_different_fusion_setting() {
    // The two settings name different kernel sequences for the same weights.
    // Accepting a fused plan on a driver that has fusion disabled would run
    // something the plan does not describe — and, because the driver caches the
    // materialized artifact, would do so from a cache entry the other setting
    // wrote (`architecture.md` §8.1).
    let dir = std::env::temp_dir().join(format!("pie_capi_fusion_{}", std::process::id()));
    write_snapshot(&dir);
    let dir_text = dir.to_string_lossy().into_owned();
    let mut checkpoint: *mut PieLoaderCheckpoint = std::ptr::null_mut();
    let status = unsafe {
        pie_loader_open_checkpoint(bytes(&dir_text), &mut checkpoint, std::ptr::null_mut())
    };
    assert_eq!(status, PieLoaderStatus::Ok);

    let request = llama_request(checkpoint);
    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status =
        unsafe { pie_loader_compile_model(&request, &mut plan, std::ptr::null_mut(), &mut diags) };
    assert_eq!(
        status,
        PieLoaderStatus::Ok,
        "compile failed: {}",
        drain(diags)
    );

    let mut diverged = request;
    diverged.target.fusion_mask = pie_loader::plan::FUSION_FP8_TO_MXFP4;
    let mut vdiags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { pie_loader_verify_model(plan, &diverged, &mut vdiags) };
    let message = drain(vdiags);
    assert_eq!(status, PieLoaderStatus::ContractViolation);
    assert!(message.contains("fusion_mask"), "got: {message}");

    unsafe {
        pie_loader_release(plan);
        pie_loader_close_checkpoint(checkpoint);
    }
    std::fs::remove_dir_all(dir).ok();
}
