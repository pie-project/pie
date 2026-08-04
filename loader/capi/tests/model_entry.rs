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
    PieLoaderFamilyKnobs, PieLoaderModelFactsView, PieLoaderModelRequest, pie_loader_compile_model,
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

    let mut header = String::from("{");
    let mut offset = 0u64;
    for (index, (name, shape)) in tensors.iter().enumerate() {
        let elements: i64 = shape.iter().product();
        let nbytes = elements as u64 * 2;
        if index > 0 {
            header.push(',');
        }
        let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{offset},{}]}}",
            dims.join(","),
            offset + nbytes
        ));
        offset += nbytes;
    }
    header.push('}');

    let mut file = Vec::with_capacity(8 + header.len() + offset as usize);
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    file.extend(std::iter::repeat_n(0u8, offset as usize));
    std::fs::create_dir_all(dir).expect("create snapshot dir");
    std::fs::write(dir.join("model.safetensors"), file).expect("write snapshot");
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

    let request = PieLoaderModelRequest {
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
        facts: PieLoaderModelFactsView {
            model_type: bytes("llama3"),
            quant_method: bytes(""),
            num_hidden_layers: 2,
            num_experts: 0,
            head_dim: 16,
            mamba_groups: 0,
        },
        projections: 0,
        naming: 0,
        runtime_quant: 0,
        moe_request: 0,
        component: 0,
        stream_routed_experts: false,
        knobs: PieLoaderFamilyKnobs {
            glm5_moe_gate_up_swapped: true,
            qwen35_fused_gdn_projection: false,
            qwen35_mtp_int8_lm_head: false,
            qwen35_moe_gate_up_swapped: true,
            qwen35_fused_shared_scalar_gate: false,
            kimi_k3_moe_gate_up_swapped: false,
            kimi_moe_gate_up_swapped: false,
            nemotron_tp_mamba_sharding: true,
        },
    };

    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { pie_loader_compile_model(&request, &mut plan, &mut diags) };
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

    // An unknown model_type answers with the fallback's name, not a guess.
    let mut unknown = request;
    unknown.facts.model_type = bytes("not_a_model");
    let mut no_plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut unknown_diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { pie_loader_compile_model(&unknown, &mut no_plan, &mut unknown_diags) };
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
