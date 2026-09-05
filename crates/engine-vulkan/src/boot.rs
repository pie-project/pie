use crate::api::{ContractFor, DeviceBoot, Vulkan};

pub const DEFAULT_GPU_MEM_UTILIZATION: f64 = 0.9;

pub fn open(config_bytes: &[u8], contract_for: ContractFor) -> Result<Vulkan, String> {
    let doc: toml::Table = std::str::from_utf8(config_bytes)
        .map_err(|error| format!("the vulkan boot config is not utf-8: {error}"))?
        .parse()
        .map_err(|error| format!("the vulkan boot config is not TOML: {error}"))?;
    tuning(&doc);
    Ok(Vulkan::new(
        DeviceBoot {
            device_index: device_index(&doc),
            gpu_mem_utilization: gpu_mem_utilization(&doc),
            validation: validation(&doc),
            pipeline_cache: pipeline_cache(&doc),
        },
        contract_for,
    ))
}

fn table(doc: &toml::Table) -> Option<&toml::Table> {
    doc.get("vulkan").and_then(toml::Value::as_table)
}

fn device_index(doc: &toml::Table) -> u32 {
    table(doc)
        .and_then(|t| t.get("device_index"))
        .and_then(toml::Value::as_integer)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(0)
}

fn gpu_mem_utilization(doc: &toml::Table) -> f64 {
    table(doc)
        .and_then(|t| t.get("gpu_mem_utilization"))
        .and_then(toml::Value::as_float)
        .filter(|fraction| fraction.is_finite() && *fraction > 0.0 && *fraction <= 1.0)
        .unwrap_or(DEFAULT_GPU_MEM_UTILIZATION)
}

fn validation(doc: &toml::Table) -> bool {
    table(doc)
        .and_then(|t| t.get("validation"))
        .and_then(toml::Value::as_bool)
        .unwrap_or(false)
}

fn pipeline_cache(doc: &toml::Table) -> Option<std::path::PathBuf> {
    table(doc)
        .and_then(|t| t.get("pipeline_cache"))
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(std::path::PathBuf::from)
}

fn tuning(doc: &toml::Table) {
    let Some(table) = table(doc)
        .and_then(|vulkan| vulkan.get("tuning"))
        .and_then(toml::Value::as_table)
    else {
        return;
    };
    let int = |key: &str| {
        table
            .get(key)
            .and_then(toml::Value::as_integer)
            .and_then(|v| u32::try_from(v).ok())
    };
    let flag = |key: &str| table.get(key).and_then(toml::Value::as_bool);

    kernels_vulkan::tuning::override_with(kernels_vulkan::tuning::Overrides {
        qmm_min_batch: int("qmm_min_batch"),
        qmm_min_batch_moe: int("qmm_min_batch_moe"),
        qmm_min_batch_emulated: int("qmm_min_batch_emulated"),
        qmm_bn_crossover_tg: int("qmm_bn_crossover_tg"),
        moe_tile_mid_per: int("moe_tile_mid_per"),
        moe_tile_wide_per: int("moe_tile_wide_per"),
        fp16_qmm: flag("fp16_qmm"),
        sdpa_tile_min_rows_per_request: int("sdpa_tile_min_rows_per_request"),
        sdpa_mma: flag("sdpa_mma"),
        gdn_scan_lanes: int("gdn_scan_lanes"),
        gdn_scan_rows: int("gdn_scan_rows"),
        moe_batch_min_per_expert: int("moe_batch_min_per_expert"),
        qmv_rows_max: int("qmv_rows_max"),
        qmv_rows_packs: int("qmv_rows_packs"),
        stream_rows_per_cut: int("stream_rows_per_cut"),
        sdpa_split_groups: int("sdpa_split_groups"),
        sdpa_split_min_keys: int("sdpa_split_min_keys"),
    });
}
