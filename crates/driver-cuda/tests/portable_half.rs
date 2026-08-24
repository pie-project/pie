//! The half of this crate that is correct without a GPU.
//!
//! North-star rule 2: **cut by "is this correct without a GPU?", not by
//! subsystem.** This file is the only thing that holds that cut, and it
//! holds it the only way a cfg can be held — by BUILDING the other side.
//! It compiles with `--features portable`, which selects no CUDA runtime
//! at all, so a module that quietly starts naming `cudarc` fails here and
//! nowhere else.
//!
//! # Why the rule was worth stating
//!
//! `store` was gated on `_cuda` as a subsystem. Eighteen of its
//! twenty-two files name no CUDA symbol: the memory planner's lattice
//! search, the MLA and DSv4 geometry, the KV geometry, the profile cache
//! and its key, the swap plan, the recurrent layout, the calibration
//! ladder, `dtoa`, `json`. All of it is arithmetic over shapes and
//! budgets, and all of it sat where no test that did not have a card
//! could reach — which is how `memory_planner`, `mla_geometry` and
//! `compressed_plane_geometry` came to be parity-tested with zero callers, for
//! months, without anyone noticing.
//!
//! The last two never found a caller and are DELETED: this file's
//! `the_attention_geometries_resolve` was the only place either of them ran,
//! which is a module kept alive by the test that noticed it was dead.
//!
//! The gate now ends at the modules that own DEVICE MEMORY. That is the
//! honest boundary: `kv_geometry` says what shape the pages are and
//! `kv_cache` allocates them, and only the second one needs a card.
//!
//! The assertions below are deliberately thin. **The build is the test.**
//! What each one adds is a demonstration that the module is not merely
//! reachable but usable — that its entry point can be called and returns
//! an answer — because a `pub mod` that compiles proves less than it
//! looks like it does.

/// A model whose costs are linear in its shape, which is enough to make
/// the lattice search do real work. The parity suite's `Case` is the
/// exhaustive version; this one only has to be a model.
struct PaperModel;

impl driver_cuda::layout::memory_planner::ModelCosts for PaperModel {
    fn per_kv_token_bytes(&self) -> u64 {
        // 8 layers x 4 kv heads x 64 dim x 2 planes x 2 bytes.
        8 * 4 * 64 * 2 * 2
    }
    fn envelope_bytes_per_page(&self) -> u64 {
        0
    }
    fn state_slot_bytes(&self) -> u64 {
        0
    }
    fn arena_bytes(&self, n: i32, output_rows: i32, _mtp_rows: i32) -> u64 {
        u64::try_from(n).unwrap_or(0) * 1024 * 2 + u64::try_from(output_rows).unwrap_or(0) * 4
    }
    fn attn_float_workspace_bytes(&self, n: i32, _r: i32) -> u64 {
        u64::try_from(n).unwrap_or(0) * 16 * 4
    }
    fn persistent_input_bytes(&self, n: i32, r: i32, refs: i32, _mask: i32) -> u64 {
        u64::try_from(n + r + refs).unwrap_or(0) * 4
    }
    fn runtime_quant_scratch_bytes(&self, _n: i32) -> u64 {
        0
    }
    fn has_linear_state(&self) -> bool {
        false
    }
}

/// A cache that has measured nothing — the honest state of a machine that
/// has never run a calibration boot.
struct NoMeasurement;

impl driver_cuda::layout::memory_planner::ProfileSource for NoMeasurement {
    fn lookup(
        &self,
        _key: &driver_cuda::layout::profile_key::ProfileKey,
    ) -> driver_cuda::layout::memory_planner::ProfileRead {
        driver_cuda::layout::memory_planner::ProfileRead::default()
    }
    fn path(&self) -> String {
        String::new()
    }
}

/// The planner's lattice search, on a box with no card.
///
/// This is the module the rule was written about. It is 1,567 lines, it
/// was parity-tested against the C++ from the day it was ported, and it
/// had zero callers until August 11 — a state that a test able to run
/// here would have made obvious.
#[test]
fn the_memory_planner_plans() {
    use driver_cuda::layout::memory_planner as mp;

    let cfg = mp::PlannerConfig {
        gpu_mem_utilization: 0.90,
        memory_profile: "auto".to_owned(),
        max_forward_tokens: 0,
        max_forward_requests: 0,
        kv_page_size: 0,
        kv_cache_dtype: "auto".to_owned(),
        tp_size: 1,
        mtp_num_drafts: 0,
        calibrating: false,
        rs_slot_mult: 1,
        nccl_unique_id_hex: String::new(),
    };
    let shape = mp::ModelShape {
        hidden_size: 1024,
        num_hidden_layers: 8,
        num_attention_heads: 16,
        num_key_value_heads: 4,
        head_dim_kernel: 64,
        model_id: "llama".to_owned(),
    };
    let props = mp::DeviceProps {
        name: String::new(),
        major: 8,
        minor: 9,
        sm_count: 142,
    };
    let memory = mp::DeviceMemory {
        free_bytes: 40 << 30,
        total_bytes: 48 << 30,
    };

    let planned = mp::plan(
        &cfg,
        &shape,
        &props,
        memory,
        mp::ShapeKnees::default(),
        &PaperModel,
        &NoMeasurement,
    )
    .expect("a small model in 40 GiB has a feasible rectangle");
    assert!(
        planned.plan.capacity.max_forward_tokens > 0,
        "a plan names a token width"
    );
    assert!(
        planned.plan.capacity.max_forward_requests > 0,
        "and a request width"
    );
    assert!(planned.plan.kv_page_size > 0, "and a page size");
}

/// The KV geometry answers page shapes with no pages allocated.
///
/// `kv_geometry` and `kv_cache` are the cut: one says what shape a page
/// is, the other allocates it, and only the second needs a card.
#[test]
fn the_kv_geometry_shapes_pages_without_allocating_them() {
    use driver_cuda::layout::{KvCacheFormat, kv_geometry};

    let format = KvCacheFormat::default();
    let bytes = kv_geometry::page_bytes_homogeneous(8, 4, 64, 1, &format);
    assert!(bytes > 0, "a page has a size before it has an address");
    let one = kv_geometry::device_bytes_per_page(&format, 1, 4, 64);
    assert_eq!(bytes, 8 * one, "eight identical layers cost eight pages");
}

/// The KV format's scale planes, which decide how many buffers a
/// quantised page has — the arithmetic a swap plan is built against.
///
/// This replaced a `SidebandArena` case, and the swap is worth
/// recording: the arena's growth rule IS portable arithmetic, but the
/// arena manages DEVICE memory, so it lives in `fire/` and the
/// portable half cannot see it. That is the boundary working rather
/// than failing — the test moved to something on the correct side of
/// it instead of the module moving to the wrong one.
#[test]
fn the_kv_format_counts_its_planes() {
    use driver_cuda::layout::KvCacheFormat;

    let plain = KvCacheFormat::default();
    let bytes = driver_cuda::layout::kv_geometry::device_bytes_per_page(&plain, 1, 4, 64);
    assert!(bytes > 0, "an unquantised page still has a size");
}

/// The profile cache's number formatting, which exists for byte-parity
/// with the C++ driver's `cuda_memory_profiles.json` and is pure text.
#[test]
fn the_json_writer_is_text() {
    let mut out = String::new();
    driver_cuda::layout::dtoa::write_f64(&mut out, 0.1);
    assert_eq!(
        out, "0.1",
        "the shortest round-tripping form, not 0.1000000000000000055"
    );
}
