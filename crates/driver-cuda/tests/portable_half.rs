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
//! `dsv4_geometry` came to be parity-tested with zero callers, for
//! months, without anyone noticing.
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

impl driver_cuda::store::memory_planner::ModelCosts for PaperModel {
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

impl driver_cuda::store::memory_planner::ProfileSource for NoMeasurement {
    fn lookup(
        &self,
        _key: &driver_cuda::store::profile_key::ProfileKey,
    ) -> driver_cuda::store::memory_planner::ProfileRead {
        driver_cuda::store::memory_planner::ProfileRead::default()
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
    use driver_cuda::store::memory_planner as mp;

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
        model_type: "llama".to_owned(),
    };
    let props = mp::DeviceProps { name: String::new(), major: 8, minor: 9, sm_count: 142 };
    let memory = mp::DeviceMemory { free_bytes: 40 << 30, total_bytes: 48 << 30 };

    let planned = mp::plan(
        &cfg,
        &shape,
        &props,
        memory,
        mp::Family::Generic,
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

/// The geometry modules resolve their shapes.
///
/// `mla_geometry` and `dsv4_geometry` are the other two the rule names.
/// They are still waiting on a forward path — there is no MLA arm in the
/// executor — so this is the ONLY place either of them runs.
#[test]
fn the_attention_geometries_resolve() {
    use driver_cuda::store::{dsv4_geometry, mla_geometry};

    // DeepSeek's numbers: 512 compressed KV, 64 rope.
    let mla = mla_geometry::MlaGeometry::new(8, 128, 16, 512, 64, driver_cuda::DType::Bf16)
        .expect("a deepseek-shaped MLA resolves");
    assert!(
        mla.ckv_layer_bytes() > 0,
        "an MLA layer has a size before it has an address"
    );
    assert_eq!(mla.kv_lora_rank() + mla.qk_rope_head_dim(), 576, "512 + 64");

    let widths = dsv4_geometry::layer_widths(&[1, 2, 4], 3, 128);
    assert_eq!(widths.len(), 3, "one width per layer");
    let per_token = dsv4_geometry::compress_bytes_per_token(&[1, 2, 4], 128);
    assert!(per_token > 0, "a compressed token costs something");
}

/// The KV geometry answers page shapes with no pages allocated.
///
/// `kv_geometry` and `kv_cache` are the cut: one says what shape a page
/// is, the other allocates it, and only the second needs a card.
#[test]
fn the_kv_geometry_shapes_pages_without_allocating_them() {
    use driver_cuda::store::{KvCacheFormat, kv_geometry};

    let format = KvCacheFormat::default();
    let bytes = kv_geometry::page_bytes_homogeneous(8, 4, 64, 1, &format);
    assert!(bytes > 0, "a page has a size before it has an address");
    let one = kv_geometry::device_bytes_per_page(&format, 1, 4, 64);
    assert_eq!(bytes, 8 * one, "eight identical layers cost eight pages");
}

/// The sideband arena's growth rule, without the allocator that serves it.
///
/// `DeviceMemory` is a trait for exactly this reason and the module says
/// so — *"the growth path frees the old block before it learns whether a
/// replacement exists, and that ordering is only testable against an
/// allocator that can be told to fail on cue"*. What this adds is a
/// BUILD in which that reason is load-bearing rather than merely stated.
#[test]
fn the_arena_grows_without_a_device() {
    use driver_cuda::model::sideband_arena::{DeviceMemory, Region, SidebandArena};

    #[derive(Default)]
    struct Paper {
        next: usize,
        freed: usize,
    }
    impl DeviceMemory for Paper {
        fn alloc(&mut self, bytes: usize) -> Option<*mut std::ffi::c_void> {
            self.next += bytes.max(1);
            Some(self.next as *mut std::ffi::c_void)
        }
        fn free(&mut self, _ptr: *mut std::ffi::c_void) {
            self.freed += 1;
        }
        fn synchronize(&mut self) -> bool {
            true
        }
    }

    let mut paper = Paper::default();
    let mut arena = SidebandArena::new();
    let first = arena
        .acquire(&mut paper, Region::Mask, 1024)
        .expect("a fresh arena grows");
    assert!(!first.is_null());
    arena.release(Region::Mask);
    arena
        .acquire(&mut paper, Region::Mask, 8 << 20)
        .expect("and grows again past its floor");
    assert!(paper.freed > 0, "growing retires the old block");
    arena.destroy(&mut paper);
}

/// The profile cache's number formatting, which exists for byte-parity
/// with the C++ driver's `cuda_memory_profiles.json` and is pure text.
#[test]
fn the_json_writer_is_text() {
    let mut out = String::new();
    driver_cuda::store::dtoa::write_f64(&mut out, 0.1);
    assert_eq!(out, "0.1", "the shortest round-tripping form, not 0.1000000000000000055");
}
