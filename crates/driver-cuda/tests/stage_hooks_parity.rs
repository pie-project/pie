//! Behavioural parity with the C++ stage-hook dispatch — gate-stage-hooks.
//!
//! The oracle in `tests/oracle/stage_hooks/` includes the REAL
//! `stage_hooks.hpp` (the header is the whole implementation) and drives
//! `invoke_stage_hook` with a recording execute. This test replays the
//! same script and requires the transcripts to be byte-identical.
//!
//! Run `tests/oracle/stage_hooks/run.sh` to regenerate
//! [`GOLDEN_FNV1A64`]. The pinned value is the **C++'s** hash.
//!
//! The C++ `context` pointer maps to the trait impl's `self`: the
//! recorder prints `ctx=ctx` because it IS the context, and the defaults
//! row prints `context=null` iff `execute` is `None` — one fact, two
//! spellings, as the module docs of the port explain.

use std::cell::RefCell;
use std::ffi::c_void;
use std::fmt::Write as _;

use driver_cuda::gpu::fire::attn_score::AttentionObservation;
use driver_cuda::gpu::fire::stage_hooks::{
    StageHookExecute, StageHookPoint, StageHookSideband, StageHooks, default_hooks,
    invoke_stage_hook,
};

/// FNV-1a 64 of the C++ oracle's transcript.
const GOLDEN_FNV1A64: u64 = 0x2bd3c1e03f7fa545;

/// Rows the transcript must contain, so a truncated sweep cannot pass.
const GOLDEN_ROWS: usize = 20;

const SEP: char = '\u{1f}';

fn obs_at<'a>(tag: usize) -> AttentionObservation<'a> {
    AttentionObservation {
        kv_page_size: Some(16),
        kv_page_indices_d: tag as *const u32,
        kv_page_indptr_d: std::ptr::null(),
        kv_last_page_lens_d: std::ptr::null(),
        qo_indptr_h: None,
        kv_page_indptr_h: None,
        kv_last_page_lens_h: None,
        num_requests: 0,
        total_tokens: 0,
    }
}

/// Identity by role: the oracle names pointers, never prints values. The
/// observations are told apart by the tag their first device pointer
/// carries.
fn who_obs(o: Option<&AttentionObservation<'_>>) -> &'static str {
    match o {
        None => "null",
        Some(o) if o.kv_page_indices_d as usize == 0x300 => "obsA",
        Some(o) if o.kv_page_indices_d as usize == 0x400 => "obsB",
        _ => "unknown",
    }
}

fn who_ptr(p: *const c_void) -> &'static str {
    match p as usize {
        0 => "null",
        0x200 => "query",
        _ => "unknown",
    }
}

struct Recorder {
    out: RefCell<String>,
    case: RefCell<String>,
}

impl Recorder {
    fn note(&self, body: &str) {
        let case = self.case.borrow().clone();
        let _ = writeln!(self.out.borrow_mut(), "{case}{SEP}{body}");
    }
}

impl StageHookExecute for Recorder {
    fn execute(
        &self,
        point: StageHookPoint,
        query_data: *const c_void,
        query_rows: u32,
        query_columns: u32,
        layer: u32,
        stream: *mut c_void,
        query_is_f32: bool,
        sideband: &StageHookSideband<'_>,
    ) {
        self.note(&format!(
            "execute ctx=ctx point={} q={} rows={query_rows} cols={query_columns} \
             layer={layer} stream={} f32={} obs={} scores={} sink={}",
            point as u8,
            who_ptr(query_data),
            if stream.is_null() { "s0" } else { "s?" },
            u8::from(query_is_f32),
            who_obs(sideband.observation),
            if sideband.scores.is_some() { "unknown" } else { "null" },
            if sideband.mask_sink.is_null() { "null" } else { "unknown" },
        ));
    }
}

fn transcript() -> String {
    let r = Recorder { out: RefCell::new(String::new()), case: RefCell::new(String::new()) };
    let query = 0x200 as *const c_void;
    let obs_a = obs_at(0x300);
    let obs_b = obs_at(0x400);

    // a. Defaults, field by field — the C++ member initializers, with the
    //    two folded members reading through their Rust homes.
    *r.case.borrow_mut() = "a-defaults".into();
    {
        let h = default_hooks();
        r.note(&format!("context={}", if h.execute.is_none() { "null" } else { "set" }));
        r.note(&format!("wants_attn_score={}", u8::from(h.wants_attn_score)));
        r.note(&format!("attn_score_window={}", h.attn_score_window));
        r.note(&format!("wants_page_mask={}", u8::from(h.wants_page_mask)));
        r.note(&format!("hook_free_prefix_rows={}", h.hook_free_prefix_rows));
        r.note(&format!("hook_rows_k={}", h.hook_rows_k));
        // The arena travels beside the hooks in the port; a fresh hook set
        // has none wired either way.
        r.note("sideband_arena=null");
        r.note(&format!("observation={}", who_obs(h.observation)));
        r.note(&format!("execute={}", if h.execute.is_none() { "null" } else { "set" }));
        r.note(&format!(
            "prepare_replay={}",
            if h.prepare_replay.is_none() { "null" } else { "set" }
        ));
        r.note(&format!(
            "verify_replay_capture={}",
            if h.prepare_replay.is_none() { "null" } else { "set" }
        ));
        let s = StageHookSideband::default();
        r.note(&format!(
            "sideband obs={} scores={} sink={}",
            who_obs(s.observation),
            if s.scores.is_some() { "unknown" } else { "null" },
            if s.mask_sink.is_null() { "null" } else { "unknown" },
        ));
    }

    // b. The no-op arms record nothing.
    *r.case.borrow_mut() = "b-noop".into();
    {
        invoke_stage_hook(
            None,
            StageHookPoint::OnAttn,
            query,
            4,
            64,
            0,
            std::ptr::null_mut(),
            false,
            StageHookSideband::default(),
        );
        let silent = default_hooks();
        invoke_stage_hook(
            Some(&silent),
            StageHookPoint::OnAttn,
            query,
            4,
            64,
            0,
            std::ptr::null_mut(),
            false,
            StageHookSideband::default(),
        );
        r.note("done");
    }

    // c. Argument forwarding, sideband defaulting included.
    *r.case.borrow_mut() = "c-forward".into();
    {
        let h = StageHooks {
            observation: Some(&obs_a),
            execute: Some(&r),
            ..default_hooks()
        };
        invoke_stage_hook(
            Some(&h),
            StageHookPoint::OnAttnProj,
            query,
            7,
            128,
            3,
            std::ptr::null_mut(),
            false,
            StageHookSideband::default(),
        );
        invoke_stage_hook(
            Some(&h),
            StageHookPoint::OnAttn,
            query,
            1,
            64,
            9,
            std::ptr::null_mut(),
            true,
            StageHookSideband { observation: Some(&obs_b), ..StageHookSideband::default() },
        );
        invoke_stage_hook(
            Some(&h),
            StageHookPoint::OnAttn,
            query,
            2,
            32,
            0,
            std::ptr::null_mut(),
            false,
            StageHookSideband::default(),
        );
    }

    // d. The Tier-2 truncation guard, at the boundary on both sides.
    *r.case.borrow_mut() = "d-truncation".into();
    {
        let mut h = StageHooks {
            observation: Some(&obs_a),
            execute: Some(&r),
            ..default_hooks()
        };
        h.hook_rows_k = 5;
        for layer in [0u32, 4, 5, 6] {
            invoke_stage_hook(
                Some(&h),
                StageHookPoint::OnAttn,
                query,
                1,
                64,
                layer,
                std::ptr::null_mut(),
                false,
                StageHookSideband::default(),
            );
        }
        r.note("swept");
        h.hook_rows_k = 0;
        invoke_stage_hook(
            Some(&h),
            StageHookPoint::OnAttn,
            query,
            1,
            64,
            0,
            std::ptr::null_mut(),
            false,
            StageHookSideband::default(),
        );
        r.note("k0-swept");
    }

    r.out.into_inner()
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn the_port_reproduces_the_cpp_transcript() {
    let text = transcript();
    let rows = text.lines().count();
    assert_eq!(rows, GOLDEN_ROWS, "row count diverged — script shape changed");
    let hash = fnv1a64(text.as_bytes());
    if hash != GOLDEN_FNV1A64 {
        let path = std::env::temp_dir().join("stage_hooks_rust_transcript.txt");
        std::fs::write(&path, &text).ok();
        panic!(
            "transcript hash 0x{hash:016x} != golden 0x{GOLDEN_FNV1A64:016x}; \
             rust transcript dumped to {}",
            path.display()
        );
    }
}
