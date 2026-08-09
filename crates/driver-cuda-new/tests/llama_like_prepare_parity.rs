//! Behavioural parity with the C++ `prepare_llama_like_decode_plan` —
//! slice B of gate-plan-state.
//!
//! The oracle in `tests/oracle/llama_like_prepare/` compiles the real
//! `llama_like.cpp` (forward body discarded by `--gc-sections`, real
//! workspaces and a real KvCache linked underneath) and replaces only the
//! flashinfer planner boundary with recorders that dump every argument —
//! the CSR arrays BY CONTENT, because the rebasing arithmetic is the thing
//! under test. This test replays the same cases against the port and
//! requires the transcripts to be byte-identical.
//!
//! Run `tests/oracle/llama_like_prepare/run.sh` to regenerate
//! [`GOLDEN_FNV1A64`]. The pinned value is the **C++'s** hash.
//!
//! The C++ caches its three env gates in function-local statics, so the
//! oracle sweeps them across processes; the port carries them as
//! [`PrepareGates`], so this side sweeps them as values. Plan-id counters
//! reset per process there and per mode-section here, for the same rows.

use std::fmt::Write as _;

use driver_cuda_new::launch::AttentionWorkspaceView;
use driver_cuda_new::model::config::HfConfig;
use driver_cuda_new::model::llama_like::{
    KvGeom, LlamaLikeForwardCfg, LlamaLikePlanState, PlannerOps, PrepareGates,
    PrepareParams, prepare_llama_like_decode_plan,
};

/// FNV-1a 64 of the C++ oracle's transcript.
const GOLDEN_FNV1A64: u64 = 0x9378763aa9425336;

/// Rows the transcript must contain, so a truncated sweep cannot pass.
const GOLDEN_ROWS: usize = 110;

const SEP: char = '\u{1f}';

/// A decode-plan handle: its creation ordinal, the C++ recorder's `dp#N`.
struct Dp(u32);
/// A prefill-plan handle, `pp#N`.
struct Pp(u32);

fn view_at(addr: usize) -> AttentionWorkspaceView {
    AttentionWorkspaceView {
        float_buffer: addr as *mut std::ffi::c_void,
        float_bytes: 0,
        int_buffer: std::ptr::null_mut(),
        int_bytes: 0,
        page_locked_int: std::ptr::null_mut(),
    }
}

/// The oracle's planner recorder, reproduced.
struct FakePlanner {
    out: String,
    case: String,
    next_decode: u32,
    next_prefill: u32,
}

impl FakePlanner {
    fn new() -> Self {
        Self { out: String::new(), case: String::new(), next_decode: 0, next_prefill: 0 }
    }

    fn ws_name(v: &AttentionWorkspaceView) -> &'static str {
        match v.float_buffer as usize {
            0x1000 => "ws-main",
            0x2000 => "ws-suffix",
            0x3000 => "ws-band0",
            0x4000 => "ws-band1",
            0x5000 => "ws-band2",
            _ => "ws?",
        }
    }

    fn join(vals: &[u32], n: usize) -> String {
        let mut s = "[".to_string();
        for (i, v) in vals.iter().take(n).enumerate() {
            if i > 0 {
                s.push(',');
            }
            let _ = write!(s, "{v}");
        }
        s.push(']');
        s
    }
}

impl PlannerOps for FakePlanner {
    type DecodePlan = Dp;
    type PrefillPlan = Pp;

    fn make_decode_plan(&mut self) -> Dp {
        let id = self.next_decode;
        self.next_decode += 1;
        Dp(id)
    }

    fn make_prefill_plan(&mut self) -> Pp {
        let id = self.next_prefill;
        self.next_prefill += 1;
        Pp(id)
    }

    fn plan_decode(
        &mut self,
        plan: &mut Dp,
        kv_page_indptr_h: &[u32],
        num_requests: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        workspace: AttentionWorkspaceView,
        enable_cuda_graph: bool,
        full_attention_variant: bool,
        hnd_layout: bool,
    ) {
        let n = usize::try_from(num_requests + 1).unwrap_or(0);
        let _ = writeln!(
            self.out,
            "{}{SEP}plan-decode{SEP}dp#{}{SEP}kvpp={}{SEP}R={num_requests}\
             {SEP}qh={num_q_heads}{SEP}kvh={num_kv_heads}{SEP}hd={head_dim}\
             {SEP}psz={page_size}{SEP}{}{SEP}s0{SEP}graph={}{SEP}fav={}\
             {SEP}hnd={}{SEP}wl=-1",
            self.case,
            plan.0,
            Self::join(kv_page_indptr_h, n),
            Self::ws_name(&workspace),
            u8::from(enable_cuda_graph),
            u8::from(full_attention_variant),
            u8::from(hnd_layout),
        );
    }

    fn plan_prefill(
        &mut self,
        plan: &mut Pp,
        qo_indptr_h: &[u32],
        kv_page_indptr_h: &[u32],
        kv_last_page_lens_h: &[u32],
        total_tokens: i32,
        num_requests: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        workspace: AttentionWorkspaceView,
        enable_cuda_graph: bool,
        window_left: i32,
        full_attention_variant: bool,
        hnd_layout: bool,
        causal_mask: bool,
        custom_mask: bool,
        wants_prefill_score: bool,
    ) {
        let n1 = usize::try_from(num_requests + 1).unwrap_or(0);
        let n = usize::try_from(num_requests).unwrap_or(0);
        let _ = writeln!(
            self.out,
            "{}{SEP}plan-prefill{SEP}pp#{}{SEP}qo={}{SEP}kvpp={}{SEP}lens={}\
             {SEP}T={total_tokens}{SEP}R={num_requests}{SEP}qh={num_q_heads}\
             {SEP}kvh={num_kv_heads}{SEP}hd={head_dim}{SEP}psz={page_size}\
             {SEP}{}{SEP}s0{SEP}graph={}{SEP}wl={window_left}{SEP}fav={}\
             {SEP}hnd={}{SEP}causal={}{SEP}custom={}{SEP}score={}",
            self.case,
            plan.0,
            Self::join(qo_indptr_h, n1),
            Self::join(kv_page_indptr_h, n1),
            Self::join(kv_last_page_lens_h, n),
            Self::ws_name(&workspace),
            u8::from(enable_cuda_graph),
            u8::from(full_attention_variant),
            u8::from(hnd_layout),
            u8::from(causal_mask),
            u8::from(custom_mask),
            u8::from(wants_prefill_score),
        );
    }

    fn xqa_decode_page_bucket(&mut self, max_pages_per_seq: i32) -> i32 {
        let mut b = 4;
        while b < max_pages_per_seq {
            b *= 2;
        }
        b
    }

    fn spatial_suffix_ws_view(&mut self) -> AttentionWorkspaceView {
        view_at(0x2000)
    }

    fn depth_band_ws_view(&mut self, band: usize) -> AttentionWorkspaceView {
        view_at(0x3000 + band * 0x1000)
    }
}

type State = LlamaLikePlanState<Dp, Pp>;

fn dp(p: &Option<Dp>) -> String {
    p.as_ref().map_or("null".into(), |d| format!("dp#{}", d.0))
}
fn pp(p: &Option<Pp>) -> String {
    p.as_ref().map_or("null".into(), |d| format!("pp#{}", d.0))
}

fn dump_state(ops: &mut FakePlanner, s: &State) {
    let bands = s.depth_band_plans.iter().map(dp).collect::<Vec<_>>().join(",");
    let band_pf = s.depth_band_prefill_plans.iter().map(pp).collect::<Vec<_>>().join(",");
    let bk = s.depth_band_k.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
    let br = s.depth_band_rows.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
    let pdqo = format!(
        "[{}]",
        s.prefill_decode_qo_indptr_h
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    let case = ops.case.clone();
    let _ = writeln!(
        ops.out,
        "{case}{SEP}state{SEP}decode={}{SEP}prefill={}{SEP}pd={}{SEP}mask={}\
         {SEP}dpfx={}{SEP}bands={bands}{SEP}band_pf={band_pf}{SEP}band_k={bk}\
         {SEP}band_rows={br}{SEP}band_n={}{SEP}mid={}{SEP}mid_start={}\
         {SEP}sm={}{SEP}sm_row={}{SEP}use_pf={}{SEP}use_pd={}{SEP}use_mask={}\
         {SEP}score_w={}{SEP}xqa={}{SEP}xqa_max={}{SEP}pd_qo={pdqo}",
        dp(&s.decode_plan),
        pp(&s.prefill_plan),
        pp(&s.prefill_decode_plan),
        pp(&s.mask_decode_plan),
        dp(&s.depth_prefix_decode_plan),
        s.depth_band_count,
        dp(&s.mixed_mid_decode_plan),
        s.mixed_mid_start,
        s.spatial_mask_split,
        s.spatial_mask_row_split,
        u8::from(s.use_prefill_plan),
        u8::from(s.use_prefill_decode_plan),
        u8::from(s.use_mask_decode_plan),
        s.prefill_score_window,
        u8::from(s.use_xqa_decode),
        s.xqa_max_pages_per_seq,
    );
}

struct Fire {
    qo: Vec<u32>,
    kvpp: Vec<u32>,
    lens: Vec<u32>,
}

impl Fire {
    fn total_tokens(&self) -> i32 {
        i32::try_from(*self.qo.last().unwrap()).unwrap()
    }
    fn requests(&self) -> i32 {
        i32::try_from(self.qo.len() - 1).unwrap()
    }
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    ops: &mut FakePlanner,
    gates: &PrepareGates,
    name: &str,
    ws: AttentionWorkspaceView,
    cache: &KvGeom,
    cfg: &HfConfig,
    state: &mut State,
    fwd: &LlamaLikeForwardCfg,
    fire: &Fire,
    is_pure_decode: bool,
    have_custom_mask: bool,
    params: &PrepareParams<'_>,
) {
    ops.case = name.to_string();
    let case = ops.case.clone();
    let _ = writeln!(
        ops.out,
        "{case}{SEP}call{SEP}R={}{SEP}T={}{SEP}pure={}{SEP}mask={}\
         {SEP}score_w={}{SEP}prefix={}{SEP}fdr={}{SEP}bands={}",
        fire.requests(),
        fire.total_tokens(),
        u8::from(is_pure_decode),
        u8::from(have_custom_mask),
        params.attn_score_window,
        params.unmasked_prefix_rows,
        params.full_depth_rows,
        params.depth_band_count,
    );
    prepare_llama_like_decode_plan(
        ops,
        gates,
        state,
        ws,
        cache,
        cfg,
        fwd,
        &fire.qo,
        &fire.kvpp,
        &fire.lens,
        fire.total_tokens(),
        fire.requests(),
        is_pure_decode,
        have_custom_mask,
        params,
    );
    dump_state(ops, state);
}

fn transcript() -> String {
    let cfg = HfConfig {
        num_attention_heads: 8,
        num_key_value_heads: 4,
        head_dim: 64,
        head_dim_kernel: 64,
        ..HfConfig::default()
    };
    let cfg_padded = HfConfig { head_dim: 80, head_dim_kernel: 96, ..cfg.clone() };

    let bf16 = KvGeom { page_size: 16, hnd_layout: false, native_bf16: true };
    let int8 = KvGeom { page_size: 16, hnd_layout: false, native_bf16: false };
    let ws = view_at(0x1000);

    let decode4 = Fire {
        qo: vec![0, 1, 2, 3, 4],
        kvpp: vec![0, 2, 5, 6, 10],
        lens: vec![3, 16, 1, 7],
    };
    let decode3 = Fire { qo: vec![0, 1, 2, 3], kvpp: vec![0, 3, 4, 9], lens: vec![5, 2, 16] };
    let prefill3 = Fire { qo: vec![0, 5, 9, 10], kvpp: vec![0, 2, 5, 6], lens: vec![3, 16, 1] };
    let mixed5 = Fire {
        qo: vec![0, 5, 9, 10, 11, 12],
        kvpp: vec![0, 2, 5, 6, 8, 9],
        lens: vec![3, 16, 1, 7, 2],
    };
    let mixed_mid = Fire {
        qo: vec![0, 5, 6, 7, 8, 9],
        kvpp: vec![0, 2, 4, 5, 7, 8],
        lens: vec![3, 1, 16, 7, 2],
    };

    let main_gates = PrepareGates {
        spatial_mask_on: true,
        mixed_mid_on: true,
        prefill_graph_plan: false,
        region_trace: false,
    };

    let mut ops = FakePlanner::new();
    let g = &main_gates;
    let base = LlamaLikeForwardCfg::default();
    let d = PrepareParams::default();

    {
        let mut st = State::default();
        run_case(&mut ops, g, "a-plain-decode", ws, &bf16, &cfg, &mut st, &base, &decode4, true, false, &d);
        run_case(&mut ops, g, "a2-reuse", ws, &bf16, &cfg, &mut st, &base, &decode4, true, false, &d);
    }
    {
        let mut st = State::default();
        let f = LlamaLikeForwardCfg { use_prefill_decode_plan: true, ..Default::default() };
        run_case(&mut ops, g, "b-prefill-decode", ws, &bf16, &cfg, &mut st, &f, &decode4, true, false, &d);
        let f2 = LlamaLikeForwardCfg {
            use_prefill_decode_plan: true,
            prefill_decode_full_attention_min_requests: 2,
            ..Default::default()
        };
        run_case(&mut ops, g, "b2-pd-fav", ws, &bf16, &cfg, &mut st, &f2, &decode4, true, false, &d);
        let f3 = LlamaLikeForwardCfg {
            use_prefill_decode_plan: true,
            prefill_decode_min_kv_pages: 4,
            ..Default::default()
        };
        run_case(&mut ops, g, "b3-pd-declined", ws, &bf16, &cfg, &mut st, &f3, &decode4, true, false, &d);
        let f4 = LlamaLikeForwardCfg {
            use_prefill_decode_plan: true,
            prefill_decode_min_kv_pages: 3,
            ..Default::default()
        };
        run_case(&mut ops, g, "b4-pd-ceiling", ws, &bf16, &cfg, &mut st, &f4, &decode4, true, false, &d);
    }
    {
        let mut st = State::default();
        let f = LlamaLikeForwardCfg { use_xqa_decode: true, ..Default::default() };
        let bands = PrepareParams {
            depth_band_k: &[8, 4],
            depth_band_rows: &[3, 1],
            depth_band_count: 2,
            ..PrepareParams::default()
        };
        run_case(&mut ops, g, "c-xqa", ws, &bf16, &cfg, &mut st, &f, &decode4, true, false, &bands);
        let mut st2 = State::default();
        run_case(&mut ops, g, "c2-xqa-nonnative", ws, &int8, &cfg, &mut st2, &f, &decode4, true, false, &d);
    }
    {
        let mut st = State::default();
        let score = PrepareParams { attn_score_window: 3, ..PrepareParams::default() };
        run_case(&mut ops, g, "d-prefill", ws, &bf16, &cfg, &mut st, &base, &prefill3, false, false, &score);
        let f = LlamaLikeForwardCfg { sliding_window: 128, ..Default::default() };
        let mut st2 = State::default();
        run_case(&mut ops, g, "d2-prefill-sliding", ws, &bf16, &cfg, &mut st2, &f, &prefill3, false, false, &score);
        let f2 = LlamaLikeForwardCfg { per_layer_window_left: vec![128, -1, 128], ..Default::default() };
        let mut st3 = State::default();
        run_case(&mut ops, g, "d3-prefill-per-layer", ws, &bf16, &cfg, &mut st3, &f2, &prefill3, false, false, &d);
    }
    {
        let mut st = State::default();
        let f = LlamaLikeForwardCfg { force_prefill_path: true, ..Default::default() };
        let bands = PrepareParams {
            depth_band_k: &[6],
            depth_band_rows: &[2],
            depth_band_count: 1,
            ..PrepareParams::default()
        };
        run_case(&mut ops, g, "e-force-prefill", ws, &bf16, &cfg, &mut st, &f, &decode3, true, false, &bands);
    }
    {
        let mut st = State::default();
        run_case(&mut ops, g, "f-mask-decode", ws, &bf16, &cfg, &mut st, &base, &decode4, true, false, &d);
        run_case(&mut ops, g, "f2-mask-dedicated", ws, &bf16, &cfg, &mut st, &base, &decode4, true, true, &d);
        let mut st2 = State::default();
        run_case(&mut ops, g, "f3-mask-prefill-shaped", ws, &bf16, &cfg, &mut st2, &base, &prefill3, false, true, &d);
    }
    {
        let mut st = State::default();
        let p2 = PrepareParams { unmasked_prefix_rows: 2, ..PrepareParams::default() };
        run_case(&mut ops, g, "g-spatial-split", ws, &bf16, &cfg, &mut st, &base, &decode4, true, true, &p2);
        let mut st2 = State::default();
        let p0 = PrepareParams { unmasked_prefix_rows: 0, ..PrepareParams::default() };
        run_case(&mut ops, g, "g2-spatial-zero", ws, &bf16, &cfg, &mut st2, &base, &decode4, true, true, &p0);
        let counts = [3u32, 2];
        let lens2 = [9u32, 11];
        let mut st3 = State::default();
        let pr = PrepareParams {
            unmasked_prefix_rows: 2,
            mask_suffix_page_counts_h: Some(&counts),
            mask_suffix_last_lens_h: Some(&lens2),
            ..PrepareParams::default()
        };
        run_case(&mut ops, g, "g3-spatial-resolved", ws, &bf16, &cfg, &mut st3, &base, &decode4, true, true, &pr);
        let mut st4 = State::default();
        run_case(&mut ops, g, "g4-spatial-padded-declined", ws, &bf16, &cfg_padded, &mut st4, &base, &decode4, true, true, &p2);
        let mut st5 = State::default();
        let fx = LlamaLikeForwardCfg { use_xqa_decode: true, ..Default::default() };
        run_case(&mut ops, g, "g5-spatial-xqa-declined", ws, &bf16, &cfg, &mut st5, &fx, &decode4, true, true, &p2);
    }
    {
        let mut st = State::default();
        let p3 = PrepareParams { unmasked_prefix_rows: 3, ..PrepareParams::default() };
        run_case(&mut ops, g, "h-mixed", ws, &bf16, &cfg, &mut st, &base, &mixed5, false, true, &p3);
        let mut st2 = State::default();
        run_case(&mut ops, g, "h2-mixed-mid", ws, &bf16, &cfg, &mut st2, &base, &mixed_mid, false, true, &p3);
        let mut st3 = State::default();
        let p7 = PrepareParams { unmasked_prefix_rows: 7, ..PrepareParams::default() };
        run_case(&mut ops, g, "h3-mixed-declined", ws, &bf16, &cfg, &mut st3, &base, &mixed5, false, true, &p7);
    }
    {
        let mut st = State::default();
        let fd2 = PrepareParams { full_depth_rows: 2, ..PrepareParams::default() };
        run_case(&mut ops, g, "i-depth-prefix", ws, &bf16, &cfg, &mut st, &base, &decode4, true, false, &fd2);
        let mut st2 = State::default();
        let fd4 = PrepareParams { full_depth_rows: 4, ..PrepareParams::default() };
        run_case(&mut ops, g, "i2-depth-full-declined", ws, &bf16, &cfg, &mut st2, &base, &decode4, true, false, &fd4);
    }
    {
        let mut st = State::default();
        let bands = PrepareParams {
            depth_band_k: &[8, 4, 2],
            depth_band_rows: &[2, 0, 1],
            depth_band_count: 3,
            ..PrepareParams::default()
        };
        run_case(&mut ops, g, "j-bands-decode", ws, &bf16, &cfg, &mut st, &base, &decode4, true, false, &bands);
        run_case(&mut ops, g, "j1b-bands-cleared", ws, &bf16, &cfg, &mut st, &base, &decode4, true, false, &d);
        let mut st2 = State::default();
        let f = LlamaLikeForwardCfg { use_prefill_decode_plan: true, ..Default::default() };
        let bands2 = PrepareParams {
            depth_band_k: &[6, 3],
            depth_band_rows: &[3, 1],
            depth_band_count: 2,
            ..PrepareParams::default()
        };
        run_case(&mut ops, g, "j2-bands-pd", ws, &bf16, &cfg, &mut st2, &f, &decode4, true, false, &bands2);
        run_case(&mut ops, g, "j2b-bands-pd-cleared", ws, &bf16, &cfg, &mut st2, &f, &decode4, true, false, &d);
    }
    {
        let mut st = State::default();
        let f = LlamaLikeForwardCfg { tp_size: 2, ..Default::default() };
        run_case(&mut ops, g, "k-tp2", ws, &bf16, &cfg, &mut st, &f, &decode4, true, false, &d);
    }

    let mut out = ops.out;

    // spatial-off: PIE_SPATIAL_MASK=0 in a fresh process.
    let mut ops = FakePlanner::new();
    let gates = PrepareGates { spatial_mask_on: false, ..main_gates };
    let mut st = State::default();
    let p2 = PrepareParams { unmasked_prefix_rows: 2, ..PrepareParams::default() };
    run_case(&mut ops, &gates, "z-spatial-off", ws, &bf16, &cfg, &mut st, &base, &decode4, true, true, &p2);
    out.push_str(&ops.out);

    // mid-off: PIE_MIXED_MID=0.
    let mut ops = FakePlanner::new();
    let gates = PrepareGates { mixed_mid_on: false, ..main_gates };
    let mut st = State::default();
    let p3 = PrepareParams { unmasked_prefix_rows: 3, ..PrepareParams::default() };
    run_case(&mut ops, &gates, "z-mid-off", ws, &bf16, &cfg, &mut st, &base, &mixed_mid, false, true, &p3);
    out.push_str(&ops.out);

    // graph-plan-on: PIE_PREFILL_GRAPH_PLAN=1.
    let mut ops = FakePlanner::new();
    let gates = PrepareGates { prefill_graph_plan: true, ..main_gates };
    let mut st = State::default();
    let score = PrepareParams { attn_score_window: 3, ..PrepareParams::default() };
    run_case(&mut ops, &gates, "z-graph-plan", ws, &bf16, &cfg, &mut st, &base, &prefill3, false, false, &score);
    out.push_str(&ops.out);

    out
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
    assert_eq!(rows, GOLDEN_ROWS, "row count diverged — case shape changed");
    let hash = fnv1a64(text.as_bytes());
    if hash != GOLDEN_FNV1A64 {
        let path = std::env::temp_dir().join("llama_like_prepare_rust_transcript.txt");
        std::fs::write(&path, &text).ok();
        panic!(
            "transcript hash 0x{hash:016x} != golden 0x{GOLDEN_FNV1A64:016x}; \
             rust transcript dumped to {}",
            path.display()
        );
    }
}

/// The gates' env parsing agrees with the C++ shapes for THIS process's
/// environment — the one observation a single process can make.
#[test]
fn gates_from_env_matches_the_documented_shapes() {
    let gates = PrepareGates::from_env();
    // Unset in the test environment: the two on-unless-zero gates default
    // ON, the off-unless-set gate defaults OFF.
    if std::env::var_os("PIE_SPATIAL_MASK").is_none() {
        assert!(gates.spatial_mask_on);
    }
    if std::env::var_os("PIE_PREFILL_GRAPH_PLAN").is_none() {
        assert!(!gates.prefill_graph_plan);
    }
}
