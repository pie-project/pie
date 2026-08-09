//! Behavioural parity with the C++ Qwen3.5 linear-attention workspace —
//! gate-linear-attn-ws.
//!
//! The oracle in `tests/oracle/qwen35_la_ws/` compiles the real
//! `qwen3_5_forward.cpp` (forward body discarded by `--gc-sections`) and
//! drives `Qwen3_5LinearAttnWorkspace::allocate` over four dim shapes,
//! recording every allocation's ordinal and bytes plus the member →
//! buffer mapping. This test replays the same sweep and requires the
//! transcripts to be byte-identical.
//!
//! Run `tests/oracle/qwen35_la_ws/run.sh` to regenerate
//! [`GOLDEN_FNV1A64`]. The pinned value is the **C++'s** hash.

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt::Write as _;

use driver_cuda_new::model::qwen3_5::{
    LinearAttnDims, LinearAttnWorkspace, Qwen35ForwardCfg, Qwen35PlanState,
};
use driver_cuda_new::model::sideband_arena::DeviceMemory;

/// FNV-1a 64 of the C++ oracle's transcript.
const GOLDEN_FNV1A64: u64 = 0xbdfd6530fa2ad79d;

/// Rows the transcript must contain, so a truncated sweep cannot pass.
const GOLDEN_ROWS: usize = 100;

const SEP: char = '\u{1f}';

struct FakeMem {
    out: String,
    case: String,
    names: HashMap<usize, String>,
    next: usize,
    next_addr: usize,
}

impl FakeMem {
    fn new() -> Self {
        Self {
            out: String::new(),
            case: String::new(),
            names: HashMap::new(),
            next: 0,
            next_addr: 0x1000,
        }
    }

    fn begin(&mut self, name: &str) {
        self.case = name.to_string();
        self.names.clear();
        self.next = 0;
    }

    fn note(&mut self, body: &str) {
        let case = self.case.clone();
        let _ = writeln!(self.out, "{case}{SEP}{body}");
    }

    fn buf_of(&self, p: *mut c_void) -> String {
        if p.is_null() {
            return "null".into();
        }
        self.names.get(&(p as usize)).cloned().unwrap_or_else(|| "unknown".into())
    }
}

impl DeviceMemory for FakeMem {
    fn alloc(&mut self, bytes: usize) -> Option<*mut c_void> {
        let p = self.next_addr;
        self.next_addr += 0x1000;
        let name = format!("buf{}", self.next);
        self.next += 1;
        self.names.insert(p, name.clone());
        self.note(&format!("alloc {name} bytes={bytes}"));
        Some(p as *mut c_void)
    }

    fn free(&mut self, _ptr: *mut c_void) {}

    fn synchronize(&mut self) -> bool {
        true
    }
}

fn run_case(
    mem: &mut FakeMem,
    name: &str,
    (max_tokens, conv_dim, v_h, k_h, k_d, v_d, hq): (i32, i32, i32, i32, i32, i32, i32),
) {
    mem.begin(name);
    mem.note(&format!(
        "case-begin N={max_tokens} conv={conv_dim} vh={v_h} kh={k_h} kd={k_d} \
         vd={v_d} hq={hq}"
    ));
    let dims = LinearAttnDims { max_tokens, conv_dim, v_h, k_h, k_d, v_d, hq };
    let mut ws = LinearAttnWorkspace::allocate(mem, &dims);
    let row = format!(
        "members mixed_qkv={} mixed_qkvz={} ba={} z={} a={} b={} \
         mixed_qkv_post={} q_norm={} k_norm={} v_fp32={} g_log={} beta={} \
         core_out={} core_out_bf16={} q_raw={} k_raw={} v_raw={} q_pre={} \
         k_pre={} fa_qg_packed={} fa_gate={} qo_ext={} rs_write_state_mask={} \
         qo_split={} split_slot_head={} split_slot_tail={} split_mask_head={} \
         max_tokens={}",
        mem.buf_of(ws.mixed_qkv),
        mem.buf_of(ws.mixed_qkvz),
        mem.buf_of(ws.ba),
        mem.buf_of(ws.z),
        mem.buf_of(ws.a),
        mem.buf_of(ws.b),
        mem.buf_of(ws.mixed_qkv_post),
        mem.buf_of(ws.q_norm),
        mem.buf_of(ws.k_norm),
        mem.buf_of(ws.v_fp32),
        mem.buf_of(ws.g_log),
        mem.buf_of(ws.beta),
        mem.buf_of(ws.core_out),
        mem.buf_of(ws.core_out_bf16),
        mem.buf_of(ws.q_raw),
        mem.buf_of(ws.k_raw),
        mem.buf_of(ws.v_raw),
        mem.buf_of(ws.q_pre),
        mem.buf_of(ws.k_pre),
        mem.buf_of(ws.fa_qg_packed),
        mem.buf_of(ws.fa_gate),
        mem.buf_of(ws.qo_ext),
        mem.buf_of(ws.rs_write_state_mask),
        mem.buf_of(ws.qo_split),
        mem.buf_of(ws.split_slot_head),
        mem.buf_of(ws.split_slot_tail),
        mem.buf_of(ws.split_mask_head),
        ws.max_tokens,
    );
    mem.note(&row);
    ws.release(mem);
}

fn transcript() -> String {
    let mut mem = FakeMem::new();
    run_case(&mut mem, "a-asym", (64, 97, 5, 3, 7, 11, 13));
    run_case(&mut mem, "b-4b", (128, 4096, 32, 16, 128, 128, 32));
    run_case(&mut mem, "c-ones", (1, 1, 1, 1, 1, 1, 1));
    run_case(&mut mem, "d-zero-tokens", (0, 96, 4, 2, 8, 16, 8));

    mem.begin("e-cfg-defaults");
    {
        let c = Qwen35ForwardCfg::default();
        let rows = [
            format!("force_prefill_path={}", u8::from(c.force_prefill_path)),
            format!(
                "small_prefill_naive_attention_max_tokens={}",
                c.small_prefill_naive_attention_max_tokens
            ),
            format!("tp_size={}", c.tp_size),
            format!("tp_comm_null={}", u8::from(c.tp_comm.is_null())),
            format!(
                "mtp_global_cache_uses_prefix_position={}",
                u8::from(c.mtp_global_cache_uses_prefix_position)
            ),
        ];
        for row in rows {
            mem.note(&row);
        }
        let s: Qwen35PlanState<(), ()> = Qwen35PlanState::default();
        let rows = [
            format!("decode_plan_null={}", u8::from(s.decode_plan.is_none())),
            format!("prefill_plan_null={}", u8::from(s.prefill_plan.is_none())),
            format!("use_prefill_plan={}", u8::from(s.use_prefill_plan)),
        ];
        for row in rows {
            mem.note(&row);
        }
    }

    mem.out
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
    assert_eq!(rows, GOLDEN_ROWS, "row count diverged — sweep shape changed");
    let hash = fnv1a64(text.as_bytes());
    if hash != GOLDEN_FNV1A64 {
        let path = std::env::temp_dir().join("qwen35_la_ws_rust_transcript.txt");
        std::fs::write(&path, &text).ok();
        panic!(
            "transcript hash 0x{hash:016x} != golden 0x{GOLDEN_FNV1A64:016x}; \
             rust transcript dumped to {}",
            path.display()
        );
    }
}
