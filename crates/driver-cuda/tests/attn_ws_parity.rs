//! Behavioural parity with the C++ `AttentionWorkspace` — gate-attn-ws.
//!
//! The oracle in `tests/oracle/attn_ws/` compiles the real
//! `attention_workspace.cpp`, replaces `DeviceTensor::allocate` and the six
//! CUDA entry points with recorders, and drives eight scripts: allocation
//! shapes, the slot rotation at depths 1/2/3, a move-assignment over a
//! pending upload, and four failure landings. This test replays the same
//! scripts against [`AttentionWorkspace`] and requires the transcripts to
//! be byte-identical.
//!
//! Run `tests/oracle/attn_ws/run.sh` to regenerate [`GOLDEN_FNV1A64`]. The
//! pinned value is the **C++'s** hash, never this file's.
//!
//! The C++ move-assignment maps to `b.release(&mut ops); b = a;` here —
//! the target's cleanup (sync the pending upload, tear down the slots) is
//! exactly [`AttentionWorkspace::release`], and Rust's move needs no
//! source-emptying because the source is statically gone.

use std::collections::HashMap;
use std::ffi::c_void;

use driver_cuda::gpu::fire::attention_workspace::{
    AttentionWorkspace, StagingError, StagingOps,
};

/// FNV-1a 64 of the C++ oracle's transcript.
const GOLDEN_FNV1A64: u64 = 0xddbd891048ef7a23;

/// Rows the transcript must contain, so a truncated sweep cannot pass.
const GOLDEN_ROWS: usize = 149;

const SEP: char = '\u{1f}';

/// The oracle's recorders, reproduced: pins and events named by creation
/// ordinal, device allocations logged in `DeviceTensor::allocate`'s format,
/// two logs drained cuda-first at every flush — the same interleave the
/// C++ harness produces.
struct FakeOps {
    cuda_log: Vec<String>,
    dev_log: Vec<String>,
    pins: HashMap<usize, String>,
    streams: HashMap<usize, String>,
    next_pin: usize,
    next_event: usize,
    next_addr: usize,
    fail_next_malloc: bool,
    fail_next_event: bool,
}

impl FakeOps {
    fn new() -> Self {
        Self {
            cuda_log: Vec::new(),
            dev_log: Vec::new(),
            pins: HashMap::new(),
            streams: HashMap::new(),
            next_pin: 0,
            next_event: 0,
            next_addr: 0x1000,
            fail_next_malloc: false,
            fail_next_event: false,
        }
    }

    fn fresh_addr(&mut self) -> *mut c_void {
        let a = self.next_addr;
        self.next_addr += 0x1000;
        a as *mut c_void
    }

    fn name_stream(&mut self, s: *mut c_void, name: &str) {
        self.streams.insert(s as usize, name.to_string());
    }

    fn stream_name(&self, s: *mut c_void) -> String {
        if s.is_null() {
            return "s0".into();
        }
        self.streams.get(&(s as usize)).cloned().unwrap_or_else(|| "unknown".into())
    }

    fn pin_name(&self, p: *mut c_void) -> String {
        if p.is_null() {
            return "null".into();
        }
        self.pins.get(&(p as usize)).cloned().unwrap_or_else(|| "unknown".into())
    }
}

impl StagingOps for FakeOps {
    type Event = usize;

    fn malloc_host(&mut self, bytes: usize) -> Option<*mut c_void> {
        if self.fail_next_malloc {
            self.fail_next_malloc = false;
            self.cuda_log.push(format!("pin-fail{SEP}{bytes}"));
            return None;
        }
        let p = self.fresh_addr();
        let name = format!("pin#{}", self.next_pin);
        self.next_pin += 1;
        self.pins.insert(p as usize, name.clone());
        self.cuda_log.push(format!("pin{SEP}{bytes}{SEP}{name}"));
        Some(p)
    }

    fn free_host(&mut self, ptr: *mut c_void) {
        let name = self.pin_name(ptr);
        self.cuda_log.push(format!("unpin{SEP}{name}"));
        self.pins.remove(&(ptr as usize));
    }

    fn event_create(&mut self) -> Option<usize> {
        if self.fail_next_event {
            self.fail_next_event = false;
            self.cuda_log.push("evc-fail".into());
            return None;
        }
        let k = self.next_event;
        self.next_event += 1;
        self.cuda_log.push(format!("evc{SEP}ev#{k}"));
        Some(k)
    }

    fn event_destroy(&mut self, event: usize) {
        self.cuda_log.push(format!("evd{SEP}ev#{event}"));
    }

    fn event_synchronize(&mut self, event: &usize) -> bool {
        self.cuda_log.push(format!("evs{SEP}ev#{event}"));
        // The RECORDER always succeeds: it is the transcript of what a
        // working device would do, and the C++ oracle it is held against
        // has no failing leg to transcribe.
        true
    }

    fn event_record(&mut self, event: &usize, stream: *mut c_void) -> bool {
        let s = self.stream_name(stream);
        self.cuda_log.push(format!("evr{SEP}ev#{event}{SEP}{s}"));
        true
    }

    fn alloc_device(&mut self, bytes: usize) -> Option<*mut c_void> {
        self.dev_log.push(format!("u8[{bytes}]={bytes}"));
        Some(self.fresh_addr())
    }

    fn free_device(&mut self, _ptr: *mut c_void) {}
}

/// The transcript under construction, with the oracle's flush discipline.
struct Harness {
    out: String,
    script: String,
    dev_names: HashMap<usize, &'static str>,
}

type Ws = AttentionWorkspace<usize>;

impl Harness {
    fn new() -> Self {
        Self { out: String::new(), script: String::new(), dev_names: HashMap::new() }
    }

    fn begin_script(&mut self, ops: &mut FakeOps, name: &str) {
        *ops = FakeOps::new();
        self.dev_names.clear();
        self.script = name.to_string();
        ops.cuda_log.push("case-begin".into());
        self.flush(ops);
    }

    fn flush(&mut self, ops: &mut FakeOps) {
        for row in ops.cuda_log.drain(..) {
            self.out.push_str(&self.script);
            self.out.push(SEP);
            self.out.push_str(&row);
            self.out.push('\n');
        }
        for row in ops.dev_log.drain(..) {
            self.out.push_str(&self.script);
            self.out.push(SEP);
            self.out.push_str("dev");
            self.out.push(SEP);
            self.out.push_str(&row);
            self.out.push('\n');
        }
    }

    fn call(&mut self, ops: &mut FakeOps, what: &str) {
        ops.cuda_log.push(format!("call{SEP}{what}"));
        self.flush(ops);
    }

    fn name_buffers(&mut self, ws: &Ws) {
        self.dev_names.insert(ws.float_buffer() as usize, "dev#float");
        self.dev_names.insert(ws.int_buffer() as usize, "dev#int");
    }

    fn dev_name(&self, p: *mut c_void) -> String {
        if p.is_null() {
            return "null".into();
        }
        self.dev_names.get(&(p as usize)).map_or_else(|| "unknown".into(), |s| (*s).into())
    }

    fn view_row(&mut self, ops: &mut FakeOps, ws: &Ws) {
        let v = ws.view();
        let row = format!(
            "view{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}",
            self.dev_name(v.float_buffer),
            v.float_bytes,
            self.dev_name(v.int_buffer),
            v.int_bytes,
            ops.pin_name(v.page_locked_int),
        );
        ops.cuda_log.push(row);
        self.flush(ops);
    }

    fn note_result(&mut self, ops: &mut FakeOps, r: Result<(), StagingError>) {
        ops.cuda_log
            .push(if r.is_ok() { "no-throw".into() } else { "threw".into() });
        self.flush(ops);
    }
}

fn transcript() -> String {
    let mut h = Harness::new();
    let mut ops = FakeOps::new();
    let s_a = 0xA0 as *mut c_void;
    let s_b = 0xB0 as *mut c_void;

    // a. Boot shape.
    h.begin_script(&mut ops, "a-alloc");
    ops.name_stream(s_a, "sA");
    h.call(&mut ops, "allocate(1024,512,1)");
    let mut ws = Ws::allocate(&mut ops, 1024, 512, 1).unwrap();
    h.name_buffers(&ws);
    h.view_row(&mut ops, &ws);
    h.call(&mut ops, "drop");
    ws.release(&mut ops);
    h.flush(&mut ops);

    // b. Zero slots clamps to one; a single slot fences itself.
    h.begin_script(&mut ops, "b-one-slot");
    ops.name_stream(s_a, "sA");
    ops.name_stream(s_b, "sB");
    h.call(&mut ops, "allocate(2048,256,0)");
    let mut ws = Ws::allocate(&mut ops, 2048, 256, 0).unwrap();
    h.name_buffers(&ws);
    h.call(&mut ops, "begin");
    ws.begin_plan_update(&mut ops).unwrap();
    h.view_row(&mut ops, &ws);
    h.call(&mut ops, "end(sA)");
    ws.end_plan_update(&mut ops, s_a);
    h.call(&mut ops, "begin");
    ws.begin_plan_update(&mut ops).unwrap();
    h.view_row(&mut ops, &ws);
    h.call(&mut ops, "end(sB)");
    ws.end_plan_update(&mut ops, s_b);
    h.call(&mut ops, "drop");
    ws.release(&mut ops);
    h.flush(&mut ops);

    // c. Depth 3: lazy pins, the wraparound fence, teardown with two
    //    uploads pending.
    h.begin_script(&mut ops, "c-rotate-3");
    ops.name_stream(s_a, "sA");
    ops.name_stream(s_b, "sB");
    h.call(&mut ops, "allocate(64,32,3)");
    let mut ws = Ws::allocate(&mut ops, 64, 32, 3).unwrap();
    h.name_buffers(&ws);
    for (end_label, stream) in
        [("end(sA)", s_a), ("end(sB)", s_b), ("end(sA)", s_a)]
    {
        h.call(&mut ops, "begin");
        ws.begin_plan_update(&mut ops).unwrap();
        h.view_row(&mut ops, &ws);
        h.call(&mut ops, end_label);
        ws.end_plan_update(&mut ops, stream);
    }
    h.call(&mut ops, "begin");
    ws.begin_plan_update(&mut ops).unwrap();
    h.view_row(&mut ops, &ws);
    h.call(&mut ops, "drop");
    ws.release(&mut ops);
    h.flush(&mut ops);

    // d. Move-assignment onto a workspace of its own.
    h.begin_script(&mut ops, "d-move-assign");
    ops.name_stream(s_a, "sA");
    h.call(&mut ops, "allocate-a(16,16,2)");
    let mut a = Ws::allocate(&mut ops, 16, 16, 2).unwrap();
    h.name_buffers(&a);
    h.call(&mut ops, "a.begin");
    a.begin_plan_update(&mut ops).unwrap();
    h.call(&mut ops, "a.end(sA)");
    a.end_plan_update(&mut ops, s_a);
    h.call(&mut ops, "allocate-b(16,16,1)");
    let mut b = Ws::allocate(&mut ops, 16, 16, 1).unwrap();
    h.call(&mut ops, "b.begin");
    b.begin_plan_update(&mut ops).unwrap();
    h.call(&mut ops, "b.end(sA)");
    b.end_plan_update(&mut ops, s_a);
    h.call(&mut ops, "b=move(a)");
    b.release(&mut ops);
    b = a;
    h.call(&mut ops, "b.begin");
    b.begin_plan_update(&mut ops).unwrap();
    h.view_row(&mut ops, &b);
    h.call(&mut ops, "drop");
    b.release(&mut ops);
    h.flush(&mut ops);

    // e. The pin fails during allocate.
    h.begin_script(&mut ops, "e-alloc-pin-fail");
    h.call(&mut ops, "allocate(8,8,1) [pin will fail]");
    ops.fail_next_malloc = true;
    match Ws::allocate(&mut ops, 8, 8, 1) {
        // The success arm only runs when the injected failure DIDN'T land;
        // releasing keeps Drop quiet so the hash mismatch is the report.
        Ok(mut w) => {
            w.release(&mut ops);
            ops.cuda_log.push("no-throw".into());
        }
        Err(_) => ops.cuda_log.push("threw".into()),
    }
    h.flush(&mut ops);

    // f. The event create fails during allocate; the pin is freed.
    h.begin_script(&mut ops, "f-alloc-event-fail");
    h.call(&mut ops, "allocate(8,8,1) [event will fail]");
    ops.fail_next_event = true;
    match Ws::allocate(&mut ops, 8, 8, 1) {
        Ok(mut w) => {
            w.release(&mut ops);
            ops.cuda_log.push("no-throw".into());
        }
        Err(_) => ops.cuda_log.push("threw".into()),
    }
    h.flush(&mut ops);

    // g. The lazy pin fails mid-rotation; the machine keeps working.
    h.begin_script(&mut ops, "g-lazy-pin-fail");
    ops.name_stream(s_a, "sA");
    ops.name_stream(s_b, "sB");
    h.call(&mut ops, "allocate(8,8,2)");
    let mut ws = Ws::allocate(&mut ops, 8, 8, 2).unwrap();
    h.name_buffers(&ws);
    h.call(&mut ops, "begin");
    ws.begin_plan_update(&mut ops).unwrap();
    h.call(&mut ops, "end(sA)");
    ws.end_plan_update(&mut ops, s_a);
    h.call(&mut ops, "begin [pin will fail]");
    ops.fail_next_malloc = true;
    let r = ws.begin_plan_update(&mut ops);
    h.note_result(&mut ops, r);
    h.call(&mut ops, "begin");
    ws.begin_plan_update(&mut ops).unwrap();
    h.view_row(&mut ops, &ws);
    h.call(&mut ops, "end(sB)");
    ws.end_plan_update(&mut ops, s_b);
    h.call(&mut ops, "drop");
    ws.release(&mut ops);
    h.flush(&mut ops);

    // h. The lazy event create fails after its pin succeeded; teardown
    //    frees the orphan pin.
    h.begin_script(&mut ops, "h-lazy-event-fail");
    ops.name_stream(s_a, "sA");
    h.call(&mut ops, "allocate(8,8,2)");
    let mut ws = Ws::allocate(&mut ops, 8, 8, 2).unwrap();
    h.name_buffers(&ws);
    h.call(&mut ops, "begin");
    ws.begin_plan_update(&mut ops).unwrap();
    h.call(&mut ops, "end(sA)");
    ws.end_plan_update(&mut ops, s_a);
    h.call(&mut ops, "begin [event will fail]");
    ops.fail_next_event = true;
    let r = ws.begin_plan_update(&mut ops);
    h.note_result(&mut ops, r);
    h.call(&mut ops, "begin");
    ws.begin_plan_update(&mut ops).unwrap();
    h.view_row(&mut ops, &ws);
    h.call(&mut ops, "drop");
    ws.release(&mut ops);
    h.flush(&mut ops);

    h.out
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
        let path = std::env::temp_dir().join("attn_ws_rust_transcript.txt");
        std::fs::write(&path, &text).ok();
        panic!(
            "transcript hash 0x{hash:016x} != golden 0x{GOLDEN_FNV1A64:016x}; \
             rust transcript dumped to {}",
            path.display()
        );
    }
}
