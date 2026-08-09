//! Behavioural parity with the C++ lora staging — gate-lora slice A.
//!
//! The oracle in `tests/oracle/lora_stage/` compiles the real
//! `llama_like.cpp` and drives `stage_qkv_adapters` over lane tables:
//! the validation chain, the arena discipline, the casts, the grouping,
//! the pointer slab, and the fingerprint. This test replays the same
//! cases — twice, the second pass with the grouped lowering off, exactly
//! as run.sh sweeps `PIE_LORA_GROUPED=0` across processes — and requires
//! the transcripts to be byte-identical.
//!
//! Run `tests/oracle/lora_stage/run.sh` to regenerate
//! [`GOLDEN_FNV1A64`]. The pinned value is the **C++'s** hash.
//!
//! Addresses are fabricated with the ORACLE'S OWN assignment rules (the
//! workspace views' bases derive from the registry's size, the arena
//! blocks from an ordinal) because the fingerprint MIXES addresses — the
//! value is only comparable if both sides compute over the same numbers.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt::Write as _;

use driver_cuda::fire::lora::{
    LORA_SITE_K, LORA_SITE_Q, LORA_SITE_V, LoraFireState, LoraForm, LoraLaneView, LoraOps,
    LoraStageArena, LoraStageRows, LoraTable, stage_qkv_adapters,
};
use driver_cuda::fire::sideband_arena::DeviceMemory;

/// FNV-1a 64 of the C++ oracle's transcript.
const GOLDEN_FNV1A64: u64 = 0x859d5a2da5642b23;

/// Rows the transcript must contain, so a truncated sweep cannot pass.
const GOLDEN_ROWS: usize = 148;

const SEP: char = '\u{1f}';

struct Recorder {
    out: String,
    case: String,
    prefix: &'static str,
    regions: BTreeMap<usize, (String, usize)>,
    next_dev: usize,
}

impl Recorder {
    fn new() -> Self {
        Self {
            out: String::new(),
            case: String::new(),
            prefix: "",
            regions: BTreeMap::new(),
            next_dev: 0,
        }
    }

    fn note(&mut self, body: &str) {
        let (prefix, case) = (self.prefix, self.case.clone());
        let _ = writeln!(self.out, "{prefix}{case}{SEP}{body}");
    }

    fn name_region(&mut self, addr: usize, bytes: usize, name: String) {
        self.regions.insert(addr, (name, bytes));
    }

    fn where_of(&self, p: *const c_void) -> String {
        if p.is_null() {
            return "null".into();
        }
        let addr = p as usize;
        if let Some((base, (name, len))) = self.regions.range(..=addr).next_back()
            && addr - base < *len
        {
            return format!("{name}+{}", addr - base);
        }
        "unknown".into()
    }
}

impl DeviceMemory for Recorder {
    fn alloc(&mut self, bytes: usize) -> Option<*mut c_void> {
        if bytes == 0 {
            return None;
        }
        let p = 0x1000_0000usize + self.next_dev * 0x0100_0000;
        self.name_region(p, bytes, format!("dev#{}", self.next_dev));
        self.next_dev += 1;
        self.note(&format!(
            "dev-alloc dev#{} bytes={bytes}",
            self.next_dev - 1
        ));
        Some(p as *mut c_void)
    }

    fn free(&mut self, _ptr: *mut c_void) {}

    fn synchronize(&mut self) -> bool {
        true
    }
}

impl LoraOps for Recorder {
    fn cast_fp32_to_bf16(&mut self, src: *const c_void, dst: *mut c_void, elems: usize) {
        let body = format!(
            "cast src={} dst={} elems={elems}",
            self.where_of(src),
            self.where_of(dst.cast_const())
        );
        self.note(&body);
    }

    fn upload_slab(&mut self, dst: *mut c_void, slots: &[*const c_void]) {
        let mut body = format!(
            "slab dst={} kind=1 slots=[",
            self.where_of(dst.cast_const())
        );
        for (i, &p) in slots.iter().enumerate() {
            if i > 0 {
                body.push(',');
            }
            body.push_str(&self.where_of(p));
        }
        body.push(']');
        self.note(&body);
    }
}

/// The oracle's fixture, address rules included.
struct Fixture {
    rows: LoraStageRows,
    arena: LoraStageArena,
    staged: Option<LoraFireState>,
    hk: i32,
}

const K_H: i32 = 8;
const K_HQ: i32 = 12;
const K_HK: i32 = 4;
const K_I: i32 = 16;
const K_LAYERS: i32 = 2;

impl Fixture {
    fn new(r: &mut Recorder, max_tokens: i32, kv_width: i32) -> Self {
        let mut view = |name: &str, width: i32| -> *mut c_void {
            let base = 0x1000 * (1 + r.regions.len());
            let bytes = max_tokens as usize * width as usize * 2;
            r.name_region(base, bytes, name.to_string());
            base as *mut c_void
        };
        let y = view("ws.y", K_H);
        let norm_x = view("ws.norm_x", K_H);
        let q = view("ws.q", K_HQ);
        let v = view("ws.v", kv_width);
        let gate = view("ws.gate", K_I);
        Self {
            rows: LoraStageRows {
                y: y.cast_const(),
                norm_x: norm_x.cast_const(),
                q,
                v,
                gate,
            },
            arena: LoraStageArena::default(),
            staged: None,
            hk: kv_width,
        }
    }
}

fn adapter(r: &mut Recorder, name: &str, ordinal: usize) -> *const c_void {
    let p = 0x4000_0000usize + ordinal * 0x0010_0000;
    r.name_region(p, 0x0010_0000, name.to_string());
    p as *const c_void
}

#[allow(clippy::too_many_arguments)]
fn lane(
    a: *const c_void,
    b: *const c_void,
    sites: u64,
    start: u32,
    count: u32,
    layers: u32,
    rank: u32,
    d_in: u32,
    d_out: u32,
    form: LoraForm,
) -> LoraLaneView {
    LoraLaneView {
        a,
        b,
        sites_bits: sites,
        token_start: start,
        token_count: count,
        num_layers: layers,
        rank,
        d_in,
        d_out,
        form,
    }
}

fn run_stage(
    r: &mut Recorder,
    f: &mut Fixture,
    name: &str,
    lanes: &[LoraLaneView],
    total_tokens: i32,
    grouped: bool,
) {
    r.case = name.to_string();
    r.note(&format!("call lanes={} N={total_tokens}", lanes.len()));
    let table = LoraTable { lanes };
    match stage_qkv_adapters(
        r,
        &mut f.arena,
        Some(&table),
        K_LAYERS,
        total_tokens,
        K_H,
        K_HQ,
        f.hk,
        K_I,
        1,
        false,
        &f.rows,
        grouped,
    ) {
        Ok((fp, staged)) => {
            f.staged = staged;
            let desc = f
                .staged
                .as_ref()
                .map_or_else(|| "-".to_string(), LoraFireState::grouping_desc);
            r.note(&format!(
                "staged fp=0x{fp:016x} handle={} table={} desc={desc}",
                if f.staged.is_some() { "set" } else { "null" },
                "this",
            ));
        }
        Err(e) => r.note(&format!("threw {e}")),
    }
}

#[allow(clippy::too_many_lines)]
fn sweep(r: &mut Recorder, grouped: bool) {
    let q = LORA_SITE_Q;
    let v = LORA_SITE_V;
    let k = LORA_SITE_K;
    let lr = LoraForm::LowRank;

    // a. The no-program paths.
    {
        let mut f = Fixture::new(r, 16, K_HK);
        r.case = "a-null".into();
        let (fp, staged) = stage_qkv_adapters(
            r,
            &mut f.arena,
            None,
            K_LAYERS,
            16,
            K_H,
            K_HQ,
            K_HK,
            K_I,
            1,
            false,
            &f.rows,
            grouped,
        )
        .expect("null stages to nothing");
        r.note(&format!(
            "fp={fp} handle={}",
            if staged.is_some() { "set" } else { "null" }
        ));
        let empty = LoraTable { lanes: &[] };
        let (fp2, _) = stage_qkv_adapters(
            r,
            &mut f.arena,
            Some(&empty),
            K_LAYERS,
            16,
            K_H,
            K_HQ,
            K_HK,
            K_I,
            1,
            false,
            &f.rows,
            grouped,
        )
        .expect("empty stages to nothing");
        r.note(&format!("empty fp={fp2}"));
    }

    // b. One low-rank lane on q.
    {
        let mut f = Fixture::new(r, 16, K_HK);
        let a0 = adapter(r, "A0", 0);
        let b0 = adapter(r, "B0", 1);
        let lanes = [lane(a0, b0, q, 0, 8, 2, 2, 8, 12, lr)];
        run_stage(r, &mut f, "b-solo", &lanes, 16, grouped);
    }

    // b2. Re-staging the same state: arena reset, no regrowth.
    {
        let mut f = Fixture::new(r, 16, K_HK);
        let a0 = adapter(r, "A0", 0);
        let b0 = adapter(r, "B0", 1);
        let lanes = [lane(a0, b0, q, 0, 8, 2, 2, 8, 12, lr)];
        run_stage(r, &mut f, "b2-first", &lanes, 16, grouped);
        run_stage(r, &mut f, "b2-restage", &lanes, 16, grouped);
    }

    // c. Two same-shape disjoint lanes group.
    {
        let mut f = Fixture::new(r, 16, K_HK);
        let (a0, b0) = (adapter(r, "A0", 0), adapter(r, "B0", 1));
        let (a1, b1) = (adapter(r, "A1", 2), adapter(r, "B1", 3));
        let lanes = [
            lane(a0, b0, q, 0, 6, 2, 2, 8, 12, lr),
            lane(a1, b1, q, 6, 4, 2, 2, 8, 12, lr),
        ];
        run_stage(r, &mut f, "c-grouped", &lanes, 16, grouped);
    }

    // c2. Mixed q and v members in one group (Hq == Hk fixture).
    {
        let mut f = Fixture::new(r, 16, K_HQ);
        let (a0, b0) = (adapter(r, "A0", 0), adapter(r, "B0", 1));
        let (a1, b1) = (adapter(r, "A1", 2), adapter(r, "B1", 3));
        let lanes = [
            lane(a0, b0, q, 0, 6, 2, 2, 8, 12, lr),
            lane(a1, b1, v, 6, 4, 2, 2, 8, 12, lr),
        ];
        run_stage(r, &mut f, "c2-grouped-qv", &lanes, 16, grouped);
    }

    // d / d2. Shapes differ; a scale lane never groups.
    {
        let mut f = Fixture::new(r, 16, K_HK);
        let (a0, b0) = (adapter(r, "A0", 0), adapter(r, "B0", 1));
        let (a1, b1) = (adapter(r, "A1", 2), adapter(r, "B1", 3));
        let lanes = [
            lane(a0, b0, q, 0, 6, 2, 2, 8, 12, lr),
            lane(a1, b1, q, 6, 4, 2, 4, 8, 12, lr),
        ];
        run_stage(r, &mut f, "d-shapes-differ", &lanes, 16, grouped);
    }
    {
        let mut f = Fixture::new(r, 16, K_HK);
        let l0 = adapter(r, "L0", 0);
        let (a1, b1) = (adapter(r, "A1", 2), adapter(r, "B1", 3));
        let lanes = [
            lane(l0, std::ptr::null(), v, 0, 6, 2, 0, 0, 4, LoraForm::Scale),
            lane(a1, b1, q, 6, 4, 2, 2, 8, 12, lr),
        ];
        run_stage(r, &mut f, "d2-scale-lane", &lanes, 16, grouped);
    }

    // e. Overlapping spans fall back to per-lane pairs.
    {
        let mut f = Fixture::new(r, 16, K_HK);
        let (a0, b0) = (adapter(r, "A0", 0), adapter(r, "B0", 1));
        let (a1, b1) = (adapter(r, "A1", 2), adapter(r, "B1", 3));
        let lanes = [
            lane(a0, b0, q, 0, 8, 2, 2, 8, 12, lr),
            lane(a1, b1, q, 4, 8, 2, 2, 8, 12, lr),
        ];
        run_stage(r, &mut f, "e-overlap", &lanes, 16, grouped);
    }

    // f. A zero-count lane is silently dropped.
    {
        let mut f = Fixture::new(r, 16, K_HK);
        let (a0, b0) = (adapter(r, "A0", 0), adapter(r, "B0", 1));
        let (a1, b1) = (adapter(r, "A1", 2), adapter(r, "B1", 3));
        let lanes = [
            lane(a0, b0, q, 0, 0, 2, 2, 8, 12, lr),
            lane(a1, b1, q, 0, 8, 2, 2, 8, 12, lr),
        ];
        run_stage(r, &mut f, "f-empty-span", &lanes, 16, grouped);
    }

    // g. The refusal chain.
    {
        let mut f = Fixture::new(r, 16, K_HK);
        let (a0, b0) = (adapter(r, "A0", 0), adapter(r, "B0", 1));
        let cases: [(&str, LoraLaneView); 9] = [
            (
                "g1-null-adapter",
                lane(std::ptr::null(), b0, q, 0, 8, 2, 2, 8, 12, lr),
            ),
            ("g2-no-sites", lane(a0, b0, 0, 0, 8, 2, 2, 8, 12, lr)),
            (
                "g3-unknown-bits",
                lane(a0, b0, 1 << 9, 0, 8, 2, 2, 8, 12, lr),
            ),
            ("g4-reserved-site", lane(a0, b0, k, 0, 8, 2, 2, 8, 12, lr)),
            ("g5-layers", lane(a0, b0, q, 0, 8, 7, 2, 8, 12, lr)),
            ("g6-d-in", lane(a0, b0, q, 0, 8, 2, 2, 5, 12, lr)),
            ("g7-d-out", lane(a0, b0, q, 0, 8, 2, 2, 8, 5, lr)),
            ("g8-rank", lane(a0, b0, q, 0, 8, 2, 17, 8, 12, lr)),
            ("g9-span", lane(a0, b0, q, 10, 8, 2, 2, 8, 12, lr)),
        ];
        for (name, l) in cases {
            let lanes = [l];
            run_stage(r, &mut f, name, &lanes, 16, grouped);
        }
    }
}

fn transcript() -> String {
    let mut r = Recorder::new();
    sweep(&mut r, true);
    // The off-process sweep: fresh registries, fresh ordinals, exactly as
    // a new process has, and run.sh's "off:" prefix.
    let mut off = Recorder::new();
    off.prefix = "off:";
    sweep(&mut off, false);
    r.out + &off.out
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
    assert_eq!(
        rows, GOLDEN_ROWS,
        "row count diverged — sweep shape changed"
    );
    let hash = fnv1a64(text.as_bytes());
    if hash != GOLDEN_FNV1A64 {
        let path = std::env::temp_dir().join("lora_stage_rust_transcript.txt");
        std::fs::write(&path, &text).ok();
        panic!(
            "transcript hash 0x{hash:016x} != golden 0x{GOLDEN_FNV1A64:016x}; \
             rust transcript dumped to {}",
            path.display()
        );
    }
}
