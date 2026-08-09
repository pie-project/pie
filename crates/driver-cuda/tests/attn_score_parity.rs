//! Behavioural parity with the C++ score captures — gate-score-capture.
//!
//! The oracle in `tests/oracle/attn_score/` compiles the real
//! `model/attn_score.cu` against the real `hook_sideband_arena.cpp` and
//! drives both captures over their guard chains, CSR arithmetic and arena
//! traffic; the recorders print slot offsets as `blk#K+N` and the CSR
//! upload by CONTENT. This test replays the same scripts against the port
//! — including the REAL Rust [`SidebandArena`], so the slot carve is the
//! two arenas agreeing, not a mock agreeing with itself — and requires the
//! transcripts to be byte-identical, the per-process window sweep
//! included (here the pure `_from` form answers it).
//!
//! Run `tests/oracle/attn_score/run.sh` to regenerate [`GOLDEN_FNV1A64`].
//! The pinned value is the **C++'s** hash — see that constant for why the
//! regenerator no longer runs and what that costs.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt::Write as _;

use driver_cuda::fire::attn_score::{
    AttentionObservation, AttentionScores, LayerPrefillScoreCapture, LayerScoreCapture,
    ScoreHookView, ScoreOps, ScoreScratch, default_attn_score_window_from,
    prepare_decode_score_capture,
};
use driver_cuda::fire::sideband_arena::{DeviceMemory, SidebandArena};

/// FNV-1a 64 of the C++ oracle's transcript.
///
/// # This can no longer be regenerated, and that is a fact about the tree
///
/// `tests/oracle/attn_score/run.sh` compiles `driver-cuda/csrc/src/model/
/// attn_score.cu` against `hook_sideband_arena.cpp`. **Neither file exists**:
/// commit `4569b9e4b` ("Delete crates/driver-cuda") removed the whole
/// directory, and the `driver-cuda` in the tree today is the Rust rewrite.
/// `run.sh` starts `set -euo pipefail` and its first `cp` fails, so the
/// script cannot produce a number for anyone.
///
/// The value below is therefore FROZEN. It is still worth having — it pins
/// the port against the last C++ that existed, which is the whole claim the
/// test makes — but it can only ever be *preserved*, never re-derived, and
/// any change that adds a row to the transcript makes this test permanently
/// red rather than one command stale.
///
/// **That is why the three capture post-kernels record nothing.** Their
/// launches moved into `LayerScoreCapture::publish` and
/// `LayerPrefillScoreCapture::publish` (see `fire::attn_score`'s header),
/// but they were the tail of `attention_flashinfer.cu`'s capture dispatch —
/// a translation unit this oracle never compiled and whose launches were
/// never in this transcript. Recording them would compare the Rust against a
/// C++ program that did not contain them, using a golden nobody can rebuild.
/// So [`Recorder`]'s three new methods are silent, the omission is stated at
/// each one, and the geometry those launches carry is pinned instead by
/// `tests/attn_score_post_geometry.rs`, which needs no oracle because it is
/// a transcription check rather than a behavioural one.
const GOLDEN_FNV1A64: u64 = 0x132d0dce71b9a15c;

/// Rows the transcript must contain, so a truncated sweep cannot pass.
const GOLDEN_ROWS: usize = 68;

const SEP: char = '\u{1f}';

/// The fixture's device CSRs, matching [`Fire::obs`]'s literals. `QO_D` has
/// no counterpart in the observation — the C++ capture dispatch took the
/// query CSR as its own argument — so it is a fourth fixed address.
const QO_D: *const u32 = 0x40 as *const u32;
const KVPP_D: *const u32 = 0x20 as *const u32;
const LENS_D: *const u32 = 0x30 as *const u32;

/// The oracle's recorders: one region registry serving `where()`, shared
/// by the arena's memory calls and the stream ops — which is why the port
/// takes ONE `DeviceMemory + ScoreOps` value.
#[derive(Default)]
struct Recorder {
    out: String,
    case: String,
    regions: BTreeMap<usize, (String, usize)>,
    next_block: usize,
    next_addr: usize,
}

impl Recorder {
    fn new() -> Self {
        Self {
            next_addr: 0x1000_0000,
            ..Self::default()
        }
    }

    fn note(&mut self, body: &str) {
        let case = self.case.clone();
        let _ = writeln!(self.out, "{case}{SEP}{body}");
    }

    fn name_fixed(&mut self, addr: usize, name: &str) {
        self.regions.insert(addr, (name.to_string(), 1));
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
        let p = self.next_addr;
        self.next_addr += bytes.max(1).next_multiple_of(0x1000) + 0x1000;
        let name = format!("blk#{}", self.next_block);
        self.next_block += 1;
        self.regions.insert(p, (name.clone(), bytes.max(1)));
        self.note(&format!("malloc {name} bytes={bytes}"));
        Some(p as *mut c_void)
    }

    fn free(&mut self, ptr: *mut c_void) {
        if !ptr.is_null() {
            let w = self.where_of(ptr.cast_const());
            self.note(&format!("free {w}"));
            self.regions.remove(&(ptr as usize));
        }
    }

    fn synchronize(&mut self) -> bool {
        self.note("sync");
        true
    }
}

impl ScoreOps for Recorder {
    fn memset_async(&mut self, dst: *mut u8, value: u8, bytes: usize) {
        let w = self.where_of(dst.cast_const().cast());
        self.note(&format!("memset {w} val={value} len={bytes}"));
    }

    fn upload_csr(&mut self, dst: *mut i32, src: &[i32]) {
        let w = self.where_of(dst.cast_const().cast());
        let csr = src
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        self.note(&format!("upload dst={w} kind=1 csr=[{csr}]"));
    }

    // ── the three that record NOTHING, and why ────────────────────────
    //
    // These launches are new to `publish()`; they are not new to the driver.
    // They were the tail of `attention_flashinfer.cu`'s two capture
    // dispatches, and this oracle compiles `model/attn_score.cu` — a
    // different translation unit, which never issued them. The golden is a
    // transcript of THAT program and cannot be rebuilt (see
    // `GOLDEN_FNV1A64`), so emitting a row here would not be a stale hash,
    // it would be a permanent divergence from a frozen number.
    //
    // Recording nothing is therefore the honest reading of what this test
    // claims: "the port reproduces the C++ transcript" is a claim about
    // `attn_score.cu`, and these three were never in it. What is NOT claimed
    // — that their operands and geometry are right — is claimed by
    // `attn_score_post_geometry.rs` instead, and that file exists because
    // this one cannot make the claim.

    fn normalize_decode(
        &mut self,
        _scores: *mut f32,
        _score_indptr_d: *const i32,
        _kv_page_indptr_d: *const u32,
        _kv_last_page_lens_d: *const u32,
        _page_size: i32,
        _num_requests: i32,
        _num_q_heads: i32,
    ) {
    }

    fn normalize_prefill(
        &mut self,
        _scores: *mut f32,
        _score_indptr_d: *const i32,
        _qo_indptr_d: *const u32,
        _kv_page_indptr_d: *const u32,
        _kv_last_page_lens_d: *const u32,
        _page_size: i32,
        _num_requests: i32,
        _num_q_heads: i32,
        _window: i32,
    ) {
    }

    fn fold_prefill(
        &mut self,
        _scores: *const f32,
        _folded: *mut f32,
        _score_indptr_d: *const i32,
        _qo_indptr_d: *const u32,
        _kv_page_indptr_d: *const u32,
        _kv_last_page_lens_d: *const u32,
        _page_size: i32,
        _num_requests: i32,
        _num_q_heads: i32,
        _window: i32,
    ) {
    }

    fn fold_heads(
        &mut self,
        raw: *const f32,
        score_indptr_d: *const i32,
        kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
        folded: *mut f32,
    ) {
        let body = format!(
            "fold raw={} indptr={} kvpp={} lens={} psz={page_size} R={num_requests} \
             qh={num_q_heads} out={}",
            self.where_of(raw.cast()),
            self.where_of(score_indptr_d.cast()),
            self.where_of(kv_page_indptr_d.cast()),
            self.where_of(kv_last_page_lens_d.cast()),
            self.where_of(folded.cast_const().cast()),
        );
        self.note(&body);
    }
}

/// The oracle's fire fixture.
struct Fire {
    kvpp_h: Vec<u32>,
    lens_h: Vec<u32>,
    qo_h: Vec<u32>,
}

impl Fire {
    fn new(kvpp: Vec<u32>, lens: Vec<u32>) -> Self {
        let requests = u32::try_from(kvpp.len() - 1).unwrap();
        Self {
            kvpp_h: kvpp,
            lens_h: lens,
            qo_h: (0..=requests).collect(),
        }
    }

    fn obs(&self, page_size: i32) -> AttentionObservation<'_> {
        AttentionObservation {
            kv_page_size: Some(page_size),
            kv_page_indices_d: 0x10 as *const u32,
            kv_page_indptr_d: 0x20 as *const u32,
            kv_last_page_lens_d: 0x30 as *const u32,
            qo_indptr_h: Some(&self.qo_h),
            kv_page_indptr_h: Some(&self.kvpp_h),
            kv_last_page_lens_h: Some(&self.lens_h),
            num_requests: i32::try_from(self.kvpp_h.len() - 1).unwrap(),
            total_tokens: i32::try_from(self.kvpp_h.len() - 1).unwrap(),
        }
    }
}

fn hooks<'a>(obs: &'a AttentionObservation<'a>) -> ScoreHookView<'a> {
    ScoreHookView {
        wants_attn_score: true,
        observation: Some(obs),
    }
}

fn payload_row(r: &mut Recorder, label: &str, s: Option<&AttentionScores>, expect: u32) {
    let Some(s) = s else {
        r.note(&format!("{label} payload=null"));
        return;
    };
    // SAFETY: the offsets point into the scratch, which outlives the case.
    let offs = unsafe {
        std::slice::from_raw_parts(s.offsets_h, usize::try_from(s.num_requests).unwrap() + 1)
    };
    let joined = offs
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(
        "{label} payload values={} offsets=[{joined}] R={} layer={} usable={} expectR={expect}",
        r.where_of(s.values.cast()),
        s.num_requests,
        s.layer,
        u8::from(s.usable()),
    );
    r.note(&body);
}

#[allow(clippy::too_many_lines)]
fn transcript() -> String {
    let mut r = Recorder::new();
    r.name_fixed(0x20, "kvpp_d");
    r.name_fixed(0x30, "lens_d");
    let mut arena = SidebandArena::new();
    let mut scratch = ScoreScratch::default();

    // a. The decode capture's happy path, twice on one fire.
    r.case = "a-decode".into();
    r.note("case-begin");
    {
        let f = Fire::new(vec![0, 2, 5, 6], vec![3, 16, 1]);
        let obs = f.obs(16);
        let h = hooks(&obs);
        for layer in [0u32, 7] {
            let mut cap = LayerScoreCapture::new(
                &mut r,
                Some(&mut arena),
                &mut scratch,
                Some(&h),
                layer,
                4,
                true,
            );
            let body = format!(
                "L{layer} active={} raw={} indptr={}",
                u8::from(cap.active()),
                r.where_of(cap.raw().cast_const().cast()),
                r.where_of(cap.indptr_d().cast()),
            );
            r.note(&body);
            payload_row(&mut r, "pre-publish", cap.scores(), 3);
            cap.publish(
                &mut r,
                Some(&h),
                obs.kv_page_indptr_d,
                obs.kv_last_page_lens_d,
                16,
            )
            .unwrap();
            payload_row(&mut r, "published", cap.scores(), 3);
            cap.publish(
                &mut r,
                Some(&h),
                obs.kv_page_indptr_d,
                obs.kv_last_page_lens_d,
                16,
            )
            .unwrap();
            cap.release(&mut arena, &mut scratch);
        }
    }

    // b. The guard chain.
    r.case = "b-guards".into();
    r.note("case-begin");
    {
        let f = Fire::new(vec![0, 2], vec![3]);
        let obs = f.obs(16);
        let refuse = |r: &mut Recorder,
                      scratch: &mut ScoreScratch,
                      arena: &mut SidebandArena,
                      label: &str,
                      h: Option<&ScoreHookView<'_>>,
                      heads: u32,
                      capturable: bool| {
            let mut cap = LayerScoreCapture::new(r, Some(arena), scratch, h, 0, heads, capturable);
            r.note(&format!("{label} active={}", u8::from(cap.active())));
            cap.release(arena, scratch);
        };
        refuse(
            &mut r,
            &mut scratch,
            &mut arena,
            "null-hooks",
            None,
            4,
            true,
        );
        let unwanted = ScoreHookView {
            wants_attn_score: false,
            observation: Some(&obs),
        };
        refuse(
            &mut r,
            &mut scratch,
            &mut arena,
            "unwanted",
            Some(&unwanted),
            4,
            true,
        );
        let h = hooks(&obs);
        refuse(
            &mut r,
            &mut scratch,
            &mut arena,
            "uncapturable",
            Some(&h),
            4,
            false,
        );
        refuse(
            &mut r,
            &mut scratch,
            &mut arena,
            "zero-heads",
            Some(&h),
            0,
            true,
        );
        let no_obs = ScoreHookView {
            wants_attn_score: true,
            observation: None,
        };
        refuse(
            &mut r,
            &mut scratch,
            &mut arena,
            "no-obs",
            Some(&no_obs),
            4,
            true,
        );
        let empty = Fire::new(vec![0, 0, 0], vec![0, 0]);
        let empty_obs = empty.obs(16);
        let eh = hooks(&empty_obs);
        refuse(
            &mut r,
            &mut scratch,
            &mut arena,
            "all-empty",
            Some(&eh),
            4,
            true,
        );
    }

    // c. Nested captures: the inner one stands down.
    r.case = "c-nested".into();
    r.note("case-begin");
    {
        let f = Fire::new(vec![0, 2, 5], vec![3, 16]);
        let obs = f.obs(16);
        let h = hooks(&obs);
        let mut outer =
            LayerScoreCapture::new(&mut r, Some(&mut arena), &mut scratch, Some(&h), 0, 2, true);
        let mut inner =
            LayerScoreCapture::new(&mut r, Some(&mut arena), &mut scratch, Some(&h), 0, 2, true);
        let body = format!(
            "outer={} inner={}",
            u8::from(outer.active()),
            u8::from(inner.active())
        );
        r.note(&body);
        inner.release(&mut arena, &mut scratch);
        outer.release(&mut arena, &mut scratch);
    }

    // c2. What the depth guard protects: a refused inner capture with
    //     DIFFERENT geometry must not clobber the scratch the outer
    //     capture's published offsets point into.
    r.case = "c2-nested-clobber".into();
    r.note("case-begin");
    {
        let fa = Fire::new(vec![0, 2, 5, 6], vec![3, 16, 1]);
        let fb = Fire::new(vec![0, 9], vec![4]);
        let obs_a = fa.obs(16);
        let obs_b = fb.obs(16);
        let ha = hooks(&obs_a);
        let hb = hooks(&obs_b);
        let mut outer = LayerScoreCapture::new(
            &mut r,
            Some(&mut arena),
            &mut scratch,
            Some(&ha),
            0,
            2,
            true,
        );
        {
            let mut inner = LayerScoreCapture::new(
                &mut r,
                Some(&mut arena),
                &mut scratch,
                Some(&hb),
                0,
                2,
                true,
            );
            r.note(&format!("inner={}", u8::from(inner.active())));
            inner.release(&mut arena, &mut scratch);
        }
        outer
            .publish(
                &mut r,
                Some(&ha),
                obs_a.kv_page_indptr_d,
                obs_a.kv_last_page_lens_d,
                16,
            )
            .unwrap();
        payload_row(&mut r, "outer", outer.scores(), 3);
        outer.release(&mut arena, &mut scratch);
    }

    // d. Publish after the fire geometry is torn down.
    r.case = "d-publish-lost-geometry".into();
    r.note("case-begin");
    {
        let f = Fire::new(vec![0, 2, 5], vec![3, 16]);
        let obs = f.obs(16);
        let h = hooks(&obs);
        let mut cap =
            LayerScoreCapture::new(&mut r, Some(&mut arena), &mut scratch, Some(&h), 3, 2, true);
        let gone = ScoreHookView {
            wants_attn_score: true,
            observation: None,
        };
        match cap.publish(
            &mut r,
            Some(&gone),
            obs.kv_page_indptr_d,
            obs.kv_last_page_lens_d,
            16,
        ) {
            Ok(()) => r.note("no-throw"),
            Err(_) => r.note("threw"),
        }
        cap.release(&mut arena, &mut scratch);
    }

    // e. The u32 ceiling on raw elements.
    r.case = "e-u32-ceiling".into();
    r.note("case-begin");
    {
        let f = Fire::new(vec![0, 2_000_000], vec![256]);
        let obs = f.obs(256);
        let h = hooks(&obs);
        let mut cap =
            LayerScoreCapture::new(&mut r, Some(&mut arena), &mut scratch, Some(&h), 0, 9, true);
        let body = format!("active={}", u8::from(cap.active()));
        r.note(&body);
        cap.release(&mut arena, &mut scratch);
    }

    // f. `prepare_decode_score_capture`.
    r.case = "f-prepare".into();
    r.note("case-begin");
    {
        let f = Fire::new(vec![0, 2, 5, 6], vec![3, 16, 1]);
        let obs = f.obs(16);
        let plan = prepare_decode_score_capture(&mut r, Some(&mut arena), &mut scratch, &obs, 4);
        // SAFETY: the plan's host pointers point into the scratch.
        let (csr_h, folded_h) = unsafe {
            let n = usize::try_from(plan.num_requests).unwrap() + 1;
            (
                std::slice::from_raw_parts(plan.indptr_h_data, n),
                std::slice::from_raw_parts(plan.folded_offsets_h, n),
            )
        };
        let join_i = |v: &[i32]| {
            v.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        let join_u = |v: &[u32]| {
            v.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        let body = format!(
            "plan ok={} folded={} indptr={} csr_h=[{}] folded_h=[{}] R={}",
            u8::from(plan.ok),
            r.where_of(plan.folded.cast()),
            r.where_of(plan.indptr_d.cast()),
            join_i(csr_h),
            join_u(folded_h),
            plan.num_requests,
        );
        r.note(&body);
        let refused = prepare_decode_score_capture(&mut r, None, &mut scratch, &obs, 4);
        r.note(&format!("null-arena ok={}", u8::from(refused.ok)));
        let no_heads =
            prepare_decode_score_capture(&mut r, Some(&mut arena), &mut scratch, &obs, 0);
        r.note(&format!("zero-heads ok={}", u8::from(no_heads.ok)));
    }

    // g. The prefill capture.
    r.case = "g-prefill".into();
    r.note("case-begin");
    {
        let f = Fire::new(vec![0, 2, 5, 6], vec![3, 16, 1]);
        let obs = f.obs(16);
        let h = hooks(&obs);
        let mut cap = LayerPrefillScoreCapture::new(
            &mut r,
            Some(&mut arena),
            &mut scratch,
            Some(&h),
            11,
            4,
            8,
            true,
        );
        let body = format!(
            "active={} raw={} folded={} indptr={} window={}",
            u8::from(cap.active()),
            r.where_of(cap.raw().cast_const().cast()),
            r.where_of(cap.folded().cast_const().cast()),
            r.where_of(cap.indptr_d().cast()),
            cap.window(),
        );
        r.note(&body);
        payload_row(&mut r, "pre-publish", cap.scores(), 3);
        // The four device arguments the C++ dispatch passed its own tail;
        // the recorder is silent for both launches, so only the payload rows
        // either side of them reach the transcript.
        cap.publish(&mut r, QO_D, KVPP_D, LENS_D, 4);
        payload_row(&mut r, "published", cap.scores(), 3);
        cap.release(&mut arena, &mut scratch);
    }

    // h. The prefill ceilings.
    r.case = "h-prefill-ceilings".into();
    r.note("case-begin");
    {
        let big = Fire::new(vec![0, 2_000_000], vec![256]);
        let big_obs = big.obs(256);
        let bh = hooks(&big_obs);
        let mut cap = LayerPrefillScoreCapture::new(
            &mut r,
            Some(&mut arena),
            &mut scratch,
            Some(&bh),
            0,
            4,
            64,
            true,
        );
        let body = format!("huge active={}", u8::from(cap.active()));
        r.note(&body);
        cap.release(&mut arena, &mut scratch);
        let f = Fire::new(vec![0, 2], vec![3]);
        let obs = f.obs(16);
        let h = hooks(&obs);
        let mut no_window = LayerPrefillScoreCapture::new(
            &mut r,
            Some(&mut arena),
            &mut scratch,
            Some(&h),
            0,
            4,
            0,
            true,
        );
        let body = format!("zero-window active={}", u8::from(no_window.active()));
        r.note(&body);
        no_window.release(&mut arena, &mut scratch);
        let mut windowed = LayerPrefillScoreCapture::new(
            &mut r,
            Some(&mut arena),
            &mut scratch,
            Some(&h),
            0,
            4,
            8,
            false,
        );
        let body = format!("uncapturable active={}", u8::from(windowed.active()));
        r.note(&body);
        windowed.release(&mut arena, &mut scratch);
    }

    // i. Decode and prefill scratch are SEPARATE — and the shared arena
    //    slot is what refuses the decode capture inside a live prefill.
    r.case = "i-scratch-separation".into();
    r.note("case-begin");
    {
        let f = Fire::new(vec![0, 2, 5, 6], vec![3, 16, 1]);
        let obs = f.obs(16);
        let h = hooks(&obs);
        let mut pf = LayerPrefillScoreCapture::new(
            &mut r,
            Some(&mut arena),
            &mut scratch,
            Some(&h),
            0,
            2,
            4,
            true,
        );
        pf.publish(&mut r, QO_D, KVPP_D, LENS_D, 4);
        payload_row(&mut r, "prefill", pf.scores(), 3);
        {
            let mut dec = LayerScoreCapture::new(
                &mut r,
                Some(&mut arena),
                &mut scratch,
                Some(&h),
                1,
                2,
                true,
            );
            let body = format!("decode active={}", u8::from(dec.active()));
            r.note(&body);
            dec.release(&mut arena, &mut scratch);
        }
        payload_row(&mut r, "prefill-after-decode", pf.scores(), 3);
        pf.release(&mut arena, &mut scratch);
    }
    // The arena's teardown at end of main (the C++ destructor's free).
    arena.destroy(&mut r);

    // The per-process window sweep, answered by the pure form.
    for (label, value) in [
        ("w-unset", None),
        ("w-empty", Some("")),
        ("w-0", Some("0")),
        ("w--1", Some("-1")),
        ("w-33", Some("33")),
        ("w-4096", Some("4096")),
        ("w-4097", Some("4097")),
        ("w-abc", Some("abc")),
        ("w-1e3", Some("1e3")),
    ] {
        r.case = label.into();
        let w = default_attn_score_window_from(value.map(std::ffi::OsStr::new));
        r.note(&format!("window={w}"));
    }

    r.out
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
        "row count diverged — script shape changed"
    );
    let hash = fnv1a64(text.as_bytes());
    if hash != GOLDEN_FNV1A64 {
        let path = std::env::temp_dir().join("attn_score_rust_transcript.txt");
        std::fs::write(&path, &text).ok();
        panic!(
            "transcript hash 0x{hash:016x} != golden 0x{GOLDEN_FNV1A64:016x}; \
             rust transcript dumped to {}",
            path.display()
        );
    }
}
