//! The registry of hardware-validated shapes: one table of what ran right
//! on CS-3 and what is known to fail, each with its reason. The planner
//! picks a ring's micro-batches and CAP from it ([`ran`]), and a build's
//! parameters are checked against it before they compile ([`check`]).
//!
//! A shape is refused when a known failure matches it ([`KNOWN_BAD`], in
//! order), or when its design keeps a record of validated shapes
//! ([`RAN`]) for builds of its kind and none of them holds it. Adding a
//! validated shape is one more [`Ran`] row, with the run that showed it.

use super::codegen::Design;

type R<T> = std::result::Result<T, String>;

/// A build's shape as the registry weighs it, from its parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub design: Design,
    /// `B`, `HB`, `MMAX`, `CAP`.
    pub lanes: usize,
    pub head_batch: usize,
    pub mmax: usize,
    pub cap: usize,
    /// serve2d's one-sequence grids (`GRID` 0).
    pub one_sequence: bool,
    /// A ring build (serve2d_pipe with `B`, serve8b_region_pipe's `PIPE` 1):
    /// the shapes the registry records.
    pub ring: bool,
    pub vm: Vm,
    /// `VM` 1: the VM's program on the head's VM columns.
    pub vm_on: bool,
    /// serve8b_region's VM block (`VMX`), KV staging (`KVST`), grid rows.
    pub vm_block: bool,
    pub staging: usize,
    pub py: usize,
}

/// The eta VM a shape carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vm {
    Off,
    /// Set A (and B), by turns.
    Sets,
    /// Both sets at once (`VMCONC`): set A's and set B's lanes, and whether
    /// a set B exists.
    Concurrent {
        lanes: (usize, usize),
        set_b: bool,
    },
}

impl Shape {
    /// The shape of `design` built with `params` (a parameter absent reads
    /// as 0).
    #[must_use]
    pub fn of(design: Design, params: &[(String, String)]) -> Shape {
        let get = |k: &str| params.iter().find(|(n, _)| n == k).map(|(_, v)| v.trim());
        let num = |k: &str| get(k).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let vm = if num("VMCONC") == 1 {
            Vm::Concurrent {
                lanes: (num("vm_lanes"), num("b_vm_lanes")),
                set_b: num("VM") == 1 && num("VMB") > 0,
            }
        } else if num("VM") == 1 {
            Vm::Sets
        } else {
            Vm::Off
        };
        Shape {
            design,
            lanes: num("B"),
            head_batch: num("HB"),
            mmax: num("MMAX"),
            cap: num("CAP"),
            one_sequence: get("GRID") == Some("0"),
            ring: match design {
                Design::Serve2dPipe => get("B").is_some(),
                Design::Serve8bRegionPipe => num("PIPE") == 1,
                _ => false,
            },
            vm,
            vm_on: num("VM") == 1,
            vm_block: num("VMX") != 0,
            staging: num("KVST"),
            py: num("PY"),
        }
    }
}

/// A shape that ran right on hardware: lanes in `lanes`, the head batch
/// equal to the lanes (where the design has one), at most `mmax`
/// micro-batches and `cap` iterations, the VM as `vm`, and (on
/// serve8b_region_pipe) grids of `py` rows without the VM block or
/// staging.
#[derive(Debug, Clone, Copy)]
pub struct Ran {
    pub design: Design,
    pub lanes: (usize, usize),
    pub mmax: usize,
    pub cap: usize,
    pub vm: Vm,
    pub py: Option<usize>,
    pub reason: &'static str,
}

impl Ran {
    fn holds(&self, s: &Shape) -> bool {
        let head_batch = s.design != Design::Serve2dPipe || s.head_batch == s.lanes;
        let vm = match (self.vm, s.vm) {
            (Vm::Concurrent { lanes, .. }, Vm::Concurrent { lanes: l, set_b }) => {
                lanes == l && set_b
            }
            (a, b) => a == b,
        };
        self.design == s.design
            && (self.lanes.0..=self.lanes.1).contains(&s.lanes)
            && head_batch
            && s.mmax <= self.mmax
            && (s.design != Design::Serve8bRegionPipe || s.mmax >= 1)
            && s.cap <= self.cap
            && vm
            && self
                .py
                .is_none_or(|py| py == s.py && !s.vm_block && s.staging == 0)
    }
}

/// The validated ring shapes.
pub const RAN: &[Ran] = &[
    Ran {
        design: Design::Serve2dPipe,
        lanes: (0, 4),
        mmax: 8,
        cap: 32,
        vm: Vm::Off,
        py: None,
        reason: "B 4 with HB 4 (one head group a micro-batch) at up to 8 micro-batches and CAP \
                 32 ran exact on CS-3, even with every lane's winner the same every step; smaller \
                 B (simulator builds) stays open with HB = B",
    },
    Ran {
        design: Design::Serve2dPipe,
        lanes: (0, 4),
        mmax: 4,
        cap: 16,
        vm: Vm::Sets,
        py: None,
        reason: "the VM sets ran at up to 4 micro-batches and CAP 16",
    },
    Ran {
        design: Design::Serve2dPipe,
        lanes: (4, 4),
        mmax: 4,
        cap: 16,
        vm: Vm::Concurrent {
            lanes: (8, 8),
            set_b: true,
        },
        py: None,
        reason: "VMCONC ran as the symmetric 8/8: B 4 HB 4, 8 lanes on each VM set",
    },
    Ran {
        design: Design::Serve8bRegionPipe,
        lanes: (4, 4),
        mmax: 4,
        cap: 16,
        vm: Vm::Off,
        py: Some(144),
        reason: "real Qwen3-8B, every lane exact, with the units' output-queue drain",
    },
    Ran {
        design: Design::Serve8bRegionPipe,
        lanes: (2, 2),
        mmax: 8,
        cap: 32,
        vm: Vm::Off,
        py: Some(144),
        reason: "real Qwen3-8B, every lane exact, on B 4's grids",
    },
    Ran {
        design: Design::Serve8bRegionPipe,
        lanes: (1, 1),
        mmax: 12,
        cap: 32,
        vm: Vm::Off,
        py: Some(144),
        reason: "real Qwen3-8B, every lane exact, on B 4's grids",
    },
    Ran {
        design: Design::Serve8bRegionPipe,
        lanes: (1, 1),
        mmax: 12,
        cap: 32,
        vm: Vm::Off,
        py: Some(152),
        reason: "real Llama 3.1 8B (QKN 0, eps 1e-5, llama3 RoPE table) on 152-row grids: \
                 pie serve x12 (0402) all streams done, stream 0 equal to the f32 golden for \
                 its 144 tokens, 4,361 tok/s median over 10 bursts; B 1 region on the same \
                 grids equal to the f32 golden up to near ties (0400b, 0401)",
    },
    // serve8b_region (no ring; recorded, not gating): the cerebras-8b-speed
    // builds (SPD 1, LSE 1, HV's PS 1 / NB 16 / E 5, MODE 2), bare B 1.
    Ran {
        design: Design::Serve8bRegion,
        lanes: (1, 1),
        mmax: 0,
        cap: 128,
        vm: Vm::Off,
        py: Some(152),
        reason: "real Llama 3.1 8B, B 1, SPD + LSE + HV + MODE 2 at CAP 128: option A \
                 (RPY 240) 1,284.9 tok/s (0469) and with the one-search append 1,303.2 (0471), \
                 greedy = the f32 golden for 225 tokens (the 0.0107 tie), 0 teacher-forced \
                 mismatches; lever D (YD 624, RR 5, RPY 208, RHc 20, RTR 5) 1,365.4 (0472), \
                 greedy = golden for 512, mismatches only at the two known ties (225, 274); \
                 with OP2 (scalar SiLU, set_axis skip; the plan's `ops2`, its codegen files \
                 byte-identical to 0480's) 1,403.7 (0480), the same accuracy; pie B 1 \
                 (frame 4, depth 15) 94.9-95.6% of bare over 6 reps (0481), on OP2 \
                 1,345.0-1,356.7 tok/s = 95.1-96.0% of bare 1,413.9 (0484); with QKV1 (one QKV \
                 matvec, the plan's `qkv1`) 1,443.0 (0485, regions 46.2-46.6k a pass), greedy = \
                 golden for 512, mismatches only at the two known ties; pie B 1 on it \
                 1,368.6-1,378.5 = 94.8-95.5% of bare 1,443.3 in 5 of 6 reps (0487); the same \
                 program with the lane's pages on the rows by o/l all-reduce slack (bench8b \
                 BENCH8B_PAGEORDER ar:208:16, host only) 1,461.5 (0488, regions 45.6-46.0k a \
                 pass; control 1,442.9), teacher-forced 1 mismatch (the 225 tie); bare \
                 1,460.0-1,461.3 and pie B 1 with PIE_CEREBRAS_PAGE_ORDER 1,386.2-1,401.7 = \
                 94.9-96.0% (0490; 0487 1,368.6-1,378.5), every pie stream = the f32 golden \
                 for 512 or off it only at the 225 tie; with ARF 1 (the regions' four Y \
                 all-reduces as streamed halves with FIFO dual-input merges, host routes) \
                 1,494.1 / 1,494.8 (0497, regions 44.59-44.75k a pass = 14.9k a layer; \
                 same-session control 0496 1,461.1-1,461.7, 45.66k), teacher-forced \
                 mismatches only at the two known ties (225, 274); bare 1,494.1-1,495.1 and pie \
                 B 1 (the plan's `arf`, PIE_CEREBRAS_PAGE_ORDER) 1,422.0-1,434.9 = 95.1-96.0% of \
                 bare (0502), every pie stream one of the known three (= golden for 512, or off \
                 it only at the 225 or 274 tie); ARF 3 (+ the X full all-reduces, the plan's \
                 `arfx`) + MEM 1 1,530.5 / 1,531.0 (0508, regions 43.45-43.57k a pass = 14.5k a \
                 layer), mismatches only at the two known ties; the IEEE f16 build (F16 1) with \
                 its probabilities kept finite 1,451.8 / 1,451.6 (0509, 0504's program + the \
                 fix), ARF + F16 2 (f16-accumulating region matvecs) + MEM 1 1,565.5 / 1,565.6 \
                 (0509, regions 42.2k a pass = 14.07k a layer), both greedy = golden for 512 and \
                 0 teacher-forced mismatches; ARF 3 + MEM 3 + OP3 (RoPE's tables once a pass, the \
                 plan's `op3`) 1,594.2 / 1,594.7 (0510, regions 41.48k a pass = 13.83k a layer), \
                 mismatches only at the two ties; with F16 2 (the plan's `f16acc`) 1,663.3 / \
                 1,662.9 (0511, regions 39.35k a pass = 13.12k a layer, token 467.5k cycles), \
                 greedy = golden for 512, 0 teacher-forced mismatches; F16 3 (the grids' matvecs \
                 f16-accumulating too, the plan's `f16accg`) 1,683.4 / 1,683.8 (0515, grids \
                 18.96k / 20.84k a pass), the same accuracy; pie B 1 on the F16 2 program \
                 (the plan's `arfx` + `mem` + `op3` + `f16acc`) 1,555.3-1,586.2 = 93.6-95.4% of \
                 bare 1,662.1-1,662.7 in 5 of 6 reps (0516; rep 1 1,304.5 with a 63.5 ms host \
                 gap), streams = golden for 512 x5 or off it only at the 274 tie x1; F16 3 with \
                 the probabilities' f16 copy scaled by a power of two (bf16's redo window, no \
                 redos) 1,678.2 / 1,679.4 (0518, grids 18.8 / 18.4k), the same accuracy; pie B 1 \
                 on it (the plan's `f16accg` too) 1,585.2-1,602.3 = 94.4-95.5% of bare \
                 1,677.5-1,678.9 (0519), every pie and bare stream = golden for 512; with the \
                 grids taking ARF 3 + OP3 + MEM 3 too (GRD 7, the plan's `grid_levers`) \
                 1,692.5 / 1,693.5 (0522, grids 16.8 / 16.6k), greedy = golden for 512, 0 \
                 teacher-forced mismatches; pie B 1 on it (bundle x5) 1,598.6-1,614.6 = \
                 94.5-95.4% of bare 1,692.2-1,692.6 in 5 of 6 reps (0525; rep 1 1,322.0 with a \
                 99.6 ms host fire stall), every pie and bare stream = golden for 512; + the \
                 regions' f16 copy clamped with 0515's window (F16CL) 1,705.8 / 1,706.4 (0526, \
                 regions 39.20-39.37k a pass) and + XA on its own colours (XAC) 1,709.9 / \
                 1,711.6 (0527, regions 39.10-39.26k), both = golden for 512, 0 mismatches; pie \
                 B 1 on it (bundle x6: + `f16_clamp` + `xac`) 1,614.8-1,631.4 = 94.4-95.4% of bare \
                 1,709.7-1,710.6 (0528), every pie and bare stream = golden for 512; + the \
                 units' exact op cuts (TRM 5) 1,762.3 / 1,762.6 (0530, regions 37.9-38.0k a pass), \
                 + exp at degree 4 and the residuals from the receives (TRM 29) 1,804.7 / 1,804.7 \
                 (0531), + x's sums of squares by a scalar loop (TRM 31) 1,832.2 / 1,832.7 (0531, \
                 regions 36.3-36.4k a pass, grids 16.2 / 15.9k), each = golden for 512, 0 \
                 teacher-forced mismatches (TRM 7's stride-0 accumulate: garbage, 0530); + HV's \
                 zeroings by reads and entries once a pass, Y groups of 16 and centred vertical \
                 handoffs (TRM 223, ARY 16, HVC 1) 1,892.8 / 1,891.8 (0533, regions 34.9-35.1k a \
                 pass), + rsqrt in two Newton steps (TRM 479) 1,903.1 / 1,902.5 (0533), + the \
                 head's split_x alone (TRM 255) 1,897.4 / 1,897.7 (0533, head matvec mean 4.28k \
                 against 5.34k), each = golden for 512, 0 teacher-forced mismatches; ARF 7 (the \
                 segment all-reduce as streamed halves) hung (0532: kernel ARF constant 1); \
                 TRM 511 + ARY 16 + HVC 1 1,908.7 / 1,908.2 (0534), golden for 512, 0 \
                 mismatches; pie B 1 on it (bundle x7: + `trim` + `ary16` + `hvc`) 1,820.6, \
                 1,824.0, 1,805.8, 1,820.5, 1,805.7 = 94.6-95.6% of bare (rep 1 1,455.6: the \
                 host fire stall), every stream = golden for 512; + the segment all-reduce \
                 as streamed halves with the kernels' ARF constant fixed (ARF 7) 1,927.8 / \
                 1,927.2 (0535, regions 34.35-34.51k a pass), golden for 512, 0 mismatches; + x g \
                 in f16 (TRM 1023) 1,946.8 / 1,946.3 (0537, regions 33.97-34.14k), golden for \
                 512, 0 mismatches; + the MLP's act in f16 on every PE, no act all-reduce, \
                 the f16 gammas in place (TRM 6143) 1,997.6 / 1,997.3 (0538); + the head's \
                 packed argmax round (TRM 22527) 2,025.7 / 2,025.1 and + the SiLU's scales in \
                 f16 (TRM 24575) 2,033.3 / 2,032.5 (0540, regions 32.37-32.63k), golden for \
                 512, 0 mismatches; Pie B 1 on TRM 22527 (0541, depth 63, posted:64) \
                 1,993.2-2,010.6, every stream = golden for 512",
    },
    Ran {
        design: Design::Serve8bRegion,
        lanes: (1, 1),
        mmax: 0,
        cap: 64,
        vm: Vm::Off,
        py: Some(144),
        reason: "real Qwen3-8B, B 1, SPD + LSE + HV + MODE 2 at CAP 64, option A: 1,108.7 \
                 tok/s (0470), 1,120.3 with the one-search append (0473); the same 14 \
                 teacher-forced mismatches as the bf16 reference (model differences), the \
                 160-token free-running stream identical across them; lever D by the U order \
                 (1 + 10 x 3 + 5, YD 208, NCOL 5, RPY 208, RHc 20, RTR 5) 1,166.8 (0476), the \
                 same 14 mismatches at the same margins; + OP2 + RGX 16 1,208.2 (0483), + QKV1 \
                 1,234.8 (0486), the same 14 mismatches, both streams equal to 0483's for 160; \
                 the same program with the pages on the rows by all-reduce slack (page order \
                 ar:208:16, host only) 1,244.6 (0489, regions 45.5-45.9k a pass), the same 14 \
                 mismatches, rep 0 = 0486 rep 1 for 160; with ARF 1 (no RGX) 1,260.3 / 1,261.1 \
                 (0500, regions 45.0-45.1k a pass), the same 14 mismatches (token, got, want, \
                 margin to 4 places); ARF 3 + MEM 1 + OP3 1,337.6 / 1,337.9 (0512, regions \
                 41.5-41.7k a pass = 13.9k a layer) and with F16 2 1,360.9 / 1,359.4 (0512, \
                 regions 40.0-43.7k a pass), both the same 14 mismatches; F16 3 with the scaled \
                 f16 copy 1,407.4 / 1,407.4 (0518, regions 40.0-40.2k a pass, no redos), the same \
                 14 mismatches; pie B 1 on it (160 tokens) 1,212.4-1,233.6 = 86.2-87.7% of bare \
                 1,406.0-1,406.7 (0520; device 0.722 ms a step against 0.711), every pie stream = \
                 bare's; with GRD 7 (the six grids: ARF 3 + OP3 + MEM 1) 1,435.0 / 1,436.4 (0523, \
                 grids 15.9-16.4k), the same 14 mismatches; + XAC 1,445.9 / 1,443.8 (0529, \
                 regions 39.8-39.9k a pass), the same 14 mismatches; + TRM 31 1,521.8 / 1,520.6 \
                 (0531) with 13 teacher-forced mismatches, not 0529's 14 (9 differ): TRM's \
                 inexact bits (2: another summation order, 8: exp at degree 4) move Qwen's; \
                 the exact bits (TRM 245) + HVC 1 1,537.7 / 1,539.4 and + ARY 16 1,539.4 / \
                 1,538.5 (0534), both with 0529's 14 mismatches exactly; + ARF 7 1,564.0 / \
                 1,563.6 (0535, regions 36.6-36.9k a pass), the same 14; + the head's packed \
                 argmax round (TRM 16629, exact) 1,566.1 / 1,564.5 (0540), the same 14",
    },
];

/// A shape known to fail, and why.
#[derive(Clone, Copy)]
pub struct Bad {
    pub design: Design,
    pub when: fn(&Shape) -> bool,
    pub reason: &'static str,
}

/// The known failures, checked in order before [`RAN`].
pub const KNOWN_BAD: &[Bad] = &[
    Bad {
        design: Design::Serve2d,
        when: |s| s.one_sequence,
        reason: "serve2d GRID:0 is unsupported (it hung on hardware); use GRID:1 with B:1 for \
                 one sequence",
    },
    Bad {
        design: Design::Serve2dPipe,
        when: |s| {
            s.ring
                && matches!(s.vm, Vm::Concurrent { .. })
                && !RAN
                    .iter()
                    .any(|r| matches!(r.vm, Vm::Concurrent { .. }) && r.holds(s))
        },
        reason: "serve2d_pipe VMCONC runs only as validated on hardware: B 4 HB 4, MMAX <= 4, \
                 CAP <= 16, 8 lanes on each VM set",
    },
    Bad {
        design: Design::Serve2dPipe,
        when: |s| s.ring && !matches!(s.vm, Vm::Concurrent { .. }) && s.head_batch > 4,
        reason: "serve2d_pipe HB > 4 is unsupported: an 8-lane head group hangs on some winner \
                 patterns",
    },
    Bad {
        design: Design::Serve2dPipe,
        when: |s| s.ring && !matches!(s.vm, Vm::Concurrent { .. }) && s.lanes > 8,
        reason: "serve2d_pipe B > 8 is unsupported: lanes past the first get wrong embeddings",
    },
];

/// What a refusal says about `s` beyond its reason.
fn context(s: &Shape, why: &str) -> String {
    match s.design {
        Design::Serve2dPipe => format!(
            "{why} (B:{} HB:{} MMAX:{} CAP:{} VM:{}); hardware ran B:4 HB:4 MMAX<=8 CAP<=32, \
MMAX<=4 CAP<=16 with the VM sets, and their 8/8 VMCONC",
            s.lanes,
            s.head_batch,
            s.mmax,
            s.cap,
            usize::from(s.vm_on)
        ),
        Design::Serve8bRegionPipe => format!(
            "serve8b_region_pipe B:{} MMAX:{} CAP:{} VMX:{} KVST:{} PY:{} is unsupported: only \
B:4 MMAX<=4 CAP<=16, B:2 MMAX<=8 CAP<=32 and B:1 MMAX<=12 CAP<=32 on B 4's grids (PY 144, \
grid_lanes 4), and B:1 MMAX<=12 CAP<=32 on PY 152 (Llama 3.1 8B), without the VM block or staging \
ran on hardware",
            s.lanes,
            s.mmax,
            s.cap,
            usize::from(s.vm_block),
            s.staging,
            s.py
        ),
        _ => why.to_string(),
    }
}

/// Refuses `design` built with `params` where a known failure matches it,
/// or where its design records validated ring shapes and none holds it.
pub fn check(design: Design, params: &[(String, String)]) -> R<()> {
    let s = Shape::of(design, params);
    if let Some(bad) = KNOWN_BAD
        .iter()
        .find(|b| b.design == design && (b.when)(&s))
    {
        return Err(context(&s, bad.reason));
    }
    let recorded = RAN.iter().any(|r| r.design == design);
    if s.ring && recorded && !RAN.iter().any(|r| r.holds(&s)) {
        return Err(context(&s, "serve2d_pipe's shape is unrun on hardware"));
    }
    Ok(())
}

/// The validated ring shape of `design` at `lanes`: its most
/// micro-batches and its CAP.
#[must_use]
pub fn ran(design: Design, lanes: usize) -> Option<(usize, usize)> {
    RAN.iter()
        .find(|r| r.design == design && r.vm == Vm::Off && r.lanes == (lanes, lanes))
        .map(|r| (r.mmax, r.cap))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row carries its reason, and a ring design's every validated
    /// shape passes its own check.
    #[test]
    fn every_validated_shape_passes_and_says_why() {
        for r in RAN {
            assert!(!r.reason.is_empty());
            let mut p: Vec<(String, String)> = [
                ("B", r.lanes.1),
                ("HB", r.lanes.1),
                ("MMAX", r.mmax),
                ("CAP", r.cap),
                ("PY", r.py.unwrap_or(0)),
                ("PIPE", 1),
            ]
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.to_string()))
            .collect();
            match r.vm {
                Vm::Off => {}
                Vm::Sets => p.push(("VM".into(), "1".into())),
                Vm::Concurrent { lanes, .. } => p.extend(
                    [
                        ("VM", 1),
                        ("VMB", 1),
                        ("VMCONC", 1),
                        ("vm_lanes", lanes.0),
                        ("b_vm_lanes", lanes.1),
                    ]
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), v.to_string())),
                ),
            }
            assert!(check(r.design, &p).is_ok(), "{r:?}");
        }
        assert!(KNOWN_BAD.iter().all(|b| !b.reason.is_empty()));
        assert_eq!(ran(Design::Serve8bRegionPipe, 2), Some((8, 32)));
        assert_eq!(ran(Design::Serve8bRegionPipe, 8), None);
    }
}
