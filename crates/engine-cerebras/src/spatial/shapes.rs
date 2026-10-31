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
grid_lanes 4) without the VM block or staging ran on hardware",
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
