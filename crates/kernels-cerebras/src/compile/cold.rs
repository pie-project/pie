//! The search's cold start: a model's first partitions from its layer graph alone. Each kernel set's constructor reads a few structural keys (the units,
//! the columns and rows a unit, the KV entries a row, the segments); their values are drawn from
//! what the shape divides (the layers into units of one layer pattern, the heads over the
//! columns, the hidden over the rows) and from the fabric's size, the rest at the kernels'
//! defaults ([`DEFAULTS`]). Every combination the constructor plans, serving the positions asked
//! for, is a candidate; the search walks from the fastest few ([`crate::compile::search`]).

use crate::compile::kernels::KernelSet;
use crate::compile::plan::{LayerKind, Preset, Shape};

/// The fabric's rows a program may take (the system's, past memcpy's halo).
pub const FABRIC_ROWS: u32 = 1153;
/// Its columns.
pub const FABRIC_COLS: u32 = 750;

/// A posted token loop's most steps (`LOOP`), every kernel set's: a loop's rows come back when it
/// ends, so a longer one holds a stream's last tokens back (back-to-back posted loops already hide
/// a launch's fixed cost).
pub const LOOP: u32 = 32;

/// A kernel set's levers and the keys no partition moves: the kernels' defaults.
#[must_use]
pub fn defaults(set: KernelSet) -> (&'static str, &'static [(&'static str, f64)]) {
    match set {
        KernelSet::Own => (
            "NPX:1,DIAG:0,TSTAMP:0,XKEEP:0,HACC16:0,ATTL:2",
            &[("G2", 12.0), ("WL", 1.0), ("LOOP", LOOP as f64)],
        ),
        KernelSet::Shared => ("", &[("G2", 12.0), ("G2W", 24.0), ("LOOP", LOOP as f64)]),
        KernelSet::Windows => (
            "DIAG:0,GELUT:1",
            &[("G2", 16.0), ("G2W", 24.0), ("LOOP", LOOP as f64)],
        ),
        KernelSet::Experts => (
            "DIAG:0,GLUV:1,CYV:2",
            &[
                ("G2", 16.0),
                ("G2W", 24.0),
                ("LOOP", LOOP as f64),
                ("NLV", 5.0),
                ("HSCALE", 32768.0),
            ],
        ),
    }
}

fn divisors(n: u32) -> Vec<u32> {
    (1..=n).filter(|d| n.is_multiple_of(*d)).collect()
}

/// The layer counts a unit may hold: divisors of the layers whose units repeat one pattern.
fn unit_layers(s: &Shape) -> Vec<u32> {
    let n = s.layers.len() as u32;
    divisors(n)
        .into_iter()
        .filter(|&k| {
            let p = &s.layers[..k as usize];
            s.layers.chunks(k as usize).all(|u| u == p)
        })
        .collect()
}

/// `vocab` over `rows` rows, a multiple of 8.
fn vocab_rows(vocab: u32, rows: u32) -> u32 {
    vocab.div_ceil(rows).div_ceil(8) * 8
}

fn preset(set: KernelSet, keys: &[(&str, u32)]) -> Preset {
    let (levers, fixed) = defaults(set);
    let mut p = Preset::new("cold", &[]);
    for &(k, v) in fixed {
        p.keys.insert(k.to_string(), v);
    }
    for &(k, v) in keys {
        p.keys.insert(k.to_string(), f64::from(v));
    }
    p.levers = levers.to_string();
    p
}

/// Every cold candidate of `s` serving `positions` (the constructor still to accept each).
#[must_use]
pub fn candidates(s: &Shape, positions: u32) -> Vec<Preset> {
    let set = KernelSet::of_shape(s);
    let n = s.layers.len() as u32;
    let mut out = Vec::new();
    let hr = FABRIC_ROWS;
    let vr = vocab_rows(s.vocab, hr);
    let rows_range = || (32..=320).step_by(8);
    match set {
        KernelSet::Own => {
            // a column count every head count divides, the MLP over it
            let mut heads = vec![s.attn.kv_heads];
            if let Some(g) = s.gdn {
                heads.extend([g.key_heads, g.value_heads]);
            }
            let pxs: Vec<u32> = (16..=FABRIC_COLS / 2)
                .filter(|px| {
                    heads.iter().all(|h| *h > 0 && px.is_multiple_of(*h))
                        && s.mlp.intermediate.is_multiple_of(*px)
                })
                .collect();
            for k in unit_layers(s) {
                let units = n / k;
                for unx in [1u32, 2, 3] {
                    for uny in [units.div_ceil(unx), (units + 1).div_ceil(unx)] {
                        if unx * uny < units || unx * uny > units + 1 {
                            continue;
                        }
                        for &px in &pxs {
                            if unx * (px + 1) > FABRIC_COLS {
                                continue;
                            }
                            for py in rows_range() {
                                if uny * (py + 1) > FABRIC_ROWS {
                                    continue;
                                }
                                let ne = positions.div_ceil(py);
                                out.push(preset(
                                    set,
                                    &[
                                        ("K", k),
                                        ("UNX", unx),
                                        ("UNY", uny),
                                        ("PX", px),
                                        ("PY", py),
                                        ("NE", ne),
                                        ("HR", hr),
                                        ("VR", vr),
                                    ],
                                ));
                            }
                        }
                    }
                }
            }
        }
        KernelSet::Shared => {
            let window = s
                .layers
                .iter()
                .find_map(|l| match l {
                    LayerKind::Sliding { window } => Some(*window),
                    _ => None,
                })
                .unwrap_or(1);
            for nu in divisors(n).into_iter().filter(|u| (1..=16).contains(u)) {
                for px in (64..=FABRIC_COLS / 2).step_by(8) {
                    for py in rows_range().filter(|py| s.hidden.is_multiple_of(*py)) {
                        // (the global KV over the window's KV rows, as the constructor lays
                        // them: the positions asked)
                        let nes = window.div_ceil(py);
                        let pyw = if window.is_multiple_of(nes) {
                            window / nes
                        } else {
                            py
                        };
                        let neg = positions.div_ceil(pyw);
                        out.push(preset(
                            set,
                            &[
                                ("NU", nu),
                                ("PX", px),
                                ("PY", py),
                                ("NEG", neg),
                                ("HR", hr - 23),
                            ],
                        ));
                    }
                }
            }
        }
        KernelSet::Windows => {
            let kv = s.attn.kv_heads.max(1);
            let window = s
                .layers
                .iter()
                .find_map(|l| match l {
                    LayerKind::Sliding { window } => Some(*window),
                    _ => None,
                })
                .unwrap_or(1024);
            for nu in divisors(n).into_iter().filter(|u| (1..=16).contains(u)) {
                for swa in 8..=64 {
                    let px = kv * swa;
                    if px > FABRIC_COLS / 2 {
                        break;
                    }
                    for py in rows_range().filter(|py| *py >= 64) {
                        for ra in [64u32, 128, 256].into_iter().filter(|ra| *ra < py) {
                            // the attention rows' dims, the plain rows' covering the rest
                            for hca in [8u32, 12, 16] {
                                let rest = s.hidden.saturating_sub(ra * hca);
                                let hcw = rest.div_ceil(py - ra).max(1);
                                let neg = positions.div_ceil(ra);
                                let _ = window;
                                out.push(preset(
                                    set,
                                    &[
                                        ("NU", nu),
                                        ("PX", px),
                                        ("PY", py),
                                        ("RA", ra),
                                        ("SWA", swa),
                                        ("HcA", hca),
                                        ("HcW", hcw),
                                        ("NEG", neg),
                                        ("GA", 6),
                                        ("GW", 5),
                                        ("HR", hr),
                                    ],
                                ));
                            }
                        }
                    }
                }
            }
        }
        KernelSet::Experts => {
            let Some(moe) = s.moe else { return out };
            let hd = s.attn.head_dim;
            for units in divisors(n) {
                for unx in [1u32, 2] {
                    if !units.is_multiple_of(unx) {
                        continue;
                    }
                    let uny = units / unx;
                    for swa in divisors(hd)
                        .into_iter()
                        .filter(|w| (hd / w).is_multiple_of(2))
                    {
                        let ax = s.attn.kv_heads * swa;
                        for mx in (moe.experts..=FABRIC_COLS / unx)
                            .step_by(moe.experts as usize)
                            .filter(|m| moe.expert_intermediate.is_multiple_of(*m))
                        {
                            if unx * (ax + mx) > FABRIC_COLS {
                                continue;
                            }
                            for py in rows_range().filter(|py| s.hidden.is_multiple_of(*py)) {
                                if uny * (py + 1) > FABRIC_ROWS {
                                    continue;
                                }
                                let ne = positions.div_ceil(py);
                                let hvc = unx * (ax + mx);
                                out.push(preset(
                                    set,
                                    &[
                                        ("AX", ax),
                                        ("MX", mx),
                                        ("PY", py),
                                        ("UNX", unx),
                                        ("UNY", uny),
                                        ("SWA", swa),
                                        ("SWR", moe.experts),
                                        ("HVC", hvc),
                                        ("NE", ne),
                                    ],
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    out
}
