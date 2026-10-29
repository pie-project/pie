//! What a guest table costs the head per token, before it runs: each
//! instruction's cycles from the VM's profile (`vm_prof`, Fork 3's coverage
//! sweep, `spatial/eta/COVERAGE.md`), summed over `pre` (overlaps the
//! model's token) and `main` (after the logits: on the critical path). The
//! placement weighs both against the model's token time: a table whose
//! `pre` does not hide under it, or whose `main` adds too much to it, stays
//! on the host.
//!
//! Calibrated at Vt = 32 vocabulary words a VM PE (vocab 512 on a 2-column
//! VM, and serve2d's full vocabulary alike): over the sweep's 31 programs the
//! forecast's median is 0.98 of the measured `main` and 1.01 of `pre`
//! (a profile counts `pre` once a step and once more before the first). Wide
//! element ops scale with a PE's words; reductions, top-k and pivots (the
//! hub chain) are taken as measured.

use super::Table;
use super::isa::*;

/// The model's per-token time (cycles) at Qwen3-0.6B and at 8B
/// (`COVERAGE.md`); smaller models are taken at the 0.6B figure.
const TOKEN_06B: f64 = 340_000.0;
const TOKEN_8B: f64 = 800_000.0;

/// A table's forecast cycles a token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Forecast {
    pub pre: u64,
    pub main: u64,
}

/// When a table runs on the head: `pre` at most `pre_margin` of the token,
/// `main` adding at most `main_fraction` of it.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub token_cycles: u64,
    pub pre_margin: f64,
    pub main_fraction: f64,
}

impl Placement {
    pub const PRE_MARGIN: f64 = 0.95;
    /// `main` is on the critical path but the host path ships the
    /// vocabulary's logits every token, so only a table doubling the token
    /// goes back (`COVERAGE.md` keeps beam search's 61% on the wafer).
    pub const MAIN_FRACTION: f64 = 1.0;

    /// Why `f` stays on the host, if it does.
    pub fn refuses(&self, f: Forecast) -> Option<String> {
        let token = self.token_cycles as f64;
        if f.pre as f64 > token * self.pre_margin {
            return Some(format!(
                "pre ~{}k cycles does not hide under the model's token (~{}k × {})",
                f.pre / 1000,
                self.token_cycles / 1000,
                self.pre_margin
            ));
        }
        if f.main as f64 > token * self.main_fraction {
            return Some(format!(
                "main ~{}k cycles adds {:.0}% to the model's token (~{}k; at most {:.0}%)",
                f.main / 1000,
                100.0 * f.main as f64 / token,
                self.token_cycles / 1000,
                100.0 * self.main_fraction
            ));
        }
        None
    }
}

/// The model's per-token cycles from its parameter count: log-linear through
/// 0.6B and 8B, never under the 0.6B figure.
pub fn token_cycles(params: f64) -> u64 {
    let slope = (TOKEN_8B / TOKEN_06B).ln() / (8.0f64 / 0.6).ln();
    let cycles = TOKEN_06B * (params.max(0.6e9) / 0.6e9).powf(slope);
    cycles.round() as u64
}

/// `table`'s cycles a token on a VM whose PEs hold `vt` vocabulary words.
pub fn forecast(table: &Table, vt: u32) -> Forecast {
    let cut = table.pre_len.min(table.ins.len());
    let sum = |ins: &[[u32; 6]]| ins.iter().map(|i| cycles(i, vt)).sum();
    Forecast {
        pre: sum(&table.ins[..cut]),
        main: sum(&table.ins[cut..]),
    }
}

/// One instruction's cycles (the profile's per-op medians).
pub fn cycles(ins: &[u32; 6], vt: u32) -> u64 {
    let [op, fl, _, _, _, c] = *ins;
    let n = u64::from(fl >> 8).max(1);
    // An element op over a wide row: its cost at Vt 32 scaled to `vt`.
    let wide = |c32: u64| c32 * u64::from(vt.max(1)) / 32;
    match op {
        HALT => 0,
        WBIN => wide(if c == B_DIV { 1400 } else { 260 }),
        WCMP => wide(1750),
        WUN => wide(match c {
            U_NEG => 250,
            U_LOG => 9000,
            _ => 2500,
        }),
        WSEL => wide(1670),
        WFILL | WIOTA => wide(200),
        WRNG => wide(if c == 0 { 22_400 } else { 35_900 }),
        WMOV => wide(215),
        RRED => 2300,
        WGATHER => 500 + 70 * n,
        WSCATTER => 250 + 23 * n,
        TOPK => 10_000 + 750 * u64::from(c),
        PIVRANK | PIVMASS => 23_000,
        QBIN => if c >= B_DIV { 340 + 130 * n } else { 265 + 51 * n },
        QCMP => 300 + 75 * n,
        QUN => 360 + 44 * n,
        QSEL => 330 + 19 * n,
        QIOTA => 230 + 15 * n,
        QCONST => 220,
        QMOV => 300 + 54 * n,
        QGATHER => 410 + 240 * n,
        QSCATTER => 390,
        QCUM => 100 + 35 * n,
        QRED => 220 + 47 * n,
        QRNG => 1700 * n,
        OUT => 210,
        _ => 1000,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(ins: Vec<[u32; 6]>, pre_len: usize) -> Table {
        Table {
            ins,
            pre_len,
            homes: Vec::new(),
            outputs: Vec::new(),
            wide_outputs: Vec::new(),
            per_step: Vec::new(),
            tok_home: Some(0),
            nreg: 1,
            pool_words: 1,
            feat: 0,
            remat: false,
            bank_regs: Vec::new(),
            bank_pool: Vec::new(),
        }
    }

    #[test]
    fn token_time_runs_through_the_two_measured_models() {
        assert_eq!(token_cycles(0.6e9), 340_000);
        assert!(token_cycles(8.0e9).abs_diff(800_000) < 10);
        assert_eq!(token_cycles(1.0e6), 340_000);
    }

    #[test]
    fn a_draw_heavy_pre_goes_back_at_0_6b_and_stays_at_8b() {
        // Ten gumbel draws over the vocabulary before the logits (~359k: a
        // synthid tournament's are ~540k), then an argmax.
        let mut ins = vec![[WRNG, 0, 1, 0, 0, 1]; 10];
        ins.push([HALT, 0, 0, 0, 0, 0]);
        ins.extend([[RRED, 0, 0, 1, 0, 0], [HALT, 0, 0, 0, 0, 0]]);
        let f = forecast(&table(ins, 11), 32);
        assert_eq!(f, Forecast { pre: 359_000, main: 2300 });
        let at = |params| Placement {
            token_cycles: token_cycles(params),
            pre_margin: Placement::PRE_MARGIN,
            main_fraction: Placement::MAIN_FRACTION,
        };
        assert!(at(0.6e9).refuses(f).is_some_and(|why| why.starts_with("pre ~359k")));
        assert_eq!(at(8.0e9).refuses(f), None);
    }

    #[test]
    fn main_past_its_fraction_goes_back() {
        let ins = vec![[HALT, 0, 0, 0, 0, 0], [TOPK, 0, 0, 0, 0, 128], [HALT, 0, 0, 0, 0, 0]];
        let f = forecast(&table(ins, 1), 32);
        assert_eq!(f.main, 106_000);
        let p = Placement { token_cycles: 340_000, pre_margin: 0.95, main_fraction: 0.25 };
        assert!(p.refuses(f).is_some_and(|why| why.contains("adds 31%")));
        assert_eq!(Placement { main_fraction: 1.0, ..p }.refuses(f), None);
    }

    #[test]
    fn wide_element_ops_scale_with_a_pe_s_words() {
        assert_eq!(cycles(&[WRNG, 0, 0, 0, 0, 0], 64), 44_800);
        assert_eq!(cycles(&[RRED, 0, 0, 0, 0, 0], 64), 2300);
    }
}
