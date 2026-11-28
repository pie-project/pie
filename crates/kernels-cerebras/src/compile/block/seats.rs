//! A ring of seats: the program holds `m` sequences at once, pass `s` of a launch seat `s % m`'s
//! lap `s / m` (`ops/seats.csl`). Each seat has its own state: a KV window of the shared pool's
//! entries from the first position the host states (CY's `sbase`), its log-sum-exp references,
//! GDN states and conv windows, its first x at a launch (CX's `xs`), its position (CY's `spos`)
//! and its forced tokens (the head's `nforce`); the head marks
//! the last `m` passes' tokens as taken by none. Seats ride every kernel set
//! ([`crate::compile::kernels::KernelSet`]; the shared caches': the shared-KV line carries the
//! pass's seat word, each pass's credited back before the next; the per-layer inputs of a
//! launch's first lap the host's (CY's `pls`), the bus's one token a pass after; the routed experts': a unit's halves one
//! stage) and the token loop, at most `units - 1` of them
//! ([`crate::compile::check::hazard::seats`]). A sliding cache holds a ring a seat, or each
//! seat's ring within its window's entries where those take fewer.
//!
//! The kernels' seat code is in their sources' lines marked `//+seats` (written only for a ring)
//! and the one-sequence lines they replace, `//-seats` (written only without one): a program of
//! one seat is the sources' text before the seats, byte for byte ([`source`]).

use crate::compile::abi::{Kernels, Value};
use crate::compile::program::Program;
use crate::compile::role::Kind;

use super::Block;

type R<T> = Result<T, String>;

/// Marks a line written only for a ring, and one written only without one.
const RING: &str = " //+seats";
const ONE: &str = " //-seats";

/// The seats' exported names: their positions and windows (CY) and first x (CX).
pub const SYMBOLS: &[(&str, &str)] = &[("spos", "u32"), ("sbase", "u32"), ("xs", "f32")];
/// And with per-layer inputs: a launch's first lap's words (CY).
pub const SYMBOLS_PLE: &[(&str, &str)] = &[
    ("spos", "u32"),
    ("sbase", "u32"),
    ("xs", "f32"),
    ("pls", "f32"),
];

/// The seats a ring may hold: CY's word to the attention carries 2 seat + 1 under 2^8.
const MOST: u32 = 127;

/// `m` seats (1: one sequence; the block then only writes the sources); `ple`: per-layer inputs.
pub struct Seats {
    pub m: u32,
    pub ple: bool,
}

/// `text` as a program of `m` seats writes it: the ring's lines (`//+seats`) or the one-sequence
/// lines (`//-seats`), their marks dropped.
#[must_use]
pub fn source(text: &str, m: u32) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let (body, end) = line.strip_suffix('\n').map_or((line, ""), |b| (b, "\n"));
        let keep = if let Some(b) = body.strip_suffix(RING) {
            (m > 1).then_some(b)
        } else if let Some(b) = body.strip_suffix(ONE) {
            (m == 1).then_some(b)
        } else {
            Some(body)
        };
        if let Some(b) = keep {
            out.push_str(b);
            out.push_str(end);
        }
    }
    out
}

impl Block for Seats {
    fn kernels(&self, prog: &Program, ks: &mut Kernels) -> R<()> {
        let m = self.m;
        if m == 1 {
            return Ok(());
        }
        let (plan, p) = (&prog.plan, &prog.plan.partition);
        if !plan.contract.token_loop {
            return Err(format!(
                "seats: {m} seats want the token loop ({} here)",
                prog.roles.name()
            ));
        }
        if let Some(e) = crate::compile::check::hazard::seats(m, p.units) {
            return Err(format!("seats: {e}"));
        }
        // (the token words: one a pass to the PEs that turn routes)
        if let Some(e) = crate::sources::get("head_ctl.csl")
            .and_then(|t| crate::compile::check::hazard::seat_turns(&source(t, m)))
        {
            return Err(format!("seats: {e}"));
        }
        let pool = p.kv_entries.per_row * p.kv_rows;
        if let Some(e) = crate::compile::check::hazard::seat_pool(pool) {
            return Err(format!("seats: {e}"));
        }
        if m > MOST {
            return Err(format!("seats: {m} seats, past the {MOST} a ring holds"));
        }
        let ne = p.kv_entries.per_row;
        if ne < m {
            return Err(format!("seats: {m} seats over {ne} KV entries a row"));
        }
        // every weight, CX, CY and head control kernel (a set's halves each)
        for kb in ks.values_mut() {
            let owns: &[&str] = match kb.source {
                "weight.csl" | "head_ctl.csl" => &[],
                "cx.csl" => &["xs"],
                "cy.csl" => &["spos", "sbase"],
                _ => continue,
            };
            kb.params.set("SEATS", Value::Int(i64::from(m)));
            kb.owns.extend(owns.iter().copied());
        }
        // the per-layer inputs: a launch's first lap's the host's, a lap of words a CY (`pls`);
        // the bus carries one token a pass (its owners turn routes: never a lap back to back)
        if self.ple {
            let lv = prog
                .floor
                .stores
                .iter()
                .map(|st| super::ple::levels(plan, st.units.len()));
            let lv = lv.max().unwrap_or(0);
            if let Some(e) = crate::compile::check::hazard::seat_bus(plan.contract.cap, lv) {
                return Err(format!("seats: {e}"));
            }
            let cy = ks.get_mut(&Kind::CY).ok_or("seats: no CY kernel")?;
            cy.params.set("PLV", Value::Int(i64::from(lv)));
            cy.owns.push("pls");
        }
        Ok(())
    }

    fn symbols(&self) -> &'static [(&'static str, &'static str)] {
        match (self.m > 1, self.ple) {
            (false, _) => &[],
            (true, false) => SYMBOLS,
            (true, true) => SYMBOLS_PLE,
        }
    }

    fn source(&self, name: &str, text: String) -> Option<String> {
        if name == "seats.csl" && self.m == 1 {
            return None;
        }
        Some(source(&text, self.m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One seat writes every source as it was before the seats: no seat line, every other line
    /// as is; a ring writes each marked pair's other side.
    #[test]
    fn one_seat_writes_the_sources_as_they_were() {
        for (name, text) in crate::sources::KERNELS.iter().chain(crate::sources::OPS) {
            let one = source(text, 1);
            assert!(!one.contains("//+seats"), "{name}: a seat line at one seat");
            let plain: Vec<&str> = text
                .lines()
                .filter(|l| !l.ends_with(RING) && !l.ends_with(ONE))
                .collect();
            let ring = source(text, 3);
            assert!(
                !ring.contains("//-seats") && !ring.contains("//+seats"),
                "{name}"
            );
            for l in plain {
                assert!(one.contains(l) && ring.contains(l), "{name}: {l}");
            }
        }
        assert_eq!(source("a\nb //-seats\nc //+seats\nd", 1), "a\nb\nd");
        assert_eq!(source("a\nb //-seats\nc //+seats\nd", 2), "a\nc\nd");
    }

    /// The head's token words go one a pass, a lap of them back to back is refused.
    #[test]
    fn the_token_words_go_one_a_pass() {
        use crate::compile::check::hazard::seat_turns;
        let head = crate::sources::get("head_ctl.csl").unwrap();
        assert_eq!(seat_turns(&source(head, 1)), None);
        assert_eq!(seat_turns(&source(head, 4)), None);
        let burst = "task t() void {\n  while (s < n) : (s += 1) {\n    if (s == 0) {\n      \
                     while (j < SEATS) : (j += 1) { @fmovs(fab.fout(q_tb, 1), d); }\n    }\n  \
                     }\n}\n";
        assert!(seat_turns(burst).is_some());
    }
}
