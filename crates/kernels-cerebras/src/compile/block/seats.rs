//! A ring of seats: the program holds `m` sequences at once, pass `s` of a launch seat `s % m`'s
//! lap `s / m` (`ops/seats.csl`). Each seat has its own state: a KV window of the shared pool's
//! entries from the first position the host states (CY's `sbase`), its log-sum-exp references,
//! GDN states and conv windows, its first x at a launch (CX's `xs`), its position (CY's `spos`)
//! and its forced tokens (the head's `nforce`); the head marks
//! the last `m` passes' tokens as taken by none. Seats ride the own caches' kernels
//! ([`KernelSet::Own`]) and the token loop, at most `units - 1` of them
//! ([`crate::compile::check::hazard::seats`]).
//!
//! The kernels' seat code is in their sources' lines marked `//+seats` (written only for a ring)
//! and the one-sequence lines they replace, `//-seats` (written only without one): a program of
//! one seat is the sources' text before the seats, byte for byte ([`source`]).

use crate::compile::abi::{Kernels, Value};
use crate::compile::kernels::KernelSet;
use crate::compile::program::Program;
use crate::compile::role::Kind;

use super::Block;

type R<T> = Result<T, String>;

/// Marks a line written only for a ring, and one written only without one.
const RING: &str = " //+seats";
const ONE: &str = " //-seats";

/// The seats' exported names: their positions and windows (CY) and first x (CX).
pub const SYMBOLS: &[(&str, &str)] = &[("spos", "u32"), ("sbase", "u32"), ("xs", "f32")];

/// The seats a ring may hold: CY's word to the attention carries 2 seat + 1 under 2^8.
const MOST: u32 = 127;

/// `m` seats (1: one sequence; the block then only writes the sources).
pub struct Seats {
    pub m: u32,
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
        if prog.roles != KernelSet::Own || !plan.contract.token_loop {
            return Err(format!(
                "seats: {m} seats want the own caches' kernels and the token loop ({} here)",
                prog.roles.name()
            ));
        }
        if let Some(e) = crate::compile::check::hazard::seats(m, p.units) {
            return Err(format!("seats: {e}"));
        }
        if m > MOST {
            return Err(format!("seats: {m} seats, past the {MOST} a ring holds"));
        }
        let ne = p.kv_entries.per_row;
        if ne < m {
            return Err(format!("seats: {m} seats over {ne} KV entries a row"));
        }
        for (kind, owns) in [
            (Kind::WEIGHT, &[][..]),
            (Kind::CX, &["xs"][..]),
            (Kind::CY, &["spos", "sbase"][..]),
            (Kind::HEAD_CTL, &[][..]),
        ] {
            let kb = ks
                .get_mut(&kind)
                .ok_or_else(|| format!("seats: no {} kernel", kind.name))?;
            kb.params.set("SEATS", Value::Int(i64::from(m)));
            kb.owns.extend(owns.iter().copied());
        }
        Ok(())
    }

    fn symbols(&self) -> &'static [(&'static str, &'static str)] {
        if self.m > 1 { SYMBOLS } else { &[] }
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
}
