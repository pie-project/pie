//! A matmul in the GEMM regime (`gemm_pe.csl`'s `gemm`): a class's rows times a weight block off
//! the tape below the array, every PE of a row reading every other's piece along the row line.

use crate::compile::stream::gemm::{GOp, GemmProgram, piece};
use crate::kernel::{GemmForm, Kernel, LineAlong, Source};
use crate::tuning;

/// Op `i`'s cycles, by term: each column piece's turn, its input piece, each output segment's
/// block off the column's tree and its MACs (a call a row and half).
fn price(p: &GemmProgram, i: usize, terms: &mut [f64; 5]) -> f64 {
    let GOp::Gemm {
        k, y, x32, lane, ..
    } = &p.l.ops[i]
    else {
        return 0.0;
    };
    let (c, tb) = (p.c, p.tb);
    let rows = if *lane { 1 } else { tb };
    let segs = crate::compile::stream::gemm::local(&p.l.vals[y], c, 0);
    for cp in 0..c {
        let (a, b) = piece(cp, c, *k);
        let kc = b - a;
        if kc == 0 {
            continue;
        }
        let xw = (rows * kc).div_ceil(2) * if *x32 { 2 } else { 1 };
        terms[0] += tuning::get().gemm_turn + tuning::get().gemm_piece;
        terms[1] += 3.0 * f64::from(xw);
        for s in &segs {
            let w = s.4;
            let wb = f64::from(kc * w / 2);
            // the block off the column's tree
            terms[2] += wb + tuning::get().gemm_call;
            // (a call a row and half: its block's MACs)
            let calls = f64::from(rows) * if *x32 { 2.0 } else { 1.0 };
            terms[3] += calls * tuning::get().gemm_mac_call;
            terms[4] += calls * tuning::get().gemm_mac_elem * f64::from(kc * w);
        }
    }
    terms.iter().sum()
}

/// A GEMM.
pub struct Gemm;

/// The registry's GEMM.
pub static GEMM: Gemm = Gemm;

/// Its GEMM form: the row line, which orders the row (no PE leaves before every other has
/// reached it, but for its own tail).
static FORM: GemmForm = GemmForm {
    func: "gemm",
    what: "a GEMM",
    line: LineAlong::Row,
    masked: true,
    orders_row: |_| true,
    what_price: "gemm",
    price,
    drift: "",
    drift_after: "",
};

impl Kernel for Gemm {
    fn name(&self) -> &'static str {
        "linear.gemm"
    }

    fn ops(&self) -> &'static [&'static str] {
        &["linear.matmul"]
    }

    fn sources(&self) -> &'static [Source] {
        &[]
    }

    fn gemm(&self) -> Option<&'static GemmForm> {
        Some(&FORM)
    }
}
