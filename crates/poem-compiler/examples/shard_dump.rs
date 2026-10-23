//! Every catalog deployment's shard plan at tp 1, 2, 4 and 8, by dataflow
//! alone: the params, every value's per-rank type, each node's operands and
//! every inserted collective. Two trees' dumps are equal when an IR change
//! left sharding alone; a refusal names its op, so a rename shows there only.
//!
//! ```text
//! cargo run --release -p pie-poem-compiler --example shard_dump -- OUT
//! ```
//!
//! `scripts/equivalence/README.md` is the procedure this is one part of.

use poem::Platform;
use poem_ir::{Operands, Operation};
use std::fmt::Write as _;

fn main() {
    let path = std::env::args().nth(1).expect("usage: shard_dump OUT");
    let mut out = String::new();
    for row in poem_compiler::catalog::deployments() {
        let Ok(trace) = row.try_trace(Platform::Cuda) else {
            let _ = writeln!(out, "## {} untraceable", row.name);
            continue;
        };
        for tp in [1u32, 2, 4, 8] {
            let _ = writeln!(out, "## {} tp{tp}", row.name);
            match poem_compiler::shard::shard(trace.clone(), tp) {
                Err(why) => {
                    let _ = writeln!(out, "refused: {why}");
                }
                Ok(t) => {
                    let _ = writeln!(out, "params {:?}", t.params);
                    let _ = writeln!(out, "values {:?}", t.values);
                    for node in &t.nodes {
                        let (mut i, mut o) = (Vec::new(), Vec::new());
                        node.op.inputs(&mut i);
                        node.op.outputs(&mut o);
                        i.sort();
                        i.dedup();
                        if matches!(node.op, Operation::Collective(_)) {
                            let _ = writeln!(out, "C {:?}", node.op);
                        } else {
                            let _ = writeln!(out, "n {i:?} -> {o:?}");
                        }
                    }
                }
            }
        }
    }
    std::fs::write(&path, out).unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
}
