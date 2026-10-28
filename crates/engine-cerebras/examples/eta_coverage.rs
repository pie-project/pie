//! Which registered guest programs the wafer's guest region could run:
//! `eta_coverage <dir of .ptir> [vocab]` (the files `PIE_SPATIAL_DUMP_PTIR`
//! makes) decodes each program, translates its epilogue to a guest3 table
//! with stand-in channel inputs of the declared shapes, and prints a line a
//! program: translated (instructions, registers) or why not, and the ops it
//! uses.

use std::collections::BTreeMap;

use eta_exec::Value;
use eta_ir::container::ChanDType;
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::Dtype;

use engine_cerebras::spatial::guest3::{numel, translate};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: eta_coverage <dir> [vocab]");
    let vocab: u32 = args.next().map_or(151_936, |v| v.parse().expect("a vocabulary size"));
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("the directory")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "ptir"))
        .collect();
    files.sort();
    for path in files {
        let name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        let bytes = std::fs::read(&path).expect("a program");
        let container = match eta_ir::container::decode(&bytes) {
            Ok(c) => c,
            Err(e) => {
                println!("{name} undecodable: {e:?}");
                continue;
            }
        };
        let Some(stage) = container.stages.iter().find(|s| s.stage == Stage::Epilogue) else {
            println!("{name} no epilogue");
            continue;
        };
        let mut kinds: BTreeMap<String, u32> = BTreeMap::new();
        for op in &stage.ops {
            let kind = format!("{op:?}");
            let kind = kind.split(['(', ' ', '{']).next().unwrap_or("?").to_string();
            *kinds.entry(kind).or_default() += 1;
        }
        // Stand-ins for what the epilogue takes: the shapes and dtypes declared.
        let mut seeds = BTreeMap::new();
        for op in &stage.ops {
            if let Op::ChanTake(c) | Op::ChanRead(c) = op {
                let decl = &container.channels[*c as usize];
                let n = numel(&decl.shape) as usize;
                let value = match decl.dtype {
                    ChanDType::Concrete(Dtype::F32) => Value::F32(vec![0.0; n]),
                    ChanDType::Concrete(Dtype::Bool) => Value::Bool(vec![0; n]),
                    ChanDType::Concrete(Dtype::U32) => Value::U32(vec![0; n]),
                    ChanDType::Concrete(Dtype::I32) => Value::I32(vec![0; n]),
                    _ => continue,
                };
                seeds.insert(*c, value);
            }
        }
        let verdict = match translate(&container, &seeds, vocab) {
            Ok(t) => format!("TRANSLATED {} instructions, {} wide registers", t.ins.len(), t.nreg),
            Err(why) => format!("host: {why}"),
        };
        println!("{name} {verdict} | ops {kinds:?}");
    }
}
