//! The Neural Engine program of a split MLP, run through the private interface
//! in `private.rs`. TEMPORARY like that interface: the program format (MIL
//! text with `tensor_buffer` IOSurface inputs) is undocumented and may change.

use std::path::Path;

use crate::private::{Binding, Program};

mod memory;
mod program;
mod shape;

pub use memory::{Memory, Weights};
pub use program::{function_name, function_rows, signs};
pub use shape::*;

pub struct Ffn {
    pub program: Program,
    pub evaluations: Vec<(u32, [Binding; 2])>,
}

impl Ffn {
    pub fn compile(shape: &Shape, memory: &Memory, cache: &Path) -> Result<Ffn, String> {
        let list = memory::inputs(shape, memory);
        let rows = function_rows();
        let mut text = String::from("program(1.3)\n{\n");
        for &count in &rows {
            text.push_str(&program::function(shape, &list, &memory.partial, count));
        }
        text.push_str("}\n");
        let program = Program::compile(&text, &program::rotation_blob(shape.ane, &signs()), cache)?;
        let mut evaluations = Vec::new();
        for &count in &rows {
            let procedure = program
                .procedure(&function_name(count))
                .ok_or_else(|| format!("the program has no {}", function_name(count)))?;
            let names = program.inputs(procedure);
            let bind = |set: usize| -> Result<Binding, String> {
                let surfaces = names
                    .iter()
                    .map(|name| {
                        list.iter()
                            .find(|input| &input.name == name)
                            .map(|input| input.surfaces[set])
                            .ok_or_else(|| format!("the program has an unknown input {name}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                program.bind(procedure, &surfaces, &memory.partial)
            };
            evaluations.push((count, [bind(0)?, bind(1)?]));
        }
        Ok(Ffn {
            program,
            evaluations,
        })
    }

    #[must_use]
    pub fn evaluation(&self, rows: u32) -> Option<usize> {
        self.evaluations
            .iter()
            .position(|(count, _)| *count >= rows)
    }
}
