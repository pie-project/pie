//! Encoding dispatches onto the GPU. The one half that needs a device.
//!
//! [`dispatch::plan`] decided everything: which entry point, which shader,
//! which grid, which addresses. What is left is three calls per launch — set
//! the pipeline, bind the operands, dispatch — and one compile pass in front
//! of them.
//!
//! # A symbol is a name, and that is the whole argument
//!
//! `Compiler::compile_batch` builds a pipeline from `(path, entry name)`. So
//! every symbol a text states reaches the GPU through the *same* three lines,
//! and adding a kernel to a text costs no code here. That is what
//! `driver-cuda-new`'s executor cannot do: a CUDA launcher is an authored C++
//! function, so its bridge grows an arm per kernel.
//!
//! [`dispatch::plan`]: super::dispatch::plan

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use objc2::runtime::ProtocolObject;
use objc2::rc::Retained;
use objc2_metal::MTLComputePipelineState;

use crate::error::{Error, Result};
use crate::metal::Context;
use crate::metal::{ArgumentTable, Compiler, StepEncoder};
use crate::shader::Request;

use super::dispatch::{Dispatch, pipelines_needed};

/// The pipelines a fire's symbols compile to, keyed by entry point.
///
/// Built once per fire — or once per process, since the map is additive and a
/// second fire naming the same symbols finds them. Nothing evicts: a model's
/// symbol set is bounded by its text, and a driver that recompiled a kernel
/// per fire would spend more time in the compiler than on the GPU.
pub struct Pipelines {
    /// Where the shader tree is rooted; a row's `file` is relative to it.
    root: PathBuf,
    built: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl std::fmt::Debug for Pipelines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipelines")
            .field("root", &self.root)
            .field("built", &self.built.len())
            .finish()
    }
}

impl Pipelines {
    /// An empty cache over the shader tree at `root`.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            built: HashMap::new(),
        }
    }

    /// The shader tree this cache compiles from.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether `symbol` has been compiled already.
    #[must_use]
    pub fn holds(&self, symbol: &str) -> bool {
        self.built.contains_key(symbol)
    }

    /// The pipeline for `symbol`, if it has been compiled.
    #[must_use]
    pub fn get(&self, symbol: &str) -> Option<&ProtocolObject<dyn MTLComputePipelineState>> {
        self.built.get(symbol).map(|p| &**p)
    }

    /// Compile every symbol `dispatches` names that is not held yet.
    ///
    /// One batch, so the shared files become one `MTLLibrary` each and a
    /// second run is served from the archive rather than rebuilt.
    ///
    /// # Errors
    ///
    /// The first symbol that would not build, named. A batch is positional, so
    /// a typo in one shader does not cost the twenty-nine that compiled — but
    /// a fire missing one kernel cannot run, so this reports rather than
    /// continues.
    pub fn ensure(
        &mut self,
        context: &Context,
        compiler: &Compiler,
        dispatches: &[Dispatch<'_>],
    ) -> Result<()> {
        let wanted: Vec<(&'static str, &str)> = pipelines_needed(dispatches)
            .into_iter()
            .filter(|(_, symbol)| !self.built.contains_key(*symbol))
            .collect();
        if wanted.is_empty() {
            return Ok(());
        }
        let requests: Vec<Request> = wanted
            .iter()
            .map(|(file, symbol)| Request::new(self.root.join(file), *symbol))
            .collect();
        let compiled = compiler.compile_batch(context, &requests);
        for ((_, symbol), built) in wanted.iter().zip(compiled.pipelines) {
            self.built.insert((*symbol).to_string(), built?);
        }
        Ok(())
    }
}

/// Encode one dispatch: pipeline, operands, grid.
///
/// The operands are bound at their **stated index**: argument `i` of the trace
/// is buffer `i` of the kernel. That is the trace's order (`inputs, outputs,
/// weights`) and nothing here reorders it — a driver that reordered operands
/// would be describing the kernel, which is the table's job.
///
/// # Errors
///
/// A symbol with no compiled pipeline, an operand past the table's bind count,
/// or a grid the pipeline refuses.
pub fn encode_one(
    encoder: &mut StepEncoder<'_>,
    table: &ArgumentTable,
    pipelines: &Pipelines,
    dispatch: &Dispatch<'_>,
) -> Result<()> {
    let pipeline = pipelines
        .get(dispatch.symbol)
        .ok_or_else(|| Error::Create {
            what: "dispatch",
            message: format!(
                "`{}` has no compiled pipeline; call `Pipelines::ensure` first",
                dispatch.symbol
            ),
        })?;
    encoder.set_pipeline(pipeline);
    for (index, arg) in dispatch.args.iter().enumerate() {
        table.bind_address(index, arg.slice.address)?;
    }
    encoder.set_argument_table(table);
    encoder.dispatch(
        [
            dispatch.grid[0] as usize,
            dispatch.grid[1] as usize,
            dispatch.grid[2] as usize,
        ],
        [
            dispatch.threadgroup[0] as usize,
            dispatch.threadgroup[1] as usize,
            dispatch.threadgroup[2] as usize,
        ],
    )
}

/// Encode a whole fire, in the order the lowering stated.
///
/// **This is the executor.** Six lines, one loop, no branch on anything.
///
/// # Errors
///
/// The first dispatch that would not encode, which stops the fire: a partially
/// encoded fire computes a prefix and leaves the rest of the arena holding
/// whatever the last one left.
pub fn encode(
    encoder: &mut StepEncoder<'_>,
    table: &ArgumentTable,
    pipelines: &Pipelines,
    dispatches: &[Dispatch<'_>],
) -> Result<()> {
    for dispatch in dispatches {
        encode_one(encoder, table, pipelines, dispatch)?;
    }
    Ok(())
}
