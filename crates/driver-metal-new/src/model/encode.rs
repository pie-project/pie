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
use crate::metal::{ArgumentTable, Compiler, Handle, StepEncoder, allocate};
use crate::region::Region as _;
use crate::shader::Request;

use super::dispatch::{Dispatch, pipelines_needed};

/// The scalars a fire's statements state, in one device buffer.
///
/// MTL4 argument tables bind **addresses**, not bytes, so a kernel taking a
/// `const constant uint&` needs a buffer to point at. One buffer per fire
/// rather than one per launch: the values are known before any encoding
/// starts, so they are written once and each dispatch binds a slice of the
/// same region. A buffer per dispatch would allocate 367 times for a fire
/// whose scalars total a few dozen bytes.
pub struct Params {
    region: Handle,
    /// Byte offset of each dispatch's run, parallel to the dispatch list.
    offsets: Vec<u64>,
}

impl std::fmt::Debug for Params {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Params")
            .field("runs", &self.offsets.len())
            .finish_non_exhaustive()
    }
}

impl Params {
    /// Stage every dispatch's scalars into one buffer, in dispatch order.
    ///
    /// # Errors
    ///
    /// The allocation, or a write past it.
    pub fn stage(context: &Context, dispatches: &[Dispatch<'_>]) -> Result<Self> {
        let total: usize = dispatches.iter().map(|d| d.params.len()).sum();
        // A fire whose statements state no scalars still gets a region: a
        // zero-length allocation has no address to bind, and an unbound slot
        // is a kernel reading whatever the last step left there.
        let bytes = (total * size_of::<u32>()).max(size_of::<u32>()) as u64;
        let region = allocate(context, bytes, "launch params")?;
        let mut offsets = Vec::with_capacity(dispatches.len());
        let mut at = 0u64;
        for d in dispatches {
            offsets.push(at);
            if d.params.is_empty() {
                continue;
            }
            let raw: &[u8] = unsafe {
                core::slice::from_raw_parts(
                    d.params.as_ptr().cast::<u8>(),
                    core::mem::size_of_val(d.params),
                )
            };
            // SAFETY: `region` was allocated to hold every run; this one
            // starts at `at`, which advances by exactly the bytes written.
            unsafe { region.write(at, raw)? };
            at += core::mem::size_of_val(d.params) as u64;
        }
        Ok(Self { region, offsets })
    }

    /// The GPU address of dispatch `index`'s scalars.
    #[must_use]
    pub fn address_of(&self, index: usize) -> Option<u64> {
        self.offsets
            .get(index)
            .map(|at| self.region.gpu_address() + at)
    }
}

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
    params: &Params,
    index: usize,
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
    for (slot, arg) in dispatch.args.iter().enumerate() {
        table.bind_address(slot, arg.slice.address)?;
    }
    // The scalars, at the slots the ROW placed them. Scalar `i` binds at
    // `base + i * 4`, which serves both spellings in the tree: a packed
    // `constant RouterParams&` is the address of its first field, and a
    // separate `const constant int&` is the address of that scalar. A row
    // stating one `Param(0)` describes both at once.
    //
    // One slot and not one each, because that is what the shader tree already
    // does: `moe/route.metal` takes `constant RouterParams&`, `norm/rms.metal`
    // takes its own struct, and every such struct is a run of `unsigned int`
    // with no padding. A statement's `params` in stated order IS that struct,
    // so the address of the run is the address of the struct.
    //
    // The alternative — a slot per scalar — was what this did first, and it
    // serves exactly one kernel: the QKV split, whose shader was written here
    // and could be written either way. Every kernel that already existed
    // wanted the packed form, so the packed form is the convention and the
    // split's shader was changed to match it.
    if !dispatch.params.is_empty() {
        let base = params.address_of(index).ok_or_else(|| Error::Create {
            what: "dispatch",
            message: format!(
                "`{}` states {} scalar(s) but was not staged",
                dispatch.symbol,
                dispatch.params.len()
            ),
        })?;
        for &(slot, which) in &dispatch.param_slots {
            table.bind_address(slot, base + u64::from(which) * size_of::<u32>() as u64)?;
        }
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
    params: &Params,
    dispatches: &[Dispatch<'_>],
) -> Result<()> {
    for (index, dispatch) in dispatches.iter().enumerate() {
        encode_one(encoder, table, pipelines, params, index, dispatch)?;
    }
    Ok(())
}
