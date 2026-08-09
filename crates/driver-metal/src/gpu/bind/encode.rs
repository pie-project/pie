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
//! `driver-cuda`'s executor cannot do: a CUDA launcher is an authored C++
//! function, so its bridge grows an arm per kernel.
//!
//! [`dispatch::plan`]: crate::lowering::dispatch::plan

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use objc2::runtime::ProtocolObject;
use objc2::rc::Retained;
use objc2_metal::MTLComputePipelineState;

use crate::device::recording::{Bind, Command};

use crate::error::{Error, Result};
use crate::Context;
use crate::{ArgumentTable, Compiler, StepEncoder};
use crate::layout::region::Region as _;
use crate::layout::shader::Request;

use crate::lowering::dispatch::{Dispatch, pipelines_needed};

/// The scalars a fire's statements state, in one device buffer.
///
/// MTL4 argument tables bind **addresses**, not bytes, so a kernel taking a
/// `const constant uint&` needs a buffer to point at. One buffer per fire
/// rather than one per launch: the values are known before any encoding
/// starts, so they are written once and each dispatch binds a slice of the
/// same region. A buffer per dispatch would allocate 367 times for a fire
/// whose scalars total a few dozen bytes.
pub struct Params {
    /// LEASED, not allocated. See `gpu::fire::scratch`: allocating this per fire
    /// leaks it into the residency set permanently and moves its address, and
    /// the address is one of only three things that vary between two fires of
    /// one shape.
    region: crate::Lease,
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
        Self::stage_in(context, &crate::Scratch::new(), dispatches)
    }

    /// The same, out of a pool the caller keeps.
    ///
    /// The pooled form is the one a serving path wants: `stage` above makes a
    /// throwaway pool, which allocates every time and is right only for a
    /// caller that fires once.
    ///
    /// # Errors
    ///
    /// As [`Self::stage`].
    pub fn stage_in(
        context: &Context,
        scratch: &crate::Scratch,
        dispatches: &[Dispatch<'_>],
    ) -> Result<Self> {
        // Each dispatch's run is as wide as its layout says, because a row
        // may widen a scalar: `sdpa_vector_decode` reads its strides as
        // `size_t`, eight bytes, from a channel whose values are `u32`.
        let width = |d: &Dispatch<'_>| -> usize {
            if d.params.is_empty() {
                return 0;
            }
            d.param_slots
                .iter()
                .map(|p| {
                    if p.packed {
                        // The whole run from this slot's first scalar.
                        p.at as usize
                            + (d.params.len() - usize::from(p.value.unwrap_or(0)))
                                * size_of::<u32>()
                    } else {
                        (p.at + p.bytes) as usize
                    }
                })
                .max()
                .unwrap_or(0)
        };
        let total: usize = dispatches.iter().map(|d| width(d).div_ceil(8) * 8).sum();
        // A fire whose statements state no scalars still gets a region: a
        // zero-length allocation has no address to bind, and an unbound slot
        // is a kernel reading whatever the last step left there.
        let bytes = total.max(size_of::<u64>()) as u64;
        let region = scratch.take(context, bytes, "launch params")?;
        // A REUSED region holds the last fire's scalars, and a dispatch whose
        // run is shorter than its predecessor's would read the tail of that
        // one. Zeroed for the same reason the arena is.
        //
        // SAFETY: leased exclusively; nothing is encoded against it yet.
        unsafe { region.zero(0, region.len())? };
        let mut offsets = Vec::with_capacity(dispatches.len());
        let mut at = 0u64;
        for d in dispatches {
            offsets.push(at);
            if d.params.is_empty() {
                continue;
            }
            // Written per SLOT, widened where the row says. A four-byte value
            // handed to an eight-byte read would otherwise take the next
            // scalar as its high half.
            let mut run = vec![0u8; width(d)];
            for p in &d.param_slots {
                let Some(v) = p.value.and_then(|i| d.params.get(i as usize)) else {
                    continue;
                };
                let start = p.at as usize;
                if p.packed {
                    // Every remaining scalar, in stated order — the struct.
                    let from = usize::from(p.value.unwrap_or(0));
                    for (n, value) in d.params[from..].iter().enumerate() {
                        let at = start + n * size_of::<u32>();
                        run[at..at + 4].copy_from_slice(&value.to_le_bytes());
                    }
                } else if p.bytes == 8 {
                    run[start..start + 8].copy_from_slice(&u64::from(*v).to_le_bytes());
                } else {
                    run[start..start + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            // SAFETY: `region` was allocated to hold every run; this one
            // starts at `at`, which advances by exactly the bytes written.
            unsafe { region.write(at, &run)? };
            at += (run.len().div_ceil(8) * 8) as u64;
        }
        Ok(Self { region, offsets })
    }

    /// The region the scalars live in.
    ///
    /// For `gpu::device::recording::record`, which has to turn a scalar's ADDRESS back into
    /// the buffer holding it — a recorded command binds a buffer.
    #[must_use]
    pub fn region(&self) -> &crate::Handle {
        &self.region
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
        for p in &dispatch.param_slots {
            table.bind_address(p.slot, base + u64::from(p.at))?;
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
/// **This is the executor.** One loop, a barrier, no branch on anything.
///
/// # The barrier
///
/// Metal does NOT order two dispatches in one compute encoder. Without a
/// barrier they may overlap, so a statement reading what the previous one
/// wrote reads whatever was there — and the failure is not a crash but a
/// *number*, sometimes right.
///
/// Measured before this line existed: three runs of one fire over one
/// checkpoint's weights gave widest activations of 11.7, 23.1 and 4.5e12.
/// Two of the three looked entirely plausible. That is the whole argument for
/// the barrier being unconditional here — a hazard that produces a believable
/// answer two times in three is one no amount of eyeballing finds.
///
/// After EVERY dispatch, not between the ones that alias. `lowering::executor::bind` asks
/// `barrier_after` because its DAG states which launches run concurrently;
/// this walk has no such statement and inventing one would be the driver
/// deciding something about a text. A text that wants concurrency can say so
/// when there is something to say it with; until then the correct order is the
/// stated one.
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
        encoder.barrier(crate::Visibility::Device);
    }
    Ok(())
}

/// Lower a fire's dispatches into the commands a recording is made of.
///
/// # Why this is here and not in `recording`
///
/// `.wiki/driver/real-metal-north-star.md` §9: **layers point down.**
/// `gpu::device::recording` used to import `lowering::dispatch::Dispatch`
/// while `gpu::fire::run` imported `Recordings` — a cycle, and an ICB path
/// that knew what a fire was. What a recording needs is a pipeline, some
/// addresses and a grid; turning a `Dispatch` into those three is this
/// layer's job, because this is the layer where `Pipelines` and `Params`
/// already live and where the identical walk is done for encoding.
///
/// The walk IS identical, and that is the point of it being one function:
/// a recording that binds a different set of slots from the encode it stands
/// in for is the failure `device_icb.rs` exists to catch, and it is caught
/// by comparing outputs rather than by two people reading two loops.
///
/// # Errors
///
/// A symbol with no compiled pipeline, or a dispatch that states scalars
/// while `Params` has no staged region for it. Both are refused by name:
/// a recording is replayed blind, so a slot filled with a guess is a kernel
/// reading someone else's memory and saying nothing.
pub fn commands<'a>(
    pipelines: &'a Pipelines,
    params: &Params,
    dispatches: &'a [Dispatch<'a>],
) -> Result<Vec<Command<'a>>> {
    dispatches
        .iter()
        .enumerate()
        .map(|(index, dispatch)| {
            let pipeline = pipelines.get(dispatch.symbol).ok_or_else(|| Error::Create {
                what: "recording",
                message: format!("`{}` has no compiled pipeline", dispatch.symbol),
            })?;
            let mut binds: Vec<Bind> = dispatch
                .args
                .iter()
                .enumerate()
                .map(|(slot, arg)| Bind {
                    address: arg.slice.address,
                    slot,
                })
                .collect();
            if !dispatch.params.is_empty() {
                let base = params.address_of(index).ok_or_else(|| Error::Create {
                    what: "recording",
                    message: format!("`{}` states scalars but was not staged", dispatch.symbol),
                })?;
                binds.extend(dispatch.param_slots.iter().map(|p| Bind {
                    address: base + u64::from(p.at),
                    slot: p.slot,
                }));
            }
            Ok(Command {
                pipeline,
                binds,
                grid: dispatch.grid,
                threadgroup: dispatch.threadgroup,
                symbol: dispatch.symbol,
            })
        })
        .collect()
}
