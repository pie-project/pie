//! The checkpoint onto the device, through `model::produce` and nothing else.
//!
//! # What this replaced
//!
//! `weights/{load,stage}.rs` and `loader/plan.rs`: a LOAD PLAN authored by
//! `model::boot::compile_load_plan_for` from a catalog row and an `Encoding`,
//! run through `model_loader::executor::Execution` in ≤4 GiB chunks, whose
//! output was a `HashMap<String, Slice>` keyed by CHECKPOINT names that a
//! separate `Names` table then joined to the trace's. Four crates and three
//! name spaces to get bytes from a file into a buffer.
//!
//! `model::produce` is the whole of it now. It takes the SKU's import table —
//! the same table `model::identify` matched the checkpoint against, so
//! identification and loadability are one question — and answers with the
//! tensors the plan's `params` name, already dense, row-major and canonical.
//! `driver-cuda/src/baker/mod.rs::upload` is the sibling and says the rest:
//! **the upload has no decision in it.**
//!
//! # One arena, and on this plane it is one memcpy
//!
//! Every tensor is a span of one allocation, for cuda's reason (an allocator's
//! live-byte accounting should not disagree with the device's) and one more of
//! this plane's: an argument table binds a BUFFER and a residency set tracks
//! one, so 260 allocations would be 260 residency entries.
//!
//! And there is no staging copy. `Allocation::new` takes
//! `MTLResourceOptions::StorageModeShared`, which on Apple silicon is memory
//! the host and the GPU both address, so a produced tensor is written straight
//! into the buffer at its offset. The chunking `weights/stage.rs` carried was
//! the ICB bind ceiling's — 4 GiB per bound buffer — and it belongs to the
//! BINDING rather than to the transfer; a model past that ceiling refuses by
//! name here rather than being silently split into buffers nothing can bind
//! as one.

use std::collections::BTreeMap;

use crate::baker::{Bank, Slice, arenas_of, join, readable_base};
use crate::device::{Allocation, Context};
use crate::error::{Error, Result};
use crate::layout::region::Region as _;
use model_ir::plan::Plan;

/// The ceiling a single Metal buffer may be bound at.
///
/// 4 GiB, and it is the INDIRECT COMMAND BUFFER's rather than the device's:
/// an ICB argument table encodes a buffer's length in 32 bits.
///
/// # It bounds ONE ARENA, not the model
///
/// It bounded the model until `gptoss-20b` asked for 12.82 GiB, and the
/// paragraph that stood here said why the alternative was worse: "several
/// buffers with the arena's offsets spanning them is a `Slice` that no single
/// binding can address". That objection is about a bank STRADDLING two
/// buffers, and it is right — a slice half in one allocation and half in the
/// next has no address a binding can use.
///
/// `walk::lane::arenas_of` never straddles. It fills an arena until the next
/// bank would cross the cap and then opens a new one, so **every bank lies
/// wholly inside exactly one allocation** and every allocation is under the
/// ceiling. The 32-bit length holds for each, which is the whole of what the
/// ICB asks.
///
/// What several allocations cost instead is residency: each is a `Regions`
/// span and a residency entry rather than one. That is a real cost and a
/// bounded one — four for gpt-oss — and it is the price of the plane holding
/// a model past two billion parameters at all.
const BIND_CEILING: u64 = 4 * 1024 * 1024 * 1024;

/// Every weight the plan names, on the device.
pub struct Weights {
    /// The arenas the banks are spans of, held to keep them alive.
    ///
    /// One for every model this plane held until `gptoss-20b`, and as many as
    /// [`BIND_CEILING`] needs since. Every [`Bank`] is a region inside exactly
    /// one of them, so dropping the vector frees every weight the lane is
    /// about to fire against.
    ///
    /// Read, unlike the single field it replaces: [`Weights::arenas`] hands
    /// them to `Regions` so a recorded command can name the buffer an address
    /// is in, and that was one call when there was one arena.
    owned: Vec<Allocation>,
    /// Every param the plan named, by the name the plan names it.
    pub banks: BTreeMap<String, Bank>,
}

impl Weights {
    /// Produce `rank`'s share of `sku`'s tensors out of `snapshot` and put
    /// them on the device.
    ///
    /// `rank` is the caller's, and on this plane it is always zero — see
    /// `serve::load`, which is the one site that states it and the one that
    /// refuses the rows a rank would be a guess for.
    ///
    /// # Errors
    ///
    /// A SKU with no safetensors import, a snapshot that does not read, a
    /// production the import refuses, a weight set past [`BIND_CEILING`], a
    /// device that declines the arena, or a param the import does not satisfy
    /// — each naming itself.
    pub fn produce(
        context: &Context,
        sku: &str,
        snapshot: &std::path::Path,
        plan: &Plan,
        rank: u32,
    ) -> Result<Self> {
        let unserved = |message: String| Error::Unserved {
            what: "load_model",
            message,
        };
        let base = readable_base(sku).map_err(unserved)?;
        let import = model::import_of(sku, base)
            .ok_or_else(|| unserved(format!("`{sku}` names no `{base}` import")))?;
        // `Snapshot::at` AND NOT `Snapshot::open`: `open` resolves a cache-dir
        // NAME under `$HOME/.cache/huggingface/hub`, and a driver is handed the
        // snapshot directory itself.
        let snap = model::snapshot::Snapshot::at(snapshot.to_path_buf()).ok_or_else(|| {
            unserved(format!(
                "no safetensors snapshot at {} — this driver produces its weights \
                 through `model::produce`, which reads a checkpoint directory holding \
                 `model.safetensors` or a shard index",
                snapshot.display()
            ))
        })?;
        // The plan's own `params` column, handed to the interpreter that is
        // about to be joined against it — `driver-cuda/src/baker/mod.rs::load`
        // is the sibling and passes the same pair. `Param::shape` is what THIS
        // RANK holds, so the cut is applied while the bytes are still host
        // bytes and the write loop below keeps having no decision in it. At
        // world 1 the cut is the identity and no byte moves.
        let produced = model::produce::produce(&import, &plan.params, rank, &|n| snap.read(n))
            .map_err(|e| unserved(format!("production refused: {e}")))?;

        // AS MANY ARENAS AS THE ICB'S CEILING NEEDS — see [`BIND_CEILING`] for
        // why several is sound where a single oversized one is not. One for
        // every SKU this plane held before `gptoss-20b`, four for that.
        let arenas = arenas_of(&produced, BIND_CEILING).map_err(unserved)?;
        let mut owned = Vec::with_capacity(arenas.len());
        let mut banks = BTreeMap::new();
        for arena in &arenas {
            let held = Allocation::new(context, arena.bytes, "a baker weight arena")?;
            let base_address = held.gpu_address();
            owned.push(held);
            let held = owned.last().expect("just pushed");
            for &(i, offset) in &arena.banks {
                let (name, t) = &produced[i];
                // SHARED STORAGE, SO THE UPLOAD IS THE WRITE. `Region::write`
                // bounds the span against the allocation's own length before
                // the first byte, which is what makes `arenas_of`'s arithmetic
                // auditable rather than trusted.
                //
                // SAFETY: `t.bytes` is a live host slice that cannot overlap
                // the device allocation, which was created for this load.
                unsafe { held.write(offset, &t.bytes) }?;
                banks.insert(
                    name.clone(),
                    Bank {
                        // THE ARENA'S OWN BASE, which is the whole of what
                        // several allocations changes down here: a bank in the
                        // fourth arena is an address like one in the first and
                        // no claim body can tell.
                        slice: Slice {
                            address: base_address + offset,
                            bytes: t.bytes.len() as u64,
                        },
                        shape: t.shape.clone(),
                        dtype: t.dtype,
                        // The demand side's own column, carried across by
                        // name. A produced row the plan binds no param for
                        // keeps an empty repr, which is a refusal at a bank
                        // slot — and no statement can name it anyway.
                        repr: plan
                            .params
                            .iter()
                            .find(|p| p.name == *name)
                            .map_or_else(String::new, |p| p.repr.clone()),
                    },
                );
            }
        }
        drop(produced);
        // The join `baker_load` proves, restated as a precondition: a missing
        // bank would be a zero-length binding at a `Const` slot and a shader
        // reading nothing, which is the worst place to find out.
        join(plan, &banks).map_err(unserved)?;
        Ok(Self { owned, banks })
    }

    /// The arenas, so a recorded fire can name the buffer an address is in.
    ///
    /// One call per allocation at the caller, because `Regions::add` takes one
    /// handle and the number of them is a fact about the model rather than
    /// about this driver.
    pub fn arenas(&self) -> impl Iterator<Item = &crate::device::Handle> {
        self.owned.iter().map(Allocation::handle)
    }
}
