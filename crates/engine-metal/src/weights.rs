//! The checkpoint, resident: one device allocation, one row per
//! `Trace::params` — and, in the same allocation and the same table, the
//! adapter banks (design §8).
//!
//! # Why the banks live HERE and not in `store/`
//!
//! They are runtime-MUTABLE, which is `store/`'s whole subject, and the pull
//! toward putting them beside the kv pages is real. Three things decide it the
//! other way.
//!
//! **The table has to be one table.** `Def::Weight(i)` resolves through
//! `WeightTable` and nowhere else (`Run::tensor` is the crate's heart, and its
//! whole point is that provenance handling exists exactly once). A bank is a
//! `Def::Weight` — that is the seat design §8's open item asked for, and the
//! reason it is the right seat is that MoE already proved a routed bank needs
//! no `Def` of its own. Splitting the STORAGE while sharing the TABLE would
//! put one invariant in two modules; splitting both would mean a second
//! resolution path in `Run` for a value that is a weight.
//!
//! **What `store/` owns is per-SEQUENCE and this is not.** A kv page belongs
//! to a slot, a recurrent slab belongs to a slot, and `Pools::clear(slot)`
//! is a sentence about a sequence beginning. A bank row belongs to no
//! sequence: many lanes route to it in one fire, it outlives every one of
//! them, and nothing about it is per slot.
//!
//! **Mutability here is not the fire path's.** `store/` is written by
//! LAUNCHES, inside the graph, every fire. A bank is written by
//! `register_adapter`, between fires, on the host — the same instant and the
//! same mechanism the checkpoint itself was landed by. It is a second load of
//! a few rows, not a pool.
//!
//! What the banks do inherit from the pools is the stability rule: the store
//! is reserved at the model text's declared capacity and never grows, so a
//! view recorded at load stays the bytes a later fire reads. Registering the
//! thirty-second adapter moves nothing.
//!
//! # ONE ALLOCATION, AND A STREAMED LOAD STILL ONLY HAS ONE
//!
//! A load whose `device_weight_budget` cannot hold its routed banks
//! (`crate::experts`) does not get a second device allocation and does not get
//! a second table. What it gets is a smaller RESERVATION for the bands it
//! streams — `slots` expert seats where the plan declares `experts` of them,
//! which is [`places`] reading [`Plan::resident`] and nothing more — and one
//! host mapping holding those bands whole ([`crate::host_source`]), which the
//! landing sink fills instead of the store. The weight rows are minted exactly as they always
//! were: a `Tensor` over the seats, at the same alignment, resolved through
//! the same handle table. Every kernel that reads one computes
//! `base + e * stride` and is told nothing, because between two command
//! buffers the host has rewritten `e` to name a SEAT (`crate::experts`'s
//! header is where that argument lives).
//!
//! So the two facts this module's header opens with are both still true, and
//! the third — that the store never moves — is what makes the whole mechanism
//! legal: a seat is written in place, at a fixed address, while no command
//! buffer is running.
//!
//! # A weight row is a HANDLE, and that is what "stability" means here
//!
//! **THE CUDA SIBLING'S ROWS ARE ADDRESSES; THESE ARE TABLE ROWS.** A
//! `kernels_metal::Tensor` carries a `u32` into [`Handles`], because a
//! compute encoder binds a buffer and an offset and there is no pointer to
//! hand out. So [`Weights::resident`] takes the handle table and mints one
//! row per param — and those rows are LOAD-LIVED, which is the whole reason
//! [`Handles::seal`] exists: the caller seals immediately after this
//! constructor returns, and every fire afterwards mints above the watermark
//! and rewinds back down to it. Sealing is the CALLER's call, not this
//! module's, because the seal covers the pools' and the inputs' load-time
//! rows too and only the shell knows when the last of them is minted.
//!
//! The reservation itself takes a [`Context`](crate::device::Context) for the
//! same reason every reservation in this shell does: a Metal buffer is made
//! by a call ON a device, where `cudaMalloc` reads the thread's current one
//! out of ambient state.
//!
//! # The contract arrives; the family does not
//!
//! This module takes a [`ModelContract`] and never asks which model it
//! describes. That is decision #18 read from the residency side: the runtime
//! links `model`, traces the `Trace` and states the load contract; the shell
//! compiles the contract against a checkpoint and lands the bytes. A shell
//! that reached for `models::qwen_3` would be a shell that has to grow an arm
//! per family, which is exactly the shape the string-plan era had.
//!
//! # Names are the plan's, both sides
//!
//! The load contract publishes one `Visibility::Public` entry per plan param,
//! **under the param's own name** — `embed`, `layer.7.q_proj` — and the
//! checkpoint's spellings live inside the expressions. So the bijection this
//! module needs is `trace.params[i].name == published name`, which is the
//! property `model/tests/the_zt_contract_states_the_cut.rs` asserts for every
//! SKU in the catalog. Both directions are checked here anyway, because the
//! consequence of a hole is not a missing tensor: it is
//! [`WeightTable`]'s `None`, reached at the first fire, in a panic.
//!
//! # Why the transforms run on the host
//!
//! The plan is compiled at `BackendKind::Metal` — that is what fixes the
//! alignment and the tile budget — but the arena handed to the executor is a
//! `Vec<u8>`, so `ArenaBacking::runs_named_kernels` is false and every cast
//! runs host-side. Here that is not a temporary state of affairs the way it
//! is on the CUDA side: there is one device arena for load transforms in this
//! tree and it is `engine-cuda`'s `weights::arena`, with no metal twin, so
//! there is no device path to fall back FROM. (It stood in `checkpoint`'s
//! `executor` behind a `cuda` feature until it moved to the crate that owns
//! the device; the seam it plugs into, `ArenaBacking`, is still the loader's
//! and is still backend-neutral.) The mask the loader admits for this backend
//! (`METAL_TILE_MAP_MASK`) is chosen on exactly that basis, and every
//! transform it admits has a host implementation. For the SKUs this shell
//! serves today it is a handful of bf16→f32 widenings on norm scales.
//!
//! A STREAMED load hands no arena at all — see the branch in
//! [`Weights::resident`] — and the sentence above still holds for it: the
//! executor's streaming residency owns each buffer as a host allocation and
//! frees it at its last use, so the transforms run in the same place, on
//! bytes nothing kept.
//!
//! [`Handles`]: crate::device::Handles
//! [`Handles::seal`]: crate::device::Handles::seal

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;

// The two readers. `zt` comes in by module because its door is spelled
// `parse` — the module path is what says what is parsed, and a bare `parse`
// beside `parse_metadata` would say nothing about which of the two doors
// this line opened.
use checkpoint::file::read::parse_metadata;
use checkpoint::file::serve;
use checkpoint::file::zt;
use checkpoint::contract::ModelContract;
use checkpoint::error::Error as LoadError;
use checkpoint::executor::{Execution, sink::TensorSink};
use checkpoint::plan::{LoadPlan, StorageTarget, compile, compile_streaming};
// The `pie.serving/1` vocabulary, consumed and never re-spelled: `Stamp::of`
// is the constructor that fills the four policy fields from the profile's own
// constants, and `Mismatch::refuse` is the one place the refusal is written.
use checkpoint::serving::{self, Stamp};
use checkpoint::types::{BackendKind, ScaleForm, TensorId};
use kernels_metal::Tensor;
use model_ir::{Dtype, ParamSource, Trace};

use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use crate::experts::{Attachments, Plan, Tier};
use crate::host_source::HostSource;
use crate::run::{WeightRow, WeightTable};

/// What a matrix operand wants, and what a Metal buffer's own base is already
/// aligned to — so a view into the store is as aligned as its own reservation
/// would have been. The same number the loader's `StorageTarget` states for
/// this backend, for the same reason; `newBufferWithLength:` returns
/// page-aligned storage, so the constant is about the VIEWS inside it.
pub(crate) const ALIGN: u64 = 256;

/// One plane of a registered adapter, as the caller hands it over.
///
/// **FULL CAPACITY, NOT THE ADAPTER'S OWN RANK.** The bytes are exactly one
/// slot of `bank`, which is that bank's declared `[rank, in]` or `[out, rank]`
/// rectangle in the bank's own dtype. An adapter trained at a lower rank is
/// zero-padded by the CALLER, because only the caller knows the layout it is
/// padding: `A`'s unused ranks are trailing rows and `B`'s are a stride inside
/// every row, so "write the prefix and leave the rest" is right for one plane
/// and wrong for the other. A short plane is a refusal naming both numbers,
/// never a partial write.
#[derive(Debug, Clone, Copy)]
pub struct AdapterPlane<'a> {
    /// The bank param this plane fills, as `Trace::params` names it.
    pub bank: &'a str,
    /// One slot's worth of bytes.
    pub bytes: &'a [u8],
}

/// Every weight this model needs, on the device — the checkpoint's and the
/// banks'.
#[derive(Debug)]
pub struct Weights {
    store: Buffer,
    table: WeightTable,
    /// **The routed-expert tier, opened only for a streamed load**
    /// (`crate::experts`): the host band table, the wired slabs, and the seat
    /// bookkeeping the segment cut drives.
    ///
    /// `None` is full residency and is what every load in this workspace does
    /// today; everything downstream asks this field whether a fire is cut at
    /// all, so a full-residency load pays nothing for the mechanism existing.
    ///
    /// A `RefCell` because the swap happens INSIDE the walk — `walk_once`
    /// takes `&self`, the encode sink takes `&self`, and what moves between
    /// two segments is seats. The cell's borrow is taken and dropped inside
    /// one cut, between two command buffers, on the lane thread.
    tier: Option<RefCell<Tier>>,
    /// Bank name -> (param index, adapters, bytes per adapter slot). Built at
    /// load off `ParamSource::Registered`, which is the only place a bank is
    /// declared; `register_adapter` is a lookup in here and a write.
    banks: BTreeMap<String, Bank>,
}

/// One declared adapter bank: where its slots are and how big they are.
#[derive(Debug, Clone, Copy)]
struct Bank {
    /// Where the bank's first slot starts in the store.
    offset: u64,
    /// How many adapters the first axis seats — the ceiling
    /// `Budget::max_adapters` is checked against, stated as a shape.
    adapters: u32,
    /// One adapter's bytes in this bank.
    slot: u64,
    /// The leading axis of ONE slot.
    rows: u64,
    /// The trailing axes of one slot, multiplied out.
    cols: u64,
    /// One element in bytes.
    elem: u64,
}

/// **ONE BANK, AS THE ADAPTER RESOLVER READS IT** (alto adapter §6.3).
///
/// Everything [`crate::adapter`] needs to slice a `[layers, ...]` seed into
/// one full-capacity plane per bank: the name it registers under, the
/// capacity, the slot's bytes, and the rectangle those bytes are. A flattened
/// [`Bank`] — the resolver lives in another module and a private field is not
/// a contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankSeat {
    /// The param's own name, which is what a registration names.
    pub name: String,
    /// How many adapters it seats.
    pub adapters: u32,
    /// One adapter's bytes.
    pub slot: u64,
    /// The leading axis of one slot.
    pub rows: u64,
    /// The trailing axes of one slot, multiplied out.
    pub cols: u64,
    /// One element in bytes.
    pub elem: u64,
}


/// The banks a plan declares, read off `ParamSource::Registered`.
///
/// **ONE STATEMENT OF WHAT A BANK IS, AND IT IS THE PARAM'S OWN.** No suffix
/// convention, no name matching, no pairing table: a bank is a param the model
/// text marked registered, its capacity is that param's leading axis, and one
/// adapter's slot is everything after it. `A` and `B` are two independent
/// banks here — the op pairs them, this module does not, and a registration
/// names each by the name the plan gave it.
fn banks(trace: &Trace, places: &[Place]) -> BTreeMap<String, Bank> {
    trace.params
        .iter()
        .zip(places)
        .filter(|(param, _)| param.source == ParamSource::Registered)
        .map(|(param, place)| {
            let adapters =
                u32::try_from(param.shape.first().copied().unwrap_or(0)).unwrap_or(u32::MAX);
            // The DECLARED plane, not the reserved one: an adapter bank is
            // `ParamSource::Registered` and is never routed, so the two are
            // equal — and reading `full` is what keeps them equal the day a
            // plan states a bank that is both.
            let slot = if adapters == 0 {
                0
            } else {
                place.full / u64::from(adapters)
            };
            // The slot's own rectangle: the param's shape with the adapters
            // axis cut off. `[adapters, rank, in]` is a `[rank, in]` slot and
            // `[adapters, out, rank]` is an `[out, rank]` one.
            let (rows, cols) = rectangle(param.shape.get(1..).unwrap_or(&[]));
            (
                param.name.clone(),
                Bank {
                    offset: place.offset,
                    adapters,
                    slot,
                    rows,
                    cols,
                    elem: model_compiler::arena::elem_bytes(param.dtype).unwrap_or(0),
                },
            )
        })
        .collect()
}

impl Weights {
    /// Land `contract` against the checkpoint at `path`.
    ///
    /// `path` is a snapshot directory or a single container; a directory is
    /// discovered the way `pie model import` discovers one, a file is read
    /// directly.
    ///
    /// `device` is the reservation's and `handles` is the table the returned
    /// [`WeightTable`]'s rows point into — the two parameters the CUDA twin
    /// does not carry, for the two reasons this module's header gives. The
    /// caller seals `handles` once every load-time row is minted; this
    /// constructor does not, because it is not the last minter.
    ///
    /// `tp_size` and `precision` are **the two deployment facts a shell cannot
    /// look up** (§M-4c): a shell must not know a model family, so the
    /// degree and the served numeric form arrive from the runtime, through
    /// `LoadRequest` and `Boot`. The other three the artifact is checked
    /// against are already here — `trace.platform.backend()` is the backend and
    /// `trace.name` is the SKU — and are deliberately not restated beside
    /// them, for the reason `request_of` gives: a fact carried twice is a fact
    /// that can disagree with itself. [`serves_this_deployment`] is where the
    /// whole of the rule is written down.
    ///
    /// # Errors
    ///
    /// [`Fault::Recipe`] for a serving artifact stamped for another
    /// deployment, [`Fault::Load`] for a checkpoint the contract does not fit,
    /// [`Fault::Param`] for a plan and a contract that do not name the same
    /// tensors, [`Fault::Ceiling`] for a store past `maxBufferLength` or a
    /// view that leaves it, [`Fault::Device`] when the device declined the
    /// length, [`Fault::Deviceless`] for a non-Apple build.
    pub fn resident(
        device: &Context,
        handles: &Handles,
        trace: &Trace,
        contract: &ModelContract,
        path: &Path,
        plan: &Plan,
        tp_size: u64,
        precision: &str,
    ) -> Result<Weights> {
        // **BEFORE THE METADATA PARSE**, because this refusal is about the
        // TRACE and not about the file: a plane order this shell cannot read
        // is unreadable whichever checkpoint states it, and answering it with
        // a checkpoint's sentence would send the reader to the wrong crate.
        // See `readable_plane_orders`.
        readable_plane_orders(trace)?;

        // **AND THEN THE STAMP, STILL BEFORE ANYTHING IS TAKEN** (§M-4c).
        // Every line below this one spends something the machine cannot give
        // back for free — `Buffer::zeroed` reserves the whole store,
        // `HostSource::open` takes a descriptor and an address range, the
        // executor's arena is the image's own size — and an artifact compiled
        // for another shell's kernels is refused whichever of them it would
        // have filled. The order is the promise: a cross-recipe boot costs one
        // manifest read and no reservation at all.
        serves_this_deployment(
            path,
            trace.platform.backend(),
            &trace.name,
            tp_size,
            precision,
        )?;

        let (metadata, snapshot) = if path.is_dir() {
            (parse_metadata(path)?, path)
        } else {
            (zt::parse(path)?, path.parent().unwrap_or(Path::new(".")))
        };

        // tp=1: the plan's `Shard::Cut` segments still describe the whole
        // tensor, and a rank of one takes all of them. `tp_rank`/`tp_size`
        // are the loader's whole notion of a rank, so a shell that grows
        // tensor parallelism states it here and nowhere else.
        let target = StorageTarget::for_backend(BackendKind::Metal, 0, 1);
        let landing = compile(&metadata, contract, target.clone())?;

        let places = places(trace, plan)?;
        let total = places.last().map_or(0, |p| p.offset + p.reserved);
        let mut store = Buffer::zeroed(device, total)?;

        let index: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();
        // The host band table, sized off the plan and empty for a
        // full-residency load — where a streamed band's bytes go instead of
        // into the store.
        //
        // **IT IS A MAPPING AND NOT A `Vec`**, which is the whole of W-b's
        // interim step: an unlinked temporary file the kernel may page these
        // bytes back to, rather than anonymous memory whose only home under
        // pressure is swap ([`crate::host_source`] argues it, and states why
        // nothing ever binds a buffer over it). A plan that streams nothing
        // asks for zero bytes and creates no file at all, so a full-residency
        // load is untouched by this line.
        let mut host = HostSource::open(plan.source_bytes())?;
        let mut sink = Landing {
            store: &mut store,
            host: &mut host,
            plan,
            places: &places,
            index: &index,
            landed: vec![false; places.len()],
        };
        // ── **A STREAMED LOAD HAS NO ARENA AT ALL** (alto §K.3's Metal
        //    reading; the CUDA twin states the same branch over its own
        //    memory). The arena is where a transform's intermediates live
        //    while the schedule runs, and it is planned at the whole image's
        //    size: a resident load wants it, because its image IS the store's
        //    and every finalized tensor is copied out of it. A streamed
        //    load's image is not staged anywhere — each tensor goes to the
        //    device store or to the host band table as it finalizes — so the
        //    executor's own streaming residency owns every buffer and frees
        //    it at its last use, and what the load holds at once is one
        //    tensor's chain rather than the whole model.
        //
        //    **AND ON THIS PLANE THE ARENA IS NOT SOMEBODY ELSE'S MEMORY.**
        //    The CUDA arena is host staging beside a discrete card's own
        //    budget; here the store, the band table and the arena are all
        //    bytes of the ONE pool `Context::working_set` reports and
        //    `Accounting` admitted the weight budget against (`api.rs`). A
        //    load that streams because it did not fit was then allocating a
        //    second whole-image term on the very side it was shrunk to fit —
        //    which is the term this branch subtracts, and the reason it is
        //    worth a second compile here at all.
        //
        //    The plan is compiled a SECOND TIME for that, and that is the
        //    whole price: `compile_streaming` is this same contract against
        //    this same Metal target, minus the two passes that exist to serve
        //    an arena (it says there why the target cannot be weakened to
        //    `Unknown` instead). Milliseconds, against an image.
        //
        //    **AND THE SECOND PLAN IS NOT WHAT ANYTHING ELSE READS.** It is
        //    compiled, run, and dropped inside this branch. `landing` — the
        //    one the whole pipeline lowered — is what `pairings` reads below,
        //    for the same reason the CUDA side keeps it as its key: a plan
        //    that differed by how its caller intended to run it must never be
        //    the plan anything downstream identifies the load by.
        let landed = if plan.streams() {
            let streaming = compile_streaming(&metadata, contract, target)?;
            Execution::new(&streaming, snapshot)
                .streaming()
                .sink(&mut sink)
                .run()?;
            sink.landed
        } else {
            // The executor's own arena: where a transform's intermediates
            // live while it runs. Host memory, because the transforms run
            // host-side; it is dropped the moment the load is over, and only
            // the finalized tensors the sink took survive.
            let mut scratch = vec![0u8; usize::try_from(landing.memory.arena_bytes()).unwrap_or(0)];
            let mut backing: &mut [u8] = &mut scratch;
            Execution::new(&landing, snapshot)
                .arena(&mut backing)
                .sink(&mut sink)
                .run()?;
            let landed = sink.landed;
            drop(scratch);
            landed
        };

        let pairings = pairings(&landing, &index)?;

        let mut table = Vec::with_capacity(places.len());
        for (at, place) in places.iter().enumerate() {
            // A REGISTERED PLANE IS ONE THE CHECKPOINT DOES NOT HAVE, and
            // demanding it is exactly what would refuse every adapter-capable
            // load (design §8's open item, from the residency side). It is
            // already reserved — `places` sizes every param, whatever its
            // provenance — and already zeroed, and a zeroed low-rank `A`
            // makes the whole bank the identity until something registers
            // into it. What lands here is `register_adapter`'s business.
            if !landed[at] && trace.params[at].source == ParamSource::Checkpoint {
                return Err(Fault::Param {
                    name: trace.params[at].name.clone(),
                    why: "is a plan param the load contract never published",
                });
            }
            let dense = |place: &Place| -> Result<Tensor> {
                Ok(Tensor::new(
                    handles.bind(&store, place.offset, place.bytes)?,
                    place.rows,
                    place.width,
                    place.dtype,
                ))
            };
            table.push(Some(match pairings.get(trace.params[at].name.as_str()) {
                Some(pairing) => WeightRow::Planes(kernels_metal::Bank {
                    codes: dense(place)?,
                    scales: dense(&places[pairing.scales])?,
                    biases: pairing.biases.map(|at| dense(&places[at])).transpose()?,
                    group: pairing.group,
                    bits: pairing.bits,
                }),
                None => WeightRow::Dense(dense(place)?),
            }));
        }
        // **THE TIER IS OPENED AFTER THE LANDING AND BEFORE THE FIRST FIRE**,
        // which is the only instant where both halves exist: the host table is
        // full, the store's seats are reserved and still zeroed, and no
        // command buffer has been committed. `Tier::open` copies the identity
        // prefix in, so a load that never routes outside it never copies again.
        let tier = plan
            .streams()
            .then(|| {
                let offsets: Vec<u64> = places.iter().map(|place| place.offset).collect();
                Tier::open(plan, &store, host, &offsets).map(RefCell::new)
            })
            .transpose()?;
        Ok(Weights {
            store,
            table: WeightTable(table),
            tier,
            banks: banks(trace, &places),
        })
    }

    /// **The routed-expert tier this load opened, or `None` for a
    /// full-residency one.**
    ///
    /// The one question `serve` asks to decide whether a fire is walked in one
    /// command buffer or in segments — and the handle the encode sink takes
    /// its swap through.
    #[must_use]
    pub fn tier(&self) -> Option<&RefCell<Tier>> {
        self.tier.as_ref()
    }

    /// Write one adapter's planes into the banks (design §8).
    ///
    /// **A POOL WRITE AND A TABLE ROW, AND NOT A RECAPTURE** (decision 17).
    /// Nothing about the composition key is a function of a bank's contents —
    /// the key is the fire's composition — so this is a `memcpy` into a span
    /// that was reserved at load and will not move, and the handle rows that
    /// name that span do not change either. On this platform "not a transfer"
    /// is literal: the store is shared storage, and the write is a plain
    /// memcpy into the mapping the GPU reads. The per-lane id is what a fire
    /// says afterwards, and it says it in a submission.
    ///
    /// **RE-REGISTERING ZEROES THE SLOT FIRST**, because the planes are
    /// full-capacity and a caller that skipped one would otherwise leave the
    /// previous adapter's plane in place beside the new one's — an adapter
    /// that is half of each. A bank this call does not name keeps whatever it
    /// held, which is what makes a per-site registration expressible; naming
    /// every site is the caller's business and the refusal names any bank it
    /// invented.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a bank this plan does not declare, an id past
    /// the bank's capacity, or a plane whose bytes are not one slot's;
    /// [`Fault::Ceiling`] for a span past the store, [`Fault::Deviceless`]
    /// for a non-Apple build.
    pub fn register_adapter(&mut self, id: u32, planes: &[AdapterPlane<'_>]) -> Result<()> {
        // Checked whole before anything is written: a registration that
        // refuses halfway leaves a bank holding an adapter nobody described.
        for plane in planes {
            let bank = self.banks.get(plane.bank).ok_or_else(|| Fault::Adapter {
                bank: plane.bank.to_string(),
                why: "not a bank this plan declares; a bank is a weight the model text \
                      marked `registered`, and this plan marked none by that name"
                    .to_string(),
            })?;
            if id >= bank.adapters {
                return Err(Fault::Adapter {
                    bank: plane.bank.to_string(),
                    why: format!(
                        "seats {} adapters while this registration is id {id}; capacity \
                         is a shape the model text declared, so the fix is the model \
                         text and not a retry",
                        bank.adapters
                    ),
                });
            }
            if plane.bytes.len() as u64 != bank.slot {
                return Err(Fault::Adapter {
                    bank: plane.bank.to_string(),
                    why: format!(
                        "seats {} bytes per adapter while this plane carries {}; a plane \
                         is one whole slot, zero-padded by the caller past its own rank",
                        bank.slot,
                        plane.bytes.len()
                    ),
                });
            }
        }
        for plane in planes {
            let bank = self.banks[plane.bank];
            let at = bank.offset + u64::from(id) * bank.slot;
            self.store.zero_span(at, bank.slot)?;
            self.store.write(at, plane.bytes)?;
        }
        Ok(())
    }

    /// The banks this load declared: name, capacity, and bytes per slot.
    /// What a caller sizes its planes against, and what a gate asserts on.
    #[must_use]
    pub fn banks(&self) -> Vec<(&str, u32, u64)> {
        self.banks
            .iter()
            .map(|(name, bank)| (name.as_str(), bank.adapters, bank.slot))
            .collect()
    }

    /// **THE BANKS, AS THE ADAPTER RESOLVER READS THEM** — name, capacity,
    /// slot bytes, the slot's rectangle and its element size.
    ///
    /// [`Weights::banks`]'s longer twin, and a second method rather than a
    /// widened one because the two have different readers: a caller sizing a
    /// plane by hand wants three numbers, and [`crate::adapter::planes_of`]
    /// slicing a `[layers, ...]` seed needs the shape to check the out-major
    /// statute against (alto adapter §6.3).
    #[must_use]
    pub fn seats(&self) -> Vec<BankSeat> {
        self.banks
            .iter()
            .map(|(name, bank)| BankSeat {
                name: name.clone(),
                adapters: bank.adapters,
                slot: bank.slot,
                rows: bank.rows,
                cols: bank.cols,
                elem: bank.elem,
            })
            .collect()
    }

    /// **HOW MANY ADAPTERS THIS LOAD CAN HOLD RESIDENT AT ONCE** — the
    /// smallest capacity any declared bank states, and zero for a plan that
    /// declares none.
    ///
    /// The SMALLEST, because an adapter occupies one slot of every bank it
    /// fills: a load whose `A` seats eight and whose `B` seats four holds
    /// four adapters, and a fifth would have nowhere to put its `B`. Model
    /// texts declare the two alike today and this is the honest reading of a
    /// text that does not.
    #[must_use]
    pub fn adapter_seats(&self) -> u32 {
        self.banks
            .values()
            .map(|bank| bank.adapters)
            .min()
            .unwrap_or(0)
    }

    /// The table a fire resolves `Def::Weight(i)` through.
    #[must_use]
    pub fn table(&self) -> &WeightTable {
        &self.table
    }

    /// Every byte the store holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes()
    }
}

/// **A routed bank's other device planes, off the load plan** — what
/// [`experts::Plan::of`] needs before a byte is landed, and the reason it can
/// be asked for without one.
///
/// A quantized bank is codes plus factors plus (for an affine codec) zero
/// points, and all three are indexed by the number the routing vector carries
/// — so a residency decision that moved the codes alone would leave two
/// planes reading somebody else's expert. The pairing is the LOAD PLAN's, read
/// exactly as [`pairings`] reads it and never off a name; this door exists so
/// that the decision can be made BEFORE `Weights::resident` reserves the
/// store, which is what puts the admission gate in front of the landing rather
/// than behind it.
///
/// Costs one metadata parse and one plan compile, and reads no tensor bytes.
///
/// # Errors
///
/// [`Fault::Load`] for a checkpoint the contract does not fit,
/// [`Fault::Param`] for an attachment this plan cannot resolve.
///
/// [`experts::Plan::of`]: crate::experts::Plan::of
pub fn attachments(trace: &Trace, contract: &ModelContract, path: &Path) -> Result<Attachments> {
    // The admission gate runs in front of the landing, so it meets an
    // unreadable plane order first — see `readable_plane_orders`.
    readable_plane_orders(trace)?;
    let metadata = if path.is_dir() {
        parse_metadata(path)?
    } else {
        zt::parse(path)?
    };
    let landing = compile(
        &metadata,
        contract,
        StorageTarget::for_backend(BackendKind::Metal, 0, 1),
    )?;
    let index: BTreeMap<&str, usize> = trace
        .params
        .iter()
        .enumerate()
        .map(|(at, param)| (param.name.as_str(), at))
        .collect();
    let mut out = Attachments::new();
    for (name, pairing) in pairings(&landing, &index)? {
        let Some(&at) = index.get(name) else {
            continue;
        };
        let mut planes = vec![pairing.scales];
        planes.extend(pairing.biases);
        out.insert(at, planes);
    }
    Ok(out)
}

/// One quantized weight's other planes, as rows of [`places`].
#[derive(Debug, Clone, Copy)]
struct Pairing {
    scales: usize,
    /// The zero points, for an affine scheme; `None` for a symmetric one.
    biases: Option<usize>,
    group: u32,
    bits: u32,
}

/// Which params are quantized weights, and what their other planes are.
///
/// # The pairing is the PLAN's, and it is read structurally
///
/// Every scale plane in this tree is named `<weight>.scales` and every zero
/// point `<weight>.biases`, and none of that is what this function reads. A
/// suffix is a convention, and a convention is exactly what silently pairs a
/// weight with somebody else's scales the day a checkpoint ships a tensor
/// whose own name ends in `.scales`. `LoadPlan::attachments` is the pairing
/// STATED, by the two places that create a scale plane — the contract that
/// declares one the checkpoint shipped, and the encode that writes one the
/// loader computed — each recording it at the moment it knew both halves.
///
/// # The format is [`ScaleForm`], and the numbers are the attachment's
///
/// **ONE FIELD DECIDES, AND IT IS THE FIELD WHOSE WHOLE JOB IS TO DECIDE.**
/// `ScaleForm` is the loader's answer to "what does a kernel reading these
/// bytes get" — `RawE8M0` for exponent bytes that ARE the dequantization,
/// `Bf16AffineFactors` for factors that are half of one and imply the zero
/// points beside them. Its own doc records that an engine inferring this from
/// `group_size == 32` was right only by accident, because mxfp4 happened to
/// be the one scheme at that group. So the group size and the bit width are
/// read as NUMBERS here and never as evidence: the group off the attachment,
/// the width off the plan's per-tensor `QuantSpec` (`affine_point_of`), which
/// is the only reading that survives `mlx_lm`'s 4-bit stack with its 8-bit
/// router gate.
///
/// An attachment naming a tensor this trace has no param for is not an
/// error: a contract may compute internal tensors, and only the ones the
/// trace declares become weight rows.
fn pairings<'a>(
    landing: &'a LoadPlan,
    index: &BTreeMap<&str, usize>,
) -> Result<BTreeMap<&'a str, Pairing>> {
    // Keyed by the id's own number: `TensorId` is `Hash` and `Eq` and not
    // `Ord`, and a derive on the loader's type is not this shell's to add.
    let named: BTreeMap<u32, &str> = landing
        .tensors
        .iter()
        .map(|decl| (decl.id.0, decl.name.as_str()))
        .collect();
    let mut out = BTreeMap::new();
    for attachment in &landing.attachments {
        let Some(of) = named.get(&attachment.tensor.0) else {
            continue;
        };
        // Only a plane the trace declares becomes a weight row; a contract
        // may compute internal tensors, and an attachment on one of those is
        // a pairing about bytes no `Def::Weight` names.
        if !index.contains_key(of) {
            continue;
        }
        let name = of;
        let row = |id: TensorId, what: &'static str| -> Result<usize> {
            named
                .get(&id.0)
                .and_then(|plane| index.get(plane))
                .copied()
                .ok_or_else(|| Fault::Param {
                    name: (*name).to_string(),
                    why: what,
                })
        };
        let bits = match attachment.scale_form {
            // E2M1 is four bits by its own name, and the plan does not carry
            // a `QuantSpec` width for it to disagree with: `affine_point_of`
            // answers `None` for an mxfp4 tensor on purpose, because an
            // mxfp4 bank is not read at an affine point at all.
            ScaleForm::RawE8M0 => 4,
            ScaleForm::Bf16AffineFactors => match landing.affine_point_of(name) {
                Some((_, bits)) => bits,
                None => {
                    return Err(Fault::Param {
                        name: (*name).to_string(),
                        why: "carries affine scale factors and no quantized encoding for \
                              them to be factors OF; the point a kernel is selected at is \
                              the tensor's own `QuantSpec`, and this one has none",
                    });
                }
            },
            // The third form expands to f32 multipliers before a kernel sees
            // them, which is a plane this shell's points do not read: every
            // one of them takes its scales in the stored width.
            ScaleForm::F32Factors => {
                return Err(Fault::Param {
                    name: (*name).to_string(),
                    why: "wants its scales expanded to f32 factors, and every quantized \
                          point this shell stamps reads them in the width they are stored",
                });
            }
        };
        // **AN AFFINE BANK WITHOUT ITS ZERO POINTS IS REFUSED, NOT
        // DOWNGRADED.** `code * scale` alone is the right spread around the
        // wrong centre — a model that computes, produces no NaN, and is
        // wrong — so a form that says the groups are offset and an
        // attachment that names nothing to offset by is a hole in the
        // contract and is answered as one. A contract states the plane with
        // `TensorContract::offsetting`; an encode the loader performs
        // records the id itself.
        let biases = match (attachment.scale_form, attachment.zero_point_tensor) {
            (_, Some(id)) => Some(row(
                id,
                "is an affine bank whose zero points this plan does not publish as a \
                 param of their own",
            )?),
            (ScaleForm::Bf16AffineFactors, None) => {
                return Err(Fault::Param {
                    name: (*name).to_string(),
                    why: "is an affine bank whose scales are half of its dequantization, \
                          and this plan names no zero points for the other half; a \
                          contract states them with `TensorContract::offsetting`",
                });
            }
            (_, None) => None,
        };
        out.insert(
            *name,
            Pairing {
                scales: row(
                    attachment.scale_tensor,
                    "is a quantized weight whose scales this plan does not publish as a \
                     param of their own",
                )?,
                biases,
                group: attachment.group_size,
                bits,
            },
        );
    }
    Ok(out)
}

/// **A PLANE ORDER THIS SHELL HAS NO READER FOR IS REFUSED BY NAME**, before
/// a plan is compiled and before a byte is reserved.
///
/// `Dtype::U4g64tiled` is `U4g64`'s codes in m16n8k16 fragment order — the
/// same scheme, the same group of sixty-four, the same two bf16 companions,
/// and a different ORDER, written for `kernels_cuda::linear::tiled`. This
/// shell has no point that reads it: `pairings` seats every affine bank as
/// codes + scales + biases with a `group` and a `bits`, and
/// `kernels_metal::linear::quant`'s qmm/qmv arms index that rectangle
/// ROW-MAJOR. Handed a relaid plane they compute finite, deterministic
/// nonsense.
///
/// # What it was written for, and what closed that
///
/// It was written because a correct load PRODUCED one. `qwen_3::model`'s
/// projection flip declared `U4g64tiled` off the weight's width alone — no
/// platform, no backend — so all seven `*-mlxu4-*` qwen_3 rows asked this
/// shell for an order it does not read. A raw MLX snapshot refused at
/// `checkpoint::plan::passes::validate`'s `validate_target_support`
/// (`METAL_TILE_MAP_MASK` carries no `TILE_MAP_REPACK`, and a serving plan
/// does not convert), and a `pie model import` artifact — whose projections
/// the import HAD relaid, and which `qwen_3::import`'s `read_own` arm then
/// binds with no transform — sailed past that and landed here, loading in
/// 0.1s and answering the first-light prompt `"一时的وات**!.energy…"`.
///
/// **§J4c CLOSED IT AT THE DECLARATION, WHICH IS WHERE THE PARAGRAPH ABOVE
/// SAID IT BELONGED.** `model_dsl::place` resolves a placed dtype against
/// the platform the declaration is being read for (`Platform::placement`),
/// and BOTH readings of a family's text go through one — the trace through
/// `catalog!`, the load contract through `runtime::engine::load`. So a Metal
/// trace of those rows declares the canonical `U4g64` its qmm and qmv arms
/// already read, its contract states no repack, and the raw snapshot serves
/// as stored with no import at all. That is §M's own ruling in a dtype: a
/// `.zt` is a function of the RECIPE, so one text is two artifacts.
///
/// # What it is still for, and what it is NOT
///
/// **IT IS THE NET UNDER THE DECLARATION.** A text that reaches for a
/// placement `place` was not asked about — a literal `Dtype::U4g64tiled` in a
/// declaration, or a future placed variant whose `Platform::placement` row
/// nobody wrote — arrives here rather than at a wrong number. `models`'
/// `no_trace_declares_a_plane_its_platform_cannot_read` is the same fact
/// asserted over the whole catalog, one layer earlier and with no device in
/// the room; this is the one that runs on the load path.
///
/// **IT IS NOT THE NET UNDER THE ARTIFACT, AND CANNOT BE.** It reads
/// `Trace::params` — a declaration — so a `.zt` CONVERTED FOR CUDA and fed to
/// this shell sails past it: the trace is this shell's and says `U4g64`, and
/// the artifact says nothing to the contrary, because a repack moves no value
/// and `U4g64tiled` shares `U4g64`'s TERM (`dtype::Dtype::U4g64tiled`) while
/// a `.zt` records terms. Measured: the CUDA artifact of the 0.8B loads here
/// in 0.1s and answers `"productiveeldahar打造成…"`.
///
/// **AND THE OPEN ITEM IS CLOSED, ONE FUNCTION DOWN** (§M-4c). A guard on the
/// NAME was tried and does not hold: it read "a projection under the weight's
/// own name is a plane some repack produced", which was true while a promotion
/// moved only the repack and stopped being true at §M-4a, where a promotion
/// moves the whole landing and a row-major artifact holds its projections under
/// their own names too. What this paragraph said would settle it — *the
/// artifact SAYING which recipe it was written for* — is what
/// [`serves_this_deployment`] now asks, off the `pie.serving/1` stamp `pie
/// model import` writes, on the line after this check and before anything is
/// reserved. The division of labour stands as it was stated: this one is about
/// the TRACE and refuses a declaration no artifact could rescue; that one is
/// about the FILE and refuses a file no declaration can.
///
/// # Errors
///
/// [`Fault::Param`] naming the first param declared in an order this shell
/// cannot read.
pub(crate) fn readable_plane_orders(trace: &Trace) -> Result<()> {
    match trace
        .params
        .iter()
        .find(|param| param.dtype == Dtype::U4g64tiled)
    {
        None => Ok(()),
        Some(param) => Err(Fault::Param {
            name: param.name.clone(),
            why: "is declared U4g64tiled — MLX affine codes in m16n8k16 fragment order, which \
                  this shell has no reader for: its qmm and qmv arms index an affine bank \
                  row-major and would answer nonsense off a relaid plane. The order is \
                  `kernels_cuda::linear::tiled`'s, and a model text reaches it only by asking \
                  for it: `model_dsl::place` resolves a placed dtype against the platform the \
                  declaration is read for, and this platform's answer is the canonical \
                  row-major sibling. So either this plane came out of an artifact converted \
                  FOR the cuda shell — convert it again on this box, or serve it there — or a \
                  text stated the order outright, in which case it is the text that has to ask",
        }),
    }
}

/// **IS THIS ARTIFACT FOR THIS DEPLOYMENT?** — the `pie.serving/1` stamp gate,
/// asked before a byte of the machine is spent (§M-4c).
///
/// This is the net under the ARTIFACT that
/// [`readable_plane_orders`] says it cannot be. That one reads
/// `Trace::params` — a DECLARATION — so a `.zt` converted for the CUDA shell
/// sails past it: the trace is this shell's and says `U4g64`, the artifact
/// says nothing to the contrary because a repack moves no value and shares the
/// canonical term, and the measured result is a 0.1s load that answers
/// nonsense. What settles it is the file SAYING which recipe it was written
/// for, and `pie model import` writes exactly that down.
///
/// # Two absences, two meanings — the cut both shells implement
///
/// **A raw snapshot or an ordinary `.zt` has no stamp and PROCEEDS.** This is
/// not a hole left open: a checkpoint that carries no `pie.serving/1` block is
/// not claiming to be anybody's serving artifact, and every load this tree has
/// ever run — every device gate, every four-bit first light — is one. Refusing
/// them would be refusing the format that predates the profile.
///
/// **A stamped artifact is CHECKED, field by field**, and the refusal is
/// `serving::Mismatch::refuse`'s sentence rather than one written here: what
/// disagreed, that nothing on this path rewrote or deleted the file, and the
/// `pie model import --force` that writes it again. `Stamp::check` compares
/// the version, the backend, the tp degree, the SKU, the precision, the layout
/// revision, the block size, the block algorithm and the zeroed-adapters
/// assertion, and stops at the FIRST one that differs — which is what makes
/// the sentence name a fact an operator can act on instead of saying only
/// *different*.
///
/// **A REQUEST THAT STATES NO PRECISION IS REFUSED, and it is refused BEFORE
/// the file is looked at.** `runtime::engine::load::request_of` fills that
/// field unconditionally off the catalog, so an empty one is a runtime that
/// could not assemble the comparison — and the tempting reading, "nothing to
/// compare against, so no check", is precisely the silent cross-recipe landing
/// this function exists to prevent. Asked first rather than after the read
/// because it is a fact about the LOAD and not about the file: it is true of an
/// ordinary checkpoint too, and a shell that only noticed when the artifact
/// happened to be stamped would report a caller's bug at random.
///
/// # What it costs, and what it is not
///
/// One positioned manifest read (`serve::stamp_of`, which opens no mapping)
/// for a single-container load, and nothing at all for a snapshot DIRECTORY: a
/// serving artifact is one file by construction, so a directory is answered
/// without touching the disk.
///
/// **It hashes nothing.** The blocks are the verify doors' business
/// (`Artifact::verify_all`); this asks whether the recipe matches, which is a
/// question about nine text fields.
///
/// **And it is asserted rather than assumed.** The CUDA twin makes the same
/// claim structurally — positioned reads, and the call's position in the file;
/// `a_cross_recipe_artifact_refuses_before_it_allocates` measures it, around
/// this call, off `device::reservations` and `host_source::descriptors`.
///
/// # The claim and the reading are separate questions, and both are asked
///
/// [`serve::stamp_of`] answers in the shape that has three answers:
/// `Ok(None)` is a container with no `pie.serving` key — an ordinary
/// checkpoint, the ONLY outcome that proceeds. `Ok(Some(_))` is a stamp that
/// read back, and is checked. `Err(_)` is a file that CLAIMS a serving
/// profile (`serving::stated_profile`) but whose stamp does not read back — a
/// rotted member, a future `pie.serving/<n>`, a manifest-less container — and
/// is refused, because landing it as the ordinary checkpoint it also is would
/// be a stamped artifact losing its stamp to decay and passing the very check
/// the stamp exists to feed. (This split was the recorded ask of the sibling
/// lane; `eaff5950d` answered it with the public predicate and this door.)
///
/// # Errors
///
/// [`Fault::Recipe`] for a stamp that disagrees, for a serving version this
/// build does not read, and for a load that states no precision.
pub(crate) fn serves_this_deployment(
    path: &Path,
    backend: &str,
    sku: &str,
    tp_size: u64,
    precision: &str,
) -> Result<()> {
    if precision.is_empty() {
        return Err(Fault::Recipe(format!(
            "internal: this load states no precision, so nothing here can check that {} \
             was written for it. `runtime::engine::load::request_of` fills that field \
             from the catalog and a request without one did not come through it; \
             landing the planes unchecked is the cross-recipe boot the `{}` stamp \
             exists to prevent, so this is loud rather than skipped",
            path.display(),
            serving::PROFILE,
        )));
    }
    // A snapshot directory is a set of source files and not one container, so
    // it holds no file-level attributes and can carry no stamp. Asked before
    // the door is opened rather than through it, because `read_head` on a
    // directory would refuse for a reason that has nothing to do with serving.
    if path.is_dir() {
        return Ok(());
    }
    let artifact = match serve::stamp_of(path) {
        // No `pie.serving/1` key at all: an ordinary checkpoint, which is
        // every load this tree ran before the profile existed — the ONLY
        // outcome that proceeds. See the header.
        Ok(None) => return Ok(()),
        Ok(Some(stamp)) => stamp,
        // The file CLAIMS a serving profile and its stamp does not read back
        // (a rotted member, a future profile, no manifest at all). Serving it
        // as the ordinary checkpoint it also is would be the quiet cousin of
        // the cross-recipe boot: a stamped artifact losing its stamp to decay
        // and passing the very check the stamp exists to feed.
        Err(why) => return Err(Fault::Recipe(format!("checkpoint: {why}"))),
    };
    // The four policy fields are NOT spelled here. `Stamp::of` fills them from
    // `serving::LAYOUT_REVISION` and `serving::BLOCK_BYTES`, which is the
    // whole reason the constructor exists: a constant spelled twice is a field
    // that can disagree with itself, and `check` compares field by field.
    let deployment = Stamp::of(backend, tp_size, sku, precision, None);
    artifact.check(&deployment).map_err(|mismatch| {
        // `refuse` takes the artifact's name and the SOURCE the refusing load
        // should be re-imported from. The artifact's own `model_id` is that
        // source when it carries one — it is what the operator named at import
        // — and the command is left as a slot when it does not, rather than
        // printing this artifact's own path as the thing to convert.
        Fault::Recipe(mismatch.refuse(
            &path.display().to_string(),
            artifact.model_id.as_deref(),
        ))
    })
}

/// Where one param's plane sits in the store.
#[derive(Debug, Clone, Copy)]
struct Place {
    offset: u64,
    /// What the store RESERVES for it — the plane's own bytes for a param
    /// held whole, and `slots x stride` for a band the residency plan
    /// streams.
    bytes: u64,
    /// Those bytes, rounded up to the next view alignment.
    reserved: u64,
    /// **What the plane DECLARES**, which is what the checkpoint publishes
    /// and what the landing sink checks an arriving tensor against. Equal to
    /// [`bytes`](Place::bytes) for every param of a full-residency load; the
    /// whole bank for a streamed band, whose bytes go to the host table
    /// instead of here.
    full: u64,
    /// Whether the residency plan streams this param — the one bit that sends
    /// its bytes to the host band table rather than into the store.
    streamed: bool,
    rows: u32,
    width: u32,
    dtype: Dtype,
}

/// **What every param's plane DECLARES**, in bytes, `Trace::params`-indexed —
/// the arithmetic [`places`] lays out and [`experts::Plan`] budgets against,
/// stated once so that the two cannot disagree about how big a bank is.
///
/// **THE TWO PACKED PLANES ARE DECLARED IN TWO DIFFERENT UNITS, AND
/// `elem_bytes` CAN ANSWER FOR NEITHER.** `model_dsl::Weight::planes` gives an
/// mxfp4 bank the rectangle it OCCUPIES — each 32-code block folded into a
/// trailing axis of sixteen, which is the block's bytes — and gives an MLX
/// affine bank its LOGICAL rectangle, four bits an element, because that is
/// the shape the qmm and qmv points index it by. So neither is
/// `elements x element size`, and taking `elem_bytes`'s honest `None` for the
/// four-bit code as a refusal is what refuses every quantized SKU at its first
/// param.
///
/// # Errors
///
/// [`Fault::Param`] for a param declared in a storage element that has no
/// byte size.
///
/// [`experts::Plan`]: crate::experts::Plan
pub(crate) fn plane_bytes(trace: &Trace) -> Result<Vec<u64>> {
    trace
        .params
        .iter()
        .map(|param| {
            let (rows, width) = rectangle(&param.shape);
            Ok(match param.dtype {
                Dtype::Mxfp4 => rows.saturating_mul(width),
                // A tiled affine plane is the same nibbles relaid; the band
                // padding rides in the shape, so it is this arm's number.
                // This shell has no reader for the order and refuses one by
                // name at the two load doors (`readable_plane_orders`), so
                // nothing reaches here declaring it — the arm stays because
                // sizing is a fact about the rectangle and not about who can
                // read it, and because `experts::Plan` budgets off this
                // function without going through a door.
                Dtype::U4g64 | Dtype::U4g32 | Dtype::U4g64tiled => {
                    rows.saturating_mul(width).div_ceil(2)
                }
                // Two bits a code, sixteen to the `u32` word MLX packs them
                // into — so a row of `width` codes is `width / 4` bytes, and
                // the `div_ceil` is the four-bit arm's own argument one notch
                // narrower: a row whose width is not a multiple of four would
                // round up into the next, and none of the three groups (32,
                // 64, 128) admits such a row in the first place. The group
                // does not appear here; it sizes the COMPANION planes, which
                // are their own params with their own rectangles.
                Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128 => {
                    rows.saturating_mul(width).div_ceil(4)
                }
                // The same logical rectangle at a whole byte a code — see
                // `dtype::Dtype::U8g64`, and note that the `div_ceil` above is
                // the one thing that does NOT carry over: an eight-bit code
                // owns its byte, so no row can round up into the next.
                Dtype::U8g64 => rows.saturating_mul(width),
                other => {
                    let element =
                        model_compiler::arena::elem_bytes(other).ok_or_else(|| Fault::Param {
                            name: param.name.clone(),
                            why: "is declared in a packed storage element that has no element \
                                  size",
                        })?;
                    rows.saturating_mul(width).saturating_mul(element)
                }
            })
        })
        .collect()
}

/// The store's layout, decided before a byte is read.
///
/// STATED AHEAD, NOT ACCUMULATED, so that the length the checkpoint publishes
/// meets a length the plan predicted. A sink that allocated as it went would
/// take whatever arrived; this one refuses a plane that is not the size its
/// own declaration says it is, which is the only cheap check there is that a
/// contract and a plan describe the same model.
///
/// **AND `plan` IS THE ONE MAP THAT MAKES A BANK SMALLER THAN ITS
/// DECLARATION.** A param the residency plan streams reserves `slots` expert
/// seats instead of `experts` of them — the leading axis is the only axis
/// that shrinks, because a seat is one whole rectangle at intra-seat offset
/// zero (`experts::Plan::of` proves that before it plans anything). Its
/// declared size is kept in `full`, because that is still what the checkpoint
/// publishes and what the landing sink checks.
fn places(trace: &Trace, plan: &Plan) -> Result<Vec<Place>> {
    let bytes = plane_bytes(trace)?;
    let mut out = Vec::with_capacity(trace.params.len());
    let mut at = 0u64;
    for (index, param) in trace.params.iter().enumerate() {
        let (rows, width) = rectangle(&param.shape);
        let full = bytes[index];
        let (reserve, rows) = match plan.resident(index) {
            Some(slots) if rows > 0 => (full / rows * u64::from(slots), u64::from(slots)),
            _ => (full, rows),
        };
        out.push(Place {
            offset: at,
            bytes: reserve,
            reserved: reserve.next_multiple_of(ALIGN),
            full,
            streamed: plan.resident(index).is_some(),
            rows: u32::try_from(rows).unwrap_or(u32::MAX),
            width: u32::try_from(width).unwrap_or(u32::MAX),
            dtype: param.dtype,
        });
        at += reserve.next_multiple_of(ALIGN);
    }
    Ok(out)
}

/// A declared shape, read as `rows x width` — the IR's own rule, and the one
/// the arena carve reads too.
///
/// The numbers are honest rather than load-bearing: `linear.matmul` takes its
/// `m`, `n` and `k` from the ACTIVATION and the RESULT, never from the
/// weight, and the norm entries read a weight's handle alone. So a handle
/// here says what the plan declared, which is what a `{:?}` in a panic should
/// print, and nothing reads it back as a promise.
fn rectangle(shape: &[u64]) -> (u64, u64) {
    match shape.split_first() {
        Some((rows, rest)) => (*rows, rest.iter().product()),
        None => (1, 1),
    }
}

/// The sink that puts each finalized tensor where the layout said it goes —
/// the store for a param held whole, the host band table for one the
/// residency plan streams.
///
/// **THE DIVERT IS A BRANCH ON THE PLAN AND NOT A SECOND SINK.** Every
/// transform already runs host-side here (this module's header argues why), so
/// a streamed band arrives at exactly the same instant, in exactly the same
/// finalized bytes, as a landed one — the only question is which mapping they
/// are copied into. Answering it here keeps `Execution` unaware that residency
/// exists, which is what lets the loader stay backend-neutral.
struct Landing<'a> {
    store: &'a mut Buffer,
    /// Every expert of every streamed band, at the offsets
    /// [`Plan::host_at`](crate::experts::Plan::host_at) states.
    host: &'a mut [u8],
    plan: &'a Plan,
    places: &'a [Place],
    index: &'a BTreeMap<&'a str, usize>,
    landed: Vec<bool>,
}

impl TensorSink for Landing<'_> {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> std::result::Result<(), LoadError> {
        let at = *self.index.get(name).ok_or_else(|| {
            LoadError::Contract(format!(
                "the load contract publishes `{name}`, which this plan does not \
                 name — the two were not written from each other"
            ))
        })?;
        let place = self.places[at];
        // The DECLARED size, not the reserved one: a streamed band reserves
        // `slots` seats and the checkpoint still publishes every expert.
        if bytes.len() as u64 != place.full {
            return Err(LoadError::Contract(format!(
                "`{name}` lands {} bytes and the plan declares {} — a plane read \
                 at the wrong width is a model that computes",
                bytes.len(),
                place.full
            )));
        }
        if place.streamed {
            let from = usize::try_from(self.plan.host_at(at).unwrap_or(0)).unwrap_or(usize::MAX);
            let into = self
                .host
                .get_mut(from..from + bytes.len())
                .ok_or_else(|| {
                    LoadError::Internal(format!(
                        "`{name}` is a streamed band whose {} bytes leave the host band \
                         table at offset {from}",
                        bytes.len()
                    ))
                })?;
            into.copy_from_slice(bytes);
        } else {
            self.store
                .write(place.offset, bytes)
                .map_err(|fault| LoadError::Internal(fault.to_string()))?;
        }
        self.landed[at] = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use model_ir::Platform;

    use super::*;

    #[test]
    fn the_store_is_laid_out_aligned_disjoint_and_in_plan_order() {
        let trace =
            models::trace_of("qwen35-d0.8b-bf16-kv-bf16").expect("the catalog ships the SKU");
        let trace = trace(Platform::Metal);
        let places = places(&trace, &Plan::default())
            .expect("every param of a bf16 SKU has an element size");

        assert_eq!(places.len(), trace.params.len());
        let mut end = 0u64;
        for (place, param) in places.iter().zip(&trace.params) {
            assert!(
                place.offset >= end,
                "`{}` overlaps its predecessor",
                param.name
            );
            assert_eq!(place.offset % ALIGN, 0, "`{}` is misaligned", param.name);
            assert!(place.bytes > 0, "`{}` reserves nothing", param.name);
            end = place.offset + place.reserved;
        }

        // The embedding is the SKU's largest plane and its first: 248320
        // rows of 1024 bf16, and the head is tied to it, so it is landed once.
        assert_eq!(places[0].offset, 0);
        assert_eq!(places[0].rows, 248_320);
        assert_eq!(places[0].width, 1024);
        assert_eq!(places[0].bytes, 248_320 * 1024 * 2);
    }
}
