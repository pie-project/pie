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
//! host `Vec<u8>` holding those bands whole, which the landing sink fills
//! instead of the store. The weight rows are minted exactly as they always
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
//! that reached for `model::qwen_3` would be a shell that has to grow an arm
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
use checkpoint::file::zt;
use checkpoint::contract::ModelContract;
use checkpoint::error::Error as LoadError;
use checkpoint::executor::{Execution, sink::TensorSink};
use checkpoint::plan::{LoadPlan, StorageTarget, compile};
use checkpoint::types::{BackendKind, ScaleForm, TensorId};
use kernels_metal::Tensor;
use model_ir::{Dtype, ParamSource, Trace};

use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use crate::experts::{Attachments, Plan, Tier};
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
            (
                param.name.clone(),
                Bank {
                    offset: place.offset,
                    adapters,
                    slot,
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
    /// # Errors
    ///
    /// [`Fault::Load`] for a checkpoint the contract does not fit,
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
    ) -> Result<Weights> {
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
        let landing = compile(&metadata, contract, target)?;

        let places = places(trace, plan)?;
        let total = places.last().map_or(0, |p| p.offset + p.reserved);
        let mut store = Buffer::zeroed(device, total)?;

        // The executor's own arena: where a transform's intermediates live
        // while it runs. Host memory, because the transforms run host-side;
        // it is dropped the moment the load is over, and only the finalized
        // tensors the sink took survive.
        let mut scratch = vec![0u8; usize::try_from(landing.memory.arena_bytes()).unwrap_or(0)];
        let mut backing: &mut [u8] = &mut scratch;

        let index: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();
        // The host band table, sized off the plan and empty for a
        // full-residency load — where a streamed band's bytes go instead of
        // into the store.
        let mut host =
            vec![0u8; usize::try_from(plan.source_bytes()).unwrap_or(0)];
        let mut sink = Landing {
            store: &mut store,
            host: &mut host,
            plan,
            places: &places,
            index: &index,
            landed: vec![false; places.len()],
        };
        Execution::new(&landing, snapshot)
            .arena(&mut backing)
            .sink(&mut sink)
            .run()?;

        let landed = sink.landed;
        drop(scratch);

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
                Dtype::MlxU4 => rows.saturating_mul(width).div_ceil(2),
                // The same logical rectangle at a whole byte a code — see
                // `dtype::Dtype::MlxU8`, and note that the `div_ceil` above is
                // the one thing that does NOT carry over: an eight-bit code
                // owns its byte, so no row can round up into the next.
                Dtype::MlxU8 => rows.saturating_mul(width),
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
            model::trace_of("qwen35-d0.8b-bf16-kv-bf16").expect("the catalog ships the SKU");
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
