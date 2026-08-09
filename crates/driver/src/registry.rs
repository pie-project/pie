//! Programs, channels and instances, and the rules that decide whether an
//! instance may attach to a channel.
//!
//! # What this is for
//!
//! A launch program is registered once and run many times. Each run is an
//! *instance*, and an instance names the channels it will read and write. The
//! program declares what it expects of each of those channels -- dtype, shape,
//! capacity, who on the host may touch it, whether it is imported or exported
//! -- and the channel was registered separately, by someone else, with its own
//! idea of all five. This module is where those two ideas are compared, and
//! refusing a mismatch here is the only thing standing between a shape
//! disagreement and a kernel reading a ring with the wrong stride.
//!
//! # Why each rejection is its own function
//!
//! The C++ makes the whole comparison one boolean expression: eight clauses
//! OR-ed together, and when it is true the caller gets a single message that
//! prints every field of both sides and leaves the reader to spot which pair
//! disagrees. That is what a condition that cannot name itself forces. Here
//! each rule is a named check that returns its own reason, so a refusal says
//! `capacity` or `extern direction` rather than handing over a dump.
//!
//! # Attachment, and the rule that is not "one instance per channel"
//!
//! A channel outlives the instance that created it. A prefill pass hands its
//! channel to the decode pass that follows, so a second attachment to a
//! private channel is normal rather than an error -- provided nobody on the
//! host is reading or writing it and it carries no seed. Those two conditions
//! are what make an instance's use of it invisible to anyone else, which is
//! what makes handing it on safe. An imported channel may have one importer
//! and an exported one may have one exporter, because two writers to one ring
//! is a race and two readers each see half the cells.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::sync::Arc;

use driver_api::local::{
    PIE_CHANNEL_DTYPE_ACT, PIE_CHANNEL_DTYPE_BOOL, PIE_CHANNEL_DTYPE_F32, PIE_CHANNEL_DTYPE_I32,
    PIE_CHANNEL_DTYPE_U32, PIE_CHANNEL_EXTERN_EXPORT, PIE_CHANNEL_EXTERN_IMPORT,
    PIE_CHANNEL_EXTERN_NONE, PIE_CHANNEL_HOST_READER, PIE_CHANNEL_HOST_ROLE_NONE,
    PIE_CHANNEL_HOST_ROLE_READER, PIE_CHANNEL_HOST_ROLE_WRITER, PIE_CHANNEL_HOST_VISIBLE,
    PIE_CHANNEL_SEEDED, PIE_GEOMETRY_CLASS_DECODE_ENVELOPE, PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY,
    PIE_GEOMETRY_CLASS_HOST,
};
use driver_api::plan::{LaunchChannel, LaunchPackage};

use super::channel::{ChannelState, InterpInstance, make_host_channel_state, make_instance};
use super::plan::{ExecPlan, adopt_launch_package};
use super::value::{Value, concrete_dtype, decode_wire, wire_cell_bytes};
use crate::{Error, Result};

/// How a program's geometry is decided.
///
/// Kept as an enum rather than the wire's `u32` because the C++ gate accepted
/// only `Host` for long after the device resolver existed, which refused every
/// device-resolved decode at bind time. A `u32` compared against three
/// constants is a rule that can silently fall behind the constants; a match
/// that must be exhaustive cannot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Geometry {
    /// The host computes the shapes and sends them.
    Host,
    /// A decode envelope: the shapes are bounded and the device picks within.
    DecodeEnvelope,
    /// The device resolves the shapes from descriptors.
    Device,
}

impl Geometry {
    /// The wire value, or an error naming what arrived.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] if `class` is not one of the three.
    pub fn from_wire(class: u32) -> Result<Self> {
        match class {
            PIE_GEOMETRY_CLASS_HOST => Ok(Self::Host),
            PIE_GEOMETRY_CLASS_DECODE_ENVELOPE => Ok(Self::DecodeEnvelope),
            PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY => Ok(Self::Device),
            other => Err(Error::Program {
                message: format!("geometry class {other} is not one this driver binds"),
            }),
        }
    }

    /// Back to the wire value.
    #[must_use]
    pub const fn to_wire(self) -> u32 {
        match self {
            Self::Host => PIE_GEOMETRY_CLASS_HOST,
            Self::DecodeEnvelope => PIE_GEOMETRY_CLASS_DECODE_ENVELOPE,
            Self::Device => PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY,
        }
    }
}

/// Which side of an external boundary a channel sits on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Direction {
    /// Not external: private to the instances that attach to it.
    Private,
    /// Values arrive from outside.
    Import,
    /// Values leave for outside.
    Export,
}

impl Direction {
    /// What a program's channel declaration says.
    #[must_use]
    pub const fn of(channel: &LaunchChannel) -> Self {
        match channel.extern_dir {
            0 => Self::Import,
            1 => Self::Export,
            _ => Self::Private,
        }
    }

    /// The direction a `PIE_CHANNEL_EXTERN_*` byte names.
    ///
    /// The inverse of [`Direction::to_wire`], for a caller that receives a
    /// registration over the ABI rather than reading a program's declaration.
    /// Anything that is not import or export is private, which is the same
    /// reading [`Direction::of`] gives an unrecognised `extern_dir`.
    #[must_use]
    pub fn from_wire(dir: u8) -> Self {
        match dir {
            PIE_CHANNEL_EXTERN_IMPORT => Self::Import,
            PIE_CHANNEL_EXTERN_EXPORT => Self::Export,
            _ => Self::Private,
        }
    }

    /// The endpoint's `PIE_CHANNEL_EXTERN_*` byte.
    #[must_use]
    pub const fn to_wire(self) -> u8 {
        match self {
            Self::Private => PIE_CHANNEL_EXTERN_NONE,
            Self::Import => PIE_CHANNEL_EXTERN_IMPORT,
            Self::Export => PIE_CHANNEL_EXTERN_EXPORT,
        }
    }
}

/// Who on the host may touch a channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HostRole {
    /// Nobody: the channel is device-only.
    None,
    /// The host writes into it.
    Writer,
    /// The host reads out of it.
    Reader,
}

impl HostRole {
    /// What a program's channel declaration says.
    #[must_use]
    pub const fn of(channel: &LaunchChannel) -> Self {
        if channel.flags & PIE_CHANNEL_HOST_VISIBLE == 0 {
            Self::None
        } else if channel.flags & PIE_CHANNEL_HOST_READER == 0 {
            Self::Writer
        } else {
            Self::Reader
        }
    }

    /// The role a `PIE_CHANNEL_HOST_ROLE_*` byte names.
    ///
    /// The inverse of [`HostRole::to_wire`]. An unrecognised byte reads as
    /// `None`, which is the conservative answer: a channel nobody on the host
    /// may touch is refused a host operation rather than granted one.
    #[must_use]
    pub fn from_wire(role: u8) -> Self {
        match role {
            PIE_CHANNEL_HOST_ROLE_WRITER => Self::Writer,
            PIE_CHANNEL_HOST_ROLE_READER => Self::Reader,
            _ => Self::None,
        }
    }

    /// The endpoint's `PIE_CHANNEL_HOST_ROLE_*` byte.
    #[must_use]
    pub const fn to_wire(self) -> u8 {
        match self {
            Self::None => PIE_CHANNEL_HOST_ROLE_NONE,
            Self::Writer => PIE_CHANNEL_HOST_ROLE_WRITER,
            Self::Reader => PIE_CHANNEL_HOST_ROLE_READER,
        }
    }
}

/// The `PIE_CHANNEL_DTYPE_*` byte for a program's declared dtype.
#[must_use]
pub const fn channel_dtype(dtype: u8) -> u8 {
    // The interpreter's dtype byte and the channel wire's are different
    // encodings of the same set; folding them by name rather than by value
    // is what stops one gaining a member and the other silently mapping it
    // to F32.
    match dtype {
        1 => PIE_CHANNEL_DTYPE_I32,
        2 => PIE_CHANNEL_DTYPE_U32,
        3 => PIE_CHANNEL_DTYPE_BOOL,
        4 => PIE_CHANNEL_DTYPE_ACT,
        _ => PIE_CHANNEL_DTYPE_F32,
    }
}

/// One host-emitted kernel, or the reason the host chose not to emit it.
///
/// [`driver_api::plan::EmittedKernel`], re-exported rather than restated. The
/// port declared its own field-identical copy of this struct — six fields, the
/// same six names — and the two were never reconciled because nothing in the
/// Metal shell called both halves: `Registry::register_program` took the copy
/// and [`Emitted::index`](crate::Emitted::index) took the ABI type, so the
/// duplication was invisible until a second shell tried to hand a registered
/// program's kernels to the compiler and found the two types nominally
/// distinct.
///
/// The ABI type is the right survivor. It is the one that arrives over the
/// wire, the one the emitted-table index already reads, and the one whose
/// `error`-with-no-source form is documented where a driver author will look
/// for it. The empty-source-with-an-error case is deliberate and is not a
/// failure to register: it is the host saying "I could not emit this one", and
/// the runtime takes its fallback path.
pub use driver_api::plan::EmittedKernel;

/// A registered program: its plan, its channel declarations, its kernels.
#[derive(Debug)]
pub struct Program {
    /// The id this registry assigned.
    pub id: u64,
    /// The hash the caller supplied, and the key programs are deduplicated on.
    pub hash: u64,
    /// What the program expects of each channel it binds, in slot order.
    pub channels: Vec<LaunchChannel>,
    /// The adopted execution plan.
    pub plan: ExecPlan,
    /// What the host emitted for it.
    pub kernels: Vec<EmittedKernel>,
}

/// What a caller says about a channel it is registering.
///
/// A struct rather than nine positional arguments, four of which are a `u32`,
/// a `bool`, and two enums that a caller could put in any order and the
/// compiler would accept several of the wrong ones.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelSpec {
    /// The id the caller chooses.
    pub id: u64,
    /// The `PIE_CHANNEL_DTYPE_*` byte.
    pub dtype: u8,
    /// The cell shape.
    pub shape: Vec<u32>,
    /// How many cells the ring holds.
    pub capacity: u32,
    /// Who on the host may touch it.
    pub role: HostRole,
    /// Whether the program that declares it supplies a seed.
    pub seeded: bool,
    /// Which side of an external boundary it is on.
    pub direction: Direction,
    /// The name it is imported or exported under.
    pub extern_name: Vec<u8>,
}

/// A registered channel endpoint and the ring behind it.
#[derive(Debug)]
pub struct Channel {
    /// The id the caller chose.
    pub id: u64,
    /// The `PIE_CHANNEL_DTYPE_*` byte.
    pub dtype: u8,
    /// The cell shape.
    pub shape: Vec<u32>,
    /// How many cells the ring holds.
    pub capacity: u32,
    /// Who on the host may touch it.
    pub role: HostRole,
    /// Whether the program that declares it supplies a seed.
    pub seeded: bool,
    /// Which side of an external boundary it is on.
    pub direction: Direction,
    /// The name it is imported or exported under.
    pub extern_name: Vec<u8>,
    /// Instances currently attached, and the direction each attached in.
    pub attachments: BTreeMap<u64, Direction>,
    state: Arc<ChannelState>,
}

impl Channel {
    /// The lane count of one cell.
    #[must_use]
    pub fn numel(&self) -> u64 {
        super::shape_numel(&self.shape)
    }

    /// The ring itself.
    #[must_use]
    pub fn state(&self) -> &Arc<ChannelState> {
        &self.state
    }
}

/// What a caller needs to reach a channel's ring without going through here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endpoint {
    /// The channel this describes.
    pub channel_id: u64,
    /// Bytes in one cell.
    pub cell_bytes: u32,
    /// Cells in the ring.
    pub capacity: u32,
    /// Total bytes of cell storage.
    pub mirror_bytes: usize,
    /// Total bytes of the control words.
    pub word_bytes: usize,
    /// Address of the cell storage, for the ABI's endpoint binding.
    pub mirror_base: u64,
    /// Address of the four control words, in the ABI's order: head, tail,
    /// poison, closed.
    pub word_base: u64,
}

/// A bound instance: one run of a program over a named set of channels.
#[derive(Debug)]
pub struct Instance {
    /// The id this registry assigned, or the one the caller requested.
    pub id: u64,
    /// The program it runs.
    pub program_id: u64,
    /// That program's hash, copied so a stale instance can be spotted.
    pub program_hash: u64,
    /// How its geometry is decided.
    pub geometry: Geometry,
    /// The channels it attached to, in program slot order.
    pub channel_ids: Vec<u64>,
    /// How many fires it has taken.
    pub fire_seq: u64,
    /// The interpreter state.
    pub interp: InterpInstance,
}

/// Programs, channels and instances, and the rules between them.
///
/// `Default` is written rather than derived, and the difference is not
/// cosmetic: both counters start at ONE, because zero is the ABI's "none" for
/// a program id and an instance id alike -- `validate_instance_binding`
/// refuses a binding whose `instance_id` is zero, and the engine's
/// `InstanceBindingPlan` spells "any instance" as a requested id of zero. A
/// derived `Default` handed out zero as the first instance, and a registry
/// built that way refused its own first bind.
#[derive(Debug)]
pub struct Registry {
    next_program: u64,
    next_instance: u64,
    programs: BTreeMap<u64, Program>,
    program_by_hash: BTreeMap<u64, u64>,
    instances: BTreeMap<u64, Instance>,
    channels: BTreeMap<u64, Channel>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_program: 1,
            next_instance: 1,
            programs: BTreeMap::new(),
            program_by_hash: BTreeMap::new(),
            instances: BTreeMap::new(),
            channels: BTreeMap::new(),
        }
    }

    /// Register a program, or return the id of one already registered under
    /// the same hash.
    ///
    /// Deduplication is by hash and the package is not re-compared. That is
    /// the caller's contract -- the hash is over the package -- and it is
    /// stated here because the consequence of breaking it is that two
    /// different programs share one plan and neither reports anything.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] if the package has no stages or cannot be adopted.
    pub fn register_program(
        &mut self,
        hash: u64,
        package: LaunchPackage,
        kernels: Vec<EmittedKernel>,
    ) -> Result<u64> {
        if let Some(&existing) = self.program_by_hash.get(&hash) {
            return Ok(existing);
        }
        let channels = package.channels.clone();
        let plan = adopt_launch_package(package)?;
        let id = self.next_program;
        self.next_program += 1;
        self.programs.insert(
            id,
            Program {
                id,
                hash,
                channels,
                plan,
                kernels,
            },
        );
        self.program_by_hash.insert(hash, id);
        Ok(id)
    }

    /// Register a channel endpoint and allocate its ring.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] if `id` is taken, if any extent is zero, or if the
    /// cell size does not fit the `u32` the endpoint reports it in.
    pub fn register_channel(&mut self, spec: ChannelSpec) -> Result<Endpoint> {
        let ChannelSpec {
            id,
            dtype,
            shape,
            capacity,
            role,
            seeded,
            direction,
            extern_name,
        } = spec;
        if self.channels.contains_key(&id) {
            return Err(Error::Program {
                message: format!("channel {id} is already registered"),
            });
        }
        // A zero extent is refused rather than accepted as an empty cell: the
        // ring would have a stride of zero, every push would write over the
        // last, and nothing downstream checks for it.
        if let Some(position) = shape.iter().position(|&d| d == 0) {
            return Err(Error::Program {
                message: format!("channel {id}: extent {position} of the cell shape is zero"),
            });
        }
        let numel = super::shape_numel(&shape);
        // The endpoint reports the cell size as a u32. A shape whose byte
        // count passes that would hand the caller a stride the ring does not
        // use, so it is refused here rather than truncated.
        let lanes = usize::try_from(numel).unwrap_or(usize::MAX);
        let bytes =
            u64::try_from(wire_cell_bytes(concrete_dtype(dtype), lanes)).unwrap_or(u64::MAX);
        if bytes == 0 || bytes > u64::from(u32::MAX) {
            return Err(Error::Program {
                message: format!(
                    "channel {id}: a cell of {numel} lanes is {bytes} bytes, which does \
                     not fit the u32 the endpoint reports it in"
                ),
            });
        }
        let state = make_host_channel_state(dtype, &shape, capacity);
        let endpoint = Endpoint {
            channel_id: id,
            cell_bytes: u32::try_from(bytes).unwrap_or(u32::MAX),
            capacity,
            mirror_bytes: state.cells_len(),
            word_bytes: state.words_len(),
            mirror_base: state.mirror_base(),
            word_base: state.word_base(),
        };
        self.channels.insert(
            id,
            Channel {
                id,
                dtype,
                shape,
                capacity,
                role,
                seeded,
                direction,
                extern_name,
                attachments: BTreeMap::new(),
                state,
            },
        );
        Ok(endpoint)
    }

    /// Bind an instance of `program_id` over `channel_ids`, in slot order.
    ///
    /// Every rule is checked before anything is changed, and the seeds are
    /// written last. A bind that fails leaves the registry exactly as it was
    /// -- the C++ pushes each seed as it decodes it and returns on the first
    /// failure, so a bad third seed leaves the first two sitting in rings
    /// that no instance is attached to, where the next bind of those channels
    /// finds them non-empty and refuses for a reason that has nothing to do
    /// with it.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] naming the single rule that refused.
    pub fn bind_instance(
        &mut self,
        program_id: u64,
        requested_id: Option<u64>,
        geometry: Geometry,
        channel_ids: &[u64],
        seeds: &[(u64, Vec<u8>)],
    ) -> Result<u64> {
        let program = self
            .programs
            .get(&program_id)
            .ok_or_else(|| Error::Program {
                message: format!("no program {program_id}"),
            })?;
        let instance_id = requested_id.unwrap_or(self.next_instance);
        if self.instances.contains_key(&instance_id) {
            return Err(Error::Program {
                message: format!("instance {instance_id} is already bound"),
            });
        }
        if channel_ids.len() != program.channels.len() {
            return Err(Error::Program {
                message: format!(
                    "instance supplies {} channel(s); program {program_id} declares {}",
                    channel_ids.len(),
                    program.channels.len()
                ),
            });
        }
        let mut seen: Vec<u64> = channel_ids.to_vec();
        seen.sort_unstable();
        if seen.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::Program {
                message: "the instance names the same channel twice".to_owned(),
            });
        }

        for (slot, &channel_id) in channel_ids.iter().enumerate() {
            let endpoint = self
                .channels
                .get(&channel_id)
                .ok_or_else(|| Error::Program {
                    message: format!("channel {channel_id} is not registered"),
                })?;
            check_slot(slot, endpoint, &program.channels[slot])?;
        }

        // Seeds are decoded here and applied further down, so a seed that
        // does not decode refuses the bind without having written anything.
        let mut decoded: Vec<(u64, Value)> = Vec::with_capacity(seeds.len());
        for (channel_id, bytes) in seeds {
            let slot = channel_ids
                .iter()
                .position(|id| id == channel_id)
                .ok_or_else(|| Error::Program {
                    message: format!(
                        "seed names channel {channel_id}, which this instance does not bind"
                    ),
                })?;
            if decoded.iter().any(|(id, _)| id == channel_id) {
                return Err(Error::Program {
                    message: format!("channel {channel_id} is seeded twice"),
                });
            }
            let declared = &program.channels[slot];
            if declared.flags & PIE_CHANNEL_SEEDED == 0 {
                return Err(Error::Program {
                    message: format!(
                        "channel {channel_id} carries a seed, but the program declares none \
                         for slot {slot}"
                    ),
                });
            }
            let endpoint = &self.channels[channel_id];
            let lanes = usize::try_from(endpoint.numel()).unwrap_or(usize::MAX);
            let want = wire_cell_bytes(concrete_dtype(endpoint.dtype), lanes);
            if bytes.len() != want {
                return Err(Error::Program {
                    message: format!(
                        "the seed for channel {channel_id} is {} bytes; a cell is {want}",
                        bytes.len()
                    ),
                });
            }
            let value =
                decode_wire(bytes, concrete_dtype(endpoint.dtype), lanes).ok_or_else(|| {
                    Error::Program {
                        message: format!("the seed for channel {channel_id} does not decode"),
                    }
                })?;
            if !endpoint.state.is_empty() {
                return Err(Error::Program {
                    message: format!(
                        "channel {channel_id} already holds a cell, so its seed would be \
                         the second value in the ring rather than the first"
                    ),
                });
            }
            decoded.push((*channel_id, value));
        }

        // Everything above only reads. From here nothing can fail.
        for (channel_id, value) in &decoded {
            let pushed = self.channels[channel_id].state.push(value);
            debug_assert!(
                pushed,
                "the ring was checked empty and its capacity is at least one"
            );
        }
        let states: Vec<Arc<ChannelState>> = channel_ids
            .iter()
            .map(|id| Arc::clone(&self.channels[id].state))
            .collect();
        let program = &self.programs[&program_id];
        let interp = make_instance(&program.plan, states);
        let hash = program.hash;
        let directions: Vec<Direction> = program.channels.iter().map(Direction::of).collect();
        for (&channel_id, &direction) in channel_ids.iter().zip(directions.iter()) {
            self.channels
                .get_mut(&channel_id)
                .expect("checked above")
                .attachments
                .insert(instance_id, direction);
        }
        self.instances.insert(
            instance_id,
            Instance {
                id: instance_id,
                program_id,
                program_hash: hash,
                geometry,
                channel_ids: channel_ids.to_vec(),
                fire_seq: 0,
                interp,
            },
        );
        if requested_id.is_none() {
            self.next_instance += 1;
        }
        Ok(instance_id)
    }

    /// The program with this id.
    #[must_use]
    pub fn program(&self, id: u64) -> Option<&Program> {
        self.programs.get(&id)
    }

    /// The channel with this id.
    #[must_use]
    pub fn channel(&self, id: u64) -> Option<&Channel> {
        self.channels.get(&id)
    }

    /// The instance with this id.
    #[must_use]
    pub fn instance(&self, id: u64) -> Option<&Instance> {
        self.instances.get(&id)
    }

    /// The instance with this id, to fire.
    pub fn instance_mut(&mut self, id: u64) -> Option<&mut Instance> {
        self.instances.get_mut(&id)
    }

    /// Run one instance's channel-plane pass over a fire's read-out.
    ///
    /// The registry's job, because the two halves live here and nowhere else:
    /// [`step`] needs an instance's interpreter state AND its program's plan,
    /// and a caller holding only `&mut Registry` cannot borrow both — the
    /// instance mutably and the program immutably — through the accessors.
    /// Splitting the borrow across two FIELDS is legal inside the type and
    /// impossible outside it, so putting the fire anywhere else costs a clone
    /// of the plan per instance per fire.
    ///
    /// The fire counter advances on a COMMIT only. A blocked pass changed
    /// nothing and did not happen; a faulted one poisoned the instance and the
    /// count of passes it completed is still the count it completed.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] if no such instance is bound, or if its program has
    /// gone — which is a registry that let a program close under an instance.
    pub fn fire(&mut self, id: u64, inputs: &super::PassInputs) -> Result<super::StepOutcome> {
        let instance = self.instances.get_mut(&id).ok_or_else(|| Error::Program {
            message: format!("no instance {id}"),
        })?;
        let program = self
            .programs
            .get(&instance.program_id)
            .ok_or_else(|| Error::Program {
                message: format!(
                    "instance {id} names program {} which is gone",
                    instance.program_id
                ),
            })?;
        let outcome = super::step(&mut instance.interp, &program.plan, inputs);
        if outcome == super::StepOutcome::Committed {
            instance.fire_seq += 1;
        }
        // A fault kills the instance, and the instance was somebody's
        // producer. `step` latches the poison flag on the INSTANCE, which is
        // a fact only this crate can see; the ring's poison WORD is the one
        // an external agent reads, and a host parked on a channel this dead
        // instance was going to put to would otherwise wait for a cell that
        // is never coming. So the fault is published on the rings whose host
        // side READS -- those and no others, because a channel the host
        // writes into has no waiter to tell, and poisoning it would fail a
        // producer for a consumer's death.
        if matches!(outcome, super::StepOutcome::Faulted(_)) {
            let dead: Vec<u64> = instance.channel_ids.clone();
            for channel_id in dead {
                if let Some(channel) = self.channels.get(&channel_id)
                    && channel.role == HostRole::Reader
                {
                    channel.state().fault();
                }
            }
        }
        Ok(outcome)
    }

    /// Detach an instance, freeing its channels for the pass that follows.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] if no such instance is bound.
    pub fn close_instance(&mut self, id: u64) -> Result<()> {
        let instance = self.instances.remove(&id).ok_or_else(|| Error::Program {
            message: format!("no instance {id}"),
        })?;
        for channel_id in instance.channel_ids {
            if let Some(channel) = self.channels.get_mut(&channel_id) {
                channel.attachments.remove(&id);
            }
        }
        Ok(())
    }

    /// Close a channel, marking its ring closed for anyone still holding it.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] if no such channel exists, or if an instance is
    /// still attached -- closing under a live instance would leave it reading
    /// a ring whose closed word says stop while its plan says continue.
    pub fn close_channel(&mut self, id: u64) -> Result<()> {
        let Entry::Occupied(entry) = self.channels.entry(id) else {
            return Err(Error::Program {
                message: format!("no channel {id}"),
            });
        };
        if !entry.get().attachments.is_empty() {
            return Err(Error::Program {
                message: format!(
                    "channel {id} still has {} instance(s) attached",
                    entry.get().attachments.len()
                ),
            });
        }
        entry.get().state.close();
        entry.remove();
        Ok(())
    }
}

/// Whether a second instance may attach to a private channel.
///
/// A prefill pass hands its channel to the decode pass that follows, so a
/// second attachment is the normal case rather than the error the C++ made it
/// for a while -- it refused every second attachment on a non-extern channel,
/// which made the second pass of every two-pass program fail to bind. The two
/// conditions are what make one instance's use of the ring invisible to
/// anyone else: nobody on the host is watching it, and it carries no seed
/// that a second attach would have to decide what to do with.
fn chainable(declared: &LaunchChannel) -> bool {
    HostRole::of(declared) == HostRole::None && declared.flags & PIE_CHANNEL_SEEDED == 0
}

/// Compare one endpoint against what the program declares for its slot.
fn check_slot(slot: usize, endpoint: &Channel, declared: &LaunchChannel) -> Result<()> {
    let refuse = |what: &str, found: String, want: String| Error::Program {
        message: format!(
            "channel {} (slot {slot}): {what} is {found}, but the program declares {want}",
            endpoint.id
        ),
    };

    let want_dtype = channel_dtype(declared.dtype);
    if endpoint.dtype != want_dtype {
        return Err(refuse(
            "dtype",
            endpoint.dtype.to_string(),
            want_dtype.to_string(),
        ));
    }
    if endpoint.shape != declared.shape {
        return Err(refuse(
            "cell shape",
            format!("{:?}", endpoint.shape),
            format!("{:?}", declared.shape),
        ));
    }
    if endpoint.capacity != declared.capacity {
        return Err(refuse(
            "capacity",
            endpoint.capacity.to_string(),
            declared.capacity.to_string(),
        ));
    }
    let want_role = HostRole::of(declared);
    if endpoint.role != want_role {
        return Err(refuse(
            "host role",
            format!("{:?}", endpoint.role),
            format!("{want_role:?}"),
        ));
    }
    let want_seeded = declared.flags & PIE_CHANNEL_SEEDED != 0;
    if endpoint.seeded != want_seeded {
        return Err(refuse(
            "seeding",
            endpoint.seeded.to_string(),
            want_seeded.to_string(),
        ));
    }
    let want_direction = Direction::of(declared);
    if endpoint.direction != want_direction {
        return Err(refuse(
            "extern direction",
            format!("{:?}", endpoint.direction),
            format!("{want_direction:?}"),
        ));
    }

    match want_direction {
        Direction::Private => {
            if !endpoint.attachments.is_empty() && !chainable(declared) {
                return Err(Error::Program {
                    message: format!(
                        "channel {} (slot {slot}) is private and already attached to {} \
                         instance(s); it is not chainable because it is {}",
                        endpoint.id,
                        endpoint.attachments.len(),
                        if want_role == HostRole::None {
                            "seeded"
                        } else {
                            "host-visible"
                        }
                    ),
                });
            }
        }
        Direction::Import | Direction::Export => {
            if endpoint.extern_name != declared.extern_name {
                return Err(refuse(
                    "extern name",
                    String::from_utf8_lossy(&endpoint.extern_name).into_owned(),
                    String::from_utf8_lossy(&declared.extern_name).into_owned(),
                ));
            }
            if endpoint
                .attachments
                .values()
                .any(|&existing| existing == want_direction)
            {
                return Err(Error::Program {
                    message: format!(
                        "channel {} (slot {slot}) already has an instance attached as \
                         {want_direction:?}; two would race on the same ring",
                        endpoint.id
                    ),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use driver_api::plan::{LaunchStage, LaunchStagePlan};
    use tensor_ir::registry::Stage;

    use super::*;

    fn channel_decl(id: u32, flags: u8, extern_dir: i8) -> LaunchChannel {
        LaunchChannel {
            id,
            capacity: 1,
            dtype: 0,
            flags,
            extern_dir,
            readiness: 0,
            shape: vec![],
            extern_name: vec![],
        }
    }

    fn package(channels: Vec<LaunchChannel>) -> LaunchPackage {
        LaunchPackage {
            values: vec![],
            channels,
            ports: vec![],
            names: vec![],
            stages: vec![LaunchStage {
                kind: Stage::Epilogue as u8,
                ops: vec![],
                puts: vec![],
                takes: vec![],
                reads: vec![],
            }],
            plans: vec![LaunchStagePlan::default()],
        }
    }

    fn register_matching(registry: &mut Registry, id: u64, decl: &LaunchChannel) -> Endpoint {
        registry
            .register_channel(ChannelSpec {
                id,
                dtype: channel_dtype(decl.dtype),
                shape: decl.shape.clone(),
                capacity: decl.capacity,
                role: HostRole::of(decl),
                seeded: decl.flags & PIE_CHANNEL_SEEDED != 0,
                direction: Direction::of(decl),
                extern_name: decl.extern_name.clone(),
            })
            .expect("a matching endpoint")
    }

    #[test]
    fn the_same_hash_registers_once_and_returns_the_first_id() {
        let mut registry = Registry::new();
        let first = registry
            .register_program(0xABC, package(vec![]), vec![])
            .expect("first");
        let second = registry
            .register_program(0xABC, package(vec![]), vec![])
            .expect("second");
        assert_eq!(
            first, second,
            "a program registered twice under one hash must not get two plans; a \
             second plan means two instances of the same program cannot share a \
             compiled kernel"
        );
        assert!(registry.program(first).is_some());
    }

    #[test]
    fn a_program_with_no_stages_is_refused_rather_than_registered_empty() {
        let mut registry = Registry::new();
        let mut empty = package(vec![]);
        empty.stages.clear();
        empty.plans.clear();
        assert!(
            registry.register_program(1, empty, vec![]).is_err(),
            "a program with nothing to run was registered; every fire of it would \
             report success having done nothing"
        );
    }

    #[test]
    fn a_channel_with_a_zero_extent_is_refused_because_its_stride_would_be_zero() {
        let mut registry = Registry::new();
        let refused = registry.register_channel(ChannelSpec {
            id: 1,
            dtype: PIE_CHANNEL_DTYPE_F32,
            shape: vec![4, 0, 4],
            capacity: 2,
            role: HostRole::None,
            seeded: false,
            direction: Direction::Private,
            extern_name: vec![],
        });
        assert!(
            refused.is_err(),
            "a cell of zero lanes gives the ring a stride of zero, so every push \
             writes over the last one and the ring reports success"
        );
    }

    /// The fire the seam calls, which is what makes `step` reachable at all.
    ///
    /// Before `Registry::fire` existed, `pipeline::step` had no production
    /// caller: the engine seam ran every launch of a frame, waited, and
    /// dropped the arena — so the interpreter was exercised only by tests that
    /// built their own inputs, and the channel plane was dead code in a
    /// running deployment.
    ///
    /// Two claims, and the second is the one worth having: a fire that commits
    /// ADVANCES the instance, and a fire against an instance that is not bound
    /// refuses by name rather than firing something else.
    #[test]
    fn a_fire_advances_the_instance_it_committed_and_refuses_one_that_is_gone() {
        let decl = channel_decl(0, 0, -1);
        let mut registry = Registry::new();
        let program = registry
            .register_program(1, package(vec![decl.clone()]), vec![])
            .expect("program");
        register_matching(&mut registry, 10, &decl);
        let instance = registry
            .bind_instance(program, None, Geometry::Host, &[10], &[])
            .expect("instance");

        assert_eq!(registry.instance(instance).expect("bound").fire_seq, 0);
        let outcome = registry
            .fire(instance, &super::super::PassInputs::none())
            .expect("a bound instance fires");
        assert_eq!(outcome, super::super::StepOutcome::Committed);
        assert_eq!(
            registry.instance(instance).expect("bound").fire_seq,
            1,
            "a committed pass has to advance the instance, or a program that \
             counts its own fires counts none of them"
        );

        // An id nobody bound. The seam derives this from a roster row, so a
        // frame built against a registry that has moved on must refuse rather
        // than fire whatever else is in the map.
        let missing = registry.fire(instance + 999, &super::super::PassInputs::none());
        assert!(missing.is_err(), "an unbound instance cannot fire");
    }

    /// A fault publishes poison on the rings the host reads, and nowhere else.
    ///
    /// # Why a log line is not a report
    ///
    /// `step` latches the fault on the INSTANCE, which is a fact only this
    /// crate can see, and every seam above it treats a faulted member as one
    /// program's problem rather than the frame's -- correctly, since the other
    /// requests in the batch ran. But the guest behind the faulted one is
    /// parked on its output ring waiting for a cell that a dead instance will
    /// never put, and `pipeline::channel` learns that only by reading the
    /// ring's POISON word out of the shared mapping. Left unwritten, the fault
    /// is a hang: no error reaches the guest, no completion arrives, and the
    /// only trace is a warning in the server's log.
    ///
    /// The second half is the limit. A channel the host WRITES into has no
    /// waiter to tell -- the host is its producer -- and poisoning it would
    /// report a consumer's death to a producer that is still perfectly able
    /// to write. So this is the one direction, checked here because "poison
    /// everything attached" is the obvious implementation and passes any test
    /// that looks only at the output ring.
    #[test]
    fn a_faulted_instance_poisons_the_rings_its_host_reads_and_not_the_ones_it_writes() {
        // Capacity one, already full, and the epilogue puts: `commit` refuses
        // the overflow and faults. Readiness is 0 -- unstated -- so the gate
        // above does not turn this into a Blocked, which is the difference
        // between "not yet" and "never".
        let out = LaunchChannel {
            id: 0,
            capacity: 1,
            dtype: PIE_CHANNEL_DTYPE_F32,
            flags: PIE_CHANNEL_HOST_VISIBLE | PIE_CHANNEL_HOST_READER,
            extern_dir: -1,
            readiness: 0,
            shape: vec![1],
            extern_name: vec![],
        };
        let mut inbox = out.clone();
        inbox.id = 1;
        // No reader bit: host-visible and WRITTEN by the host.
        inbox.flags = PIE_CHANNEL_HOST_VISIBLE;

        let mut package = package(vec![out.clone(), inbox.clone()]);
        package.values = vec![driver_api::plan::LaunchValue {
            id: 0,
            source: driver_api::local::PIE_VALUE_CONST,
            dtype: PIE_CHANNEL_DTYPE_F32,
            intrinsic: 0,
            channel: 0,
            literal_bits: 0,
            shape: vec![1],
        }];
        package.stages[0].puts = vec![driver_api::plan::LaunchPut {
            channel: 0,
            value: 0,
        }];

        let mut registry = Registry::new();
        let program = registry
            .register_program(0xDEAD, package, vec![])
            .expect("program");
        register_matching(&mut registry, 10, &out);
        register_matching(&mut registry, 11, &inbox);
        let instance = registry
            .bind_instance(program, None, Geometry::Host, &[10, 11], &[])
            .expect("instance");

        // Fill the output ring, so the put has nowhere to land.
        assert!(
            registry
                .channel(10)
                .expect("the ring")
                .state()
                .push(&super::super::Value::F32(vec![0.0])),
            "the fixture could not fill its own ring"
        );

        let outcome = registry
            .fire(instance, &super::super::PassInputs::none())
            .expect("a bound instance fires");
        assert!(
            matches!(outcome, super::super::StepOutcome::Faulted(_)),
            "the fixture did not fault, so this test proves nothing: {outcome:?}"
        );
        assert_ne!(
            registry.channel(10).expect("the ring").state().poison(),
            0,
            "the fault was not published on the ring the host reads, so a guest \
             parked on it waits for a cell a dead instance will never put"
        );
        assert_eq!(
            registry.channel(11).expect("the ring").state().poison(),
            0,
            "a channel the host WRITES was poisoned by its consumer's death, \
             which fails a producer that is still able to produce"
        );
    }

    #[test]
    fn every_mismatched_field_refuses_by_name_rather_than_by_dump() {
        let decl = channel_decl(0, 0, -1);
        let mut registry = Registry::new();
        let program = registry
            .register_program(1, package(vec![decl.clone()]), vec![])
            .expect("program");

        // Capacity is the one that differs, and the message has to say so.
        registry
            .register_channel(ChannelSpec {
                id: 10,
                dtype: PIE_CHANNEL_DTYPE_F32,
                shape: vec![],
                capacity: 9,
                role: HostRole::None,
                seeded: false,
                direction: Direction::Private,
                extern_name: vec![],
            })
            .expect("endpoint");
        let error = registry
            .bind_instance(program, None, Geometry::Host, &[10], &[])
            .expect_err("capacity 9 against a declared 1");
        let text = error.to_string();
        assert!(
            text.contains("capacity"),
            "the refusal must name the field that disagreed; it said: {text}"
        );
        assert!(
            !text.contains("dtype"),
            "the refusal named fields that agreed, which is the dump the C++ \
             produces and the reason a bind failure is a bisect: {text}"
        );
    }

    #[test]
    fn a_private_channel_may_be_handed_to_the_pass_that_follows() {
        let decl = channel_decl(0, 0, -1);
        let mut registry = Registry::new();
        let program = registry
            .register_program(1, package(vec![decl.clone()]), vec![])
            .expect("program");
        register_matching(&mut registry, 10, &decl);

        let first = registry
            .bind_instance(program, None, Geometry::Host, &[10], &[])
            .expect("prefill");
        let second = registry
            .bind_instance(program, None, Geometry::Host, &[10], &[])
            .expect(
                "the decode pass could not attach to the channel the prefill pass \
                 created, so every two-pass program fails to bind",
            );
        assert_ne!(first, second);
        assert_eq!(registry.channel(10).expect("channel").attachments.len(), 2);
    }

    #[test]
    fn a_host_visible_channel_admits_only_one_instance() {
        let decl = channel_decl(0, PIE_CHANNEL_HOST_VISIBLE, -1);
        let mut registry = Registry::new();
        let program = registry
            .register_program(1, package(vec![decl.clone()]), vec![])
            .expect("program");
        register_matching(&mut registry, 10, &decl);
        registry
            .bind_instance(program, None, Geometry::Host, &[10], &[])
            .expect("first");
        assert!(
            registry
                .bind_instance(program, None, Geometry::Host, &[10], &[])
                .is_err(),
            "two instances share a channel the host is also writing, so a cell \
             the host put in for one of them is taken by whichever fires first"
        );
    }

    #[test]
    fn two_exporters_of_one_channel_are_refused() {
        let mut decl = channel_decl(0, 0, 1);
        decl.extern_name = b"kv".to_vec();
        let mut registry = Registry::new();
        let program = registry
            .register_program(1, package(vec![decl.clone()]), vec![])
            .expect("program");
        register_matching(&mut registry, 10, &decl);
        registry
            .bind_instance(program, None, Geometry::Host, &[10], &[])
            .expect("first exporter");
        let error = registry
            .bind_instance(program, None, Geometry::Host, &[10], &[])
            .expect_err("two exporters race on one ring");
        assert!(error.to_string().contains("Export"), "{error}");
    }

    #[test]
    fn naming_one_channel_twice_is_refused_before_anything_attaches() {
        let decl = channel_decl(0, 0, -1);
        let mut registry = Registry::new();
        let program = registry
            .register_program(1, package(vec![decl.clone(), decl.clone()]), vec![])
            .expect("program");
        register_matching(&mut registry, 10, &decl);
        assert!(
            registry
                .bind_instance(program, None, Geometry::Host, &[10, 10], &[])
                .is_err(),
            "one ring is bound to two slots, so a put through one slot is seen as \
             an arrival on the other"
        );
        assert!(
            registry
                .channel(10)
                .expect("channel")
                .attachments
                .is_empty()
        );
    }

    #[test]
    fn a_bad_seed_leaves_no_earlier_seed_sitting_in_a_ring() {
        let seeded = channel_decl(0, PIE_CHANNEL_SEEDED, -1);
        let mut registry = Registry::new();
        let program = registry
            .register_program(1, package(vec![seeded.clone(), seeded.clone()]), vec![])
            .expect("program");
        register_matching(&mut registry, 10, &seeded);
        register_matching(&mut registry, 11, &seeded);

        let good = 1.0f32.to_le_bytes().to_vec();
        let error = registry.bind_instance(
            program,
            None,
            Geometry::Host,
            &[10, 11],
            &[(10, good), (11, vec![0u8; 3])],
        );
        assert!(
            error.is_err(),
            "a three-byte seed for an f32 cell was accepted"
        );

        // The claim: channel 10's good seed must not be in its ring. The C++
        // pushes as it decodes, so the first seed lands, the second fails, and
        // the next attempt to bind these channels finds 10 non-empty and
        // refuses for a reason that has nothing to do with what went wrong.
        assert!(
            registry.channel(10).expect("channel").state().is_empty(),
            "a failed bind left an earlier seed in the ring, so retrying the bind \
             refuses with 'already holds a cell'"
        );
        assert!(registry.instance(1).is_none());
    }

    #[test]
    fn a_closed_instance_frees_its_channel_for_the_next_one() {
        let decl = channel_decl(0, PIE_CHANNEL_HOST_VISIBLE, -1);
        let mut registry = Registry::new();
        let program = registry
            .register_program(1, package(vec![decl.clone()]), vec![])
            .expect("program");
        register_matching(&mut registry, 10, &decl);
        let first = registry
            .bind_instance(program, None, Geometry::Host, &[10], &[])
            .expect("first");
        registry.close_instance(first).expect("close");
        registry
            .bind_instance(program, None, Geometry::Host, &[10], &[])
            .expect("the closed instance still held its attachment");
    }

    #[test]
    fn a_channel_with_an_instance_on_it_cannot_be_closed() {
        let decl = channel_decl(0, 0, -1);
        let mut registry = Registry::new();
        let program = registry
            .register_program(1, package(vec![decl.clone()]), vec![])
            .expect("program");
        register_matching(&mut registry, 10, &decl);
        let instance = registry
            .bind_instance(program, None, Geometry::Host, &[10], &[])
            .expect("bind");
        assert!(
            registry.close_channel(10).is_err(),
            "the ring was freed under a live instance, which would read a closed \
             word saying stop while its plan says continue"
        );
        registry.close_instance(instance).expect("close instance");
        registry.close_channel(10).expect("now it may close");
        assert!(registry.channel(10).is_none());
    }

    #[test]
    fn every_geometry_class_the_wire_can_carry_is_bound() {
        // The C++ gate stayed at HOST-only long after the device resolver
        // existed, refusing every device-resolved decode at bind.
        for class in [
            PIE_GEOMETRY_CLASS_HOST,
            PIE_GEOMETRY_CLASS_DECODE_ENVELOPE,
            PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY,
        ] {
            let geometry = Geometry::from_wire(class).expect("a class the wire carries");
            assert_eq!(geometry.to_wire(), class, "the round trip must be exact");
        }
        assert!(
            Geometry::from_wire(3).is_err(),
            "an unknown class was accepted"
        );
    }

    /// A default-built registry hands out the same ids a `new` one does.
    ///
    /// # What this is guarding
    ///
    /// `Default` used to be DERIVED, which started both counters at zero.
    /// Zero is the ABI's "none": `validate_instance_binding` refuses a
    /// binding whose `instance_id` is zero, so the first instance bound out
    /// of a default-built registry was rejected by the layer above -- and
    /// every caller in this crate's own tests used `new`, so nothing here
    /// saw it. `driver-vulkan`'s `Programs` is `#[derive(Default)]` over
    /// this type, which is how it reached a real driver.
    ///
    /// Asserted through `bind_instance` rather than by reading the fields,
    /// because the field is private and the ID IS THE OBSERVABLE.
    #[test]
    fn a_default_registry_does_not_hand_out_the_abis_none() {
        let decl = channel_decl(1, PIE_CHANNEL_HOST_VISIBLE, -1);
        let mut registry = Registry::default();
        let program = registry
            .register_program(1, package(vec![decl.clone()]), vec![])
            .expect("a one-stage program");
        assert_ne!(program, 0, "program zero is the ABI's `no program`");
        register_matching(&mut registry, 1, &decl);
        let instance = registry
            .bind_instance(program, None, Geometry::Host, &[1], &[])
            .expect("an instance over the one channel");
        assert_ne!(
            instance, 0,
            "instance zero is refused by `validate_instance_binding`, so this \
             registry cannot bind its own first instance"
        );
    }
}

#[cfg(test)]
mod thread_safety {
    /// A [`Registry`] can be held by a driver the engine registers.
    ///
    /// # Why this is a test and not a doc line
    ///
    /// `engine` keeps every driver in a `'static RwLock<Vec<
    /// Option<DriverRegistration>>>`, so a backend that owns a registry must
    /// be `Send + Sync`. It was neither: `ChannelState` used a `RefCell` for
    /// its cell bytes and instances held rings behind an `Rc`, which made the
    /// whole registry `!Send` and `!Sync` by construction.
    ///
    /// Nothing said so. The failure was a compile error on the engine's
    /// static, in another crate, naming a type this file does not mention --
    /// and the fix looked like it belonged there. So the property is asserted
    /// HERE, where losing it happens: one `Rc` or one `RefCell` reintroduced
    /// anywhere reachable from a `Registry` fails this, in this crate, with
    /// the offending field named.
    #[test]
    fn a_registry_can_cross_threads_and_be_shared() {
        const fn require<T: Send + Sync>() {}
        require::<super::Registry>();
        require::<crate::channel::ChannelState>();
        require::<super::Channel>();
        require::<super::Instance>();
        require::<super::Program>();
    }
}
