use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::sync::Arc;

use driver_api::program::{LaunchChannel, LaunchPackage};
use tensor_ir::container::{ChanDType, ExternDir};

use super::channel::{ChannelState, InterpInstance, make_host_channel_state, make_instance};
use super::plan::{ExecPlan, adopt_launch_package};
use super::value::{Value, concrete_dtype, decode_wire, wire_cell_bytes};
use crate::{Error, Result};

/// How much of a fire's geometry an instance's descriptor resolves on device.
///
/// The port registry's, now that the ports live there (palo decision 19). This
/// plane used to declare a three-variant copy of it plus `from_wire`/`to_wire`
/// against three `PIE_GEOMETRY_CLASS_*` `u32` constants, which a `const`
/// assertion in `driver-api` held in step with a fourth spelling.
pub use tensor_ir::registry::GeometryClass as Geometry;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Direction {
    Private,

    Import,

    Export,
}

impl Direction {
    /// Which way `channel` crosses, if it crosses.
    ///
    /// This function and the `from_wire` that stood beside it read the SAME
    /// axis out of two different encodings — `extern_dir: i8` as `0 => Import,
    /// 1 => Export, else Private`, and `PIE_CHANNEL_EXTERN_*` as
    /// `1 => Import, 2 => Export, 0 => Private` — and nothing said they were
    /// about the same thing. There is one encoding now, and it is an
    /// `Option<ExternDir>`.
    #[must_use]
    pub const fn of(channel: &LaunchChannel) -> Self {
        match channel.extern_dir {
            Some(ExternDir::Import) => Self::Import,
            Some(ExternDir::Export) => Self::Export,
            None => Self::Private,
        }
    }
}

/// Which end of a channel the host holds.
///
/// PTIR's own, re-exported. This plane used to declare a three-variant copy
/// and derive it from two `flags` bits — a two-bit encoding of three states,
/// whose fourth pattern was reachable and meant nothing.
pub use tensor_ir::container::HostRole;

pub use driver_api::program::EmittedKernel;

#[derive(Debug)]
pub struct Program {
    pub id: u64,

    pub hash: u64,

    pub channels: Vec<LaunchChannel>,

    pub plan: ExecPlan,

    pub kernels: Vec<EmittedKernel>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelSpec {
    pub id: u64,

    pub dtype: ChanDType,

    pub shape: Vec<u32>,

    pub capacity: u32,

    pub role: HostRole,

    pub seeded: bool,

    pub direction: Direction,

    pub extern_name: Vec<u8>,
}

#[derive(Debug)]
pub struct Channel {
    pub id: u64,

    pub dtype: ChanDType,

    pub shape: Vec<u32>,

    pub capacity: u32,

    pub role: HostRole,

    pub seeded: bool,

    pub direction: Direction,

    pub extern_name: Vec<u8>,

    pub attachments: BTreeMap<u64, Direction>,
    state: Arc<ChannelState>,
}

impl Channel {
    #[must_use]
    pub fn numel(&self) -> u64 {
        crate::shape_numel(&self.shape)
    }

    #[must_use]
    pub fn state(&self) -> &Arc<ChannelState> {
        &self.state
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub channel_id: u64,

    pub cell_bytes: u32,

    pub capacity: u32,

    pub mirror_bytes: usize,

    pub word_bytes: usize,

    pub mirror_base: u64,

    pub word_base: u64,
}

#[derive(Debug)]
pub struct Instance {
    pub id: u64,

    pub program_id: u64,

    pub program_hash: u64,

    pub geometry: Geometry,

    pub channel_ids: Vec<u64>,

    pub fire_seq: u64,

    pub interp: InterpInstance,
}

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

        if let Some(position) = shape.iter().position(|&d| d == 0) {
            return Err(Error::Program {
                message: format!("channel {id}: extent {position} of the cell shape is zero"),
            });
        }
        let numel = crate::shape_numel(&shape);

        let lanes = usize::try_from(numel).unwrap_or(usize::MAX);
        let bytes = u64::try_from(wire_cell_bytes(concrete_dtype(dtype), lanes))
            .unwrap_or(u64::MAX);
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
            if !declared.seeded {
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

    #[must_use]
    pub fn program(&self, id: u64) -> Option<&Program> {
        self.programs.get(&id)
    }

    #[must_use]
    pub fn channel(&self, id: u64) -> Option<&Channel> {
        self.channels.get(&id)
    }

    #[must_use]
    pub fn instance(&self, id: u64) -> Option<&Instance> {
        self.instances.get(&id)
    }

    pub fn instance_mut(&mut self, id: u64) -> Option<&mut Instance> {
        self.instances.get_mut(&id)
    }

    pub fn fire(&mut self, id: u64, inputs: &crate::PassInputs) -> Result<crate::StepOutcome> {
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
        let outcome = crate::step(&mut instance.interp, &program.plan, inputs);
        if outcome == crate::StepOutcome::Committed {
            instance.fire_seq += 1;
        }

        if matches!(outcome, crate::StepOutcome::Faulted(_)) {
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

fn chainable(declared: &LaunchChannel) -> bool {
    declared.host_role == HostRole::None && !declared.seeded
}

fn check_slot(slot: usize, endpoint: &Channel, declared: &LaunchChannel) -> Result<()> {
    let refuse = |what: &str, found: String, want: String| Error::Program {
        message: format!(
            "channel {} (slot {slot}): {what} is {found}, but the program declares {want}",
            endpoint.id
        ),
    };

    if endpoint.dtype != declared.dtype {
        return Err(refuse(
            "dtype",
            format!("{:?}", endpoint.dtype),
            format!("{:?}", declared.dtype),
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
    let want_role = declared.host_role;
    if endpoint.role != want_role {
        return Err(refuse(
            "host role",
            format!("{:?}", endpoint.role),
            format!("{want_role:?}"),
        ));
    }
    let want_seeded = declared.seeded;
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
