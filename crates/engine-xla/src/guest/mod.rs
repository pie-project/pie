//! Device forms of eta guest programs: every stage of a program lowered to
//! StableHLO (`lower`), compiled per batch shape and run on the device
//! (`run`). A program any of whose stages does not lower is refused a device
//! form and runs in the host interpreter.

pub mod lower;
pub mod run;

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::program::ProgramId;
use eta_compiler::codegen::launch::LaunchPackage;

pub use lower::Refused;

/// A program's device form: every stage lowers.
#[derive(Debug)]
pub struct Compiled {
    /// The package's content hash: programs with one content share their
    /// compiled stages and run as one group.
    pub key: [u8; 32],
    pub stages: usize,
    /// Channels whose cell can stay on the device between passes.
    pub carried: Vec<u32>,
    /// Each carried channel's cell in u32 words, a lane (an instance) one
    /// cell: what its `Channel` slots hold.
    pub carried_words: Vec<(u32, u32)>,
    /// Channels the next fire reads off the device (`embed_tokens`): each
    /// with its cell's u32 words and its ring's capacity. A pass's put
    /// lands in the ring's `Channel` slots too, where the fire gathers it.
    pub fed: Vec<(u32, u32, u32)>,
}

impl Compiled {
    /// The channels a pass keeps in the pool: the carried ones (read and
    /// written there) and the fed ones (written there): its stages' key.
    #[must_use]
    pub fn pooled(&self) -> Vec<u32> {
        let mut out = self.carried.clone();
        out.extend(self.fed.iter().map(|&(c, _, _)| c));
        out
    }
}

/// The channels the next fire reads off the device: a non-constant
/// `embed_tokens` port's that a stage puts to, with its cell's u32 words
/// and ring capacity.
#[must_use]
pub fn fed_channels(package: &LaunchPackage) -> Vec<(u32, u32, u32)> {
    package
        .ports
        .iter()
        .filter(|p| !p.is_const && p.port == eta_ir::registry::Port::EmbedTokens)
        // A stage puts its cells: the device produces them.
        .filter(|p| {
            package
                .stages
                .iter()
                .any(|s| s.puts.iter().any(|put| put.channel == p.channel))
        })
        .filter_map(|p| {
            let decl = package.channels.get(p.channel as usize)?;
            let dtype = eta_exec::concrete_dtype(decl.dtype);
            let n = decl.shape.iter().map(|&d| d as usize).product::<usize>().max(1);
            Some((
                p.channel,
                u32::try_from(lower::words(dtype, n)).unwrap_or(u32::MAX),
                decl.capacity,
            ))
        })
        .collect()
}

/// The u32 words of each carried channel's cell (`carried_channels`): the
/// value its one put lands, packed as the stages pack it.
#[must_use]
pub fn carried_words(package: &LaunchPackage) -> Vec<(u32, u32)> {
    carried_channels(package)
        .into_iter()
        .filter_map(|c| {
            let put = package
                .stages
                .iter()
                .flat_map(|s| s.puts.iter())
                .find(|p| p.channel == c)?;
            let value = &package.values[put.value as usize];
            let n = value.shape.iter().map(|&d| d as usize).product::<usize>().max(1);
            Some((c, u32::try_from(lower::words(value.dtype, n)).unwrap_or(u32::MAX)))
        })
        .collect()
}

/// A `Channel` slot's bytes (`SlotRole::Channel`, the pool's
/// [`crate::store::layout::CHANNEL`] shape).
pub const CHANNEL_SLOT: u64 = 1 << 10;

impl Compiled {
    /// The rings an instance keeps in the pool's `Channel` slots: each
    /// carried channel, one cell of its words, one contiguous run.
    #[must_use]
    pub fn rings(&self) -> Vec<engine::program::RingFacts> {
        let ring = |channel: u32, words: u32, capacity: u32| {
            let bytes = u64::from(words) * 4;
            // Each cell whole slots, back to back.
            let slots = bytes.div_ceil(CHANNEL_SLOT) * u64::from(capacity.max(1));
            engine::program::RingFacts {
                channel,
                cell_bytes: bytes,
                capacity,
                footprint: engine::program::Footprint {
                    bytes: slots * CHANNEL_SLOT,
                    slots: u32::try_from(slots).unwrap_or(u32::MAX),
                    contiguous: true,
                },
            }
        };
        self.carried_words
            .iter()
            .map(|&(c, w)| ring(c, w, 1))
            .chain(self.fed.iter().map(|&(c, w, cap)| ring(c, w, cap)))
            .collect()
    }
}

/// The channels a pass carries to the next one and nobody else sees: a
/// one-cell ring the host neither reads nor writes, bound to no port, input
/// or extern, taken and put by one stage and touched by no other. Their
/// cells can stay on the device between passes (`Plane::fire_guests`).
#[must_use]
pub fn carried_channels(package: &LaunchPackage) -> Vec<u32> {
    let mut out = Vec::new();
    for (c, decl) in package.channels.iter().enumerate() {
        let c = c as u32;
        if decl.capacity != 1
            || decl.host_role != eta_ir::container::HostRole::None
            || decl.extern_dir.is_some()
            || package.ports.iter().any(|p| !p.is_const && p.channel == c)
            || package.inputs.contains(&c)
        {
            continue;
        }
        let touching: Vec<&eta_compiler::codegen::launch::LaunchStage> = package
            .stages
            .iter()
            .filter(|s| {
                s.takes.contains(&c)
                    || s.reads.contains(&c)
                    || s.puts.iter().any(|p| p.channel == c)
            })
            .collect();
        let [stage] = touching.as_slice() else {
            continue;
        };
        let takes = stage.takes.iter().filter(|&&t| t == c).count();
        let puts = stage.puts.iter().filter(|p| p.channel == c).count();
        if takes == 1 && puts == 1 && !stage.reads.contains(&c) {
            out.push(c);
        }
    }
    out
}

#[derive(Debug, Default)]
pub struct Forms {
    compiled: BTreeMap<ProgramId, Arc<Compiled>>,
    refused: BTreeMap<ProgramId, Refused>,
}

/// Whether every stage of `package` lowers.
pub fn admits(package: &LaunchPackage) -> Result<(), Refused> {
    if std::env::var_os("PIE_XLA_GUESTS_ON_HOST").is_some_and(|v| v != "0") {
        return Err(Refused(
            "PIE_XLA_GUESTS_ON_HOST asks for the host interpreter".into(),
        ));
    }
    for at in 0..package.stages.len() {
        lower::lower(
            package,
            at,
            lower::Batch {
                lanes: 1,
                reads: Vec::new(),
            },
            &carried_channels(package),
            1,
        )?;
    }
    Ok(())
}

impl Forms {
    pub fn admit(&mut self, program: ProgramId, package: &LaunchPackage) -> bool {
        match admits(package) {
            Ok(()) => {
                let key = *blake3::hash(format!("{package:?}").as_bytes()).as_bytes();
                self.compiled.insert(
                    program,
                    Arc::new(Compiled {
                        key,
                        stages: package.stages.len(),
                        carried: carried_channels(package),
                        carried_words: carried_words(package),
                        fed: fed_channels(package),
                    }),
                );
                true
            }
            Err(why) => {
                self.refused.insert(program, why);
                false
            }
        }
    }

    #[must_use]
    pub fn get(&self, program: ProgramId) -> Option<&Arc<Compiled>> {
        self.compiled.get(&program)
    }

    #[must_use]
    pub fn refusal(&self, program: ProgramId) -> Option<&Refused> {
        self.refused.get(&program)
    }

    pub fn forget(&mut self, program: ProgramId) {
        self.compiled.remove(&program);
        self.refused.remove(&program);
    }

    #[must_use]
    pub fn tally(&self) -> (usize, usize) {
        (self.compiled.len(), self.refused.len())
    }
}
