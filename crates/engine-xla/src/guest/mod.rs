//! Device forms of eta guest programs: every stage of a program lowered to
//! StableHLO (`lower`), compiled per batch shape and run on the device
//! (`run`). A program any of whose stages does not lower is refused a device
//! form and runs in the host interpreter.

pub mod lower;
pub mod run;

use std::collections::BTreeMap;
use std::sync::Arc;

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
}

/// The channels a pass carries to the next one and nobody else sees: a
/// one-cell ring the host neither reads nor writes, bound to no port and
/// no extern, taken and put by one stage and touched by no other. Their
/// cells can stay on the device between passes (`Plane::fire_guests`).
#[must_use]
pub fn carried_channels(package: &LaunchPackage) -> Vec<u32> {
    if std::env::var_os("PIE_XLA_GUESTS_CARRY").is_some_and(|v| v == "0") {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (c, decl) in package.channels.iter().enumerate() {
        let c = c as u32;
        if decl.capacity != 1
            || decl.host_role != eta_ir::container::HostRole::None
            || decl.extern_dir.is_some()
            || package.ports.iter().any(|p| !p.is_const && p.channel == c)
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
    compiled: BTreeMap<u64, Arc<Compiled>>,
    refused: BTreeMap<u64, Refused>,
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
                logits: None,
                mtp: None,
            },
            &carried_channels(package),
        )?;
    }
    Ok(())
}

impl Forms {
    pub fn admit(&mut self, program: u64, package: &LaunchPackage) -> bool {
        match admits(package) {
            Ok(()) => {
                let key = *blake3::hash(format!("{package:?}").as_bytes()).as_bytes();
                self.compiled.insert(
                    program,
                    Arc::new(Compiled {
                        key,
                        stages: package.stages.len(),
                        carried: carried_channels(package),
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
    pub fn get(&self, program: u64) -> Option<&Arc<Compiled>> {
        self.compiled.get(&program)
    }

    #[must_use]
    pub fn refusal(&self, program: u64) -> Option<&Refused> {
        self.refused.get(&program)
    }

    pub fn forget(&mut self, program: u64) {
        self.compiled.remove(&program);
        self.refused.remove(&program);
    }

    #[must_use]
    pub fn tally(&self) -> (usize, usize) {
        (self.compiled.len(), self.refused.len())
    }
}
