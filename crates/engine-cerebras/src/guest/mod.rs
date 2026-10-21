//! Device forms of eta guest programs: every stage of a program lowered to
//! a CSL phase program (`lower`), compiled per batch shape and run on the
//! fabric (`run`). A program any of whose stages does not lower (or does
//! not fit one PE) is refused a device form and runs in the host
//! interpreter.

pub mod lower;
pub mod run;

use std::collections::BTreeMap;

use eta_compiler::codegen::launch::LaunchPackage;

pub use lower::Refused;

/// A program's device form: every stage lowers.
#[derive(Debug, Clone, Copy)]
pub struct Compiled {
    /// The package's content hash: programs with one content share their
    /// compiled stages.
    pub key: [u8; 32],
    pub stages: usize,
    /// PEs a lane spreads over: the fewest the lowering fits, widened when
    /// the compiler finds a stage too big for the PE (`Forms::widen`).
    pub cols: u32,
}

#[derive(Debug, Default)]
pub struct Forms {
    compiled: BTreeMap<u64, Compiled>,
    refused: BTreeMap<u64, Refused>,
}

/// Whether every stage of `package` lowers (for one lane, every intrinsic
/// fed as a value), and over how many PEs the lane spreads.
pub fn admits(package: &LaunchPackage) -> Result<u32, Refused> {
    if std::env::var_os("PIE_CEREBRAS_GUESTS_ON_HOST").is_some_and(|v| v != "0") {
        return Err(Refused(
            "PIE_CEREBRAS_GUESTS_ON_HOST asks for the host interpreter".into(),
        ));
    }
    if let Some(cols) = run::columns_above(package, 0) {
        return Ok(cols);
    }
    // The refusal of the widest spread, for the record.
    let widest = lower::wide_axis(package)
        .and_then(|v| lower::column_choices(v).last().copied())
        .unwrap_or(1);
    for at in 0..package.stages.len() {
        lower::lower(
            package,
            at,
            lower::Batch {
                lanes: 1,
                cols: widest,
                logits: None,
                mtp: None,
            },
        )?;
    }
    Err(Refused("no spread of the lane over PEs fits".into()))
}

impl Forms {
    pub fn admit(&mut self, program: u64, package: &LaunchPackage) -> bool {
        match admits(package) {
            Ok(cols) => {
                let key = *blake3::hash(format!("{package:?}").as_bytes()).as_bytes();
                self.compiled.insert(
                    program,
                    Compiled {
                        key,
                        stages: package.stages.len(),
                        cols,
                    },
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
    pub fn get(&self, program: u64) -> Option<&Compiled> {
        self.compiled.get(&program)
    }

    /// Spreads `program`'s lanes over more PEs after the compiler found a
    /// stage too big for the PE at the current spread (`why`): the next
    /// spread every stage lowers at, or none, when the program loses its
    /// device form and runs in the interpreter from now on.
    pub fn widen(&mut self, program: u64, package: &LaunchPackage, why: &str) -> Option<u32> {
        let current = self.compiled.get(&program)?.cols;
        // A quarter wider at least: a spread one PE wider rarely links
        // where this one did not, and each try is a compile.
        match run::columns_above(package, current + current / 4) {
            Some(cols) => {
                if let Some(form) = self.compiled.get_mut(&program) {
                    form.cols = cols;
                }
                Some(cols)
            }
            None => {
                self.demote(
                    program,
                    format!("no spread of the lane over PEs fits the PE (at {current}: {why})"),
                );
                None
            }
        }
    }

    /// Takes `program`'s device form away: it runs in the interpreter from
    /// now on, `why` on record.
    pub fn demote(&mut self, program: u64, why: String) {
        self.compiled.remove(&program);
        self.refused.insert(program, Refused(why));
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
