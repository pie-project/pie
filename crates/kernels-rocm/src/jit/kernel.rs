//! A kernel: one instantiation of one carried unit, made once per process.
//! Its unit and every device's load of it hang off it, so a fire finds them
//! by following a reference rather than by naming them again.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "rocm")]
use crate::jit::Fault;
use crate::jit::Root;
#[cfg(feature = "rocm")]
use crate::jit::cache::{self, Resolved};

/// The devices one process loads a kernel on, by ordinal.
#[cfg(feature = "rocm")]
const MAX_DEVICES: usize = 16;

pub struct Kernel {
    pub file: &'static str,
    pub name: &'static str,
    /// The unit `file` names; `None` when the crate carries no such unit.
    pub root: Option<Root>,
    #[cfg(feature = "rocm")]
    /// The load's outcome: kept once it succeeds or its unit does not
    /// compile (which a retry would only repeat); a device's refusal (no
    /// room for the module, say) is not kept, and the next fire tries
    /// again.
    loaded: [OnceLock<Result<Resolved, Fault>>; MAX_DEVICES],
}

/// Every kernel made so far: by unit, then by instantiation.
type Table = HashMap<&'static str, HashMap<Box<str>, &'static Kernel>>;

/// The kernel `name` of the unit `file`: the same reference for every call
/// with the same two.
pub fn kernel(file: &'static str, name: &str) -> &'static Kernel {
    made(file, name, || Root::of(file))
}

/// The kernel `name` of `root`, a generated unit ([`Root::generated`]): the
/// same reference for every call with the same unit and name.
pub(crate) fn generated(root: Root, name: &str) -> &'static Kernel {
    made(root.file, name, || Some(root))
}

fn made(file: &'static str, name: &str, root: impl FnOnce() -> Option<Root>) -> &'static Kernel {
    static KERNELS: OnceLock<Mutex<Table>> = OnceLock::new();
    let mut table = KERNELS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("kernel table lock poisoned");
    let named = table.entry(file).or_default();
    if let Some(&found) = named.get(name) {
        return found;
    }
    let made: &'static Kernel = Box::leak(Box::new(Kernel {
        file,
        name: Box::leak(name.into()),
        root: root(),
        #[cfg(feature = "rocm")]
        loaded: [const { OnceLock::new() }; MAX_DEVICES],
    }));
    named.insert(name.into(), made);
    made
}

impl Kernel {
    /// Compiles this kernel (or reads it from the disk cache) and loads it
    /// on the current device now, so that no fire of it does: a module load
    /// waits on the device, and mid-fire that is whatever the device runs.
    pub fn load(&'static self) -> Result<(), crate::Error> {
        #[cfg(feature = "rocm")]
        {
            self.resolve()
                .map(|_| ())
                .map_err(|fault| fault.at("jit.load"))
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(crate::jit::runtimeless("jit.load"))
        }
    }

    /// Whether [`Kernel::load`] (or a fire) loaded this kernel on the
    /// current device.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        #[cfg(feature = "rocm")]
        {
            cache::current_device()
                .ok()
                .and_then(|ordinal| usize::try_from(ordinal).ok())
                .and_then(|at| self.loaded.get(at))
                .is_some_and(|slot| matches!(slot.get(), Some(Ok(_))))
        }
        #[cfg(not(feature = "rocm"))]
        {
            false
        }
    }
}

#[cfg(feature = "rocm")]
impl Kernel {
    /// This kernel loaded on the current device: compiled or read from the
    /// disk cache the first time, the same handles after.
    pub(crate) fn resolve(&'static self) -> Result<&'static Resolved, Fault> {
        let Some(root) = self.root else {
            return Err(Fault::Compile {
                unit: self.file,
                log: format!("no carried unit is named `{}`", self.file),
            });
        };
        let ordinal = cache::current_device()?;
        let Some(slot) = usize::try_from(ordinal)
            .ok()
            .and_then(|at| self.loaded.get(at))
        else {
            return Err(Fault::Compile {
                unit: self.file,
                log: format!(
                    "device {ordinal} is past the {MAX_DEVICES} devices a process loads kernels on"
                ),
            });
        };
        if let Some(kept) = slot.get() {
            return kept.as_ref().map_err(Clone::clone);
        }
        match cache::load(&root, self.name, ordinal) {
            Err(fault) if !matches!(fault, Fault::Compile { .. }) => Err(fault),
            outcome => {
                // Two threads may load at once; the first kept is every
                // caller's.
                let _ = slot.set(outcome);
                slot.get()
                    .expect("set just above")
                    .as_ref()
                    .map_err(Clone::clone)
            }
        }
    }
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", self.file, self.name)
    }
}

/// One kernel is one reference: two are the same exactly when they are.
impl PartialEq for Kernel {
    fn eq(&self, other: &Kernel) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Eq for Kernel {}
