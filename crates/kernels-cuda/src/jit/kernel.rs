//! A kernel: one instantiation of one carried unit, made once per process.
//! Its unit and every device's load of it hang off it, so a fire finds them
//! by following a reference rather than by naming them again.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "cuda")]
use crate::jit::Fault;
use crate::jit::Root;
#[cfg(feature = "cuda")]
use crate::jit::cache::{self, Resolved};

/// The devices one process loads a kernel on, by ordinal.
#[cfg(feature = "cuda")]
const MAX_DEVICES: usize = 16;

pub struct Kernel {
    pub file: &'static str,
    pub name: &'static str,
    /// The unit `file` names; `None` when the crate carries no such unit.
    pub root: Option<Root>,
    #[cfg(feature = "cuda")]
    loaded: [OnceLock<Result<Resolved, Fault>>; MAX_DEVICES],
}

/// Every kernel made so far: by unit, then by instantiation.
type Table = HashMap<&'static str, HashMap<Box<str>, &'static Kernel>>;

/// The kernel `name` of the unit `file`: the same reference for every call
/// with the same two.
pub fn kernel(file: &'static str, name: &str) -> &'static Kernel {
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
        root: Root::of(file),
        #[cfg(feature = "cuda")]
        loaded: [const { OnceLock::new() }; MAX_DEVICES],
    }));
    named.insert(name.into(), made);
    made
}

#[cfg(feature = "cuda")]
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
        cache::bind_context()?;
        let ordinal = cache::current_device()?;
        let Some(slot) = self.loaded.get(ordinal) else {
            return Err(Fault::Device {
                call: "cudaGetDevice",
                code: i32::try_from(ordinal).unwrap_or(i32::MAX),
            });
        };
        slot.get_or_init(|| cache::load(&root, self.name))
            .as_ref()
            .map_err(Clone::clone)
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
