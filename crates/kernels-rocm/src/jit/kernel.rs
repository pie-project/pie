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
            return Err(Fault::Device {
                call: "hipGetDevice",
                code: ordinal,
            });
        };
        slot.get_or_init(|| cache::load(&root, self.name, ordinal))
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

#[cfg(all(test, feature = "rocm"))]
mod tests {
    use super::{Root, generated};

    /// A unit the crate does not carry compiles with the carried headers in
    /// scope and loads on the device; the same unit and name are one kernel.
    #[test]
    fn a_generated_unit_compiles_against_the_carried_headers() {
        if crate::jit::cache::current_device().is_err() {
            return;
        }
        const TEXT: &str = "#include \"prim/device.hip\"\n\
                            namespace pie::generated_test {\n\
                            __global__ void fill(float* __restrict__ y) { y[threadIdx.x] = 1.f; }\n\
                            }\n";
        let root = Root::generated("synth/generated_test.hip", TEXT);
        assert_eq!(root.name, "synth/generated_test");
        let made = generated(root, "::pie::generated_test::fill");
        made.load().expect("the generated unit compiles and loads");
        assert!(std::ptr::eq(
            made,
            generated(root, "::pie::generated_test::fill")
        ));
    }
}
