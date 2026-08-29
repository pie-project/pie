use std::collections::HashMap;

use engine_api::program::{EmittedKernel, KernelKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot<'a> {
    Kernel { source: &'a str, entry: &'a str },

    Refused(&'a str),

    Absent,

    Malformed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Duplicate {
    pub kind: KernelKind,

    pub stage: u32,

    pub region: u32,
}

#[derive(Debug)]
pub struct Emitted<'a> {
    slots: HashMap<(KernelKind, u32, u32), &'a EmittedKernel>,
}

impl<'a> Emitted<'a> {
    pub fn index(kernels: &'a [EmittedKernel]) -> Result<Emitted<'a>, Duplicate> {
        let mut slots = HashMap::with_capacity(kernels.len());
        for kernel in kernels {
            let key = (kernel.kind, kernel.stage_index, kernel.region_index);
            if slots.insert(key, kernel).is_some() {
                return Err(Duplicate {
                    kind: key.0,
                    stage: key.1,
                    region: key.2,
                });
            }
        }
        Ok(Emitted { slots })
    }

    #[must_use]
    pub fn get(&self, kind: KernelKind, stage: u32, region: u32) -> Slot<'a> {
        let Some(kernel) = self.slots.get(&(kind, stage, region)) else {
            return Slot::Absent;
        };

        if !kernel.error.is_empty() {
            return Slot::Refused(&kernel.error);
        }
        if kernel.source.is_empty() {
            return Slot::Malformed;
        }
        Slot::Kernel {
            source: &kernel.source,
            entry: &kernel.entry_name,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}
