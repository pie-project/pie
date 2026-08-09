//! The index over the kernels the host emitted, and what a lookup can find.
//!
//! A [`ProgramRegistration`] carries a flat `Vec<EmittedKernel>`, each tagged
//! with `(kind, stage_index, region_index)`. The driver needs them by that
//! triple, so it builds a map once at bind. That is `HostEmittedKernels` in the
//! C++ and [`Emitted`] here.
//!
//! ## An empty source and an empty error are not the same as a kernel
//!
//! `EmittedKernel` has three states packed into two strings. A populated
//! `source` is a kernel. An empty `source` with a populated `error` is a
//! *deliberate* refusal — the host looked at the region, decided it could not
//! emit for it, and said why, so the driver can take its slower path knowing
//! this is a designed fallback and not a bug. The C++ knew this and has a
//! comment about it:
//!
//! > callers check `error` before `source` to preserve the fallback that the
//! > old `emit_*_msl(...) == false` return took.
//!
//! A convention enforced by a comment on the container, at a call site the
//! comment is not next to. [`Slot`] makes the three states three variants, so
//! there is no order to check them in.
//!
//! There is also a fourth state the C++ had no words for: both strings empty.
//! `find` returned that entry like any other and the caller compiled `""`.
//! [`Slot::Malformed`] names it.
//!
//! [`ProgramRegistration`]: driver_abi::plan::ProgramRegistration

use std::collections::HashMap;

use driver_abi::plan::EmittedKernel;

/// What the host left in one `(kind, stage, region)` slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot<'a> {
    /// A kernel, with its entry point.
    Kernel {
        /// The backend source.
        source: &'a str,
        /// The entry-point symbol.
        entry: &'a str,
    },
    /// The host declined to emit for this region, and said why.
    ///
    /// Take the fallback path. This is not a failure to report.
    Refused(&'a str),
    /// The host emitted nothing for this slot.
    Absent,
    /// An entry with neither a source nor a reason.
    ///
    /// Not a kernel, not a refusal, and not the absence of one — the host said
    /// something and the something is empty. The C++ handed this back as a
    /// kernel with an empty source.
    Malformed,
}

/// Two kernels claiming the same slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Duplicate {
    /// `PIE_KERNEL_*`.
    pub kind: u32,
    /// The stage.
    pub stage: u32,
    /// The region within the stage.
    pub region: u32,
}

/// The host's emitted kernels, indexed by `(kind, stage, region)`.
///
/// Borrows the registration's vector rather than copying it. The C++ stored raw
/// `const HostEmittedKernel*` into a span it was handed, which is the same
/// thing with the lifetime left to the reader.
#[derive(Debug)]
pub struct Emitted<'a> {
    slots: HashMap<(u32, u32, u32), &'a EmittedKernel>,
}

impl<'a> Emitted<'a> {
    /// Index `kernels`.
    ///
    /// # Errors
    ///
    /// [`Duplicate`] when two entries claim one slot. The C++ used
    /// `unordered_map::emplace`, which *keeps the entry already there* and
    /// discards the new one without saying so, so a host that emitted two
    /// different sources for one region got whichever came first in the
    /// vector. That is a silent choice between two kernels, made by array
    /// order, in a driver that cannot tell which one the host meant — and if
    /// the two differ at all, one of them is wrong. It is an ABI bug and it is
    /// reported.
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

    /// What the host left in one slot.
    #[must_use]
    pub fn get(&self, kind: u32, stage: u32, region: u32) -> Slot<'a> {
        let Some(kernel) = self.slots.get(&(kind, stage, region)) else {
            return Slot::Absent;
        };
        // The error is checked first, as the C++ comment asks, and here that is
        // a property of the function rather than of every caller.
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

    /// How many slots are filled.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the host emitted nothing at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SINGLETON: u32 = 0;
    const FUSED: u32 = 1;

    fn kernel(kind: u32, stage: u32, region: u32, source: &str) -> EmittedKernel {
        EmittedKernel {
            kind,
            stage_index: stage,
            region_index: region,
            entry_name: "main0".into(),
            source: source.into(),
            error: String::new(),
        }
    }

    fn refusal(kind: u32, stage: u32, region: u32, why: &str) -> EmittedKernel {
        EmittedKernel {
            kind,
            stage_index: stage,
            region_index: region,
            entry_name: String::new(),
            source: String::new(),
            error: why.into(),
        }
    }

    #[test]
    fn a_kernel_is_found_by_its_kind_stage_and_region() {
        let kernels = [kernel(SINGLETON, 2, 3, "kernel void a() {}")];
        let emitted = Emitted::index(&kernels).expect("no duplicates");
        assert_eq!(
            emitted.get(SINGLETON, 2, 3),
            Slot::Kernel {
                source: "kernel void a() {}",
                entry: "main0"
            }
        );
    }

    #[test]
    fn all_three_parts_of_the_key_matter() {
        let kernels = [kernel(SINGLETON, 2, 3, "src")];
        let emitted = Emitted::index(&kernels).expect("no duplicates");
        assert_eq!(emitted.get(FUSED, 2, 3), Slot::Absent);
        assert_eq!(emitted.get(SINGLETON, 9, 3), Slot::Absent);
        assert_eq!(emitted.get(SINGLETON, 2, 9), Slot::Absent);
    }

    #[test]
    fn a_slot_the_host_never_filled_is_absent() {
        let emitted = Emitted::index(&[]).expect("no duplicates");
        assert_eq!(emitted.get(SINGLETON, 0, 0), Slot::Absent);
        assert!(emitted.is_empty());
    }

    #[test]
    fn a_deliberate_refusal_is_not_a_kernel_and_not_an_absence() {
        let kernels = [refusal(FUSED, 0, 0, "too many bound channels")];
        let emitted = Emitted::index(&kernels).expect("no duplicates");
        assert_eq!(
            emitted.get(FUSED, 0, 0),
            Slot::Refused("too many bound channels")
        );
    }

    #[test]
    fn the_error_is_read_before_the_source_without_the_caller_choosing_to() {
        // An entry with both is still a refusal: the C++ comment says callers
        // must check `error` first, and this is where that is enforced.
        let kernels = [EmittedKernel {
            error: "declined".into(),
            ..kernel(FUSED, 0, 0, "some stale source")
        }];
        let emitted = Emitted::index(&kernels).expect("no duplicates");
        assert_eq!(emitted.get(FUSED, 0, 0), Slot::Refused("declined"));
    }

    #[test]
    fn an_entry_with_neither_a_source_nor_a_reason_is_not_compiled() {
        let kernels = [kernel(SINGLETON, 0, 0, "")];
        let emitted = Emitted::index(&kernels).expect("no duplicates");
        assert_eq!(emitted.get(SINGLETON, 0, 0), Slot::Malformed);
    }

    #[test]
    fn two_kernels_claiming_one_slot_is_reported_rather_than_resolved_by_order() {
        let kernels = [
            kernel(SINGLETON, 1, 1, "first"),
            kernel(SINGLETON, 1, 1, "second"),
        ];
        assert_eq!(
            Emitted::index(&kernels).err(),
            Some(Duplicate {
                kind: SINGLETON,
                stage: 1,
                region: 1
            })
        );
    }

    #[test]
    fn a_duplicate_refusal_is_a_duplicate_too() {
        let kernels = [kernel(FUSED, 0, 0, "src"), refusal(FUSED, 0, 0, "declined")];
        assert!(Emitted::index(&kernels).is_err());
    }

    #[test]
    fn keys_that_the_cpps_hash_conflated_are_still_distinct_entries() {
        // `KeyHash` packed `stage << 24` over a full-width `region`, so
        // (stage 1, region 0) and (stage 0, region 0x0100_0000) hashed alike.
        // The map compared full keys so it was only a slowdown, but the two
        // are different slots and must stay so.
        let kernels = [
            kernel(SINGLETON, 1, 0, "one"),
            kernel(SINGLETON, 0, 0x0100_0000, "other"),
        ];
        let emitted = Emitted::index(&kernels).expect("distinct keys");
        assert_eq!(emitted.len(), 2);
        assert_eq!(
            emitted.get(SINGLETON, 1, 0),
            Slot::Kernel {
                source: "one",
                entry: "main0"
            }
        );
        assert_eq!(
            emitted.get(SINGLETON, 0, 0x0100_0000),
            Slot::Kernel {
                source: "other",
                entry: "main0"
            }
        );
    }
}
