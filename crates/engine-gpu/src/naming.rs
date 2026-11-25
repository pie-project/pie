//! A load's own kernels, named before any is built: the walk a fire makes,
//! run against a sink that fires nothing. Each kernel a dispatch asks for is
//! one name; a host that picks by the fire's rows names every kernel its
//! widths allow ([`kernels_slang::Naming`]), so the set a load's walks leave
//! is every pipeline a fire of it can ask for.

use std::cell::RefCell;
use std::collections::BTreeSet;

use kernels_slang::{ArgValue, Error, Fire, Target};

/// One kernel as a pipeline is built from it: the unit its module is
/// compiled from and its key at the tier it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Kernel {
    pub unit: &'static str,
    pub key: &'static str,
}

impl Kernel {
    /// The kernel `fire` asks for; refused where it is not instantiated at
    /// the fire's tier, in the words a backend's sink refuses it.
    pub fn of(fire: Fire) -> Result<Kernel, Error> {
        let key = kernels_slang::key(fire.entrypoint, fire.tier).ok_or_else(|| Error::Backend {
            op: fire.entrypoint,
            detail: format!("not instantiated at the {} tier", fire.tier.name()),
        })?;
        Ok(Kernel {
            unit: fire.file,
            key,
        })
    }
}

/// The ways one host goes, open: the dispatches before it, and the longest
/// way closed so far.
#[derive(Debug)]
struct Fork {
    before: u32,
    longest: u32,
}

#[derive(Debug, Default)]
struct Named {
    kernels: BTreeSet<Kernel>,
    /// The dispatches of the walk so far, a forked host at its open way.
    dispatches: u32,
    forks: Vec<Fork>,
}

/// The naming sink and the set it leaves.
#[derive(Debug)]
pub struct Names {
    target: Target,
    named: RefCell<Named>,
    /// The most rows a fire of the load may bring ([`Names::bound_rows`]).
    most_rows: std::cell::Cell<u32>,
}

impl Names {
    #[must_use]
    pub fn new(target: Target) -> Names {
        Names {
            target,
            named: RefCell::new(Named::default()),
            most_rows: std::cell::Cell::new(0),
        }
    }

    /// States the most rows a fire of the load may bring, which a host
    /// whose dispatches grow with the rows counts at.
    pub fn bound_rows(&self, rows: u32) {
        self.most_rows.set(rows);
    }

    /// Every kernel named so far, in order.
    #[must_use]
    pub fn kernels(&self) -> Vec<Kernel> {
        self.named.borrow().kernels.iter().copied().collect()
    }

    #[must_use]
    pub fn holds(&self, unit: &str, key: &str) -> bool {
        self.named
            .borrow()
            .kernels
            .iter()
            .any(|kernel| kernel.unit == unit && kernel.key == key)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.named.borrow().kernels.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.named.borrow().kernels.is_empty()
    }

    /// Names `fire`'s kernel beside the walks': one a fire makes outside its
    /// walk.
    pub fn name(&self, fire: Fire) -> Result<(), Error> {
        self.named.borrow_mut().kernels.insert(Kernel::of(fire)?);
        Ok(())
    }

    /// The dispatches made since the last take: one walk's, each forked
    /// host at its longest way.
    pub fn take_dispatches(&self) -> u32 {
        let mut named = self.named.borrow_mut();
        debug_assert!(named.forks.is_empty(), "a walk ended inside a forked host");
        std::mem::take(&mut named.dispatches)
    }

    /// Drops what a walk that a host refused left: its dispatches and the
    /// ways it had open.
    pub fn abandon(&self) {
        let mut named = self.named.borrow_mut();
        named.forks.clear();
        named.dispatches = 0;
    }

    /// The dispatches made since the last take, the walk still going: read
    /// between hosts, where none is forked.
    #[must_use]
    pub fn dispatched(&self) -> u32 {
        let named = self.named.borrow();
        debug_assert!(named.forks.is_empty(), "read inside a forked host");
        named.dispatches
    }
}

impl kernels_slang::Encode for Names {
    fn fire(&self, fire: Fire, _args: &[ArgValue]) -> Result<(), Error> {
        let kernel = Kernel::of(fire)?;
        let mut named = self.named.borrow_mut();
        named.kernels.insert(kernel);
        named.dispatches += 1;
        Ok(())
    }

    fn absent(&self) -> Result<ArgValue, Error> {
        Ok(ArgValue::Buffer(crate::handles::NIL))
    }

    fn target(&self) -> Target {
        self.target
    }

    fn naming(&self) -> Option<&dyn kernels_slang::Naming> {
        Some(self)
    }
}

impl kernels_slang::Naming for Names {
    fn either(&self, alternatives: &[Fire]) -> Result<(), Error> {
        let kernels = alternatives
            .iter()
            .map(|fire| Kernel::of(*fire))
            .collect::<Result<Vec<Kernel>, Error>>()?;
        let mut named = self.named.borrow_mut();
        // A fire makes one of them.
        named.dispatches += u32::from(!kernels.is_empty());
        named.kernels.extend(kernels);
        Ok(())
    }

    fn fork(&self) {
        let mut named = self.named.borrow_mut();
        let before = named.dispatches;
        named.forks.push(Fork { before, longest: 0 });
    }

    fn or(&self) {
        let mut named = self.named.borrow_mut();
        let at = named.dispatches;
        let fork = named
            .forks
            .last_mut()
            .expect("a way ends inside the fork that opened it");
        fork.longest = fork.longest.max(at - fork.before);
        named.dispatches = fork.before;
    }

    fn most_rows(&self) -> u32 {
        self.most_rows.get()
    }

    fn join(&self) {
        self.or();
        let mut named = self.named.borrow_mut();
        let fork = named.forks.pop().expect("`or` just read the open fork");
        named.dispatches = fork.before + fork.longest;
    }
}

/// A class of lanes the kernels refuse a fire of: the load's naming fire of
/// the class alone was refused, in `why`. The load says it once and stands;
/// no fire of the class is served, so a bind whose lane only such classes
/// serve is refused ([`refuses`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unservable {
    pub class: usize,
    pub why: String,
}

/// Why an instance that reads `reads` (each an output and the lane of its
/// attachment that reads it) is not bound, where the load's `unservable`
/// classes are why. A lane's fire lands it in a class that writes every
/// output it reads; where each such class of `classes` is one the kernels
/// refuse, its first fire would be refused as the load's naming fire was,
/// and the bind is refused instead: by the lane, its outputs, the classes
/// and the kernels' own words.
#[must_use]
pub fn refuses(
    unservable: &[Unservable],
    classes: &poem_ir::LaneClasses,
    reads: &[(String, u32)],
) -> Option<String> {
    let mut lanes: Vec<u32> = reads.iter().map(|(_, lane)| *lane).collect();
    lanes.sort_unstable();
    lanes.dedup();
    lanes.into_iter().find_map(|lane| {
        let names: Vec<&str> = reads
            .iter()
            .filter(|(_, of)| *of == lane)
            .map(|(name, _)| name.as_str())
            .collect();
        let writers: Vec<usize> = (0..classes.classes.len())
            .filter(|&class| {
                let outputs = &classes.classes[class].outputs;
                names
                    .iter()
                    .all(|name| outputs.iter().any(|decl| decl.name == *name))
            })
            .collect();
        let refused: Vec<&Unservable> = writers
            .iter()
            .filter_map(|class| unservable.iter().find(|held| held.class == *class))
            .collect();
        let first = refused.first()?;
        (refused.len() == writers.len()).then(|| {
            format!(
                "lane {lane} reads {names:?}, which only the lane classes {writers:?} write, \
                 and this load holds each unservable: the kernels refuse a fire of it ({})",
                first.why
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(outputs: &[&str]) -> poem_ir::LaneClass {
        poem_ir::LaneClass {
            words: Vec::new(),
            reads_kv: false,
            reads_state: false,
            embeds: true,
            wide: false,
            outputs: outputs
                .iter()
                .map(|name| poem_ir::OutputDecl {
                    name: (*name).to_string(),
                    axis: poem_ir::InputAxis::Lane,
                    width: 1,
                    dtype: poem_ir::Dtype::F32,
                    layers: 1,
                    readout: false,
                    tail: 0,
                    planes: Vec::new(),
                })
                .collect(),
            patch: None,
        }
    }

    /// A bind is refused where every class that writes what a lane of it
    /// reads is one the load holds unservable: by the lane, its outputs,
    /// the classes and the kernel's words. A lane that a class the kernels
    /// admit serves is bound, whatever else the load refused.
    #[test]
    fn a_bind_only_unservable_classes_serve_is_refused_by_name() {
        let classes = poem_ir::LaneClasses {
            mask: 0,
            classes: vec![
                class(&["logits"]),
                class(&["logits", "attn.scores"]),
                class(&["logits", "attn.scores"]),
            ],
        };
        let why = "`attn.score_capture` would not enqueue: the 512-wide head is wider than \
                   the 256-wide dot this kernel is stamped for";
        let held = |of: &[usize]| -> Vec<Unservable> {
            of.iter()
                .map(|&class| Unservable {
                    class,
                    why: why.to_string(),
                })
                .collect()
        };
        let reads = |of: &[(&str, u32)]| -> Vec<(String, u32)> {
            of.iter()
                .map(|(name, lane)| ((*name).to_string(), *lane))
                .collect()
        };
        let scores = held(&[1, 2]);

        // The lane of the attachment that reads the scores is the one named.
        assert_eq!(
            refuses(
                &scores,
                &classes,
                &reads(&[("logits", 0), ("attn.scores", 1)])
            ),
            Some(format!(
                "lane 1 reads [\"attn.scores\"], which only the lane classes [1, 2] write, \
                 and this load holds each unservable: the kernels refuse a fire of it ({why})"
            ))
        );
        // A lane that reads them beside what every class writes lands in
        // the same classes.
        assert!(
            refuses(
                &scores,
                &classes,
                &reads(&[("logits", 0), ("attn.scores", 0)])
            )
            .is_some()
        );
        // A class the kernels admit writes these.
        assert_eq!(refuses(&scores, &classes, &reads(&[("logits", 0)])), None);
        assert_eq!(
            refuses(&held(&[2]), &classes, &reads(&[("attn.scores", 0)])),
            None
        );
        // No class refused, and an output no class writes: neither is this
        // refusal's.
        assert_eq!(refuses(&[], &classes, &reads(&[("attn.scores", 0)])), None);
        assert_eq!(refuses(&scores, &classes, &reads(&[("hidden", 0)])), None);
    }
}
