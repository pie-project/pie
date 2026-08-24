use std::cell::RefCell;
use std::collections::BTreeSet;

#[derive(Debug, Default, Clone)]
pub struct Ledger {
    pub resolved: BTreeSet<(&'static str, String)>,

    pub refused: Vec<Refused>,

    pub took: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct Refused {
    pub root: &'static str,
    pub instantiation: String,
    pub why: String,
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "`{}` in `{}`: {}",
            self.instantiation, self.root, self.why
        )
    }
}

impl Ledger {
    #[must_use]
    pub fn ms(&self) -> f64 {
        self.took.as_secs_f64() * 1e3
    }
}

thread_local! {

    static OPEN: RefCell<Option<Ledger>> = const { RefCell::new(None) };
}

#[must_use = "a warm pass that is not closed reports nothing"]
pub struct Pass {
    outermost: bool,
}

pub fn pass() -> Pass {
    OPEN.with(|open| {
        let mut open = open.borrow_mut();
        if open.is_some() {
            return Pass { outermost: false };
        }
        *open = Some(Ledger::default());
        Pass { outermost: true }
    })
}

impl Pass {
    pub fn close(self) -> Ledger {
        let outermost = self.outermost;
        core::mem::forget(self);
        if !outermost {
            return Ledger::default();
        }
        OPEN.with(|open| open.borrow_mut().take())
            .unwrap_or_default()
    }
}

impl Drop for Pass {
    fn drop(&mut self) {
        if self.outermost {
            OPEN.with(|open| {
                open.borrow_mut().take();
            });
        }
    }
}

#[must_use]
pub fn warming() -> bool {
    OPEN.with(|open| open.borrow().is_some())
}

pub(crate) fn note(
    root: &'static str,
    instantiation: &str,
    why: Option<String>,
    took: std::time::Duration,
) {
    OPEN.with(|open| {
        let mut open = open.borrow_mut();
        let Some(ledger) = open.as_mut() else { return };
        ledger.took += took;
        match why {
            None => {
                ledger.resolved.insert((root, instantiation.to_owned()));
            }
            Some(why) => ledger.refused.push(Refused {
                root,
                instantiation: instantiation.to_owned(),
                why,
            }),
        }
    });
}

#[cfg(feature = "_cuda")]
pub(crate) fn resolve_only(root: &crate::jit::Root, instantiation: &str) {
    let _ = crate::jit::cache::resolve(root, instantiation);
}

#[cfg(not(feature = "_cuda"))]
pub(crate) fn resolve_only(root: &crate::jit::Root, instantiation: &str) {
    note(
        root.name,
        instantiation,
        Some("this build selected no CUDA runtime".to_owned()),
        std::time::Duration::ZERO,
    );
}
