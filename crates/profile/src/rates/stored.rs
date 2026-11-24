//! A stored measurement's key: what it was measured by and on, and
//! whether this build on this device still takes it.

use super::{DENSE_SETTLED, KERNEL_CURVES, KEYED, MEASURED_ON, Rates};

/// What one stored measurement was measured by, beside the device it was
/// measured on ([`Rates::measured_on`]): the version of its measuring method
/// (bumped only when the method changes, stated beside the measuring code)
/// and each kernel it fired with its code's fingerprint in the build that
/// measured it ([`Now::code`]). A measurement whose method or any of whose
/// kernels' code differs from this build's is measured again; nothing else
/// of the build (its version, its commit, the day) keys it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MeasuredBy {
    pub method: u32,
    pub kernels: Vec<(String, u32)>,
}

impl MeasuredBy {
    /// Why a measurement by `self` is not valid under `now`, `None` where
    /// it is: its method's version moved, or a kernel's code changed or left
    /// the build.
    #[must_use]
    pub fn stale(&self, what: &str, now: &dyn Now) -> Option<String> {
        match now.method(what) {
            None => return Some(format!("`{what}` is measured by no method of this build")),
            Some(method) if method != self.method => {
                return Some(format!(
                    "its measuring method is version {method} now, {} when stored",
                    self.method
                ));
            }
            Some(_) => {}
        }
        for (kernel, code) in &self.kernels {
            match now.code(kernel) {
                None => return Some(format!("kernel `{kernel}` is not in this build")),
                Some(now) if now != *code => {
                    return Some(format!(
                        "kernel `{kernel}`'s code changed ({code:#010x} stored, {now:#010x} now)"
                    ));
                }
                Some(_) => {}
            }
        }
        None
    }

    /// The key as the load log states it: `method v<n>, <kernel> <code>, ...`.
    #[must_use]
    pub fn said(&self) -> String {
        let kernels: Vec<String> = self
            .kernels
            .iter()
            .map(|(k, c)| format!("{k} {c:#010x}"))
            .collect();
        format!("method v{}, kernels [{}]", self.method, kernels.join(", "))
    }
}

/// What a stored measurement's key is checked against: this build on this
/// device. A backend that stores measurements states it.
pub trait Now {
    /// The device as its measurements depend on it: vendor, device id and
    /// driver version.
    fn on(&self) -> String;
    /// The version of the method that measures `what` (a `[rates]` key), or
    /// `None` where this build measures no such thing.
    fn method(&self, what: &str) -> Option<u32>;
    /// The fingerprint of `kernel`'s compiled code in this build, `None`
    /// where the build carries no such kernel.
    fn code(&self, kernel: &str) -> Option<u32>;
}

/// The fingerprint of `bytes` a stored key holds: FNV-1a folded to 32 bits,
/// the same on every build and toolchain (std's hasher is not).
#[must_use]
pub fn fingerprint(bytes: &[u8]) -> u32 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (hash ^ (hash >> 32)) as u32
}

impl Rates {
    /// The key `key` was stored under, where the file states one.
    #[must_use]
    pub fn key_of(&self, key: &str) -> Option<&MeasuredBy> {
        self.measured_by
            .iter()
            .flatten()
            .find(|(rate, _)| rate == key)
            .map(|(_, by)| by)
    }

    /// A user's file's measurements (`self`, its `[rates]` alone) checked
    /// against `now`, this build on this device ([`Rates::sift_as`], said as
    /// `stored`).
    pub fn sift(&mut self, now: &dyn Now) -> Vec<String> {
        self.sift_as(now, "stored")
    }

    /// Stored measurements (`self`, a file's `[rates]` alone: the user's,
    /// `what` "stored", or a committed measured profile's, "committed")
    /// checked against `now`, this build on this device: each whose key
    /// matches is kept, to be taken with no measurement; each whose key
    /// does not (or that states none) is dropped, by itself. Measured on
    /// another device (or driver), every one is dropped. Each drop is said,
    /// a line each, with why; what is kept is said once the layers are
    /// joined ([`Rates::whence`]), by the layer that decides it. A figure
    /// dropped for its key alone (this device's) is kept as an estimate
    /// ([`Rates::stale`]).
    pub fn sift_as(&mut self, now: &dyn Now, what: &str) -> Vec<String> {
        let mut said = Vec::new();
        let on = now.on();
        let stored = KEYED.iter().any(|key| self.states(key))
            || self.kernel_curves.is_some()
            || !self.dense_settled.is_empty();
        if !stored {
            said.push(format!("{what} measurements: none for `{on}`"));
            *self = Rates::default();
            return said;
        }
        if self.measured_on.as_deref() != Some(on.as_str()) {
            said.push(format!(
                "{what} measurements: taken on `{}`, this device is `{on}`: none is taken",
                self.measured_on.as_deref().unwrap_or("no device stated")
            ));
            *self = Rates::default();
            return said;
        }
        let mut stale = Rates::default();
        for key in KEYED {
            if !self.states(key) {
                continue;
            }
            let why = match self.key_of(key) {
                None => Some("stored without a key".to_string()),
                Some(by) => by.stale(key, now),
            };
            if let Some(why) = why {
                said.push(format!(
                    "{what} `{key}`: not taken as measured, {why}; kept as an estimate"
                ));
                let figure = self.only(&[key]);
                stale = figure.over(stale);
                self.unstate(key);
            }
        }
        // A key whose rate is not stated keys nothing.
        let stated: Vec<&str> = KEYED.into_iter().filter(|key| self.states(key)).collect();
        self.measured_by = self
            .measured_by
            .take()
            .map(|by| {
                by.into_iter()
                    .filter(|(rate, _)| stated.contains(&rate.as_str()))
                    .collect::<Vec<_>>()
            })
            .filter(|by| !by.is_empty());
        if let Some(curves) = self.kernel_curves.take() {
            let mut kept = Vec::with_capacity(curves.len());
            let mut estimates = Vec::new();
            for curve in curves {
                match curve.by.stale(KERNEL_CURVES, now) {
                    None => kept.push(curve),
                    Some(why) => {
                        said.push(format!(
                            "{what} curve {}: not taken as measured, {why}; kept as an estimate",
                            curve.name()
                        ));
                        estimates.push(curve);
                    }
                }
            }
            self.kernel_curves = (!kept.is_empty()).then_some(kept);
            if !estimates.is_empty() {
                stale = Rates {
                    kernel_curves: Some(estimates),
                    ..Rates::default()
                }
                .over(stale);
            }
        }
        for settled in std::mem::take(&mut self.dense_settled) {
            match now.method(DENSE_SETTLED) {
                Some(method) if method == settled.method => {
                    self.dense_settled.push(settled);
                }
                now => said.push(format!(
                    "{what} dense settle {:#010x}: not taken, its method is version {} now, {} \
                     when stored",
                    settled.of,
                    now.map_or_else(|| "none".to_string(), |m| m.to_string()),
                    settled.method
                )),
            }
        }
        stale.measured_on = None;
        stale.measured_by = None;
        self.stale = (stale != Rates::default()).then(|| Box::new(stale));
        said
    }

    /// Each entry `self` (the rates a load takes, the user's stored over a
    /// committed profile's, each sifted: [`Rates::sift_as`]) states by its
    /// key, said once by the layer that decides it: `stored` (`user`, the
    /// user's sifted `[rates]`, from `user_from`) where the user's file
    /// states it, else `committed` (from `committed_from`). The dense
    /// settle's key is checked by the settle itself, which says which it
    /// takes.
    #[must_use]
    pub fn whence(&self, user: &Rates, user_from: &str, committed_from: &str) -> Vec<String> {
        let mut said = Vec::new();
        for key in KEYED {
            if !self.states(key) {
                continue;
            }
            let Some(by) = self.key_of(key) else {
                continue;
            };
            let from = if user.states(key) {
                format!("stored ({user_from})")
            } else {
                format!("committed ({committed_from})")
            };
            said.push(format!("`{key}`: {from}, its key matches: {}", by.said()));
        }
        for curve in self.kernel_curves.iter().flatten() {
            let mine = user
                .kernel_curves
                .iter()
                .flatten()
                .any(|c| c.id() == curve.id());
            let from = if mine {
                format!("stored ({user_from})")
            } else {
                format!("committed ({committed_from})")
            };
            said.push(format!(
                "curve {}: {from}, its key matches: {}",
                curve.name(),
                curve.by.said()
            ));
        }
        said
    }

    /// The first entry of `self` a committed measured profile could not
    /// take ([`crate::Strict::Measured`]): no device stated, or a rate (one
    /// of [`KEYED`]) stored without a key. `None` where every entry is keyed.
    #[must_use]
    pub fn unkeyed(&self) -> Option<String> {
        if self.measured_on.is_none() {
            return Some(format!("`rates.{MEASURED_ON}` names no device"));
        }
        KEYED
            .iter()
            .find(|key| self.states(key) && self.key_of(key).is_none())
            .map(|key| format!("`rates.{key}` is stored without a key"))
    }
}
