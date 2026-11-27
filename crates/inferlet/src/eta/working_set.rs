use super::*;

/// One sequence's model state. Slots below [`WorkingSet::folded`] are frozen;
/// the rest is a buffer a pass may write again.
pub struct WorkingSet {
    pub(super) ws: Rc<wit_ws::WorkingSet>,
}

impl WorkingSet {
    pub fn new() -> WorkingSet {
        WorkingSet {
            ws: Rc::new(wit_ws::WorkingSet::new()),
        }
    }

    /// The handle the launch moved in under `name`.
    pub fn take(name: &str) -> Option<WorkingSet> {
        use crate::pie::inferlet::session::{Handle, HandleKind, take};
        match take(name, HandleKind::WorkingSet)? {
            Handle::WorkingSet(ws) => Some(WorkingSet { ws: Rc::new(ws) }),
            _ => None,
        }
    }

    /// Hands this set back to the client under the launch's `name`: every
    /// pass bound to it must be dropped first.
    pub fn give(self, name: &str) -> Result<(), String> {
        let ws = Rc::try_unwrap(self.ws)
            .map_err(|_| format!("`{name}` is still bound to a forward pass"))?;
        crate::pie::inferlet::session::give(
            name,
            crate::pie::inferlet::session::Handle::WorkingSet(ws),
        )
    }

    /// Lets passes write slots below `end`: needed only by a pass whose slots
    /// are computed on the device.
    pub fn reserve(&self, end: u32) -> Result<(), String> {
        self.ws.reserve(end)
    }

    /// The frozen length; while a device-counted fold is in flight, the length
    /// before it.
    pub fn folded(&self) -> u32 {
        self.ws.folded()
    }

    /// A child sharing the frozen part, with a copy of the buffer. It
    /// shares as this set does.
    pub fn fork(&self, on: &Pipeline) -> Result<WorkingSet, String> {
        Ok(WorkingSet {
            ws: Rc::new(self.ws.fork(&on.wit)?),
        })
    }

    /// Publishes this set's frozen slots within `scope`, and lets its passes
    /// reuse what that scope holds: prefixes others computed, or are
    /// computing. Unasked, a set is private. Ask before its first pass.
    pub fn share(&self, scope: ShareScope) -> Result<(), String> {
        self.ws.share(&scope)
    }

    /// [`WorkingSet::new`], shared within `scope` ([`WorkingSet::share`]).
    pub fn shared(scope: ShareScope) -> Result<WorkingSet, String> {
        let ws = WorkingSet::new();
        ws.share(scope)?;
        Ok(ws)
    }
}

impl Default for WorkingSet {
    fn default() -> Self {
        WorkingSet::new()
    }
}

/// A `WorkingSet` in a typed input is the handle the client wrote there, as
/// `{"$handle": "<name>"}`.
impl<'de> serde::Deserialize<'de> for WorkingSet {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let handle = crate::handle_marker(deserializer)?;
        WorkingSet::take(&handle).ok_or_else(|| {
            serde::de::Error::custom(format!("the launch moved in no handle `{handle}`"))
        })
    }
}

/// How many more slots each lane freezes after a fire writes.
#[derive(Clone, Copy, Default)]
pub enum Fold<'a> {
    /// None: everything stays in the buffer.
    #[default]
    None,
    /// Everything the lane has written.
    All,
    /// Per lane (or one for all), possibly computed on the device; lanes
    /// sharing a working set fold by the first one's.
    Count(&'a Channel),
}

/// Where a pass's rows land and what they read. Left out, rows append, are
/// roped at their slot and read every slot up to their own.
#[derive(Clone, Copy, Default)]
pub struct StateGeometry<'a> {
    /// Per row, the RoPE position.
    pub positions: Option<&'a Channel>,
    /// Per row, the slot it writes; a lane's rows write consecutive slots.
    pub slots: Option<&'a Channel>,
    /// Per lane, how many slots the lane reads.
    pub kv_len: Option<&'a Channel>,
    pub fold: Fold<'a>,
    /// `[lanes]` u32: per lane, the beam of the fire before it continues; the
    /// engine keeps the beams' own page lists (`forward.wit`, `crate::beam`).
    pub parents: Option<&'a Channel>,
    /// With `parents`, the slot past the search's last row: the depth its
    /// page lists are laid for (`crate::beam::BeamGroup::geometry`).
    pub parents_end: Option<u32>,
}

impl StateGeometry<'_> {
    /// Append and freeze everything written.
    pub fn folding() -> StateGeometry<'static> {
        StateGeometry {
            fold: Fold::All,
            ..StateGeometry::default()
        }
    }
}
