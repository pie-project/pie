//! Per-key admissibility table (widened, memoized) and the cuttable check
//! that reads it.

use crate::record;
use crate::window::Windows;

use super::Shell;

/// One key's segmentation, memoized — [`Shell::segments`]'s value.
pub(super) struct Segmented {
    /// Not a key coordinate: a fire whose answer differs is refused a body
    /// rather than re-derived against it.
    copies: bool,
    /// `Windows::admits` widened, one entry per template region. Shared, not
    /// cloned, across the `Run`, the capture loop and `record::cuts`.
    admits: std::sync::Arc<[crate::window::Admit]>,
    /// `None` until the cutting question is first asked for this key.
    cuttable: Option<bool>,
}

impl Shell {
    /// This key's admissibility table, derived once and memoized in
    /// [`Shell::segments`]. `copies` is the one input not carried by the key;
    /// a fire whose answer disagrees is refused a body rather than
    /// re-derived. Returns the table and whether this fire is in the key's
    /// world.
    pub(super) fn segmentation(
        &mut self,
        key: &record::BodyKey,
        windows: &Windows,
        totals: model_ir::PerAxis<u32>,
        copies: bool,
    ) -> (std::sync::Arc<[crate::window::Admit]>, bool) {
        // Same `get`, so table and world can't be answered off different entries.
        let held = self
            .segments
            .get(key)
            .map(|held| (std::sync::Arc::clone(&held.admits), held.copies));
        if let Some((admits, world)) = held {
            if world != copies {
                // Key is in another world; this fire isn't served from it.
                self.cache.eager_copy_world();
                return (admits, false);
            }
            // Debug-only: catches the table drifting from a pure function of the key.
            debug_assert!(
                admits.as_ref()
                    == record::widen(
                        &self.compiled,
                        &windows.admits_axes(totals, &self.shifted, &self.lane_shifted)
                    ),
                "the admissibility table for {key} is not what this key derived \
                 before, so `Windows::admits` has grown an input the key does \
                 not carry",
            );
            return (admits, true);
        }
        // Widened here and nowhere else: one call, one table, three readers.
        let admits: std::sync::Arc<[crate::window::Admit]> =
            record::widen(
                &self.compiled,
                &windows.admits_axes(totals, &self.shifted, &self.lane_shifted),
            )
            .into();
        // Bounded: past the seat count, keep only keys still holding or
        // refused a body; the rest re-derive on next use.
        if self.segments.len() > record::MAX_BODIES * 4 {
            let cache = &self.cache;
            self.segments
                .retain(|key, _| cache.holds_body(key) || cache.body_refused(key));
        }
        self.segments.insert(key.clone(), Segmented {
            copies,
            admits: std::sync::Arc::clone(&admits),
            cuttable: None,
        });
        // First fire of a key writes the world; later fires are measured against it.
        (admits, true)
    }

    /// Is there anything left for a graph to hold? Memoized per key so a
    /// steady stream allocates no `Vec<Cut>` per fire.
    pub(super) fn cuttable(
        &mut self,
        key: &record::BodyKey,
        admits: &[crate::window::Admit],
    ) -> bool {
        if let Some(Some(held)) = self.segments.get(key).map(|seg| seg.cuttable) {
            return held;
        }
        let script = record::cuts(&self.compiled, admits);
        let verdict = match script {
            Ok(_) => true,
            Err(uncut) => {
                // Widening consumed the composition entirely; declined once per key.
                eprintln!(
                    "engine-cuda: body {key} holds nothing a graph can keep — {uncut}. \
                     This composition walks eagerly for the life of the load; \
                     `record::widen` grew its islands to the nearest legal boundary \
                     first, and `record::Uncut` names what was left."
                );
                self.cache.body_refuse(key.clone());
                false
            }
        };
        if let Some(seg) = self.segments.get_mut(key) {
            seg.cuttable = Some(verdict);
        }
        verdict
    }
}
