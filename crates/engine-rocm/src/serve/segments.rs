use crate::record;
use crate::window::Windows;

use super::Shell;

pub(super) struct Segmented {
    copies: bool,
    admits: std::sync::Arc<[crate::window::Admit]>,
    cuttable: Option<bool>,
}

impl Shell {
    pub(super) fn segmentation(
        &mut self,
        key: &record::BodyKey,
        windows: &Windows,
        totals: poem_ir::PerAxis<u32>,
        copies: bool,
    ) -> (std::sync::Arc<[crate::window::Admit]>, bool) {
        let held = self
            .segments
            .get(key)
            .map(|held| (std::sync::Arc::clone(&held.admits), held.copies));
        if let Some((admits, world)) = held {
            if world != copies {
                self.cache.eager_copy_world();
                return (admits, false);
            }
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
        let admits: std::sync::Arc<[crate::window::Admit]> = record::widen(
            &self.compiled,
            &windows.admits_axes(totals, &self.shifted, &self.lane_shifted),
        )
        .into();
        if self.segments.len() > record::MAX_BODIES * 4 {
            let cache = &self.cache;
            self.segments
                .retain(|key, _| cache.holds_body(key) || cache.body_refused(key));
        }
        self.segments.insert(
            key.clone(),
            Segmented {
                copies,
                admits: std::sync::Arc::clone(&admits),
                cuttable: None,
            },
        );
        (admits, true)
    }

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
