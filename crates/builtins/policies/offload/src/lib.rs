//! Sends a long pass's prefill to a peer: a rule for every process that asks
//! for one by its `offload` prop, and `cost` from 1024 tokens for the rest.

use policy::{Offload, OffloadMode, Subject, SubjectKind, candidates, prop, worker};

const MIN_TOKENS: u32 = 1024;

#[derive(Default)]
struct OffloadLong;

fn rule(mode: OffloadMode, min_tokens: u32) -> Offload {
    Offload {
        mode,
        min_tokens,
        executors: Vec::new(),
    }
}

impl worker::Policy for OffloadLong {
    fn plan(&mut self, _now_ms: u64) {
        let entries: Vec<_> = candidates(SubjectKind::Process)
            .into_iter()
            .filter_map(|process| {
                let of = Subject::Process(process.clone());
                let mode = match prop(&of, "offload").text()? {
                    "never" => OffloadMode::Never,
                    "always" => OffloadMode::Always,
                    "cost" => OffloadMode::Cost,
                    _ => return None,
                };
                let min = prop(&of, "offload-min-tokens")
                    .int()
                    .and_then(|n| u32::try_from(n).ok())
                    .unwrap_or(if mode == OffloadMode::Always {
                        0
                    } else {
                        MIN_TOKENS
                    });
                Some((process, rule(mode, min)))
            })
            .collect();
        let default = rule(OffloadMode::Cost, MIN_TOKENS);
        let _ = worker::install_offload(&entries, Some(&default));
    }
}

policy::worker!(OffloadLong);
