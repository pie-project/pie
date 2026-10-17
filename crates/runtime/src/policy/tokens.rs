//! Processes' own token counts (`session.set-tokens`), by PLEX id. The host
//! never sees a prompt and a decode samples on the device, so the counts are
//! the process's word. An exited process's are kept for the last [`KEEP`]
//! exits.

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;

use crate::inferlet::process::ProcessId;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    pub prompt: u64,
    pub generated: u64,
}

const KEEP: usize = 1024;

#[derive(Default)]
struct Table {
    live: HashMap<ProcessId, Counts>,
    exited: HashMap<ProcessId, Counts>,
    first_token_ms: HashMap<ProcessId, u64>,
    order: VecDeque<ProcessId>,
}

static TABLE: Mutex<Option<Table>> = Mutex::new(None);

pub(crate) fn set(process: ProcessId, counts: Counts) {
    let mut table = TABLE.lock();
    let table = table.get_or_insert_with(Table::default);
    if counts.generated > 0 {
        table.first_token_ms.entry(process).or_insert_with(now_ms);
    }
    table.live.insert(process, counts);
}

pub fn first_token_ms(process: ProcessId) -> Option<u64> {
    TABLE.lock().as_ref()?.first_token_ms.get(&process).copied()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

pub fn get(process: ProcessId) -> Option<Counts> {
    let table = TABLE.lock();
    let table = table.as_ref()?;
    table
        .live
        .get(&process)
        .or_else(|| table.exited.get(&process))
        .copied()
}

pub(crate) fn exit(process: ProcessId) {
    let mut table = TABLE.lock();
    let Some(table) = table.as_mut() else {
        return;
    };
    let Some(counts) = table.live.remove(&process) else {
        return;
    };
    if table.exited.insert(process, counts).is_none() {
        table.order.push_back(process);
    }
    while table.order.len() > KEEP {
        if let Some(oldest) = table.order.pop_front() {
            table.exited.remove(&oldest);
            table.first_token_ms.remove(&oldest);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid() -> ProcessId {
        ProcessId::new_v4()
    }

    fn counts(prompt: u64, generated: u64) -> Counts {
        Counts { prompt, generated }
    }

    #[test]
    fn counts_outlive_their_process_for_the_last_exits() {
        let (first, counted, silent) = (pid(), pid(), pid());
        set(first, counts(1, 0));
        set(counted, counts(10, 0));
        set(counted, counts(10, 3));
        assert_eq!(get(counted), Some(counts(10, 3)));
        assert_eq!(get(silent), None);
        assert!(first_token_ms(counted).is_some(), "it has generated");
        assert_eq!(first_token_ms(first), None, "it has not");

        exit(first);
        exit(counted);
        exit(silent);
        assert_eq!(get(counted), Some(counts(10, 3)));
        for _ in 0..KEEP {
            let other = pid();
            set(other, Counts::default());
            exit(other);
        }
        assert_eq!(get(first), None, "the oldest exits are forgotten");
        assert_eq!(get(counted), None);
    }
}
