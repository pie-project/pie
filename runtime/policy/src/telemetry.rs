use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq)]
pub struct TelemetryRecord {
    pub name: String,
    pub value: f64,
}

#[derive(Clone)]
pub(crate) struct TelemetryBuffer {
    inner: Arc<Mutex<VecDeque<TelemetryRecord>>>,
    capacity: usize,
}

impl TelemetryBuffer {
    pub(crate) fn new(capacity: usize) -> Self {
        assert!(
            capacity != 0,
            "telemetry capacity is validated by PolicyEngine"
        );
        Self {
            inner: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            capacity,
        }
    }

    pub(crate) fn push(&self, record: TelemetryRecord) {
        let mut records = self.inner.lock().unwrap();
        if records.len() == self.capacity {
            records.pop_front();
        }
        records.push_back(record);
    }

    pub(crate) fn drain(&self) -> Vec<TelemetryRecord> {
        self.inner.lock().unwrap().drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_drops_oldest_records_without_affecting_decisions() {
        let buffer = TelemetryBuffer::new(2);
        for value in [1.0, 2.0, 3.0] {
            buffer.push(TelemetryRecord {
                name: "metric".into(),
                value,
            });
        }
        assert_eq!(
            buffer
                .drain()
                .into_iter()
                .map(|record| record.value)
                .collect::<Vec<_>>(),
            vec![2.0, 3.0]
        );
    }
}
