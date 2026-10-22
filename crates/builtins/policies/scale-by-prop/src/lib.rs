//! Keeps the cluster at the worker counts launches ask for: the last ask of
//! each shape stands, installed whole as the scale table every call.

use std::collections::BTreeMap;

use policy::{Id, Scale, Subject, cluster, prop};

#[derive(Default)]
struct ScaleByProp {
    /// (model, engine) to workers, as last asked.
    asked: BTreeMap<(Option<String>, String), u32>,
}

impl cluster::Policy for ScaleByProp {
    fn on_arrived(&mut self, process: &Id) {
        let of = Subject::Process(process.clone());
        let (Some(workers), Some(engine)) = (
            prop(&of, "scale-workers").int(),
            prop(&of, "scale-engine").text().map(str::to_string),
        ) else {
            return;
        };
        let model = prop(&of, "scale-model").text().map(str::to_string);
        let workers = u32::try_from(workers.max(0)).unwrap_or(u32::MAX);
        self.asked.insert((model, engine), workers);
    }

    fn plan(&mut self, _now_ms: u64) {
        if self.asked.is_empty() {
            return;
        }
        let table: Vec<Scale> = self
            .asked
            .iter()
            .map(|((model, engine), workers)| Scale {
                model: model.clone(),
                engine: engine.clone(),
                workers: *workers,
            })
            .collect();
        cluster::install_scale(&table);
    }
}

policy::cluster!(ScaleByProp);
