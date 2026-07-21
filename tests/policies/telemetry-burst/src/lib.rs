use plex::pie::plex::telemetry;
use plex::types::{PolicyError, ScheduleInput, ScheduleOutput, ServiceDecision, TelemetryRecord};

struct TelemetryBurst;

impl plex::Policy for TelemetryBurst {
    fn schedule(input: ScheduleInput) -> Result<ScheduleOutput, PolicyError> {
        for value in [1.0, 2.0, 3.0] {
            let _ = telemetry::emit(&TelemetryRecord {
                name: "paper.schedule.score".into(),
                value,
            });
        }
        Ok(ScheduleOutput {
            decisions: input
                .runnable
                .iter()
                .map(|_| ServiceDecision {
                    score: 0.0,
                    token_budget: None,
                })
                .collect(),
            mutations: Vec::new(),
        })
    }
}

plex::export_policy!(TelemetryBurst);
