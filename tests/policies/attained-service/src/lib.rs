use plex::types::{
    FeedbackInput, FeedbackOutput, PolicyError, ScheduleInput, ScheduleOutput, ServiceDecision,
};

struct AttainedService;

impl plex::Policy for AttainedService {
    fn schedule(input: ScheduleInput) -> Result<ScheduleOutput, PolicyError> {
        let decisions = input
            .runnable
            .iter()
            .enumerate()
            .map(|(index, _)| ServiceDecision {
                score: -(index as f64),
                token_budget: None,
            })
            .collect();
        Ok(ScheduleOutput {
            decisions,
            mutations: Vec::new(),
        })
    }

    fn feedback(_input: FeedbackInput) -> Result<FeedbackOutput, PolicyError> {
        Ok(FeedbackOutput {
            mutations: Vec::new(),
        })
    }
}

plex::export_policy!(AttainedService);
