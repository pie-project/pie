use plex::types::{PolicyError, ScheduleInput, ScheduleOutput};

struct Malformed;

impl plex::Policy for Malformed {
    fn schedule(_input: ScheduleInput) -> Result<ScheduleOutput, PolicyError> {
        Ok(ScheduleOutput {
            decisions: Vec::new(),
            mutations: Vec::new(),
        })
    }
}

plex::export_policy!(Malformed);
