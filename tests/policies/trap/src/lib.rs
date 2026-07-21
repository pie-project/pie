use plex::types::{PolicyError, ScheduleInput, ScheduleOutput};

struct Trap;

impl plex::Policy for Trap {
    fn schedule(_input: ScheduleInput) -> Result<ScheduleOutput, PolicyError> {
        panic!("injected PLEX policy trap")
    }
}

plex::export_policy!(Trap);
