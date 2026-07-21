use plex::types::{PolicyError, ScheduleInput, ScheduleOutput};

struct Spin;

impl plex::Policy for Spin {
    fn schedule(_input: ScheduleInput) -> Result<ScheduleOutput, PolicyError> {
        loop {
            core::hint::spin_loop();
        }
    }
}

plex::export_policy!(Spin);
