use plex::{Host, Policy, RouteContext, RoutePlan, State, policy_error};

struct MutateRequestFacts;

impl Policy for MutateRequestFacts {
    fn route(_ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        Err(policy_error(
            "facts-immutable",
            "v0.6 state-update has no host-facts mutation field",
        ))
    }
}

plex::export_policy!(MutateRequestFacts);
