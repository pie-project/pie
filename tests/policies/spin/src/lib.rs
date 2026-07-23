use plex::{Host, Policy, RouteContext, RoutePlan, State};

struct Spin;

impl Policy for Spin {
    fn route(_ctx: &RouteContext, _state: &mut State, _host: &Host) -> plex::Result<RoutePlan> {
        loop {
            core::hint::spin_loop();
        }
    }
}

plex::export_policy!(Spin);
