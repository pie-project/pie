use plex::exports::pie::plex::policy::*;

struct MalformedStateUpdate;

impl Guest for MalformedStateUpdate {
    fn admit(_: AdmitInvocation) -> Result<AdmitOutput, PolicyError> {
        Err(fallback())
    }

    fn route(input: RouteInvocation) -> Result<RouteOutput, PolicyError> {
        plex::link_host_interface();
        Ok(RouteOutput {
            plan: RoutePlan {
                decisions: vec![RouteDecision::Defer; input.context.requests.len()],
            },
            state_update: StateUpdate {
                shared: Some("{".into()),
                groups: Vec::new(),
                requests: Vec::new(),
            },
        })
    }

    fn schedule(_: ScheduleInvocation) -> Result<ScheduleOutput, PolicyError> {
        Err(fallback())
    }

    fn cache(_: CacheInvocation) -> Result<CacheOutput, PolicyError> {
        Err(fallback())
    }

    fn feedback(_: FeedbackInvocation) -> Result<FeedbackOutput, PolicyError> {
        Err(fallback())
    }
}

fn fallback() -> PolicyError {
    PolicyError {
        code: "fallback-required".into(),
        message: "operation is not implemented".into(),
        details: "{}".into(),
    }
}

plex::export!(MalformedStateUpdate with_types_in plex);
