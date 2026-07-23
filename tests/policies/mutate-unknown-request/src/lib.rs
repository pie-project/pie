use plex::exports::pie::plex::policy::*;

struct MutateUnknownRequest;

impl Guest for MutateUnknownRequest {
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
                shared: None,
                groups: Vec::new(),
                requests: vec![RequestStateUpdate {
                    request_id: "forged".into(),
                    fields: Some("{}".into()),
                    scratch: None,
                }],
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

plex::export!(MutateUnknownRequest with_types_in plex);
