//! Typed Rust guest SDK for `pie:plex@0.6.0`.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

pub use pie_plex::v0_6::*;
pub use serde_json;
pub use wit_bindgen;

mod wire;

pub type Document = Value;
pub type Result<T> = std::result::Result<T, PolicyError>;

wit_bindgen::generate!({
    path: "wit",
    world: "plex-policy",
    pub_export_macro: true,
    generate_all,
});

#[derive(Debug, Clone, PartialEq)]
pub struct State {
    pub shared: Document,
    groups: BTreeMap<String, Group>,
    requests: BTreeMap<String, Request>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    group_id: GroupId,
    principal_id: PrincipalId,
    status: GroupStatus,
    limits: GroupLimits,
    member_count: u32,
    facts: Document,
    pub scratch: Document,
}

impl Group {
    pub fn id(&self) -> &GroupId {
        &self.group_id
    }

    pub fn principal_id(&self) -> &PrincipalId {
        &self.principal_id
    }

    pub fn status(&self) -> GroupStatus {
        self.status
    }

    pub fn limits(&self) -> &GroupLimits {
        &self.limits
    }

    pub fn member_count(&self) -> u32 {
        self.member_count
    }

    pub fn facts(&self) -> &Document {
        &self.facts
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    request: RequestRef,
    status: RequestStatus,
    facts: Document,
    pub fields: Document,
    pub scratch: Document,
}

impl Request {
    pub fn reference(&self) -> &RequestRef {
        &self.request
    }

    pub fn status(&self) -> RequestStatus {
        self.status
    }

    pub fn facts(&self) -> &Document {
        &self.facts
    }
}

impl State {
    pub fn group(&self, id: &str) -> Result<&Group> {
        self.groups.get(id).ok_or_else(|| {
            policy_error(
                "missing-group-state",
                format!("group state {id} is not in this invocation"),
            )
        })
    }

    pub fn group_mut(&mut self, id: &str) -> Result<&mut Group> {
        self.groups.get_mut(id).ok_or_else(|| {
            policy_error(
                "missing-group-state",
                format!("group state {id} is not in this invocation"),
            )
        })
    }

    pub fn group_ids(&self) -> impl Iterator<Item = &str> {
        self.groups.keys().map(String::as_str)
    }

    pub fn request(&self, id: &str) -> Result<&Request> {
        self.requests.get(id).ok_or_else(|| {
            policy_error(
                "missing-request-state",
                format!("request state {id} is not in this invocation"),
            )
        })
    }

    pub fn request_mut(&mut self, id: &str) -> Result<&mut Request> {
        self.requests.get_mut(id).ok_or_else(|| {
            policy_error(
                "missing-request-state",
                format!("request state {id} is not in this invocation"),
            )
        })
    }

    pub fn request_ids(&self) -> impl Iterator<Item = &str> {
        self.requests.keys().map(String::as_str)
    }

    fn from_policy_state(state: PolicyState) -> Result<Self> {
        validate_policy_state(&state)
            .map_err(|error| policy_error("invalid-state", error.to_string()))?;
        let groups = state
            .groups
            .into_iter()
            .map(|group| {
                (
                    group.group_id.0.clone(),
                    Group {
                        group_id: group.group_id,
                        principal_id: group.principal_id,
                        status: group.status,
                        limits: group.limits,
                        member_count: group.member_count,
                        facts: group.facts,
                        scratch: group.scratch,
                    },
                )
            })
            .collect();
        let requests = state
            .requests
            .into_iter()
            .map(|request| {
                (
                    request.request.request_id.0.clone(),
                    Request {
                        request: request.request,
                        status: request.status,
                        facts: request.facts,
                        fields: request.fields,
                        scratch: request.scratch,
                    },
                )
            })
            .collect();
        Ok(Self {
            shared: state.shared,
            groups,
            requests,
        })
    }

    fn updates_since(&self, initial: &Self) -> Result<StateUpdate> {
        require_object(&self.shared, "shared state")?;
        if self.groups.keys().collect::<BTreeSet<_>>()
            != initial.groups.keys().collect::<BTreeSet<_>>()
        {
            return Err(policy_error(
                "group-membership-mutation",
                "policy changed the group working set",
            ));
        }
        if self.requests.keys().collect::<BTreeSet<_>>()
            != initial.requests.keys().collect::<BTreeSet<_>>()
        {
            return Err(policy_error(
                "request-membership-mutation",
                "policy changed the request working set",
            ));
        }

        let shared = (self.shared != initial.shared).then(|| self.shared.clone());
        let mut groups = Vec::new();
        for (id, group) in &self.groups {
            let original = &initial.groups[id];
            if group.group_id != original.group_id
                || group.principal_id != original.principal_id
                || group.status != original.status
                || group.limits != original.limits
                || group.member_count != original.member_count
                || group.facts != original.facts
            {
                return Err(policy_error(
                    "group-structure-mutation",
                    format!("policy changed host-owned group structure for {id}"),
                ));
            }
            require_object(&group.scratch, "group scratch")?;
            if group.scratch != original.scratch {
                groups.push(GroupStateUpdate {
                    group_id: group.group_id.clone(),
                    scratch: group.scratch.clone(),
                });
            }
        }

        let mut requests = Vec::new();
        for (id, request) in &self.requests {
            let original = &initial.requests[id];
            if request.request != original.request
                || request.status != original.status
                || request.facts != original.facts
            {
                return Err(policy_error(
                    "request-structure-mutation",
                    format!("policy changed host-owned request structure for {id}"),
                ));
            }
            require_object(&request.fields, "request fields")?;
            require_object(&request.scratch, "request scratch")?;
            let fields = (request.fields != original.fields).then(|| request.fields.clone());
            let scratch = (request.scratch != original.scratch).then(|| request.scratch.clone());
            if fields.is_some() || scratch.is_some() {
                requests.push(RequestStateUpdate {
                    request_id: request.request.request_id.clone(),
                    fields,
                    scratch,
                });
            }
        }
        Ok(StateUpdate {
            shared,
            groups,
            requests,
        })
    }
}

#[derive(Debug, Default)]
pub struct Host {
    _private: (),
}

impl Host {
    pub fn query_raw(&self, method: &str, args: &Document) -> Result<Document> {
        require_object(args, "query arguments")?;
        let args_json = serde_json::to_string(args)
            .map_err(|error| policy_error("query-encode", error.to_string()))?;
        let result_json = pie::plex::host::query(method, &args_json)
            .map_err(|error| policy_error("query-failed", error))?;
        let result: Document = serde_json::from_str(&result_json)
            .map_err(|error| policy_error("query-decode", error.to_string()))?;
        require_object(&result, "query result")?;
        Ok(result)
    }

    pub fn action_raw(&self, method: &str, args: &Document) -> Result<ActionId> {
        require_object(args, "action arguments")?;
        pie::plex::host::action(
            method,
            &serde_json::to_string(args)
                .map_err(|error| policy_error("action-encode", error.to_string()))?,
        )
        .map(ActionId)
        .map_err(|error| policy_error("action-failed", error))
    }

    pub fn cancel_request(
        &self,
        request_id: &str,
        idempotency_key: &str,
        reason: Option<&str>,
    ) -> Result<ActionId> {
        let mut args = serde_json::json!({
            "request_id": request_id,
            "idempotency_key": idempotency_key,
        });
        if let Some(reason) = reason {
            args["reason"] = serde_json::json!(reason);
        }
        self.action_raw("pie.request.cancel@1", &args)
    }

    pub fn cancel_group(
        &self,
        group_id: &str,
        propagation: &str,
        idempotency_key: &str,
    ) -> Result<ActionId> {
        self.action_raw(
            "pie.group.cancel@1",
            &serde_json::json!({
                "group_id": group_id,
                "propagation": propagation,
                "idempotency_key": idempotency_key,
            }),
        )
    }

    pub fn prefetch_cache(
        &self,
        object_id: &str,
        target_id: Option<&str>,
        idempotency_key: &str,
    ) -> Result<ActionId> {
        let mut args = serde_json::json!({
            "object_id": object_id,
            "idempotency_key": idempotency_key,
        });
        if let Some(target_id) = target_id {
            args["target_id"] = serde_json::json!(target_id);
        }
        self.action_raw("pie.cache.prefetch@1", &args)
    }

    pub fn swap_cache(
        &self,
        object_id: &str,
        tier: &str,
        idempotency_key: &str,
    ) -> Result<ActionId> {
        self.action_raw(
            "pie.cache.swap@1",
            &serde_json::json!({
                "object_id": object_id,
                "tier": tier,
                "idempotency_key": idempotency_key,
            }),
        )
    }

    pub fn rebalance_request(
        &self,
        request_id: &str,
        target_id: &str,
        idempotency_key: &str,
    ) -> Result<ActionId> {
        self.action_raw(
            "pie.request.rebalance@1",
            &serde_json::json!({
                "request_id": request_id,
                "target_id": target_id,
                "idempotency_key": idempotency_key,
            }),
        )
    }
}

pub trait Policy {
    fn admit(_ctx: &AdmitContext, _state: &mut State, _host: &Host) -> Result<AdmitPlan> {
        Err(policy_error(
            "fallback-required",
            "admit is not implemented",
        ))
    }

    fn route(_ctx: &RouteContext, _state: &mut State, _host: &Host) -> Result<RoutePlan> {
        Err(policy_error(
            "fallback-required",
            "route is not implemented",
        ))
    }

    fn schedule(_ctx: &ScheduleContext, _state: &mut State, _host: &Host) -> Result<SchedulePlan> {
        Err(policy_error(
            "fallback-required",
            "schedule is not implemented",
        ))
    }

    fn cache(_ctx: &CacheContext, _state: &mut State, _host: &Host) -> Result<CachePlan> {
        Err(policy_error(
            "fallback-required",
            "cache is not implemented",
        ))
    }

    fn feedback(_ctx: &FeedbackContext, _state: &mut State, _host: &Host) -> Result<()> {
        Err(policy_error(
            "fallback-required",
            "feedback is not implemented",
        ))
    }
}

pub fn policy_error(code: impl Into<String>, message: impl Into<String>) -> PolicyError {
    PolicyError {
        code: code.into(),
        message: message.into(),
        details: serde_json::json!({}),
    }
}

#[doc(hidden)]
pub fn invoke_admit<P: Policy>(
    input: exports::pie::plex::policy::AdmitInvocation,
) -> std::result::Result<
    exports::pie::plex::policy::AdmitOutput,
    exports::pie::plex::policy::PolicyError,
> {
    link_host_interface();
    let (context, state) = wire::admit_invocation_from_wire(input)?;
    let initial = State::from_policy_state(state).map_err(wire::policy_error_to_wire)?;
    let mut state = initial.clone();
    let plan =
        P::admit(&context, &mut state, &Host::default()).map_err(wire::policy_error_to_wire)?;
    let update = state
        .updates_since(&initial)
        .map_err(wire::policy_error_to_wire)?;
    wire::admit_output_to_wire(plan, update)
}

#[doc(hidden)]
pub fn invoke_route<P: Policy>(
    input: exports::pie::plex::policy::RouteInvocation,
) -> std::result::Result<
    exports::pie::plex::policy::RouteOutput,
    exports::pie::plex::policy::PolicyError,
> {
    link_host_interface();
    let (context, state) = wire::route_invocation_from_wire(input)?;
    let initial = State::from_policy_state(state).map_err(wire::policy_error_to_wire)?;
    let mut state = initial.clone();
    let plan =
        P::route(&context, &mut state, &Host::default()).map_err(wire::policy_error_to_wire)?;
    let update = state
        .updates_since(&initial)
        .map_err(wire::policy_error_to_wire)?;
    wire::route_output_to_wire(plan, update)
}

#[doc(hidden)]
pub fn invoke_schedule<P: Policy>(
    input: exports::pie::plex::policy::ScheduleInvocation,
) -> std::result::Result<
    exports::pie::plex::policy::ScheduleOutput,
    exports::pie::plex::policy::PolicyError,
> {
    link_host_interface();
    let (context, state) = wire::schedule_invocation_from_wire(input)?;
    let initial = State::from_policy_state(state).map_err(wire::policy_error_to_wire)?;
    let mut state = initial.clone();
    let plan =
        P::schedule(&context, &mut state, &Host::default()).map_err(wire::policy_error_to_wire)?;
    let update = state
        .updates_since(&initial)
        .map_err(wire::policy_error_to_wire)?;
    wire::schedule_output_to_wire(plan, update)
}

#[doc(hidden)]
pub fn invoke_cache<P: Policy>(
    input: exports::pie::plex::policy::CacheInvocation,
) -> std::result::Result<
    exports::pie::plex::policy::CacheOutput,
    exports::pie::plex::policy::PolicyError,
> {
    link_host_interface();
    let (context, state) = wire::cache_invocation_from_wire(input)?;
    let initial = State::from_policy_state(state).map_err(wire::policy_error_to_wire)?;
    let mut state = initial.clone();
    let plan =
        P::cache(&context, &mut state, &Host::default()).map_err(wire::policy_error_to_wire)?;
    let update = state
        .updates_since(&initial)
        .map_err(wire::policy_error_to_wire)?;
    wire::cache_output_to_wire(plan, update)
}

#[doc(hidden)]
pub fn invoke_feedback<P: Policy>(
    input: exports::pie::plex::policy::FeedbackInvocation,
) -> std::result::Result<
    exports::pie::plex::policy::FeedbackOutput,
    exports::pie::plex::policy::PolicyError,
> {
    link_host_interface();
    let (context, state) = wire::feedback_invocation_from_wire(input)?;
    let initial = State::from_policy_state(state).map_err(wire::policy_error_to_wire)?;
    let mut state = initial.clone();
    P::feedback(&context, &mut state, &Host::default()).map_err(wire::policy_error_to_wire)?;
    let update = state
        .updates_since(&initial)
        .map_err(wire::policy_error_to_wire)?;
    wire::feedback_output_to_wire(update)
}

#[doc(hidden)]
#[inline(never)]
pub fn link_host_interface() {
    core::hint::black_box(
        pie::plex::host::query as fn(&str, &str) -> std::result::Result<String, String>,
    );
    core::hint::black_box(
        pie::plex::host::action as fn(&str, &str) -> std::result::Result<u64, String>,
    );
}

fn require_object(value: &Document, field: &'static str) -> Result<()> {
    if !value.is_object() {
        return Err(policy_error(
            "document-not-object",
            format!("{field} must be a JSON object"),
        ));
    }
    Ok(())
}

#[macro_export]
macro_rules! export_policy {
    ($policy:ty) => {
        struct __PlexGuest;

        impl $crate::exports::pie::plex::policy::Guest for __PlexGuest {
            fn admit(
                input: $crate::exports::pie::plex::policy::AdmitInvocation,
            ) -> std::result::Result<
                $crate::exports::pie::plex::policy::AdmitOutput,
                $crate::exports::pie::plex::policy::PolicyError,
            > {
                $crate::invoke_admit::<$policy>(input)
            }

            fn route(
                input: $crate::exports::pie::plex::policy::RouteInvocation,
            ) -> std::result::Result<
                $crate::exports::pie::plex::policy::RouteOutput,
                $crate::exports::pie::plex::policy::PolicyError,
            > {
                $crate::invoke_route::<$policy>(input)
            }

            fn schedule(
                input: $crate::exports::pie::plex::policy::ScheduleInvocation,
            ) -> std::result::Result<
                $crate::exports::pie::plex::policy::ScheduleOutput,
                $crate::exports::pie::plex::policy::PolicyError,
            > {
                $crate::invoke_schedule::<$policy>(input)
            }

            fn cache(
                input: $crate::exports::pie::plex::policy::CacheInvocation,
            ) -> std::result::Result<
                $crate::exports::pie::plex::policy::CacheOutput,
                $crate::exports::pie::plex::policy::PolicyError,
            > {
                $crate::invoke_cache::<$policy>(input)
            }

            fn feedback(
                input: $crate::exports::pie::plex::policy::FeedbackInvocation,
            ) -> std::result::Result<
                $crate::exports::pie::plex::policy::FeedbackOutput,
                $crate::exports::pie::plex::policy::PolicyError,
            > {
                $crate::invoke_feedback::<$policy>(input)
            }
        }

        $crate::export!(__PlexGuest with_types_in $crate);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> PolicyState {
        PolicyState {
            shared: serde_json::json!({"calls": 1}),
            groups: vec![GroupState {
                group_id: "G".into(),
                principal_id: "tenant".into(),
                status: GroupStatus::Open,
                limits: GroupLimits {
                    max_members: 2,
                    max_scratch_bytes: 1024,
                },
                member_count: 1,
                facts: serde_json::json!({
                    "group_id": "G",
                    "principal_id": "tenant"
                }),
                scratch: serde_json::json!({}),
            }],
            requests: vec![RequestState {
                request: RequestRef {
                    request_id: "A".into(),
                    generation_id: 0,
                    group_id: Some("G".into()),
                    principal_id: "tenant".into(),
                },
                status: RequestStatus::Active,
                facts: serde_json::json!({
                    "request_id": "A",
                    "generation_id": 0,
                    "group_id": "G",
                    "principal_id": "tenant"
                }),
                fields: serde_json::json!({}),
                scratch: serde_json::json!({}),
            }],
        }
    }

    #[test]
    fn state_tracks_only_mutable_namespaces() {
        let initial = State::from_policy_state(state()).unwrap();
        let mut changed = initial.clone();
        changed.group_mut("G").unwrap().scratch["service"] = serde_json::json!(4);
        changed.request_mut("A").unwrap().fields["priority"] = serde_json::json!(2);
        let update = changed.updates_since(&initial).unwrap();
        assert_eq!(update.groups[0].scratch["service"], 4);
        assert_eq!(update.requests[0].fields.as_ref().unwrap()["priority"], 2);
        assert!(update.requests[0].scratch.is_none());
    }

    #[test]
    fn state_rejects_host_owned_mutation() {
        let initial = State::from_policy_state(state()).unwrap();
        let mut changed = initial.clone();
        changed.requests.get_mut("A").unwrap().status = RequestStatus::Completed;
        assert!(changed.updates_since(&initial).is_err());
    }
}
