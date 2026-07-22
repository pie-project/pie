//! JSON guest SDK for the breaking PLEX v0.5 proof of concept.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

pub use serde_json;
pub use wit_bindgen;

pub type Document = Value;
pub type Result<T> = std::result::Result<T, String>;
pub type ActionId = u64;

wit_bindgen::generate!({
    path: "wit",
    world: "plex-policy",
    pub_export_macro: true,
    generate_all,
});

#[derive(Debug, Clone, PartialEq)]
pub struct State {
    pub shared: Document,
    requests: BTreeMap<String, Request>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    facts: Document,
    pub fields: Document,
    pub scratch: Document,
}

impl Request {
    pub fn facts(&self) -> &Document {
        &self.facts
    }
}

impl State {
    pub fn request(&self, id: &str) -> Result<&Request> {
        self.requests
            .get(id)
            .ok_or_else(|| format!("request state {id} is not in this invocation"))
    }

    pub fn request_mut(&mut self, id: &str) -> Result<&mut Request> {
        self.requests
            .get_mut(id)
            .ok_or_else(|| format!("request state {id} is not in this invocation"))
    }

    pub fn request_ids(&self) -> impl Iterator<Item = &str> {
        self.requests.keys().map(String::as_str)
    }

    fn parse(snapshot_json: &str) -> Result<Self> {
        let snapshot: Document = serde_json::from_str(snapshot_json)
            .map_err(|error| format!("invalid state snapshot JSON: {error}"))?;
        let snapshot = snapshot
            .as_object()
            .ok_or("state snapshot must be a top-level JSON object")?;
        require_exact_keys(snapshot, &["requests", "shared"], "state snapshot")?;
        let shared = snapshot["shared"]
            .as_object()
            .ok_or("state snapshot shared must be a JSON object")?;
        let requests = snapshot["requests"]
            .as_object()
            .ok_or("state snapshot requests must be a JSON object")?;

        let mut parsed = BTreeMap::new();
        for (id, request) in requests {
            if id.is_empty() {
                return Err("state snapshot request ID must not be empty".into());
            }
            let request = request
                .as_object()
                .ok_or_else(|| format!("state snapshot request {id} must be a JSON object"))?;
            require_exact_keys(
                request,
                &["facts", "fields", "scratch"],
                "state snapshot request",
            )?;
            let facts = request["facts"]
                .as_object()
                .ok_or_else(|| format!("state snapshot request {id} facts must be an object"))?;
            if facts.get("logical_request_id").and_then(Value::as_str) != Some(id)
                || facts.get("generation_id").and_then(Value::as_u64).is_none()
            {
                return Err(format!(
                    "state snapshot request {id} has invalid host identity facts"
                ));
            }
            let fields = request["fields"]
                .as_object()
                .ok_or_else(|| format!("state snapshot request {id} fields must be an object"))?;
            let scratch = request["scratch"]
                .as_object()
                .ok_or_else(|| format!("state snapshot request {id} scratch must be an object"))?;
            parsed.insert(
                id.clone(),
                Request {
                    facts: Value::Object(facts.clone()),
                    fields: Value::Object(fields.clone()),
                    scratch: Value::Object(scratch.clone()),
                },
            );
        }

        Ok(Self {
            shared: Value::Object(shared.clone()),
            requests: parsed,
        })
    }

    fn updates_since(&self, initial: &Self) -> Result<Document> {
        if !self.shared.is_object() {
            return Err("shared state must remain a JSON object".into());
        }
        if self.requests.keys().collect::<BTreeSet<_>>()
            != initial.requests.keys().collect::<BTreeSet<_>>()
        {
            return Err("policy changed the request working set".into());
        }

        let mut updates = Map::new();
        if self.shared != initial.shared {
            updates.insert("shared".into(), self.shared.clone());
        }

        let mut request_updates = Map::new();
        for (id, request) in &self.requests {
            let original = &initial.requests[id];
            if request.facts != original.facts {
                return Err(format!("policy changed host-owned facts for request {id}"));
            }
            if !request.fields.is_object() {
                return Err(format!("request {id} fields must remain a JSON object"));
            }
            if !request.scratch.is_object() {
                return Err(format!("request {id} scratch must remain a JSON object"));
            }
            if request.fields != original.fields || request.scratch != original.scratch {
                request_updates.insert(
                    id.clone(),
                    serde_json::json!({
                        "fields": request.fields,
                        "scratch": request.scratch,
                    }),
                );
            }
        }
        if !request_updates.is_empty() {
            updates.insert("requests".into(), Value::Object(request_updates));
        }
        Ok(Value::Object(updates))
    }
}

#[derive(Debug, Default)]
pub struct Host {
    _private: (),
}

impl Host {
    pub fn query_raw(&self, method: &str, args: &Document) -> Result<Document> {
        let args_json = serde_json::to_string(args)
            .map_err(|error| format!("failed to serialize query arguments: {error}"))?;
        let result_json = pie::plex::host::query(method, &args_json)
            .map_err(|error| format!("host query {method} failed: {error}"))?;
        serde_json::from_str(&result_json)
            .map_err(|error| format!("host query {method} returned invalid JSON: {error}"))
    }

    pub fn action_raw(&self, method: &str, args: &Document) -> Result<ActionId> {
        pie::plex::host::action(
            method,
            &serde_json::to_string(args)
                .map_err(|error| format!("failed to serialize action arguments: {error}"))?,
        )
        .map_err(|error| format!("host action {method} failed: {error}"))
    }

    pub fn kv_lookup(&self, request_id: &str, target: &str) -> Result<Document> {
        self.query_raw(
            "pie.kv.lookup@1",
            &serde_json::json!({
                "request_id": request_id,
                "target": target,
            }),
        )
    }

    pub fn cluster_capacity(&self, model: &str) -> Result<Document> {
        self.query_raw(
            "pie.cluster.capacity@1",
            &serde_json::json!({"model": model}),
        )
    }

    pub fn model_config(&self) -> Result<Document> {
        self.query_raw("pie.model.config@1", &serde_json::json!({}))
    }

    pub fn now_ms(&self) -> Result<u64> {
        let value = self.query_raw("pie.clock.now@1", &serde_json::json!({}))?;
        value
            .as_u64()
            .or_else(|| value.get("now_ms").and_then(Value::as_u64))
            .ok_or_else(|| "pie.clock.now@1 did not return an unsigned timestamp".into())
    }

    pub fn prefetch_kv(&self, request_id: &str, target: &str) -> Result<ActionId> {
        self.action_raw(
            "pie.kv.prefetch@1",
            &serde_json::json!({
                "request_id": request_id,
                "target": target,
            }),
        )
    }

    pub fn preempt(&self, request_id: &str) -> Result<ActionId> {
        self.action_raw(
            "pie.schedule.preempt@1",
            &serde_json::json!({"request_id": request_id}),
        )
    }

    pub fn replicate(&self, request_id: &str, targets: &[&str]) -> Result<ActionId> {
        self.action_raw(
            "pie.route.replicate@1",
            &serde_json::json!({
                "request_id": request_id,
                "targets": targets,
            }),
        )
    }

    pub fn set_retention(&self, request_id: &str, ttl_ms: u64) -> Result<ActionId> {
        self.action_raw(
            "pie.retention.set@1",
            &serde_json::json!({
                "request_id": request_id,
                "ttl_ms": ttl_ms,
            }),
        )
    }

    pub fn arm_timer(&self, request_id: &str, delay_ms: u64) -> Result<ActionId> {
        self.action_raw(
            "pie.timer.arm@1",
            &serde_json::json!({
                "request_id": request_id,
                "delay_ms": delay_ms,
            }),
        )
    }
}

pub trait Policy {
    fn route(ctx: &Document, state: &mut State, host: &Host) -> Result<Document> {
        let _ = (ctx, state, host);
        Err("fallback-required".into())
    }

    fn admit(ctx: &Document, state: &mut State, host: &Host) -> Result<Document> {
        let _ = (ctx, state, host);
        Err("fallback-required".into())
    }

    fn schedule(ctx: &Document, state: &mut State, host: &Host) -> Result<Document> {
        let _ = (ctx, state, host);
        Err("fallback-required".into())
    }

    fn evict(ctx: &Document, state: &mut State, host: &Host) -> Result<Document> {
        let _ = (ctx, state, host);
        Err("fallback-required".into())
    }

    fn feedback(ctx: &Document, state: &mut State, host: &Host) -> Result<Document> {
        let _ = (ctx, state, host);
        Err("fallback-required".into())
    }
}

pub fn invoke(
    input: exports::pie::plex::policy::Invocation,
    operation: impl FnOnce(&Document, &mut State, &Host) -> Result<Document>,
) -> Result<exports::pie::plex::policy::PolicyOutput> {
    link_host_interface();
    invoke_with(input, operation)
}

fn invoke_with(
    input: exports::pie::plex::policy::Invocation,
    operation: impl FnOnce(&Document, &mut State, &Host) -> Result<Document>,
) -> Result<exports::pie::plex::policy::PolicyOutput> {
    let context: Document = serde_json::from_str(&input.context_json)
        .map_err(|error| format!("invalid JSON context: {error}"))?;
    if !context.is_object() {
        return Err("context must be a top-level JSON object".into());
    }

    let initial = State::parse(&input.state_json)?;
    let mut state = initial.clone();
    let result = operation(&context, &mut state, &Host::default())
        .map_err(|error| format!("policy error: {error}"))?;
    let updates = state.updates_since(&initial)?;
    Ok(exports::pie::plex::policy::PolicyOutput {
        result_json: serde_json::to_string(&result)
            .map_err(|error| format!("failed to serialize policy result: {error}"))?,
        state_update_json: serde_json::to_string(&updates)
            .map_err(|error| format!("failed to serialize state updates: {error}"))?,
    })
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

fn require_exact_keys(object: &Map<String, Value>, expected: &[&str], name: &str) -> Result<()> {
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!("{name} contains invalid namespaces"));
    }
    Ok(())
}

#[macro_export]
macro_rules! export_policy {
    ($policy:ty) => {
        struct __PlexGuest;

        impl $crate::exports::pie::plex::policy::Guest for __PlexGuest {
            fn route(
                input: $crate::exports::pie::plex::policy::Invocation,
            ) -> $crate::Result<$crate::exports::pie::plex::policy::PolicyOutput> {
                $crate::invoke(input, <$policy as $crate::Policy>::route)
            }

            fn admit(
                input: $crate::exports::pie::plex::policy::Invocation,
            ) -> $crate::Result<$crate::exports::pie::plex::policy::PolicyOutput> {
                $crate::invoke(input, <$policy as $crate::Policy>::admit)
            }

            fn schedule(
                input: $crate::exports::pie::plex::policy::Invocation,
            ) -> $crate::Result<$crate::exports::pie::plex::policy::PolicyOutput> {
                $crate::invoke(input, <$policy as $crate::Policy>::schedule)
            }

            fn evict(
                input: $crate::exports::pie::plex::policy::Invocation,
            ) -> $crate::Result<$crate::exports::pie::plex::policy::PolicyOutput> {
                $crate::invoke(input, <$policy as $crate::Policy>::evict)
            }

            fn feedback(
                input: $crate::exports::pie::plex::policy::Invocation,
            ) -> $crate::Result<$crate::exports::pie::plex::policy::PolicyOutput> {
                $crate::invoke(input, <$policy as $crate::Policy>::feedback)
            }
        }

        $crate::export!(__PlexGuest with_types_in $crate);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> String {
        serde_json::json!({
            "shared": {"routes": 1},
            "requests": {
                "L": {
                    "facts": {
                        "logical_request_id": "L",
                        "generation_id": 0,
                        "attained_service": 8
                    },
                    "fields": {
                        "body": {"prompt": "hello"},
                        "metadata": {}
                    },
                    "scratch": {}
                }
            }
        })
        .to_string()
    }

    fn input(context_json: &str) -> exports::pie::plex::policy::Invocation {
        exports::pie::plex::policy::Invocation {
            context_json: context_json.into(),
            state_json: snapshot(),
        }
    }

    #[test]
    fn wrapper_returns_explicit_result_and_changed_state() {
        let output = invoke_with(input(r#"{"request_id":"L"}"#), |ctx, state, _host| {
            let request = state.request_mut(ctx["request_id"].as_str().unwrap())?;
            request.fields["body"]["prompt"] = serde_json::json!("rewritten");
            request.scratch["admissions"] = serde_json::json!(1);
            state.shared["routes"] = serde_json::json!(2);
            Ok(serde_json::json!({"decision": "accept"}))
        })
        .unwrap();

        assert_eq!(
            serde_json::from_str::<Document>(&output.result_json).unwrap(),
            serde_json::json!({"decision": "accept"})
        );
        let updates: Document = serde_json::from_str(&output.state_update_json).unwrap();
        assert_eq!(updates["shared"], serde_json::json!({"routes": 2}));
        assert_eq!(
            updates["requests"]["L"]["fields"]["body"]["prompt"],
            "rewritten"
        );
        assert_eq!(updates["requests"]["L"]["scratch"]["admissions"], 1);
        assert!(updates["requests"]["L"].get("facts").is_none());
    }

    #[test]
    fn wrapper_returns_empty_update_and_clear_policy_errors() {
        let output =
            invoke_with(input("{}"), |_ctx, _state, _host| Ok(serde_json::json!({}))).unwrap();
        assert_eq!(output.state_update_json, "{}");

        let error = invoke_with(input("{}"), |_ctx, _state, _host| {
            Err("fallback-required".into())
        })
        .unwrap_err();
        assert_eq!(error, "policy error: fallback-required");
    }

    #[test]
    fn state_exposes_facts_read_only_and_unique_request_ids() {
        let state = State::parse(&snapshot()).unwrap();
        assert_eq!(state.request("L").unwrap().facts()["attained_service"], 8);
        assert_eq!(state.request_ids().collect::<Vec<_>>(), vec!["L"]);
        assert!(state.request("M").is_err());
    }

    #[test]
    fn wrapper_rejects_bad_context_and_snapshot_shapes() {
        assert_eq!(
            invoke_with(input("[]"), |_ctx, _state, _host| {
                Ok(serde_json::json!({}))
            })
            .unwrap_err(),
            "context must be a top-level JSON object"
        );
        let mut invalid = input("{}");
        invalid.state_json = r#"{"shared":{},"requests":{},"extra":{}}"#.into();
        assert!(
            invoke_with(invalid, |_ctx, _state, _host| { Ok(serde_json::json!({})) })
                .unwrap_err()
                .contains("invalid namespaces")
        );
    }
}
