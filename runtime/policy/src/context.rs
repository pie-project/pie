use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use wasmtime::{Store, StoreLimits, StoreLimitsBuilder};

use pie_plex::v0_6::{MechanicId, MechanicKind, Operation, STANDARD_MECHANICS};

use crate::bindings::pie::plex::host;
use crate::bindings_v0_6::pie::plex::host as host_v0_6;
use crate::error::{InvocationFailure, InvocationFailureKind};
use crate::host::{QueryHandler, StagedAction};

pub(crate) const MAX_CORE_INSTANCES_PER_INVOCATION: u32 = 4;
pub(crate) const MAX_MEMORIES_PER_INVOCATION: u32 = 1;
pub(crate) const MAX_TABLES_PER_INVOCATION: u32 = 4;
pub(crate) const MAX_TABLE_ELEMENTS: usize = 1024;

pub(crate) struct InvocationContext {
    limits: StoreLimits,
    staged_actions: Vec<StagedAction>,
    query_handler: Arc<dyn QueryHandler>,
    supported_actions: Arc<BTreeSet<String>>,
    authorization: Option<InvocationAuthorizationV0_6>,
    action_keys: BTreeMap<(String, String), u64>,
    max_host_calls: u32,
    max_host_call_bytes: u64,
    host_calls: u32,
    host_call_bytes: u64,
    fatal_failure: Option<InvocationFailure>,
    reported_failure: Option<InvocationFailure>,
}

pub(crate) struct InvocationContextConfig {
    pub memory_bytes: usize,
    pub query_handler: Arc<dyn QueryHandler>,
    pub supported_actions: Arc<BTreeSet<String>>,
    pub authorization: Option<InvocationAuthorizationV0_6>,
    pub max_host_calls: u32,
    pub max_host_call_bytes: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct InvocationAuthorizationV0_6 {
    pub operation: Operation,
    pub mechanics: BTreeSet<MechanicId>,
    pub request_ids: BTreeSet<String>,
    pub group_ids: BTreeSet<String>,
    pub cache_object_ids: BTreeSet<String>,
    pub target_ids: BTreeSet<String>,
}

impl InvocationContext {
    pub(crate) fn store(engine: &wasmtime::Engine, config: InvocationContextConfig) -> Store<Self> {
        let limits = StoreLimitsBuilder::new()
            .memory_size(config.memory_bytes)
            .table_elements(MAX_TABLE_ELEMENTS)
            .instances(MAX_CORE_INSTANCES_PER_INVOCATION as usize)
            .tables(MAX_TABLES_PER_INVOCATION as usize)
            .memories(MAX_MEMORIES_PER_INVOCATION as usize)
            .build();
        let mut store = Store::new(
            engine,
            Self {
                limits,
                staged_actions: Vec::new(),
                query_handler: config.query_handler,
                supported_actions: config.supported_actions,
                authorization: config.authorization,
                action_keys: BTreeMap::new(),
                max_host_calls: config.max_host_calls,
                max_host_call_bytes: config.max_host_call_bytes,
                host_calls: 0,
                host_call_bytes: 0,
                fatal_failure: None,
                reported_failure: None,
            },
        );
        store.limiter(|context| &mut context.limits);
        store
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<StagedAction>, InvocationFailure> {
        if let Some(failure) = self.fatal_failure.take() {
            return Err(failure);
        }
        Ok(std::mem::take(&mut self.staged_actions))
    }

    pub(crate) fn take_reported_failure(
        &mut self,
        policy_error: &str,
    ) -> Option<InvocationFailure> {
        if let Some(failure) = self.fatal_failure.take() {
            return Some(failure);
        }
        self.reported_failure
            .take()
            .filter(|failure| policy_error.contains(&failure.message))
    }

    fn begin_call(
        &mut self,
        request_bytes: usize,
        kind: InvocationFailureKind,
    ) -> Result<(), String> {
        self.host_calls = self.host_calls.saturating_add(1);
        self.host_call_bytes = self.host_call_bytes.saturating_add(request_bytes as u64);
        if self.host_calls > self.max_host_calls {
            return self.fail(
                kind,
                format!(
                    "policy exceeded the host-call limit of {}",
                    self.max_host_calls
                ),
            );
        }
        if self.host_call_bytes > self.max_host_call_bytes {
            return self.fail(
                kind,
                format!(
                    "policy exceeded the host-call byte limit of {}",
                    self.max_host_call_bytes
                ),
            );
        }
        Ok(())
    }

    fn finish_call(
        &mut self,
        response_bytes: usize,
        kind: InvocationFailureKind,
    ) -> Result<(), String> {
        self.host_call_bytes = self.host_call_bytes.saturating_add(response_bytes as u64);
        if self.host_call_bytes > self.max_host_call_bytes {
            return self.fail(
                kind,
                format!(
                    "policy exceeded the host-call byte limit of {}",
                    self.max_host_call_bytes
                ),
            );
        }
        Ok(())
    }

    fn fail<T>(&mut self, kind: InvocationFailureKind, message: String) -> Result<T, String> {
        if self.fatal_failure.is_none() {
            self.fatal_failure = Some(InvocationFailure::new(kind, message.clone()));
        }
        Err(message)
    }

    fn report<T>(&mut self, kind: InvocationFailureKind, message: String) -> Result<T, String> {
        self.reported_failure = Some(InvocationFailure::new(kind, message.clone()));
        Err(message)
    }
}

impl host::Host for InvocationContext {
    fn query(&mut self, method: String, args_json: String) -> Result<String, String> {
        self.begin_call(
            method.len().saturating_add(args_json.len()),
            InvocationFailureKind::Query,
        )?;
        if !is_versioned_method(&method) {
            return self.report(
                InvocationFailureKind::Query,
                format!("query method {method:?} must be a non-empty versioned name"),
            );
        }

        let args: pie_plex::Document = match serde_json::from_str(&args_json) {
            Ok(args) => args,
            Err(error) => {
                return self.report(
                    InvocationFailureKind::Query,
                    format!("query arguments are invalid JSON: {error}"),
                );
            }
        };
        if !args.is_object() {
            return self.report(
                InvocationFailureKind::Query,
                "query arguments must be a JSON object".into(),
            );
        }
        let result = match self.query_handler.query(&method, &args) {
            Ok(result) => result,
            Err(error) => {
                return self.report(InvocationFailureKind::Query, error.to_string());
            }
        };
        let result_json = match serde_json::to_string(&result) {
            Ok(result) => result,
            Err(error) => {
                return self.report(
                    InvocationFailureKind::Query,
                    format!("failed to serialize query result: {error}"),
                );
            }
        };
        self.finish_call(result_json.len(), InvocationFailureKind::Query)?;
        Ok(result_json)
    }

    fn action(&mut self, method: String, args_json: String) -> Result<u64, String> {
        self.begin_call(
            method.len().saturating_add(args_json.len()),
            InvocationFailureKind::ActionValidation,
        )?;
        if !is_versioned_method(&method) {
            return self.report(
                InvocationFailureKind::ActionValidation,
                format!("action method {method:?} must be a non-empty versioned name"),
            );
        }
        let args: pie_plex::Document = match serde_json::from_str(&args_json) {
            Ok(args) => args,
            Err(error) => {
                return self.report(
                    InvocationFailureKind::ActionValidation,
                    format!("action arguments are invalid JSON: {error}"),
                );
            }
        };
        if !args.is_object() {
            return self.report(
                InvocationFailureKind::ActionValidation,
                "action arguments must be a JSON object".into(),
            );
        }
        let idempotency_key = match &self.authorization {
            Some(authorization) => match validate_standard_action(authorization, &method, &args) {
                Ok(key) => key,
                Err((kind, message)) => return self.report(kind, message),
            },
            None => None,
        };
        if !self.supported_actions.contains(&method) {
            return self.report(
                if self.authorization.is_some() {
                    InvocationFailureKind::UnsupportedMechanic
                } else {
                    InvocationFailureKind::ActionValidation
                },
                format!("unsupported action method {method}"),
            );
        }
        if let Some(key) = &idempotency_key
            && let Some(id) = self.action_keys.get(&(method.clone(), key.clone()))
        {
            return Ok(*id);
        }
        let id = match u64::try_from(self.staged_actions.len()) {
            Ok(id) => id,
            Err(_) => {
                return self.report(
                    InvocationFailureKind::ActionValidation,
                    "action count exceeds the invocation-local ID range".into(),
                );
            }
        };
        self.finish_call(
            std::mem::size_of::<u64>(),
            InvocationFailureKind::ActionValidation,
        )?;
        if let Some(key) = idempotency_key {
            self.action_keys.insert((method.clone(), key), id);
        }
        self.staged_actions.push(StagedAction { id, method, args });
        Ok(id)
    }
}

fn validate_standard_action(
    authorization: &InvocationAuthorizationV0_6,
    method: &str,
    args: &pie_plex::Document,
) -> Result<Option<String>, (InvocationFailureKind, String)> {
    let Some(mechanic) = STANDARD_MECHANICS
        .iter()
        .find(|mechanic| mechanic.method == Some(method))
    else {
        return Ok(args
            .get("idempotency_key")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned));
    };
    let mechanic_id = MechanicId::from(mechanic.id);
    if !authorization.mechanics.contains(&mechanic_id) {
        return Err((
            InvocationFailureKind::UnsupportedMechanic,
            format!(
                "action {method} requires negotiated mechanic {}",
                mechanic.id
            ),
        ));
    }
    if mechanic.kind != MechanicKind::Action
        || !mechanic.operations.contains(&authorization.operation)
    {
        return Err((
            InvocationFailureKind::ActionValidation,
            format!(
                "action {method} is not valid from operation {:?}",
                authorization.operation
            ),
        ));
    }
    let object = args.as_object().expect("action arguments were validated");
    let idempotency_key = required_string(object, "idempotency_key")?;
    if idempotency_key.len() > 128 {
        return Err((
            InvocationFailureKind::ActionValidation,
            "action idempotency_key exceeds 128 bytes".into(),
        ));
    }
    match method {
        "pie.request.cancel@1" => {
            let request_id = required_string(object, "request_id")?;
            require_authorized(&authorization.request_ids, request_id, "request", method)?;
            optional_string(object, "reason", 256)?;
            optional_u64(object, "expires_at_ms")?;
            require_only(
                object,
                &["request_id", "idempotency_key", "reason", "expires_at_ms"],
            )?;
        }
        "pie.group.cancel@1" => {
            let group_id = required_string(object, "group_id")?;
            require_authorized(&authorization.group_ids, group_id, "group", method)?;
            match required_string(object, "propagation")? {
                "group-only" | "live-requests" => {}
                value => {
                    return Err((
                        InvocationFailureKind::ActionValidation,
                        format!("invalid group cancellation propagation {value:?}"),
                    ));
                }
            }
            optional_string(object, "reason", 256)?;
            optional_u64(object, "expires_at_ms")?;
            require_only(
                object,
                &[
                    "group_id",
                    "propagation",
                    "idempotency_key",
                    "reason",
                    "expires_at_ms",
                ],
            )?;
        }
        "pie.cache.prefetch@1" => {
            let object_id = required_string(object, "object_id")?;
            require_authorized(
                &authorization.cache_object_ids,
                object_id,
                "cache object",
                method,
            )?;
            if let Some(target_id) = optional_string(object, "target_id", 128)?
                && !authorization.target_ids.is_empty()
                && !authorization.target_ids.contains(target_id)
            {
                return Err((
                    InvocationFailureKind::ActionValidation,
                    format!("target {target_id:?} is outside the invocation"),
                ));
            }
            if let Some(urgency) = optional_u64(object, "urgency")?
                && urgency > 1000
            {
                return Err((
                    InvocationFailureKind::ActionValidation,
                    "cache prefetch urgency exceeds 1000".into(),
                ));
            }
            optional_u64(object, "expires_at_ms")?;
            require_only(
                object,
                &[
                    "object_id",
                    "target_id",
                    "urgency",
                    "expires_at_ms",
                    "idempotency_key",
                ],
            )?;
        }
        "pie.cache.swap@1" => {
            let object_id = required_string(object, "object_id")?;
            require_authorized(
                &authorization.cache_object_ids,
                object_id,
                "cache object",
                method,
            )?;
            let tier = required_string(object, "tier")?;
            if !valid_atom(tier, 64) {
                return Err((
                    InvocationFailureKind::ActionValidation,
                    format!("invalid cache tier {tier:?}"),
                ));
            }
            require_only(object, &["object_id", "tier", "idempotency_key"])?;
        }
        "pie.request.rebalance@1" => {
            let request_id = required_string(object, "request_id")?;
            require_authorized(&authorization.request_ids, request_id, "request", method)?;
            let target_id = required_string(object, "target_id")?;
            if !authorization.target_ids.is_empty() && !authorization.target_ids.contains(target_id)
            {
                return Err((
                    InvocationFailureKind::ActionValidation,
                    format!("target {target_id:?} is outside the invocation"),
                ));
            }
            optional_string(object, "reason", 256)?;
            require_only(
                object,
                &["request_id", "target_id", "idempotency_key", "reason"],
            )?;
        }
        _ => unreachable!("standard action registry and validator diverged"),
    }
    Ok(Some(idempotency_key.to_owned()))
}

fn required_string<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &'static str,
) -> Result<&'a str, (InvocationFailureKind, String)> {
    object
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            (
                InvocationFailureKind::ActionValidation,
                format!("action field {field} must be a non-empty string"),
            )
        })
}

fn optional_string<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &'static str,
    maximum: usize,
) -> Result<Option<&'a str>, (InvocationFailureKind, String)> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    let value = value.as_str().ok_or_else(|| {
        (
            InvocationFailureKind::ActionValidation,
            format!("action field {field} must be a string"),
        )
    })?;
    if value.is_empty() || value.len() > maximum {
        return Err((
            InvocationFailureKind::ActionValidation,
            format!("action field {field} must contain 1-{maximum} bytes"),
        ));
    }
    Ok(Some(value))
}

fn optional_u64(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &'static str,
) -> Result<Option<u64>, (InvocationFailureKind, String)> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    value.as_u64().map(Some).ok_or_else(|| {
        (
            InvocationFailureKind::ActionValidation,
            format!("action field {field} must be an unsigned integer"),
        )
    })
}

fn require_authorized(
    allowed: &BTreeSet<String>,
    value: &str,
    kind: &'static str,
    method: &str,
) -> Result<(), (InvocationFailureKind, String)> {
    if allowed.contains(value) {
        Ok(())
    } else {
        Err((
            InvocationFailureKind::ActionValidation,
            format!("{kind} {value:?} is not authorized for {method}"),
        ))
    }
}

fn require_only(
    object: &serde_json::Map<String, serde_json::Value>,
    allowed: &[&str],
) -> Result<(), (InvocationFailureKind, String)> {
    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err((
            InvocationFailureKind::ActionValidation,
            format!("action arguments contain unsupported field {field:?}"),
        ));
    }
    Ok(())
}

fn valid_atom(value: &str, maximum: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum
        && value.bytes().enumerate().all(|(index, byte)| match byte {
            b'a'..=b'z' => true,
            b'0'..=b'9' | b'.' | b'_' | b'-' => index > 0,
            _ => false,
        })
}

impl host_v0_6::Host for InvocationContext {
    fn query(&mut self, method: String, args: String) -> Result<String, String> {
        <Self as host::Host>::query(self, method, args)
    }

    fn action(&mut self, method: String, args: String) -> Result<u64, String> {
        <Self as host::Host>::action(self, method, args)
    }
}

pub(crate) fn is_versioned_method(method: &str) -> bool {
    method.rsplit_once('@').is_some_and(|(name, version)| {
        !name.is_empty() && !version.is_empty() && version.bytes().all(|byte| byte.is_ascii_digit())
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pie_plex::v0_6::{MechanicId, Operation};
    use serde_json::json;

    use super::{
        InvocationAuthorizationV0_6, InvocationFailureKind, is_versioned_method,
        validate_standard_action,
    };

    #[test]
    fn helper_names_require_numeric_versions() {
        assert!(is_versioned_method("pie.kv.prefetch@1"));
        assert!(!is_versioned_method(""));
        assert!(!is_versioned_method("pie.kv.prefetch"));
        assert!(!is_versioned_method("pie.kv.prefetch@v1"));
    }

    #[test]
    fn standard_actions_validate_schema_authority_and_operation() {
        let authorization = InvocationAuthorizationV0_6 {
            operation: Operation::Schedule,
            mechanics: BTreeSet::from([
                MechanicId::from("request.cancel@1"),
                MechanicId::from("group.cancel@1"),
            ]),
            request_ids: BTreeSet::from(["A".into()]),
            group_ids: BTreeSet::from(["G".into()]),
            cache_object_ids: BTreeSet::new(),
            target_ids: BTreeSet::new(),
        };
        assert_eq!(
            validate_standard_action(
                &authorization,
                "pie.request.cancel@1",
                &json!({"request_id": "A", "idempotency_key": "cancel-A"})
            )
            .unwrap(),
            Some("cancel-A".into())
        );
        assert!(matches!(
            validate_standard_action(
                &authorization,
                "pie.request.cancel@1",
                &json!({
                    "request_id": "B",
                    "idempotency_key": "cancel-B",
                    "unknown": true
                })
            ),
            Err((InvocationFailureKind::ActionValidation, _))
        ));

        let route = InvocationAuthorizationV0_6 {
            operation: Operation::Route,
            ..authorization
        };
        assert!(matches!(
            validate_standard_action(
                &route,
                "pie.request.cancel@1",
                &json!({"request_id": "A", "idempotency_key": "cancel-A"})
            ),
            Err((InvocationFailureKind::ActionValidation, _))
        ));
    }
}
