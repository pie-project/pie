use std::collections::BTreeSet;
use std::sync::Arc;

use wasmtime::{Store, StoreLimits, StoreLimitsBuilder};

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
    pub max_host_calls: u32,
    pub max_host_call_bytes: u64,
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
        if !self.supported_actions.contains(&method) {
            return self.report(
                InvocationFailureKind::ActionValidation,
                format!("unsupported action method {method}"),
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
        self.staged_actions.push(StagedAction { id, method, args });
        Ok(id)
    }
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
    use super::is_versioned_method;

    #[test]
    fn helper_names_require_numeric_versions() {
        assert!(is_versioned_method("pie.kv.prefetch@1"));
        assert!(!is_versioned_method(""));
        assert!(!is_versioned_method("pie.kv.prefetch"));
        assert!(!is_versioned_method("pie.kv.prefetch@v1"));
    }
}
