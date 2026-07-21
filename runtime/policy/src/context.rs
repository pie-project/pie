use pie_plex::{MapHandle, PolicyLimits, TypedValue};
use wasmtime::component::{Access, HasSelf};
use wasmtime::{AsContext, AsContextMut, Store, StoreLimits, StoreLimitsBuilder};

use crate::bindings::pie::plex::{maps, telemetry, types};
use crate::convert;
use crate::engine::InvocationPermit;
use crate::error::{InvocationFailure, InvocationFailureKind};
use crate::maps::{InvocationTransaction, MapAccessError};
use crate::telemetry::{TelemetryBuffer, TelemetryRecord};

pub(crate) struct InvocationContext {
    limits: StoreLimits,
    transaction: Option<InvocationTransaction>,
    map_calls_remaining: u32,
    map_bytes_remaining: u64,
    telemetry_records_remaining: u32,
    telemetry_bytes_remaining: u64,
    telemetry: TelemetryBuffer,
    _permit: InvocationPermit,
    failure: Option<InvocationFailure>,
}

impl InvocationContext {
    pub(crate) fn store(
        engine: &wasmtime::Engine,
        memory_bytes: usize,
        transaction: InvocationTransaction,
        policy_limits: &PolicyLimits,
        telemetry: TelemetryBuffer,
        permit: InvocationPermit,
    ) -> Store<InvocationContext> {
        let limits = StoreLimitsBuilder::new()
            .memory_size(memory_bytes)
            .table_elements(1024)
            .instances(4)
            .tables(4)
            .memories(1)
            .build();
        let mut store = Store::new(
            engine,
            Self {
                limits,
                transaction: Some(transaction),
                map_calls_remaining: policy_limits.map_calls,
                map_bytes_remaining: policy_limits.map_bytes,
                telemetry_records_remaining: policy_limits.telemetry_records,
                telemetry_bytes_remaining: policy_limits.telemetry_bytes,
                telemetry,
                _permit: permit,
                failure: None,
            },
        );
        store.limiter(|context| &mut context.limits);
        store
    }

    pub(crate) fn take_transaction(&mut self) -> InvocationTransaction {
        self.transaction
            .take()
            .expect("invocation transaction is present exactly once")
    }

    pub(crate) fn take_failure(&mut self) -> Option<InvocationFailure> {
        self.failure.take()
    }

    fn fail(&mut self, kind: InvocationFailureKind, message: impl Into<String>) {
        if self.failure.is_none() {
            self.failure = Some(InvocationFailure::new(kind, message));
        }
    }
}

impl types::Host for InvocationContext {}
impl maps::Host for InvocationContext {}
impl telemetry::Host for InvocationContext {}

impl maps::HostWithStore<InvocationContext> for HasSelf<InvocationContext> {
    fn get(
        mut host: Access<InvocationContext, Self>,
        handle: types::MapHandle,
        key: types::MapKey,
    ) -> anyhow::Result<Result<Option<types::MapValue>, types::MapError>> {
        let key = convert::contract_map_key(key);
        let key_bytes = u64::try_from(key.payload_len()).unwrap_or(u64::MAX);
        charge_host_fuel(&mut host, 100u64.saturating_add(key_bytes))?;

        let value_budget = {
            let context = host.data_mut();
            if context.map_calls_remaining == 0 {
                context.fail(
                    InvocationFailureKind::MapLimitExceeded,
                    "policy exceeded its map call limit",
                );
                return Ok(Err(types::MapError::CallLimit));
            }

            context.map_calls_remaining -= 1;
            if key_bytes > context.map_bytes_remaining {
                context.fail(
                    InvocationFailureKind::MapLimitExceeded,
                    "policy exceeded its map byte limit",
                );
                return Ok(Err(types::MapError::ByteLimit));
            }
            context.map_bytes_remaining -= key_bytes;
            context.map_bytes_remaining
        };
        let fuel_budget = host.as_context().get_fuel()?;
        let lookup = {
            let context = host.data_mut();
            let transaction = context
                .transaction
                .as_mut()
                .expect("transaction exists during map call");
            transaction.get_bounded(
                MapHandle::new(handle.value),
                &key,
                value_budget.min(fuel_budget),
            )
        };
        let value = match lookup {
            Ok(value) => value,
            Err(error @ MapAccessError::InvocationByteLimit { actual, .. })
                if actual > value_budget =>
            {
                host.data_mut()
                    .fail(InvocationFailureKind::MapLimitExceeded, error.to_string());
                return Ok(Err(types::MapError::ByteLimit));
            }
            Err(MapAccessError::InvocationByteLimit { .. }) => {
                host.as_context_mut().set_fuel(0)?;
                host.data_mut().fail(
                    InvocationFailureKind::FuelExhausted,
                    "policy exhausted fuel while reading a map value",
                );
                anyhow::bail!("policy exhausted fuel while reading a map value");
            }
            Err(error) => return Ok(Err(map_error(error))),
        };

        let value_bytes = value
            .as_ref()
            .map(TypedValue::payload_len)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(0);
        {
            let context = host.data_mut();
            context.map_bytes_remaining -= value_bytes;
        }
        charge_host_fuel(&mut host, value_bytes)?;
        Ok(Ok(value.map(convert::wit_map_value)))
    }
}

impl telemetry::HostWithStore<InvocationContext> for HasSelf<InvocationContext> {
    fn emit(
        mut host: Access<InvocationContext, Self>,
        record: types::TelemetryRecord,
    ) -> anyhow::Result<Result<(), types::TelemetryError>> {
        let name_bytes = u64::try_from(record.name.len()).unwrap_or(u64::MAX);
        charge_host_fuel(&mut host, 50u64.saturating_add(name_bytes))?;
        if record.name.is_empty() || !record.value.is_finite() {
            return Ok(Err(types::TelemetryError::InvalidRecord));
        }
        let charged = name_bytes.saturating_add(8);
        let context = host.data_mut();
        if context.telemetry_records_remaining == 0 {
            return Ok(Err(types::TelemetryError::RecordLimit));
        }
        if charged > context.telemetry_bytes_remaining {
            return Ok(Err(types::TelemetryError::ByteLimit));
        }
        context.telemetry_records_remaining -= 1;
        context.telemetry_bytes_remaining -= charged;
        context.telemetry.push(TelemetryRecord {
            name: record.name,
            value: record.value,
        });
        Ok(Ok(()))
    }
}

fn charge_host_fuel(
    host: &mut Access<'_, InvocationContext, HasSelf<InvocationContext>>,
    amount: u64,
) -> anyhow::Result<()> {
    let remaining = host.as_context().get_fuel()?;
    if remaining < amount {
        host.as_context_mut().set_fuel(0)?;
        host.data_mut().fail(
            InvocationFailureKind::FuelExhausted,
            "policy exhausted fuel in a map host call",
        );
        anyhow::bail!("policy exhausted fuel in a map host call");
    }
    host.as_context_mut().set_fuel(remaining - amount)?;
    Ok(())
}

fn map_error(error: MapAccessError) -> types::MapError {
    match error {
        MapAccessError::UnknownMap(_) => types::MapError::Unavailable,
        MapAccessError::KeyType => types::MapError::TypeMismatch,
        MapAccessError::KeyTooLarge => types::MapError::KeyTooLarge,
        MapAccessError::InvocationByteLimit { .. } => types::MapError::ByteLimit,
        MapAccessError::Mutation(_) => types::MapError::TypeMismatch,
    }
}
