//! Rust guest bindings for the fixed PLEX v0.1 component world.

#![forbid(unsafe_code)]

pub use wit_bindgen;

wit_bindgen::generate!({
    path: "wit",
    world: "plex-policy",
    pub_export_macro: true,
    generate_all,
});

pub use pie::plex::types;

pub trait LinkSetExt {
    fn fact(&self, declaration: usize) -> Option<types::FactHandle>;
    fn metadata(&self, declaration: usize) -> Option<types::MetadataHandle>;
    fn map(&self, declaration: usize) -> Option<types::MapHandle>;
    fn event(&self, declaration: usize) -> Option<types::EventHandle>;
    fn capability(&self, declaration: usize) -> Option<types::CapabilityHandle>;
}

impl LinkSetExt for types::LinkSet {
    fn fact(&self, declaration: usize) -> Option<types::FactHandle> {
        self.facts.get(declaration).copied().flatten()
    }

    fn metadata(&self, declaration: usize) -> Option<types::MetadataHandle> {
        self.metadata.get(declaration).copied().flatten()
    }

    fn map(&self, declaration: usize) -> Option<types::MapHandle> {
        self.maps.get(declaration).copied().flatten()
    }

    fn event(&self, declaration: usize) -> Option<types::EventHandle> {
        self.events.get(declaration).copied().flatten()
    }

    fn capability(&self, declaration: usize) -> Option<types::CapabilityHandle> {
        self.capabilities.get(declaration).copied().flatten()
    }
}

pub trait RecordBatchExt {
    fn fact(&self, handle: types::FactHandle) -> Option<&types::ColumnValues>;
    fn metadata(&self, handle: types::MetadataHandle) -> Option<&types::ColumnValues>;
}

impl RecordBatchExt for types::RecordBatch {
    fn fact(&self, handle: types::FactHandle) -> Option<&types::ColumnValues> {
        self.facts
            .iter()
            .find(|column| column.handle.value == handle.value)
            .map(|column| &column.values)
    }

    fn metadata(&self, handle: types::MetadataHandle) -> Option<&types::ColumnValues> {
        self.metadata
            .iter()
            .find(|column| column.handle.value == handle.value)
            .map(|column| &column.values)
    }
}

pub trait LogicalRequestIdExt {
    fn to_be_bytes(self) -> [u8; 16];
}

impl LogicalRequestIdExt for types::LogicalRequestId {
    fn to_be_bytes(self) -> [u8; 16] {
        let mut bytes = [0; 16];
        bytes[..8].copy_from_slice(&self.high.to_be_bytes());
        bytes[8..].copy_from_slice(&self.low.to_be_bytes());
        bytes
    }
}

pub trait MapKeyType {
    fn into_map_key(self) -> types::MapKey;
}

impl MapKeyType for bool {
    fn into_map_key(self) -> types::MapKey {
        types::MapKey::Boolean(self)
    }
}

impl MapKeyType for i64 {
    fn into_map_key(self) -> types::MapKey {
        types::MapKey::Signed64(self)
    }
}

impl MapKeyType for u64 {
    fn into_map_key(self) -> types::MapKey {
        types::MapKey::Unsigned64(self)
    }
}

impl MapKeyType for String {
    fn into_map_key(self) -> types::MapKey {
        types::MapKey::Text(self)
    }
}

impl MapKeyType for Vec<u8> {
    fn into_map_key(self) -> types::MapKey {
        types::MapKey::Bytes(self)
    }
}

pub trait MapValueType: Sized {
    fn into_map_value(self) -> types::MapValue;
    fn from_map_value(value: types::MapValue) -> Option<Self>;
}

macro_rules! map_value {
    ($ty:ty, $variant:ident) => {
        impl MapValueType for $ty {
            fn into_map_value(self) -> types::MapValue {
                types::MapValue::$variant(self)
            }

            fn from_map_value(value: types::MapValue) -> Option<Self> {
                match value {
                    types::MapValue::$variant(value) => Some(value),
                    _ => None,
                }
            }
        }
    };
}

map_value!(bool, Boolean);
map_value!(i64, Signed64);
map_value!(u64, Unsigned64);
map_value!(f64, Float64);
map_value!(String, Text);
map_value!(Vec<u8>, Bytes);

#[derive(Debug, Clone, Copy)]
pub struct TypedMap<K, V> {
    handle: types::MapHandle,
    marker: core::marker::PhantomData<fn(K) -> V>,
}

impl<K: MapKeyType, V: MapValueType> TypedMap<K, V> {
    pub const fn new(handle: types::MapHandle) -> Self {
        Self {
            handle,
            marker: core::marker::PhantomData,
        }
    }

    pub fn get(&self, key: K) -> Result<Option<V>, types::MapError> {
        match crate::pie::plex::maps::get(self.handle, &key.into_map_key())? {
            Some(value) => V::from_map_value(value)
                .map(Some)
                .ok_or(types::MapError::TypeMismatch),
            None => Ok(None),
        }
    }

    pub fn upsert(&self, key: K, value: V, ttl_ms: Option<u64>) -> types::MapMutation {
        types::MapMutation::Upsert(types::MapUpsert {
            handle: self.handle,
            key: key.into_map_key(),
            value: value.into_map_value(),
            ttl_ms,
        })
    }

    pub fn delete(&self, key: K) -> types::MapMutation {
        types::MapMutation::Delete(types::MapDelete {
            handle: self.handle,
            key: key.into_map_key(),
        })
    }
}

impl<K: MapKeyType> TypedMap<K, u64> {
    pub fn add(&self, key: K, delta: u64, ttl_ms: Option<u64>) -> types::MapMutation {
        types::MapMutation::AddU64(types::MapAddU64 {
            handle: self.handle,
            key: key.into_map_key(),
            delta,
            ttl_ms,
        })
    }
}

impl<K: MapKeyType> TypedMap<K, i64> {
    pub fn add(&self, key: K, delta: i64, ttl_ms: Option<u64>) -> types::MapMutation {
        types::MapMutation::AddI64(types::MapAddI64 {
            handle: self.handle,
            key: key.into_map_key(),
            delta,
            ttl_ms,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct U64Map {
    handle: types::MapHandle,
}

impl U64Map {
    pub const fn new(handle: types::MapHandle) -> Self {
        Self { handle }
    }

    pub fn get_bytes(&self, key: &[u8]) -> Result<Option<u64>, types::MapError> {
        match crate::pie::plex::maps::get(self.handle, &types::MapKey::Bytes(key.to_vec()))? {
            Some(types::MapValue::Unsigned64(value)) => Ok(Some(value)),
            None => Ok(None),
            Some(_) => Err(types::MapError::TypeMismatch),
        }
    }

    pub fn add_bytes(&self, key: Vec<u8>, delta: u64, ttl_ms: Option<u64>) -> types::MapMutation {
        types::MapMutation::AddU64(types::MapAddU64 {
            handle: self.handle,
            key: types::MapKey::Bytes(key),
            delta,
            ttl_ms,
        })
    }

    pub fn upsert_bytes(
        &self,
        key: Vec<u8>,
        value: u64,
        ttl_ms: Option<u64>,
    ) -> types::MapMutation {
        types::MapMutation::Upsert(types::MapUpsert {
            handle: self.handle,
            key: types::MapKey::Bytes(key),
            value: types::MapValue::Unsigned64(value),
            ttl_ms,
        })
    }
}

/// Author-facing policy trait. Unimplemented operations request native fallback.
pub trait Policy {
    fn admit(input: types::AdmissionInput) -> Result<types::AdmissionOutput, types::PolicyError> {
        let _ = input;
        Err(types::PolicyError::FallbackRequired)
    }

    fn route(input: types::PlacementInput) -> Result<types::DenseOutput, types::PolicyError> {
        let _ = input;
        Err(types::PolicyError::FallbackRequired)
    }

    fn schedule(input: types::ScheduleInput) -> Result<types::ScheduleOutput, types::PolicyError> {
        let _ = input;
        Err(types::PolicyError::FallbackRequired)
    }

    fn evict(input: types::EvictionInput) -> Result<types::DenseOutput, types::PolicyError> {
        let _ = input;
        Err(types::PolicyError::FallbackRequired)
    }

    fn feedback(input: types::FeedbackInput) -> Result<types::FeedbackOutput, types::PolicyError> {
        let _ = input;
        Err(types::PolicyError::FallbackRequired)
    }
}

/// Export one [`Policy`] implementation through the fixed PLEX component world.
#[macro_export]
macro_rules! export_policy {
    ($policy:ty) => {
        struct __PlexGuest;

        impl $crate::exports::pie::plex::policy::Guest for __PlexGuest {
            fn admit(
                input: $crate::types::AdmissionInput,
            ) -> Result<$crate::types::AdmissionOutput, $crate::types::PolicyError> {
                <$policy as $crate::Policy>::admit(input)
            }

            fn route(
                input: $crate::types::PlacementInput,
            ) -> Result<$crate::types::DenseOutput, $crate::types::PolicyError> {
                <$policy as $crate::Policy>::route(input)
            }

            fn schedule(
                input: $crate::types::ScheduleInput,
            ) -> Result<$crate::types::ScheduleOutput, $crate::types::PolicyError> {
                <$policy as $crate::Policy>::schedule(input)
            }

            fn evict(
                input: $crate::types::EvictionInput,
            ) -> Result<$crate::types::DenseOutput, $crate::types::PolicyError> {
                <$policy as $crate::Policy>::evict(input)
            }

            fn feedback(
                input: $crate::types::FeedbackInput,
            ) -> Result<$crate::types::FeedbackOutput, $crate::types::PolicyError> {
                <$policy as $crate::Policy>::feedback(input)
            }
        }

        $crate::export!(__PlexGuest with_types_in $crate);
    };
}
