use serde::{Deserialize, Serialize};

/// One admitted unit of serving work, potentially spanning trusted
/// continuations and multiple generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LogicalRequestId([u8; 16]);

impl LogicalRequestId {
    pub const fn new(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn into_bytes(self) -> [u8; 16] {
        self.0
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

/// A generation within one logical request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GenerationId(u64);

impl GenerationId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Host-minted identity for one feedback delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DeliveryId([u8; 16]);

impl DeliveryId {
    pub const fn new(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn into_bytes(self) -> [u8; 16] {
        self.0
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

macro_rules! handle {
    ($name:ident) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
        )]
        pub struct $name(u32);

        impl $name {
            pub const fn new(value: u32) -> Self {
                Self(value)
            }

            pub const fn get(self) -> u32 {
                self.0
            }
        }
    };
}

handle!(FactHandle);
handle!(MetadataHandle);
handle!(MapHandle);
handle!(CapabilityHandle);
handle!(EventHandle);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_round_trip_through_json() {
        let logical = LogicalRequestId::new([7; 16]);
        let encoded = serde_json::to_string(&logical).unwrap();
        assert_eq!(
            serde_json::from_str::<LogicalRequestId>(&encoded).unwrap(),
            logical
        );

        let delivery = DeliveryId::new([9; 16]);
        let encoded = serde_json::to_string(&delivery).unwrap();
        assert_eq!(
            serde_json::from_str::<DeliveryId>(&encoded).unwrap(),
            delivery
        );
    }
}
