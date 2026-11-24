//! Whether a load has a host tier at all: on a device whose memory is the
//! host's (unified memory: a phone's or an integrated GPU, the Spark's,
//! Apple's), a move from the device to the host is a copy within the same
//! memory. It frees nothing physical and only moves bytes between two
//! budgets of one ledger, so such a device states no host tier (neither
//! for kv nor for role slots) and suspended kv goes straight to the disk.
//! The only lending within one memory left there is the activation rungs'
//! to the range. Discrete devices are unchanged.

/// Whose memory a device's is, as its backend states it at bind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemory {
    /// Memory of its own: a host tier frees device memory.
    Own,
    /// The host's: the device and the host draw on the same memory.
    Hosts,
}

/// What a load seats beside its device for suspended kv and role slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostTier {
    /// No host tier: the device's memory is the host's.
    Absent,
    /// The host tier the engine seats, from the operator's
    /// `[engine.memory] budget_host` where stated (else its own default).
    Seated(Option<u64>),
}

impl DeviceMemory {
    /// The load line of a device that states no host tier.
    pub const NO_HOST_TIER: &'static str =
        "no host kv tier: the device's memory is the host's; suspended kv goes to the disk";

    /// Of `stated`, the operator's `[engine.memory] budget_host` bytes,
    /// the host tier a load on this device seats.
    ///
    /// # Errors
    ///
    /// A host tier stated on a device whose memory is the host's, refused
    /// by name.
    pub fn host_tier(self, stated: Option<u64>) -> Result<HostTier, String> {
        match (self, stated) {
            (DeviceMemory::Own, stated) => Ok(HostTier::Seated(stated)),
            (DeviceMemory::Hosts, None) => Ok(HostTier::Absent),
            (DeviceMemory::Hosts, Some(bytes)) => Err(format!(
                "`[engine.memory] budget_host` ({} MiB) states a host tier, and this device's \
                 memory is the host's: a move to the host would copy within the same memory \
                 and free nothing; omit budget_host (suspended kv goes to the disk)",
                bytes >> 20
            )),
        }
    }

    /// Whether the device's memory is the host's.
    #[must_use]
    pub fn unified(self) -> bool {
        self == DeviceMemory::Hosts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_unified_device_states_no_host_tier_and_refuses_a_stated_one_by_name() {
        assert_eq!(DeviceMemory::Hosts.host_tier(None), Ok(HostTier::Absent));
        let refused = DeviceMemory::Hosts.host_tier(Some(8 << 30)).unwrap_err();
        assert!(refused.contains("budget_host"), "{refused}");
        assert!(refused.contains("the host's"), "{refused}");
        assert!(DeviceMemory::Hosts.unified());
    }

    #[test]
    fn a_discrete_device_seats_what_is_stated_or_its_default() {
        assert_eq!(DeviceMemory::Own.host_tier(None), Ok(HostTier::Seated(None)));
        assert_eq!(
            DeviceMemory::Own.host_tier(Some(8 << 30)),
            Ok(HostTier::Seated(Some(8 << 30)))
        );
        assert!(!DeviceMemory::Own.unified());
    }
}
