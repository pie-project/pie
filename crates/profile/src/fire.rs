//! Tuned results a load reads to choose its top rung of fires: the measured
//! fire times of a model class on a device, and the rates a device's fire
//! time is estimated from where nothing was measured. Both are keyed by the
//! device's profile id ([`crate::id_of`] of its name); the fire times also
//! by the model class (the deployment, as `engine::load::model_class` names
//! it). Checked in, and empty until a device is measured: a lookup that finds
//! nothing returns an empty table, or no rates.
//!
//! The device's own file under a backend's `profiles/` is where these move
//! once the GEMM tuning lands in the profile under the same device key: one
//! file a device holds both.

/// The measured fire times of `model` on `device`: `(tokens, ns)` of a fire
/// at each rung width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FireTable {
    pub device: &'static str,
    pub model: &'static str,
    pub fire_ns: &'static [(u32, u64)],
}

/// What `device` computes and moves a nanosecond, where it was measured or
/// stated: bf16 flops, bytes of device memory, and the time a fire's
/// launches cost beside its work.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceRates {
    pub device: &'static str,
    pub flops_per_ns: Option<f64>,
    pub bytes_per_ns: Option<f64>,
    pub launch_ns: Option<u64>,
}

/// Every measured fire table.
pub const FIRE_NS: &[FireTable] = &[];

/// Every device's rates.
pub const RATES: &[DeviceRates] = &[];

/// The fire times measured of `model` on `device`, or none.
#[must_use]
pub fn fire_ns(device: &str, model: &str) -> &'static [(u32, u64)] {
    lookup(FIRE_NS, device, model)
}

/// The rates of `device`, where any are known.
#[must_use]
pub fn rates(device: &str) -> Option<DeviceRates> {
    RATES.iter().copied().find(|r| r.device == device)
}

fn lookup(table: &[FireTable], device: &str, model: &str) -> &'static [(u32, u64)] {
    table
        .iter()
        .find(|t| t.device == device && t.model == model)
        .map_or(&[], |t| t.fire_ns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lookup_finds_its_entry_or_nothing() {
        const TABLE: &[FireTable] = &[FireTable {
            device: "nvidia-rtx-pro-4500-blackwell",
            model: "qwen3-8b-bf16-kv-bf16",
            fire_ns: &[(1024, 50_000_000), (2048, 90_000_000)],
        }];
        assert_eq!(
            lookup(
                TABLE,
                "nvidia-rtx-pro-4500-blackwell",
                "qwen3-8b-bf16-kv-bf16"
            ),
            &[(1024, 50_000_000), (2048, 90_000_000)]
        );
        assert!(lookup(TABLE, "nvidia-rtx-pro-4500-blackwell", "qwen3-0-6b").is_empty());
        assert!(fire_ns("nvidia-rtx-pro-4500-blackwell", "qwen3-8b-bf16-kv-bf16").is_empty());
        assert_eq!(rates("nvidia-rtx-pro-4500-blackwell"), None);
    }

    #[test]
    fn every_checked_in_table_is_one_entry_a_key_ascending() {
        for (k, t) in FIRE_NS.iter().enumerate() {
            assert!(
                FIRE_NS[..k]
                    .iter()
                    .all(|o| (o.device, o.model) != (t.device, t.model)),
                "{} / {} twice",
                t.device,
                t.model
            );
            assert!(t.fire_ns.windows(2).all(|w| w[0].0 < w[1].0));
        }
        for (k, r) in RATES.iter().enumerate() {
            assert!(RATES[..k].iter().all(|o| o.device != r.device));
        }
    }
}
