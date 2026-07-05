use anyhow::{Result, anyhow, ensure};
use pie_driver_abi::PieInstanceBinding;

use super::frame::InstanceBindingPlan;

pub(crate) fn validate_instance_binding(
    binding: &PieInstanceBinding,
    plan: &InstanceBindingPlan,
) -> Result<()> {
    pie_driver_abi::validate_instance_binding(binding)
        .map_err(|err| anyhow!("invalid native instance binding: {err}"))?;
    if plan.requested_instance_id != 0 {
        ensure!(
            binding.instance_id == plan.requested_instance_id,
            "native binding returned instance {} for requested {}",
            binding.instance_id,
            plan.requested_instance_id
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(requested_instance_id: u64) -> InstanceBindingPlan {
        InstanceBindingPlan {
            driver_id: 0,
            program_id: 1,
            requested_instance_id,
            pacing_wait_id: 11,
            channel_ids: vec![101],
            seed_values: Vec::new(),
        }
    }

    #[test]
    fn accepts_driver_or_requested_identity() {
        validate_instance_binding(&PieInstanceBinding { instance_id: 9 }, &plan(0)).unwrap();
        validate_instance_binding(&PieInstanceBinding { instance_id: 7 }, &plan(7)).unwrap();
    }

    #[test]
    fn rejects_zero_or_mismatched_identity() {
        assert!(validate_instance_binding(&PieInstanceBinding::default(), &plan(0)).is_err());
        assert!(
            validate_instance_binding(&PieInstanceBinding { instance_id: 8 }, &plan(7)).is_err()
        );
    }
}
