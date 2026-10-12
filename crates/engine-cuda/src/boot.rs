use crate::api::{ContractFor, Cuda, DeviceBoot};

pub fn open(boot: DeviceBoot, contract_for: ContractFor) -> Result<Cuda, String> {
    let fraction = boot.knobs.gpu_mem_utilization;
    if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
        return Err(format!(
            "`gpu_mem_utilization` must be finite and in (0.0, 1.0]; this boot \
             says {fraction}. It is the fraction of the whole card this \
             deployment lets pie hold, weights included — 1.0 is the whole \
             card, which is what the elastic pool took before the knob \
             reached a shell at all."
        ));
    }
    Ok(Cuda::new(boot, contract_for))
}

#[must_use]
pub fn ordinal_of(device: &str) -> i32 {
    device
        .rsplit(':')
        .next()
        .and_then(|ordinal| ordinal.trim().parse::<i32>().ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::ordinal_of;

    #[test]
    fn the_device_spelling_is_read_in_every_form_a_deployment_writes() {
        assert_eq!(ordinal_of("cuda:3"), 3);
        assert_eq!(ordinal_of("2"), 2);
        assert_eq!(ordinal_of(""), 0, "saying nothing means device 0");
    }
}
