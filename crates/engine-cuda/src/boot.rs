use crate::api::{ContractFor, Cuda, DeviceBoot};

pub fn open(boot: DeviceBoot, contract_for: ContractFor) -> Result<Cuda, String> {
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
