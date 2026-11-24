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
