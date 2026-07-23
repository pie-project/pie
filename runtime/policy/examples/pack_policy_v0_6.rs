use std::collections::BTreeSet;
use std::path::PathBuf;

use pie_plex::v0_6::{ContractVersion, Manifest, MechanicId, Operation, PolicyLimits};
use pie_policy::PolicyPackageV0_6;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let component = PathBuf::from(args.next().ok_or(
        "usage: pack_policy_v0_6 <component> <output> <name> <operations> [requires] [optional]",
    )?);
    let output = PathBuf::from(args.next().ok_or("missing output path")?);
    let package_name = args.next().ok_or("missing package name")?;
    let implements = parse_operations(&args.next().ok_or("missing operation list")?)?;
    let requires = parse_mechanics(args.next().as_deref().unwrap_or(""))?;
    let optional = parse_mechanics(args.next().as_deref().unwrap_or(""))?;
    if args.next().is_some() {
        return Err("too many arguments".into());
    }

    let package = PolicyPackageV0_6::new(
        Manifest {
            contract: ContractVersion::V0_6,
            package_name,
            package_version: "0.6.0".into(),
            implements,
            requires,
            optional,
            schemas: BTreeSet::new(),
            limits: PolicyLimits {
                memory_bytes: 4 * 1024 * 1024,
                fuel: 2_000_000,
                deadline_ms: 100,
                input_bytes: 1024 * 1024,
                output_bytes: 1024 * 1024,
                host_calls: 64,
                host_call_bytes: 1024 * 1024,
            },
        },
        std::fs::read(component)?,
    )?;
    std::fs::write(output, package.encode()?)?;
    Ok(())
}

fn parse_operations(value: &str) -> Result<BTreeSet<Operation>, String> {
    value
        .split(',')
        .filter(|value| !value.is_empty())
        .map(|value| match value {
            "admit" => Ok(Operation::Admit),
            "route" => Ok(Operation::Route),
            "schedule" => Ok(Operation::Schedule),
            "cache" => Ok(Operation::Cache),
            "feedback" => Ok(Operation::Feedback),
            _ => Err(format!("unknown operation {value:?}")),
        })
        .collect()
}

fn parse_mechanics(value: &str) -> Result<BTreeSet<MechanicId>, String> {
    Ok(value
        .split(',')
        .filter(|value| !value.is_empty())
        .map(MechanicId::from)
        .collect())
}
