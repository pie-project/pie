use std::collections::BTreeSet;
use std::path::PathBuf;

use pie_plex::v0_5::{ContractVersion, Manifest, Operation, PolicyLimits};
use pie_policy::PolicyPackageV0_5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let component = PathBuf::from(
        args.next()
            .ok_or("usage: pack_policy <component> <output> <name> <operations>")?,
    );
    let output = PathBuf::from(
        args.next()
            .ok_or("usage: pack_policy <component> <output> <name> <operations>")?,
    );
    let package_name = args
        .next()
        .ok_or("usage: pack_policy <component> <output> <name> <operations>")?;
    let operations = args
        .next()
        .ok_or("usage: pack_policy <component> <output> <name> <operations>")?
        .split(',')
        .map(parse_operation)
        .collect::<Result<BTreeSet<_>, _>>()?;
    if args.next().is_some() {
        return Err("pack_policy received unexpected arguments".into());
    }

    let package = PolicyPackageV0_5::new(
        Manifest {
            contract: ContractVersion::V0_5,
            package_name,
            package_version: "0.5.0".into(),
            operations,
            limits: PolicyLimits {
                memory_bytes: 4 << 20,
                fuel: 2_000_000,
                deadline_ms: 100,
                input_bytes: 1 << 20,
                output_bytes: 1 << 20,
            },
        },
        std::fs::read(component)?,
    )?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output, package.encode()?)?;
    Ok(())
}

fn parse_operation(value: &str) -> Result<Operation, String> {
    match value {
        "route" => Ok(Operation::Route),
        "admit" => Ok(Operation::Admit),
        "schedule" => Ok(Operation::Schedule),
        "evict" => Ok(Operation::Evict),
        "feedback" => Ok(Operation::Feedback),
        _ => Err(format!("unknown operation {value:?}")),
    }
}
