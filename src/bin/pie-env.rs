use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use pie::process::report::Resolved;
use pie::process::{BootSpec, GlobalArgs};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Field {
    Home,
    Config,
    LogLevel,
    Metrics,
}

#[derive(Parser)]
#[command(
    name = "pie-env",
    version,
    about = "Show what a pie daemon would resolve at boot (config path, $PIE_HOME, log level, /metrics) without starting it."
)]
struct Cli {
    #[command(flatten)]
    global: GlobalArgs,

    #[arg(
        long,
        value_name = "ROLE",
        value_parser = clap::builder::PossibleValuesParser::new(BootSpec::ROLES),
    )]
    role: Option<String>,

    #[arg(long, value_enum, value_name = "FIELD")]
    field: Option<Field>,

    #[arg(long)]
    check: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let specs: Vec<BootSpec> = match cli.role.as_deref() {
        Some(r) => BootSpec::for_role(r).into_iter().collect(),
        None => BootSpec::ROLES
            .iter()
            .filter_map(|r| BootSpec::for_role(r))
            .collect(),
    };
    let resolved: Vec<Resolved> = specs
        .iter()
        .map(|spec| pie::process::report::resolve(spec, &cli.global))
        .collect();

    match cli.field {
        Some(field) => {
            if cli.role.is_none() {
                eprintln!("pie-env: --field requires --role (values differ per role)");
                return ExitCode::FAILURE;
            }
            println!("{}", field_of(&resolved[0], field));
        }
        None => {
            for (i, r) in resolved.iter().enumerate() {
                if i > 0 {
                    println!();
                }
                print!("{r}");
            }
        }
    }

    if cli.check && resolved.iter().any(|r| !r.problems().is_empty()) {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn field_of(r: &Resolved, field: Field) -> &str {
    match field {
        Field::Home => &r.home.value,
        Field::Config => &r.config_path.value,
        Field::LogLevel => &r.log_level.value,
        Field::Metrics => r.metrics.as_ref().map(|s| s.value.as_str()).unwrap_or(""),
    }
}
