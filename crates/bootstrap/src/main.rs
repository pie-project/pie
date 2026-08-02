//! `bootstrap` — show what a pie daemon would resolve, without starting one.
//!
//! Every pie process resolves the same things at boot: `$PIE_HOME`, *which*
//! config file to read (`--config` → `$PIE_CONFIG` → `$PIE_HOME/<role>.toml`,
//! and the default filename differs per role), the log filter, and the
//! `/metrics` address. That resolution is invisible until the process is already
//! running — and when it goes wrong the failure is silent: a config file at a
//! path nobody reads produces a daemon that boots happily on defaults.
//!
//! This command answers those questions up front, from the same code the
//! daemons run ([`bootstrap::report`]), so it cannot report a path they wouldn't
//! read.
//!
//! ```text
//! pie-env                                # report every role
//! pie-env --role worker                  # just the worker
//! pie-env --role worker --check          # exit non-zero if it couldn't boot
//! pie-env --role worker --field config   # raw value, for scripts
//! ```

use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use bootstrap::report::Resolved;
use bootstrap::{BootSpec, GlobalArgs};

/// A single value to print raw, for shell consumption:
/// `cfg=$(pie-env --role worker --field config)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Field {
    /// `$PIE_HOME`.
    Home,
    /// The config file path this role would read.
    Config,
    /// The effective tracing filter.
    LogLevel,
    /// The `/metrics` address (empty line if the role has none).
    Metrics,
}

#[derive(Parser)]
#[command(
    name = "pie-env",
    version,
    about = "Show what a pie daemon would resolve at boot (config path, $PIE_HOME, log level, /metrics) without starting it."
)]
struct Cli {
    /// The shared flags, so resolution sees exactly what a daemon would.
    #[command(flatten)]
    global: GlobalArgs,

    /// Which role to resolve for. Roles differ in their default config
    /// filename and metrics port; omit to report all of them.
    #[arg(
        long,
        value_name = "ROLE",
        value_parser = clap::builder::PossibleValuesParser::new(BootSpec::ROLES),
    )]
    role: Option<String>,

    /// Print one raw value instead of the report (requires `--role`).
    #[arg(long, value_enum, value_name = "FIELD")]
    field: Option<Field>,

    /// Exit non-zero if any reported role could not boot.
    #[arg(long)]
    check: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    // `--role` is validated by clap against `BootSpec::ROLES`, so `for_role`
    // cannot miss here; the filter keeps the two lists impossible to desync.
    let specs: Vec<BootSpec> = match cli.role.as_deref() {
        Some(r) => BootSpec::for_role(r).into_iter().collect(),
        None => BootSpec::ROLES
            .iter()
            .filter_map(|r| BootSpec::for_role(r))
            .collect(),
    };
    let resolved: Vec<Resolved> = specs
        .iter()
        .map(|spec| bootstrap::report::resolve(spec, &cli.global))
        .collect();

    match cli.field {
        Some(field) => {
            // Per-role values differ, so a single raw line would be a lie if we
            // don't know which role is meant.
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
        // An empty string (not "none") keeps `[ -z "$x" ]` working.
        Field::Metrics => r.metrics.as_ref().map(|s| s.value.as_str()).unwrap_or(""),
    }
}
