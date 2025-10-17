use anyhow::Result;
use clap::{Parser, Subcommand};

mod config;
mod engine;
mod model;
mod output;
mod path;
mod run;
mod serve;

use config::ConfigCommands;
use model::ModelCommands;
use run::RunArgs;
use serve::ServeArgs;

#[derive(Parser, Debug)]
#[command(author, version, about = "Pie Command Line Interface")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the Pie engine and enter an interactive session.
    Serve(ServeArgs),
    /// Run an inferlet with a one-shot Pie engine.
    Run(RunArgs),
    #[command(subcommand)]
    /// Manage local models.
    Model(ModelCommands),
    #[command(subcommand)]
    /// Manage configuration.
    Config(ConfigCommands),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command.unwrap_or(Commands::Serve(ServeArgs::default())) {
        Commands::Serve(args) => {
            let (engine_config, backend_configs) = engine::parse_engine_and_backend_config(
                args.config,
                args.no_auth,
                args.host,
                args.port,
                args.verbose,
                args.log,
            )?;

            // Initialize logging based on the config and get the file-writer guard
            let _guard = output::init_logging(&engine_config)?;

            serve::handle_serve_command(engine_config, backend_configs).await?;
        }
        Commands::Run(args) => {
            // Build both engine and backend configs.
            let (engine_config, backend_configs) = engine::parse_engine_and_backend_config(
                args.config,
                false,
                None,
                None,
                false,
                args.log,
            )?;

            // Initialize logging based on the config and get the file-writer guard
            let _guard = output::init_logging(&engine_config)?;

            run::handle_run_command(
                engine_config,
                backend_configs,
                args.inferlet,
                args.arguments,
            )
            .await?;
        }
        Commands::Model(cmd) => {
            // Model commands don't start the engine, so they can use a simple logger
            output::init_simple_logging()?;
            model::handle_model_command(cmd).await?;
        }
        Commands::Config(cmd) => {
            // Config commands don't start the engine, so they can use a simple logger
            output::init_simple_logging()?;
            config::handle_config_command(cmd).await?;
        }
    }
    Ok(())
}
