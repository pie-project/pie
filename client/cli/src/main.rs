use anyhow::Result;
use clap::{Parser, Subcommand};

mod config;
mod engine;
mod path;
mod submit;

use config::ConfigCommands;
use submit::SubmitArgs;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Programmable Inference Command Line Interface"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Submit an inferlet to a running Pie engine.
    Submit(SubmitArgs),
    #[command(subcommand)]
    /// Manage configuration.
    Config(ConfigCommands),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Submit(args) => {
            submit::handle_submit_command(
                args.config,
                args.host,
                args.port,
                args.username,
                args.private_key_path,
                args.auth_secret,
                args.inferlet,
                args.arguments,
            )
            .await?;
        }
        Commands::Config(cmd) => {
            config::handle_config_command(cmd).await?;
        }
    }

    Ok(())
}
