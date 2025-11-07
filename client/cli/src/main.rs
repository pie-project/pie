use anyhow::Result;
use clap::{Parser, Subcommand};

mod config;
mod engine;
mod list;
mod path;
mod ping;
mod submit;

use config::ConfigCommands;
use list::ListArgs;
use ping::PingArgs;
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
    /// Check if the Pie engine is alive and responsive.
    Ping(PingArgs),
    /// List all running inferlet instances.
    List(ListArgs),
    #[command(subcommand)]
    /// Manage configuration.
    Config(ConfigCommands),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Submit(args) => {
            submit::handle_submit_command(args).await?;
        }
        Commands::Ping(args) => {
            ping::handle_ping_command(args).await?;
        }
        Commands::List(args) => {
            list::handle_list_command(args).await?;
        }
        Commands::Config(cmd) => {
            config::handle_config_command(cmd).await?;
        }
    }

    Ok(())
}
