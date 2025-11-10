use anyhow::Result;
use clap::{Parser, Subcommand};

mod abort;
mod attach;
mod config;
mod engine;
mod list;
mod path;
mod ping;
mod submit;

use abort::AbortArgs;
use attach::AttachArgs;
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
    /// Attach to a running inferlet instance and stream its output.
    Attach(AttachArgs),
    /// Terminate a running inferlet instance.
    Abort(AbortArgs),
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
        Commands::Attach(args) => {
            attach::handle_attach_command(args).await?;
        }
        Commands::Abort(args) => {
            abort::handle_abort_command(args).await?;
        }
        Commands::Config(cmd) => {
            config::handle_config_command(cmd).await?;
        }
    }

    Ok(())
}
