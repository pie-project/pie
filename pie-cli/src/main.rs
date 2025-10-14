use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use pie::Config as EngineConfig;
use rustyline::Editor;
use rustyline::error::ReadlineError;
use rustyline::history::FileHistory;
use std::path::PathBuf;

mod config;
mod engine;
mod model;
mod output;
mod path;
mod run;

use config::ConfigCommands;
use engine::ClientConfig;
use model::ModelCommands;
use output::{MyHelper, SharedPrinter};
use run::RunArgs;

//================================================================================//
// SECTION: CLI Command & Config Structs
//================================================================================//

#[derive(Parser, Debug)]
#[command(author, version, about = "Pie Command Line Interface")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
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

#[derive(Args, Debug)]
/// Arguments for starting the Pie engine.
pub struct ServeArgs {
    /// Path to a custom TOML configuration file.
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// The network host to bind to.
    #[arg(long)]
    pub host: Option<String>,
    /// The network port to use.
    #[arg(long)]
    pub port: Option<u16>,
    /// Disable authentication.
    #[arg(long)]
    pub no_auth: bool,
    /// A log file to write to.
    #[arg(long)]
    pub log: Option<PathBuf>,
    /// Enable verbose console logging.
    #[arg(long, short)]
    pub verbose: bool,
}

/// Helper for clap to expand `~` in path arguments.
fn expand_tilde(s: &str) -> Result<PathBuf, std::convert::Infallible> {
    Ok(PathBuf::from(shellexpand::tilde(s).as_ref()))
}

#[derive(Parser, Debug)]
/// Arguments to submit an inferlet (Wasm program) to the engine in the shell.
pub struct ShellRunArgs {
    /// Path to the .wasm inferlet file.
    #[arg(value_parser = expand_tilde)]
    pub inferlet_path: PathBuf,

    /// Run the inferlet in the background and print its instance ID.
    #[arg(long, short)]
    pub detach: bool,

    /// Arguments to pass to the Wasm program.
    pub arguments: Vec<String>,
}

//================================================================================//
// SECTION: Main Entrypoint
//================================================================================//

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
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

            handle_serve_command(engine_config, backend_configs).await?;
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

//================================================================================//
// SECTION: Command Handlers
//================================================================================//

/// Handles the `pie serve` command.
async fn handle_serve_command(
    engine_config: EngineConfig,
    backend_configs: Vec<toml::Value>,
) -> Result<()> {
    let (rl, printer) = output::create_editor_and_printer_with_history().await?;

    // Start the engine and backend services
    let (shutdown_tx, server_handle, backend_processes, client_config) =
        engine::start_engine_and_backend(engine_config, backend_configs, printer.clone()).await?;

    // Start the interactive session, passing both configs
    run_shell(&client_config, rl, printer).await?;

    // Terminate the engine and backend services
    engine::terminate_engine_and_backend(
        &client_config,
        backend_processes,
        shutdown_tx,
        server_handle,
    )
    .await?;

    Ok(())
}

/// Parses and executes commands in the shell.
async fn handle_shell_command(
    command: &str,
    args: &[&str],
    client_config: &ClientConfig,
    printer: &SharedPrinter,
) -> Result<bool> {
    match command {
        "run" => {
            // Prepend a dummy command name so clap can parse the args slice.
            let clap_args = std::iter::once("run").chain(args.iter().copied());

            match ShellRunArgs::try_parse_from(clap_args) {
                Ok(run_args) => {
                    if let Err(e) = engine::run_inferlet(
                        client_config,
                        run_args.inferlet_path,
                        run_args.arguments,
                        run_args.detach,
                        printer,
                    )
                    .await
                    {
                        // Use the printer to avoid corrupting the prompt.
                        let _ = output::print_with_printer(
                            printer,
                            format!("Error: running inferlet: {e}"),
                        )
                        .await;
                    }
                }
                Err(e) => {
                    // Clap's error messages are user-friendly and include usage.
                    let _ = output::print_with_printer(printer, e.to_string()).await;
                }
            }
        }
        "query" => {
            println!("(Query functionality not yet implemented)");
        }
        "stat" => {
            engine::print_backend_stats(client_config, printer).await?;
        }
        "help" => {
            println!("Available commands:");
            println!(
                "  run [--detach] <path> [ARGS]... - Run a .wasm inferlet with optional arguments"
            );
            println!("  query                  - (Placeholder) Query the engine state");
            println!("  exit                   - Exit the Pie session");
            println!("  help                   - Show this help message");
        }
        "exit" => {
            println!("Exiting...");
            return Ok(true);
        }
        _ => {
            println!(
                "Unknown command: '{}'. Type 'help' for a list of commands.",
                command
            );
        }
    }
    Ok(false)
}

//================================================================================//
// SECTION: Shell Interaction
//================================================================================//

async fn run_shell(
    client_config: &ClientConfig,
    mut rl: Editor<MyHelper, FileHistory>,
    printer: SharedPrinter,
) -> Result<()> {
    println!("Entering interactive session. Type 'help' for commands or use ↑/↓ for history.");

    // The main interactive loop
    loop {
        match rl.readline("pie> ") {
            Ok(line) => {
                let _ = rl.add_history_entry(line.as_str());
                let parts: Vec<String> = match shlex::split(&line) {
                    Some(parts) => parts,
                    None => {
                        eprintln!("Error: Mismatched quotes in command.");
                        continue;
                    }
                };
                if parts.is_empty() {
                    continue;
                }

                match handle_shell_command(
                    &parts[0],
                    &parts[1..].iter().map(AsRef::as_ref).collect::<Vec<_>>(),
                    client_config,
                    &printer,
                )
                .await
                {
                    Ok(should_exit) if should_exit => break,
                    Ok(_) => (),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            Err(ReadlineError::Interrupted) => println!("(To exit, type 'exit' or press Ctrl-D)"),
            Err(ReadlineError::Eof) => {
                println!("Exiting...");
                break;
            }
            Err(err) => {
                eprintln!("Error reading line: {}", err);
                break;
            }
        }
    }

    println!("Shutting down services...");
    if let Err(err) = rl.save_history(&path::get_shell_history_path()?) {
        eprintln!("Warning: Failed to save command history: {}", err);
    }

    Ok(())
}
