//! Serve command implementation for the Pie CLI.
//!
//! This module implements the `pie serve` subcommand which starts the Pie engine
//! and provides an interactive shell session for running inferlets and managing
//! the engine state.

use crate::{engine, output, path};
use anyhow::Result;
use clap::{Args, Parser};
use pie::Config as EngineConfig;
use rustyline::Editor;
use rustyline::error::ReadlineError;
use rustyline::history::FileHistory;
use std::path::PathBuf;

/// Arguments for the `pie serve` command.
#[derive(Args, Debug, Default)]
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

/// Arguments for running inferlets within the interactive shell.
#[derive(Parser, Debug)]
pub struct ShellRunArgs {
    /// Path to the .wasm inferlet file.
    #[arg(value_parser = path::expand_tilde)]
    pub inferlet_path: PathBuf,

    /// Stream the inferlet output to the console.
    #[arg(long, short)]
    pub stream_output: bool,

    /// Arguments to pass to the Wasm program.
    pub arguments: Vec<String>,
}

/// Handles the `pie serve` command.
///
/// This function:
/// 1. Creates an editor and printer for the interactive shell
/// 2. Starts the Pie engine and backend services
/// 3. Runs the interactive shell session
/// 4. Terminates the engine and backend services on exit
pub async fn handle_serve_command(
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

/// Parses and executes commands in the interactive shell.
async fn handle_shell_command(
    command: &str,
    args: &[&str],
    client_config: &engine::ClientConfig,
    printer: &output::SharedPrinter,
) -> Result<bool> {
    match command {
        "run" => {
            // Prepend a dummy command name so clap can parse the args slice.
            let clap_args = std::iter::once("run").chain(args.iter().copied());

            match ShellRunArgs::try_parse_from(clap_args) {
                Ok(run_args) => {
                    if let Err(e) = engine::submit_detached_inferlet(
                        client_config,
                        run_args.inferlet_path,
                        run_args.arguments,
                        run_args.stream_output,
                        printer.clone(),
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

/// Runs the interactive shell session.
///
/// This function provides the main interactive loop for the `pie serve` command,
/// allowing users to run inferlets, query engine state, and manage the session.
async fn run_shell(
    client_config: &engine::ClientConfig,
    mut rl: Editor<output::MyHelper, FileHistory>,
    printer: output::SharedPrinter,
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
