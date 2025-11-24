//! Submit command implementation for the Pie CLI.
//!
//! This module implements the `pie submit` subcommand for submitting inferlets
//! to an existing running Pie engine instance.

use crate::engine;
use crate::path;
use anyhow::Context;
use anyhow::Result;
use clap::Args;
use pie_client::client;
use std::fs;
use std::path::PathBuf;
use wac_graph::{CompositionGraph, EncodeOptions};
use wac_types::Package;

/// Arguments for the `pie submit` command.
#[derive(Args, Debug)]
pub struct SubmitArgs {
    /// Path to the .wasm inferlet file.
    #[arg(value_parser = path::expand_tilde)]
    pub inferlet: PathBuf,
    /// Path to a custom TOML configuration file.
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// The network host to connect to.
    #[arg(long)]
    pub host: Option<String>,
    /// The network port to connect to.
    #[arg(long)]
    pub port: Option<u16>,
    /// The username to use for authentication.
    #[arg(long)]
    pub username: Option<String>,
    /// Path to the private key file to use for authentication.
    #[arg(long)]
    pub private_key_path: Option<PathBuf>,
    /// Run the inferlet in detached mode.
    #[arg(short, long, default_value = "false")]
    pub detached: bool,
    /// Paths to .wasm library files to link with the inferlet.
    #[arg(short, long, value_parser = path::expand_tilde)]
    pub link: Vec<PathBuf>,
    /// Arguments to pass to the inferlet after `--`.
    #[arg(last = true)]
    pub arguments: Vec<String>,
}

/// Compose a component with a library. The library (plug) is linked into the
/// main program (socket).
fn compose_one_component(socket_bytes: Vec<u8>, plug_bytes: Vec<u8>) -> Result<Vec<u8>> {
    // Create a new composition graph
    let mut graph = CompositionGraph::new();

    // Prepare the socket package
    let socket_package = Package::from_bytes("socket", None, socket_bytes, graph.types_mut())
        .context("Failed to load socket component")?;
    let socket_id = graph
        .register_package(socket_package)
        .context("Failed to register socket package")?;

    // Prepare the plug package
    let plug_package = Package::from_bytes("plug", None, plug_bytes, graph.types_mut())
        .context("Failed to load plug component")?;
    let plug_id = graph
        .register_package(plug_package)
        .context("Failed to register plug package")?;

    // Perform the composition: connect the plug's exports to the socket's imports
    wac_graph::plug(&mut graph, vec![plug_id], socket_id)
        .context("Failed to compose components")?;

    // Encode the composed component to bytes
    let composed_bytes = graph
        .encode(EncodeOptions::default())
        .context("Failed to encode composed component")?;

    Ok(composed_bytes)
}

/// Compose a component with multiple libraries. The libraries are linked into the main program
/// sequentially in the order of the library paths.
///
/// Notably, the composition is done iteratively. At each iteration, a library is linked into
/// the main program. This allows an interface instrumented by a library to be instrumented
/// again by another following library.
fn compose_components(program_bytes: Vec<u8>, library_paths: &[PathBuf]) -> Result<Vec<u8>> {
    let mut socket_bytes = program_bytes;

    for library_path in library_paths {
        let plug_bytes = fs::read(library_path)
            .context(format!("Failed to read library file at {:?}", library_path))?;
        let composed_bytes = compose_one_component(socket_bytes, plug_bytes)?;
        socket_bytes = composed_bytes;
    }

    Ok(socket_bytes)
}

/// Handles the `pie submit` command.
///
/// This function:
/// 1. Reads configuration from the specified config file or default config
/// 2. Creates a client configuration from config and command-line arguments
/// 3. Connects to the existing Pie engine server
/// 4. If libraries are specified, composes them with the inferlet on the client side
/// 5. Submits the composed inferlet with the provided arguments
/// 6. In non-detached mode, streams the inferlet output with signal handling:
///    - Ctrl-C (SIGINT): Terminates the inferlet on the server
///    - Ctrl-D (EOF): Detaches from the inferlet (continues running on server)
pub async fn handle_submit_command(args: SubmitArgs) -> Result<()> {
    let client_config = engine::ClientConfig::new(
        args.config,
        args.host,
        args.port,
        args.username,
        args.private_key_path,
    )?;

    let client = engine::connect_and_authenticate(&client_config).await?;

    // Read the main inferlet
    let inferlet_blob = fs::read(&args.inferlet)
        .context(format!("Failed to read Wasm file at {:?}", args.inferlet))?;

    // If libraries are specified, compose them with the main inferlet
    let final_blob = if args.link.is_empty() {
        inferlet_blob
    } else {
        compose_components(inferlet_blob, &args.link)
            .context("Failed to compose inferlet with libraries")?
    };

    // Calculate the hash of the final composed blob
    let hash = client::hash_blob(&final_blob);
    println!("Final inferlet hash: {}", hash);

    // Upload the composed inferlet to the server
    if !client.program_exists(&hash).await? {
        client.upload_program(&final_blob).await?;
        println!("✅ Inferlet upload successful.");
    } else {
        println!("Inferlet already exists on server.");
    }

    let cmd_name = args
        .inferlet
        .file_stem()
        .context("Inferlet path must have a valid file name")?
        .to_string_lossy()
        .to_string();

    let instance = client
        .launch_instance(hash, cmd_name, args.arguments, args.detached)
        .await?;

    println!("✅ Inferlet launched with ID: {}", instance.id());

    if !args.detached {
        engine::stream_inferlet_output(instance, client).await?;
    }

    Ok(())
}
