//! Standalone-process deployment form.
//!
//! [`run_as_process`] wraps an [`InProcController`] in a TCP control server for
//! the distributed topology, where workers reach the coordinator over the
//! network (`--controller=addr`). It is the *same* coordination logic the
//! on-device worker links in-proc — only the shell (a socket + a thread per
//! connection) differs. Workers dial it with [`crate::RemoteController`].

use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::controller::{ControllerConfig, InProcController};
use crate::error::{ControllerError, Result};
use crate::protocol::ControlRequest;
use crate::rpc::{dispatch, read_message, write_message};

/// Configuration for the standalone controller process.
#[derive(Debug, Clone)]
pub struct ProcessConfig {
    /// Address the control endpoint listens on (e.g. `"0.0.0.0:7000"`).
    pub listen_addr: String,
    /// Coordination knobs passed through to the embedded [`InProcController`].
    pub controller: ControllerConfig,
    /// How often the liveness clock advances (a missed report ages a worker out
    /// of routing). See [`ControllerConfig`] for the grading thresholds.
    pub tick_interval: Duration,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:7000".to_string(),
            controller: ControllerConfig::default(),
            tick_interval: Duration::from_secs(1),
        }
    }
}

/// Run the controller as a standalone process: bind the control endpoint and
/// serve [`ControlRequest`]s until the process is stopped.
///
/// One [`InProcController`] is shared (`Arc`) across a thread per connection and
/// a background liveness ticker. The accept loop runs until the listener errors.
pub fn run_as_process(config: ProcessConfig) -> Result<()> {
    let controller = Arc::new(InProcController::new(config.controller));
    let listener = TcpListener::bind(&config.listen_addr)
        .map_err(|e| ControllerError::Transport(format!("bind {}: {e}", config.listen_addr)))?;
    tracing::info!(listen_addr = %config.listen_addr, "pie-controller serving control plane");

    // Background liveness ticker: ages out workers that stop reporting.
    {
        let controller = Arc::clone(&controller);
        let interval = config.tick_interval;
        thread::spawn(move || {
            loop {
                thread::sleep(interval);
                controller.tick();
            }
        });
    }

    for stream in listener.incoming() {
        let stream = stream.map_err(|e| ControllerError::Transport(format!("accept: {e}")))?;
        let controller = Arc::clone(&controller);
        thread::spawn(move || {
            if let Err(e) = serve_connection(stream, &controller) {
                tracing::warn!(error = %e, "control connection ended");
            }
        });
    }
    Ok(())
}

/// Serve one worker connection: decode → dispatch → encode, until the peer
/// closes the connection.
fn serve_connection(stream: TcpStream, controller: &InProcController) -> Result<()> {
    let peer = stream.peer_addr().ok();
    let mut reader = stream
        .try_clone()
        .map_err(|e| ControllerError::Transport(format!("clone stream: {e}")))?;
    let mut writer = stream;
    while let Some(req) = read_message::<_, ControlRequest>(&mut reader)? {
        let resp = dispatch(controller, req);
        write_message(&mut writer, &resp)?;
    }
    tracing::debug!(?peer, "control connection closed");
    Ok(())
}
