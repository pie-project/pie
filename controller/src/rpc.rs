//! Control-RPC: length-prefixed framing + the distributed [`RemoteController`].
//!
//! The minimal concrete control channel: each message is a `u32` big-endian
//! length prefix followed by an rmp-serde (MessagePack) body — the wire format
//! the rest of the repo already uses. Low-rate metadata (register / report /
//! route / pair) only; KV never travels here (that's the data-plane transport).
//!
//! Blocking `std::net` — control ops happen at worker startup plus periodic
//! reports, so a thread-per-connection server and a blocking client are
//! sufficient. The format is swappable behind the [`crate::Controller`] trait if
//! perf ever warrants (e.g. rkyv), with no logic churn.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Mutex;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::controller::Controller;
use crate::error::{ControllerError, Result};
use crate::protocol::{ControlRequest, ControlResponse};
use pie_schema::{LoadState, Placement, RequestId, RequestMeta, WorkerId, WorkerInfo};

fn transport(e: impl std::fmt::Display) -> ControllerError {
    ControllerError::Transport(e.to_string())
}

/// Hard cap on a single control-RPC frame. Control messages are tiny
/// (register/report/route/pair) — well under a KiB — so a 1 MiB ceiling is
/// generous. Rejecting an oversized length prefix stops a malicious or buggy
/// peer from driving a multi-GiB allocation off a 4-byte length field.
const MAX_FRAME_LEN: usize = 1024 * 1024;

/// Write one length-prefixed (`u32` BE) rmp-serde frame and flush.
pub(crate) fn write_message<W: Write, T: Serialize>(w: &mut W, msg: &T) -> Result<()> {
    let bytes = rmp_serde::to_vec(msg).map_err(transport)?;
    let len = u32::try_from(bytes.len()).map_err(|_| transport("control message exceeds 4 GiB"))?;
    w.write_all(&len.to_be_bytes()).map_err(transport)?;
    w.write_all(&bytes).map_err(transport)?;
    w.flush().map_err(transport)?;
    Ok(())
}

/// Read one frame. `Ok(None)` on a clean EOF at a frame boundary (peer closed).
pub(crate) fn read_message<R: Read, T: DeserializeOwned>(r: &mut R) -> Result<Option<T>> {
    let mut len_buf = [0u8; 4];
    let mut filled = 0;
    while filled < 4 {
        match r.read(&mut len_buf[filled..]).map_err(transport)? {
            0 if filled == 0 => return Ok(None),
            0 => return Err(transport("eof mid frame header")),
            n => filled += n,
        }
    }
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_FRAME_LEN {
        return Err(transport(format!(
            "control frame too large: {len} bytes (max {MAX_FRAME_LEN})"
        )));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(transport)?;
    let msg = rmp_serde::from_slice(&buf).map_err(transport)?;
    Ok(Some(msg))
}

/// Server-side dispatch: map a decoded request onto a [`Controller`] and render
/// the response. Coordination errors become [`ControlResponse::Error`] so they
/// relay to the remote caller rather than dropping the connection.
pub(crate) fn dispatch(controller: &dyn Controller, req: ControlRequest) -> ControlResponse {
    match req {
        ControlRequest::Register(info) => match controller.register(info) {
            Ok(id) => ControlResponse::Registered(id),
            Err(e) => ControlResponse::Error(e.to_string()),
        },
        ControlRequest::Report { worker, load } => match controller.report(worker, load) {
            Ok(()) => ControlResponse::Ack,
            Err(e) => ControlResponse::Error(e.to_string()),
        },
        ControlRequest::Route(meta) => match controller.route(&meta) {
            Ok(p) => ControlResponse::Routed(p),
            Err(e) => ControlResponse::Error(e.to_string()),
        },
        ControlRequest::Pair(req) => match controller.pair(req) {
            Ok((p, d)) => ControlResponse::Paired(p, d),
            Err(e) => ControlResponse::Error(e.to_string()),
        },
    }
}

/// Distributed [`Controller`] — dials a standalone controller process and frames
/// each trait call as a [`ControlRequest`]. Blocking I/O behind a mutex so the
/// trait's shared-`&self` contract holds.
#[derive(Debug)]
pub struct RemoteController {
    stream: Mutex<TcpStream>,
}

impl RemoteController {
    /// Connect to the controller's control endpoint.
    pub fn connect(addr: impl ToSocketAddrs) -> Result<Self> {
        let stream = TcpStream::connect(addr).map_err(transport)?;
        stream.set_nodelay(true).ok();
        Ok(Self {
            stream: Mutex::new(stream),
        })
    }

    fn request(&self, req: ControlRequest) -> Result<ControlResponse> {
        let mut stream = self.stream.lock().expect("rpc stream mutex poisoned");
        write_message(&mut *stream, &req)?;
        read_message::<_, ControlResponse>(&mut *stream)?
            .ok_or_else(|| transport("controller closed the connection"))
    }
}

impl Controller for RemoteController {
    fn register(&self, worker: WorkerInfo) -> Result<WorkerId> {
        match self.request(ControlRequest::Register(worker))? {
            ControlResponse::Registered(id) => Ok(id),
            other => Err(unexpected("register", other)),
        }
    }

    fn report(&self, worker: WorkerId, load: LoadState) -> Result<()> {
        match self.request(ControlRequest::Report { worker, load })? {
            ControlResponse::Ack => Ok(()),
            other => Err(unexpected("report", other)),
        }
    }

    fn route(&self, req: &RequestMeta) -> Result<Placement> {
        match self.request(ControlRequest::Route(*req))? {
            ControlResponse::Routed(p) => Ok(p),
            other => Err(unexpected("route", other)),
        }
    }

    fn pair(&self, req: RequestId) -> Result<(WorkerId, WorkerId)> {
        match self.request(ControlRequest::Pair(req))? {
            ControlResponse::Paired(p, d) => Ok((p, d)),
            other => Err(unexpected("pair", other)),
        }
    }
}

/// Interpret a non-matching response: a controller-side [`ControlResponse::Error`]
/// becomes [`ControllerError::Remote`]; anything else is a protocol violation.
fn unexpected(op: &str, resp: ControlResponse) -> ControllerError {
    match resp {
        ControlResponse::Error(msg) => ControllerError::Remote(msg),
        other => ControllerError::Transport(format!("unexpected {op} response: {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::{ControllerConfig, InProcController};
    use pie_schema::Role;
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn remote_round_trip_over_socket() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let controller = Arc::new(InProcController::new(ControllerConfig::default()));

        let server = {
            let controller = Arc::clone(&controller);
            thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                let mut reader = stream.try_clone().unwrap();
                let mut writer = stream;
                while let Some(req) = read_message::<_, ControlRequest>(&mut reader).unwrap() {
                    let resp = dispatch(&*controller, req);
                    write_message(&mut writer, &resp).unwrap();
                }
            })
        };

        let client = RemoteController::connect(addr).unwrap();
        let w = client
            .register(WorkerInfo {
                control_addr: "x".to_string(),
                preferred_role: Some(Role::Prefill),
            })
            .unwrap();
        assert_eq!(w, WorkerId(0));

        client
            .report(
                w,
                LoadState {
                    active_requests: 1,
                    kv_pages_free: 5,
                },
            )
            .unwrap();

        let placement = client
            .route(&RequestMeta {
                id: RequestId(7),
                prompt_tokens: 3,
            })
            .unwrap();
        assert_eq!(placement.worker, w);
        assert_eq!(client.pair(RequestId(7)).unwrap(), (w, w));

        // a coordination error relays back as ControllerError::Remote
        let err = client
            .report(
                WorkerId(99),
                LoadState {
                    active_requests: 0,
                    kv_pages_free: 0,
                },
            )
            .unwrap_err();
        assert!(matches!(err, ControllerError::Remote(_)));

        drop(client); // closes the connection → server loop returns
        server.join().unwrap();
    }

    #[test]
    fn read_message_rejects_oversized_frame() {
        // An oversized length prefix is rejected before any allocation, so a
        // malicious/buggy peer can't drive a multi-GiB alloc off 4 bytes.
        let oversized = (MAX_FRAME_LEN as u32 + 1).to_be_bytes();
        let mut cursor = std::io::Cursor::new(oversized.to_vec());
        let err = read_message::<_, ControlResponse>(&mut cursor).unwrap_err();
        assert!(matches!(err, ControllerError::Transport(_)));
    }
}
