//! Aux-IPC client for the embedded driver.
//!
//! Speaks the wire protocol defined in `driver/portable/src/aux_ipc.hpp`
//! over a unix domain socket. Replaces the Python `_CtrlClient` in
//! `pie_driver_portable/worker.py` for the standalone path.
//!
//! Wire format (little-endian):
//!
//!   * Page-copy command (CopyD2H/H2D/D2D/H2H):
//!         AuxCmdHeader (24 B) + AuxPagePair (8 B) × n_pages
//!   * LoadAdapter command:
//!         AuxCmdHeader (24 B, n_pages = 0) + AuxLoadAdapterPayload (16 B) +
//!         path bytes (no NUL terminator)
//!   * Ack (after every command):
//!         AuxAck (16 B) — magic, status, req_id
//!
//! The cuda driver currently has a different wire format (see
//! `driver/cuda/src/control_socket.hpp`) reachable only via subprocess
//! `--control-fd`. The standalone's cuda branch will gain a matching
//! `[aux_ipc].socket_path` listener in a follow-up — until then the
//! cold-path RPC dispatcher refuses copy_*/swap_*/load_adapter for
//! cuda with an explicit error rather than silently misrouting.

/// Method tags. Must stay in sync with `driver/portable/src/aux_ipc.hpp`
/// `enum class Method`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum Method {
    CopyD2H = 1,
    CopyH2D = 2,
    CopyD2D = 3,
    CopyH2H = 4,
    LoadAdapter = 5,
}

#[cfg(unix)]
mod unix_impl {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    use anyhow::{Result, anyhow, bail};

    use super::Method;

    const MAGIC: u32 = 0x41454950; // 'PIEA'
    const ACK_MAGIC: u32 = 0x4B414950; // 'PIAK'
    const HEADER_SIZE: usize = 24;
    const PAIR_SIZE: usize = 8;
    const LOAD_ADAPTER_PAYLOAD_SIZE: usize = 16;
    const ACK_SIZE: usize = 16;

    /// Status codes from `AuxAck.status`. Mirrors `enum class Status`.
    fn status_to_str(s: u32) -> &'static str {
        match s {
            0 => "ok",
            1 => "bad_magic",
            2 => "bad_method",
            3 => "out_of_bounds",
            4 => "no_swap_pool",
            5 => "backend_error",
            _ => "unknown",
        }
    }

    /// Connected aux-IPC client. Cheap to clone via `Arc`; the underlying
    /// stream is mutex-protected because cold-path RPC dispatch can race
    /// across threads (one shmem-server poller, plus signal handlers).
    pub struct AuxIpcClient {
        sock: Mutex<UnixStream>,
        /// Surfaced via [`Self::socket_path`] for debug logging; not read
        /// by the dispatcher itself.
        #[allow(dead_code)]
        socket_path: PathBuf,
        next_req_id: AtomicU64,
    }

    impl AuxIpcClient {
        /// Open a connection to the driver's aux socket. The driver's
        /// `AuxServer` is constructed *before* `ready_cb` fires (see
        /// `driver/portable/src/entry.cpp`), so by the time
        /// `EmbeddedDriver::start` returns, this connect is race-free.
        pub fn connect(socket_path: PathBuf) -> Result<Self> {
            let sock = UnixStream::connect(&socket_path)
                .map_err(|e| anyhow!("connect aux-ipc socket {socket_path:?}: {e}"))?;
            Ok(Self {
                sock: Mutex::new(sock),
                socket_path,
                next_req_id: AtomicU64::new(1),
            })
        }

        #[allow(dead_code)] // exposed for debug logging.
        pub fn socket_path(&self) -> &Path {
            &self.socket_path
        }

        /// Issue a page-copy command. `pairs` are written verbatim — caller
        /// is responsible for the runtime's source/destination semantics
        /// (e.g. CopyH2D needs `(cpu_src, gpu_dst)` order).
        pub fn copy(&self, method: Method, pairs: &[(u32, u32)]) -> Result<()> {
            debug_assert!(
                !matches!(method, Method::LoadAdapter),
                "use load_adapter() for LoadAdapter, not copy()"
            );

            let req_id = self.next_req_id.fetch_add(1, Ordering::Relaxed);
            let n_pages = u32::try_from(pairs.len())
                .map_err(|_| anyhow!("aux-ipc: too many pages ({})", pairs.len()))?;

            let mut buf = Vec::with_capacity(HEADER_SIZE + pairs.len() * PAIR_SIZE);
            write_header(&mut buf, method, n_pages, req_id);
            for (src, dst) in pairs {
                buf.extend_from_slice(&src.to_le_bytes());
                buf.extend_from_slice(&dst.to_le_bytes());
            }

            let status = self.send_and_recv(&buf, req_id)?;
            check_status(method, status)
        }

        /// Issue a LoadAdapter command. The driver reads weights from
        /// `path` itself; the runtime's `adapter_data: Vec<u8>` is
        /// materialized to disk by the dispatcher before this is called.
        pub fn load_adapter(&self, adapter_id: u64, path: &Path) -> Result<()> {
            let path_bytes = path
                .to_str()
                .ok_or_else(|| anyhow!("aux-ipc: adapter path is not utf-8: {path:?}"))?
                .as_bytes();
            let path_len = u32::try_from(path_bytes.len()).map_err(|_| {
                anyhow!(
                    "aux-ipc: adapter path too long ({} bytes)",
                    path_bytes.len()
                )
            })?;

            let req_id = self.next_req_id.fetch_add(1, Ordering::Relaxed);

            let mut buf =
                Vec::with_capacity(HEADER_SIZE + LOAD_ADAPTER_PAYLOAD_SIZE + path_bytes.len());
            write_header(&mut buf, Method::LoadAdapter, /*n_pages=*/ 0, req_id);
            buf.extend_from_slice(&adapter_id.to_le_bytes());
            buf.extend_from_slice(&path_len.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
            buf.extend_from_slice(path_bytes);

            let status = self.send_and_recv(&buf, req_id)?;
            check_status(Method::LoadAdapter, status)
        }

        /// Lock the socket, send a fully-formed frame, read the ack,
        /// validate magic + req_id. Returns the raw status code.
        fn send_and_recv(&self, frame: &[u8], expect_req_id: u64) -> Result<u32> {
            let mut sock = self.sock.lock().expect("aux-ipc socket mutex poisoned");
            sock.write_all(frame)
                .map_err(|e| anyhow!("aux-ipc write: {e}"))?;

            let mut ack = [0u8; ACK_SIZE];
            sock.read_exact(&mut ack)
                .map_err(|e| anyhow!("aux-ipc read ack: {e}"))?;

            let magic = u32::from_le_bytes(ack[0..4].try_into().unwrap());
            let status = u32::from_le_bytes(ack[4..8].try_into().unwrap());
            let req_id = u64::from_le_bytes(ack[8..16].try_into().unwrap());

            if magic != ACK_MAGIC {
                bail!("aux-ipc ack: bad magic 0x{magic:08x} (expected 0x{ACK_MAGIC:08x})");
            }
            if req_id != expect_req_id {
                bail!(
                    "aux-ipc ack: req_id mismatch (expected {expect_req_id}, got {req_id}) — \
                 socket frame desync"
                );
            }
            Ok(status)
        }
    }

    fn write_header(buf: &mut Vec<u8>, method: Method, n_pages: u32, req_id: u64) {
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&(method as u32).to_le_bytes());
        buf.extend_from_slice(&n_pages.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
        buf.extend_from_slice(&req_id.to_le_bytes());
    }

    fn check_status(method: Method, status: u32) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            bail!(
                "aux-ipc: {method:?} returned status={status} ({})",
                status_to_str(status)
            )
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::os::unix::net::UnixListener;
        use std::thread;

        /// Read a fixed-size header off `stream` and write a canned ack.
        /// Used by the round-trip tests below to stand in for the C++ aux
        /// server. Returns the parsed (method, n_pages, req_id) tuple.
        fn fake_server(
            listener: UnixListener,
            ack_status: u32,
            body_extra_bytes: usize,
        ) -> thread::JoinHandle<(u32, u32, u64, Vec<u8>)> {
            thread::spawn(move || {
                let (mut conn, _) = listener.accept().unwrap();

                let mut hdr = [0u8; HEADER_SIZE];
                conn.read_exact(&mut hdr).unwrap();
                let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
                assert_eq!(magic, MAGIC);
                let method = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
                let n_pages = u32::from_le_bytes(hdr[8..12].try_into().unwrap());
                let req_id = u64::from_le_bytes(hdr[16..24].try_into().unwrap());

                // Drain page-pair body OR the LoadAdapter payload+path.
                let body_len = (n_pages as usize) * PAIR_SIZE + body_extra_bytes;
                let mut body = vec![0u8; body_len];
                conn.read_exact(&mut body).unwrap();

                // Send ack.
                let mut ack = Vec::with_capacity(ACK_SIZE);
                ack.extend_from_slice(&ACK_MAGIC.to_le_bytes());
                ack.extend_from_slice(&ack_status.to_le_bytes());
                ack.extend_from_slice(&req_id.to_le_bytes());
                conn.write_all(&ack).unwrap();

                (method, n_pages, req_id, body)
            })
        }

        fn fresh_socket() -> (PathBuf, UnixListener) {
            let dir = tempfile::tempdir().unwrap().keep();
            let path = dir.join("aux.sock");
            let listener = UnixListener::bind(&path).unwrap();
            (path, listener)
        }

        #[test]
        fn copy_round_trip_succeeds_on_status_ok() {
            let (path, listener) = fresh_socket();
            let server = fake_server(listener, /*ack_status=*/ 0, /*body_extra=*/ 0);

            let client = AuxIpcClient::connect(path).unwrap();
            let pairs = vec![(10, 20), (11, 21), (12, 22)];
            client.copy(Method::CopyD2H, &pairs).unwrap();

            let (method, n_pages, _req_id, body) = server.join().unwrap();
            assert_eq!(method, Method::CopyD2H as u32);
            assert_eq!(n_pages, 3);
            assert_eq!(body.len(), 3 * PAIR_SIZE);
            // src=10, dst=20 (little-endian u32 pair)
            assert_eq!(&body[0..4], &10u32.to_le_bytes());
            assert_eq!(&body[4..8], &20u32.to_le_bytes());
        }

        #[test]
        fn copy_returns_error_on_nonzero_status() {
            let (path, listener) = fresh_socket();
            let _server = fake_server(listener, /*ack_status=*/ 4, /*body_extra=*/ 0);

            let client = AuxIpcClient::connect(path).unwrap();
            let err = client
                .copy(Method::CopyD2H, &[(1, 2)])
                .unwrap_err()
                .to_string();
            assert!(err.contains("no_swap_pool"), "got: {err}");
            assert!(err.contains("status=4"), "got: {err}");
        }

        #[test]
        fn load_adapter_writes_path_payload() {
            let (path, listener) = fresh_socket();
            // LoadAdapter has a 16-byte payload before the path bytes;
            // payload+path go in body_extra.
            let server = fake_server(
                listener,
                /*ack_status=*/ 0,
                /*body_extra=*/ LOAD_ADAPTER_PAYLOAD_SIZE + b"/tmp/lora.safetensors".len(),
            );

            let client = AuxIpcClient::connect(path).unwrap();
            client
                .load_adapter(0xDEADBEEF, Path::new("/tmp/lora.safetensors"))
                .unwrap();

            let (method, n_pages, _req_id, body) = server.join().unwrap();
            assert_eq!(method, Method::LoadAdapter as u32);
            assert_eq!(n_pages, 0);
            let adapter_id = u64::from_le_bytes(body[0..8].try_into().unwrap());
            let path_len = u32::from_le_bytes(body[8..12].try_into().unwrap());
            assert_eq!(adapter_id, 0xDEADBEEF);
            assert_eq!(path_len as usize, b"/tmp/lora.safetensors".len());
            assert_eq!(&body[16..], b"/tmp/lora.safetensors");
        }

        #[test]
        fn ack_bad_magic_surfaces_clear_error() {
            let dir = tempfile::tempdir().unwrap().keep();
            let path = dir.join("aux.sock");
            let listener = UnixListener::bind(&path).unwrap();

            let _bad = thread::spawn(move || {
                let (mut conn, _) = listener.accept().unwrap();
                let mut hdr = [0u8; HEADER_SIZE];
                conn.read_exact(&mut hdr).unwrap();
                let mut body = [0u8; PAIR_SIZE];
                conn.read_exact(&mut body).unwrap();
                // Corrupt the magic bytes.
                let mut ack = Vec::with_capacity(ACK_SIZE);
                ack.extend_from_slice(&0xCAFEBABEu32.to_le_bytes());
                ack.extend_from_slice(&0u32.to_le_bytes());
                ack.extend_from_slice(&1u64.to_le_bytes());
                conn.write_all(&ack).unwrap();
            });

            let client = AuxIpcClient::connect(path).unwrap();
            let err = client
                .copy(Method::CopyD2D, &[(0, 1)])
                .unwrap_err()
                .to_string();
            assert!(err.contains("bad magic"), "got: {err}");
        }
    }
}

#[cfg(unix)]
pub use unix_impl::*;

#[cfg(windows)]
use std::path::{Path, PathBuf};
#[cfg(windows)]
use std::sync::atomic::AtomicU64;

#[cfg(windows)]
use anyhow::{Result, bail};

/// Windows embedded portable builds currently run without aux IPC; basic
/// inference uses the shmem fast path, while cold-path copy/load-adapter
/// calls return an explicit unsupported error.
#[cfg(windows)]
pub struct AuxIpcClient {
    socket_path: PathBuf,
    #[allow(dead_code)]
    next_req_id: AtomicU64,
}

#[cfg(windows)]
impl AuxIpcClient {
    pub fn connect(socket_path: PathBuf) -> Result<Self> {
        Ok(Self {
            socket_path,
            next_req_id: AtomicU64::new(1),
        })
    }

    #[allow(dead_code)]
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    pub fn copy(&self, _method: Method, _pairs: &[(u32, u32)]) -> Result<()> {
        bail!("aux IPC is not supported on Windows yet")
    }

    pub fn load_adapter(&self, _adapter_id: u64, _path: &Path) -> Result<()> {
        bail!("aux IPC is not supported on Windows yet")
    }
}
