//! Descriptor-ports → `LaunchPlan` geometry mapping (thrust-3 P2c-fire).
//!
//! A **pure function** from a trace container's descriptor ports (+ the current
//! per-channel values at fire time) to the request's forward geometry: the
//! token family (`embed_tokens`/`positions`/`embed_indptr`/`readout` →
//! `token_ids`/`position_ids`/`qo_indptr`/`sampling_*`) and the port-provided KV
//! family (`pages`/`page_indptr`/`kv_len` → `kv_page_indices`/`kv_page_indptr`/
//! `kv_last_page_lens`). Unit-testable in isolation against the locked
//! bound program contract — no driver decode, no GPU.
//!
//! The one geometry piece that is NOT pure-from-ports is the **sugar** KV arity
//! (`attn_working_set(&ws, &len)`, only `kv_len` bound): the page indices come
//! from the instance's `WorkingSet` allocation, not a port — that derivation is
//! staged for the fire wiring (needs the ws page map). This mapper fills every
//! field the ports DO provide; absent-port fields stay empty for the caller.
//!
//! Complete pipeline domain API: some methods here (relaxed geometry
//! variants, per-channel introspection, the pure `instantiate`/registry
//! probe entry points, device-geometry lease internals) are not yet
//! called by the current single-model/mock-driver fire path, but are
//! exercised by this module's own unit tests and reserved for upcoming
//! wiring (multi-pass channels, device-geometry beams) — kept rather
//! than deleted, allowed rather than silently masked.
#![allow(dead_code)]

use pie_ptir::container::{PortSource, TraceContainer};
use pie_ptir::registry::Port;

/// The forward geometry a PTIR pass contributes to a `LaunchPlan`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReqGeometry {
    /// Input token ids (from `embed_tokens`).
    pub token_ids: Vec<u32>,
    /// RoPE positions (from `positions`, else append-order `0..nnz`).
    pub position_ids: Vec<u32>,
    /// Per-lane token CSR (from `embed_indptr`, else one lane over all tokens).
    pub qo_indptr: Vec<u32>,
    /// KV page slot ids (from `pages`; empty for the sugar arity — ws-derived).
    pub kv_page_indices: Vec<u32>,
    /// Per-lane page CSR (from `page_indptr`; empty for the sugar arity).
    pub kv_page_indptr: Vec<u32>,
    /// Valid tokens in each lane's last KV page (derived from `kv_len`).
    pub kv_last_page_lens: Vec<u32>,
    /// Read-out positions (from `readout`, else the last token of each lane).
    pub sampling_indices: Vec<u32>,
    /// Per-lane read-out CSR.
    pub sampling_indptr: Vec<u32>,
}

/// A geometry-mapping failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeometryError {
    /// A port bound to a channel had no value at fire time (unfilled cell).
    MissingChannelValue { port: Port, channel: u32 },
    /// A port's byte payload isn't a whole number of `u32`s.
    BadPayload { port: Port, bytes: usize },
    /// No `embed_tokens` port — every pass embeds tokens (§5.1).
    NoEmbed,
}

impl ReqGeometry {
    /// Write this geometry into a [`LaunchPlan`]'s forward-geometry fields
    /// (the host-known prefill for a PTIR fire). One-to-one with the request's
    /// `token_ids`/`position_ids`/`qo_indptr`/`kv_page_*`/`kv_last_page_lens`/
    /// `sampling_*` — leaves every other field (samplers, masks, carrier) intact.
    pub fn apply_to(&self, req: &mut crate::driver::LaunchPlan) {
        req.token_ids = self.token_ids.clone();
        req.position_ids = self.position_ids.clone();
        req.qo_indptr = self.qo_indptr.clone();
        req.kv_page_indices = self.kv_page_indices.clone();
        req.kv_page_indptr = self.kv_page_indptr.clone();
        req.kv_last_page_lens = self.kv_last_page_lens.clone();
        req.sampling_indices = self.sampling_indices.clone();
        req.sampling_indptr = self.sampling_indptr.clone();
    }
}

/// Per-channel values at fire time: `values[i]` is channel `i`'s current cell
/// bytes (little-endian, per its dtype), or `None` if unfilled.
pub type ChannelValues<'a> = &'a [Option<Vec<u8>>];

/// Map a container's ports to the forward geometry (P2c-fire, pure).
pub fn map_geometry(
    container: &TraceContainer,
    values: ChannelValues<'_>,
    page_size: u32,
) -> Result<ReqGeometry, GeometryError> {
    map_geometry_impl(container, values, page_size, false)
}

/// **Relaxed** geometry map for a DEVICE-GEOMETRY fire (plan W3.4): a descriptor
/// port bound to a device-produced channel that has no host-known value at fire
/// time leaves its wire field EMPTY instead of erroring
/// ([`GeometryError::MissingChannelValue`]). The driver resolves those ports
/// pre-forward from the channel cells (W1.1); the host still prefills whatever
/// const / host-known ports it can. `NoEmbed` is likewise relaxed (empty
/// tokens). `BadPayload` still errors — a malformed const is a real bug.
pub fn map_geometry_relaxed(
    container: &TraceContainer,
    values: ChannelValues<'_>,
    page_size: u32,
) -> Result<ReqGeometry, GeometryError> {
    map_geometry_impl(container, values, page_size, true)
}

fn map_geometry_impl(
    container: &TraceContainer,
    values: ChannelValues<'_>,
    page_size: u32,
    relaxed: bool,
) -> Result<ReqGeometry, GeometryError> {
    let mut g = ReqGeometry::default();

    // -- token family --
    // In relaxed mode a missing embed channel leaves tokens empty (the driver
    // reads them pre-forward); strict mode errors as before.
    let tokens = match resolve_opt(container, values, Port::EmbedTokens, relaxed)? {
        Some(t) => t,
        None if relaxed => Vec::new(),
        None => return Err(GeometryError::NoEmbed),
    };
    g.token_ids = as_u32(Port::EmbedTokens, &tokens)?;
    let nnz = g.token_ids.len() as u32;

    g.qo_indptr = match resolve_opt(container, values, Port::EmbedIndptr, relaxed)? {
        Some(b) => as_u32(Port::EmbedIndptr, &b)?,
        None => vec![0, nnz], // one lane over all tokens
    };
    let lanes = g.qo_indptr.len().saturating_sub(1);

    g.position_ids = match resolve_opt(container, values, Port::Positions, relaxed)? {
        Some(b) => as_u32(Port::Positions, &b)?,
        None => (0..nnz).collect(), // append order (caller rebases by seq_len)
    };

    // read-out: explicit positions, else the last token of each lane.
    match resolve_opt(container, values, Port::Readout, relaxed)? {
        Some(b) => {
            g.sampling_indices = as_u32(Port::Readout, &b)?;
            let n = g.sampling_indices.len() as u32;
            g.sampling_indptr = vec![0, n];
        }
        None => {
            g.sampling_indices = (0..lanes)
                .map(|l| g.qo_indptr[l + 1].saturating_sub(1))
                .collect();
            g.sampling_indptr = (0..=lanes as u32).collect();
        }
    }

    // -- KV family (port-provided; sugar arity leaves indices/indptr empty) --
    if let Some(b) = resolve_opt(container, values, Port::Pages, relaxed)? {
        g.kv_page_indices = as_u32(Port::Pages, &b)?;
    }
    if let Some(b) = resolve_opt(container, values, Port::PageIndptr, relaxed)? {
        g.kv_page_indptr = as_u32(Port::PageIndptr, &b)?;
    }
    if let Some(b) = resolve_opt(container, values, Port::KvLen, relaxed)? {
        let kv_len = as_u32(Port::KvLen, &b)?;
        g.kv_last_page_lens = kv_len
            .iter()
            .map(|&len| last_page_len(len, page_size))
            .collect();
    }

    Ok(g)
}

/// Valid tokens in the last KV page for a physical span `len` (§5.1): a full
/// span ends the page exactly (`page_size`), else the remainder; `0` for empty.
fn last_page_len(len: u32, page_size: u32) -> u32 {
    if len == 0 || page_size == 0 {
        0
    } else {
        ((len - 1) % page_size) + 1
    }
}

/// Resolve a port's value: its const payload, or the current value of the
/// channel it binds. `None` if the container has no such port.
fn resolve(
    container: &TraceContainer,
    values: ChannelValues<'_>,
    port: Port,
) -> Result<Option<Vec<u8>>, GeometryError> {
    resolve_opt(container, values, port, false)
}

/// Like [`resolve`] but, in `relaxed` mode, a port bound to a channel with no
/// host-known value returns `None` (the driver resolves it pre-forward, W1.1)
/// instead of erroring [`GeometryError::MissingChannelValue`].
fn resolve_opt(
    container: &TraceContainer,
    values: ChannelValues<'_>,
    port: Port,
    relaxed: bool,
) -> Result<Option<Vec<u8>>, GeometryError> {
    let Some(binding) = container.ports.iter().find(|p| p.port == port) else {
        return Ok(None);
    };
    match &binding.source {
        PortSource::Const { data, .. } => Ok(Some(data.clone())),
        PortSource::Channel(c) => match values.get(*c as usize).and_then(|v| v.clone()) {
            Some(v) => Ok(Some(v)),
            None if relaxed => Ok(None),
            None => Err(GeometryError::MissingChannelValue { port, channel: *c }),
        },
    }
}

/// Reinterpret a little-endian byte payload as `u32`s (4 bytes each). Token ids
/// stored `i32` reinterpret bit-for-bit (the driver's `token_ids` is `u32`).
fn as_u32(port: Port, bytes: &[u8]) -> Result<Vec<u32>, GeometryError> {
    if bytes.len() % 4 != 0 {
        return Err(GeometryError::BadPayload {
            port,
            bytes: bytes.len(),
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_ptir::container::{
        ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
    };
    use pie_ptir::op::Op;
    use pie_ptir::registry::Stage;
    use pie_ptir::types::{DType, Shape};

    fn u32_bytes(v: &[u32]) -> Vec<u8> {
        v.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
    fn const_port(port: Port, words: &[u32]) -> PortBinding {
        PortBinding {
            port,
            source: PortSource::Const {
                dtype: DType::U32,
                shape: Shape::vector(words.len() as u32),
                data: u32_bytes(words),
            },
        }
    }
    fn chan(shape: Shape, dtype: DType) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: HostRole::None,
            seeded: true,
        }
    }

    /// §3 sugar: embed tok (chan 0) + embed_indptr const [0,1] + kv_len (chan 1).
    fn section3_container() -> TraceContainer {
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::I32), // 0 tok
                chan(Shape::vector(1), DType::U32), // 1 len
            ],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
                const_port(Port::EmbedIndptr, &[0, 1]),
                PortBinding {
                    port: Port::KvLen,
                    source: PortSource::Channel(1),
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![Op::ChanTake(0)],
            }],
        }
    }

    #[test]
    fn section3_single_seq_decode_geometry() {
        let c = section3_container();
        // tok = [42] (i32), len = [5] (u32); page_size 16.
        let values: Vec<Option<Vec<u8>>> = vec![
            Some(42i32.to_le_bytes().to_vec()),
            Some(5u32.to_le_bytes().to_vec()),
        ];
        let g = map_geometry(&c, &values, 16).unwrap();

        assert_eq!(g.token_ids, vec![42]);
        assert_eq!(g.qo_indptr, vec![0, 1], "one lane, one token");
        assert_eq!(g.position_ids, vec![0], "append order, one token");
        assert_eq!(
            g.sampling_indices,
            vec![0],
            "read out the lane's last (only) token"
        );
        assert_eq!(g.sampling_indptr, vec![0, 1]);
        assert_eq!(
            g.kv_last_page_lens,
            vec![5],
            "len 5 in a 16-page → last page holds 5"
        );
        assert!(
            g.kv_page_indices.is_empty(),
            "sugar arity: page indices are ws-derived (staged)"
        );
    }

    /// §6.2 rectangular batch: B=2 lanes, full KV arity from ports.
    fn beam_container(b: u32, p: u32) -> TraceContainer {
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(b), DType::I32),    // 0 toks
                chan(Shape::vector(b), DType::U32),    // 1 pos
                chan(Shape::matrix(b, p), DType::U32), // 2 pages
                chan(Shape::vector(b), DType::U32),    // 3 klen
            ],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
                const_port(Port::EmbedIndptr, &(0..=b).collect::<Vec<_>>()),
                PortBinding {
                    port: Port::Positions,
                    source: PortSource::Channel(1),
                },
                PortBinding {
                    port: Port::Pages,
                    source: PortSource::Channel(2),
                },
                const_port(
                    Port::PageIndptr,
                    &(0..=b).map(|i| i * p).collect::<Vec<_>>(),
                ),
                PortBinding {
                    port: Port::KvLen,
                    source: PortSource::Channel(3),
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![Op::ChanTake(0)],
            }],
        }
    }

    #[test]
    fn beam_rectangular_batch_geometry() {
        let c = beam_container(2, 3);
        let values: Vec<Option<Vec<u8>>> = vec![
            Some(u32_bytes(&[100, 200])), // 0 toks (reinterpret i32→u32)
            Some(u32_bytes(&[7, 9])),     // 1 pos
            Some(u32_bytes(&[10, 11, 12, 20, 21, 22])), // 2 pages [B,P] flat
            Some(u32_bytes(&[20, 33])),   // 3 klen (physical spans)
        ];
        let g = map_geometry(&c, &values, 16).unwrap();

        assert_eq!(g.token_ids, vec![100, 200]);
        assert_eq!(g.qo_indptr, vec![0, 1, 2], "one token per lane");
        assert_eq!(g.position_ids, vec![7, 9]);
        assert_eq!(
            g.sampling_indices,
            vec![0, 1],
            "last token of each of 2 lanes"
        );
        assert_eq!(g.sampling_indptr, vec![0, 1, 2]);
        assert_eq!(g.kv_page_indices, vec![10, 11, 12, 20, 21, 22]);
        assert_eq!(g.kv_page_indptr, vec![0, 3, 6]);
        // len 20 in 16-page → last page holds 4; len 33 → 1.
        assert_eq!(g.kv_last_page_lens, vec![4, 1]);
    }

    #[test]
    fn missing_channel_value_errors() {
        let c = section3_container();
        let values: Vec<Option<Vec<u8>>> = vec![None, Some(5u32.to_le_bytes().to_vec())];
        let e = map_geometry(&c, &values, 16).unwrap_err();
        assert_eq!(
            e,
            GeometryError::MissingChannelValue {
                port: Port::EmbedTokens,
                channel: 0
            }
        );
    }

    /// W3.4 relaxed gate: a device-geometry fire whose port channels are unfilled
    /// leaves the wire fields EMPTY (the driver resolves them pre-forward, W1.1)
    /// instead of erroring — but const/host-known ports still prefill.
    #[test]
    fn relaxed_leaves_device_ports_empty() {
        let c = beam_container(2, 3);
        // Nothing produced yet (all device channels unfilled).
        let values: Vec<Option<Vec<u8>>> = vec![None, None, None, None];
        let g = map_geometry_relaxed(&c, &values, 16).unwrap();
        assert!(g.token_ids.is_empty(), "device embed_tokens left empty");
        assert!(g.kv_page_indices.is_empty(), "device pages left empty");
        assert!(g.kv_last_page_lens.is_empty(), "device kv_len left empty");
        // The const embed_indptr port ([0,1,2]) still prefills.
        assert_eq!(
            g.qo_indptr,
            vec![0, 1, 2],
            "const port prefilled even in relaxed mode"
        );
        // strict mode still errors on the same input.
        assert!(
            map_geometry(&c, &values, 16).is_err(),
            "strict gate still errors"
        );
    }

    #[test]
    fn last_page_len_boundaries() {
        assert_eq!(last_page_len(0, 16), 0);
        assert_eq!(last_page_len(1, 16), 1);
        assert_eq!(last_page_len(16, 16), 16, "a full page ends exactly");
        assert_eq!(last_page_len(17, 16), 1);
        assert_eq!(last_page_len(32, 16), 16);
    }
}
