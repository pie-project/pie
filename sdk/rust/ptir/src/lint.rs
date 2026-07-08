//! SDK span lints (overview P1.3): SPSC double-endpoint, readiness-direction
//! conflict, and sink stage-precedence — caught during assembly with source
//! spans. Echo's [`bind`](pie_ptir::validate::bind) is the
//! authoritative SPSC gate (host-role vs pass; the descriptor + all stages are
//! one pass endpoint, so a channel touched by both the descriptor and a stage is
//! legal). These run first for friendly, span-rich author errors and mirror
//! echo's model.

use alloc::vec::Vec;

use pie_ptir::registry::{SinkScope, Stage};

use crate::context::{ChannelRef, SinkCall};
use crate::error::{Endpoint, Span, TraceError};

/// Run the span lints over the interned channels + recorded sinks.
pub(crate) fn lint(
    channels: &[ChannelRef],
    sinks: &[(Stage, SinkCall)],
    errs: &mut Vec<TraceError>,
) {
    for ch in channels {
        let st = ch.borrow();
        let name = st.name.clone();

        let host_writes = st.host_puts.first().copied();
        let host_consumes =
            st.host_takes.first().copied().or_else(|| st.host_reads.first().copied());
        let stage_puts = !st.prog_puts.is_empty();
        let stage_consumes = !st.prog_takes.is_empty() || !st.desc_takes.is_empty();

        // Double-endpoint (SPSC, T2): the host claims *both* endpoints (writes
        // and consumes the same channel) — no pass endpoint remains. (Host-vs-
        // stage conflicts are structurally avoided by role derivation; echo's
        // bind is the authoritative gate on the container.)
        if let (Some(w), Some(c)) = (host_writes, host_consumes) {
            errs.push(TraceError::DoubleEndpoint {
                channel: name.clone(),
                role: "host",
                first: (Endpoint::Host, w),
                second: (Endpoint::Host, c),
            });
        }

        // Readiness-direction (T3): a consumed channel must be produced or
        // seeded, else its `take`/`read` can never become full.
        let produced = stage_puts || host_writes.is_some() || st.seeded || st.seed.is_some();
        let consumed = stage_consumes
            || !st.prog_reads.is_empty()
            || !st.desc_reads.is_empty()
            || host_consumes.is_some();
        if consumed && !produced {
            let span = st
                .prog_takes
                .first()
                .map(|(_, s)| *s)
                .or_else(|| st.prog_reads.first().map(|(_, s)| *s))
                .or_else(|| st.desc_takes.first().copied())
                .or(host_consumes)
                .or_else(|| st.desc_reads.first().copied())
                .unwrap_or_else(|| Span::of(core::panic::Location::caller()));
            errs.push(TraceError::ReadinessConflict {
                channel: name,
                detail: alloc::string::String::from("consumed but never produced or seeded"),
                span,
            });
        }
    }

    // Sink stage-precedence (T11).
    for (stage, s) in sinks {
        let ok = match s.scope {
            SinkScope::PassWide => *stage == Stage::Prologue,
            SinkScope::Attention => matches!(stage, Stage::Prologue | Stage::OnAttnProj),
        };
        if !ok {
            errs.push(TraceError::SinkMisplacement {
                sink: s.name.clone(),
                stage: *stage,
                span: s.span,
            });
        }
    }
}
