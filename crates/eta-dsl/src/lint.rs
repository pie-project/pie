use alloc::vec::Vec;

use eta_ir::registry::Stage;

use crate::context::{ChannelRef, SinkCall};
use crate::error::{Endpoint, Span, TraceError};

pub(crate) fn lint(
    channels: &[ChannelRef],
    sinks: &[(Stage, SinkCall)],
    errs: &mut Vec<TraceError>,
) {
    for ch in channels {
        let st = ch.borrow();
        let name = st.name.clone();

        let host_writes = st.host_puts.first().copied();
        let host_consumes = st
            .host_takes
            .first()
            .copied()
            .or_else(|| st.host_reads.first().copied());
        let stage_puts = !st.prog_puts.is_empty();
        let stage_consumes = !st.prog_takes.is_empty() || !st.desc_takes.is_empty();

        if let (Some(w), Some(c)) = (host_writes, host_consumes) {
            errs.push(TraceError::DoubleEndpoint {
                channel: name.clone(),
                role: "host",
                first: (Endpoint::Host, w),
                second: (Endpoint::Host, c),
            });
        }

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

    for (stage, s) in sinks {
        if *stage != Stage::Prologue {
            errs.push(TraceError::SinkMisplacement {
                sink: s.name.clone(),
                stage: *stage,
                span: s.span,
            });
        }
    }
}
