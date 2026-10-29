use crate::error::{Error, Result};
use crate::fire::{FoldLen, Masking, RsVerb, buffered_reads};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    pub pages: Vec<u32>,
    pub from: u32,
    pub count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct LanePlan {
    pub replay: u32,
    pub commit: u32,
    pub gather: Option<Run>,
    pub gathers: Vec<Run>,
    pub scatter: Option<Run>,
}

impl LanePlan {
    pub fn of(
        verb: &RsVerb,
        rows: u32,
        first: u32,
        mask: Option<&Masking>,
        device_fold: Option<u32>,
    ) -> Result<LanePlan> {
        let RsVerb::Buffer {
            pages,
            base,
            folded,
            fold,
        } = verb
        else {
            return Ok(LanePlan {
                commit: rows,
                ..LanePlan::default()
            });
        };
        if first < *folded || *folded < *base {
            return Err(Error::Invalid(format!(
                "its rows land at slot {first}, and its state holds slots below {folded} \
                 with its buffer from slot {base}; a lane writes past what its state holds, \
                 into its buffer"
            )));
        }
        let reads = buffered_reads(mask, *folded, first, rows)?;
        let replay: u32 = reads
            .iter()
            .map(|r| u32::try_from(r.len()).unwrap_or(0))
            .sum();
        let fold = match fold {
            FoldLen::Host(n) => *n,
            FoldLen::Device => device_fold.ok_or_else(|| {
                Error::Invalid(
                    "it states a device-resident fold length, and the program attached to \
                     it resolved no `fold` port this fire"
                        .into(),
                )
            })?,
        };
        let to = folded.saturating_add(fold).min(first + rows);
        let below: u32 = reads
            .iter()
            .map(|span| span.end.min(to).saturating_sub(span.start))
            .sum();
        let commit = below + to.saturating_sub(first).min(rows);
        let at = first - base;
        let gathers: Vec<Run> = reads
            .into_iter()
            .filter(|span| !span.is_empty())
            .map(|span| Run {
                pages: pages.clone(),
                from: span.start - base,
                count: span.end - span.start,
            })
            .collect();
        Ok(LanePlan {
            replay,
            commit,
            gather: gathers.first().cloned(),
            gathers,
            scatter: (rows > 0).then(|| Run {
                pages: pages.clone(),
                from: at,
                count: rows,
            }),
        })
    }
}
