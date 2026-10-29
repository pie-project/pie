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
        let replay = first - folded;
        let reads = buffered_reads(mask, *folded, first, rows)?;
        if reads.iter().map(ExactSizeIterator::len).sum::<usize>() != replay as usize {
            return Err(Error::Invalid(
                "its mask hides buffered slots its recurrent layers would replay; this \
                 engine replays the buffer whole"
                    .into(),
            ));
        }
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
        let at = first - base;
        Ok(LanePlan {
            replay,
            commit: fold.min(replay.saturating_add(rows)),
            gather: (replay > 0).then(|| Run {
                pages: pages.clone(),
                from: at - replay,
                count: replay,
            }),
            scatter: (rows > 0).then(|| Run {
                pages: pages.clone(),
                from: at,
                count: rows,
            }),
        })
    }
}
