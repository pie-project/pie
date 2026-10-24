use engine::fire::Masking;

use crate::error::{Fault, Result};

#[derive(Debug, Clone, Copy)]
pub struct LaneMask<'a> {
    pub mask: Option<&'a Masking>,
    pub have: u32,
    pub rows: u32,
    /// The lane's class attends both ways: every row reads every key the
    /// lane holds (its own rows included), the row's enable flag is 2 and,
    /// without a stated mask, every bit is kept.
    pub wide: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Staged {
    pub bytes: Vec<u8>,
    pub enabled: Vec<u8>,
    pub stride: u32,
}

pub fn stage(lanes: &[LaneMask<'_>]) -> Result<Option<Staged>> {
    if lanes.iter().all(|lane| lane.mask.is_none() && !lane.wide) {
        return Ok(None);
    }

    let mut widest = 0u64;
    for (at, lane) in lanes.iter().enumerate() {
        let kv = u64::from(lane.have) + u64::from(lane.rows);
        let Some(masking) = lane.mask else {
            if lane.wide {
                widest = widest.max(kv);
            }
            continue;
        };

        for mask in masking.masks() {
            if mask.total < kv {
                return Err(Fault::Mask {
                    lane: at as u32,
                    stated: mask.total,
                    extent: kv,
                });
            }
        }

        if let Some(stated) = masking.stated_rows()
            && stated != lane.rows as usize
        {
            return Err(Fault::MaskRows {
                lane: at as u32,
                stated: stated as u64,
                rows: lane.rows,
            });
        }
        widest = widest.max(kv);
    }
    let stride = u32::try_from(widest).map_err(|_| Fault::Ceiling {
        what: "key positions in one mask row",
        need: widest,
        have: u64::from(u32::MAX),
    })?;

    let rows: u64 = lanes.iter().map(|lane| u64::from(lane.rows)).sum();
    let cells = rows.saturating_mul(u64::from(stride));
    let size = usize::try_from(cells).map_err(|_| Fault::Ceiling {
        what: "bytes of one fire's mask plane",
        need: cells,
        have: usize::MAX as u64,
    })?;
    let mut out = Staged {
        bytes: vec![0; size],
        enabled: vec![0; rows as usize],
        stride,
    };

    let mut row = 0usize;
    for lane in lanes {
        let kv = u64::from(lane.have) + u64::from(lane.rows);
        let flag = if lane.wide { 2 } else { 1 };
        let Some(masking) = lane.mask else {
            if lane.wide {
                for q in 0..u64::from(lane.rows) {
                    out.enabled[row + q as usize] = flag;
                    for key in 0..kv {
                        keep(&mut out.bytes, stride, row as u64 + q, key);
                    }
                }
            }
            row += lane.rows as usize;
            continue;
        };
        for q in 0..lane.rows as usize {
            out.enabled[row + q] = flag;
        }
        let base = row as u64;
        match masking {
            Masking::Extent(mask) => {
                let mut at_position = 0u64;
                for (index, &run) in mask.runs.iter().enumerate() {
                    let end = at_position.saturating_add(u64::from(run)).min(kv);
                    if index % 2 == 1 {
                        for key in at_position..end {
                            let first = if lane.wide {
                                0
                            } else {
                                key.saturating_sub(u64::from(lane.have))
                            };
                            for q in first..u64::from(lane.rows) {
                                keep(&mut out.bytes, stride, base + q, key);
                            }
                        }
                    }
                    if end == kv {
                        break;
                    }
                    at_position = end;
                }
            }
            Masking::Rows(masks) => {
                for (q, mask) in masks.iter().enumerate() {
                    let q = q as u64;

                    let bound = if lane.wide {
                        kv
                    } else {
                        u64::from(lane.have) + q
                    };
                    let mut at_position = 0u64;
                    for (index, &run) in mask.runs.iter().enumerate() {
                        let end = at_position.saturating_add(u64::from(run)).min(kv);
                        if index % 2 == 1 {
                            for key in at_position..end.min(bound + 1) {
                                keep(&mut out.bytes, stride, base + q, key);
                            }
                        }
                        if end == kv {
                            break;
                        }
                        at_position = end;
                    }
                }
            }
        }
        row += lane.rows as usize;
    }
    Ok(Some(out))
}

fn keep(bytes: &mut [u8], stride: u32, row: u64, key: u64) {
    let cell = row * u64::from(stride) + key;
    if let Some(word) = bytes.get_mut(cell as usize) {
        *word = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::fire::Mask;

    #[test]
    fn mask_every_case() {
        a_wide_lane_keeps_the_keys_after_the_row();
        a_per_row_mask_of_the_wrong_height_is_refused();
        a_short_per_row_mask_is_refused_for_its_extent_first();
        a_fire_with_no_masks_stages_nothing();
        a_mask_short_of_its_lanes_extent_is_refused();
    }

    fn a_wide_lane_keeps_the_keys_after_the_row() {
        let all = Masking::Extent(Mask::new(vec![0, 3], 3));
        let causal = stage(&[LaneMask {
            mask: Some(&all),
            have: 1,
            rows: 2,
            wide: false,
        }])
        .unwrap()
        .unwrap();
        assert_eq!(causal.enabled, vec![1, 1]);
        assert_eq!(causal.bytes, vec![1, 1, 0, 1, 1, 1]);
        let wide = stage(&[LaneMask {
            mask: Some(&all),
            have: 1,
            rows: 2,
            wide: true,
        }])
        .unwrap()
        .unwrap();
        assert_eq!(wide.enabled, vec![2, 2]);
        assert_eq!(wide.bytes, vec![1, 1, 1, 1, 1, 1]);
        let rows = Masking::Rows(vec![Mask::new(vec![0, 3], 3), Mask::new(vec![1, 2], 3)]);
        let wide = stage(&[LaneMask {
            mask: Some(&rows),
            have: 1,
            rows: 2,
            wide: true,
        }])
        .unwrap()
        .unwrap();
        assert_eq!(wide.bytes, vec![1, 1, 1, 0, 1, 1]);
    }

    fn a_per_row_mask_of_the_wrong_height_is_refused() {
        let short = Masking::Rows(vec![Mask::new(vec![0, 3], 3), Mask::new(vec![0, 3], 3)]);
        let refused = stage(&[LaneMask {
            mask: Some(&short),
            have: 0,
            rows: 3,
            wide: false,
        }]);
        assert!(
            matches!(
                refused,
                Err(Fault::MaskRows {
                    lane: 0,
                    stated: 2,
                    rows: 3
                })
            ),
            "a two-row masking on a three-row lane is refused: {refused:?}"
        );
        let said = Fault::MaskRows {
            lane: 0,
            stated: 2,
            rows: 3,
        }
        .to_string();
        assert!(
            said.contains("Masking::Rows"),
            "the refusal names the form it is about: {said}"
        );
    }

    fn a_short_per_row_mask_is_refused_for_its_extent_first() {
        let short = Masking::Rows(vec![Mask::new(vec![0, 2], 2), Mask::new(vec![0, 2], 2)]);
        let refused = stage(&[LaneMask {
            mask: Some(&short),
            have: 1,
            rows: 3,
            wide: false,
        }]);
        assert!(
            matches!(
                refused,
                Err(Fault::Mask {
                    lane: 0,
                    stated: 2,
                    extent: 4
                })
            ),
            "a mask that is both short and miscounted names its extent: {refused:?}"
        );
    }

    fn a_fire_with_no_masks_stages_nothing() {
        let staged = stage(&[LaneMask {
            mask: None,
            have: 4,
            rows: 1,
            wide: false,
        }])
        .expect("no mask is no error");
        assert_eq!(staged, None);
    }

    fn a_mask_short_of_its_lanes_extent_is_refused() {
        let mask = Masking::Extent(Mask::new(vec![0, 4], 4));
        let refused = stage(&[LaneMask {
            mask: Some(&mask),
            have: 7,
            rows: 1,
            wide: false,
        }]);
        assert!(
            matches!(
                refused,
                Err(Fault::Mask {
                    lane: 0,
                    stated: 4,
                    extent: 8
                })
            ),
            "a 4-position mask on an 8-position lane is refused: {refused:?}"
        );
    }
}
