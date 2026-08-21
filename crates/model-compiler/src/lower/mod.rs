//! Lowering turns a traced fire into rows, buffers, and a flat launch list.

use std::collections::BTreeMap;
use std::ops::Range;

use model_ir::kernels::{self, Backend};
use model_ir::trace::{DType, Dim, ForwardPlan, GuardPred, Op, OpKind, PeelWindow, ValueId};

/// One row of a fire, in engine seriation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Row {
    pub multi_token: bool,
    pub custom_mask: bool,
    pub hooked: bool,
    pub lora: bool,
    /// Truncated at layer `k`, or `None` for full depth.
    pub depth_k: Option<u32>,
    /// The fire steers graph replay with explicit KV write descriptors.
    pub write_desc: bool,
    /// The fire's attached programs read attention scores.
    pub wants_scores: bool,
    /// This row's logits are read.
    pub samples: bool,
}

/// The region-signature bits, as the wire states them.
pub mod region_sig {
    /// The region's members carry multi-token qo windows.
    pub const MULTI_TOKEN: u32 = 1 << 0;
    /// Attention-stage hook programs.
    pub const HOOK: u32 = 1 << 1;
    /// A user (custom) attention mask.
    pub const MASK: u32 = 1 << 2;
    /// A depth truncation; the region's `k` is its `region_k`.
    pub const TRUNCATED: u32 = 1 << 3;
    /// A span-grouped correction (lora) program.
    pub const LORA: u32 = 1 << 4;
    /// The region's hooks write the `attn_page_mask` sink.
    pub const HOOK_PAGE_MASK: u32 = 1 << 5;
    /// `region_k`'s full-depth sentinel.
    pub const MAX_LAYERS_FULL: u32 = u32::MAX;
}

/// Why a step's tables do not describe its rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionDrift {
    /// The three region arrays disagree on how many regions there are.
    Ragged {
        segments: usize,
        sigs: usize,
        ks: usize,
    },
    /// A region names rows the step does not have.
    OutOfRange {
        region: usize,
        end: u32,
        rows: usize,
    },
    /// Two regions claim one row, or no region claims it.
    DoNotTile { at: u32 },
    /// A read-out names a row past the end of the request that named it.
    NotItsRow {
        request: usize,
        /// In that request's own numbering.
        row: u32,
        /// How many rows the request has.
        span: u32,
    },
    /// Read-out rows with no CSR that says which request named them.
    RaggedReadout {
        rows: usize,
        segments: usize,
        requests: usize,
    },
}

/// The fire's read-out table, and the row geometry that gives it meaning.
#[derive(Clone, Copy, Debug, Default)]
pub struct Readouts<'a> {
    /// Rows read, each numbered inside its own request.
    pub indices: &'a [u32],
    /// Request -> readout CSR over `indices`, empty when none are named.
    pub indptr: &'a [u32],
    /// Request -> row CSR for the fire.
    pub qo_indptr: &'a [u32],
}

impl<'a> Readouts<'a> {
    /// The rows request `r` reads, in its own numbering.
    fn of(&self, r: usize) -> &'a [u32] {
        let (Some(&lo), Some(&hi)) = (self.indptr.get(r), self.indptr.get(r + 1)) else {
            return &[];
        };
        if hi < lo {
            return &[];
        }
        self.indices.get(lo as usize..hi as usize).unwrap_or(&[])
    }

    /// Which rows of the fire this table marks as read.
    fn samples(&self, rows: usize) -> Result<Vec<bool>, RegionDrift> {
        let mut samples = vec![false; rows];
        let requests = self.qo_indptr.len().saturating_sub(1);
        if !self.indices.is_empty() && self.indptr.len() != requests + 1 {
            return Err(RegionDrift::RaggedReadout {
                rows: self.indices.len(),
                segments: self.indptr.len().saturating_sub(1),
                requests,
            });
        }
        for r in 0..requests {
            let (lo, hi) = (self.qo_indptr[r], self.qo_indptr[r + 1]);
            let span = hi.saturating_sub(lo);
            if span == 0 {
                continue;
            }
            let named = self.of(r);
            if named.is_empty() {
                samples[(hi - 1) as usize] = true;
                continue;
            }
            for &row in named {
                if row >= span {
                    return Err(RegionDrift::NotItsRow {
                        request: r,
                        row,
                        span,
                    });
                }
                samples[(lo + row) as usize] = true;
            }
        }
        Ok(samples)
    }
}

impl Row {
    /// The row a region's signature describes.
    #[must_use]
    pub const fn from_region(sig: u32, k: u32, samples: bool) -> Self {
        Self {
            multi_token: sig & region_sig::MULTI_TOKEN != 0,
            custom_mask: sig & region_sig::MASK != 0,
            hooked: sig & (region_sig::HOOK | region_sig::HOOK_PAGE_MASK) != 0,
            lora: sig & region_sig::LORA != 0,
            depth_k: if sig & region_sig::TRUNCATED != 0 && k != region_sig::MAX_LAYERS_FULL {
                Some(k)
            } else {
                None
            },
            write_desc: false,
            wants_scores: false,
            samples,
        }
    }
}

/// The rows a step lowers over, from its region table and readout list.
pub fn rows_from_regions(
    rows: usize,
    readouts: Readouts<'_>,
    region_row_indptr: &[u32],
    region_sig_bits: &[u32],
    region_k: &[u32],
) -> Result<Vec<Row>, RegionDrift> {
    let samples = readouts.samples(rows)?;

    let segments = region_row_indptr.len().saturating_sub(1);
    if region_row_indptr.is_empty() {
        // NO TABLE IS NOT "NO AXES", AND READING IT THAT WAY COST A PREFILL
        // ITS ATTENTION KERNEL.
        //
        // A sig of zero says every axis is off, and `MULTI_TOKEN` is one of
        // them: `GuardPred::WindowOne` reads it as "this row's query window is
        // one row", which is what makes a fire take `sdpa_paged_decode`
        // instead of the tiled or mma kernel. So a 2048-token prefill arriving
        // without a table was lowered onto the per-row decode kernel, which
        // re-reads the whole key run once per query row. It is not wrong --
        // the logits are the same and every MLX gate still passes -- it is
        // quadratic where the tiled form is not. Measured on Llama-3.2-1B by
        // `attention_is_a_minority_of_a_long_prefill`: 2.104e-4 ms per token²
        // against the mma kernel's 2.589e-5, attention 52% of the fire instead
        // of 8%.
        //
        // And an empty table is a PRODUCTION case, not just a fixture's
        // shorthand: `engine::scheduler::batch::planned_region_table` declines
        // to build one whenever the attribution CSR is absent or any member
        // owns no wire row, and returns three empty vectors.
        //
        // The width is knowable without the table, because `qo_indptr` is the
        // same fact the table was built from -- that function's own rule is
        // `w[1] - w[0] > 1` over the request's windows. Deriving it here is
        // therefore not a guess standing in for the table; it is the one
        // remaining reader computing what the absent producer would have said.
        // The other axes are NOT derivable -- a hook, a mask, a LoRA and a
        // depth are the scheduler's claims about a request and nothing in the
        // step implies them -- so they stay off, which is what an absent table
        // has always meant for them.
        let mut sig = vec![0u32; rows];
        for r in 0..readouts.qo_indptr.len().saturating_sub(1) {
            let (lo, hi) = (readouts.qo_indptr[r], readouts.qo_indptr[r + 1]);
            if hi <= lo || hi.saturating_sub(lo) == 1 {
                continue;
            }
            for row in lo..hi.min(u32::try_from(rows).unwrap_or(u32::MAX)) {
                sig[row as usize] = region_sig::MULTI_TOKEN;
            }
        }
        return Ok((0..rows)
            .map(|i| Row::from_region(sig[i], region_sig::MAX_LAYERS_FULL, samples[i]))
            .collect());
    }
    if segments != region_sig_bits.len() || segments != region_k.len() {
        return Err(RegionDrift::Ragged {
            segments,
            sigs: region_sig_bits.len(),
            ks: region_k.len(),
        });
    }

    let mut out: Vec<Option<Row>> = vec![None; rows];
    for region in 0..segments {
        let (start, end) = (region_row_indptr[region], region_row_indptr[region + 1]);
        if end as usize > rows || start > end {
            return Err(RegionDrift::OutOfRange { region, end, rows });
        }
        for row in start..end {
            let slot = &mut out[row as usize];
            if slot.is_some() {
                return Err(RegionDrift::DoNotTile { at: row });
            }
            *slot = Some(Row::from_region(
                region_sig_bits[region],
                region_k[region],
                samples[row as usize],
            ));
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(i, row)| {
            row.ok_or(RegionDrift::DoNotTile {
                at: u32::try_from(i).unwrap_or(u32::MAX),
            })
        })
        .collect()
}

mod buffers;
mod semantics;
mod shapes;
mod walk;

pub use buffers::*;
pub use shapes::*;
pub use walk::*;

#[cfg(test)]
mod region_tests {
    use super::*;

    /// A STEP THAT ARRIVES WITHOUT A REGION TABLE STILL KNOWS HOW WIDE ITS
    /// QUERY WINDOWS ARE.
    ///
    /// `MULTI_TOKEN` is the one axis `qo_indptr` implies, and reading an
    /// absent table as "every axis off" told `GuardPred::WindowOne` that a
    /// 2048-token prefill was a one-row window — which picks the per-row
    /// decode attention kernel over the tiled one and makes the fire
    /// quadratic. The other axes stay off, because nothing in a step implies
    /// a hook, a mask, a LoRA or a truncated depth.
    #[test]
    fn no_region_table_still_reads_a_wide_request_as_multi_token() {
        // Two requests: a four-token prefill and a one-token decode.
        let qo = [0u32, 4, 5];
        let rows = rows_from_regions(
            5,
            Readouts {
                indices: &[],
                indptr: &[],
                qo_indptr: &qo,
            },
            &[],
            &[],
            &[],
        )
        .expect("a step with no table lowers");
        assert!(
            rows[..4].iter().all(|r| r.multi_token),
            "every row of the wide request is a multi-token window"
        );
        assert!(!rows[4].multi_token, "and the decode's row is not");
        assert!(
            rows.iter()
                .all(|r| !r.hooked && !r.custom_mask && !r.lora && r.depth_k.is_none()),
            "no other axis is invented from a step that states none"
        );
    }

    #[test]
    fn each_axis_bit_lands_on_its_own_field() {
        let base = Row {
            samples: true,
            ..Row::default()
        };
        for (bit, expected) in [
            (
                region_sig::MULTI_TOKEN,
                Row {
                    multi_token: true,
                    ..base
                },
            ),
            (
                region_sig::MASK,
                Row {
                    custom_mask: true,
                    ..base
                },
            ),
            (
                region_sig::HOOK,
                Row {
                    hooked: true,
                    ..base
                },
            ),
            (
                region_sig::HOOK_PAGE_MASK,
                Row {
                    hooked: true,
                    ..base
                },
            ),
            (region_sig::LORA, Row { lora: true, ..base }),
        ] {
            assert_eq!(
                Row::from_region(bit, region_sig::MAX_LAYERS_FULL, true),
                expected,
                "bit {bit:#x} landed on the wrong field"
            );
        }
    }

    #[test]
    fn a_truncation_needs_both_the_bit_and_the_depth() {
        assert_eq!(
            Row::from_region(region_sig::TRUNCATED, 7, true).depth_k,
            Some(7)
        );
        assert_eq!(
            Row::from_region(region_sig::TRUNCATED, region_sig::MAX_LAYERS_FULL, true).depth_k,
            None,
            "the bit plus the sentinel is FULL depth, not a truncation at u32::MAX"
        );
        assert_eq!(
            Row::from_region(0, 7, true).depth_k,
            None,
            "no bit, no truncation"
        );
    }

    fn readouts<'a>(indices: &'a [u32], indptr: &'a [u32], qo: &'a [u32]) -> Readouts<'a> {
        Readouts {
            indices,
            indptr,
            qo_indptr: qo,
        }
    }

    #[test]
    fn an_empty_readout_list_samples_each_requests_last_row() {
        let rows = rows_from_regions(3, readouts(&[], &[], &[0, 3]), &[], &[], &[])
            .expect("legacy discipline");
        assert_eq!(
            rows.iter().map(|r| r.samples).collect::<Vec<_>>(),
            [false, false, true],
            "one request: its last row"
        );
        let many = rows_from_regions(3, readouts(&[], &[], &[0, 1, 2, 3]), &[], &[], &[])
            .expect("legacy discipline");
        assert_eq!(
            many.iter().map(|r| r.samples).collect::<Vec<_>>(),
            [true, true, true]
        );
        let named = rows_from_regions(3, readouts(&[0, 1], &[0, 2], &[0, 3]), &[], &[], &[])
            .expect("named readout");
        assert_eq!(
            named.iter().map(|r| r.samples).collect::<Vec<_>>(),
            [true, true, false]
        );
    }

    #[test]
    fn a_readout_row_is_numbered_inside_its_own_request() {
        let rows = rows_from_regions(4, readouts(&[2, 0], &[0, 1, 2], &[0, 3, 4]), &[], &[], &[])
            .expect("a fire");
        assert_eq!(
            rows.iter().map(|r| r.samples).collect::<Vec<_>>(),
            [false, false, true, true]
        );
    }

    #[test]
    fn a_readout_past_its_own_request_is_refused() {
        assert_eq!(
            rows_from_regions(4, readouts(&[2], &[0, 1, 1], &[0, 2, 4]), &[], &[], &[]),
            Err(RegionDrift::NotItsRow {
                request: 0,
                row: 2,
                span: 2
            }),
            "it lands inside the FIRE, so only a per-request bound catches it"
        );
    }

    #[test]
    fn readout_rows_without_a_csr_to_place_them_are_refused() {
        assert_eq!(
            rows_from_regions(4, readouts(&[2, 0], &[], &[0, 3, 4]), &[], &[], &[]),
            Err(RegionDrift::RaggedReadout {
                rows: 2,
                segments: 0,
                requests: 2
            })
        );
    }

    #[test]
    fn a_table_that_does_not_tile_is_refused() {
        let gap = rows_from_regions(3, readouts(&[], &[], &[0, 3]), &[0, 2], &[0], &[u32::MAX]);
        assert_eq!(gap, Err(RegionDrift::DoNotTile { at: 2 }));
        let ragged = rows_from_regions(
            2,
            readouts(&[], &[], &[0, 2]),
            &[0, 1, 2],
            &[0],
            &[u32::MAX, u32::MAX],
        );
        assert!(matches!(ragged, Err(RegionDrift::Ragged { .. })));
        let over = rows_from_regions(2, readouts(&[], &[], &[0, 2]), &[0, 5], &[0], &[u32::MAX]);
        assert!(matches!(over, Err(RegionDrift::OutOfRange { .. })));
    }

    #[test]
    fn a_region_carrying_an_adapter_marks_only_its_own_rows() {
        let rows = rows_from_regions(
            4,
            readouts(&[3], &[0, 1], &[0, 4]),
            &[0, 2, 4],
            &[0, region_sig::LORA],
            &[region_sig::MAX_LAYERS_FULL; 2],
        )
        .expect("a tiling table");
        assert_eq!(
            rows.iter().map(|r| r.lora).collect::<Vec<_>>(),
            [false, false, true, true],
            "the adapter is the second region's, not the fire's"
        );
        assert_eq!(
            rows.iter().map(|r| r.samples).collect::<Vec<_>>(),
            [false, false, false, true]
        );
    }
}

#[cfg(test)]
mod raised_operand {
    use super::*;
    use model_ir::trace::{
        DType, Dim, ForwardPlan, Op, OpKind, PrepKind, Shape as VShape, ValueInfo,
    };

    /// A prep, then a launch that names what it raised.
    ///
    /// Hand-built rather than traced: no DSL states this yet — stage 4 is
    /// where a real launcher takes the operand — and the point of the fixture
    /// is that the LOWERING can carry one before anything asks it to.
    fn plan_with_a_raise() -> ForwardPlan {
        let prep = PrepKind::PrefillAttention { head_dim: 128 };
        ForwardPlan {
            family: "fixture.cuda".to_string(),
            values: vec![
                // 0: the raise the prep publishes.
                ValueInfo::raise(prep.key()),
                // 1: an ordinary activation, so the arena has something to do.
                ValueInfo {
                    shape: VShape(vec![Dim::Tokens, Dim::Const(64)]),
                    dtype: DType::BF16,
                    dyn_axis: None,
                    raised: None,
                },
            ],
            ops: vec![
                Op {
                    kind: OpKind::Prep { prep },
                    inputs: Vec::new(),
                    outputs: vec![0],
                    layer: None,
                    dest: Vec::new(),
                },
                Op {
                    kind: OpKind::Launch {
                        kernel: "attn::dispatch_attention_flashinfer_prefill_bf16".to_string(),
                        weights: Vec::new(),
                        state: None,
                        params: Vec::new(),
                        param_extents: Vec::new(),
                    },
                    inputs: vec![0],
                    outputs: vec![1],
                    layer: None,
                    dest: Vec::new(),
                },
            ],
            depth_window: false,
            seams: Vec::new(),
        }
    }

    fn rows(n: usize) -> Vec<Row> {
        vec![
            Row {
                samples: true,
                ..Row::default()
            };
            n
        ]
    }

    /// THE STAGE 3 GATE: a raise reaches the lowering as itself.
    #[test]
    fn a_raise_lowers_to_its_own_arg_and_carries_the_word() {
        let plan = plan_with_a_raise();
        let out = lower(&plan, &rows(4), Fire::default()).expect("the fixture lowers");

        let launch = out.launches.first().expect("the launch was emitted");
        let args = &out.args[launch.args.start as usize..launch.args.end as usize];
        let first = args.first().expect("the launch placed its operand");

        match first {
            Arg::Raised { value, key } => {
                assert_eq!(*value, 0);
                assert_eq!(key, PrepKind::PrefillAttention { head_dim: 128 }.key());
            }
            other => panic!("the raise lowered as {other:?}, not as itself"),
        }
    }

    /// THE ARENA DECLINED IT, and that is not incidental.
    ///
    /// A raise reaches the allocator's output loop like any value. Without the
    /// guard, `value_bytes` sizes it from the empty shape and `take_block`
    /// hands back a real offset for zero bytes — after which `slot` reads
    /// `Arg::Arena` and the raise is a rectangle at a place in the activation
    /// arena, which is a wrong answer that binds.
    #[test]
    fn the_arena_gives_a_raise_no_block() {
        let plan = plan_with_a_raise();
        let out = lower(&plan, &rows(4), Fire::default()).expect("the fixture lowers");

        assert_eq!(
            out.value_offset[0],
            Buffers::NAMED,
            "the raise took an arena block"
        );
        assert_ne!(
            out.value_offset[1],
            Buffers::NAMED,
            "and the activation beside it still got one, so the guard is not a blanket"
        );
    }

    /// THE EDGE, END TO END: the prep's `value` is what the consumer's
    /// `Arg::Raised` names.
    ///
    /// This is what lets a driver answer BY VALUE. The key cannot tell two
    /// apart -- a stack whose layers disagree about head dim wants one
    /// schedule per width and both spell `fa2.prefill` -- so a resolver keyed
    /// on the word could only ever hand back whichever was raised last.
    #[test]
    fn the_preps_value_is_the_one_the_consumer_names() {
        let plan = plan_with_a_raise();
        let out = lower(&plan, &rows(4), Fire::default()).expect("the fixture lowers");

        let prep = out.preps.first().expect("the prep was carried");
        assert_eq!(prep.value, plan.ops[prep.at_op as usize].outputs[0]);

        let launch = out.launches.first().expect("the launch was emitted");
        let first = &out.args[launch.args.start as usize];
        match first {
            Arg::Raised { value, .. } => assert_eq!(
                *value, prep.value,
                "the consumer names a different object than the prep published"
            ),
            other => panic!("the raise lowered as {other:?}"),
        }
    }

    /// A prep states no launch of its own; only the statement that reads it does.
    #[test]
    fn the_prep_itself_emits_no_launch() {
        let plan = plan_with_a_raise();
        let out = lower(&plan, &rows(4), Fire::default()).expect("the fixture lowers");

        assert_eq!(out.launches.len(), 1, "the prep is not a launch");
        assert_eq!(out.preps.len(), 1, "and it is still stated as a prep");
        assert_eq!(out.preps[0].at_op, 0);
    }
}
