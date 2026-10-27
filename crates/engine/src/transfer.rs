use serde::{Deserialize, Serialize};

use poem_ir::Dtype;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryDomain {
    HostPinned,
    CudaDevice(u32),
    RocmDevice(u32),
    MetalShared,
    MetalPrivate,
    VulkanDevice(u32),
    WgpuDevice(u32),
    XlaDevice(u32),
    CerebrasDevice(u32),
    /// The engine's own slot file, one kv page per slot.
    LocalDisk,
}

impl MemoryDomain {
    #[must_use]
    pub const fn ordinal(self) -> Option<u32> {
        match self {
            MemoryDomain::CudaDevice(ordinal)
            | MemoryDomain::RocmDevice(ordinal)
            | MemoryDomain::VulkanDevice(ordinal)
            | MemoryDomain::WgpuDevice(ordinal)
            | MemoryDomain::XlaDevice(ordinal)
            | MemoryDomain::CerebrasDevice(ordinal) => Some(ordinal),
            MemoryDomain::HostPinned
            | MemoryDomain::MetalShared
            | MemoryDomain::MetalPrivate
            | MemoryDomain::LocalDisk => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvLayoutKind {
    KvSeparate,
    FusedLatent,
}

impl KvLayoutKind {
    #[must_use]
    pub const fn planes(self) -> u64 {
        match self {
            KvLayoutKind::KvSeparate => 2,
            KvLayoutKind::FusedLatent => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvLayout {
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_size: u32,
    pub dtype: Dtype,
    pub kind: KvLayoutKind,
    #[serde(default)]
    pub storage_format: String,
    #[serde(default)]
    pub region_page_bytes: Vec<u64>,
}

impl KvLayout {
    #[must_use]
    pub fn page_bytes(&self) -> u64 {
        if !self.region_page_bytes.is_empty() {
            return self.region_page_bytes.iter().copied().sum();
        }
        let elements = u64::from(self.num_layers)
            * self.kind.planes()
            * u64::from(self.num_kv_heads)
            * u64::from(self.head_dim)
            * u64::from(self.page_size);
        (elements * self.dtype.bits()).div_ceil(8)
    }

    #[must_use]
    pub fn compatible_with(&self, other: &KvLayout) -> bool {
        self == other
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvRegion {
    pub base: u64,
    pub len: u64,
    pub page_stride: u64,
    pub domain: MemoryDomain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvHandle {
    pub regions: Vec<KvRegion>,
    pub layout: KvLayout,
}

impl KvHandle {
    #[must_use]
    pub fn page_bytes(&self) -> u64 {
        self.layout.page_bytes()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvMove {
    pub dst_page_id: u32,
    pub dst_token_offset: u32,
    pub src_page_id: u32,
    pub src_token_offset: u32,
}

/// `tokens` kv cells copied from one page to another, the unit a device
/// copies in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvRun {
    pub src_page: u32,
    pub src_token: u32,
    pub dst_page: u32,
    pub dst_token: u32,
    pub tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCopy {
    pub src: MemoryDomain,
    pub dst: MemoryDomain,
    pub src_page_ids: Vec<u32>,
    pub dst_page_ids: Vec<u32>,
    pub moves: Vec<KvMove>,
    /// Windowed kv pages `(src, dst)` moved in the copy's direction, in the
    /// windowed planes' own page ids: a ring's pages parking on the host.
    #[serde(default)]
    pub windowed: Vec<(u32, u32)>,
}

impl Default for KvCopy {
    fn default() -> KvCopy {
        KvCopy {
            src: MemoryDomain::HostPinned,
            dst: MemoryDomain::HostPinned,
            src_page_ids: Vec::new(),
            dst_page_ids: Vec::new(),
            moves: Vec::new(),
            windowed: Vec::new(),
        }
    }
}

impl KvCopy {
    pub fn validate(&self) -> crate::Result<()> {
        if self.src_page_ids.len() != self.dst_page_ids.len() {
            return Err(crate::Error::Invalid(format!(
                "src_page_ids has {} entries and dst_page_ids {}",
                self.src_page_ids.len(),
                self.dst_page_ids.len()
            )));
        }
        Ok(())
    }

    /// The copy over pages of `page_size` tokens as runs: one per whole
    /// page pair, and the per-token moves merged where each continues the
    /// last within one page. A run whose two ends overlap on one page is
    /// refused: a device copy of overlapping regions is undefined, not a
    /// shift.
    pub fn runs(&self, page_size: u32) -> crate::Result<Vec<KvRun>> {
        self.validate()?;
        let mut runs: Vec<KvRun> = self
            .src_page_ids
            .iter()
            .zip(&self.dst_page_ids)
            .map(|(&src_page, &dst_page)| KvRun {
                src_page,
                src_token: 0,
                dst_page,
                dst_token: 0,
                tokens: page_size,
            })
            .collect();
        for (at, cell) in self.moves.iter().enumerate() {
            if cell.src_token_offset >= page_size || cell.dst_token_offset >= page_size {
                return Err(crate::Error::Invalid(format!(
                    "kv move {at} names token offsets {}/{} in pages of {page_size} tokens",
                    cell.src_token_offset, cell.dst_token_offset
                )));
            }
            if cell.src_page_id == cell.dst_page_id
                && cell.src_token_offset == cell.dst_token_offset
            {
                continue;
            }
            let run = runs.last_mut().filter(|run| {
                run.src_page == cell.src_page_id
                    && run.dst_page == cell.dst_page_id
                    && run.src_token + run.tokens == cell.src_token_offset
                    && run.dst_token + run.tokens == cell.dst_token_offset
                    && run.src_token + run.tokens < page_size
            });
            match run {
                Some(run) => run.tokens += 1,
                None => runs.push(KvRun {
                    src_page: cell.src_page_id,
                    src_token: cell.src_token_offset,
                    dst_page: cell.dst_page_id,
                    dst_token: cell.dst_token_offset,
                    tokens: 1,
                }),
            }
        }
        if let Some(run) = runs.iter().find(|run| {
            run.src_page == run.dst_page && run.src_token.abs_diff(run.dst_token) < run.tokens
        }) {
            return Err(crate::Error::Invalid(format!(
                "a kv move of {} tokens reads page {} from token {} and writes the same page \
                 at token {} — the two ends overlap, and a device copy whose ends overlap is \
                 undefined rather than a shift",
                run.tokens, run.src_page, run.src_token, run.dst_token
            )));
        }
        Ok(runs)
    }

    /// A whole-page move between `device` and a tier below it: the tier,
    /// whether the pages leave the device, and the page ids on the device
    /// and in the tier. `None` for any other copy.
    #[must_use]
    pub fn tier_move(&self, device: MemoryDomain) -> Option<(MemoryDomain, bool, &[u32], &[u32])> {
        let tier = |domain| matches!(domain, MemoryDomain::HostPinned | MemoryDomain::LocalDisk);
        if !self.moves.is_empty() {
            return None;
        }
        if self.src == device && tier(self.dst) {
            Some((self.dst, true, &self.src_page_ids, &self.dst_page_ids))
        } else if tier(self.src) && self.dst == device {
            Some((self.src, false, &self.dst_page_ids, &self.src_page_ids))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateMove {
    pub src_slot_id: u32,
    pub dst_slot_id: u32,
    pub src_token_offset: u32,
    pub dst_token_offset: u32,
    pub token_count: u32,
}

/// Host ids index `host_state_slots` and `host_buffer_pages`, wherever the
/// engine keeps them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateDirection {
    #[default]
    DeviceToDevice,
    DeviceToHost,
    HostToDevice,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StateCopy {
    pub moves: Vec<StateMove>,
    /// Buffer pages, moved whole: the ids name pages.
    #[serde(default)]
    pub pages: Vec<StateMove>,
    #[serde(default)]
    pub direction: StateDirection,
}

impl StateCopy {
    /// Refuses a copy to or from host slots on `engine`, which reserves none.
    pub fn device_only(&self, engine: &'static str) -> crate::Result<()> {
        if self.direction == StateDirection::DeviceToDevice {
            return Ok(());
        }
        Err(crate::Error::Unsupported {
            verb: "`copy_state` to or from host slots, which this load does not reserve",
            engine,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u32 = 8;

    fn copy(src_pages: &[u32], dst_pages: &[u32], moves: Vec<KvMove>) -> KvCopy {
        KvCopy {
            src: MemoryDomain::MetalShared,
            dst: MemoryDomain::MetalShared,
            src_page_ids: src_pages.to_vec(),
            dst_page_ids: dst_pages.to_vec(),
            moves,
            windowed: Vec::new(),
        }
    }

    fn tail(src_page: u32, dst_page: u32, from: u32, tokens: u32) -> Vec<KvMove> {
        (0..tokens)
            .map(|at| KvMove {
                src_page_id: src_page,
                src_token_offset: from + at,
                dst_page_id: dst_page,
                dst_token_offset: from + at,
            })
            .collect()
    }

    #[test]
    fn a_forked_page_is_one_run_and_not_one_move_per_token_every_case() {
        a_page_pair_is_one_move_over_the_whole_page();
        a_forked_tail_is_one_run_and_not_one_move_per_token();
        a_gap_in_the_cells_cuts_the_run();
        a_run_never_walks_off_the_end_of_its_page();
        a_cell_that_moves_nowhere_is_dropped();
        an_offset_past_the_page_is_refused_by_name();
        a_move_whose_ends_overlap_is_refused_by_name();
        one_page_is_not_by_itself_an_overlap();
        page_lists_that_are_not_parallel_are_refused();
        the_two_spellings_are_one_list_of_runs();
    }

    fn a_page_pair_is_one_move_over_the_whole_page() {
        let plan = copy(&[3, 4], &[9, 10], Vec::new())
            .runs(PAGE)
            .expect("the plan");
        assert_eq!(
            plan,
            vec![
                KvRun {
                    src_page: 3,
                    src_token: 0,
                    dst_page: 9,
                    dst_token: 0,
                    tokens: PAGE
                },
                KvRun {
                    src_page: 4,
                    src_token: 0,
                    dst_page: 10,
                    dst_token: 0,
                    tokens: PAGE
                },
            ]
        );
    }

    fn a_forked_tail_is_one_run_and_not_one_move_per_token() {
        let plan = copy(&[], &[], tail(2, 9, 0, 5))
            .runs(PAGE)
            .expect("the plan");
        assert_eq!(
            plan,
            vec![KvRun {
                src_page: 2,
                src_token: 0,
                dst_page: 9,
                dst_token: 0,
                tokens: 5
            }]
        );
    }

    fn a_gap_in_the_cells_cuts_the_run() {
        let cells = vec![
            KvMove {
                src_page_id: 2,
                src_token_offset: 0,
                dst_page_id: 9,
                dst_token_offset: 0,
            },
            KvMove {
                src_page_id: 2,
                src_token_offset: 1,
                dst_page_id: 9,
                dst_token_offset: 1,
            },
            KvMove {
                src_page_id: 2,
                src_token_offset: 3,
                dst_page_id: 9,
                dst_token_offset: 3,
            },
        ];
        let plan = copy(&[], &[], cells).runs(PAGE).expect("the plan");
        assert_eq!(
            plan,
            vec![
                KvRun {
                    src_page: 2,
                    src_token: 0,
                    dst_page: 9,
                    dst_token: 0,
                    tokens: 2
                },
                KvRun {
                    src_page: 2,
                    src_token: 3,
                    dst_page: 9,
                    dst_token: 3,
                    tokens: 1
                },
            ]
        );
    }

    fn a_run_never_walks_off_the_end_of_its_page() {
        let mut cells = tail(2, 9, PAGE - 2, 2);
        cells.extend(tail(3, 10, 0, 2));
        let plan = copy(&[], &[], cells).runs(PAGE).expect("the plan");
        assert_eq!(
            plan,
            vec![
                KvRun {
                    src_page: 2,
                    src_token: PAGE - 2,
                    dst_page: 9,
                    dst_token: PAGE - 2,
                    tokens: 2
                },
                KvRun {
                    src_page: 3,
                    src_token: 0,
                    dst_page: 10,
                    dst_token: 0,
                    tokens: 2
                },
            ]
        );
    }

    fn a_cell_that_moves_nowhere_is_dropped() {
        let cells = vec![
            KvMove {
                src_page_id: 2,
                src_token_offset: 0,
                dst_page_id: 2,
                dst_token_offset: 0,
            },
            KvMove {
                src_page_id: 2,
                src_token_offset: 1,
                dst_page_id: 9,
                dst_token_offset: 1,
            },
        ];
        let plan = copy(&[], &[], cells).runs(PAGE).expect("the plan");
        assert_eq!(
            plan,
            vec![KvRun {
                src_page: 2,
                src_token: 1,
                dst_page: 9,
                dst_token: 1,
                tokens: 1
            }]
        );
    }

    fn an_offset_past_the_page_is_refused_by_name() {
        let cells = vec![KvMove {
            src_page_id: 2,
            src_token_offset: PAGE,
            dst_page_id: 9,
            dst_token_offset: 0,
        }];
        let why = copy(&[], &[], cells).runs(PAGE).expect_err("past the page");
        assert!(why.to_string().contains("token offsets"), "{why}");
        assert!(why.to_string().contains(&PAGE.to_string()), "{why}");
    }

    fn a_move_whose_ends_overlap_is_refused_by_name() {
        let why = copy(
            &[],
            &[],
            tail(2, 2, 0, 3)
                .into_iter()
                .map(|cell| KvMove {
                    dst_token_offset: cell.src_token_offset + 1,
                    ..cell
                })
                .collect(),
        )
        .runs(PAGE)
        .expect_err("overlapping ends");
        assert!(why.to_string().contains("overlap"), "{why}");
    }

    fn one_page_is_not_by_itself_an_overlap() {
        let cells: Vec<KvMove> = (0..3)
            .map(|at| KvMove {
                src_page_id: 2,
                src_token_offset: at,
                dst_page_id: 2,
                dst_token_offset: PAGE - 3 + at,
            })
            .collect();
        let plan = copy(&[], &[], cells).runs(PAGE).expect("the plan");
        assert_eq!(
            plan,
            vec![KvRun {
                src_page: 2,
                src_token: 0,
                dst_page: 2,
                dst_token: PAGE - 3,
                tokens: 3
            }]
        );
    }

    fn page_lists_that_are_not_parallel_are_refused() {
        let why = copy(&[3, 4], &[9], Vec::new())
            .runs(PAGE)
            .expect_err("unequal lists");
        assert!(why.to_string().contains("src_page_ids"), "{why}");
    }

    fn the_two_spellings_are_one_list_of_runs() {
        let plan = copy(&[0, 1], &[7, 8], tail(2, 9, 0, 5))
            .runs(PAGE)
            .expect("the plan");
        assert_eq!(
            plan,
            vec![
                KvRun {
                    src_page: 0,
                    src_token: 0,
                    dst_page: 7,
                    dst_token: 0,
                    tokens: PAGE
                },
                KvRun {
                    src_page: 1,
                    src_token: 0,
                    dst_page: 8,
                    dst_token: 0,
                    tokens: PAGE
                },
                KvRun {
                    src_page: 2,
                    src_token: 0,
                    dst_page: 9,
                    dst_token: 0,
                    tokens: 5
                },
            ]
        );
    }
}
