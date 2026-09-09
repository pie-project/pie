use engine::transfer::{KvCopy, KvMove};
use engine_metal::store::Move;

const PAGE: u32 = 8;

fn copy(src_pages: &[u32], dst_pages: &[u32], moves: Vec<KvMove>) -> KvCopy {
    KvCopy {
        src: engine::transfer::MemoryDomain::MetalShared,
        dst: engine::transfer::MemoryDomain::MetalShared,
        src_page_ids: src_pages.to_vec(),
        dst_page_ids: dst_pages.to_vec(),
        moves,
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
    let plan = Move::plan(&copy(&[3, 4], &[9, 10], Vec::new()), PAGE).expect("the plan");
    assert_eq!(
        plan,
        vec![
            Move {
                src_page: 3,
                src_token: 0,
                dst_page: 9,
                dst_token: 0,
                tokens: PAGE
            },
            Move {
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
    let plan = Move::plan(&copy(&[], &[], tail(2, 9, 0, 5)), PAGE).expect("the plan");
    assert_eq!(
        plan,
        vec![Move {
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
    let plan = Move::plan(&copy(&[], &[], cells), PAGE).expect("the plan");
    assert_eq!(
        plan,
        vec![
            Move {
                src_page: 2,
                src_token: 0,
                dst_page: 9,
                dst_token: 0,
                tokens: 2
            },
            Move {
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
    let plan = Move::plan(&copy(&[], &[], cells), PAGE).expect("the plan");
    assert_eq!(
        plan,
        vec![
            Move {
                src_page: 2,
                src_token: PAGE - 2,
                dst_page: 9,
                dst_token: PAGE - 2,
                tokens: 2
            },
            Move {
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
    let plan = Move::plan(&copy(&[], &[], cells), PAGE).expect("the plan");
    assert_eq!(
        plan,
        vec![Move {
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
    let why = Move::plan(&copy(&[], &[], cells), PAGE).expect_err("past the page");
    assert!(why.contains("token offsets"), "{why}");
    assert!(why.contains(&PAGE.to_string()), "{why}");
}

fn a_move_whose_ends_overlap_is_refused_by_name() {
    let why = Move::plan(
        &copy(
            &[],
            &[],
            tail(2, 2, 0, 3)
                .into_iter()
                .map(|cell| KvMove {
                    dst_token_offset: cell.src_token_offset + 1,
                    ..cell
                })
                .collect(),
        ),
        PAGE,
    )
    .expect_err("overlapping ends");
    assert!(why.contains("overlap"), "{why}");
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
    let plan = Move::plan(&copy(&[], &[], cells), PAGE).expect("the plan");
    assert_eq!(
        plan,
        vec![Move {
            src_page: 2,
            src_token: 0,
            dst_page: 2,
            dst_token: PAGE - 3,
            tokens: 3
        }]
    );
}

fn page_lists_that_are_not_parallel_are_refused() {
    let why = Move::plan(&copy(&[3, 4], &[9], Vec::new()), PAGE).expect_err("unequal lists");
    assert!(why.contains("src_page_ids"), "{why}");
}

fn the_two_spellings_are_one_list_of_runs() {
    let plan = Move::plan(&copy(&[0, 1], &[7, 8], tail(2, 9, 0, 5)), PAGE).expect("the plan");
    assert_eq!(
        plan,
        vec![
            Move {
                src_page: 0,
                src_token: 0,
                dst_page: 7,
                dst_token: 0,
                tokens: PAGE
            },
            Move {
                src_page: 1,
                src_token: 0,
                dst_page: 8,
                dst_token: 0,
                tokens: PAGE
            },
            Move {
                src_page: 2,
                src_token: 0,
                dst_page: 9,
                dst_token: 0,
                tokens: 5
            },
        ]
    );
}
