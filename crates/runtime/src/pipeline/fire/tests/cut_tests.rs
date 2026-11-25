use super::*;

#[test]
fn cut_every_case() {
    a_long_plain_pass_is_cut_on_page_boundaries();
    only_the_last_cut_runs_the_program();
    a_masked_pass_is_cut_with_its_rows_masks();
    the_last_piece_holds_what_the_program_reads();
    every_lane_of_a_pass_rides_its_last_fire();
    a_pass_with_media_is_cut_before_its_first_span();
}

fn prefill_lane(held: u32, rows: u32, page: u32) -> crate::engine::FireRequest {
    let mut lane: crate::engine::FireLane = ::engine::Lane::decode(0, 0, 1, 0).into();
    lane.tokens = (0..rows).map(|t| t + 100).collect();
    lane.kv.held = held;
    lane.pages = (0..(held + rows).div_ceil(page)).collect();
    lane.readout = ::engine::Readout::Last;
    let mut req = crate::engine::FireRequest::one(lane);
    req.boundary_program = true;
    req
}

fn a_long_plain_pass_is_cut_on_page_boundaries() {
    let req = prefill_lane(5, 100, 16);
    assert_eq!(
        host_cuts(&req, true, 128, 16, &[], 0),
        [(0, 100)],
        "short enough"
    );
    assert_eq!(
        host_cuts(&req, false, 32, 16, &[], 0),
        [(0, 100)],
        "not cuttable"
    );
    // Held 5: the first cut ends where page 2 does (32 - 5).
    assert_eq!(
        host_cuts(&req, true, 32, 16, &[], 0),
        [(0, 27), (27, 59), (59, 91), (91, 100)]
    );
    let mut masked = req.clone();
    masked.lanes[0].readout = ::engine::Readout::Rows(vec![3]);
    assert_eq!(
        host_cuts(&masked, true, 32, 16, &[], 0),
        [(0, 100)],
        "reads a middle row"
    );
    // Checkpoint rows cut it too, short as it is.
    assert_eq!(
        host_cuts(&req, true, 128, 16, &[43, 100, 0], 0),
        [(0, 43), (43, 100)]
    );
    assert_eq!(
        host_cuts(&req, true, 32, 16, &[43], 0),
        [(0, 27), (27, 43), (43, 59), (59, 91), (91, 100)]
    );
}

fn only_the_last_cut_runs_the_program() {
    let req = prefill_lane(5, 100, 16);
    let first = rows_of_lanes(&req, &[Some((0, 27))], false, 16);
    let lane = &first.lanes[0];
    assert_eq!(
        (lane.kv.held, lane.tokens.len(), lane.pages.len()),
        (5, 27, 2)
    );
    assert_eq!(lane.tokens[0], 100);
    assert!(matches!(lane.readout, ::engine::Readout::None));
    assert!(
        !first.boundary_program,
        "no program stages before the last fire"
    );
    let tail = rows_of_lanes(&req, &[Some((91, 100))], true, 16);
    let lane = &tail.lanes[0];
    assert_eq!(
        (lane.kv.held, lane.tokens.len(), lane.pages.len()),
        (96, 9, 7)
    );
    assert_eq!(lane.tokens[0], 191);
    assert!(matches!(lane.readout, ::engine::Readout::Last));
    assert!(tail.boundary_program);
}

fn a_pass_with_media_is_cut_before_its_first_span() {
    let mut req = prefill_lane(5, 100, 16);
    req.media.push(::engine::fire::StepMedia {
        lane: 0,
        rows: vec![10],
        routes: (60..70).chain([-1, -1]).collect(),
        token_positions: (0..300).collect(),
        ..Default::default()
    });
    let cuts = host_cuts(&req, true, 32, 16, &[], 0);
    assert_eq!(cuts, [(0, 27), (27, 59), (59, 100)], "no cut past row 60");
    let first = rows_of_lanes(&req, &[Some((0, 27))], false, 16);
    assert!(first.media.is_empty(), "the text before it carries none");
    let tail = rows_of_lanes(&req, &[Some((59, 100))], true, 16);
    assert_eq!(tail.media[0].routes[..2], [1, 2]);
    assert_eq!(tail.media[0].routes[10], -1, "a dropped row stays dropped");
    assert_eq!(tail.media[0].token_positions.len(), 3 * 41);
    assert_eq!(tail.media[0].token_positions[0], 3 * 59);
}

/// The pieces a pass is cut into end together: its last fire holds
/// every lane, and a last piece that left one out is refused by name.
fn every_lane_of_a_pass_rides_its_last_fire() {
    let mut req = prefill_lane(5, 100, 16);
    let other = prefill_lane(0, 40, 16).lanes.remove(0);
    req.lanes.push(other);
    let pieces = pass_pieces(&req, true, true, 32, 16, &[], &[]);
    assert!(pieces.len() > 1);
    assert_eq!(every_lane_in_last_piece(&pieces, 2), Ok(()));
    let short = vec![vec![Some((0, 50)), Some((0, 40))], vec![Some((50, 100)), None]];
    let refusal = every_lane_in_last_piece(&short, 2).unwrap_err();
    assert!(
        refusal.contains("a pass of 2 lanes carries 1"),
        "{refusal}"
    );
}

fn the_last_piece_holds_what_the_program_reads() {
    let mut req = prefill_lane(5, 100, 16);
    req.lanes[0].readout = ::engine::Readout::Rows(vec![96, 95]);
    let tail = rows_of_lanes(&req, &[Some((91, 100))], true, 16);
    assert_eq!(tail.lanes[0].readout, ::engine::Readout::Rows(vec![5, 4]));
    req.lanes[0].readout = ::engine::Readout::Rows(vec![60, 99]);
    assert_eq!(
        host_cuts(&req, true, 32, 16, &[], 0),
        [(0, 27), (27, 59), (59, 100)],
        "no cut past the first row read"
    );
    // An output over the lane's last rows: the last piece holds them all.
    let req = prefill_lane(5, 100, 16);
    assert_eq!(
        host_cuts(&req, true, 32, 16, &[], 32),
        [(0, 27), (27, 59), (59, 100)]
    );
}

fn a_masked_pass_is_cut_with_its_rows_masks() {
    let mut req = prefill_lane(5, 100, 16);
    let masks = (0..100)
        .map(|row| ::engine::Mask::new(vec![row, 1], 105))
        .collect();
    req.lanes[0].mask = Some(::engine::Masking::Rows(masks));
    let piece = rows_of_lanes(&req, &[Some((27, 59))], false, 16);
    let Some(::engine::Masking::Rows(masks)) = &piece.lanes[0].mask else {
        panic!("the piece keeps its rows' masks");
    };
    assert_eq!(masks.len(), 32);
    assert_eq!(masks[0].runs, [27, 1], "row 27's own mask");
}
