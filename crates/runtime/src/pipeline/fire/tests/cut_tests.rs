use super::*;

#[test]
fn cut_every_case() {
    a_long_plain_pass_is_cut_on_page_boundaries();
    only_the_last_cut_runs_the_program();
    only_the_last_cut_captures_scores();
    a_masked_pass_is_cut_with_its_rows_masks();
    the_last_piece_holds_what_the_program_reads();
    several_lanes_are_cut_and_end_together();
    every_lane_of_a_pass_rides_its_last_fire();
    a_cut_pass_behind_fires_of_its_frame_goes_in_frames_of_its_own();
    a_lane_only_in_the_last_piece_is_not_written_before_it();
    a_pass_with_media_is_cut_before_its_first_span();
    a_pass_with_media_keys_the_rows_before_it();
    the_rest_of_a_pass_is_cut_again_wider();
}

/// A raise landing mid-pass: what is left from the last cut is cut at
/// the wider share, on the same page boundaries and checkpoint rows.
fn the_rest_of_a_pass_is_cut_again_wider() {
    let req = prefill_lane(5, 200, 16);
    let narrow = host_cuts(&req, true, 32, 16, &[123], 0);
    assert_eq!(narrow[..2], [(0, 27), (27, 59)]);
    assert_eq!(
        host_cuts_from(&req, true, 160, 16, &[123], 0, 59),
        [(59, 123), (123, 200)],
        "the checkpoint row still cuts"
    );
    assert_eq!(
        host_cuts_from(&req, true, 64, 16, &[], 0, 59),
        [(59, 123), (123, 187), (187, 200)]
    );
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

/// A pass whose program reads its attention scores, cut into several
/// fires: only the last runs the program, so only its lanes capture
/// them; an earlier cut's would have no reader attached.
fn only_the_last_cut_captures_scores() {
    let req = prefill_lane(5, 100, 16);
    let switched = Switched {
        lanes: vec![Vec::new()],
        on: vec![vec![poem_ir::Builtin::Scores]],
        tails: vec![0],
        rows_read: true,
    };
    let cuts = host_cuts(&req, true, 32, 16, &[], 0);
    assert_eq!(cuts.len(), 4, "the pass is cut");
    let captures: Vec<bool> = cuts
        .iter()
        .enumerate()
        .map(|(at, &cut)| {
            let last = at + 1 == cuts.len();
            let piece = rows_of_lanes(&req, &[Some(cut)], last, 16);
            assert_eq!(piece.boundary_program, last);
            let requests = lane_requests(&piece, &switched, &[0], false, None);
            assert_eq!(requests.len(), 1);
            requests[0].captures_scores()
        })
        .collect();
    assert_eq!(
        captures,
        [false, false, false, true],
        "only the fire carrying the reading program captures scores"
    );
    // The pass uncut is its own last fire: it captures.
    let whole = lane_requests(&req, &switched, &[0], false, None);
    assert!(whole[0].captures_scores());
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

fn a_pass_with_media_keys_the_rows_before_it() {
    let mut req = prefill_lane(5, 100, 16);
    assert_eq!(keyed_rows(&req), 100);
    req.media.push(::engine::fire::StepMedia {
        lane: 0,
        rows: vec![10],
        routes: (60..70).collect(),
        ..Default::default()
    });
    assert_eq!(keyed_rows(&req), 60, "up to its first media row");
    let cuts = host_cuts(&req, true, 128, 16, &[], 0);
    assert_eq!(cuts, [(0, 59), (59, 100)]);
    let first = rows_of_lanes(&req, &[Some(cuts[0])], false, 16);
    assert!(first.media.is_empty(), "it logs its rows and seals");
    assert_eq!(keyed_rows(&first), 59);
    let tail = rows_of_lanes(&req, &[Some(cuts[1])], true, 16);
    assert!(!tail.media.is_empty(), "it cuts the log");
    req.lanes[0].positions = (0..100).collect();
    assert_eq!(keyed_rows(&req), 0, "rows rotated off their slots");
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

/// A cut pass sends its leading pieces unframed and its last with its
/// frame: behind fires of its own frame it goes in frames of its own;
/// first in its frame, alone in it, or uncut, it keeps the frame.
fn a_cut_pass_behind_fires_of_its_frame_goes_in_frames_of_its_own() {
    let stamp = |fires| crate::scheduler::FrameStamp {
        lane: uuid::Uuid::from_u128(1),
        seq: 3,
        slot: 1,
        fires,
    };
    assert!(reframes(stamp(2), 1, 3), "slot 1 behind slot 0, cut");
    assert!(!reframes(stamp(2), 0, 3), "first in its frame");
    assert!(!reframes(stamp(1), 0, 3), "a frame of its own");
    assert!(!reframes(stamp(2), 1, 1), "uncut");
}

fn several_lanes_are_cut_and_end_together() {
    let mut req = prefill_lane(5, 100, 16);
    let other = prefill_lane(0, 40, 16).lanes.remove(0);
    req.lanes.push(other);
    let pieces = pass_pieces(&req, true, true, 32, 16, &[], &[]);
    let last = pieces.last().unwrap();
    assert_eq!(
        last,
        &[Some((91, 100)), Some((32, 40))],
        "both lanes' tails"
    );
    assert_eq!(
        pieces[0],
        [Some((0, 11)), None],
        "the short lane starts later"
    );
    for lane in 0..2 {
        let rows: u32 = pieces
            .iter()
            .filter_map(|p| p[lane])
            .map(|(a, b)| b - a)
            .sum();
        assert_eq!(rows, [100, 40][lane], "every row once");
    }
    let first = rows_of_lanes(&req, &pieces[0], false, 16);
    assert_eq!(first.lanes.len(), 1);
    assert!(!first.boundary_program);
    let tail = rows_of_lanes(&req, last, true, 16);
    assert_eq!(tail.lanes.len(), 2);
    assert!(tail.boundary_program);
    assert_eq!(
        pass_pieces(&req, true, false, 32, 16, &[], &[]).len(),
        1,
        "not where the lanes carry recurrent state"
    );
}

fn a_lane_only_in_the_last_piece_is_not_written_before_it() {
    // A prompt cut into pieces beside a decode lane whose one row rides only in the last.
    let mut req = prefill_lane(0, 100, 16);
    let decode = prefill_lane(17, 1, 16).lanes.remove(0);
    req.lanes.push(decode);
    let pieces = pass_pieces(&req, true, true, 32, 16, &[], &[]);
    assert!(pieces.len() > 1, "the prompt is cut");
    let groups = [(0u32, vec![0]), (1u32, vec![1])];
    for (at, piece) in pieces.iter().enumerate() {
        let kept: Vec<usize> = (0..req.lanes.len())
            .filter(|&lane| piece[lane].is_some())
            .collect();
        let piece_groups: Vec<(u32, Vec<usize>)> = groups
            .iter()
            .map(|(ws, lanes)| {
                (
                    *ws,
                    lanes
                        .iter()
                        .filter_map(|lane| kept.iter().position(|k| k == lane))
                        .collect(),
                )
            })
            .collect();
        let last = at + 1 == pieces.len();
        let written = written_groups(&piece_groups);
        assert_eq!(
            written,
            if last { vec![0, 1] } else { vec![0] },
            "piece {at}"
        );
        if !last {
            // what the decode lane's set would have logged here: a write at slot 0, inside
            // the 17 slots it already froze
            let rows = rows_of_lanes(&req, piece, false, 16);
            assert_eq!(written_span_of(&rows, &piece_groups[1].1, 16), (0..0, 0..0));
        }
    }
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
