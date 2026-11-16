//! A token run of the rs list shape (`SlotCopy::runs` on `RS`) lands row
//! for row: a run that starts and ends mid-page, its start and length no
//! multiple of the page's tokens, puts every row of every rs plane of its
//! source tokens at its destination tokens, and leaves every byte outside
//! the run as it was, in the destination and the source. A whole-page copy
//! of the same pages does not (the control). qwen35-tiny's rs list, its
//! slots patterned through the host tier.

use engine::chunk::{SlotCopy, SlotRun};
use engine::transfer::MemoryDomain;
use engine_gpu::store::layout::RS;

use crate::backend;
use crate::model_logits::{CONTEXT, PAGE, fixture, load_with};

/// A byte a place, odd; its `flip` differs from it in every byte.
fn pattern(flip: u8, bytes: u64) -> Vec<u8> {
    (0..bytes)
        .map(|at| ((at.wrapping_mul(2_654_435_761) >> 13) as u8 | 1) ^ flip)
        .collect()
}

/// `dst` as `runs` of `src` land in it: each of the shape's row runs
/// (`(offset, bytes a token)`), token for token.
fn landed(src: &[u8], dst: &[u8], rows: &[(u64, u64)], runs: &[(u32, u32, u32)]) -> Vec<u8> {
    let mut out = dst.to_vec();
    for &(from, to, tokens) in runs {
        for &(offset, cell) in rows {
            for t in 0..u64::from(tokens) {
                let s = (offset + (u64::from(from) + t) * cell) as usize;
                let d = (offset + (u64::from(to) + t) * cell) as usize;
                let cell = cell as usize;
                out[d..d + cell].copy_from_slice(&src[s..s + cell]);
            }
        }
    }
    out
}

#[test]
fn an_unaligned_token_run_of_the_rs_list_lands_row_for_row() {
    let fixture = fixture("qwen35-tiny", None);
    let Some(mut shell) = load_with(
        &fixture,
        PAGE,
        CONTEXT,
        None,
        engine::load::Residency {
            host_kv_budget: Some(256 << 20),
            ..engine::load::Residency::default()
        },
    ) else {
        return;
    };
    let device = <backend::Api as engine_gpu::api::Host>::domain(&backend::DeviceBoot::default());
    let rs = shell
        .pools()
        .rs_rows()
        .expect("the pool lays an rs list")
        .clone();
    let slots = shell.pools().layout().rs.expect("an rs list shape");
    let (tokens, bytes) = (rs.tokens, slots.slot_bytes);
    assert!(
        tokens >= 4,
        "an rs page of {tokens} tokens holds an unaligned run"
    );
    // The shape's rows lie apart, inside a slot, and are every rs plane.
    let mut spans: Vec<(u64, u64)> = rs
        .runs
        .iter()
        .map(|&(offset, cell)| (offset, offset + u64::from(tokens) * cell))
        .collect();
    spans.sort_unstable();
    assert!(!spans.is_empty(), "the rs list lays rows");
    assert!(
        spans.windows(2).all(|w| w[0].1 <= w[1].0),
        "rows overlap: {spans:?}"
    );
    assert!(
        spans.last().unwrap().1 <= bytes,
        "rows past the slot: {spans:?}"
    );
    for at in &rs.plane_at {
        assert!(
            rs.runs.iter().any(|&(offset, _)| offset == *at),
            "plane at {at}"
        );
    }
    eprintln!(
        "qwen35-tiny: rs slot of {tokens} tokens, {bytes} bytes, {} row runs",
        rs.runs.len()
    );

    // Device slots A (source), B and C (run destinations), D (the
    // control's whole-page destination): seat-addressed refs no fire took.
    let refs: Vec<u32> = (0..4)
        .map(|id| slots.seat_ref(id).expect("a seat-addressed rs slot"))
        .collect();
    let [a, b, c, d] = [refs[0], refs[1], refs[2], refs[3]];
    let src = pattern(0, bytes);
    let dst = pattern(0xfe, bytes);
    let host_in: Vec<u32> = (0..4).collect();
    let host_out: Vec<u32> = (4..8).collect();
    let fill: Vec<u8> = [&src[..], &dst[..], &dst[..], &dst[..]].concat();
    shell
        .write_tier_slots(&[(RS, host_in.clone())], &fill)
        .expect("the host slots take the patterns");
    let pairs = |from: &[u32], to: &[u32]| from.iter().copied().zip(to.iter().copied()).collect();
    let copy = |shell: &mut backend::Shell, src, dst, slots, runs| {
        shell
            .copy_slots(&SlotCopy {
                shape: RS,
                src,
                dst,
                slots,
                runs,
            })
            .unwrap_or_else(|why| panic!("the rs slot copy: {why}"))
    };
    copy(
        &mut shell,
        MemoryDomain::HostPinned,
        device,
        pairs(&host_in, &refs),
        Vec::new(),
    );

    // Two runs, neither start nor length a multiple of the page's tokens:
    // A's tokens 1.. (T - 3 of them) to B's 2.., A's last two to C's 1...
    let runs = [(1, 2, tokens - 3), (tokens - 2, 1, 2)];
    for &(from, to, n) in &runs {
        for v in [from, to, n] {
            assert_ne!(v % tokens, 0, "the run {from} -> {to} x{n} is unaligned");
        }
    }
    let stated: Vec<SlotRun> = [(b, runs[0]), (c, runs[1])]
        .into_iter()
        .map(|(dst, (src_token, dst_token, n))| SlotRun {
            src: a,
            src_token,
            dst,
            dst_token,
            tokens: n,
        })
        .collect();
    copy(&mut shell, device, device, Vec::new(), stated);
    // The control: a whole-page copy of A onto D.
    copy(&mut shell, device, device, vec![(a, d)], Vec::new());

    copy(
        &mut shell,
        device,
        MemoryDomain::HostPinned,
        pairs(&refs, &host_out),
        Vec::new(),
    );
    let back = shell
        .read_tier_slots(&[(RS, host_out)])
        .expect("the host slots read");
    let slot = |k: usize| &back[k * bytes as usize..(k + 1) * bytes as usize];

    let tell = |got: &[u8], want: &[u8], what: &str| {
        let wrong: Vec<usize> = (0..want.len()).filter(|&i| got[i] != want[i]).collect();
        assert!(
            wrong.is_empty(),
            "{what}: {} of {} bytes wrong, first at {:?}",
            wrong.len(),
            want.len(),
            wrong.first()
        );
    };
    tell(slot(0), &src, "the source page");
    let want_b = landed(&src, &dst, &rs.runs, &runs[..1]);
    let want_c = landed(&src, &dst, &rs.runs, &runs[1..]);
    // Every row of the run moved, and the bytes outside it stand.
    assert!(want_b.iter().zip(&dst).filter(|(x, y)| x != y).count() > 0);
    tell(slot(1), &want_b, "B, A's tokens 1.. at its 2..");
    tell(slot(2), &want_c, "C, A's last two tokens at its 1..");
    // The control: a whole-page copy is A itself, not the run's landing.
    tell(slot(3), &src, "D, the whole-page copy");
    assert_ne!(
        slot(3),
        &want_b[..],
        "a whole-page copy lands the unaligned run"
    );
    assert_ne!(
        slot(3),
        &want_c[..],
        "a whole-page copy lands the unaligned run"
    );
}
