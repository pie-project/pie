//! What a wgpu load fixes: every segment of the pool, each arm's readout
//! seat and its mirror, the host tier's stage and the slot file's. Serving
//! after it creates no buffer, and a need past what it fixed is refused by
//! name.

use crate::backend::{Api, DeviceBoot, Lane, Seated, Shell};
use crate::model_logits::{
    CONTEXT, PAGE, SEATS, fixture, load_with, prompts, serve_seats, word,
};
use engine::chunk::{Growth, SlotCopy, SlotKind};
use engine::transfer::MemoryDomain;
use engine_gpu::store::layout::LIST;

/// Two lanes prefill and decode, kv slots go down to the host tier and the
/// slot file and come back the same bytes, the lanes decode on, and a fire
/// reading past the seat, a segment past the pool's last and a release are
/// refused: through all of it the device creates no buffer
/// (`engine_wgpu::device::reservations` counts every one the process
/// creates, so this runs alone) and its ledger stands as the load left it.
#[test]
#[ignore = "counts the buffers the whole process creates: run it alone"]
fn serving_creates_no_buffer_after_the_load() {
    let fixture = fixture("qwen3-micro-text", None);
    let Some(mut shell) = load_with(
        &fixture,
        PAGE,
        CONTEXT,
        Some(64 << 10),
        engine::load::Residency {
            host_kv_budget: Some(64 << 20),
            ..engine::load::Residency::default()
        },
    ) else {
        return;
    };
    let dir = std::env::temp_dir().join(format!("pie-wgpu-fixed-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("the slot file's dir");
    let disk_pages = shell
        .seat_disk_kv(Some(16 << 20), Some(&dir), SEATS, CONTEXT)
        .expect("the slot file seats");
    assert!(disk_pages > 0, "a slot file of kv pages");

    let facts = shell.pool_chunk_facts();
    facts.validate().expect("the facts hang together");
    assert_eq!(facts.segments.growth, Growth::Fixed);
    assert_eq!(
        facts.segments.committed, facts.segments.max,
        "every segment stands from the load"
    );
    let created = engine_wgpu::device::reservations();
    let loaded = shell.device().books().ledger();
    eprintln!(
        "fixed at load: pool {} chunks of {} B in {} segments ({} B), {} arms' seats and \
         mirrors of {} B each, {created} buffers created; {loaded}",
        facts.total_chunks,
        facts.chunk_bytes,
        facts.segments.max,
        u64::from(facts.total_chunks) * facts.chunk_bytes,
        shell.readout.len(),
        shell.readout[0].bytes(),
    );

    // What each step after the load created, by name.
    let mut steps: Vec<(&str, u64, String)> = Vec::new();
    let mut seen = created;
    let mut step = |what: &'static str, shell: &Shell| {
        let now = engine_wgpu::device::reservations();
        steps.push((what, now - seen, shell.device().books().ledger().to_string()));
        seen = now;
    };

    serve_seats(&mut shell, &fixture.trace, 6);
    step("two lanes prefill and decode", &shell);

    // Free chunks above the seats' take a pattern from the host tier, go
    // down to the slot file, come back into other chunks and from there
    // to the host tier again.
    let device = <Api as engine_gpu::api::Host>::domain(&DeviceBoot::default());
    let list = facts
        .shapes
        .iter()
        .find(|shape| shape.id == LIST && matches!(shape.kind, SlotKind::List { .. }))
        .expect("a kv list");
    let per = list.slots_per_chunk;
    let low = 1 + shell.pool_layout().seat_chunks();
    assert!(
        low + 4 + facts.act_chunks_max() <= facts.total_chunks,
        "four free chunks above the seats'"
    );
    let refs = |chunks: std::ops::Range<u32>| -> Vec<u32> {
        chunks
            .flat_map(|chunk| (0..per).map(move |i| chunk * per + i))
            .collect()
    };
    let (there, back) = (refs(low..low + 2), refs(low + 2..low + 4));
    let slots = (there.len() as u32).min(disk_pages);
    let (there, back) = (&there[..slots as usize], &back[..slots as usize]);
    let host_in: Vec<u32> = (0..slots).collect();
    let host_out: Vec<u32> = (slots..2 * slots).collect();
    let on_disk: Vec<u32> = (0..slots).collect();
    let pattern: Vec<u8> = (0..u64::from(slots) * list.slot_bytes)
        .map(|at| (at.wrapping_mul(2_654_435_761) >> 13) as u8 | 1)
        .collect();
    shell
        .write_tier_slots(&[(LIST, host_in.clone())], &pattern)
        .expect("the host slots take the pattern");
    let pairs = |from: &[u32], to: &[u32]| from.iter().copied().zip(to.iter().copied()).collect();
    let copy = |shell: &mut Shell, src, dst, from: &[u32], to: &[u32]| {
        shell
            .copy_slots(&SlotCopy {
                shape: LIST,
                src,
                dst,
                slots: pairs(from, to),
                runs: Vec::new(),
            })
            .unwrap_or_else(|why| panic!("slots copy from {src:?} to {dst:?}: {why}"));
    };
    copy(&mut shell, MemoryDomain::HostPinned, device, &host_in, there);
    step("host tier to device", &shell);
    copy(&mut shell, device, MemoryDomain::LocalDisk, there, &on_disk);
    step("device to slot file", &shell);
    copy(&mut shell, MemoryDomain::LocalDisk, device, &on_disk, back);
    step("slot file to device", &shell);
    copy(&mut shell, device, MemoryDomain::HostPinned, back, &host_out);
    step("device to host tier", &shell);
    let read = shell
        .read_tier_slots(&[(LIST, host_out)])
        .expect("the host slots read");
    assert_eq!(
        read.iter().zip(&pattern).position(|(a, b)| a != b),
        None,
        "the slots come back through the slot file and the host tier as they went"
    );

    // The lanes serve on after the copies.
    serve_seats(&mut shell, &fixture.trace, 6);
    step("the lanes decode on", &shell);

    // Past what the load fixed: refused by name, nothing created.
    let past = shell
        .commit_segments(facts.segments.max + 1)
        .expect_err("no segment past the pool's last");
    assert!(
        past.to_string().contains("segments of the state pool"),
        "{past}"
    );
    let kept = shell
        .release_segments(0)
        .expect_err("a pool fixed at load keeps its segments");
    assert!(kept.to_string().contains("fixed at load"), "{kept}");
    // A prefill naming every one of its rows reads out more rows than the
    // load reserved, which the seat is cut to: refused by name, nothing
    // regrown.
    let prompt = &prompts()[0];
    let named: Vec<u32> = (0..prompt.len() as u32).collect();
    let prefill = |readout| Seated {
        held: Some(0),
        readout,
        ..Seated::of(Lane {
            slot: 0,
            word: word(&fixture.trace, prompt.len() as u32),
            tokens: prompt,
        })
    };
    let over = shell
        .fire_seated(&[prefill(Some(&named))])
        .expect_err("a fire reading out more rows than the load reserved");
    assert!(
        over.to_string().contains("the load reserved"),
        "the refusal names what the load reserved: {over}"
    );
    let got = shell
        .fire_seated(&[prefill(None)])
        .expect("the shell serves on after the refusal");
    assert!(got[0].iter().all(|v| v.is_finite()));
    step("the refusals and a prefill after them", &shell);

    let made: Vec<String> = steps
        .iter()
        .filter(|(_, made, _)| *made > 0)
        .map(|(what, made, ledger)| format!("{what}: {made} created ({ledger})"))
        .collect();
    assert!(
        made.is_empty(),
        "buffers were created after the load: {made:#?}"
    );
    // The frame uniform chunks are booked where a fire first touches them
    // (the frames piece lays them at the load): every other owner stands.
    let uniforms = engine_gpu::books::owners::UNIFORMS;
    let stands = |ledger: &engine::memory::Ledger| {
        ledger
            .entries()
            .filter(|&(class, owner, _)| (class, owner) != (uniforms.class, uniforms.name))
            .collect::<Vec<_>>()
    };
    let after = shell.device().books().ledger();
    assert_eq!(
        stands(&after),
        stands(&loaded),
        "the ledger moved after the load"
    );
    drop(shell);
    let _ = std::fs::remove_dir_all(&dir);
}
