//! A host-written channel must survive more steps than its ring has cells.
//!
//! # The defect this exists for
//!
//! A guest that feeds a descriptor port from the HOST each decode step -- puts
//! the next token id into the channel its `embed` reads, rather than letting
//! the epilogue carry it on-device -- used to deadlock on `driver-cuda` after
//! exactly `capacity` steps. At the default capacity of one, that is the SECOND
//! step.
//!
//! The ring has one consumer on that driver, `Session::pull_channels`, which
//! copies the engine's mirror cells into the device ring and advances the
//! mirror's `head` as it goes. Nothing else moves `head`; the driver never
//! writes the binding's head word from anywhere else. And `head` is exactly
//! what the engine's writer checks before staging a cell, in
//! `engine/src/pipeline/channel.rs`:
//!
//! ```text
//!     if self.writer_tail - head >= capacity { return Err(ChannelError::Full) }
//! ```
//!
//! `fire::envelope::compose` pulled only on its device-resolved branch. A
//! member whose geometry the ENGINE resolves -- the ordinary case, and the only
//! one this driver builds for anything but the decode envelope -- took an early
//! `Composed::Wire` return, or the host branch's `continue`, and never touched
//! its rings. So `head` stayed at zero for the life of the instance and the
//! writer filled after `capacity` puts and stayed full.
//!
//! What that looks like from outside is the worst available failure: the guest
//! blocks on a cell that can never be staged, the driver blocks on a fire that
//! is never submitted, and NEITHER SIDE SAYS ANYTHING. `nvidia-smi` reads 0%.
//! The planner's contention trace prints nothing, because nothing is queued.
//! `RUST_LOG=debug` prints nothing after the last program registers. It is
//! indistinguishable from a model that is merely slow, which is why it sat in
//! four `#[ignore]`d fixtures for as long as it did.
//!
//! # How it was found, since the bisect is the reusable part
//!
//! Four curated fixtures hung on CUDA and said nothing. Bisecting on
//! `max_tokens` separated them at once: `contrastive-decoding` answered at 1
//! and 2 tokens and hung at 3, and `classifier-free-guidance` did the same.
//! Two tokens is one trip through the decode loop; three is two. So the FIRST
//! fire of every pass worked and the SECOND did not, which is a cursor, not a
//! kernel.
//!
//! Raising the two host-written channels' capacity from 1 to 4 moved the wall
//! rather than removing it -- 4 tokens passed, 24 hung -- and that is the whole
//! diagnosis: a ring that fills after exactly `capacity` items has a producer
//! whose consumer never runs. The other two fixtures turned out not to hang at
//! all (`cacheback-speculative-decoding` is slow, `constrained-speculative-
//! decoding` refuses a budget too small to close its JSON), which is why this
//! file names two and not four.
//!
//! # What this asserts
//!
//! `contrastive-decoding` host-puts its amateur token every step through a
//! capacity-1 channel, so it is the sharpest available probe: unfixed it dies
//! on step two, and no amount of ring widening saves a 24-token run. This asks
//! for 24 and requires coherent text, on the default dense snapshot.
//!
//! Use `--release`. A debug engine spends about 7.5 s a token here (this
//! fixture runs two passes and several host round trips per token) and the
//! harness's 180 s deadline is not generous.
//!
//!   cargo test --release -p worker --features driver-cuda-13 \
//!       --test cuda_host_writer_channels -- --ignored --nocapture

mod common;

#[test]
#[ignore = "real-hardware: needs an RTX GPU + --features driver-cuda-13 + a local snapshot"]
fn a_host_written_descriptor_port_outlives_its_ring() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let worker = common::boot_cuda().await;
        eprintln!("[host_writer] engine up on {}", worker.url());

        // 24 steps against a capacity-1 host writer: two dozen times more puts
        // than the ring holds, so a `head` that does not move cannot hide.
        let out = common::spawn_inferlet(
            "contrastive-decoding",
            r#"{"prompt":"The capital of France is","max_tokens":24}"#,
        )
        .await
        .expect(
            "contrastive-decoding host-puts its amateur token every step through a \
             capacity-1 channel. A timeout here is the ring wedging, not the model \
             being slow -- see this file's header.",
        );
        eprintln!("[host_writer] out = {out:?}");

        // Coherence, not an exact string: the point is that 24 steps HAPPENED,
        // and a wedged run produces no answer at all rather than a short one.
        assert!(
            out.len() > 20,
            "24 steps produced {} characters, which is fewer than one step's worth: {out}",
            out.len()
        );
        worker.shutdown().await;
    });
}
