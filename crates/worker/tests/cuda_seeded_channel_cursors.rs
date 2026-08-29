//! A SEEDED host-written channel must survive its second step, and a
//! latest-value one must still accept `set` afterwards.
//!
//! # Two defects, one seed
//!
//! `cuda_host_writer_channels` is the sibling of this file, and it fixed the
//! host-writer drain for a channel declared with `Channel::writer` -- no seed,
//! head and tail both honestly zero at registration. A SEEDED channel takes a
//! different route into the engine and neither cursor came with it.
//!
//! The runtime stages a seed with `peek_seed`, keeping its own copy, and hands
//! the bytes over out of band as `ChannelValue`s riding the bind.
//! `fire::launch::ensure_sessions` writes them straight into ring slot 0. The
//! cell therefore never travels through the runtime-shared mirror, and nothing
//! on either side moves the head or tail word on its way past.
//!
//! The runtime counts it regardless. `pipeline::channel::bind` sets
//! `writer_tail = max(1)` for a seeded writer, because from the guest's side
//! one cell HAS been put. Two things the runtime owns are then wrong at once:
//!
//! * The guest's next `put` asks `writer_tail - head >= capacity`, reads the
//!   head word at zero, and gets `1 - 0 >= 1` at the default capacity of one.
//!   `Full`, and permanently -- the only thing that moves `head` is a
//!   `pull_channels` take out of a mirror that is empty. That is the same
//!   silent two-sided wait the sibling file is about: guest blocked on a cell
//!   it cannot stage, engine blocked on a fire never submitted, 0% on the GPU,
//!   nothing in any log.
//! * `Channel::set` -- which is how a LATEST-VALUE port is advanced, since
//!   nothing consumes one -- refuses with `Empty` unless the cell it replaces
//!   is still committed. With an empty mirror there is no front to replace, so
//!   every `set` after the seed is an error.
//!
//! `ensure_sessions` now publishes the seed into the mirror as well as the
//! ring, which is simply the truth about it: produced by the guest, consumed by
//! nothing. Tail moves, head does not, and both symptoms go with it.
//!
//! # And one the first fix caused
//!
//! `drain_host_writers` drained EVERY bound port when it was written. That is
//! wrong for exactly the latest-value ports above: `KvLen`, `Pages`,
//! `PageIndptr`, `EmbedIndptr` and `AttnMask` hold one committed cell that the
//! guest REPLACES with `set` rather than re-putting, because nothing device-
//! side would ever move their head. Draining that cell is what makes the `set`
//! fail. The drain is restricted to `Port::consumes()` now -- the same
//! predicate the device-resolved path had been using all along, two hundred
//! lines further down the same file.
//!
//! The two defects hid each other perfectly. While the seed cursors were wrong
//! the guest deadlocked on its second `put` and never reached a `set`, so the
//! over-broad drain could not be observed; fixing the cursors alone turned the
//! deadlock into `no cell available` at the first `set`.
//!
//! # Why `tart-masked`
//!
//! It is the only curated fixture that needs both halves. Its decode seeds
//! `token_in`, `position`, `write_slot` and `write_offset` with
//! `Channel::from(..)` and then host-`put`s each one per fire, and it seeds
//! `klen` and `page_indptr` and advances those with `set`. Its own source says
//! so, in the comment above the geometry advance, and it is right; the engine
//! was not.
//!
//! Run with `bisect: 2`, which drops the attention masks. That is NOT because
//! masks are incidental here -- it is because they are a SEPARATE axis of the
//! same fixture, and this gate should fail for its own reason or not at all.
//! When this file was written the masked arm answered `" wore of of of.. the."`
//! against the unmasked arm's `"<think>\nOkay, the user is asking"`, which was
//! a second defect entirely: the engine published the custom mask one byte per
//! pair where both kernels read it one BIT per pair. That is fixed and gated
//! now, by `cuda_element_mask_packing`, which drives the same fixture across
//! all three `bisect` settings and requires the answers to match. Keeping this
//! file on the arm with no masks at all means a future mask regression lands on
//! that gate and leaves this one alone.
//!
//! Run:
//!   cargo test --release -p worker --features engine-cuda-13 \
//!     --test cuda_seeded_channel_cursors -- --ignored --nocapture

mod common;

#[test]
#[ignore = "real-hardware: needs an RTX GPU + --features engine-cuda-13 + a local model snapshot"]
fn a_seeded_host_writer_survives_past_its_seed() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let _worker = common::boot_cuda().await;

        // Eight steps against a capacity-1 seeded ring: step two is where the
        // old engine stopped, and every step after it is a `set` the old engine
        // refused. Sixteen would prove no more than eight does and costs twice.
        let input = r#"{"prompt":"The capital of France is","max_tokens":8,"bisect":2}"#;
        let text = common::spawn_inferlet("tart-masked", input)
            .await
            .expect("tart-masked answered");

        // A DEADLOCK reaches here as the harness's "no answer within 180s" and
        // the over-broad drain as "no cell available", so both old failures are
        // already `Err` above. What is left to check is that the tokens are
        // real: the seed publish writes bytes into the mirror, and a seed that
        // arrived as zeros would still satisfy every cursor.
        assert!(
            !text.trim().is_empty(),
            "tart-masked answered nothing at all -- the seed reached the ring \
             but not the model"
        );
        eprintln!("[cuda_seeded_channel_cursors] 8 steps => {text:?}");
    });
}
