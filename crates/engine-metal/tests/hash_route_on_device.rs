//! dsv4-flash's LOOKUP router, on a real Apple GPU.
//!
//! **WHAT THIS FILE IS FOR.** The first `num_hash_layers` layers of
//! DeepSeek-V4-Flash do not score a gate at all: `ffn.gate.tid2eid` is a
//! `[vocab, top_k]` I64 table naming, for every token id, the experts that id
//! routes to. `hash_route_gather` has shipped in
//! `kernels/linear/moe_route.metal` since the port; nothing could reach it,
//! because no IR op named it and the model text routed those layers through a
//! plain softmax top-k over a router they do not read.
//! `linear.moe_hash_route` is that op, and this file is the first time the
//! entry meets a device.
//!
//! # The two things a faithful gather can still get wrong
//!
//! ```text
//! (a) the ADDRESS   tid2eid[tid · top_k + slot] — a gather that transposed
//!                   the table would read a self-consistent plane and route
//!                   every token to somebody else's experts, with no shape
//!                   check to notice
//! (b) the FALLBACK  an id at or past `vocab` reads row 0, exactly as
//!                   `embed.metal`'s gather does — not off the end
//! ```
//!
//! The weights are the third claim and the easy one: `1/top_k` on every slot,
//! which is what makes this router's output drop-in for the sorted-MoE path
//! behind it.
//!
//! # Gating
//!
//! As `device_floor`, `pool_on_device` and `hc_on_device`: `cfg`'d to Apple at
//! compile time, and SKIPS at run time when `device::present()` says no.
//!
//! ```text
//! cargo test -p engine-metal --test hash_route_on_device -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_metal::device::{self, Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use model_ir::Dtype;

/// **ONE DEVICE AT A TIME**, for `device_floor`'s reason.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn device_or_skip(what: &str) -> Option<Context> {
    if !device::present() {
        println!("SKIP {what}: this machine publishes no Metal device");
        return None;
    }
    Some(Context::bind().expect("the device binds"))
}

/// Small enough to hold in a test, big enough that a transposed read lands on
/// the wrong row rather than on the same one: `VOCAB != TOP_K`, and no row is
/// a permutation of another.
const VOCAB: usize = 37;
const TOP_K: usize = 6;

/// The table: row `t` names six experts derived from `t` so that no two rows
/// agree and no row is constant — a gather reading `tid2eid[slot · vocab +
/// tid]` (the transpose) produces a different set for every token this file
/// asks about.
fn tid2eid() -> Vec<i64> {
    (0..VOCAB)
        .flat_map(|t| (0..TOP_K).map(move |k| ((t * 11 + k * 3 + 1) % 64) as i64))
        .collect()
}

/// Fire the real entry over `ids` and read back `(routes, weights)`.
fn fire(ids: &[u32]) -> (Vec<i32>, Vec<f32>) {
    let device = Context::bind().expect("the device binds");
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    let table = tid2eid();
    let ids_buf = staged(&device, &encode_u32(ids));
    let table_buf = staged(&device, &encode_i64(&table));
    let rows = ids.len();
    let routes_buf =
        Buffer::zeroed(&device, (rows * TOP_K * 4) as u64).expect("the routes reserve");
    let weights_buf =
        Buffer::zeroed(&device, (rows * TOP_K * 4) as u64).expect("the weights reserve");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::linear::moe::hash_route(
            &sink,
            Tensor::new(
                bind_whole(&handles, &ids_buf, "the token ids"),
                rows as u32,
                1,
                Dtype::U32,
            ),
            Tensor::new(
                bind_whole(&handles, &table_buf, "the hash table"),
                VOCAB as u32,
                TOP_K as u32,
                Dtype::I64,
            ),
            VOCAB as u32,
            TOP_K as u32,
            Tensor::new(
                bind_whole(&handles, &routes_buf, "the routes"),
                rows as u32,
                TOP_K as u32,
                Dtype::I32,
            ),
            Tensor::new(
                bind_whole(&handles, &weights_buf, "the weights"),
                rows as u32,
                TOP_K as u32,
                Dtype::F32,
            ),
        )
        .expect("the lookup router encodes");
    }
    frame.commit().expect("the lookup router completes");

    (
        decode_i32(&read_back(&routes_buf, rows * TOP_K * 4)),
        decode_f32(&read_back(&weights_buf, rows * TOP_K * 4)),
    )
}

/// **THE ROW THE TOKEN NAMES, AND THE UNIFORM WEIGHT BESIDE IT.**
#[test]
fn every_token_reads_its_own_table_row_at_one_over_top_k() {
    let _serial = serialized();
    let Some(_device) = device_or_skip("the lookup router") else {
        return;
    };
    // Deliberately unsorted and with repeats, so a gather that assumed a
    // monotone id column is caught.
    let ids: Vec<u32> = vec![0, 36, 17, 3, 17, 25, 1, 9];
    let (routes, weights) = fire(&ids);

    let table = tid2eid();
    for (row, id) in ids.iter().enumerate() {
        for slot in 0..TOP_K {
            let want = table[*id as usize * TOP_K + slot] as i32;
            assert_eq!(
                routes[row * TOP_K + slot],
                want,
                "row {row} (token {id}) slot {slot} routed to {} and the table names {want}",
                routes[row * TOP_K + slot]
            );
            assert_eq!(
                weights[row * TOP_K + slot],
                1.0 / TOP_K as f32,
                "row {row} slot {slot} is not the uniform weight this router lands"
            );
        }
    }
    println!(
        "(a) hash_route_gather: {} tokens x {TOP_K} slots off a [{VOCAB}, {TOP_K}] table, exact",
        ids.len()
    );
}

/// **AN ID PAST THE VOCABULARY READS ROW 0, NOT OFF THE END.** The shader's
/// own fallback, and the same one `embed.metal`'s gather takes, so a boundary
/// token cannot walk out of the table.
#[test]
fn an_id_past_the_vocabulary_falls_to_the_first_row() {
    let _serial = serialized();
    let Some(_device) = device_or_skip("the out-of-range fallback") else {
        return;
    };
    let ids: Vec<u32> = vec![VOCAB as u32, VOCAB as u32 + 1000, 0];
    let (routes, _) = fire(&ids);

    let table = tid2eid();
    for row in 0..2 {
        for slot in 0..TOP_K {
            assert_eq!(
                routes[row * TOP_K + slot],
                table[slot] as i32,
                "an out-of-range id did not fall to the table's first row at slot {slot}"
            );
        }
    }
    // And the in-range id beside them is unaffected, so the fallback is a
    // per-row test and not a per-launch one.
    for slot in 0..TOP_K {
        assert_eq!(routes[2 * TOP_K + slot], table[slot] as i32);
    }
    println!("(b) an id at and past `vocab` falls to row 0, and token 0 still reads row 0");
}

// ---------------------------------------------------------------------------
// Host staging — `pool_on_device`'s helpers, for its reasons.
// ---------------------------------------------------------------------------

fn encode_u32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn encode_i64(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decode_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn staged(device: &Context, bytes: &[u8]) -> Buffer {
    let mut buffer = Buffer::zeroed(device, bytes.len() as u64).expect("the reservation lands");
    buffer.write(0, bytes).expect("the bytes land");
    buffer
}

fn bind_whole(handles: &Handles, buffer: &Buffer, what: &str) -> u32 {
    handles
        .bind(buffer, 0, buffer.bytes())
        .unwrap_or_else(|fault| panic!("{what} binds: {fault}"))
}

fn read_back(buffer: &Buffer, bytes: usize) -> Vec<u8> {
    let mut got = vec![0u8; bytes];
    buffer.read(0, &mut got).expect("the answer reads back");
    got
}
