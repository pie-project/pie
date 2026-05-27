//! Shmem ring transport latency. Measures the full client → server →
//! response → client roundtrip cycle inside a single process (creator
//! + attacher on different threads) for a representative payload.
//!
//! Run: `cargo bench -p pie-bridge --features "cabi ipc" --bench shmem`

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use pie_bridge::SCHEMA_HASH;
use pie_bridge::ipc::{ShmemClient, ShmemServer};

const SPIN_BUDGET_US: u64 = 100;

fn unique_name(suffix: &str) -> String {
    format!(
        "pie_bridge_bench_{}_{}_{}",
        suffix,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    )
}

/// One server thread that echoes back a 4-byte status payload. Returns
/// (server_arc, join_handle, stop_flag).
fn spawn_echo_server(
    name: &str,
    num_slots: usize,
) -> (Arc<ShmemServer>, thread::JoinHandle<()>, Arc<AtomicBool>) {
    let server = Arc::new(
        ShmemServer::create(name, num_slots, 4096, 4096, SPIN_BUDGET_US, SCHEMA_HASH)
            .expect("server create"),
    );
    let stop = Arc::new(AtomicBool::new(false));
    let server_clone = server.clone();
    let stop_clone = stop.clone();
    let h = thread::spawn(move || {
        while !stop_clone.load(Ordering::Relaxed) {
            if let Some(lease) = server_clone.poll_blocking(std::time::Duration::from_millis(10)) {
                // Read payload (touch a byte to ensure it's faulted in)
                // then commit 4 bytes of status response.
                let _ = lease.payload().first().copied();
                let _ = lease.commit_status(0);
            }
        }
    });
    (server, h, stop)
}

fn bench_shmem_roundtrip_smallish(c: &mut Criterion) {
    let name = unique_name("rt");
    let (_server, jh, stop) = spawn_echo_server(&name, 4);
    let client = ShmemClient::open(&name, SPIN_BUDGET_US, SCHEMA_HASH).expect("client open");

    // 256-byte payload. Real traffic varies but this represents a
    // cold-method request shape (Copy/Adapter).
    let payload = vec![0xa5u8; 256];

    c.bench_function("shmem_roundtrip_256B", |b| {
        let mut id = 1u32;
        b.iter(|| {
            let resp = client
                .roundtrip(id, black_box(&payload))
                .expect("roundtrip");
            id = id.wrapping_add(1);
            black_box(resp);
        });
    });

    stop.store(true, Ordering::Relaxed);
    jh.join().expect("server thread join");
}

fn bench_shmem_roundtrip_4k(c: &mut Criterion) {
    let name = unique_name("rt_4k");
    let (_server, jh, stop) = spawn_echo_server(&name, 4);
    let client = ShmemClient::open(&name, SPIN_BUDGET_US, SCHEMA_HASH).expect("client open");

    let payload = vec![0xa5u8; 4096];

    c.bench_function("shmem_roundtrip_4KB", |b| {
        let mut id = 1u32;
        b.iter(|| {
            let resp = client
                .roundtrip(id, black_box(&payload))
                .expect("roundtrip");
            id = id.wrapping_add(1);
            black_box(resp);
        });
    });

    stop.store(true, Ordering::Relaxed);
    jh.join().expect("server thread join");
}

criterion_group! {
    name = benches;
    // Shmem benches need slightly more time per sample because the spin
    // can lose a few ticks to the OS scheduler.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2));
    targets = bench_shmem_roundtrip_smallish, bench_shmem_roundtrip_4k
}
criterion_main!(benches);
