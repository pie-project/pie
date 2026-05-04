//! PageStore benchmark (Radix Trie)
//!
//! Usage:
//!   cargo bench --bench pagestore_bench

use std::time::Instant;

use pie::context::pagestore as trie;

fn make_chain(n: usize) -> Vec<u64> { (1..=n as u64).collect() }
fn make_phys(n: usize) -> Vec<u32> { (0..n as u32).collect() }

fn time_ns<F: FnMut()>(iters: u32, mut f: F) -> u64 {
    for _ in 0..iters/4 { f(); }
    let start = Instant::now();
    for _ in 0..iters { f(); }
    start.elapsed().as_nanos() as u64 / iters as u64
}

fn fmt_time(ns: u64) -> String {
    if ns >= 1_000_000 { format!("{:.1}ms", ns as f64 / 1_000_000.0) }
    else if ns >= 1_000 { format!("{:.1}µs", ns as f64 / 1_000.0) }
    else { format!("{}ns", ns) }
}

fn trie_incremental(n: usize) -> (trie::PageStore, Vec<u64>) {
    let mut store = trie::PageStore::new(16, n + 200, 0);
    let h = make_chain(n); let p = make_phys(n);
    store.commit_batch(&h[..1], &p[..1]);
    for i in 1..n { store.extend(&h[..i], &h[i..i+1], &p[i..i+1]); }
    (store, h)
}

fn main() {
    let sizes = [64, 256, 1024, 4096, 10240];
    let iters = 100;

    let hdr = format!("  {:<22} {:>6}  {:>10}", "Operation", "N", "time");

    // =====================================================================
    println!("\n=== Single-chain incremental commit ===");
    println!("{hdr}");
    for &n in &sizes {
        let t = time_ns(iters, || { trie_incremental(n); });
        println!("  {:<22} {:>6}  {:>10}", "commit", n, fmt_time(t));
    }

    // =====================================================================
    println!("\n=== Read ops on committed chain ===");
    println!("{hdr}");
    for &n in &sizes {
        let (ts, th) = trie_incremental(n);

        let t = time_ns(iters, || { let _ = ts.physical_ids(&th); });
        println!("  {:<22} {:>6}  {:>10}", "physical_ids", n, fmt_time(t));

        let t = time_ns(iters, || { let _ = ts.prefix_len(&th); });
        println!("  {:<22} {:>6}  {:>10}", "prefix_len", n, fmt_time(t));

        let t = time_ns(iters, || { let _ = ts.count_reclaimable(&th); });
        println!("  {:<22} {:>6}  {:>10}", "count_reclaimable", n, fmt_time(t));

        let t = time_ns(iters, || { let (mut s, h) = trie_incremental(n); s.release(&h); });
        println!("  {:<22} {:>6}  {:>10}", "release (w/ build)", n, fmt_time(t));
    }

    // =====================================================================
    println!("\n=== Shared-prefix workload (32 contexts, 256 output tokens) ===");
    let num_contexts = 32;
    let suffix_len = 256;

    for &prefix_len in &[64, 256, 1024, 4096] {
        let total_pages = (prefix_len + suffix_len) * num_contexts + 500;
        let label = format!("pfx={}", prefix_len);
        println!("\n  --- prefix={prefix_len}, suffix={suffix_len}, contexts={num_contexts} ---");
        println!("{hdr}");

        // -- Build all contexts --
        let t = time_ns(iters, || {
            let mut store = trie::PageStore::new(16, total_pages, 0);
            let prefix: Vec<u64> = (1..=prefix_len as u64).collect();
            let p0 = store.alloc_gpu_pages(1).unwrap();
            store.commit_batch(&prefix[..1], &p0);
            for i in 1..prefix_len {
                let p = store.alloc_gpu_pages(1).unwrap();
                store.extend(&prefix[..i], &prefix[i..i+1], &p);
            }
            for ctx in 1..num_contexts {
                store.fork(&prefix);
                let mut chain = prefix.clone();
                for j in 0..suffix_len {
                    let h = (ctx * 100000 + j + 1) as u64;
                    let p = store.alloc_gpu_pages(1).unwrap();
                    store.extend(&chain, &[h], &p);
                    chain.push(h);
                }
            }
        });
        println!("  {:<22} {:>6}  {:>10}", "build_all", label, fmt_time(t));

        // Pre-build for read benchmarks
        let mut ts = trie::PageStore::new(16, total_pages, 0);
        let prefix: Vec<u64> = (1..=prefix_len as u64).collect();
        let p0 = ts.alloc_gpu_pages(prefix_len).unwrap();
        ts.commit_batch(&prefix, &p0);
        let mut tc: Vec<Vec<u64>> = vec![prefix.clone()];
        for ctx in 1..num_contexts {
            ts.fork(&prefix);
            let mut c = prefix.clone();
            for j in 0..suffix_len {
                let h = (ctx * 100000 + j + 1) as u64;
                let p = ts.alloc_gpu_pages(1).unwrap();
                ts.extend(&c, &[h], &p);
                c.push(h);
            }
            tc.push(c);
        }

        let longest = &tc[num_contexts - 1];
        let t = time_ns(iters, || { let _ = ts.physical_ids(longest); });
        println!("  {:<22} {:>6}  {:>10}", "physical_ids(longest)", label, fmt_time(t));

        let t = time_ns(iters, || { for c in &tc { let _ = ts.prefix_len(c); } });
        println!("  {:<22} {:>6}  {:>10}", "prefix_len(all 32)", label, fmt_time(t));

        let t = time_ns(iters, || { ts.fork(&prefix); });
        println!("  {:<22} {:>6}  {:>10}", "fork(prefix)", label, fmt_time(t));
    }
}
