#![cfg(feature = "cuda")]

use crate::common::Gpu;
use kernels_cuda::channel::{WordRun, gather_words};

#[test]
fn the_word_gather_lands_each_run_where_its_table_says() {
    const RUNS: usize = 300;
    let sources: Vec<Vec<u32>> = (0..RUNS)
        .map(|run| {
            (0..1 + run % 3)
                .map(|word| (run * 1000 + word) as u32)
                .collect()
        })
        .collect();
    let width: usize = sources.iter().map(Vec::len).sum::<usize>() + 5;

    let mut gpu = Gpu::open();
    let mut runs = Vec::with_capacity(RUNS);
    let mut at = 5u32;
    let mut want = vec![0u32; width];
    for words in &sources {
        let src = gpu.up(words);
        runs.push(WordRun {
            src,
            dst_word: at,
            words: words.len() as u32,
        });
        want[at as usize..at as usize + words.len()].copy_from_slice(words);
        at += words.len() as u32;
    }
    // SAFETY: `WordRun` is plain `repr(C)` data, read as its bytes.
    let table: Vec<u8> = unsafe {
        core::slice::from_raw_parts(
            runs.as_ptr().cast::<u8>(),
            runs.len() * size_of::<WordRun>(),
        )
    }
    .to_vec();
    let table_at = gpu.up(&table);
    let dst_at = gpu.zeros(width * 4);
    gather_words(&gpu.ctx(), table_at, dst_at, RUNS as u32).expect("the gather fires");
    gpu.sync();
    let got: Vec<u32> = gpu.down(dst_at, width);
    assert_eq!(
        got, want,
        "every run lands at its own words and nowhere else"
    );
}
