#![cfg(feature = "cuda")]
#![allow(unsafe_code)]

use crate::common::Gpu;
use kernels_cuda::runtime::channel::{WordRun, gather_words};

/// 300 runs of one to three words land one after another, five words in.
#[test]
fn the_word_gather_lands_each_run_where_its_table_says() {
    let mut gpu = Gpu::open();
    let (mut runs, mut want) = (Vec::new(), vec![0u32; 5]);
    for run in 0..300u32 {
        let words: Vec<u32> = (0..1 + run % 3).map(|w| run * 1000 + w).collect();
        let (src, dst_word) = (gpu.up(&words), want.len() as u32);
        runs.push(WordRun {
            src,
            dst_word,
            words: words.len() as u32,
        });
        want.extend(words);
    }
    // SAFETY: `WordRun` is plain `repr(C)` data, read as its bytes.
    let bytes =
        unsafe { core::slice::from_raw_parts(runs.as_ptr().cast::<u8>(), size_of_val(&runs[..])) };
    let (table_at, dst_at) = (gpu.up(bytes), gpu.zeros(want.len() * 4));
    gather_words(&gpu.ctx(), table_at, dst_at, 300).expect("the gather fires");
    gpu.sync();
    assert_eq!(
        gpu.down::<u32>(dst_at, want.len()),
        want,
        "each run at its own words"
    );
}
