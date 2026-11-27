mod beam;
mod bounded_measuring;
mod buffered_fold;
mod churn;
mod device_geometry;
mod fold;
mod guest_state;
mod held_experts;
mod hosted;
mod lagged;
pub mod laid_lengths;
mod lane_refusal;
mod one_owner;
mod outputs;
mod port;
mod readout_rows;
mod roles;
mod scores;
mod settle_bound;
mod settle_key;
mod slots;
mod stale_settle_picks;
mod unplaced;

/// Writes what a guest pass read back, `bytes`, to
/// `$PIE_SUITE_GUEST_DUMP/<tag>-<n>.bin` (the `n`th pass under `tag`) when
/// that is set: runs of the same tests under different guest step forms
/// (whole, compacted, rolled) compare byte for byte.
pub(crate) fn guest_dump(tag: &str, bytes: &[u8]) {
    static PASSES: std::sync::Mutex<Vec<(String, u32)>> = std::sync::Mutex::new(Vec::new());
    let Some(dir) = std::env::var_os("PIE_SUITE_GUEST_DUMP") else {
        return;
    };
    let n = {
        let mut passes = PASSES.lock().expect("the dump count poisoned");
        let at = passes
            .iter()
            .position(|(t, _)| t == tag)
            .unwrap_or_else(|| {
                passes.push((tag.to_string(), 0));
                passes.len() - 1
            });
        passes[at].1 += 1;
        passes[at].1
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("the dump dir");
    std::fs::write(dir.join(format!("{tag}-{n}.bin")), bytes).expect("the dump writes");
}
