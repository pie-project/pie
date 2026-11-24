//! A stored settle whose key differs from the load's only by the kernels'
//! code (a kernel rebuilt since it was weighed) still serves its picks,
//! range by range, wherever its kernel is built for the load's planes: the
//! nearest settle's picks serve until the settle is weighed again behind
//! serving, never the curves' alone.

use ::engine::Engine as _;

use super::outputs::{Outputs, loaded_boot, measure_behind};
use crate::backend;

#[test]
fn a_settle_stale_by_code_alone_serves_its_picks_range_by_range() {
    let profiles = std::env::temp_dir().join(format!(
        "pie-{}-stale-settle-picks-{}",
        backend::NAME,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&profiles);
    let boot = || backend::DeviceBoot {
        profiles: Some(profiles.clone()),
        ..crate::gpu::device_boot()
    };
    let Some(mut first) = loaded_boot("stale-settle-picks", &Outputs, 4, boot()) else {
        return;
    };
    measure_behind(&mut first.engine);
    let before = first.engine.loaded_mut().expect("loaded").dense_tactics_at(1)[0];
    drop(first);

    // The one-row range given another vector kernel (one the load builds:
    // a candidate of the range), and the settle's code moved by one.
    let swapped = kernels_slang::linear::gemm::Tactic {
        vector: if before.vector == 8 { 4 } else { 8 },
        wide: false,
        ..before
    };
    let path = {
        let mut walk = vec![profiles.clone()];
        let mut found = None;
        while let Some(at) = walk.pop() {
            for entry in std::fs::read_dir(&at).into_iter().flatten().flatten() {
                let p = entry.path();
                if p.is_dir() {
                    walk.push(p);
                } else if std::fs::read_to_string(&p)
                    .is_ok_and(|t| t.contains("dense_settled_keyed"))
                {
                    found = Some(p);
                }
            }
        }
        found.expect("the first boot stored a settle")
    };
    let text = std::fs::read_to_string(&path).expect("the profile reads");
    let at = text.find("[[1, 1, ").expect("the settle decided the one-row range") + 1;
    let end = at + text[at..].find(']').expect("a range closes") + 1;
    let row = backend::Shell::settled_row(1, 1, swapped);
    let text = format!(
        "{}[{}]{}",
        &text[..at],
        row.map(|v| v.to_string()).join(", "),
        &text[end..]
    );
    // `[method, of, code, [ranges...`: the code moved, the rest kept.
    let head = text.find("dense_settled_keyed = [").expect("the settle's list");
    let entry = head + text[head..].find("\n  [").expect("an entry") + 4;
    let fields: Vec<&str> = text[entry..].splitn(4, ", ").collect();
    let code: u64 = fields[2].parse().expect("the code");
    let text = format!(
        "{}{}, {}, {}, {}",
        &text[..entry],
        fields[0],
        fields[1],
        code + 1,
        fields[3]
    );
    std::fs::write(&path, text).expect("the profile writes");

    let Some(mut second) = loaded_boot("stale-settle-picks", &Outputs, 4, boot()) else {
        return;
    };
    let pending = second.engine.measures_behind();
    let shell = second.engine.loaded_mut().expect("loaded");
    let served = shell.dense_tactics_at(1);
    drop(second);
    let _ = std::fs::remove_dir_all(&profiles);
    assert!(pending, "the stale settle is weighed again behind serving");
    assert!(
        served.iter().all(|&t| t == swapped),
        "the stale settle's pick serves the one-row range: {served:?}"
    );
}
