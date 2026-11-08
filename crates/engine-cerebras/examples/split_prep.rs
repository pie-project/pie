//! The role-split prep from a Qwen3.5-9B checkpoint (no Python).
//!
//! `split_prep <ckpt dir> <out dir> <prefix>`: writes the prep's files.
//! `split_prep <ckpt dir> --check <prep dir> <prefix> [small|head|units|all]`:
//! compares against an existing prep, array by array, word for word.

use engine_cerebras::spatial::split::npy::Npy;
use engine_cerebras::spatial::split::prep::{self, Ckpt};

fn check(name: &str, path: &std::path::Path, shape: &[usize], words: &[u32]) -> bool {
    let f = Npy::open(path).unwrap();
    let mut have = Vec::new();
    f.words(0, words.len(), &mut have).unwrap();
    let bad = have.iter().zip(words).filter(|(a, b)| a != b).count();
    let first = have.iter().zip(words).position(|(a, b)| a != b);
    let ok = f.shape == shape && bad == 0;
    println!(
        "CHECK {name} {:?} vs {:?}: {} of {} words differ{} {}",
        shape,
        f.shape,
        bad,
        words.len(),
        first.map_or(String::new(), |i| format!(
            " (first at {i}: {:#x} vs ours {:#x})",
            have[i], words[i]
        )),
        if ok { "OK" } else { "BAD" }
    );
    ok
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ckpt = Ckpt::open(std::path::Path::new(&a[1])).unwrap();
    if a[2] != "--check" {
        prep::write(&ckpt, std::path::Path::new(&a[2]), &a[3]).unwrap();
        println!("PREP written");
        return;
    }
    let dir = std::path::Path::new(&a[3]);
    let px = &a[4];
    let what = a.get(5).map_or("all", String::as_str);
    let p = |n: &str| dir.join(format!("{px}{n}"));
    let mut ok = true;
    if what == "small" || what == "all" {
        for (n, shape, words, _) in prep::small(&ckpt).unwrap().items {
            ok &= check(&n, &p(&format!("{n}.npy")), &shape, &words);
        }
    }
    if what == "head" || what == "all" {
        let hw: Vec<u32> = prep::head(&ckpt, "lm_head.weight", true)
            .unwrap()
            .into_iter()
            .map(u32::from)
            .collect();
        ok &= check(
            "hw",
            &p("hw.npy"),
            &[prep::HR, prep::HC, prep::HDC * prep::VR],
            &hw,
        );
        let he: Vec<u32> = prep::head(&ckpt, "model.language_model.embed_tokens.weight", false)
            .unwrap()
            .into_iter()
            .map(u32::from)
            .collect();
        ok &= check(
            "hemb",
            &p("hemb.npy"),
            &[prep::HR, prep::HC, prep::VR * prep::HDC],
            &he,
        );
    }
    if what.starts_with("unit") || what == "all" {
        let us: Vec<usize> = match what.strip_prefix("unit") {
            Some(n) if !n.is_empty() && n != "s" => vec![n.parse().unwrap()],
            _ => (0..prep::NU).collect(),
        };
        for u in us {
            let w: Vec<u32> = prep::unit(&ckpt, u)
                .unwrap()
                .into_iter()
                .map(u32::from)
                .collect();
            ok &= check(
                &format!("w{u}"),
                &p(&format!("w{u}.npy")),
                &[prep::PY, prep::PX, 3 * 3360 + 3200],
                &w,
            );
        }
    }
    println!(
        "{}",
        if ok {
            "PREP_CHECK OK"
        } else {
            "PREP_CHECK FAILED"
        }
    );
}
