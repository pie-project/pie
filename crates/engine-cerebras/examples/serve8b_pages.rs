//! serve8b's host page copies ([`serve8b::page_out`], [`serve8b::page_in`])
//! over a mock device whose `kc`/`vc` word i at PE (x, y) is a fixed function
//! of them: page `a` out, then in at page `b`; prints the host copy's and
//! every written row's SHA-256, for comparison with `serve_lib8b.py` over the
//! same mock. `serve8b_pages <K:V,... params> <a> <b>`

use std::collections::HashMap;

use engine_cerebras::sdk::Rect;
use engine_cerebras::spatial::serve2d::Layout2d;
use engine_cerebras::spatial::serve8b::{self, KvRows};
use sha2::{Digest, Sha256};

fn sha(words: &[u32]) -> String {
    let mut h = Sha256::new();
    for w in words {
        h.update(w.to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Word i of `sym` at PE (x, y).
fn word(sym: &str, x: usize, y: usize, i: usize) -> u32 {
    ((x * 7919 + y * 104_729 + i * 31 + if sym == "vc" { 99_991 } else { 0 }) & 0xffff) as u32
}

#[derive(Default)]
struct Mock {
    mem: HashMap<(String, usize, usize), Vec<u32>>,
}

impl KvRows for Mock {
    fn read16(&mut self, sym: &str, r: Rect, n: usize) -> Result<Vec<u32>, String> {
        let mut out = Vec::new();
        for y in r.y as usize..(r.y + r.h) as usize {
            for x in r.x as usize..(r.x + r.w) as usize {
                let m = self.mem.entry((sym.into(), x, y)).or_insert_with(|| (0..n).map(|i| word(sym, x, y, i)).collect());
                out.extend_from_slice(m);
            }
        }
        Ok(out)
    }

    fn write16(&mut self, sym: &str, r: Rect, n: usize, words: &[u32]) -> Result<(), String> {
        println!("write {sym} {} {} {} {} {n} {}", r.x, r.y, r.w, r.h, sha(words));
        for (k, y) in (r.y as usize..(r.y + r.h) as usize).enumerate() {
            for (j, x) in (r.x as usize..(r.x + r.w) as usize).enumerate() {
                let at = (k * r.w as usize + j) * n;
                self.mem.insert((sym.into(), x, y), words[at..at + n].to_vec());
            }
        }
        Ok(())
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let params: Vec<(String, usize)> = args[1]
        .split(',')
        .filter_map(|kv| kv.split_once(':'))
        .map(|(k, v)| (k.to_string(), v.parse().unwrap_or(0)))
        .collect();
    let lay = Layout2d::of(|k| params.iter().find(|p| p.0 == k).map(|p| p.1)).ok_or("not a serve8b build")??;
    let (a, b): (u32, u32) = (args[2].parse().map_err(|e| format!("{e}"))?, args[3].parse().map_err(|e| format!("{e}"))?);
    let mut dev = Mock::default();
    let out = serve8b::page_out(&lay, &mut dev, a)?;
    println!("page_out {a} {} {}", out.len(), sha(&out));
    serve8b::page_in(&lay, &mut dev, b, &out)?;
    Ok(())
}
