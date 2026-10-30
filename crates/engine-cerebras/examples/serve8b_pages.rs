//! serve8b's host page copies ([`serve8b::page_out`], [`serve8b::page_in`])
//! over a mock device whose `kc`/`vc` word i at PE (x, y) is a fixed function
//! of them: page `a` out, then in at page `b`; prints the host copy's and
//! every written row's SHA-256, for comparison with `serve_lib8b.py` over the
//! same mock (with `KVST`, serve8b's staging kernels emulated on it).
//! `serve8b_pages <K:V,... params> <a> <b>` (or comma lists: several pages at once)

use std::collections::HashMap;

use engine_cerebras::sdk::Rect;
use engine_cerebras::spatial::serve2d::Layout2d;
use engine_cerebras::spatial::serve8b::{self, KvRows, S8b};
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

/// A device of rows in memory; with `stage` (serve8b with KVST), its
/// `page_stage`/`page_unstage` copy a page's block between `kc`/`vc` and
/// `kstage`/`vstage` (two bf16 cells a 32-bit word, the even cell low) on
/// every grid PE of the page's row.
#[derive(Default)]
struct Mock {
    mem: HashMap<(String, usize, usize), Vec<u32>>,
    stage: Option<Stage>,
}

/// The staging kernels' units: (corner, px, py, layers, cells a page), the
/// pool's blocks and slots, and whether a slot word packs two cells
/// (serve8b) or holds one (serve8b_region).
struct Stage {
    units: Vec<((usize, usize), usize, usize, usize, usize)>,
    blocks: usize,
    slots: usize,
    packed: bool,
}

impl Mock {
    fn pe(&mut self, sym: &str, x: usize, y: usize, n: usize) -> &mut Vec<u32> {
        self.mem.entry((sym.into(), x, y)).or_insert_with(|| (0..n).map(|i| word(sym, x, y, i)).collect())
    }

    fn copy(&mut self, page: u32, slot: usize, back: bool) {
        let st = self.stage.as_ref().expect("a staging device");
        let (units, blocks, slots_n, packed) = (st.units.clone(), st.blocks, st.slots, st.packed);
        for ((x0, y0), px, py, nl, n) in units {
            let (row, blk) = (page as usize % py, page as usize / py);
            let kcn = blocks * n;
            let kvlen = nl * kcn + 1;
            let sn = slots_n * nl * n;
            let sw = if packed { sn.div_ceil(2) } else { sn };
            for x in x0..x0 + px {
                for (pool, slots) in [("kc", "kstage"), ("vc", "vstage")] {
                    let y = y0 + row;
                    let mut kv = self.pe(pool, x, y, kvlen).clone();
                    let mut sl = self.pe(slots, x, y, sw).clone();
                    for l in 0..nl {
                        for i in 0..n {
                            let (c, k) = ((slot * nl + l) * n + i, 1 + l * kcn + blk * n + i);
                            if !packed {
                                if back { kv[k] = sl[c] & 0xffff } else { sl[c] = kv[k] & 0xffff }
                                continue;
                            }
                            let shift = 16 * (c % 2);
                            if back {
                                kv[k] = (sl[c / 2] >> shift) & 0xffff;
                            } else {
                                sl[c / 2] = (sl[c / 2] & !(0xffff << shift)) | ((kv[k] & 0xffff) << shift);
                            }
                        }
                    }
                    self.mem.insert((pool.into(), x, y), kv);
                    self.mem.insert((slots.into(), x, y), sl);
                }
            }
        }
    }
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

    fn read32(&mut self, sym: &str, r: Rect, n: usize) -> Result<Vec<u32>, String> {
        self.read16(sym, r, n)
    }

    fn write32(&mut self, sym: &str, r: Rect, n: usize, words: &[u32]) -> Result<(), String> {
        self.write16(sym, r, n, words)
    }

    fn launch(&mut self, name: &str, args: &[u32]) -> Result<(), String> {
        println!("launch {name} {} {}", args[0], args[1]);
        self.copy(args[0], args[1] as usize, name == "page_unstage");
        Ok(())
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
    let list = |s: &str| -> Result<Vec<u32>, String> { s.split(',').map(|x| x.parse().map_err(|e| format!("{e}"))).collect() };
    let (a_all, b_all) = (list(&args[2])?, list(&args[3])?);
    let mut dev = Mock::default();
    if lay.kvst > 0 {
        let shape = lay.paged.ok_or("not a paged build")?;
        let ps = shape.page_size;
        let (units, packed) = match lay.s8b {
            Some(S8b::Units(u)) => (
                u.units(&lay).iter().map(|v| ((v.x0, v.y0), v.k.px, v.k.py, v.layers.len(), ps * v.k.dc)).collect(),
                false,
            ),
            _ => ((0..lay.layers()).map(|l| (lay.origin(l), lay.px, lay.py, 1, ps * lay.dc)).collect(), true),
        };
        dev.stage = Some(Stage { units, blocks: shape.blocks, slots: lay.kvst, packed });
    }
    if a_all.len() == 1 {
        let (a, b) = (a_all[0], b_all[0]);
        let out = serve8b::page_out(&lay, &mut dev, a)?;
        println!("page_out {a} {} {}", out.len(), sha(&out));
        serve8b::page_in(&lay, &mut dev, b, &out)?;
    } else {
        // Several pages at once (page_out_many / page_in_many).
        let outs = serve8b::page_out_many(&lay, &mut dev, &a_all)?;
        for (a, out) in a_all.iter().zip(&outs) {
            println!("page_out {a} {} {}", out.len(), sha(out));
        }
        let ins: Vec<(u32, Vec<u32>)> = b_all.iter().copied().zip(outs).collect();
        serve8b::page_in_many(&lay, &mut dev, &ins)?;
    }
    Ok(())
}
