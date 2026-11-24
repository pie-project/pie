//! What the wires between blocks cost a step: the link term the placer prices beside a regime's
//! own (position-free) price. A stream step pays its wires twice: the fill (the row stream
//! crosses each unit gap once, at its longest wire) and each (unit, K / V column) pair's round
//! trip once a step at `place_kv_hop` a hop (the rows pipeline, so not once a row).
//!
//! Congestion is reported (shared fabric edges between block wires) but not priced.

use serde::Serialize;

use crate::compile::floor::Coord;
use crate::compile::stream::build::Program;
use crate::compile::wiring::Link;
use crate::tuning;

type R<T> = Result<T, String>;

/// A unit's attention round trip: its rows a step, the hops there and back (its longest head).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Trip {
    pub unit: String,
    pub rows: u32,
    pub hops: u32,
}

/// A stream program's link term.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Links {
    /// The row stream's gaps, each its longest wire's hops.
    pub fill_hops: Vec<u32>,
    pub trips: Vec<Trip>,
    /// Directed fabric edges more than one wire between blocks crosses, and the wires past one
    /// they carry.
    pub shared_edges: u32,
    pub shared_wires: u32,
    /// Cycles a step: the fill's and the round trips'.
    pub fill: f64,
    pub attention: f64,
    pub cycles: f64,
}

/// The link term of `p` (its wires routed as its fabric lays them).
pub fn stream(p: &Program) -> R<Links> {
    let s = &p.stream;
    let mut w = s.wiring();
    let mut floor = s.floor();
    super::route::fill_layout(&mut floor, &mut w)?;
    let mut gaps: std::collections::BTreeMap<String, u32> = Default::default();
    // (block, head) -> per unit k: hops there and back
    let mut kv: std::collections::BTreeMap<(String, usize), u32> = Default::default();
    let mut edges: std::collections::HashMap<(Coord, Coord), u32> = Default::default();
    for l in &w.links {
        let Link::Path(path) = l else { continue };
        let name = path.class.name;
        let hops = path.pts.len().saturating_sub(1) as u32;
        let between = if let Some(g) = name.strip_prefix("SP").or(name.strip_prefix("SW")) {
            // a gap's wires are one class (`SW<k>`); in-band gaps alternate `SP0` / `SP1`, told
            // apart by their first point's x
            let key = format!("{}{g}@{}", &name[..2], path.pts[0].x);
            let e = gaps.entry(key).or_default();
            *e = (*e).max(hops);
            true
        } else if let Some(rest) = name.strip_prefix("KI").or(name.strip_prefix("KO")) {
            // `KI<block>_<head>_<k>`: the round trip of unit k of the block, by head
            let (bh, k) = rest.rsplit_once('_').ok_or("links: a K / V link's name")?;
            let k: usize = k.parse().map_err(|e| format!("links: {name}: {e}"))?;
            *kv.entry((bh.to_string(), k)).or_default() += hops;
            true
        } else {
            name.starts_with("LB")
        };
        if between {
            for st in path.pts.windows(2) {
                *edges.entry((st[0], st[1])).or_default() += 1;
            }
        }
    }
    let fill_hops: Vec<u32> = gaps.into_values().collect();
    // a unit's trip: its longest head's, over the blocks' K / V columns
    let mut trips: Vec<Trip> = Vec::new();
    for kc in &s.kv {
        let bh = format!("{}_{}", kc.block, kc.head);
        for (k, &u) in kc.units.iter().enumerate() {
            let hops = kv.get(&(bh.clone(), k)).copied().unwrap_or(0);
            let name = s.units[u].name.clone();
            match trips.iter_mut().find(|t| t.unit == name) {
                Some(t) => t.hops = t.hops.max(hops),
                None => trips.push(Trip {
                    unit: name,
                    rows: p.units[u].unit.rows,
                    hops,
                }),
            }
        }
    }
    let fill = tuning::get().stream_chain_hop * f64::from(fill_hops.iter().sum::<u32>());
    // (every unit's every K / V column: its round trip once a step)
    let attention = tuning::get().place_kv_hop * f64::from(kv.values().sum::<u32>());
    let shared: Vec<u32> = edges.values().copied().filter(|&n| n > 1).collect();
    Ok(Links {
        fill_hops,
        trips,
        shared_edges: shared.len() as u32,
        shared_wires: shared.iter().map(|n| n - 1).sum(),
        fill,
        attention,
        cycles: fill + attention,
    })
}
