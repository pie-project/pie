//! Shared KV (Gemma 4): the units whose layers all read an earlier layer's KV (the sharers) take
//! it from the source unit (the unit of the layer they read), whose CY row sends it north up each
//! weight column through the sharers above it in its unit column ([`LINE`]; the sharers' weight
//! PEs take a copy, the source's weight PEs and the sharers' CY PEs pass it). The sharers' kernels
//! skip their own KV (`SH 1`) and run their own MLP width (`Ic`).

use crate::abi::{Overrides, Param};
use crate::floor::{Coord, Dir, Floor};
use crate::plan::{LayerKind, Plan};
use crate::program::Program;
use crate::role::Place;
use crate::wiring::{Class, Port, Turn, Wiring};

use super::Block;

type R<T> = Result<T, String>;

/// The shared-KV line.
pub const LINE: Class = Class::new("KVS");

/// The params the sharers differ in: they read another layer's KV, their MLP width a column.
pub const SH: Param = Param("SH");
pub const IC: Param = Param("Ic");

/// The source unit (the unit of the first shared layer's source) and the sharers (units whose
/// first layer reads another's KV).
pub fn units(plan: &Plan) -> (Option<u32>, Vec<u32>) {
    let k = plan.partition.layers_per_unit;
    let layers = &plan.shape.layers;
    let dst = (0..plan.partition.units)
        .filter(|&u| matches!(layers[(u * k) as usize], LayerKind::SharedKv { .. }))
        .collect();
    let src = layers.iter().find_map(|l| match *l {
        LayerKind::SharedKv { from, .. } => Some(from / k),
        _ => None,
    });
    (src, dst)
}

/// The shared-KV line and the sharers' params.
pub struct SharedKv;

impl Block for SharedKv {
    fn wire(&self, plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        let (Some(src), dst) = units(plan) else {
            return Ok(());
        };
        let unit = |u: u32| {
            floor
                .units
                .iter()
                .find(|v| v.index == u)
                .ok_or_else(|| format!("shared kv: unit {u} not on the floor"))
        };
        let s = unit(src)?;
        let mut ytop = u32::MAX;
        for &u in &dst {
            let d = unit(u)?;
            if d.rect.x != s.rect.x || d.rect.y >= s.rect.y {
                return Err("shared kv: the sharers sit above the source in its column".into());
            }
            ytop = ytop.min(d.rect.y);
        }
        if dst.is_empty() {
            return Ok(());
        }
        for c in 0..plan.partition.cols {
            let x = s.col_x(c);
            let ysrc = s.cy_y();
            w.turn(Turn {
                at: Coord::new(x, ysrc),
                class: LINE,
                from: Port::Ramp,
                to: vec![Port::Dir(Dir::N)],
            });
            for y in (ytop..ysrc).rev() {
                let at = Coord::new(x, y);
                let u = floor
                    .unit_at(at)
                    .ok_or_else(|| format!("shared kv: the line crosses a gap at ({x}, {y})"))?;
                let mut to = Vec::new();
                if dst.contains(&u.index) && y - u.rect.y != u.cy {
                    to.push(Port::Ramp);
                }
                if y > ytop {
                    to.push(Port::Dir(Dir::N));
                }
                w.turn(Turn {
                    at,
                    class: LINE,
                    from: Port::Dir(Dir::S),
                    to,
                });
            }
        }
        Ok(())
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        vec![(LINE, 10)]
    }

    /// `SH 1` on the sharers' weight and CY PEs; where the layers' MLP widths differ, every
    /// unit's width a column (`Ic`) on them.
    fn overrides(&self, prog: &Program, o: &mut Overrides) -> R<()> {
        let plan = &prog.plan;
        let (k, cols) = (plan.partition.layers_per_unit, plan.partition.cols);
        let mlp = &plan.shape.mlp;
        let shared = |u: u32| {
            let ls = &plan.shape.layers;
            let (a, b) = ((u * k) as usize, ((u + 1) * k) as usize);
            b <= ls.len()
                && ls[a..b]
                    .iter()
                    .all(|l| matches!(l, LayerKind::SharedKv { .. }))
        };
        for (at, role) in prog.floor.roles.iter() {
            let (Place::Weight { unit, .. } | Place::Cy { unit, .. }) = role.place else {
                continue;
            };
            if shared(unit) {
                o.set(at, SH, 1u32);
            }
            if !mlp.layer_widths.is_empty() {
                o.set(at, IC, mlp.width(unit * k) / cols);
            }
        }
        Ok(())
    }
}
