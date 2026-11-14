//! A stream step on the host: several lanes at once, each its own request word, their rows side
//! by side (a diffusion transformer's denoise: a text lane and an image lane attending each
//! other). The ops a diffusion transformer's reading runs beyond a decoder's: layer norms,
//! adaLN modulations, gated residuals, sinusoidal embeddings, rotations over several position
//! axes, attention over the step's own rows (`attn.ragged`), row permutations, and the latent's
//! patches and cells.
//!
//! A node runs over every row of the step and keeps only the rows of the lanes whose word its
//! guard holds: the others are NaN, or, for an op that writes in place, what it found there. A
//! merge takes each row from the arm its lane's word holds. So a node guarded to the text lane
//! sees the image lane's rows of its inputs but lands none of them, as a GPU's per-class window
//! does.
//!
//! The step's geometry is the host's: every lane is one group (they attend each other), packed in
//! lane order; a media lane's clip is its voxels in (t, h, w) order, after the clips before it,
//! and its token rows are its patches in the same order.

use poem_ir::{
    ActFn, Attn, BinaryOp, Conv, Def, Dim, Elemwise, Gather, GeomKind, InputAxis, ModulateForm,
    Operands, RaggedMask, RopeForm, RuntimeInput, Ty, ValueId,
};

use super::{Forward, Held, Output, Run, Values, dot, sigmoid};

/// One lane of a stream step: its request word, its rows and, for a media lane, its clip's voxel
/// grid `(t, h, w)`.
#[derive(Clone, Debug)]
pub struct StreamLane {
    pub word: u64,
    pub rows: usize,
    pub clip: Option<[u32; 3]>,
}

/// A stream step: its lanes, and the guest inputs by name (`trace.inputs`), f32 as the input's
/// axis lays them: a row a lane, a row a token (the step's rows, lane after lane; a lane that
/// does not read it holds zeros), or a row a voxel (the clips' voxels, clip after clip).
#[derive(Clone, Debug)]
pub struct StreamStep<'a> {
    pub lanes: Vec<StreamLane>,
    pub inputs: Vec<(&'a str, Vec<f32>)>,
}

/// Each row's lane, by the axis a value's rows run over.
pub(super) struct RowLanes {
    pub words: Vec<u64>,
    pub token: Vec<usize>,
    pub voxel: Vec<usize>,
    pub clip: Vec<usize>,
}

impl RowLanes {
    /// The lane row `r` of a value of type `ty` belongs to, if its rows are a lane's.
    fn lane(&self, ty: &Ty, r: usize) -> Option<usize> {
        let Ty::Tensor { shape, .. } = ty else {
            return None;
        };
        match shape.first()? {
            Dim::Tokens => self.token.get(r).copied(),
            Dim::Lanes | Dim::LanesPlus(_) => (r < self.words.len()).then_some(r),
            Dim::Voxels => self.voxel.get(r).copied(),
            Dim::Clips | Dim::ClipsPlus(_) => self.clip.get(r).copied(),
            _ => None,
        }
    }
}

/// A clip as a grid input states it: `(t, h, w)` and its first row.
#[derive(Clone, Copy)]
struct Clip {
    t: usize,
    h: usize,
    w: usize,
    off: usize,
}

impl Clip {
    fn rows(self) -> usize {
        self.t * self.h * self.w
    }

    fn unravel(self, local: usize) -> (usize, usize, usize) {
        let w = local % self.w;
        let rest = local / self.w;
        (rest / self.h, rest % self.h, w)
    }

    fn ravel(self, t: usize, h: usize, w: usize) -> usize {
        self.off + (t * self.h + h) * self.w + w
    }
}

/// The clips a grid input holds, and the one row `r` falls in.
fn clips(grid: &[i32]) -> Vec<Clip> {
    grid.chunks(4)
        .map(|c| Clip {
            t: c[0].max(0) as usize,
            h: c[1].max(0) as usize,
            w: c[2].max(0) as usize,
            off: c[3].max(0) as usize,
        })
        .collect()
}

fn clip_of(clips: &[Clip], r: usize) -> Option<(usize, Clip)> {
    clips
        .iter()
        .enumerate()
        .find(|(_, c)| r >= c.off && r < c.off + c.rows())
        .map(|(i, c)| (i, *c))
}

/// erf to about 1.5e-7 (Abramowitz and Stegun 7.1.26), for the exact GELU.
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    y.copysign(x)
}

fn act(f: ActFn, x: f32) -> f32 {
    match f {
        ActFn::Identity => x,
        ActFn::Silu => x * sigmoid(x),
        ActFn::Gelu { tanh: true } => {
            0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
        }
        ActFn::Gelu { tanh: false } => 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)),
        ActFn::Tanh => x.tanh(),
        ActFn::Sqrt => x.sqrt(),
    }
}

/// A node's value as one stream step held it ([`Forward::stream_capture`]): what the node read,
/// before it ran, or what it wrote, after (the rows its guard keeps; the others NaN, or what an
/// in-place op found there).
#[derive(Clone, Debug, PartialEq)]
pub struct Captured {
    pub node: usize,
    /// `true`: one of the node's inputs, as it read it; `false`: one of its outputs.
    pub input: bool,
    pub value: ValueId,
    pub rows: usize,
    pub width: usize,
    pub values: Values,
}

impl Forward {
    /// One stream step of `step`'s lanes, every lane one group. Returns every output its lanes'
    /// words write, in the order the trace first names them.
    pub fn stream(&mut self, step: &StreamStep<'_>) -> Result<Vec<Output>, String> {
        Ok(self.stream_capture(step, &[])?.0)
    }

    /// [`Forward::stream`], and the inputs and outputs of each node of `nodes` as the step held
    /// them: the op-level oracle a primitive run on its own is compared to. A node its guard
    /// keeps from every lane captures nothing.
    pub fn stream_capture(
        &mut self,
        step: &StreamStep<'_>,
        nodes: &[usize],
    ) -> Result<(Vec<Output>, Vec<Captured>), String> {
        let mut captured: Vec<Captured> = Vec::new();
        let words: Vec<u64> = step.lanes.iter().map(|l| l.word).collect();
        let (mut token, mut positions, mut voxel, mut clip) = (vec![], vec![], vec![], vec![]);
        let (mut grid, mut tgrid) = (Vec::new(), Vec::new());
        for (i, lane) in step.lanes.iter().enumerate() {
            if let Some([t, h, w]) = lane.clip {
                grid.extend([t, h, w, voxel.len() as u32].map(|v| v as i32));
                voxel.extend(std::iter::repeat_n(i, (t * h * w) as usize));
                clip.push(i);
                tgrid.push((i, token.len()));
            }
            positions.extend(0..lane.rows as i32);
            token.extend(std::iter::repeat_n(i, lane.rows));
        }
        let rows = RowLanes {
            words,
            token,
            voxel,
            clip,
        };
        let (n, lanes) = (rows.token.len(), step.lanes.len());
        let mut vals: Vec<Option<Held>> = self.weights.clone();
        for (id, decl) in self.trace.values.iter().enumerate() {
            let Def::Input(input) = &decl.def else {
                continue;
            };
            let ints = |rows: usize, width: usize, v: Vec<i32>| Some(Held::Ints { rows, width, v });
            vals[id] = match *input {
                RuntimeInput::Positions => ints(n, 1, positions.clone()),
                RuntimeInput::Geometry {
                    kind: GeomKind::RequestOfToken,
                    ..
                } => ints(n, 1, rows.token.iter().map(|&l| l as i32).collect()),
                // every selected lane's rows, in lane order, then the rest's
                RuntimeInput::RowPermutation { select } => {
                    let (mut sel, mut rest): (Vec<i32>, Vec<i32>) = (vec![], vec![]);
                    for (r, &l) in rows.token.iter().enumerate() {
                        if select.holds(rows.words[l]) {
                            sel.push(r as i32);
                        } else {
                            rest.push(r as i32);
                        }
                    }
                    sel.extend(rest);
                    ints(n, 1, sel)
                }
                // one group: every selected lane
                RuntimeInput::Geometry {
                    kind: GeomKind::GroupIndptr { select },
                    ..
                } => {
                    let packed = (rows.token.iter())
                        .filter(|&&l| select.holds(rows.words[l]))
                        .count() as i32;
                    let mut v = vec![0, packed];
                    v.resize(lanes + 1, packed);
                    ints(lanes + 1, 1, v)
                }
                RuntimeInput::Grid => ints(grid.len() / 4, 4, grid.clone()),
                RuntimeInput::TokenGrid { p } => {
                    let v: Vec<i32> = (tgrid.iter())
                        .flat_map(|&(lane, off)| {
                            let [t, h, w] = step.lanes[lane].clip.unwrap_or([0; 3]);
                            [t / p[0], h / p[1], w / p[2], off as u32].map(|v| v as i32)
                        })
                        .collect();
                    ints(v.len() / 4, 4, v)
                }
                RuntimeInput::Guest {
                    id, axis, width, ..
                } => {
                    let name = &self.trace.inputs[id as usize].name;
                    let Some((_, v)) = step.inputs.iter().find(|(n, _)| n == name) else {
                        continue;
                    };
                    let want = match axis {
                        InputAxis::Lane => lanes,
                        InputAxis::Row => n,
                        InputAxis::Voxel => rows.voxel.len(),
                    };
                    if v.len() != want * width as usize {
                        return Err(format!(
                            "input `{name}`: {} values for {want} rows of {width}",
                            v.len()
                        ));
                    }
                    Some(Held::Floats {
                        rows: want,
                        width: width as usize,
                        v: v.clone(),
                    })
                }
                RuntimeInput::Mask { .. } | RuntimeInput::Geometry { .. } => Some(Held::Opaque),
                _ => continue,
            };
        }
        let mut run = Run {
            trace: &self.trace,
            vals,
            word: 0,
            positions: &positions,
            lanes: &mut self.lanes,
            group: self.group.as_ref(),
            placed: &self.placed,
            rows: Some(&rows),
        };
        let (mut ins, mut outs, mut aliases) = (Vec::new(), Vec::new(), Vec::new());
        for (at, node) in self.trace.nodes.iter().enumerate() {
            let live: Vec<bool> = rows.words.iter().map(|&w| node.guard.holds(w)).collect();
            ins.clear();
            node.op.inputs(&mut ins);
            aliases.clear();
            node.op.aliases(&mut aliases);
            if !live.iter().any(|&l| l) {
                for &(out, into) in &aliases {
                    run.materialize(into)?;
                    let kept = run.held(into).ok().cloned();
                    run.vals[out.0 as usize] = kept;
                }
                continue;
            }
            for &v in &ins {
                run.materialize(v)?;
            }
            let tapped = nodes.contains(&at);
            if tapped {
                for &v in &ins {
                    captured.extend(run.captured(at, true, v));
                }
            }
            run.op(&node.op)
                .map_err(|why| format!("node {at}: {why}"))?;
            outs.clear();
            node.op.outputs(&mut outs);
            if !live.iter().all(|&l| l) {
                for &o in &outs {
                    let into = aliases.iter().find(|(out, _)| *out == o).map(|&(_, i)| i);
                    run.keep(o, &live, into);
                }
            }
            if tapped {
                for &o in &outs {
                    captured.extend(run.captured(at, false, o));
                }
            }
        }
        let holds = |g: &poem_ir::Guard| rows.words.iter().any(|&w| g.holds(w));
        let mut out: Vec<Output> = Vec::new();
        for output in &self.trace.outputs {
            let written = match &self.trace.values[output.value.0 as usize].def {
                Def::Op(node) => holds(&self.trace.nodes[*node as usize].guard),
                Def::Merge(arms) => arms.iter().any(|(_, g)| holds(g)),
                _ => false,
            };
            if !written {
                continue;
            }
            run.materialize(output.value)?;
            let (rows, width, values) = match run.held(output.value)? {
                Held::Floats { rows, width, v } => (*rows, *width, Values::F32(v.clone())),
                Held::Ints { rows, width, v } => (*rows, *width, Values::I32(v.clone())),
                Held::Opaque => return Err(format!("output `{}` holds no rows", output.name)),
            };
            out.push(Output {
                name: output.name.clone(),
                layers: 0,
                rows,
                width,
                values,
            });
        }
        Ok((out, captured))
    }
}

impl Run<'_> {
    /// `v` as it is held now, node `at`'s input or output.
    fn captured(&self, at: usize, input: bool, v: ValueId) -> Option<Captured> {
        let (rows, width, values) = match self.held(v).ok()? {
            Held::Floats { rows, width, v } => (*rows, *width, Values::F32(v.clone())),
            Held::Ints { rows, width, v } => (*rows, *width, Values::I32(v.clone())),
            Held::Opaque => return None,
        };
        Some(Captured {
            node: at,
            input,
            value: v,
            rows,
            width,
            values,
        })
    }

    /// A merge `v` (in a stream step) as a value: each row from the arm its lane's word takes.
    pub(super) fn materialize(&mut self, v: ValueId) -> Result<(), String> {
        let Some(rows) = self.rows else {
            return Ok(());
        };
        let Def::Merge(arms) = &self.trace.values[v.0 as usize].def else {
            return Ok(());
        };
        if self.vals[v.0 as usize].is_some() {
            return Ok(());
        }
        for &(a, _) in arms {
            self.materialize(a)?;
        }
        let held: Vec<Option<&Held>> = arms
            .iter()
            .map(|(a, _)| self.vals[a.0 as usize].as_ref())
            .collect();
        let ty = &self.trace.values[v.0 as usize].ty;
        let Some(first) = held.iter().flatten().next() else {
            return Ok(());
        };
        let pick = |r: usize| {
            let lane = rows.lane(ty, r);
            arms.iter()
                .zip(&held)
                .position(|((_, g), h)| h.is_some() && lane.is_none_or(|l| g.holds(rows.words[l])))
        };
        let merged = match first {
            Held::Floats { rows: n, width, .. } => {
                let (n, width) = (*n, *width);
                let mut out = vec![f32::NAN; n * width];
                for r in 0..n {
                    if let Some(Some(Held::Floats { v, .. })) = pick(r).map(|i| held[i])
                        && v.len() >= (r + 1) * width
                    {
                        out[r * width..(r + 1) * width].copy_from_slice(&v[r * width..][..width]);
                    }
                }
                Held::Floats {
                    rows: n,
                    width,
                    v: out,
                }
            }
            Held::Ints { rows: n, width, .. } => {
                let (n, width) = (*n, *width);
                let mut out = vec![-1; n * width];
                for r in 0..n {
                    if let Some(Some(Held::Ints { v, .. })) = pick(r).map(|i| held[i])
                        && v.len() >= (r + 1) * width
                    {
                        out[r * width..(r + 1) * width].copy_from_slice(&v[r * width..][..width]);
                    }
                }
                Held::Ints {
                    rows: n,
                    width,
                    v: out,
                }
            }
            Held::Opaque => Held::Opaque,
        };
        self.vals[v.0 as usize] = Some(merged);
        Ok(())
    }

    /// Output `o` of a node only some lanes run: the other lanes' rows NaN, or what the node
    /// found in `into` where it writes in place.
    fn keep(&mut self, o: ValueId, live: &[bool], into: Option<ValueId>) {
        let Some(rows) = self.rows else {
            return;
        };
        let ty = &self.trace.values[o.0 as usize].ty;
        let found = into.and_then(|i| self.vals[i.0 as usize].clone());
        let Some(held) = self.vals[o.0 as usize].as_mut() else {
            return;
        };
        let dead = |r: usize| rows.lane(ty, r).is_some_and(|l| !live[l]);
        match held {
            Held::Floats { rows: n, width, v } => {
                for r in (0..*n).filter(|&r| dead(r)) {
                    let row = &mut v[r * *width..(r + 1) * *width];
                    match &found {
                        Some(Held::Floats { v: f, width: w, .. })
                            if w == width && f.len() >= (r + 1) * *w =>
                        {
                            row.copy_from_slice(&f[r * *w..(r + 1) * *w]);
                        }
                        _ => row.fill(f32::NAN),
                    }
                }
            }
            Held::Ints { rows: n, width, v } => {
                for r in (0..*n).filter(|&r| dead(r)) {
                    v[r * *width..(r + 1) * *width].fill(-1);
                }
            }
            Held::Opaque => {}
        }
    }

    /// The row of a per-lane (or per-row) operand `m` row `r` reads: `lane_of_row[r]`'s, else
    /// its own.
    fn lane_row(&self, lane_of_row: Option<ValueId>, r: usize) -> Result<usize, String> {
        match lane_of_row {
            Some(l) => {
                let l = self.ints(l)?;
                usize::try_from(l.v[r]).map_err(|_| format!("row {r} of no lane"))
            }
            None => Ok(r),
        }
    }

    pub(super) fn elemwise_more(&mut self, op: &Elemwise) -> Result<(), String> {
        match *op {
            Elemwise::Layernorm {
                x,
                weight,
                bias,
                eps,
                y,
            } => {
                let xs = self.floats(x)?;
                let w = self.opt_floats(weight)?.map(|w| w.v.to_vec());
                let b = self.opt_floats(bias)?.map(|b| b.v.to_vec());
                let mut out = Vec::with_capacity(xs.v.len());
                for t in 0..xs.rows {
                    let row = xs.row(t);
                    let n = row.len() as f32;
                    let mean = row.iter().sum::<f32>() / n;
                    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
                    let inv = 1.0 / (var + eps).sqrt();
                    for (i, v) in row.iter().enumerate() {
                        let g = w.as_ref().map_or(1.0, |w| w[i]);
                        let b = b.as_ref().map_or(0.0, |b| b[i]);
                        out.push((v - mean) * inv * g + b);
                    }
                }
                let rows = xs.rows;
                self.put(y, rows, out)
            }
            Elemwise::Modulate {
                x,
                m,
                lane_of_row,
                form,
                y,
            } => {
                let (xs, ms) = (self.floats(x)?, self.floats(m)?);
                let w = xs.width;
                let mut out = Vec::with_capacity(xs.v.len());
                for t in 0..xs.rows {
                    let mr = ms.row(self.lane_row(lane_of_row, t)?);
                    for (c, &v) in xs.row(t).iter().enumerate() {
                        out.push(match form {
                            ModulateForm::ScaleShift => v.mul_add(1.0 + mr[c], mr[w + c]),
                            ModulateForm::Scale => v * (1.0 + mr[c]),
                            ModulateForm::TanhGate => mr[c].tanh() * v,
                        });
                    }
                }
                let rows = xs.rows;
                self.put(y, rows, out)
            }
            Elemwise::Sinusoid {
                t,
                dim,
                max_period,
                flip_sin_cos,
                scale,
                y,
            } => {
                let ts = self.floats(t)?;
                let (dim, half) = (dim as usize, dim as usize / 2);
                let mut out = Vec::with_capacity(ts.rows * dim);
                for r in 0..ts.rows {
                    let t = ts.v[r * ts.width];
                    let angles: Vec<f32> = (0..half)
                        .map(|i| scale * (t * (-max_period.ln() * i as f32 / half as f32).exp()))
                        .collect();
                    let mut row: Vec<f32> = angles.iter().map(|a| a.sin()).collect();
                    row.extend(angles.iter().map(|a| a.cos()));
                    if flip_sin_cos {
                        row.rotate_left(half);
                    }
                    row.resize(dim, 0.0);
                    out.extend(row);
                }
                let rows = ts.rows;
                self.put(y, rows, out)
            }
            Elemwise::Binary { x, y, op, z } => {
                let (xs, ys) = (self.floats(x)?, self.floats(y)?);
                let mut out = Vec::with_capacity(xs.v.len());
                for t in 0..xs.rows {
                    for (c, &a) in xs.row(t).iter().enumerate() {
                        let b = ys.v[t * ys.width + if ys.width == 1 { 0 } else { c }];
                        out.push(match op {
                            BinaryOp::Add => a + b,
                            BinaryOp::Mul => a * b,
                            BinaryOp::Div => a / b,
                            BinaryOp::Mod => a - b * (a / b).floor(),
                        });
                    }
                }
                let rows = xs.rows;
                self.put(z, rows, out)
            }
            Elemwise::Act {
                s,
                x,
                act: f,
                scale,
                x_out,
            } => {
                let k = match (scale, s) {
                    (Some(k), _) => k,
                    (None, Some(s)) => self.floats(s)?.v[0],
                    (None, None) => 1.0,
                };
                let xs = self.floats(x)?;
                let out = xs.v.iter().map(|&v| act(f, v * k)).collect();
                let rows = xs.rows;
                self.put(x_out, rows, out)
            }
            Elemwise::AddBias { bias, out, out_out } => {
                let (b, xs) = (self.floats(bias)?, self.floats(out)?);
                if b.v.len() != xs.width {
                    return Err(format!("a bias of {} on rows of {}", b.v.len(), xs.width));
                }
                let v = (xs.v.iter().enumerate())
                    .map(|(i, x)| x + b.v[i % xs.width])
                    .collect();
                let rows = xs.rows;
                self.put(out_out, rows, v)
            }
            Elemwise::ResidualAdd {
                x,
                gate: Some(gate),
                y,
                y_out,
            } => {
                let (xs, gs, ys) = (self.floats(x)?, self.floats(gate.g)?, self.floats(y)?);
                let mut out = Vec::with_capacity(ys.v.len());
                for t in 0..ys.rows {
                    let g = gs.row(self.lane_row(gate.lane_of_row, t)?);
                    for (c, (&a, &b)) in xs.row(t).iter().zip(ys.row(t)).enumerate() {
                        out.push(g[c].mul_add(a, b));
                    }
                }
                let rows = ys.rows;
                self.put(y_out, rows, out)
            }
            Elemwise::RopeAxes {
                x,
                positions,
                dims,
                thetas,
                form,
                rotary_dim,
                head_dim,
                x_out,
            } => {
                let (xs, pos) = (self.floats(x)?, self.floats(positions)?);
                let (hd, angles) = (head_dim as usize, rotary_dim as usize / 2);
                let mut v = xs.v.to_vec();
                for t in 0..xs.rows {
                    for head in 0..xs.width / hd {
                        let base = t * xs.width + head * hd;
                        let (mut first_angle, mut first_channel) = (0, 0);
                        for (axis, &block) in dims.iter().enumerate() {
                            let block = block as usize;
                            for within in 0..block / 2 {
                                // the angle in f64, as diffusers' tables are made
                                let freq = f64::from(thetas[axis])
                                    .powf(-2.0 * within as f64 / block as f64);
                                let angle = f64::from(pos.v[t * pos.width + axis]) * freq;
                                let (sin, cos) = (angle.sin() as f32, angle.cos() as f32);
                                let (lo, hi) = match form {
                                    RopeForm::Interleaved => {
                                        (first_channel + 2 * within, first_channel + 2 * within + 1)
                                    }
                                    RopeForm::Neox => {
                                        (first_angle + within, first_angle + within + angles)
                                    }
                                    RopeForm::Split => {
                                        (first_channel + within, first_channel + block / 2 + within)
                                    }
                                    RopeForm::SplitLadder => {
                                        return Err("an axial rope in the split-ladder form".into());
                                    }
                                };
                                let (a, b) = (xs.v[base + lo], xs.v[base + hi]);
                                v[base + lo] = a.mul_add(cos, -(b * sin));
                                v[base + hi] = b.mul_add(cos, a * sin);
                            }
                            first_angle += block / 2;
                            first_channel += block;
                        }
                    }
                }
                let rows = xs.rows;
                self.put(x_out, rows, v)
            }
            _ => Err(format!("`{}` has no host forward", Operands::name(op))),
        }
    }

    pub(super) fn gather_more(&mut self, op: &Gather) -> Result<(), String> {
        match op {
            Gather::PackRows { x, perm, y } | Gather::UnpackRows { x, perm, y } => {
                let pack = matches!(op, Gather::PackRows { .. });
                let (xs, p) = (self.floats(*x)?, self.ints(*perm)?);
                let w = xs.width;
                let mut out = vec![f32::NAN; p.rows * w];
                for (r, &to) in p.v.iter().enumerate() {
                    let to = usize::try_from(to).map_err(|_| "a negative row in a permutation")?;
                    let (src, dst) = if pack { (to, r) } else { (r, to) };
                    if src < xs.rows && dst < p.rows {
                        out[dst * w..(dst + 1) * w].copy_from_slice(xs.row(src));
                    }
                }
                let rows = p.rows;
                self.put(*y, rows, out)
            }
            Gather::SplitRows {
                x,
                width,
                left,
                right,
            } => {
                let xs = self.floats(*x)?;
                let w = *width as usize;
                let (mut l, mut r) = (Vec::new(), Vec::new());
                for t in 0..xs.rows {
                    l.extend_from_slice(&xs.row(t)[..w]);
                    r.extend_from_slice(&xs.row(t)[w..]);
                }
                let rows = xs.rows;
                self.put(*left, rows, l)?;
                self.put(*right, rows, r)
            }
            Gather::ConcatCols { xs, y } => {
                let parts = xs
                    .iter()
                    .map(|&x| self.floats(x).map(|r| (r.rows, r.width, r.v.to_vec())))
                    .collect::<Result<Vec<_>, _>>()?;
                let rows = parts.first().map_or(0, |p| p.0);
                let mut out = Vec::new();
                for t in 0..rows {
                    for (_, w, v) in &parts {
                        out.extend_from_slice(&v[t * w..(t + 1) * w]);
                    }
                }
                self.put(*y, rows, out)
            }
            _ => Err(format!("`{}` has no host forward", Operands::name(op))),
        }
    }

    pub(super) fn attn_more(&mut self, op: &Attn) -> Result<(), String> {
        match *op {
            // each group's queries over its own keys and values, every one of them
            Attn::Ragged {
                q,
                k,
                v,
                q_indptr,
                kv_indptr,
                mask: RaggedMask::None | RaggedMask::GroupBlockDiagonal,
                head_dim,
                kv_heads,
                sm_scale,
                o,
            } => {
                let (qs, ks, vs) = (self.floats(q)?, self.floats(k)?, self.floats(v)?);
                let (qi, ki) = (self.ints(q_indptr)?, self.ints(kv_indptr)?);
                let hd = head_dim as usize;
                let heads = qs.width / hd;
                let group = heads / (kv_heads as usize).max(1);
                let mut out = vec![0.0; qs.rows * qs.width];
                for g in 0..qi.v.len().saturating_sub(1) {
                    let (q0, q1) = (qi.v[g] as usize, qi.v[g + 1] as usize);
                    let (k0, k1) = (ki.v[g] as usize, ki.v[g + 1] as usize);
                    for t in q0..q1.min(qs.rows) {
                        for h in 0..heads {
                            let qh = &qs.row(t)[h * hd..(h + 1) * hd];
                            let kh = h / group.max(1);
                            let scores: Vec<f32> = (k0..k1)
                                .map(|s| dot(qh, &ks.row(s)[kh * hd..(kh + 1) * hd]) * sm_scale)
                                .collect();
                            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
                            let row = &mut out[t * qs.width + h * hd..][..hd];
                            for (s, score) in (k0..k1).zip(&scores) {
                                let w = (score - max).exp() / sum;
                                for (o, v) in row.iter_mut().zip(&vs.row(s)[kh * hd..(kh + 1) * hd])
                                {
                                    *o += w * v;
                                }
                            }
                        }
                    }
                }
                let rows = qs.rows;
                self.put(o, rows, out)
            }
            _ => Err(format!("`{}` has no host forward", Operands::name(op))),
        }
    }

    /// The latent's patches and cells, over the step's clips.
    pub(super) fn conv(&mut self, op: &Conv) -> Result<(), String> {
        let n = self.rows.map_or(0, |r| r.token.len());
        let voxels = self.rows.map_or(0, |r| r.voxel.len());
        match *op {
            Conv::Patchify {
                x,
                grid,
                p,
                tgrid,
                y,
            } => {
                let (xs, g, tg) = (self.floats(x)?, self.ints(grid)?, self.ints(tgrid)?);
                let (g, tg) = (clips(g.v), clips(tg.v));
                let (r1, r2, r3) = (p[0] as usize, p[1] as usize, p[2] as usize);
                let r = r1 * r2 * r3;
                let c = xs.width;
                let mut out = vec![0.0; n * c * r];
                for row in 0..n {
                    let Some((l, og)) = clip_of(&tg, row) else {
                        continue;
                    };
                    let (t, h, w) = og.unravel(row - og.off);
                    for col in 0..c * r {
                        let (cin, block) = (col / r, col % r);
                        let (i1, i2, i3) = (block / (r2 * r3), (block / r3) % r2, block % r3);
                        let src = g[l].ravel(t * r1 + i1, h * r2 + i2, w * r3 + i3);
                        out[row * c * r + col] = xs.v[src * c + cin];
                    }
                }
                self.put(y, n, out)
            }
            Conv::Unpatchify {
                x,
                tgrid,
                p,
                grid,
                y,
            } => {
                let (xs, tg, g) = (self.floats(x)?, self.ints(tgrid)?, self.ints(grid)?);
                let (tg, g) = (clips(tg.v), clips(g.v));
                let (r1, r2, r3) = (p[0] as usize, p[1] as usize, p[2] as usize);
                let r = r1 * r2 * r3;
                let c = xs.width / r;
                let mut out = vec![0.0; voxels * c];
                for row in 0..voxels {
                    let Some((l, og)) = clip_of(&g, row) else {
                        continue;
                    };
                    let (t, h, w) = og.unravel(row - og.off);
                    let src = tg[l].ravel(t / r1, h / r2, w / r3);
                    let block = ((t % r1) * r2 + (h % r2)) * r3 + (w % r3);
                    for col in 0..c {
                        out[row * c + col] = xs.v[src * c * r + col * r + block];
                    }
                }
                self.put(y, voxels, out)
            }
            Conv::Cells {
                tgrid,
                positions,
                axis,
                y,
            } => {
                let (tg, pos) = (self.ints(tgrid)?, self.ints(positions)?);
                let tg = clips(tg.v);
                let axis = axis as usize;
                let width = axis.max(2) + 1 + 3;
                let mut out = vec![0.0; n * width];
                for row in 0..n {
                    let o = &mut out[row * width..(row + 1) * width];
                    match clip_of(&tg, row) {
                        Some((_, g)) => {
                            let (t, h, w) = g.unravel(row - g.off);
                            o[..3].copy_from_slice(&[t as f32, h as f32, w as f32]);
                            o[width - 3..].copy_from_slice(&[g.t as f32, g.h as f32, g.w as f32]);
                        }
                        None => o[axis] = pos.v[row] as f32,
                    }
                }
                self.put(y, n, out)
            }
            _ => Err(format!("`{}` has no host forward", Operands::name(op))),
        }
    }
}
