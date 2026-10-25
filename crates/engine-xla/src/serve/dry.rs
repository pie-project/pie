//! Synthetic fires: every class of a plan at a prefill and a decode shape.
//! Over a dry device (`Shell::load_dry`) they trace and run nothing (no
//! weights land, no pool holds a buffer); over a device that runs, they run
//! and read back (the e2e example checks what came back is finite). The coverage harness
//! (`examples/coverage.rs`) answers with it, for each catalog deployment, whether
//! engine-xla emits the plan's programs, and, on a compiling dry device,
//! whether the plugin compiles them.
//!
//! A class's lanes are synthetic as engine-cuda's arming lanes are
//! (`serve/arming.rs`): a representative request of the class (its word,
//! mask, adapter, score capture, switches), zero tokens, fresh pages; a class
//! that reads guest inputs is fed zero cells of each input's width, a class
//! over the voxel axis one clip, a class that lands media one image.

use engine::fire::{Mask, Masking};
use poem_ir::{Facts, Request};

use super::{Lane, Media, PATCH_ROUTE_DROP, Seated, Shell};
use crate::dit::{Clips, InputCell};
use crate::error::Result;

/// One dry fire: which class, at how many rows, and what came of it.
#[derive(Debug, Clone)]
pub struct Probe {
    pub class: usize,
    pub rows: u32,
    /// The representative request, as `Debug`.
    pub request: String,
    /// Programs traced (and compiled, on a compiling device) by the fire, or
    /// the refusal.
    pub outcome: std::result::Result<usize, String>,
    /// On a device that runs: whether every value the fire read back (its
    /// rows, drafts and pixels) is finite, and how many there were.
    pub finite: Option<(bool, usize)>,
}

/// Every request shape the arming pass enumerates (engine-cuda
/// `exports::landing_requests`), by the class it lands in, the plainest
/// first.
#[must_use]
pub fn landing(facts: &Facts, classes: &poem_ir::ClassTable) -> Vec<Vec<Request>> {
    let mut landing = vec![Vec::new(); classes.classes.len()];
    for request in facts.requests() {
        let word = facts.word(&request) & classes.mask;
        if let Some(class) = classes.class_of(word) {
            landing[class].push(request);
        }
    }
    landing
}

/// What one synthetic lane owns.
struct Owned {
    word: u64,
    tokens: Vec<u32>,
    pages: Vec<u32>,
    window: Vec<u32>,
    mask: Option<Masking>,
    adapter: Option<u32>,
    inputs: Vec<(u32, Vec<u8>)>,
    clip: Option<([u32; 3], Vec<u8>)>,
    media: Option<OwnedMedia>,
    kv_less: bool,
    /// A block drafter reads every row back.
    readout: Option<Vec<u32>>,
}

struct OwnedMedia {
    rows: Vec<u32>,
    patches: Vec<u8>,
    routes: Vec<i32>,
    positions: Vec<i32>,
    grids: Vec<i32>,
}

impl Shell {
    /// Traces each class of the plan at `prefill` rows and at one row (a
    /// class whose tokens a clip sets fires at the clip's), one fire per
    /// class and shape; runs nothing. The shell must be dry.
    ///
    /// `lean` skips the classes that run a custom-mask or adapter arm (a
    /// compile check keeps the device briefly).
    pub fn synthetic_fires(&mut self, prefill: u32, lean: bool) -> Vec<Probe> {
        let facts = self.trace().facts.clone();
        let landing = landing(&facts, &self.compiled_model().classes);
        let mut probes = Vec::new();
        for (class, requests) in landing.iter().enumerate() {
            // A score capture this shell does not serve.
            if self.capturing.contains(class)
                || lean && (self.masked.contains(class) || self.corrected.contains(class))
            {
                continue;
            }
            let plain = |prefill: bool| {
                requests
                    .iter()
                    .find(|r| (r.query_len() != 1) == prefill && !r.has_media())
                    .cloned()
            };
            // (request, rows per lane, lanes)
            let mut shapes: Vec<(Request, u32, u32)> = Vec::new();
            if let Some(request) = plain(true) {
                shapes.push((request, prefill, 1));
            }
            if let Some(request) = plain(false) {
                shapes.push((request, 1, 3));
            }
            if self.patch_seat.is_some()
                && let Some(media) = requests.iter().find(|r| r.has_media()).cloned()
            {
                shapes.push((media, prefill.max(16), 1));
            }
            for (request, rows, lanes) in shapes {
                let before = self.device().dry_texts().len();
                let word = facts.word(&request);
                let fired = self.dry_fire(class, &request, word, rows, lanes);
                let finite = match (&fired, self.device().is_dry()) {
                    (Ok(fired), false) => {
                        let values = fired
                            .rows
                            .iter()
                            .chain(&fired.drafts)
                            .flatten()
                            .chain(fired.pixels.iter().flat_map(|(p, _)| p));
                        let (mut all, mut n) = (true, 0usize);
                        for v in values {
                            all &= v.is_finite();
                            n += 1;
                        }
                        Some((all, n))
                    }
                    _ => None,
                };
                let outcome = fired
                    .map(|_| self.device().dry_texts().len() - before)
                    .map_err(|fault| fault.to_string());
                probes.push(Probe {
                    class,
                    rows: rows * lanes,
                    request: format!("{request:?}"),
                    outcome,
                    finite,
                });
            }
        }
        probes
    }

    fn dry_fire(
        &mut self,
        class: usize,
        request: &Request,
        word: u64,
        rows: u32,
        lanes: u32,
    ) -> Result<crate::serve::Fired> {
        if !self.device().is_dry() {
            for slot in 0..lanes {
                self.open(slot)?;
            }
        }
        let owned: Vec<Owned> = (0..lanes)
            .map(|at| self.dry_lane(class, request, word, rows, at))
            .collect::<Result<_>>()?;
        let cells: Vec<Vec<InputCell<'_>>> = owned
            .iter()
            .map(|lane| {
                lane.inputs
                    .iter()
                    .map(|(id, bytes)| InputCell { id: *id, bytes })
                    .collect()
            })
            .collect();
        let mut seated_lanes = Vec::with_capacity(owned.len());
        for (at, lane_owned) in owned.iter().enumerate() {
            let mut seated = Seated::of(Lane {
                slot: at as u32,
                word: lane_owned.word,
                tokens: &lane_owned.tokens,
            });
            seated.pages = &lane_owned.pages;
            seated.window = &lane_owned.window;
            seated.held = Some(0);
            seated.mask = lane_owned.mask.as_ref();
            seated.adapter = lane_owned.adapter;
            seated.inputs = &cells[at];
            seated.kv_less = lane_owned.kv_less;
            seated.readout = lane_owned.readout.as_deref();
            seated.media = lane_owned.media.as_ref().map(|m| Media {
                rows: &m.rows,
                patches: &m.patches,
                routes: &m.routes,
                positions: &m.positions,
                grids: &m.grids,
                token_positions: &[],
            });
            seated_lanes.push(seated);
        }
        let clips: Vec<Clips<'_>> = owned
            .iter()
            .enumerate()
            .filter_map(|(at, lane)| {
                lane.clip.as_ref().map(|(b, payload)| Clips {
                    lane: at as u32,
                    clips: std::slice::from_ref(b),
                    payload,
                })
            })
            .collect();
        self.fire_full_with(&seated_lanes, &clips)
    }

    fn dry_lane(
        &self,
        class: usize,
        request: &Request,
        word: u64,
        rows: u32,
        lane: u32,
    ) -> Result<Owned> {
        let dit = &self.dit;
        let voxel = self
            .dry_voxel_class(class)
            .then(|| dit.dry_voxel())
            .flatten();
        // A clip that patches into tokens sets the lane's rows.
        let (rows, clip) = match voxel {
            None => (rows, None),
            Some((widths, dtype, patch)) => {
                let p = patch.unwrap_or([1, 1, 1]);
                let b = [p[0], 2 * p[1], 2 * p[2]];
                let voxels = (b[0] * b[1] * b[2]) as usize;
                let elem = poem_compiler::arena::elem_bytes(dtype).unwrap_or(2) as usize;
                let payload = match dit.dry_voxel_width(class).or(widths.first().copied()) {
                    Some(w) => vec![0u8; voxels * w as usize * elem],
                    None => Vec::new(),
                };
                let rows = if patch.is_some() { 4 } else { 1 };
                (rows, Some((b, payload)))
            }
        };
        let page_size = self.paging().page_size.max(1);
        let per = rows.div_ceil(page_size).max(1);
        let pages: Vec<u32> = (lane * per..(lane + 1) * per).collect();
        let windowed = self.paging().window_pages().saturating_sub(1).max(1);
        let window = if self.pools.has_windowed() {
            pages
                .iter()
                .map(|&page| 1 + u32::try_from(u64::from(page) % windowed).unwrap_or(0))
                .collect()
        } else {
            Vec::new()
        };
        let inputs = dit
            .dry_inputs(class)
            .into_iter()
            .map(|seat| {
                let n = if seat.per_lane() { 1 } else { rows } as usize;
                (seat.id, vec![0u8; n * seat.row_bytes() as usize])
            })
            .collect();
        let media = if request.has_media() {
            self.dry_media(rows)
        } else {
            None
        };
        Ok(Owned {
            word,
            tokens: vec![0; rows as usize],
            pages,
            window,
            // The shell checks each against the arms the class runs.
            mask: self
                .masked
                .contains(class)
                .then(|| Masking::Extent(Mask::new(vec![0, rows], u64::from(rows)))),
            adapter: self.corrected.contains(class).then_some(0),
            readout: (self.trace().drafter.is_some() && !request.switches().is_empty())
                .then(|| (0..rows).collect()),
            inputs,
            clip,
            media,
            kv_less: !self.dry_class_has_kv(class),
        })
    }
}

impl Shell {
    /// Whether a class runs a node over the voxel axis.
    fn dry_voxel_class(&self, class: usize) -> bool {
        self.dry_class_runs(class, |trace, op| {
            let mut inputs = Vec::new();
            poem_ir::Operands::inputs(op, &mut inputs);
            inputs.iter().any(|v| {
                matches!(&trace.values[v.0 as usize].ty, poem_ir::Ty::Tensor { shape, .. }
                    if shape.first().and_then(|dim| dim.axis()) == Some(poem_ir::RowAxis::Voxels))
            })
        })
    }

    /// Whether a class reads or writes a kv cache.
    fn dry_class_has_kv(&self, class: usize) -> bool {
        self.dry_class_runs(class, |_, op| crate::store::kv::reads(op).is_some())
            || self.trace.caches.is_empty()
    }

    fn dry_class_runs(
        &self,
        class: usize,
        pick: impl Fn(&poem_ir::Trace, &poem_ir::Operation) -> bool,
    ) -> bool {
        self.compiled.template().iter().any(|region| {
            region.mask.contains(class)
                && region
                    .nodes
                    .clone()
                    .any(|node| pick(&self.trace, &self.trace.nodes[node as usize].op))
        })
    }

    /// One image of the plan's tower: a lattice of patches, all zero.
    fn dry_media(&self, rows: u32) -> Option<OwnedMedia> {
        let seat = self.patch_seat?;
        let fold = self.patch_fold.max(1);
        let patches = (poem_compiler::PATCH_LATTICE_FLOOR / fold).max(1) * fold;
        let n = patches as usize;
        let live = n / fold as usize;
        let mut routes = vec![
            if self.drops_patch_rows {
                PATCH_ROUTE_DROP
            } else {
                0
            };
            n
        ];
        for (j, route) in routes.iter_mut().take(live).enumerate() {
            *route = (j % rows.max(1) as usize) as i32;
        }
        Some(OwnedMedia {
            rows: vec![patches],
            patches: vec![0u8; n * seat.row_bytes as usize],
            routes,
            positions: vec![0i32; n * 3],
            grids: vec![1, 1, patches as i32],
        })
    }
}
