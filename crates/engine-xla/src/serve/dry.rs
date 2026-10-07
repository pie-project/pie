//! Synthetic fires: every class of a plan at a prefill and a decode shape.
//! Over a dry device (`Shell::load_dry`) they trace and run nothing (no
//! weights land, no pool holds a buffer); over a device that runs, they run
//! and read back (the e2e example checks what came back is finite). The coverage harness
//! (`examples/coverage.rs`) answers with it, for each catalog SKU, whether
//! engine-xla emits the plan's programs, and, on a compiling dry device,
//! whether the plugin compiles them.
//!
//! A class's lanes are synthetic as engine-cuda's arming lanes are
//! (`serve/arming.rs`): a representative request of the class (its word,
//! mask, adapter, drafts, score capture), zero tokens, fresh pages; a class
//! that reads float ports is fed zero cells of each port's width, a class
//! over the voxel axis one clip, a class that lands media one image.

use engine::fire::{Mask, Masking};
use poem_ir::{Facts, Request, Stream};

use super::{Lane, Media, PATCH_ROUTE_DROP, Seated, Shell};
use crate::dit::{Clips, PortCell, SelfCond};
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
/// `exports::landing_requests`), by the class it lands in.
#[must_use]
pub fn landing(facts: &Facts, classes: &poem_ir::ClassTable) -> Vec<Vec<Request>> {
    let mut landing = vec![Vec::new(); classes.classes.len()];
    let readings: Vec<Option<&str>> = std::iter::once(None)
        .chain(facts.values("reading").map(Some))
        .collect();
    for reading in readings {
        for stream in Stream::ALL {
            for bits in 0..256u32 {
                let request = Request::new(if bits & 1 == 0 { 1 } else { 2 }, bits & 2 != 0)
                    .adapted(bits & 4 != 0)
                    .drafting(bits & 8 != 0)
                    .capturing_scores(bits & 16 != 0)
                    .with_media(bits & 32 != 0)
                    .denoising(bits & 64 != 0)
                    .drafting_a_block(bits & 128 != 0)
                    .on_stream(stream);
                let request = match reading {
                    Some(name) => request.in_reading(name),
                    None => request,
                };
                let word = facts.word(&request) & classes.mask;
                if let Some(class) = classes.class_of(word) {
                    landing[class].push(request);
                }
            }
        }
    }
    for requests in &mut landing {
        requests.sort_by_key(|r| {
            u32::from(r.has_custom_mask())
                + u32::from(r.has_adapter())
                + u32::from(r.drafts())
                + u32::from(r.captures_scores())
                + u32::from(r.has_media())
                + u32::from(r.denoise())
                + u32::from(r.drafts_a_block())
                + u32::from(r.stream() != Stream::Text)
                + u32::from(r.reading().is_some())
        });
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
    captures: bool,
    stream: u8,
    ports: Vec<(engine::fire::PortKind, u8, Vec<f32>)>,
    self_cond: Option<(u32, Vec<i32>, Vec<f32>)>,
    clip: Option<([u32; 3], Vec<u8>)>,
    media: Option<OwnedMedia>,
    kv_less: bool,
    bidirectional: bool,
    /// A block drafter reads every row back.
    readout: Option<Vec<u32>>,
}

struct OwnedMedia {
    rows: Vec<u32>,
    patches: Vec<u8>,
    routes: Vec<i32>,
    positions: Vec<i32>,
    embed_rows: Vec<i32>,
    embed_weights: Vec<f32>,
}

impl Shell {
    /// Traces each class of the plan at `prefill` rows and at one row (a
    /// class whose tokens a clip sets fires at the clip's), one fire per
    /// class and shape; runs nothing. The shell must be dry.
    ///
    /// `lean` skips the classes that run a custom-mask, adapter or score
    /// capture arm (a compile check keeps the device briefly).
    pub fn synthetic_fires(&mut self, prefill: u32, lean: bool) -> Vec<Probe> {
        let facts = self.trace().facts.clone();
        let landing = landing(&facts, &self.compiled_model().classes);
        let mut probes = Vec::new();
        for (class, requests) in landing.iter().enumerate() {
            if lean
                && (self.masked.contains(class)
                    || self.corrected.contains(class)
                    || self.capturing.contains(class))
            {
                continue;
            }
            let plain = |prefill: bool| {
                requests
                    .iter()
                    .find(|r| (r.query_len() != 1) == prefill && !r.has_media())
                    .copied()
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
                && let Some(media) = requests.iter().find(|r| r.has_media()).copied()
            {
                shapes.push((media, prefill.max(16), 1));
            }
            for (request, rows, lanes) in shapes {
                let before = self.device().dry_texts().len();
                let fired = self.dry_fire(class, request, facts.word(&request), rows, lanes);
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
        request: Request,
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
        let cells: Vec<Vec<PortCell<'_>>> = owned
            .iter()
            .map(|lane| {
                lane.ports
                    .iter()
                    .map(|(kind, port, values)| PortCell {
                        kind: *kind,
                        port: *port,
                        values,
                    })
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
            seated.captures_scores = lane_owned.captures;
            seated.stream = lane_owned.stream;
            seated.ports = &cells[at];
            seated.kv_less = lane_owned.kv_less;
            seated.bidirectional = lane_owned.bidirectional;
            seated.readout = lane_owned.readout.as_deref();
            seated.self_cond =
                lane_owned
                    .self_cond
                    .as_ref()
                    .map(|(taps, rows, weights)| SelfCond {
                        taps: *taps,
                        rows,
                        weights,
                    });
            seated.media = lane_owned.media.as_ref().map(|m| Media {
                rows: &m.rows,
                patches: &m.patches,
                routes: &m.routes,
                positions: &m.positions,
                embed_rows: &m.embed_rows,
                embed_weights: &m.embed_weights,
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
        request: Request,
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
        let ports = dit
            .dry_ports(class)
            .into_iter()
            .map(|(kind, port, width, per_lane)| {
                let n = if per_lane { 1 } else { rows } as usize * width as usize;
                (kind, port, vec![0f32; n])
            })
            .collect();
        let taps = dit.dry_self_cond_taps();
        let self_cond = (taps > 0 && request.denoise()).then(|| {
            let n = rows as usize * taps as usize;
            (taps, vec![0i32; n], vec![0f32; n])
        });
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
            captures: self.capturing.contains(class),
            bidirectional: request.drafts_a_block(),
            readout: request.drafts_a_block().then(|| (0..rows).collect()),
            stream: request.stream().code(),
            ports,
            self_cond,
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
        let taps = seat.embed_taps as usize;
        let weight_taps = if seat.embed_weights { taps } else { 0 };
        let mut embed_weights = vec![0f32; n * weight_taps];
        for row in embed_weights.chunks_mut(weight_taps.max(1)) {
            if let Some(first) = row.first_mut() {
                *first = 1.0;
            }
        }
        Some(OwnedMedia {
            rows: vec![patches],
            patches: vec![0u8; n * seat.row_bytes as usize],
            routes,
            positions: vec![0i32; n * 3],
            embed_rows: vec![0i32; n * taps],
            embed_weights,
        })
    }
}
