//! The synthetic fires arming captures its bodies by: their lanes, media,
//! voxels and patches, seated and fired.

#[allow(clippy::wildcard_imports)]
use super::*;

impl Shell {
    pub(super) fn synthetic_lanes(&self, lanes: &[LaneRows]) -> Vec<Synthetic> {
        self.synthetic_lanes_with(lanes, &[])
    }

    /// The records, kv pages and windowed pages the lattice's synthetic
    /// fires name ([`Pools::synthetic_room`]): the widest fire's.
    pub(super) fn synthetic_room(&self) -> (u32, u64, u64) {
        let page_size = u64::from(self.pools.paging().page_size).max(1);
        let lanes_most = self.fire_lanes().max(1);
        self.pools.synthetic_room(
            (self.held.len().max(1) as u32).min(lanes_most),
            u64::from(self.budget.max_tokens).div_ceil(page_size) + u64::from(lanes_most),
        )
    }

    /// What the runtime reads of this shell while it serves.
    #[must_use]
    pub fn live(&self) -> engine::live::Live {
        self.live.clone()
    }

    pub(super) fn synthetic_lanes_with(
        &self,
        lanes: &[LaneRows],
        media: &[LaneMedia],
    ) -> Vec<Synthetic> {
        let page_size = u64::from(self.pools.paging().page_size).max(1);
        // What the lanes name, inside the chunks the fit left the pool: a
        // stated kv budget may hold fewer records and pages than the
        // lattice's widest fire names, and a synthetic lane's rows are never
        // read back, so its lanes share them there. The pool seats what a
        // fire names for good, so every fire of the lattice folds into one
        // room: the widest fire's.
        let (slots, room, windowed) = self.synthetic_room();
        let slot_of = |at: u32| at;
        let page_of = |at: u64| u32::try_from(at).unwrap_or(u32::MAX);
        let mut next_page = 0u64;
        let row_bytes = self.patch_seat.map_or(0, |seat| seat.row_bytes) as usize;
        let fold = (self.patch_fold as usize).max(1);
        // The voxel seat's patches and clips the lanes feed from, in order.
        let mut voxel_room = self.voxels.as_ref().map_or((0, 0), |store| {
            let seat = store.seat();
            (
                u32::try_from(seat.rows).unwrap_or(u32::MAX)
                    / self.token_patch().iter().product::<u32>(),
                u32::try_from(seat.clips).unwrap_or(u32::MAX),
            )
        });
        lanes
            .iter()
            .enumerate()
            .map(|(at, &(class, rows))| {
                let wants_media = media.get(at).is_some_and(|(images, _)| *images > 0);
                let request = self.representative(class, rows, wants_media);
                Synthetic {
                    word: self.trace.facts.word(&request),
                    tokens: vec![0u32; rows as usize],
                    mask: request
                        .has_custom_mask()
                        .then(|| Masking::Extent(Mask::new(vec![0, rows], u64::from(rows)))),
                    adapter: request.has_adapter().then_some(0),
                    slot: slot_of((at as u32) % slots),
                    window: Vec::new(),
                    pages: {
                        let pages = u64::from(rows).div_ceil(page_size).max(1);
                        let table: Vec<u32> = (next_page..next_page + pages)
                            .map(|page| page_of(page % room))
                            .collect();
                        next_page += pages;
                        table
                    },
                    held: Some(0),
                    voxels: self.synthetic_voxels(class, rows, &mut voxel_room),
                    media: media
                        .get(at)
                        .copied()
                        .filter(|(images, patches)| *images > 0 && *patches > 0)
                        .map(|(images, patches)| {
                            let patches = patches as usize;
                            let per = patches / images as usize;
                            let mut per_image: Vec<u32> = vec![per as u32; images as usize];
                            per_image[0] += (patches - per * images as usize) as u32;
                            let live = patches / fold;
                            let mut routes = vec![
                                if self.drops_patch_rows {
                                    PATCH_ROUTE_DROP
                                } else {
                                    0
                                };
                                patches
                            ];
                            for (j, route) in routes.iter_mut().take(live).enumerate() {
                                *route = (j % rows.max(1) as usize) as i32;
                            }
                            let grids = per_image
                                .iter()
                                .flat_map(|&rows| [1, rows as i32, 1])
                                .collect();
                            SyntheticMedia {
                                rows: per_image,
                                patches: vec![0u8; patches * row_bytes],
                                routes,
                                positions: vec![0i32; patches * MROPE_COORDS],
                                grids,
                            }
                        }),
                }
            })
            .map(|mut lane| {
                // A synthetic lane's content is never read back, so its
                // pages fold into the windowed pool past the null page.
                let windowed = self
                    .pools
                    .paging()
                    .window_pages()
                    .min(windowed)
                    .saturating_sub(1)
                    .max(1);
                if self.pools.has_windowed() {
                    lane.window = lane
                        .pages
                        .iter()
                        .map(|&page| 1 + u32::try_from(u64::from(page) % windowed).unwrap_or(0))
                        .collect();
                }
                lane
            })
            .collect()
    }

    /// The voxel input a synthetic lane of `class` and `rows` token rows
    /// feeds, out of `room` (the seat's patches and clips the fire's lanes
    /// before it left): the most patches its rows bind (a lane's patches are
    /// at most its token rows), as clips ([`Shell::clips_of`]), at its
    /// input's width, so the body recorded carries any fire's (a voxel axis
    /// is not bucketed: a body's grids are its fire's). `None` where its
    /// class reads none, the load seats no voxel axis or no room is left.
    pub(super) fn synthetic_voxels(
        &self,
        class: usize,
        rows: u32,
        room: &mut (u32, u32),
    ) -> Option<SyntheticVoxels> {
        let input = self
            .feeds
            .inputs
            .iter()
            .find(|(input, readers)| input.voxel() && readers.contains(class))
            .map(|(input, _)| *input)?;
        let (most, _) = self.seat_patches(class)?;
        let patches = most.min(rows).min(room.0);
        let (clips, voxels) = self.clips_of(patches, room.1)?;
        room.0 -= patches;
        room.1 -= u32::try_from(clips.len()).unwrap_or(room.1);
        Some(SyntheticVoxels {
            clips,
            payload: vec![0u8; usize::try_from(u64::from(voxels) * input.row_bytes()).ok()?],
        })
    }

    /// The token patches a fire of `class` feeds at most (the seat's rows its
    /// class reads, by its voxel growth, in whole patches) and the seat's
    /// clips; `None` where it reads none.
    pub(super) fn seat_patches(&self, class: usize) -> Option<(u32, u32)> {
        let seat = self.voxels.as_ref()?.seat();
        let (grown, read) = poem_compiler::arena::voxel_growth(&self.trace, &self.compiled.classes)
            .get(class)
            .copied()?;
        let rows = u32::try_from(seat.rows * u64::from(read) / u64::from(grown.max(1)))
            .unwrap_or(u32::MAX);
        let patches = rows / self.token_patch().iter().product::<u32>();
        let clips = u32::try_from(seat.clips.max(1)).unwrap_or(u32::MAX);
        (patches > 0).then_some((patches, clips))
    }

    /// The voxel seat's token patch (`[pt, ph, pw]`, a single voxel without
    /// one).
    pub(super) fn token_patch(&self) -> [u32; 3] {
        self.voxels
            .as_ref()
            .and_then(|store| store.seat().token_patch)
            .unwrap_or([1, 1, 1])
            .map(|p| p.max(1))
    }

    /// `patches` token patches as clips: laid along t over at most `clips`
    /// clips, the first taking what the even split leaves; the boxes and
    /// their voxels. `None` where there are none.
    pub(super) fn clips_of(&self, patches: u32, clips: u32) -> Option<(Vec<[u32; 3]>, u32)> {
        let [pt, ph, pw] = self.token_patch();
        let clips = clips.min(patches);
        if clips == 0 {
            return None;
        }
        let each = patches / clips;
        let boxes = (0..clips)
            .map(|at| {
                [
                    (each + if at == 0 { patches % clips } else { 0 }) * pt,
                    ph,
                    pw,
                ]
            })
            .collect();
        Some((boxes, patches * pt * ph * pw))
    }

    /// The class a lane of `class` lands in once it carries an adapter: the
    /// one whose requests depart from `class`'s as it does and by the
    /// adapter besides; `None` where the plan lands no such request.
    pub(super) fn adapted_twin(&self, class: usize) -> Option<usize> {
        let plain = self.landing.get(class)?.first()?;
        if plain.has_adapter() {
            return None;
        }
        self.landing.iter().position(|requests| {
            requests.iter().any(|request| {
                request.has_adapter()
                    && request.has_media() == plain.has_media()
                    && request.drafts() == plain.drafts()
                    && (request.query_len() == 1) == (plain.query_len() == 1)
                    && request.departures() == plain.departures() + 1
            })
        })
    }

    pub(super) fn representative(
        &self,
        class: usize,
        rows: u32,
        wants_media: bool,
    ) -> poem_ir::Request {
        let landing = &self.landing[class];
        // A bare request first (no adapter, no score capture): a switched
        // class's lands beside its adapter and scoring twins.
        let bare =
            |request: &&poem_ir::Request| !request.has_adapter() && !request.captures_scores();
        landing
            .iter()
            .filter(bare)
            .find(|request| {
                request.has_media() == wants_media && (request.query_len() == 1) == (rows == 1)
            })
            .or_else(|| {
                landing.iter().find(|request| {
                    request.has_media() == wants_media && (request.query_len() == 1) == (rows == 1)
                })
            })
            .or_else(|| {
                landing
                    .iter()
                    .find(|request| request.has_media() == wants_media)
            })
            .or_else(|| landing.first())
            .cloned()
            .unwrap_or_else(|| {
                panic!("the arming pass enumerated class {class}, which no request lands in")
            })
    }

    pub(super) fn seated(owned: &[Synthetic]) -> Vec<Seated<'_>> {
        owned
            .iter()
            .map(|lane| Seated {
                lane: Lane {
                    slot: lane.slot,
                    word: lane.word,
                    tokens: &lane.tokens,
                },
                positions: &[],
                pages: &lane.pages,
                held: lane.held,
                folded: 0,
                readable: 0,
                set: 0,
                kv_less: false,
                translation: &[],
                write_end: 0,
                window: &lane.window,
                window_fold: None,
                mask: lane.mask.as_ref(),
                adapter: lane.adapter,
                readout: None,
                rs: RsVerb::Fold,
                rs_reset: RsReset::Inferred,
                rs_slot: None,
                rs_read: None,
                rs_snapshots: &[],
                replay_stated: None,
                group: None,
                inputs: &[],
                reports: 0,
                tap: None,
            })
            .collect()
    }

    pub(super) fn fire_synthetic(&mut self, owned: &[Synthetic]) -> Result<()> {
        self.fire_synthetic_as(owned, crate::serve::Golden::Off)
            .map(|_| ())
    }

    pub(super) fn fire_synthetic_as(
        &mut self,
        owned: &[Synthetic],
        arm: crate::serve::Golden,
    ) -> Result<Vec<Vec<f32>>> {
        let seated = Self::seated(owned);

        let media: Vec<Media<'_>> = owned
            .iter()
            .enumerate()
            .filter_map(|(at, lane)| {
                lane.media.as_ref().map(|shot| Media {
                    lane: at as u32,
                    rows: &shot.rows,
                    patches: &shot.patches,
                    routes: &shot.routes,
                    positions: &shot.positions,
                    token_positions: &[],
                    grids: &shot.grids,
                })
            })
            .collect();
        let voxels: Vec<crate::voxels::Clips<'_>> = owned
            .iter()
            .enumerate()
            .filter_map(|(at, lane)| {
                lane.voxels.as_ref().map(|shot| crate::voxels::Clips {
                    lane: at as u32,
                    clips: &shot.clips,
                    payload: &shot.payload,
                })
            })
            .collect();

        self.arming = true;
        self.golden_arm = arm;
        let armed = self.fire_view(&seated, &media, &voxels);
        self.golden_arm = crate::serve::Golden::Off;
        self.arming = false;
        armed
    }
}
