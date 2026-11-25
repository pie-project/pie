//! The shell's pipelines: the cover fingerprint, the pipeline cache keys and
//! archive, and the load's walk that warms every step shape's pipelines.

use super::stage::Form;
use super::*;

impl Shell {

    /// What the runtime's host cover depends on here
    /// (`engine::Engine::cover_fingerprint`): the device and its OS release
    /// (`crate::stored::measured_on`, the part a committed cover is filed
    /// under), the guest emitter's version, the kernel sources a fire
    /// dispatches, and the picks that route among them. Not the binary or
    /// the OS's build: a build or update that changes none of these reuses
    /// an earlier boot's cover, or the device's committed one.
    #[must_use]
    pub fn cover_fingerprint(&self) -> String {
        let fnv = |bytes: &[u8]| {
            bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
        };
        let kernels = kernels_metal::sources::SOURCES.iter().fold(
            0xcbf2_9ce4_8422_2325u64,
            |h, (name, text)| {
                name.bytes()
                    .chain([0u8])
                    .chain(text.bytes())
                    .chain([0u8])
                    .fold(h, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
            },
        );
        // Every decision as the load resolved it, rung by rung (the dense
        // ladder measured or taken at load among them): the tuning's own
        // Debug names the profile's source alone, so a pick that changed
        // under one source kept the cover an earlier pick was timed under.
        let decisions: String = self
            .tuning
            .profile()
            .entries()
            .map(|(spec, rungs, _)| format!("{}={rungs:?};", spec.id))
            .collect();
        format!(
            "metal|{}|emitter {}|kernels {kernels:016x}|tuning {:016x}|picks {:016x}",
            crate::stored::measured_on(self.device.name()),
            eta_compiler::codegen::metal::METAL_M1_EMITTER_VERSION,
            // What routes beside the picks: not the tuning's Debug, which
            // names the profile's source, so a measured boot and the
            // committed boot of the same picks would key apart.
            fnv(format!(
                "wide {} cores {:?} family {}",
                self.tuning.wide_range(),
                self.tuning.cores(),
                self.device.family()
            )
            .as_bytes()),
            fnv(decisions.as_bytes()),
        )
    }

    /// Builds, before ready, every compute pipeline a fire of this load
    /// can meet: at each rung, a prefill fire of the rung's rows (one lane)
    /// and a decode fire of as many one-row lanes as the rung and the seats
    /// take, each prepared and encoded and its frame dropped uncommitted,
    /// so nothing runs and no seat, slot or rung changes. Returns the fires
    /// walked and
    /// the pipelines built.
    /// What decides the pipelines this load builds: the device, its OS
    /// build, this binary, the model's graph, the rungs, the token and lane
    /// budgets, the paging and the kernels' tuning. A later boot with the
    /// same key builds the same keys; any change and it walks again.
    #[must_use]
    pub fn pipeline_cache_key(&self) -> String {
        let fnv = |bytes: &[u8]| {
            bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
        };
        #[cfg(target_vendor = "apple")]
        let os = objc2_foundation::NSProcessInfo::processInfo()
            .operatingSystemVersionString()
            .to_string();
        #[cfg(not(target_vendor = "apple"))]
        let os = String::new();
        let binary = std::env::current_exe()
            .and_then(std::fs::metadata)
            .map(|meta| {
                let at = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_nanos());
                format!("{}:{at}", meta.len())
            })
            .unwrap_or_default();
        let trace = fnv(format!("{:?}", self.trace).as_bytes());
        let paging = self.paging();
        // Every decision the profile resolves, rung by rung: the tuning's
        // own Debug names the profile's source alone, so a decision that
        // changed under the same source (kernel.dense measured anew at
        // load, a user file edited) kept the key and built an earlier
        // boot's keys, and a fire met a kernel that list did not hold.
        let decisions: String = self
            .tuning
            .profile()
            .entries()
            .map(|(spec, rungs, _)| format!("{}={rungs:?};", spec.id))
            .collect();
        let shape = format!(
            "{:?}|{:?}|{}:{}:{}|{:?}|{:016x}",
            self.arena.rung_tokens(),
            self.budgets.tokens,
            paging.page_size,
            paging.pages_per_slot,
            paging.context(),
            self.tuning,
            fnv(decisions.as_bytes())
        );
        format!(
            "metal-pipelines-1|{}|{os}|{binary}|{trace:016x}|{:016x}",
            self.device.name(),
            fnv(shape.as_bytes())
        )
    }

    /// The keys of every pipeline built: file, entrypoint, stamp and page
    /// strides.
    #[must_use]
    pub fn pipeline_keys(&self) -> Vec<(String, String, String, Option<[u64; 2]>)> {
        self.pipelines
            .keys()
            .into_iter()
            .map(|(file, entry, stamp, strides)| (file.into(), entry.into(), stamp.into(), strides))
            .collect()
    }

    /// Builds `keys` (a list an earlier boot of the same load wrote): the
    /// pipelines, before ready, without walking. Returns how many it built.
    pub fn build_pipelines(&self, keys: &[(String, String, String, Option<[u64; 2]>)]) -> Result<u64> {
        let before = self.pipelines.compiled();
        for (file, entry, stamp, strides) in keys {
            let leak = |text: &str| -> &'static str { Box::leak(text.to_owned().into_boxed_str()) };
            let fire = kernels_metal::Fire {
                stamp: leak(stamp),
                page_strides: *strides,
                ..kernels_metal::Fire::at(leak(file), leak(entry))
            };
            self.pipelines.warm(&self.device, fire)?;
        }
        Ok(self.pipelines.compiled() - before)
    }

    /// Builds against (or records into) the binary archive at `path`.
    #[cfg(target_vendor = "apple")]
    pub fn use_pipeline_archive(&self, path: &std::path::Path) {
        self.pipelines.use_archive(self.device.device(), path);
    }

    /// Writes the binary archive to `path`.
    #[cfg(target_vendor = "apple")]
    pub fn save_pipeline_archive(&self, path: &std::path::Path) -> bool {
        self.pipelines.save_archive(path)
    }

    /// Seals the pipelines built ([`Shell::warm_pipelines`]), with every
    /// fixed pipeline beside them (`kernels_metal::runtime::fixed_pipelines`:
    /// a beam group's head step and copies, and on a load buffering
    /// recurrent rows its row copies and carried scans), which a fire names
    /// by its lanes' shape and no walked rung keys. Refuses the load by name
    /// where one does not build, so a pipeline a fire can name is built
    /// before ready or the load fails: no fire builds one after ready, and
    /// none meets one the load lacks.
    pub fn seal_pipelines(&self) -> Result<()> {
        for fire in kernels_metal::runtime::fixed_pipelines(self.rs_layout.is_some()) {
            self.pipelines.warm(&self.device, fire).map_err(|fault| Fault::Device {
                call: "building a fixed pipeline before ready",
                why: format!(
                    "`{}` of `{}`, a pipeline this load's fires can name, did not build: {fault}",
                    fire.entrypoint, fire.file
                ),
            })?;
        }
        self.pipelines.seal();
        Ok(())
    }

    /// Fires refused for an unbuilt pipeline since the last call.
    pub fn take_unbuilt_pipelines(&self) -> u64 {
        self.pipelines.take_unbuilt()
    }

    /// The step set the runtime's builder forms over this load
    /// ([`engine::StepShapes::of`] of [`engine::StepRules::runtime`] over the
    /// load's own facts), for a walk the load request did not hand one.
    #[must_use]
    pub fn step_shapes(&self) -> engine::StepShapes {
        engine::StepShapes::of(
            &engine::StepRules::runtime(),
            &engine::StepFacts::of(
                &self.pool_chunk_facts(),
                self.budgets.tokens.max_lanes,
                self.budgets.tokens.max_tokens,
                u32::try_from(poem_compiler::arena::readouts_ceiling(&self.budgets.tokens))
                    .unwrap_or(u32::MAX),
                engine::Attachments::default(),
            ),
        )
    }

    /// [`Shell::warm_pipelines_over`] the runtime's set over this load
    /// ([`Shell::step_shapes`]).
    pub fn warm_pipelines(&mut self) -> Result<(u32, u64)> {
        let steps = self.step_shapes();
        self.warm_pipelines_over(&steps)
    }

    /// Builds, before ready, every pipeline a step of `steps` dispatches:
    /// each shape in the set's order, smallest first, at
    /// its rung, under its treatment's verbs, at the rows its span reaches
    /// (one, every matmul band start inside it, each unrolled few-rows
    /// matmul's top, 2 / 4 / 8, and its top) and at each readout tier inside
    /// its `readout_rows` (the same points, every row of the prompt read
    /// out), so a dense routing a step's rows or readouts reach is built (the
    /// M1's `dense_gemv_t_rows8` in a c64 step, met after ready and
    /// refused). Metal records no bodies: every shape is walked, none eager.
    pub fn warm_pipelines_over(&mut self, steps: &engine::StepShapes) -> Result<(u32, u64)> {
        use engine::frame::Shell as FrameShell;
        let before = self.pipelines.compiled();
        // One lane on seat 0: a fire's matmul tiles follow its rows, its
        // attention one row a lane (a one-row lane reaches the decode form),
        // its lm_head the rows it reads out; no key turns on how many lanes
        // or seats carry them, so the walk backs one seat run.
        let context = self.paging().context().max(1);
        let standing = self.rung;
        self.warming = true;
        // The forms the walk fires, from the plan: every word a request
        // reaches that serving arms (a plain one, a drafter's verify: plain
        // but for reading its pass's drafts, and one capturing its
        // attention scores, alone or with drafts), at each rung's prompt
        // and at one row; and on a model serving rs verbs, each under every
        // verb a lane takes (a fold, a buffer folding nothing, a buffer
        // replaying what it held and folding it), so its verify and drain
        // kernels are built too. A guest program's pipelines are built at
        // its registration, not here.
        let facts = self.trace.facts.clone();
        let mut forms: Vec<(bool, bool)> = facts
            .requests()
            .iter()
            .filter(|request| {
                request.departures()
                    == usize::from(request.query_len() != 1)
                        + usize::from(request.drafts())
                        + usize::from(request.captures_scores())
            })
            .map(|request| (request.drafts(), request.captures_scores()))
            .collect();
        forms.sort_unstable();
        forms.dedup();
        let word = |rows: u32, (drafts, scores): (bool, bool)| {
            facts.word(
                &poem::Request::new(rows, false)
                    .drafting(drafts)
                    .capturing_scores(scores),
            )
        };
        // The walk's lanes name slot refs, as the runtime's do: each shape's
        // first slot past the engine's tables run (the null chunk and the
        // tables, chunks 0..1+tables, are no slot's), inside the pool
        // whatever the seat count (seat 0's own runs, laid shape by shape
        // for every seat, can lie past the pool's last chunk). No kernel
        // reads them.
        let layout = *self.pools.layout();
        let past_tables = 1 + layout.tables;
        let first = |shape: Option<crate::store::layout::Slots>| {
            shape.map_or(0, |s| s.per_chunk * past_tables)
        };
        let page_size = self.paging().page_size.max(1);
        let kv_pages = vec![first(layout.list); context.div_ceil(page_size) as usize];
        let record = vec![first(layout.fixed); layout.pieces as usize];
        let windows: Vec<Vec<u32>> = layout
            .window_lists()
            .map(|window| {
                vec![window.slots.per_chunk * past_tables; context.div_ceil(window.tokens.max(1)) as usize]
            })
            .collect();
        let verbs: Vec<(engine::fire::RsVerb, u32)> = if self.serves_rs_verbs() {
            let per = layout.rs_tokens.max(1);
            let rs_ref = first(layout.rs);
            let pages = |tokens: u32| vec![rs_ref; tokens.div_ceil(per).max(1) as usize];
            let held = 1u32;
            vec![
                (engine::fire::RsVerb::Fold, 0),
                (
                    engine::fire::RsVerb::Buffer {
                        pages: pages(context),
                        base: 0,
                        folded: 0,
                        fold: engine::fire::FoldLen::Host(0),
                        lineage: engine::fire::Lineage::Set,
                    },
                    0,
                ),
                (
                    engine::fire::RsVerb::Buffer {
                        pages: pages(context),
                        base: 0,
                        folded: 0,
                        fold: engine::fire::FoldLen::Host(held),
                        lineage: engine::fire::Lineage::Set,
                    },
                    held,
                ),
            ]
        } else {
            vec![(engine::fire::RsVerb::Fold, 0)]
        };
        // A scored form (capturing its scores, alone or with drafts) is
        // walked at the rungs' octaves alone (`engine::load::armed`), every
        // plain one at every rung: a scored fire's kernels past the plain
        // ones' are the capture's, which no rung's rows key.
        let rungs = self.arena.rung_tokens();
        let octaves = engine::load::armed::scored_buckets(&rungs);
        // Every band start of the matmul routing
        // (`kernels_metal::linear::gemm::row_bands`: the fixed bands and
        // where the profile's `kernel.dense` changes its pick): a shape's
        // rows, and its readouts, reach every band its span holds.
        let bands = kernels_metal::linear::gemm::row_bands(&self.tuning);
        // A fire the Neural Engine shares leaves the rows past a whole
        // procedure to the GPU: their kernels turn on the tail's rows alone,
        // so one tail in each band, and at each few-rows top, builds them. A
        // later procedure's tail takes the same ones while a 32-row tile fits
        // past it: a rung wider than a tile is a whole number of tiles (the
        // ladder's powers of two, or a halving of `FIRE_CEILING`'s tokens).
        let tails: Vec<u32> = self.ane.as_ref().map_or_else(Vec::new, |ane| {
            bands
                .iter()
                .copied()
                .chain([2, 4, 8, 32])
                .filter_map(|tail| ane.with_tail(tail))
                .collect()
        });
        let inside = |span: engine::shapes::Span, extra: &[u32]| -> Vec<u32> {
            let mut at: Vec<u32> = bands
                .iter()
                .chain(extra)
                .copied()
                .chain([span.hi])
                .filter(|&n| n >= span.lo.max(1) && n <= span.hi)
                .collect();
            at.sort_unstable();
            at.dedup();
            at
        };
        // A treatment's verbs: a folding shape's lanes fold, a buffering
        // one's buffer (folding nothing, or replaying what they held).
        let verbs_of = |treatment: engine::Treatment| -> Vec<usize> {
            match treatment {
                engine::Treatment::Folds => vec![0],
                engine::Treatment::Buffers if verbs.len() > 1 => (1..verbs.len()).collect(),
                engine::Treatment::Buffers => vec![0],
            }
        };
        // What the walk fires, each once: (rung, prompt rows, rows read
        // out, form, verb), a prompt of one row the decode forms.
        let mut fired: std::collections::HashSet<(u32, u32, u32, (bool, bool), usize)> =
            std::collections::HashSet::new();
        let mut plan: Vec<(u32, u32, u32, (bool, bool), usize)> = Vec::new();
        for shape in &steps.shapes {
            // The arena's rung the shape runs at: its own, or the
            // narrowest past it (a lattice's last rung is one fire's tokens).
            let rung = rungs
                .iter()
                .copied()
                .find(|&r| r >= shape.rung)
                .or_else(|| rungs.last().copied())
                .unwrap_or(shape.rung);
            debug_assert!(
                self.ane.is_none() || rung < 32 || rung.is_multiple_of(32),
                "a rung of {rung} rows leaves a split's tail no whole tile"
            );
            let top = rung.min(self.budgets.tokens.max_tokens).min(context).max(1);
            let mut fires: Vec<(u32, u32)> = Vec::new();
            // A step's rows: one (the decode forms), and as a prompt each
            // band start inside its rows, each unrolled few-rows matmul's top
            // (2 / 4 / 8: a decode step of 5-8 lanes runs `rows8`), and its
            // top; a prompt's matmuls are a decode step's at the same rows.
            fires.push((1, 1));
            fires.extend(
                inside(shape.rows, &[2, 4, 8])
                    .into_iter()
                    .chain(tails.iter().copied().filter(|&rows| rows >= shape.rows.lo && rows <= shape.rows.hi))
                    .filter(|&rows| rows > 1)
                    .map(|rows| (rows, 1)),
            );
            // The readout tiers: a prompt of as many rows reads out every
            // one of them, the lm_head at that row count.
            for reads in inside(shape.readout_rows, &[2, 4, 8]) {
                if reads > 1 {
                    fires.push((reads, reads));
                }
            }
            for (rows, reads) in fires {
                let rows = rows.min(top);
                let reads = reads.min(rows);
                for &form in forms.iter().filter(|(_, scores)| !scores || octaves.contains(&rung)) {
                    for verb in verbs_of(shape.treatment) {
                        let key = (rung, rows, reads, form, verb);
                        if fired.insert(key) {
                            plan.push(key);
                        }
                    }
                }
            }
        }
        // The rows one fire reads logits for at most: a tier past it is no
        // step's.
        let readouts = u32::try_from(poem_compiler::arena::readouts_ceiling(&self.budgets.tokens))
            .unwrap_or(u32::MAX)
            .max(1);
        let _ = kernels_metal::linear::gemm::record_shapes(true);
        let walked = (|| -> Result<u32> {
            let mut fires = 0u32;
            for (rung, rows, reads, form, verb) in plan {
                if self.rung != rung {
                    self.set_rung(self.rung, rung)?;
                }
                let (verb, held) = (&verbs[verb].0, verbs[verb].1);
                let rows = rows.min(context.saturating_sub(held)).max(1);
                let reads = reads.min(rows).min(readouts).max(1);
                let tokens = vec![0u32; rows as usize];
                let tail: Vec<u32> = (rows - reads..rows).collect();
                let lane = [Seated {
                    replay_stated: None,
                    held: Some(held),
                    rs: verb,
                    refs: true,
                    pages: &kv_pages,
                    record: (!record.is_empty()).then_some(&record[..]),
                    windows: &windows,
                    readout: (reads > 1).then_some(&tail[..]),
                    ..Seated::of(Lane {
                        slot: 0,
                        word: word(rows, form),
                        tokens: &tokens,
                    })
                }];
                // Each dropped fire's command buffer goes back with its own
                // pool: left to the thread's, a walk of more fires than the
                // queue holds command buffers blocks making the next one.
                pooled(|| -> Result<()> {
                    let prepared = FrameShell::prepare(
                        self,
                        StepView {
                            lanes: &lane,
                            attachments: &[],
                            media: &[],
                            clips: &[],
                            done: None,
                        },
                        None,
                    )?;
                    drop(self.walk(&prepared, Form::DryShared)?);
                    // A fire the Neural Engine shares runs on the GPU alone
                    // once the split is retired: both forms are built.
                    if self.ane.as_ref().is_some_and(|ane| ane.shares(rows)) {
                        drop(self.walk(&prepared, Form::DryAlone)?);
                    }
                    Ok(())
                })?;
                fires += 1;
            }
            Ok(fires)
        })();
        self.warming = false;
        let shapes = kernels_metal::linear::gemm::record_shapes(false);
        let fires = walked?;
        // What the walk backed goes back, and the rung stands where it
        // stood.
        if self.rung != standing {
            self.set_rung(self.rung, standing)?;
        }
        // Every dense matmul kernel a fire's rows can reach is built: each
        // shape the walk dispatched, at every row count up to the most it
        // dispatched it at, routes (gemm::route under this load's tuning)
        // to a kernel the walk built, or the load is refused by name here,
        // never a fire while serving.
        let built: std::collections::HashSet<String> = self
            .pipelines
            .keys()
            .into_iter()
            .map(|(_, entry, ..)| entry.to_string())
            .collect();
        let tuning = &self.tuning;
        let mut unbuilt: Vec<String> = Vec::new();
        for &(columns, contraction, f32_out, most) in &shapes {
            for rows in 1..=most {
                let entry =
                    kernels_metal::linear::gemm::route(tuning, rows, columns, contraction, f32_out);
                if !built.contains(&entry) {
                    unbuilt.push(format!("{entry} at {rows} x {columns} x {contraction}"));
                }
            }
        }
        tracing::info!(
            shapes = shapes.len(),
            unbuilt = unbuilt.len(),
            "metal: the walk's matmul shapes route only to kernels it built"
        );
        if let Some(first) = unbuilt.first() {
            return Err(Fault::Param {
                name: "kernel.dense".to_string(),
                why: Box::leak(
                    format!(
                        "routes a matmul the load's walk did not build its kernel for: {first} \
                         ({} such row counts); a fire there would be refused",
                        unbuilt.len()
                    )
                    .into_boxed_str(),
                ),
            });
        }
        Ok((fires, self.pipelines.compiled() - before))
    }
}
