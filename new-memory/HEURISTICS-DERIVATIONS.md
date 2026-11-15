# Derivations for the four numbers, and the inventory of the rest

Status: draft for Pie Leader's ruling, 2026-11-15. Nothing here is built.

Integrator's notes on the draft (GPU session):
- The MI300X figure section 3 cites (8.1 ms a step at rung 1024 against 5.0 ms) was withdrawn after this study began: the measuring script reused a cached prefill. On a fresh serve with distinct prompts a standing wide rung costs nothing measurable in steady decode on CUDA/ROCm; what remains is about 100 ms once per wave after a large prefill fire, cause not yet found. Section 3's rule does not rest on that figure; the open item "the cause of the MI300X figure" becomes "the 100 ms after a large prefill fire".
- Added since: the Vulkan/wgpu fire-fit floor is now derived (1 row for text; the patch ladder's least image otherwise), with the package-stated widest image as a follow-up that takes `DERIVED_PATCH_CEILING = 4096` (G2) and `PATCH_LATTICE_FLOOR = 64` with it; `SHAPES_KEPT = 64` (G8) and `BIND_CACHE_CAP` (G9) go with the recordings ruling (one recording per (arm, rung) made at load); `SPLIT_STORAGE_BUFFERS` is already gone.

Read-only study. Every `file:line` is on `origin/memory-redesign` (cec91bb4b) unless a ref is named. Other refs: `0bf0ef867` (local `sc/np-gpu-fire-fit`), `lends` (local `sc/np-rung-lends`, 9b3a3d30f), `unfolded` (`origin/sc/np-unfolded-contract`). Paths are under `crates/` unless they start with `new-memory/`. ROCm is served by `engine-cuda` (profile `engine-cuda/profiles/gfx94.toml`), so "CUDA" below covers ROCm. Nothing was built or run; figures that need a run are marked "not computed".

Symbols used throughout: `C` declared context; `r` a rung in tokens; `chunk` the pool's chunk bytes; `room` what the device grants the pool side after the device share; `floor` what stands on the device whatever the rung (dense weights, token tables and expert slabs resident, bodies); `S1(C)` one working set at the declared context, in chunks; `span(r, l)` the activation span of rung `r` carved for `l` lanes, in chunks.

## 1. `engine.working_set_percent` (default 50)

**(i) Where it is and what it decides.**
- Defined `engine-cuda/src/chip.rs:68-73` as a chip-profile decision (`constant(range(10..=90)), default = 50, tune = engine`); registered `:177`, field `:280`, copied `:314`, default asserted in a test `:795`. It is not one of the seven operator keys (`worker/src/config.rs:1277-1295`).
- Profile overrides: `engine-cuda/profiles/sm120/nvidia-geforce-rtx-5090.toml:65` = 75; `sm86/nvidia-a40.toml:67` = 65; `sm89/nvidia-geforce-rtx-4090.toml:63` = 65; `sm89/nvidia-l4.toml:46` = 65; `sm90/nvidia-h100-80gb-hbm3.toml:60` = 90 ("1.9% faster end to end"); `sm90/nvidia-h200.toml:56` = 70. `gfx94.toml` does not set it (50).
- Read 1, first fit: `engine-cuda/src/serve/load.rs:311` `share = after_weights / 100 * working_set_percent`, then `tokens_within(asked_tokens, floor, share, working_at)` `:312` (`engine-cuda/src/store.rs:156-162`, halving), then the widest lattice bucket under the share `:315-322`, then the voxel ladder's room `share.min(after_weights - cache)` `:323-328`. `floor = max_lanes.max(TOKENS_FLOOR).min(asked_tokens)` `:283`, `TOKENS_FLOOR = 1024` `:25`.
- Read 2, refit: `load.rs:674` caps `room_at` at `live / 100 * working_set_percent` when `owed`. `room_at` itself is `live - least_sequence_bytes - BODIES_FLOOR_BYTES` `:667-677`. The refit tries the asked budget `:705`, then `fit(true, false)` (with the percent), then `fit(false, true)` (no percent, decoded tiles scoped to one sequence) `:710-712`.
- It decides the widest fire (tokens, then lanes) a CUDA/ROCm load carves. The doc states the intent: "the rest is owed to the cache rows, bodies and guests" (`chip.rs:68-70`).
- Same heuristic elsewhere: Vulkan/wgpu on the tip take a bare half, `engine-gpu/src/serve.rs:711` `room = device.left().saturating_sub(weights) / 2`; on `0bf0ef867` `engine-gpu/src/tuning.rs:36` copies the percent and `fit.rs` uses `tuning.working_share(after_weights)`. Metal has no percent: the top rung is the worker's `METAL_FIRE` and the load refuses (`engine-metal/src/serve.rs:1129-1131`).
- Docs: `new-memory/REDESIGN.md:110-116` (2.5) already states the replacement and lists `working_set_percent`, `tokens_within`, `TOKENS_FLOOR` and "the lane/token refit" as removed; `new-memory/DEVICE-MEMORY-INVENTORY.md:109` still lists the key as live.

**(ii) What replaces it.** The proposal holds, with the span stated in full and one consequence named.
- "The state of one sequence at full context", in chunks, is already computed from load-known values on both sides. Runtime: `runtime/src/store/shared.rs:1107-1128` `working_set_chunks(context)` = `ceil(ceil(C / T_list) / spc_list)` list chunks + for each window list `ceil(min(ceil(C / T_w), ceil(w / T_w) + 1) / spc_w)` + one chunk where an rs list exists + `record_chunks(1)` (`:1076`); stored as `least_chunks` at `:918`. Engines state the same quantity: CUDA `chunks_at(layout, &paging, one_slot, slots)` (`engine-cuda/src/store.rs:1239-1241`), Metal `Layout::least_chunks(...) + window_chunks` (`engine-metal/src/serve.rs:1124-1125`), Vulkan/wgpu `least_chunks` (`engine-gpu/src/serve.rs:822-842`). All inputs are `ChunkFacts` shapes and the declared context.
- The check exists: `engine/src/chunk.rs:807-831` `fit(room, chunk_bytes, act_chunks, least_chunks)`. The rule solves it for the rung: the top rung is the widest power of two `r` with `floor + S1(C) * chunk + span(r) * chunk <= room`.
- `span(r)` must be every byte a fire of rung `r` takes from the pool side: the compiled arena carve, the attention workspaces, the decoded-weight tiles, the `lanes * tokens * 8` term (CUDA `working_at`, `load.rs:635-664`) **and the rs replay scratch of rung `r`** (CUDA reach: `RsScratch::need(max_rows * 3 + min(max_lanes, slots) * 2 * REPLAYED, ext_row_bytes)`, `engine-cuda/src/store/rs.rs:121-125`; `need = (rows.max(1) + 64) * ext_row_bytes + 4 MiB`, `engine-cuda/src/run.rs:216-218`). Metal's `act_chunks` per rung already includes it (`engine-metal/src/serve.rs:1056-1098`).
- What else is left beside the one sequence: only `floor`. Weights are dense on the device and are inside `floor`. On CUDA `floor` also holds `BODIES_FLOOR_BYTES = 512 MiB` (`engine-cuda/src/store.rs:126`), a bare number that stands in for the bodies until they are weighed (inventory, E3). Not verified: whether the streamed parts (experts, token tables) need a minimum device-side cache beside the one sequence; the fits on the tip subtract none beyond what `room` already excludes.
- Consequence. The percent does two jobs today: it bounds the top rung, and it guarantees the state side `(100 - p)%` while the widest rung stands. The rule keeps only the first. The second moves to the runtime: the rung at rest is what the standing work asks (section 3) and the span above it is lent to state; `widest_fit`/`fit_cap` (`runtime/src/store/shared/programs.rs:112-122`, `:162-164`) already cut fires to the widest rung that fits. While the top rung stands, exactly one full-context sequence is guaranteed room; everything else was evacuated by the raise.

**(iii) What the change touches.**
- `engine-cuda`: delete the decision (`chip.rs:68-73`, `:177`, `:280`, `:314`, `:795`) and six profile lines (list above). `serve/load.rs`: first fit `:283-327` and refit `:580-745` become one solve against `room_at(tokens, false)`; `working_at` (`:635-664`) gains the rs reach; `TOKENS_FLOOR` (`:25`) becomes the smallest rung; `fit_the_card` (`store.rs:1236-1300`) refuses only when the smallest rung fails. `store.rs:156` `tokens_within` loses its callers or stays as the halving step.
- `engine-metal`: `serve.rs:1091-1131` drops rungs from the top while `chunk::fit` fails, instead of `Fault::Residency` (in hand on the Mac side).
- `engine-gpu`: `serve.rs:711` `/ 2` goes; on `0bf0ef867`, `tuning.rs:36` and `fit.rs` `working_share` go and the probed span prices the rs scratch buffer.
- `engine`: `chunk::fit` unchanged. `runtime`, `worker`: nothing (the key never reaches them). Docs: `new-memory/REDESIGN.md:186-221` table row, `new-memory/DEVICE-MEMORY-INVENTORY.md:109`, `new-memory/BACKENDS.md`. Tests: `chip.rs:795`; the refit/first-fit tests in `engine-cuda` were not enumerated.
- User-visible: the fitted fire no longer differs by profile percent. Cards at 50-75% get a wider top rung where memory allows; a hybrid kv+rs model whose rs scratch is large gets a narrower top rung instead of a refusal; a load is refused only when the smallest rung plus one sequence does not fit.

**(iv) What could go wrong; what to measure before merge.**
- The profile values were set by measurement (H100: 90, "1.9% faster"). The solved rung must be at least as wide as the percent's rung on every profiled card; compare the fitted `(tokens, lanes)` per card and model before and after.
- A raise to a top rung that leaves one sequence evacuates all other device state. Measure a prefill wave arriving on a full cache (evacuation cost against the wider fire's gain) on CUDA, ROCm and Metal.
- The first fit runs before bodies are weighed; until E3 is replaced the 512 MiB floor decides the last rung on small cards.
- Fire width is also a latency trade, not only a memory one: `gfx94.toml:65-71` records that 8192 rows "left the first prompts' decodes waiting out a second prefill fire, 12K and up made queued prompts wait". The memory rule bounds the ladder; it does not say how wide a fire the scheduler should take (open question 1).

### Halving when rs scratch dominates (facts)
- CUDA/ROCm: the load-time halving does **not** see the rs replay scratch. The first fit's `working` (`load.rs:285-308`) and the refit's `working_at` (`:635-664`) price the arena, the attention workspaces, the decoded tiles and `lanes * tokens * 8`. The rs scratch reach is computed afterwards by `Buffers::reserve` (`load.rs:1049-1055`; `store/rs.rs:121-125`) and laid into the activation span by `lay_reach` (`load.rs:1485`, `:1517`; `store/rs.rs:487-493`, `pools.widen_activation(reach)`, `store/chunks.rs:353-378`).
- Minimum state the fit insists on: one sequence at the declared context (`least_sequence_bytes`, `store.rs:354-359`) plus the 512 MiB bodies floor (`load.rs:667-677`).
- "Fits the fire, zero state chunks" is **refused, not halved further**: after the span is widened, `fit_the_card` runs `engine::chunk::fit(limit, chunk, 0, least)` (`store.rs:1277`) and returns "the card does not hold this deployment ... one working set at the declared context needs ..." (`store.rs:1289-1296`). There is no path back into the halving.
- Vulkan/wgpu, tip: no token halving; the rs scratch is a separate buffer allocated per fire (`engine-gpu/src/serve.rs:3172-3192`, owner `books.rs:73`), outside the span the pool fit sees.
- Vulkan/wgpu, `0bf0ef867`: `engine-gpu/src/fit.rs` halves the rung further when one working set does not stand beside the probed span (`Probe::Pool`, about `:395-447`), so "fits the fire, no state" is halved there. Its probed span does not price the rs scratch: grep for `rs_scratch|RS_SCRATCH|scratch_bytes` in its `fit.rs`, `arena.rs`, `scratch.rs` finds nothing.
- Metal: the rs scratch is in `act_chunks` per rung (`engine-metal/src/serve.rs:1056-1098`), but the load refuses instead of halving (`:1129-1131`).

**Rule.** The top rung is the widest power-of-two `r` for which `floor + S1(C) + span(r) <= room`, checked by `engine::chunk::fit`, where `S1(C)` is the runtime's `working_set_chunks(C)` and `span(r)` includes the arena carve, the workspaces, the decoded tiles, the per-lane tables and the rs replay scratch of rung `r`. A rung that fails is dropped and the next narrower one is tried; the load is refused only when the smallest rung fails. No percentage, no token floor, and the same solve on CUDA/ROCm, Metal and Vulkan/wgpu.

## 2. `FIRE_LANES = 512`

**(i) Every definition and use.**
- Tip: `worker/src/backend.rs:139` `CUDA_FIRE_LANES = 512`, used `:157` in `cuda_budgets` (`:141-172`; `max_tokens: 0`, `slots: 0`). Per-backend `Fire { lanes, tokens }` (`:57-60`): `METAL_FIRE` 512/10240 `:80-83` (used `:580-584`), `VULKAN_FIRE` 512/10240 `:88-91` (`:622-626`), `WGPU_FIRE` 512/10240 `:93-96` (`:664-668`), `XLA_FIRE` 64/2608 `:98-101` (`:709-713`), `CEREBRAS_FIRE` 4/64 `:103-106` (`:775-779`). On `0bf0ef867`: one `FIRE_LANES: u32 = 512` at `worker/src/backend.rs:135` shared by cuda, vulkan and wgpu.
- What the asked lanes size on CUDA: state seats, a seat a lane (`engine-cuda/src/serve/load.rs:123`; `paging...seated(boot.budget.max_lanes)` `:921-927`); vocab-wide readout rows, `readouts_ceiling = min(max_lanes * readouts_per_lane, max_tokens)` (`poem-compiler/src/arena.rs:165-177`); lane tables at `2 * max_lanes` (`engine-cuda/src/window.rs:130`; `inputs.rs:115-138`); `lanes * tokens * 8` (`load.rs:659-663`); rs replay rows `min(max_lanes, slots) * 2 * REPLAYED` (`store/rs.rs:121-125`). Runtime: `pool.set_max_lanes(max_forward_requests)` (`runtime/src/bootstrap.rs:426`), `fire_chunks` uses `lanes = tokens.min(max_lanes)` (`runtime/src/store/slots.rs:690-697`), one COW set a lane (`slots.rs:61`). Admission: `runtime.max_concurrent_processes`, when omitted, derives from `max_forward_requests` (`worker/src/config.rs:767-769`).
- CUDA has a second cap: chip key `engine.fire_lanes`, default 256 (`engine-cuda/src/chip.rs:139-147`: decode bodies past 256 lanes armed cost "some 290 ns a node, 3.6% of its decode" on an L40S). `Shell::fire_lanes()` = `min(budget.max_lanes, cap)` (`engine-cuda/src/serve.rs:1043-1048`) is what `FireLimits.max_lanes` reports (`api.rs:774-786`). No profile under `engine-cuda/profiles` sets it. So on CUDA/ROCm 512 sizes the carve and the seats, and 256 is the widest fire.
- The refit already cuts lanes: `budget.max_lanes = lanes.min(tokens)` (`load.rs:599`); in `fit` (`:680-703`) tokens halve to the floor at the asked lanes, lanes then halve at that token count (`:685-688`), tokens then double back at the cut lanes (`:693-700`). The comment at `:689-692` records why tokens go back up (MiniMax-M3.1-Flash on an H100: pf64-512 7.05k to 10.35k tok/s).

**(ii) What replaces it.** Lanes are fitted in the same solve as tokens, lanes first on the decode diagonal, tokens second.
- Structural bounds found: `lanes <= tokens` of the rung (`load.rs:599`; `engine-gpu/src/arena.rs:88`; `slots.rs:690-697`; `new-memory/REDESIGN.md:97-108`), and one seat a lane. No narrower index width for lanes was found (grep for `u8`/`u16` lane fields and casts in `engine/src`, `runtime/src/{pipeline,scheduler,store}`, the CUDA/gpu/Metal serve paths returned nothing).
- Step 1, lanes. `L` is the widest power of two `l` for which the decode rung that carries `l` lanes stands beside its lanes' least state and one full sequence: `floor + S1(C) + l * s_lane + span(l, l) <= room`. `s_lane` is the least state a lane in a fire holds: its record slot, one COW set (`slots.rs:61`) and one list page, in chunks (the per-lane fire term, `slots.rs:115-127`).
- Step 2, tokens. With `L` fixed, the top rung is section 1's solve with `span(r) = span(r, min(r, L))`.
- Step 3, order of cutting. Lanes are never cut to widen tokens; tokens are never cut below the rung that carries `L`. This is today's refit order with `TOKENS_FLOOR = 1024` replaced by "the rung that carries the lanes", and it reproduces the measured MiniMax case (`:689-692`).
- Natural upper bound on lanes: `min(top rung, seats at s_lane)`. On a large card that is in the thousands, far past 512; see (iv).
- `span(r, l)` on CUDA, by term: `A(r, l)` = `compile_axes(...).arena.bytes`, containing the readout rows `min(l * readouts_per_lane, r)` each `width * elem` bytes when wider than 64 KiB (`poem-compiler/src/arena.rs:162-177`); `W(r, l)` = `attention_workspace_bytes` (`inputs.rs:115-138`); `D` = decoded tiles (`store.rs:164-177`); `8 * l * r` (`load.rs:659-663`); `R(r, l)` = rs reach above, zero for a model with no recurrent record.
- Qwen3-8B, CUDA sizes. `room = live - S1(C) - 512 MiB` (`load.rs:667-677`); the fit is `A + W + D + 8*l*r + R <= room`.

| term | where computed | 32 GB card | 48 GB card |
|---|---|---|---|
| `live` (card less weights and safety floor) | `load.rs:667-677`, `device/elastic.rs:28-44` | not computed | not computed |
| `S1(C)` bytes | `store.rs:295`, `:354-359` | not computed | not computed |
| `A(r, l)`, `W(r, l)`, `D` | `compile_axes`, `inputs.rs:115-138`, `store.rs:164-177` | not computed | not computed |
| `8 * l * r` at `l = 512, r = 8192` | `load.rs:659-663` | 32 MiB | 32 MiB |
| `8 * l * r` at `l = r = 8192` | same | 512 MiB | 512 MiB |
| per-lane bytes `dA/dl + dW/dl + 8r` | needs a `compile_axes` run at two lane counts | not computed | not computed |
| fitted `(r, L)` | the solve above | not computed | not computed |

  The per-lane figure cannot be read from the code: the arena's bytes are the compiler's output for the traced model. `new-memory/` holds no per-lane figure.

**(iii) What the change touches.**
- `worker/src/backend.rs`: `CUDA_FIRE_LANES` and the `lanes` field of the `Fire` constants go (`:57-106`, `:139`, `:157`, uses listed above); the engines are asked for "as many as fit".
- `engine-cuda`: `serve/load.rs` refit (`:580-745`) restated as steps 1-2; `state_slots` (`:123`) follows `L`; `engine.fire_lanes` (`chip.rs:139-147`, `serve.rs:1043-1048`) is a second lane cap and must be decided with this one (open question 2).
- `engine-metal` (`serve.rs:1056-1098`, lanes = `min(tokens, max_lanes)`) and `engine-gpu` (`fit.rs` on `0bf0ef867`: `max_lanes: asked_lanes.min(tokens)`, about `:415-420`) take the same two steps.
- `runtime`: no constant to change; `max_forward_requests`, `set_max_lanes` (`bootstrap.rs:426`) and the admission default (`worker/src/config.rs:767-769`) follow the fitted `L`.
- User-visible: the admission default changes from 256 (CUDA: `min(512, 256)`) or 512 (Metal, Vulkan, wgpu) to the fitted `L`; `FireLimits.max_lanes` and `max_readouts` change with it.

**(iv) What could go wrong; what to measure.**
- Armed bodies. `chip.rs:139-147` measured a launch cost for decode bodies past 256 lanes. With `L` in the thousands that cost returns unless bodies are armed by live lanes. Not verified: how the merged "live arming" bounds armed bodies. Measure c1 decode with the fitted `L` against main on an L40S.
- Admission. More lanes admit more processes; the state they need is then a PLEX/governor matter. Measure a c1024 run on a 48 GB card against main.
- Top-rung width. Readout rows at `L` lanes are carved in every rung above `L`; a large `L` narrows the top rung. Compare fitted `r` at `L` against `r` at 512.
- XLA (64/2608) keeps one pool for all shapes and Cerebras (4/64) is out of scope for this PR; their `Fire` values are not derived by this rule.

**Rule.** Lanes are the widest power-of-two `L` for which the decode rung carrying `L` lanes fits beside one full-context sequence and its lanes' least state (`floor + S1(C) + L * s_lane + span(L, L) <= room`); the top rung is then solved at `min(r, L)` lanes. Lanes are fitted first and never cut to widen tokens; tokens are never cut below the rung that carries the lanes. The only bounds are structural: lanes at most the rung's tokens, and one seat a lane.

## 3. `PATIENCE = 2` (idle lowering of the rung)

**(i) Where it is and what it decides.**
- One constant serves two things. `runtime/src/governor.rs:132` `pub const PATIENCE: u32 = 2` ("Congested ticks in a row before a backoff acts"). The governor's own use is `:381` (`congested_ticks >= PATIENCE`); that is the governor's fixed patience and is listed in the inventory (K4). The rung walk imports it (`runtime/src/store/shared/target.rs:12`) and uses it at `:140`.
- Rung walk: fields `ticks`, `idle_since` (`target.rs:63-64`); the `past`/`forced`/`waited` block `:132-145` ("it comes down only once it has for the governor's patience, so a pause between batches moves no rung"); `Lower if !waited => Wait` and the `Cap` arm `:165-171`; `ticked` increments and wakes `:520-522`. `TICK` is 250 ms (`governor.rs:475`), so a rung comes down 500 ms after its ask left.
- Tests: `target.rs:552-575` (`the_rung_comes_down_only_after_the_governor_s_patience`), `runtime/src/store/shared/tests/span.rs:12`, `:520`, `:552`; on `unfolded`, `runtime/src/pipeline/fire.rs:5642`, `:5664`.

**(ii) What replaces it.**
- The goal already falls by event. `walk_rows()` = `wanted_rows(engine).max(live_lanes())` (`target.rs:111-113`). `wanted_rows` falls when a pass's `RungWant` drops (`runtime/src/store/shared.rs:206-212`), when a process's hold is removed (`hold_for_process(.., None)`, `:223-233`, called `pipeline/fire.rs:3171-3175`), when a process ends (`forget_process`, `shared.rs:236-238`) and when the scheduler's `BatchWant` narrows or releases (`scheduler/worker.rs:1235`, `:1295`); `live_lanes` falls with the live count (`target.rs:65-70`).
- What is missing on the tip is the wake: `Drop for RungWant` removes the ask and notifies nobody (`shared.rs:206-212`); `grant_rung` stores without notifying (`:164-168`). The walker is woken only by freed room (`target.rs:296`), retires (`:307`, `:327`), a retarget (`:503`) and the tick (`:520-522`). The tick is the only thing that re-reads the goal after an ask leaves.
- `lends` adds the other half: `grant_rung` notifies when the grant changes, `grant_watch`, `RUNG_HOLDS` apart from `RUNG_WANTS`, `goal_rows -> (rows, short)` and `forced = fire_cap > fit_cap || short`; a kv waiter short of pages lowers at once. It keeps `PATIENCE` on the non-short path (`lends` `target.rs:178`).
- The event that replaces the tick: **the ask leaving**. `RungWant::drop`, `hold_for_process(None)`, `forget_process` and `BatchWant::ask/release` wake the walker when they lower `wanted_rows`; the walker lowers to the rung `walk_rows()` names in that turn. The wave's hold stands across the passes of one prompt (`fire.rs:3171-3175`, held while `rows_total > RUNG_WANT_FROM`), so "its prompts have landed" is the hold being removed.
- Idle lowering is not needed as a timed thing. What the patience protected was a re-raise after a pause. A re-raise costs nothing when state has not taken the lent chunks: `raise_now` raises at once "where nothing lies in the larger span and the room is there" (`shared.rs:283-320`). It costs an evacuation only when state did take them, which is the case where the lower was needed.
- What a standing wide rung costs when nobody needs its chunks. Memory: nothing on CUDA/ROCm (`span_backed_on_demand` false, `engine-cuda/src/store/chunks.rs:844-871`; the carve counts as committed, `runtime/src/store/slots.rs:656-672`); on a span backed on demand the held carve is resident memory (`target.rs:132-145`). Work sized by the standing rung or `fire_cap`, by reading:
  - the scheduler's step token limit is `min(max_forward_tokens, cap)` (`scheduler/worker.rs:1236-1238`) and a step is full only at `forward_tokens >= limits.max_forward_tokens` (`scheduler/batch.rs:173-176`), so under a 1024 cap a 64-lane decode step is never full; the frame policy holds for joins (`JOIN_HOLD_US = 2000`, `GATHER_POLL_US = 500`, `scheduler/frame.rs:59-60`, `:1227-1241`). `FramePolicy::new` takes the unstepped limit (`worker.rs:1173-1178`). Not traced: whether a not-full step is held by this path.
  - a pass is cut at `fire_cap` (`pipeline/fire.rs:3135`) and `prefill_share` is capped by it (`:695-709`); span funding reads it (`worker.rs:1008`).
  - Metal only: `set_rung` covers, drains and re-hosts regions for the standing rung (`engine-metal/src/serve.rs:1568-`, `api.rs:668`). CUDA/ROCm have no `set_rung` (default no-op, `engine/src/engine.rs:245-254`); a fire picks its carve by its own size (`engine-cuda/src/arena.rs:17-29`, `serve/enqueue.rs:258-261`).
- MI300X (8.1 ms/step at rung 1024 against 5.0 ms at rung 64-128): the cause is not identified here. On CUDA/ROCm nothing engine-side was found that scales with the standing rung; the candidates are the runtime items above, the step limit first. Under the rule the question does not arise at rest: 64 decoding lanes ask a 64-128 rung.

**(iii) What the change touches.**
- `runtime/src/store/shared.rs`: a wake in `Drop for RungWant` (`:206-212`), in `hold_for_process` when it removes or narrows (`:223-233`), in `forget_process` (`:236-238`); `grant_rung` notifies (from `lends`).
- `runtime/src/store/shared/target.rs`: delete `ticks`, `idle_since` (`:63-64`), the `waited` computation (`:132-145`), the `!waited` arms (`:165-171`), `ticked` (`:520-522`) and its caller, and the `PATIENCE` import (`:12`). Tests at `target.rs:552-575` and `tests/span.rs:520`, `:552` are rewritten to "the rung comes down when the ask leaves".
- `runtime/src/governor.rs:132` stays for `:381` only. No config surface. No engine change. Docs: the walk's description in `new-memory/REDESIGN.md`.
- User-visible: after a prefill wave the rung returns to the decoding lanes' rung in the same turn instead of 500 ms later; state gets the lent chunks sooner.

**(iv) What could go wrong; what to measure.**
- Rung changes per second under alternating prefill and decode. On Metal each change is a `set_rung` with a drain; measure pf64 + c64 mixed load on a Mac against main.
- Evacuation churn when state takes the lent chunks between waves; measure a repeated wave on a full cache on CUDA and ROCm.
- Not verified: whether the slot allocator takes the lent (top) chunks last. If it does not, state lands in the span after every lower and each wave pays an evacuation.
- The MI300X figure must be re-measured with the rung at rest by the ask (expected 64-128); the 8.1 ms cause still needs its own answer for the time a wide rung does stand.

**Rule.** The rung at rest is the rung `walk_rows()` names: the widest standing ask or the live decoding lanes. A wave's ask raises it; when the ask leaves (the pass's `RungWant` drops, the process's hold is removed, the process ends, or the scheduler's batch ask narrows) the walker is woken and lowers to what remains in that turn. There is no timer in the rung walk; `PATIENCE` stays only as the governor's debounce.

## 4. `MAX_WINDOW_LISTS = 4`

**(i) What it is and what it bounds.**
- A window list is a `SlotKind::List { tokens, rate, window: Some(w) }` shape in `ChunkFacts`, one per distinct window (`new-memory/CONTRACT-BC.md` items 19, 21). The declaration reaches the runtime in the pool build, `runtime/src/store/shared.rs:793-811`; the lists are sorted by window `:855`; more than four are refused `:856-861` (`SharedError::Shapes { kind: "window list" }`); duplicate windows are refused after `:862`.
- Defined `runtime/src/store/rs.rs:46`. Arrays sized by it: `WindowState.retired: [u32; MAX_WINDOW_LISTS]` (`:60-66`, serde), `RsUnits.windows: [u32; MAX_WINDOW_LISTS]` (`:113-118`, `Copy`), `RsUnits::of_windows` (`:147`), `window_need` (`:1300-1301`, `need[list] += 1`), `per_list` (`:1926`), `host_units` (`:2061`), `alloc_windows(need: [u32; MAX_WINDOW_LISTS])` (`:2308`), and `rs_growth` in `runtime/src/planner/pool.rs:233`. The store already holds `window: Vec<WindowGeometry>` (`rs.rs:540`) and its loops take `self.window.len()`.
- Past 4: the load is refused at `shared.rs:856-861`. Nothing indexes past the array at run time.
- The same 4 is restated in three engines: `engine-cuda/src/store/layout.rs:28` `MOST_WINDOWS = 4` (`:46`, `:139` `COUNT = 3 + MOST_WINDOWS`, refusal `:944-947`, `api.rs:1484`; ids past the first offset by `MORE_WINDOWS = 1 << 16`, `:30`); `engine-metal/src/store/layout.rs:51` `MAX_WINDOWS = 4` with fixed ids `[WINDOW, ShapeId(4), ShapeId(5), ShapeId(6)]` `:55` (`:133`, `:183`, `:231`, `store/pools.rs:272-274`, `:697`); `engine-gpu/src/store/layout.rs:61` `WINDOWS = 4`, ids 1, 4, 5, 6 (`:66-72`), below `WEIGHTS = 9` (`store/rows.rs:42`).
- `new-memory/CONTRACT-BC.md:97-100` says `ChunkFacts::validate` refuses more than 4 window lists. On the tip `engine/src/chunk.rs:545-580` checks empty kinds, an rs buffer with a window and duplicate windows; no count check was found there by grep.

**(ii) What replaces it.** The count is the number of window-list shapes the load declares; the runtime sizes from it at pool build.
- `WindowState.retired` becomes `Vec<u32>` (or a boxed slice) of `window_lists()` length; `RsUnits.windows` likewise, which removes `Copy` from `RsUnits`; `window_need`, `per_list`, `host_units`, `alloc_windows` and `rs_growth` take and return load-sized values. The refusal at `shared.rs:856-861` goes.
- Serialized formats. Snapshot manifest: `runtime/src/handle/snapshot.rs:141` `SavedState.window: Option<WindowState>`, serde_json, `VERSION = 1` (`:32`); `retired` is written as a 4-element array today and a `Vec<u32>` reads and writes the same JSON, so a reader that pads or truncates to the store's count keeps version 1. The snapshot refuses more than one window list anyway (`snapshot.rs:387`). `engine::fire` `Lane.tables: Vec<(ShapeId, Vec<u32>)>` (`engine/src/fire.rs:242`) is unbounded. No window-list count appears in WIT (grep of `inferlet/wit`: only `run-ahead-window`). `SavedCursor`: no fixed window array was found in it; not verified beyond grep.
- Engines: CUDA's `MOST_WINDOWS` becomes the declared count (its `MORE_WINDOWS` id offset already leaves room). Metal and Vulkan/wgpu use fixed shape ids 4-6 below the weights' ids; a fifth window list needs their id scheme changed.

**(iii) What the change touches.** `runtime`: `store/rs.rs` (sites above), `store/shared.rs:856-861`, `planner/pool.rs:233`, `handle/snapshot.rs:141`; `RsUnits` appears 226 times in 29 files and 29 non-test sites touch `.windows`, each a `Copy` use to revisit. `engine-cuda/src/store/layout.rs:28-46`, `:139`, `:944-947`; `engine-metal/src/store/layout.rs:51-55`; `engine-gpu/src/store/layout.rs:61-75`. Docs: `new-memory/CONTRACT-BC.md:97-100`. Tests: the refusal tests for a fifth list in the runtime and each engine layout (not enumerated). No config surface. User-visible: a load with more than four distinct windows is served where the engine's ids allow it.

**(iv) What could go wrong; what to measure.** `RsUnits` is copied on the fire path; a heap-backed `windows` adds an allocation per copy unless it is a small inline vector, so measure decode step time on a windowed model against main. Not verified: whether any kernel bounds the number of window tables a lane carries. Changing the runtime alone leaves three engine constants and the contract text saying 4.

**Rule.** The number of window lists is the number of `List { window: Some(_) }` shapes the load declares, read once at pool build; every per-list array in the runtime is that long. Four stays nowhere in the runtime. If the contract keeps a bound, it is defined once in `engine` and checked in `ChunkFacts::validate`, and the runtime and the three engine layouts read it from there (open question 4).

## Inventory

Known items, state on the tip (not re-derived):

| id | name, value | file:line | state |
|---|---|---|---|
| K1 | `DEVICE_SHARE = 0.9`; floors beside it | `engine/src/load.rs:60` (used `:87`); Metal `DRIVER_FLOOR = 64 MiB` `engine-metal/src/store/accounting.rs:5`; CUDA `SAFETY_FLOOR_BYTES = 128 MiB`, `min(128 MiB, total / 10)` `engine-cuda/src/device/elastic.rs:10`, `:28-35` | live; doc says provisional until the `outside` term is measured |
| K2 | `RUNG_WANT_FROM = 16`, `PASS_ASKS_FROM = 16` | `runtime/src/pipeline/fire.rs:731`; `runtime/src/store/shared.rs:242` | live, two constants; on `unfolded` one local `RUNG_WANT_FROM` (`fire.rs:822`) |
| K3 | `UNFOLDED_ROWS = 16` | `engine/src/fire.rs:590`; CUDA `REPLAYED` `engine-cuda/src/store/rs.rs:45` | live; gone from `engine/src/fire.rs` on `unfolded` |
| K4 | governor `PATIENCE = 2` | `runtime/src/governor.rs:132`, `:381` | live; also read by the rung walk (section 3) |

Runtime (`crates/runtime/src`), beyond the known ones. "Operator key" means one of the seven in `worker/src/config.rs:1277-1295`.

| id | name, value | file:line | decides | derivation |
|---|---|---|---|---|
| R1 | `FIRST_HOLD = 4`, `LONGEST_HOLD = 256`, hold x4 on push-back | `governor.rs:252-253`, `:397` | calm ticks before the budget probes up again | multiples of the tick with no source; not derived here |
| R2 | `SETTLED = 64` | `governor.rs:257` | when a hold counts as settled | same |
| R3 | `IDLE_AFTER = 120` (30 s) | `governor.rs:261` | when the idle target starts down | belongs to operator key `governor_release_idle`; not derived |
| R4 | `CUT_EVERY = 4`, `CUT_SHARE = 8` | `governor.rs:264-265` | pace and size of idle cuts | not derived |
| R5 | `KEEP_CACHE = 2400` (10 min) | `governor.rs:269` | when the idle cache is released | retention is PLEX's decision; a runtime timer here needs that ruling |
| R6 | `TICK = 250 ms` | `governor.rs:475` | governor period | bounded below by the cost of one sample; not measured |
| R7 | `MIN_EVICTED = 2000`, `THRASHING = 0.5` | `governor.rs:483-484` | refault ratio that counts as congestion | OS signal thresholds; not derived |
| R8 | `GROWTH_STEP = 512 MiB` | `governor/run.rs:17` | most one calm tick adds on unified memory | could be the largest single fire's bytes; not derived |
| R9 | `CONGESTED = 5.0`, `CRITICAL = 10.0` | `governor/psi.rs:11`, `:13` | PSI percentages that trigger backoff | OS signal thresholds; not derived |
| R10 | `COMMIT_AHEAD_BYTES = 64 MiB` | `store/shared.rs:73` (used `:1820`) | commit kept ahead of use | one rung step's chunks is the derived value |
| R11 | reclaim step `capacity / 64`, `elapsed / 4 ms` clamp 1..8 | `store/kv.rs:773-774` | pages freed per reclaim turn | the waiter's shortfall is the derived value |
| R12 | count freshness `max(1 s, took * 50)`; `RECLAIMABLE_FRESH = 1 s` | `store/kv.rs:1292`, `:2348`; `planner/pool.rs:146` | how long a reclaimable count stands | invalidate on change instead of by time |
| R13 | `SEATS_UNBOUNDED = 1 << 16` | `store/chunks.rs:525` | seats of a model without state slots | fitted lanes (section 2) |
| R14 | `PIECES_AHEAD = 2` | `pipeline/fire.rs:736` (used `:3259`) | pieces run before a pending raise lands | the frame dispatch depth it is already maxed with |
| R15 | `SEAT_WAIT = 10 s`, `FIRE_PROGRESS` | `pipeline/fire.rs:2188`; `planner/in_flight.rs:30` | when a waiter short of room fails | failure deadline; operator-visible, not derived |
| R16 | `WINDOW_GROWTH = 2` | `planner/pool.rs:8` | window slots a lane holds past its prompt | structural: the straddled slot and the next boundary's |
| R17 | `RESERVE_SHARE = 32` | `planner/inner.rs:19` (used `:128`, `:132`) | share of window lists held back for seated lanes' growth | `lanes * WINDOW_GROWTH` slots is the derived value |
| R18 | `HOST_RESERVE_DIVISOR = 8` | `planner/residency_planner.rs:22` (used `:373`, `:437`, `eviction.rs:207`) | host share held in reserve | not derived |
| R19 | to-end spread: fold at 3/4, "an eighth more lanes", `life * 2`, eight groups; env `PIE_TO_END_OVER` | `planner/to_end.rs:41`, `:79`, `:118-122`, `:172`, `:183`, `:191` | when seats are spread to the end | not derived |
| R20 | `DEFAULT_FRAME_SIZE = 2`, `DEFAULT_DISPATCH_DEPTH = 2` | `scheduler.rs:299`; `scheduler/frame.rs:10` | frames per dispatch and in flight | frames in flight is structural per REDESIGN section 3; `runtime.frame_size` is an operator key outside the seven |
| R21 | `JOIN_HOLD_US = 2000`, `GATHER_POLL_US = 500`, `Hold(500 us)`, `STRICT_WATCHDOG_US = 1 s` | `scheduler/frame.rs:57-60`, `:1066` | how long a frame waits for more lanes | a measured step time is the derived value; candidate in the MI300X figure |
| R22 | backstop 250 ms | `scheduler/worker.rs:1329` | re-poll when no wake arrives | covers a missed wake; not derived |
| R23 | `CLOSE_CHANNEL_BATCH_MAX = 512` | `scheduler/worker.rs:1916` | channel closes per turn | not derived |
| R24 | `LAUNCH_RUN_BEFORE_CONTROL = 2`, `CONTROL_RUN_MAX = 32`, `ENGINE_LANE_HOT_US` | `scheduler/lane.rs:211`, `:213`, `:333-335` | lane turn fairness, spin time | not derived |
| R25 | `GAP_BUDGET = 20 ms` | `scheduler/arming.rs:206` | lane time a gap gives a raise's steps | not derived |
| R26 | `AGAIN = 100 ms` | `policy/pressure.rs:233` | re-report period of a standing shortfall | not derived |
| R27 | `DEFAULT_DEADLINE_MS = 50`, `MAX_DEADLINE_MS = 1000`; policy `TICK = 1 ms`, `MEMORY_BYTES = 64 MiB`, `MAX_ENTRIES = 1 << 16` | `policy/manifest.rs:9-10`; `policy/program.rs:31-33` | PLEX program limits | sandbox limits; policy manifest surface |
| R28 | `FORK_WAIT = 250 ms`, `MAX_HANDLES = 4096`, `MAX_MEDIA_BYTES = 8 GiB` | `handle/mod.rs:25`, `:29`, `:33` | guest handle limits | guest quotas; not derived |
| R29 | snapshot `CHUNK_PAGES = 64`, `SYNCS = 8`; offload `CHUNK_PAGES = 32` | `handle/snapshot.rs:37-38`; `offload/prefill.rs:77` | I/O batching | not derived |
| R30 | `UNCAPPED_PREWARM_PROCESSES = 64`, `STAGED_COHORTS = 1`, `OUTPUT_BUFFER_CAP = 4096` | `inferlet/process.rs:226`, `:308`, `:708` | prewarm count, staging, output buffer | prewarm follows fitted lanes; rest not derived |
| R31 | `MAX_ROLLBACK_TOKENS = 64`, `TOKENIZER_OFFLOAD_THRESHOLD = 64` | `inferlet/host/grammar.rs:81`; `inferlet/host/tokenizer.rs:17` | grammar rollback depth, tokenizer offload | not derived |
| R32 | `LEASE_TTL = 60 s`, `UNTAKEN_TTL = 60 s`, `REPORT_SLICE = 1024`, `BLOB_GRACE = 60 s` | `offload/prefill.rs:660`; `offload/stream.rs:24`; `offload/directory.rs:58`; `inferlet/program/repository.rs:32` | offload and blob lifetimes | not derived |
| R33 | `DEFAULT_CAPACITY = 256`, `MAX_INFLIGHT_UPLOADS = 16`, `CORE_RESOURCES_PER_COMPONENT = 16`, `NODE_COPIES = 4` | `pipeline/program.rs:102`; `server.rs:230`; `bootstrap.rs:779`; `store/kv/page_table.rs:215` | cache capacity, upload and resource limits, page-table copies | not derived |

Judged structural or diagnostic in the runtime (not listed above): `FIXED_SETS_PER_WORKING_SET = 1`, `COW_SETS_PER_LANE = 1` (`store/slots.rs:56`, `:61`, documented "Derived, not configured"); word indices and record tags (`engine/completion.rs:435-437`, `engine/channel.rs:26-29`, `store/kv/journal.rs:22-23`); sentinels (`TICKET_NONE`, `RS_TRANSLATION_UNMAPPED`, `NONE`/`WORD`, `UNTIL_COPIED`, `FLOOR`, `PINNED`, `PATCH_ROUTE_DROP`/`PAD`); format versions (`handle/snapshot.rs:32`, `MAX_NAME_BYTES`); OS codes (`governor/macos.rs`); display widths (`inferlet/program/identity.rs`); say-after and dump timers that only log (`ADMIT_SAYS_AFTER`, `HOLD_SAYS_AFTER`, `STEP_SAYS_AFTER`, stall dumps `scheduler/worker.rs:1390`, `:1475-1476`, `RUNNER_DUMP_CAP`, `PIE_IDLE_DUMP_*`).

Engine-side numbers met while tracing the four (outside the runtime sweep, CUDA unless named):

| id | name, value | file:line | decides | derivation |
|---|---|---|---|---|
| E1 | `engine.fire_lanes = 256` | `engine-cuda/src/chip.rs:139-147` | widest fire in lanes | section 2 and open question 2 |
| E2 | `engine.forward_tokens` (0 = 8192; profiles 2304, 10240, 16384), worker `Fire.tokens` 10240 / 2608 / 64 | `chip.rs:111-116`; `profiles/sm89/nvidia-l40s.toml:42`, `nvidia-l4.toml:48`, `gfx94.toml:71`, `sm100/nvidia-b200.toml:28`, `sm90/nvidia-h200.toml:60`; `worker/src/backend.rs:80-106` | asked top rung | section 1 solve; open question 1 |
| E3 | `BODIES_FLOOR_BYTES = 512 MiB`, bodies allowance `room / 4` | `engine-cuda/src/store.rs:126`, `:135` | room held for bodies | bodies weighed per instantiation |
| E4 | `TOKENS_FLOOR = 1024` | `engine-cuda/src/serve/load.rs:25` | least fire the halving keeps | the smallest rung |
| E5 | `CUDA_PAGE_SIZE = 16`, `METAL_PAGE_SIZE = 32` | `worker/src/backend.rs:134`, `:86` | kv page tokens | not derived |
| E6 | `engine.walk_fire_rows`, `fire_rows`, `prefill_lanes`, `window_spare`, `seat_to_end`, `seat_spread`, `seated_first`, `lattice_steps` | `engine-cuda/src/chip.rs:108-147` | chip-tuned fire shape | REDESIGN 2.5 removes `walk_fire_rows` and `window_spare`; rest per-chip measured decisions |

Vulkan / wgpu (grep-only glance; `engine-gpu`, `engine-vulkan`, `engine-wgpu`):

| id | name, value | file:line | note |
|---|---|---|---|
| G1 | fire room `/ 2` | `engine-gpu/src/serve.rs:711` | the unnamed 50 percent; section 1 |
| G2 | `DEFAULT_STATE_SLOTS = 256`, `DERIVED_PATCH_CEILING = 4096` | `engine-gpu/src/api.rs:272`, `:288` | seats and patch bound |
| G3 | `READOUT_ROWS_PER_LANE = 8`, `PRICED_FROM = 16`, `PRICED_MOVES = 3`, `PRICED_FROM_MOST = 64` | `engine-gpu/src/serve.rs:3703`, `:4066`, `:4069`, `:4073` | readout carve, move pricing |
| G4 | `TARGET_CHUNK = 4 MiB`, `INSTANCE_SLOT = 4 KiB`, `CHANNEL_SLOT = 1 KiB`, `WINDOWS = 4` | `engine-gpu/src/store/layout.rs:56-79` | chunk size is structural per REDESIGN section 3; `WINDOWS` is section 4 |
| G5 | `RING = 2`, `MAX_ROUTES = 1 << 18`, `COPY_THREADS = 16`, `UPLOAD = 64 MiB`, `FILL_THREADS = 16`, `STAGE_BYTES = 32 MiB`, `GROUPS_PER_CORE = 4` | `engine-gpu/src/experts.rs:14-18`; `store/experts.rs:54`; `store/rows.rs:45`, `:48`; `store.rs:1588`; `guest/session.rs:15` | expert streaming and upload sizes |
| G6 | `UNREPORTED_CORES = 16` | `engine-vulkan/src/chip.rs:53`; `engine-wgpu/src/chip.rs:30` | core count when the device reports none |
| G7 | `STAGING_BYTES = 256 MiB`, `SPARE_FENCES = 32`, `FENCE_TIMEOUT_NS = 120 s`, `PIPELINE_CACHE_SETTLE = 2 s`, `TIMED_DISPATCHES = 8192`, `SETS_PER_POOL = 1024`, `SETS_PER_BATCH = 64`, `HAZARD_CAP = 96` | `engine-vulkan/src/device/ctx.rs:147`, `:157`, `:306`, `:450`, `:1238`, `:1281-1284`, `:1308` | device context sizes |
| G8 | `SHAPES_KEPT = 64`, `KEEP_BYTES = 64 KiB`, widen `SLOTS = 16` / `SLOT_BYTES = 256` | `engine-vulkan/src/record.rs:74`; `guest/run.rs:324`; `guest/widen.rs:23`, `:26` | recording and guest caches |
| G9 | `TAB_PREFILL_WORK = 1 << 19`, `UNIFORM_CHUNK = 1 MiB`, `SCRATCH_CHUNK = 8 MiB`, `TIMING_QUERIES = 4096`, `PREAD_CHUNK = 256 MiB`, `WAIT = 500 ms`, `BIND_CACHE_CAP = 16384`, `OUTBOX_FLOOR = 256` | `engine-wgpu/src/tuning.rs:202`; `device/ctx.rs:61-67`; `device/alloc.rs:16`; `device/host.rs:168`; `device/pipelines.rs:174-175`; `guest/session.rs:31` | wgpu device sizes |

Structural there: `ALIGN = 256` (`engine-gpu/src/scratch.rs:14`, `arena.rs:28`, `inputs.rs:11`, `weights.rs:24`), copy alignment and grain, SPIR-V opcodes, descriptor and binding counts, `WG = 256`, `NIL`/`NOT_HELD` sentinels.

## Open questions

1. **Upper bound on the top rung when memory is plentiful.** The memory rule alone picks the widest rung that leaves one sequence; the profiles record that width is also a latency trade (`gfx94.toml:65-71`) and two cards are set below the default (L4, L40S: 2304). Options: (a) memory only, as REDESIGN 2.5 says, and the scheduler decides how wide a fire to take from what is queued; (b) keep `engine.forward_tokens` as a per-chip measured cap on the ladder; (c) bound by a structural device limit where one exists (`dispatched_rung(device.max_workgroups())` on `0bf0ef867`, `engine-gpu/src/fit.rs:410-414`; none found on CUDA). Recommendation: (a) for the ladder, with (c) where structural; the latency trade is a scheduler ask, and the L4/L40S/MI300X values are re-measured before their keys are deleted.
2. **`engine.fire_lanes = 256` beside the fitted lanes.** Options: (a) delete it and bound armed bodies by live lanes; (b) keep it as a measured chip decision; (c) fit lanes and let the cap stand until arming by live lanes is verified. Recommendation: (c) for this PR, (a) as the end state; it rests on a measurement of launch cost at the fitted `L`.
3. **Per-rung lanes or one `L`.** Section 2 keeps one `L` and `lanes(r) = min(r, L)`, which matches `budget.max_lanes` and `fire_chunks`. Per-rung lanes (each rung carries what fits at its width) removes the lanes-against-tokens trade at the top rung but makes the scheduler read lanes per rung. Recommendation: one `L` now; revisit only if the fitted `L` measurably narrows the top rung.
4. **Is 4 window lists a contract bound or an accident of id numbering.** Options: (a) remove it everywhere, which means new shape ids on Metal and Vulkan/wgpu; (b) keep it as a contract bound defined once in `engine` and checked in `ChunkFacts::validate` (which does not check it today), with the runtime still load-sized. Recommendation: (b) plus load-sized runtime arrays in this PR; (a) when a model declares a fifth window.
5. **What `s_lane` is.** Section 2 uses the least state a lane in a fire holds (record, one COW set, one page). A lane at full context holds `S1(C)`; bounding lanes by that gives single digits on a 48 GB card for a long context. Recommendation: the least state, because sequence length is not known at load; admission and PLEX handle the rest.
6. **Governor timers (R1-R9).** REDESIGN section 3 keeps one fixed patience and two operator keys; nine further governor numbers have no stated source. Options: derive each from the tick and a measured quantity, or fold them under `governor_backoff` and `governor_release_idle`. Recommendation: list them in REDESIGN section 3 with their source or fold them, as a follow-up after the four numbers land.
7. **Not verified, each needs one reading or one run:** the cause of the MI300X figure; whether the slot allocator takes lent chunks last; whether a not-full step is held by the frame policy under a wide cap; the per-lane bytes for Qwen3-8B; whether streamed experts and token tables need a device-side minimum beside the one sequence; whether a kernel bounds window tables per lane.
