# North star on EVERY backend (tracking matrix)

Goal: the north star (SPEC.md) holds on every backend — CUDA, ROCm, Metal, Vulkan, wgpu, XLA/TPU, Cerebras. A feature
done on one backend is not done until it is on all of them (or the backend's spec realization says why not, e.g.
"one pool fixed at load" on XLA). No backend is "later, optional".

Legend: ✅ verified on hardware · 🟡 in progress / compile-only · 🔴 not started · — not applicable by spec

| Feature | CUDA | ROCm | Metal | Vulkan | wgpu | XLA/TPU | Cerebras |
|---|---|---|---|---|---|---|---|
| Real ChunkFacts, slot-ref lanes, fixed pieces | ✅ M34 | ✅ M43 (MI300X) | ✅ M30 · 🔴 dsv4 window by slot ref (BM5a) | 🟡 VK (kernels M43, pool next) | 🟡 VK | 🟡 XLA | 🟡 M43 host pool (no HW) |
| Grow / shrink (segments, evacuation) | ✅ M34 | — flat by spec | ✅ M30 | 🟡 VK | 🟡 VK | — fixed at load | 🟡 M43 (host segments) |
| rs list + slot_copies | ✅ M34 | 🟡 kernels, buffered goldens pending | ✅ M37 | 🟡 VK | 🟡 VK | 🟡 XLA | 🟡 M43 (host sim) |
| Window lists (own T_w, trimmed) | 🟡 one list, T_w=page (L40S: tables per list) | ✅ goldens (one list) | 🟡 one list (BM) | 🟡 VK | 🟡 VK | 🟡 XLA | 🟡 one list, T_w=page |
| Multi-piece records end to end | ✅ engine (L40S) · 🟡 runtime boot (IT r5) | 🟡 kernels, chunk golden pending | 🔴 on device (BM) | 🟡 VK | 🟡 VK | 🟡 XLA | 🟡 host sim |
| Weight streaming: PLE rows | ✅ M35/M36/M41 | ✅ M43 golden_rows | 🔴 BM | 🔴 VK | 🔴 VK | 🟡 XLA | — weights on host |
| Weight streaming: experts | 🟡 dense bf16 M40; quantized next | 🟡 kernels read tables; engine next | 🔴 BM | 🔴 VK | 🔴 VK | 🟡 XLA | — weights on host |
| Host/disk tiers of shared shapes (copy_slots) | ✅ engine probe M43 · 🟡 e2e | 🔴 AMD | 🟡 M43 compile (BM) | 🔴 VK | 🔴 VK | 🔴 XLA | 🟡 CB (host sim) |
| K2 (no kv spaces) | ✅ M39 | ✅ | ✅ M42 | compile | compile | compile | compile |

Cerebras realization (CB, `sc/cb-chunk-pool`): the column is the host-traced `Cerebras` engine. State lives in
one host chunk arena (shared, unpinned shapes: the kv list, one window list of the widest window with T_w = kv
page, the fixed record cut into equal word-range pieces, the rs list with T_rs = kv page); each fire stages the
slots its lanes name into fire-sized planes under local ids and writes back what it wrote, so the CSL kernels are
unchanged. Segments are host allocations (commit / release / evacuate are real). Weights are whole on the host
(uploaded per fire, or kept by a resident fabric server): there is no partial device copy to stream into, so no
weight shapes are stated and `weight_residency` / `weight_uses` take and report nothing. Host/disk tiers
(contract 22): host chunks are seated only where `host_kv_budget` is stated (the pool is host DRAM already),
allocated on first write, any shape; the kv slot file (`r0.kv`, kept with `disk_kv_persistent`) holds kv list
slots; `copy_slots` moves device <-> host <-> disk; the legacy `copy_kv` / `copy_state` refuse every host move
(this engine never served them). No hardware here: lib tests and a host simulation of staging and parking
against in-place slots; nothing ran on a CS system or the simulator.

The wafer `Spatial` engine keeps degenerate facts, because: (1) its kv pages are wafer-resident at fixed places
(pool page g on grid row g % P as block g / P; a lane reads at most E pages a row), so a page id is a physical
placement the program is built around: a kv-only shared pool (one kv-list shape, chunks of whole P-page rows,
`Growth::Fixed`, identity table, no evacuation) could be stated, nothing more; (2) recurrent state is not slot
memory there: one state is resident on the wafer and the others are rebuilt by replaying the tokens that made
them, so there is no record to cut into pieces, copy `[run, cow]` or park, and a shared pool stating no fixed
shape would stop the runtime seating state for the wafer's hybrid builds; (3) one-sequence builds keep kv as
replayed tokens, not pages; (4) the unmerged `cerebras-model-support` branch (the Spatial owner's) rewrites
`spatial/engine/{fire,mod,posted}.rs`, `paged.rs` and `wafer.rs` (+1149 / -62). Follow-up for that owner after
its merge: the kv-only shared pool of (1) for batched paged builds.

Pre-existing parity gaps (not memory, but in scope for parity): ROCm synth codegen, ROCm graph replay with >= 2 MoE layers (AMD d); (concat_cols, FP8 m33 fixed M43);
sm_100 tcgen05 W4A16 arms read no expert table; Metal MLA prefill unsupported.

Owners: L40S (CUDA), AMD (ROCm), BM agent (Metal, Mac Studio), VK agent (Vulkan + wgpu), XLA agent (XLA/TPU, sc/xla-chunk-pool; no TPU — verified where it runs, rest marked unverified), TIER agent (host/disk tiers), CB agent (Cerebras, sc/cb-chunk-pool; no hardware — lib tests + host simulation, rest unverified).
