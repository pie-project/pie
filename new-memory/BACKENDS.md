# North star on EVERY backend (tracking matrix)

Goal: the north star (SPEC.md) holds on every backend — CUDA, ROCm, Metal, Vulkan, wgpu, XLA/TPU, Cerebras. A feature
done on one backend is not done until it is on all of them (or the backend's spec realization says why not, e.g.
"one pool fixed at load" on XLA). No backend is "later, optional".

Legend: ✅ verified on hardware · 🟡 in progress / compile-only · 🔴 not started · — not applicable by spec

| Feature | CUDA | ROCm | Metal | Vulkan | wgpu | XLA/TPU | Cerebras |
|---|---|---|---|---|---|---|---|
| Real ChunkFacts, slot-ref lanes, fixed pieces | ✅ M34 | 🟡 AMD (B: packed only) | ✅ M30 | 🟡 VK | 🟡 VK | 🔴 degenerate | 🟡 CB (host pool) |
| Grow / shrink (segments, evacuation) | ✅ M34 | 🟡 AMD (flat: — by spec) | ✅ M30 | 🟡 VK | 🟡 VK | — fixed at load | 🟡 CB (host segments) |
| rs list + slot_copies | ✅ M34 | 🟡 AMD | ✅ M37 | 🟡 VK | 🟡 VK | 🔴 | 🟡 CB |
| Window lists (own T_w, trimmed) | 🟡 one list, T_w=page | 🟡 AMD | 🟡 one list | 🟡 VK | 🟡 VK | 🔴 | 🟡 CB (one list, T_w=page) |
| Multi-piece records end to end | 🟡 L40S | 🟡 AMD | 🔴 untested | 🟡 VK | 🟡 VK | 🔴 | 🟡 CB (host sim) |
| Weight streaming: PLE rows | ✅ M35/M36 | 🟡 AMD | 🔴 | 🟡 VK | 🟡 VK | 🔴 | — weights on host |
| Weight streaming: experts | 🟡 L40S | 🟡 AMD (after L40S) | 🔴 | 🟡 VK | 🟡 VK | 🔴 | — weights on host |
| Host/disk tiers of shared shapes (copy_slots) | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 |
| K2 (no kv spaces) | ✅ M39 | ✅ compile | 🟡 BM4 | compile | compile | compile | compile |

Cerebras realization (CB, `sc/cb-chunk-pool`): the column is the host-traced `Cerebras` engine. State lives in
one host chunk arena (shared, unpinned shapes: the kv list, one window list of the widest window with T_w = kv
page, the fixed record cut into equal word-range pieces, the rs list with T_rs = kv page); each fire stages the
slots its lanes name into fire-sized planes under local ids and writes back what it wrote, so the CSL kernels are
unchanged. Segments are host allocations (commit / release / evacuate are real). Weights are whole on the host
(uploaded per fire, or kept by a resident fabric server): there is no partial device copy to stream into, so no
weight shapes are stated and `weight_residency` / `weight_uses` take and report nothing. Host/disk tiers:
`copy_slots` refuses other domains (hook for the tier work). The wafer `Spatial` engine (kv on the wafer) still
states degenerate facts. No hardware here: lib tests and a host simulation of staging against in-place slots;
nothing ran on a CS system or the simulator.

Pre-existing parity gaps (not memory, but in scope for parity): ROCm synth codegen, `gather.concat_cols`, FP8 MFMA m33;
sm_100 tcgen05 W4A16 arms read no expert table; Metal MLA prefill unsupported.

Owners: L40S (CUDA), AMD (ROCm), BM agent (Metal, Mac Studio), VK agent (Vulkan + wgpu), XLA agent (XLA/TPU, sc/xla-chunk-pool; no TPU — verified where it runs, rest marked unverified), TIER agent (host/disk tiers), CB agent (Cerebras, sc/cb-chunk-pool; no hardware — lib tests + host simulation, rest unverified).
