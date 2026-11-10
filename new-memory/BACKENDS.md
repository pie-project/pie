# North star on EVERY backend (tracking matrix)

Goal: the north star (SPEC.md) holds on every backend — CUDA, ROCm, Metal, Vulkan, wgpu, XLA/TPU, Cerebras. A feature
done on one backend is not done until it is on all of them (or the backend's spec realization says why not, e.g.
"one pool fixed at load" on XLA). No backend is "later, optional".

Legend: ✅ verified on hardware · 🟡 in progress / compile-only · 🔴 not started · — not applicable by spec

| Feature | CUDA | ROCm | Metal | Vulkan | wgpu | XLA/TPU | Cerebras |
|---|---|---|---|---|---|---|---|
| Real ChunkFacts, slot-ref lanes, fixed pieces | ✅ M34 | 🟡 AMD (B: packed only) | ✅ M30 | 🟡 VK | 🟡 VK | 🔴 degenerate | 🔴 degenerate |
| Grow / shrink (segments, evacuation) | ✅ M34 | 🟡 AMD (flat: — by spec) | ✅ M30 | 🟡 VK | 🟡 VK | — fixed at load | — host state |
| rs list + slot_copies | ✅ M34 | 🟡 AMD | ✅ M37 | 🟡 VK | 🟡 VK | 🔴 | 🔴 |
| Window lists (own T_w, trimmed) | 🟡 one list, T_w=page | 🟡 AMD | 🟡 one list | 🟡 VK | 🟡 VK | 🔴 | 🔴 |
| Multi-piece records end to end | 🟡 L40S | 🟡 AMD | 🔴 untested | 🟡 VK | 🟡 VK | 🔴 | 🔴 |
| Weight streaming: PLE rows | ✅ M35/M36 | 🟡 AMD | 🔴 | 🟡 VK | 🟡 VK | 🔴 | 🔴 |
| Weight streaming: experts | 🟡 L40S | 🟡 AMD (after L40S) | 🔴 | 🟡 VK | 🟡 VK | 🔴 | 🔴 |
| Host/disk tiers of shared shapes (copy_slots) | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 |
| K2 (no kv spaces) | ✅ M39 | ✅ compile | 🟡 BM4 | compile | compile | compile | compile |

Pre-existing parity gaps (not memory, but in scope for parity): ROCm synth codegen, `gather.concat_cols`, FP8 MFMA m33;
sm_100 tcgen05 W4A16 arms read no expert table; Metal MLA prefill unsupported.

Owners: L40S (CUDA), AMD (ROCm), BM agent (Metal, Mac Studio), VK agent (Vulkan + wgpu), XLA agent (XLA/TPU, sc/xla-chunk-pool; no TPU — verified where it runs, rest marked unverified), TIER agent (host/disk tiers), CB agent (Cerebras, sc/cb-chunk-pool; no hardware — lib tests + host simulation, rest unverified).
