# ROCm golden baselines (GOLDEN-LOGITS-ROCM.txt, probe format, usable as PIE_GOLDEN_CHECK)
MI300X gfx942, ROCm 7.15, graphs on, new-memory after M45 (db186bbf). golden_logits == golden_logits_by_refs for all 8.
e536895fe's graphs-on hashes for dsv4/dots3/mimo are INVALID (split-plane routed GLU bug fixed in db186bbf).
mimo26 is the shaped answer; it differs from fully eager (ad8b8d30) from fire 5 by the split-kv arm (1 ulp median).
qwen35-tiny-buffered: golden_buffered; submit_buffered equal; same at PIE_GOLDEN_CHUNK=1048576 / 262144.

Re-baselined for synthesized regions on ROCm (d3f60e9c's successor, on new-memory de21b23f): qwen3, qwen35 and qwen35-tiny-buffered unchanged.
Against the hand rules (12 hashed steps): nemotron and dsv4 0.031, mimo 0.047, dots3 0.094 at most, no flips; minimax a near-tie flip at step 4 (0.31).
gemma4-l6 steps 0-3 identical, step 4 apart from one element of a layer-1 region's 250x2560 result one bf16 place apart (-0.4336 against -0.4355): the norm's square sum in another order (wavefront butterfly against block sum).
gemma4-l30 (region test fixture) traced to one element of 94,720 (row 24, col 1694) three places apart (0.011047 against 0.010864) at layer 9's region (rmsnorm, residual, layer scale, rmsnorm).
