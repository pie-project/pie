# ROCm golden baselines (GOLDEN-LOGITS-ROCM.txt, probe format, usable as PIE_GOLDEN_CHECK)
MI300X gfx942, ROCm 7.15, graphs on, new-memory after M45 (db186bbf). golden_logits == golden_logits_by_refs for all 8.
e536895fe's graphs-on hashes for dsv4/dots3/mimo are INVALID (split-plane routed GLU bug fixed in db186bbf).
mimo26 is the shaped answer; it differs from fully eager (ad8b8d30) from fire 5 by the split-kv arm (1 ulp median).
qwen35-tiny-buffered: golden_buffered; submit_buffered equal; same at PIE_GOLDEN_CHUNK=1048576 / 262144.
