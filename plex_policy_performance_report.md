# PLEX 31-policy performance reproduction audit

This report separates three claims:

1. **Adaptation proxy trend**: the compiled `.plexpkg` beats the declared
   baseline on a deterministic paper-anchored synthetic workload.
2. **Live-engine mechanism**: the policy executes on A100 vLLM without
   queue drops or unexpected fallback and preserves output tokens.
3. **Paper end-to-end ratio**: the original system, dataset, scale,
   hardware, and baseline are reproduced closely enough to compare the
   reported numeric ratio.

Proxy ratios are not presented as the original paper ratios.

- Pie commit: `5f279f28ae017a923664579e556424294e1e75a7`
- vLLM commit: `cf29e62d1ed511c305402f1aa7ed8fc253889071`
- Policies: 31
- Offline proxy trends reproduced: 31
- Live vLLM policies: 23
- Live output-equivalent policies: 23
- Live zero-drop / zero-fallback policies: 23 / 23
- Median live throughput delta: -5.494%
- Multi-seed stable trends: 31 across 4 additional seeds
- Exact paper end-to-end ratios reproduced: 0 (current milestone)

## Results

| Policy | Paper north star | Proxy | Win rate | Decision p50 | Live throughput | Live mechanism | Fidelity |
|---|---|---:|---:|---:|---:|---|---|
| `agentix` | 4-15x program throughput improvement at equal latency | 1.000x | 71.9% | 2528.9 us | -6.833% | S65/C0/F63; enact 606; drop 0; fallback 0 | material-semantic-gap |
| `continuum` | 1.12-3.66x response-time and 1.10-3.22x throughput improvement | 42.787x | 100.0% | 1214.2 us | +0.358% | S245/C213/F260; enact 243; drop 0; fallback 0 | material-semantic-gap |
| `kvflow` | up to 1.83x single-workflow and 2.19x concurrent-workflow speedup | 81.648x | 82.8% | 1528.1 us | -3.906% | S249/C212/F0; enact 261; drop 0; fallback 0 | material-semantic-gap |
| `preble` | 1.5-14.5x average-latency and 2-10x p99-latency improvement | 2.454x | 21.9% | 661.2 us | - | - | material-semantic-gap |
| `helium` | up to 1.56x end-to-end speedup and much larger gains over naive execution | 2.365x | 79.7% | 16834.5 us | -9.617% | S67/C0/F0; enact 47; drop 0; fallback 0 | material-semantic-gap |
| `vtc` | max-min token fairness across clients without sacrificing work conservation | 4.000x | 100.0% | 752.3 us | -6.422% | S74/C0/F72; enact 690; drop 0; fallback 0 | material-semantic-gap |
| `lmetric` | 92%/24% lower mean TTFT/TPOT versus vLLM on ChatBot and 39%/51% lower means in production | >999x | 45.3% | 650.9 us | - | - | material-semantic-gap |
| `fairserve` | 1.03-1.75x overall throughput, with queue-delay gains measured separately versus VTC and RPM | 1.511x | 100.0% | 697.4 us | -9.152% | S75/C0/F73; enact 714; drop 0; fallback 0 | incorrect |
| `marconi` | up to 34.4x token-hit-rate and 71.1% P95 TTFT improvement | 1.473x | 98.4% | 253.2 us | -1.998% | S0/C211/F263; enact 16; drop 0; fallback 0 | incorrect |
| `ragcache` | 1.2-4x TTFT and up to 2.1x throughput improvement | 6.256x | 80.5% | 239.7 us | -2.326% | S0/C210/F0; enact 17; drop 0; fallback 0 | material-semantic-gap |
| `dlpm` | up to 2.87x throughput and 2.90-4.06x lower latency | 1.406x | 35.9% | 987.3 us | +2.179% | S83/C0/F79; enact 217; drop 0; fallback 0 | material-semantic-gap |
| `infercept` | 1.3-12x lower normalized latency and 1.6-2x higher throughput | 2.254x | 99.2% | 1480.6 us | -8.225% | S259/C219/F0; enact 254; drop 0; fallback 0 | incorrect |
| `peek` | up to 3x cache hit, 7.9x TTFT, 6.7x E2E, and 4.5x throughput improvement | 1.931x | 100.0% | 1915.7 us | -8.078% | S259/C221/F0; enact 252; drop 0; fallback 0 | material-semantic-gap |
| `qlm` | 40-90% higher SLO attainment and 20-400% higher throughput | 1.529x | 99.2% | 8494.6 us | -4.854% | S77/C0/F75; enact 726; drop 0; fallback 0 | incorrect |
| `slos-serve` | about 2.2x average serving-capacity improvement | 1.518x | 100.0% | 9679.1 us | -3.519% | S80/C0/F0; enact 762; drop 0; fallback 0 | incorrect |
| `dynasor` | 9-52% token reduction and up to 3.3x more queries at equal quality | 1.261x | 64.1% | 25751.8 us | -6.720% | S81/C0/F78; enact 774; drop 0; fallback 0 | material-semantic-gap |
| `justitia` | 57.5% lower average agent completion time | 1.695x | 71.1% | 2537.2 us | -5.494% | S78/C0/F76; enact 749; drop 0; fallback 0 | incorrect |
| `chameleon` | 1.5x throughput, 80.7% lower P99 TTFT, and 48.1% lower P50 TTFT versus S-LoRA | 1.368x | 100.0% | 10063.9 us | -7.902% | S259/C221/F0; enact 251; drop 0; fallback 0 | material-semantic-gap |
| `hotprefix` | up to 2.25x/2x latency and 1.91x/1.64x throughput improvement over vLLM/SGLang | 1.906x | 87.5% | 5104.6 us | -2.074% | S0/C213/F262; enact 16; drop 0; fallback 0 | incorrect |
| `pard` | 16-176% higher goodput with materially lower drop rates | 1.884x | 46.1% | 35215.0 us | -6.898% | S80/C0/F77; enact 773; drop 0; fallback 0 | incorrect |
| `branch-regulation` | 1.77x goodput over regulation-off and 1.48x over eager, above 95% SLO attainment | 1.370x | 100.0% | 13039.4 us | -8.254% | S64/C0/F0; enact 570; drop 0; fallback 0 | incorrect |
| `dualmap` | 40.6-80% higher effective request capacity and 14.3-40% higher goodput | 1.157x | 53.9% | 721.1 us | - | - | material-semantic-gap |
| `llumnix` | up to 15x prefill-latency, 2.9x end-to-end-latency, and 1.5x high-priority latency acceleration | 288.941x | 80.5% | 678.8 us | - | - | material-semantic-gap |
| `smetric` | 24-37% lower median TTFT and up to 34% lower P99 TPOT | 5.945x | 26.6% | 768.7 us | - | - | material-semantic-gap |
| `thunderagent` | 1.5-3.6x serving throughput and 1.8-3.9x rollout improvement | 1.687x | 85.9% | 2447.8 us | -0.733% | S68/C0/F66; enact 152; drop 0; fallback 0 | incorrect |
| `pythia` | up to 2.9x JCT reduction and 1.12-1.96x throughput improvement | 2.304x | 92.2% | 1791.7 us | -6.281% | S81/C0/F77; enact 773; drop 0; fallback 0 | incorrect |
| `goodserve` | up to 27.4% higher goodput | 3.632x | 88.3% | 861.6 us | - | - | incorrect |
| `conserve` | 51.08% lower P95 time-to-first-effective-token and 7.51% higher energy efficiency | >999x | 75.0% | 754.2 us | - | - | material-semantic-gap |
| `parrot` | up to 11.7x speedup and 1.8-2.4x batch speedup | 1.702x | 78.9% | 4821.9 us | +2.401% | S83/C0/F0; enact 192; drop 0; fallback 0 | material-semantic-gap |
| `saga` | 1.73x/1.55x TCT speedup and about 1.21x/1.22x memory-utilization improvement | 114.466x | 90.6% | 3326.7 us | -5.007% | S78/C0/F0; enact 750; drop 0; fallback 0 | incorrect |
| `routebalance` | 2.6-4.1x lower E2E latency than enhanced BEST-Route at high load while preserving quality/cost tradeoffs | 1.347x | 95.3% | 11531.0 us | - | - | incorrect |

## Interpretation

A positive proxy result proves that the committed policy kernel
implements a useful ordering, admission, routing, reclaim, or
feedback mechanism on the declared scenario. It does not prove the
paper's full-system speedup when predictor training, migration,
multi-GPU execution, cache movement, provisioning, or private traces
remain deferred.

The machine-readable companion records per-trial wins/losses,
decision latency, live worker counters, output equivalence, and
independent fidelity findings.
