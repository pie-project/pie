# PLEX 31-policy performance reproduction evidence

This report separates three claims:

1. **Policy-kernel proxy trend**: the compiled `.plexpkg` beats the declared
   baseline on a deterministic paper-anchored synthetic workload.
2. **Live-engine mechanism**: the policy executes on A100 vLLM without
   queue drops or unexpected fallback and preserves output tokens.
3. **Paper end-to-end ratio**: the original system, dataset, scale,
   hardware, and baseline are reproduced closely enough to compare the
   reported numeric ratio.

Proxy ratios are not presented as the original paper ratios.

- Pie commit: `8f448e37f`
- vLLM commit: `e14fbf5f0`
- Policies: 31
- Offline proxy trends reproduced: 31
- Live vLLM policies: 23
- Live output-equivalent policies: 23
- Live zero-drop / zero-fallback policies: 23 / 23
- Median live throughput delta: -3.562%
- Multi-seed stable trends: 31 across 4 additional seeds
- Exact paper end-to-end ratios reproduced: 0 (current milestone)

## Results

| Policy | Paper north star | Proxy | Win rate | Decision p50 | Live throughput | Live mechanism | Fidelity |
|---|---|---:|---:|---:|---:|---|---|
| `agentix` | 4-15x program throughput improvement at equal latency | 1.249x | 100.0% | 8159.7 us | +1.077% | S52/C0/F49; enact 442; drop 0; fallback 0 | faithful-with-deferred-mechanics |
| `continuum` | 1.12-3.66x response-time and 1.10-3.22x throughput improvement | 1.054x | 5.5% | 2724.0 us | -3.419% | S250/C216/F253; enact 249; drop 0; fallback 0 | faithful-with-deferred-mechanics |
| `kvflow` | up to 1.83x single-workflow and 2.19x concurrent-workflow speedup | 501.000x | 100.0% | 3646.2 us | -3.990% | S253/C219/F253; enact 257; drop 0; fallback 0 | faithful-with-deferred-mechanics |
| `preble` | 1.5-14.5x average-latency and 2-10x p99-latency improvement | 1.045x | 50.0% | 1374.4 us | - | - | faithful-with-deferred-mechanics |
| `helium` | up to 1.56x end-to-end speedup and much larger gains over naive execution | 1.741x | 100.0% | 12691.8 us | +2.011% | S53/C0/F51; enact 138; drop 0; fallback 0 | faithful-with-deferred-mechanics |
| `vtc` | max-min token fairness across clients without sacrificing work conservation | 512.000x | 100.0% | 1863.1 us | -2.262% | S52/C0/F49; enact 449; drop 0; fallback 0 | faithful |
| `lmetric` | 92%/24% lower mean TTFT/TPOT versus vLLM on ChatBot and 39%/51% lower means in production | 48.581x | 100.0% | 880.2 us | - | - | faithful |
| `fairserve` | 1.03-1.75x overall throughput, with queue-delay gains measured separately versus VTC and RPM | 1.502x | 100.0% | 2909.9 us | -3.447% | S51/C0/F47; enact 430; drop 0; fallback 0 | faithful |
| `marconi` | up to 34.4x token-hit-rate and 71.1% P95 TTFT improvement | 1.082x | 90.6% | 267.3 us | -3.562% | S0/C217/F252; enact 18; drop 0; fallback 0 | faithful-with-deferred-mechanics |
| `ragcache` | 1.2-4x TTFT and up to 2.1x throughput improvement | 8.152x | 85.2% | 3076.7 us | -3.409% | S251/C219/F254; enact 251; drop 0; fallback 0 | faithful-with-deferred-mechanics |
| `dlpm` | up to 2.87x throughput and 2.90-4.06x lower latency | 1.482x | 50.0% | 8979.6 us | -6.141% | S55/C0/F53; enact 485; drop 0; fallback 0 | faithful |
| `infercept` | 1.3-12x lower normalized latency and 1.6-2x higher throughput | 2.142x | 100.0% | 1439.2 us | -8.225% | S259/C219/F0; enact 254; drop 0; fallback 0 | incorrect |
| `peek` | up to 3x cache hit, 7.9x TTFT, 6.7x E2E, and 4.5x throughput improvement | 26.800x | 100.0% | 1989.3 us | -10.807% | S259/C219/F0; enact 255; drop 0; fallback 0 | faithful |
| `qlm` | 40-90% higher SLO attainment and 20-400% higher throughput | 1.522x | 100.0% | 9082.7 us | -4.854% | S77/C0/F75; enact 726; drop 0; fallback 0 | incorrect |
| `slos-serve` | about 2.2x average serving-capacity improvement | 1.532x | 97.7% | 9921.8 us | -3.519% | S80/C0/F0; enact 762; drop 0; fallback 0 | incorrect |
| `dynasor` | 9-52% token reduction and up to 3.3x more queries at equal quality | 1.330x | 71.1% | 25915.3 us | -6.720% | S81/C0/F78; enact 774; drop 0; fallback 0 | material-semantic-gap |
| `justitia` | 57.5% lower average agent completion time | 2.002x | 76.6% | 3658.4 us | -9.057% | S60/C0/F56; enact 37; drop 0; fallback 0 | faithful-with-deferred-mechanics |
| `chameleon` | 1.5x throughput, 80.7% lower P99 TTFT, and 48.1% lower P50 TTFT versus S-LoRA | 1.391x | 100.0% | 9376.4 us | -7.902% | S259/C221/F0; enact 251; drop 0; fallback 0 | material-semantic-gap |
| `hotprefix` | up to 2.25x/2x latency and 1.91x/1.64x throughput improvement over vLLM/SGLang | 2.035x | 86.7% | 5118.6 us | -2.074% | S0/C213/F262; enact 16; drop 0; fallback 0 | incorrect |
| `pard` | 16-176% higher goodput with materially lower drop rates | 1.333x | 100.0% | 37208.6 us | -5.246% | S59/C0/F57; enact 538; drop 0; fallback 0 | faithful-with-deferred-mechanics |
| `branch-regulation` | 1.77x goodput over regulation-off and 1.48x over eager, above 95% SLO attainment | 1.373x | 99.2% | 13425.1 us | -8.254% | S64/C0/F0; enact 570; drop 0; fallback 0 | incorrect |
| `dualmap` | 40.6-80% higher effective request capacity and 14.3-40% higher goodput | 1.185x | 54.7% | 717.2 us | - | - | material-semantic-gap |
| `llumnix` | up to 15x prefill-latency, 2.9x end-to-end-latency, and 1.5x high-priority latency acceleration | 335.025x | 80.5% | 680.0 us | - | - | material-semantic-gap |
| `smetric` | 24-37% lower median TTFT and up to 34% lower P99 TPOT | 24.800x | 75.0% | 735.2 us | - | - | faithful-with-deferred-mechanics |
| `thunderagent` | 1.5-3.6x serving throughput and 1.8-3.9x rollout improvement | 1.653x | 86.7% | 2396.8 us | -0.733% | S68/C0/F66; enact 152; drop 0; fallback 0 | incorrect |
| `pythia` | up to 2.9x JCT reduction and 1.12-1.96x throughput improvement | 2.509x | 91.4% | 1864.1 us | -6.281% | S81/C0/F77; enact 773; drop 0; fallback 0 | incorrect |
| `goodserve` | up to 27.4% higher goodput | 5.133x | 89.1% | 818.2 us | - | - | incorrect |
| `conserve` | 51.08% lower P95 time-to-first-effective-token and 7.51% higher energy efficiency | >999x | 71.9% | 756.1 us | - | - | material-semantic-gap |
| `parrot` | up to 11.7x speedup and 1.8-2.4x batch speedup | 1.579x | 75.0% | 4504.9 us | +2.401% | S83/C0/F0; enact 192; drop 0; fallback 0 | material-semantic-gap |
| `saga` | 1.73x/1.55x TCT speedup and about 1.21x/1.22x memory-utilization improvement | 5.675x | 100.0% | 2171.0 us | -2.082% | S58/C0/F55; enact 521; drop 0; fallback 0 | faithful-with-deferred-mechanics |
| `routebalance` | 2.6-4.1x lower E2E latency than enhanced BEST-Route at high load while preserving quality/cost tradeoffs | 1.203x | 83.6% | 12285.9 us | - | - | faithful |

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
