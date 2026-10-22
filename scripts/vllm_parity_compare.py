"""Grade pie's probe dump against vLLM's top-k reading of the same battery
(`scripts/vllm_parity_ref.py`).

    python3 scripts/vllm_parity_compare.py DIR [--a pie] [--b vllm]

Per probe: teacher-forced argmax agreement, each flip's margin in vLLM's
reading (a flip under ~0.1 nats is a tie, not a fault), the mean |Δ logprob|
over vLLM's top-k, and the greedy continuations' first divergence with its
margin on both sides.
"""

import argparse
import json
import os

import numpy as np


def logsoftmax(x):
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def margin(top):
    return top[0][1] - top[1][1] if len(top) > 1 else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--a", default="pie")
    ap.add_argument("--b", default="vllm")
    args = ap.parse_args()
    probes = json.load(open(os.path.join(args.dir, "probes.json")))["probes"]
    total = agree_total = 0
    for probe in probes:
        name = probe["name"]
        ref = json.load(open(os.path.join(args.dir, f"{name}.{args.b}.json")))
        tf = np.fromfile(os.path.join(args.dir, f"{name}.{args.a}.tf.f32"), dtype=np.float32)
        n = len(probe["ids"])
        vocab = tf.size // n
        tf = logsoftmax(tf.reshape(n, vocab).astype(np.float64))
        agree, flips, deltas = 0, [], []
        for i, top in enumerate(ref["tf_top"]):
            if not top:
                continue
            mine = int(tf[i].argmax())
            if mine == top[0][0]:
                agree += 1
            else:
                flips.append((i, round(margin(top), 3), round(float(tf[i, mine] - tf[i, top[0][0]]), 3)))
            deltas.extend(abs(tf[i, t] - lp) for t, lp in top[:5])
        total += len(ref["tf_top"])
        agree_total += agree
        line = f"  {name:10s} tf agree {agree}/{len(ref['tf_top'])}  mean|Δlp| top5 {np.mean(deltas):.3f}"
        if flips:
            line += f"  flips (pos, vllm margin, pie margin) {flips[:4]}"
        print(line)
        gen_path = os.path.join(args.dir, f"{name}.{args.a}.gen.f32")
        mine_json = os.path.join(args.dir, f"{name}.{args.a}.json")
        if os.path.exists(mine_json) and os.path.exists(gen_path):
            mine_gen = json.load(open(mine_json)).get("gen", [])
            gen = np.fromfile(gen_path, dtype=np.float32).reshape(-1, vocab)
            same = 0
            for a, b in zip(mine_gen, ref["gen"]):
                if a != b:
                    break
                same += 1
            note = ""
            if same < min(len(mine_gen), len(ref["gen"])):
                top = ref["gen_top"][same]
                row = logsoftmax(gen[same].astype(np.float64))
                note = f"  diverge @{same}: vllm margin {margin(top):.3f}, pie margin {row[mine_gen[same]] - row[ref['gen'][same]]:.3f}"
            print(f"  {'':10s} gen same prefix {same}/{min(len(mine_gen), len(ref['gen']))}{note}")
    print(f"teacher-forced argmax agreement: {agree_total}/{total}")


if __name__ == "__main__":
    main()
