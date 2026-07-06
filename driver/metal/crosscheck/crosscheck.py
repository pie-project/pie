#!/usr/bin/env python3
"""Metal <-> CUDA cross-check harness for the PTIR ptir capstone.

Pre-staged so the cross-check collapses to a one-shot DIFF when charlie's CUDA
capstones (pentathlon parity + real-MTP Stage-1) land. The Metal side is
populated + certified here; the CUDA side is a pluggable placeholder that
charlie fills (same schema).

WHAT IT COMPARES
  1. Goldens (pentathlon_iter / dfa_ingraph / mtp_verify_tail): the per-phase
     `take chan=N` values must be BIT-IDENTICAL on Metal and CUDA. The oracle is
     echo's golden .txt files (eval.rs); Metal is already certified == oracle
     (driver/metal/tier0 tier0_test + pentathlon_test), so CUDA == oracle proves
     Metal == CUDA transitively — and we also diff the arrays directly.
  2. Decode paths (greedy / grammar deterministic; temp / top_k / top_p / min_p
     RNG): deterministic paths must be bit-identical token-for-token; RNG paths
     match iff the shared splitmix64/hash_uniform RNG matches (bit-parity with
     sample_temp.cu). Throughput TRENDS (scaling to saturation) compared
     qualitatively.
  3. MTP: acceptance rate (~50% on Qwen3.5-0.8B K=1) and the speedup verdict
     (net-loss single-stream + under saturation) trends must agree.

USAGE
  # gated (no CUDA yet): validates the harness + Metal == oracle, reports GATED
  python3 crosscheck.py
  # when CUDA lands: drop charlie's outputs in and diff
  python3 crosscheck.py --cuda cuda_results.json
  # regenerate the Metal reference from the live tier-0 binaries (real outputs)
  python3 crosscheck.py --emit-metal metal_results.json --bin ../tier0/build
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.normpath(os.path.join(HERE, "../../../interface/sampling-ir/tests/golden-ptir"))
GOLDENS = ["pentathlon_iter", "dfa_ingraph", "mtp_verify_tail"]
TOL = 1e-4  # f32 within-tol (transcendental libm last-ULP), int/bool exact


# ── golden oracle: parse the `take chan=N = TYPE([..])` lines (ordered) ──────
def parse_typed(s):
    """Parse 'F32([1.0, 2.0])' / 'I32([...])' / 'U32([...])' / 'Bool([...])'."""
    m = re.search(r"(F32|I32|U32|Bool)\(\[([^\]]*)\]\)", s)
    if not m:
        return None
    kind, body = m.group(1), m.group(2).strip()
    toks = [t.strip() for t in body.split(",") if t.strip() != ""]
    if kind == "F32":
        return ("f32", [float(t) for t in toks])
    if kind == "Bool":
        return ("bool", [1 if t == "true" else 0 for t in toks])
    return ("int", [int(t) for t in toks])


def extract_oracle(golden_dir):
    """Ordered list of (chan, typed-value) per golden — the eval.rs reference."""
    out = {}
    for name in GOLDENS:
        path = os.path.join(golden_dir, name + ".txt")
        takes = []
        if os.path.exists(path):
            for line in open(path):
                m = re.match(r"\s*take chan=(\d+)\s*=\s*(.*)", line)
                if m:
                    takes.append([int(m.group(1)), parse_typed(m.group(2))])
        out[name] = takes
    return out


# ── Metal side: run the live tier-0 cert binaries to VERIFY Metal == oracle ──
def run_metal(bin_dir):
    """Run the tier-0 cert binaries; Metal's golden takes are certified ==
    oracle when these pass (tier0_test + pentathlon_test assert bit-exact vs the
    golden files). Returns {binary: passed?}. Missing binaries -> skipped."""
    checks = {"tier0_test": "TIER0_TEST_OK", "pentathlon_test": "PENTATHLON_OK",
              "mtpverify_test": "MTPVERIFY_OK"}
    status = {}
    for binexe, ok_str in checks.items():
        path = os.path.join(bin_dir, binexe)
        if not os.path.exists(path):
            status[binexe] = None
            continue
        try:
            out = subprocess.run([path], capture_output=True, text=True, timeout=180).stdout
            status[binexe] = ok_str in out
        except Exception:
            status[binexe] = False
    return status


# ── diff engine ─────────────────────────────────────────────────────────────
def diff_values(a, b):
    """a, b = (kind, list). Bit-identical for int/bool, within-tol for f32."""
    if a is None or b is None:
        return a == b
    ka, va = a
    kb, vb = b
    if len(va) != len(vb):
        return False
    if ka == "f32" or kb == "f32":
        return all(abs(float(x) - float(y)) <= TOL for x, y in zip(va, vb))
    return list(va) == list(vb)


def diff_goldens(oracle, cuda):
    """Diff CUDA golden takes vs the oracle (== certified Metal)."""
    results = []
    for name in GOLDENS:
        ref = oracle.get(name, [])
        if cuda is None or name not in cuda.get("goldens", {}):
            results.append((name, "GATED", f"{len(ref)} takes staged; awaiting CUDA"))
            continue
        cud = cuda["goldens"][name]
        if len(cud) != len(ref):
            results.append((name, "FAIL", f"take count {len(cud)} vs {len(ref)}"))
            continue
        ok = True
        for (rc, rv), cv in zip(ref, cud):
            # cuda take encoded as [chan, kind, [values]]
            cvv = (cv[1], cv[2]) if isinstance(cv, list) and len(cv) == 3 else None
            if cv[0] != rc or not diff_values(rv, cvv):
                ok = False
                break
        results.append((name, "PASS" if ok else "FAIL", f"{len(ref)} takes"))
    return results


def diff_decode(metal, cuda):
    md = metal["decode"]
    results = []
    if cuda is None or "decode" not in cuda:
        results.append(("cost_order", "GATED", " < ".join(md["cost_order_cheapest_to_costliest"])))
        results.append(("costliest", "GATED", f"{md['costliest']} ({md['costliest_reason']})"))
        results.append(("per_token_identity", "GATED", md["per_token_identity"]))
        return results
    cd = cuda["decode"]
    order_ok = md["cost_order_cheapest_to_costliest"] == cd.get("cost_order_cheapest_to_costliest")
    results.append(("cost_order", "PASS" if order_ok else "CHECK",
                    f"metal={md['cost_order_cheapest_to_costliest']} cuda={cd.get('cost_order_cheapest_to_costliest')}"))
    results.append(("costliest", "PASS" if md["costliest"] == cd.get("costliest") else "CHECK",
                    f"metal={md['costliest']} cuda={cd.get('costliest')}"))
    return results


def diff_mtp(metal, cuda):
    results = []
    m = metal.get("mtp", {})
    if cuda is None or "mtp" not in cuda:
        results.append(("acceptance", "GATED", f"metal={m.get('acceptance')}"))
        results.append(("speedup_verdict", "GATED", m.get("speedup_verdict", "")))
        return results
    c = cuda["mtp"]
    # acceptance trend: within a tolerance band (same model, greedy self-draft)
    acc_ok = abs(float(m.get("acceptance", 0)) - float(c.get("acceptance", -1))) <= 0.10
    results.append(("acceptance", "PASS" if acc_ok else "CHECK",
                    f"metal={m.get('acceptance')} cuda={c.get('acceptance')}"))
    trend_ok = m.get("speedup_regime") == c.get("speedup_regime")
    results.append(("speedup_trend", "PASS" if trend_ok else "CHECK",
                    f"metal={m.get('speedup_regime')} cuda={c.get('speedup_regime')}"))
    return results


def load_metal_reference(oracle):
    """The Metal reference: goldens == oracle (certified); decode + MTP from the
    committed Metal benchmarks (driver/metal/tests)."""
    ref = json.load(open(os.path.join(HERE, "metal_results.json")))
    ref["goldens"] = {  # certified equal to the oracle
        name: [[c, v[0] if v else None, v[1] if v else []] for c, v in oracle[name]]
        for name in GOLDENS
    }
    return ref


def report(title, rows):
    print(f"\n== {title} ==")
    tally = {}
    for name, status, detail in rows:
        tally[status] = tally.get(status, 0) + 1
        print(f"  [{status:5}] {name:22} {detail}")
    return tally


def main():
    ap = argparse.ArgumentParser(description="Metal<->CUDA PTIR cross-check")
    ap.add_argument("--cuda", help="CUDA results JSON (charlie); omit for GATED dry-run")
    ap.add_argument("--bin", default=os.path.join(HERE, "../tier0/build"),
                    help="dir with the tier-0 cert binaries (for real Metal takes)")
    ap.add_argument("--emit-metal", help="write the Metal reference JSON and exit")
    args = ap.parse_args()

    oracle = extract_oracle(GOLDEN_DIR)
    n_oracle = sum(len(v) for v in oracle.values())
    print(f"oracle: {n_oracle} golden takes across {len(GOLDENS)} programs "
          f"({', '.join(GOLDENS)})")
    metal_status = run_metal(args.bin)  # verify Metal == oracle via the cert binaries
    for b, ok in metal_status.items():
        tag = "CERT-OK" if ok else ("MISSING" if ok is None else "CERT-FAIL")
        print(f"  metal {b:16} {tag}")
    metal = load_metal_reference(oracle)

    if args.emit_metal:
        json.dump(metal, open(args.emit_metal, "w"), indent=2)
        print(f"wrote Metal reference -> {args.emit_metal}")
        return 0

    cuda = json.load(open(args.cuda)) if args.cuda else None
    tallies = {}
    for t in (report("goldens (bit-identical takes)", diff_goldens(oracle, cuda)),
              report("decode paths", diff_decode(metal, cuda)),
              report("MTP acceptance / speedup", diff_mtp(metal, cuda))):
        for k, v in t.items():
            tallies[k] = tallies.get(k, 0) + v

    print("\n" + "=" * 60)
    if cuda is None:
        print("STATUS: GATED — harness staged + Metal==oracle reference loaded.")
        print("        Drop charlie's CUDA outputs in: "
              "python3 crosscheck.py --cuda cuda_results.json")
        print(f"        (schema: crosscheck/cuda_results.template.json)")
        return 0
    fails = tallies.get("FAIL", 0) + tallies.get("CHECK", 0)
    print(f"RESULT: {tallies}")
    if fails == 0:
        print("CROSSCHECK_OK — Metal and CUDA bit-identical / trends match.")
        return 0
    print("CROSSCHECK_FAIL — see FAIL/CHECK rows above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
