"""One run of the device-resident speculative loop, its first rounds traced
(what the head proposed, what the trunk answered), dumped as JSON — the pie
side of a round-for-round reading against `scripts/gemma4_assistant_ref.py`.

    python tests/inferlets/trace_eagle.py --engine metal --model <overlay.zt> \
        [--engine-option ...]     # TRACE_PROMPT, TRACE_K, TRACE_TOKENS, TRACE_OUT
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from conftest import run_inferlet, run_tests  # noqa: E402

PROMPT = os.environ.get("TRACE_PROMPT", "The capital of France is")
K = int(os.environ.get("TRACE_K", "3"))
MAX_TOKENS = int(os.environ.get("TRACE_TOKENS", "32"))
OUT = os.environ.get("TRACE_OUT", "trace_eagle.json")


async def trace(client, args):
    out = await run_inferlet(
        client,
        "mtp-speculative-decoding",
        {"prompt": PROMPT, "max_tokens": MAX_TOKENS, "k": K, "trace": True},
        timeout=args.timeout,
    )
    r = json.loads(out[out.find("{"):])
    print(f"  {len(r['tokens'])} tokens over {r['rounds']} rounds: accepted {r['accepted']}/{r['drafted']} (k={r['k']}, depth={r['depth']})")
    for i, (p, t) in enumerate(zip(r["proposed_trace"], r["truth_trace"])):
        print(f"  round {i}: proposed {p} truth {t}")
    print(f"  {r['text'][:200]!r}")
    json.dump(r, open(OUT, "w"))


if __name__ == "__main__":
    run_tests([trace], "Speculative loop trace")
