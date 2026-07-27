"""Does `envelope_dot` score the pages the request actually owns?

Quest's kernel is handed one flat page-id array for the whole fire and has to
find its own request's slice in it. There are two candidate CSRs: the host one
and the device one. Under decode envelopes -- the path Quest runs on, because
its page table is composed on device -- they are NOT the same array. The host
copy is `plan_kv_page_indptr`, a prefix sum of the *declared* page-channel
capacity, so it is an upper bound; the device copy is a prefix sum of the real
per-request page counts. Slicing with the host copy is wrong twice over:

  * the OFFSET is wrong whenever an earlier request has fewer pages than it
    declared, so request `r` is scored against request `r-1`'s keys;
  * the COUNT is wrong whenever this request has fewer pages than it declared,
    so slots past the end are scored as though they held real keys instead of
    getting the `-inf` that keeps a top-k consumer away from them.

Both failures are silent. A wrong-offset score is a plausible float; a
past-the-end score is a plausible float. Nothing NaNs, nothing throws, and the
generated text stays fluent because Quest still keeps *some* pages. The only
way to see either is to construct a case where bound != real and check the
arithmetic exactly.

This file does that. `test_quest_page_slots` makes `max_tokens` large relative
to the prompt, so the declared bound exceeds the real page count at every step,
and asserts the full slot census against host arithmetic. `test_quest_two_
requests` runs two requests of deliberately different lengths concurrently and
requires the co-batched scores to equal the solo ones -- the offset half of the
bug, which no single-request test can reach.

Run from the repo root with PYTHONPATH=sdk/python-server/python.
"""

import asyncio
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ.setdefault("PIE_CUDA_KV_ENVELOPES", "1")
os.environ.setdefault("PIE_QUEST_DEBUG", "1")

from conftest import run_inferlet, run_tests  # noqa: E402


class _CaptureFd2:
    """Capture the driver's stderr, then replay it.

    `resolve_lane_envelope` is C++ and writes to fd 2 directly, so Python-level
    redirection cannot see it. The test needs one line from there -- the fire's
    request count -- to know whether the engine actually co-batched, which is a
    scheduling outcome no program can request. Everything captured is written
    back out on exit, so nothing is swallowed.
    """

    def __enter__(self):
        sys.stderr.flush()
        self._tmp = tempfile.TemporaryFile(mode="w+")
        self._saved = os.dup(2)
        os.dup2(self._tmp.fileno(), 2)
        return self

    def __exit__(self, *exc):
        os.dup2(self._saved, 2)
        os.close(self._saved)
        self._tmp.seek(0)
        self.text = self._tmp.read()
        self._tmp.close()
        sys.stderr.write(self.text)
        sys.stderr.flush()
        return False

    def max_requests(self):
        seen = [int(m) for m in re.findall(r"\[quest\] R_max=(\d+)", self.text)]
        return max(seen) if seen else 0


# Long enough to span many pages, so `page_budget` is a real constraint and the
# page census has something to say.
LONG_PROMPT = (
    "The capital of France is Paris. "
    + "Paris is a large European city with a long history. " * 24
)
# Deliberately a different length, so a wrong slice offset for the second
# request in a co-batched fire lands somewhere visibly wrong.
SHORT_PROMPT = "The capital of Japan is Tokyo. Tokyo is a large city."


async def _quest(client, args, prompt, max_tokens, budget, seed=9182):
    out = await run_inferlet(
        client,
        "quest-attention",
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "page_budget": budget,
            "temperature": 0.001,
            "seed": seed,
        },
        timeout=args.timeout,
    )
    return json.loads(out[out.find("{"):])


def _census(report):
    """The slot census `envelope_dot` must produce, derived independently.

    Mirrors the kernel's three branches from the fire's own `kv_len`, which the
    inferlet drains as `kv_len_last`:

        page_count   = ceil(kv_len / page_size)          real pages
        kv_before    = kv_len - qo_len                   what envelopes describe
        scored_pages = min(kv_before // page_size, page_count)

    `qo_len` is 1: every fire in the decode loop is one token wide.
    """
    ps = report["page_size"]
    kv_len = report["kv_len_last"]
    page_count = -(-kv_len // ps)
    scored = min((kv_len - 1) // ps, page_count)
    return {
        "pages_finite": scored,
        "pages_pinned": page_count - scored,
        "pages_absent": report["max_pages"] - page_count,
        "page_count": page_count,
    }


async def test_quest_page_slots(client, args):
    # `max_tokens` is large relative to the prompt, so the declared page bound
    # -- `ceil((n + max_tokens + 1) / page_size)` -- runs well ahead of the
    # pages the request actually holds. This is the configuration the old
    # host-sliced code got wrong; at `max_tokens=8` the two coincide, which is
    # exactly why every earlier test passed.
    report = await _quest(client, args, LONG_PROMPT, max_tokens=64, budget=4)
    want = _census(report)

    print(f"  page_size={report['page_size']} kv_len_last={report['kv_len_last']}")
    print(f"  max_pages={report['max_pages']} real_pages={want['page_count']}")
    print(f"  finite={report['pages_finite']} pinned={report['pages_pinned']} "
          f"absent={report['pages_absent']} nan={report['pages_nan']}")

    assert report["layers_observed"] > 0, report
    assert report["pages_nan"] == 0, report

    # The premise of the test. If this fails the case is not exercising the bug
    # and every assertion below is vacuous, so it is checked first and loudly.
    assert want["pages_absent"] > 0, (
        "test is vacuous: the declared page bound equals the real page count, "
        f"so no slot is out of range\n  {report}"
    )

    assert report["pages_absent"] == want["pages_absent"], (
        "envelope_dot scored slots the request does not own. This is the "
        "host-CSR slice bug: the page count came from the declared bound "
        f"instead of the device CSR.\n  want {want}\n  got  {report}"
    )
    assert report["pages_finite"] == want["pages_finite"], (
        f"wrong number of finalised pages\n  want {want}\n  got  {report}"
    )
    assert report["pages_pinned"] == want["pages_pinned"], (
        f"wrong number of in-flight (+inf) pages\n  want {want}\n  got  {report}"
    )
    # The census must be a partition: every declared slot got exactly one of
    # the three treatments, so no slot was left holding a stale value.
    total = (report["pages_finite"] + report["pages_pinned"]
             + report["pages_absent"])
    assert total == report["max_pages"], (
        f"slot census does not cover the result row: {total} of "
        f"{report['max_pages']}\n  {report}"
    )
    print(f"  [ ok ] {want['pages_absent']} out-of-range slots correctly -inf")

    # And the consequence that matters to the policy: a page the request does
    # not own must never be selected. `kept_pages` is the host mirror of what
    # `pivot_threshold(.., rank_le(budget))` sends to `attn_page_mask`.
    assert all(p < want["page_count"] for p in report["kept_pages"]), (
        "Quest selected a page outside the request's page list\n"
        f"  real_pages={want['page_count']} kept={report['kept_pages']}"
    )
    print(f"  [ ok ] kept pages {report['kept_pages']} all within the request")


async def test_quest_two_requests(client, args):
    # Solo reference for the long request.
    solo = await _quest(client, args, LONG_PROMPT, max_tokens=48, budget=4)

    # ...and the same request again, this time sharing the engine with a much
    # shorter one. If the slice offset came from the host bound, the long
    # request's pages would start at the wrong place the moment it is not
    # request 0 -- or the short one's would, when it is not. Either way the
    # scores move, because they would be computed over different keys.
    with _CaptureFd2() as cap:
        both = await asyncio.gather(
            _quest(client, args, LONG_PROMPT, max_tokens=48, budget=4),
            _quest(client, args, SHORT_PROMPT, max_tokens=48, budget=4),
        )
    conc_long, conc_short = both
    r_max = cap.max_requests()

    print(f"  co-batched fire width (R_max) = {r_max}")
    print(f"  solo   max_pages={solo['max_pages']} kv_len={solo['kv_len_last']}")
    print(f"  conc   max_pages={conc_long['max_pages']} "
          f"kv_len={conc_long['kv_len_last']}")
    print(f"  short  max_pages={conc_short['max_pages']} "
          f"kv_len={conc_short['kv_len_last']}")

    for r in (conc_long, conc_short):
        assert r["layers_observed"] > 0, r
        assert r["pages_nan"] == 0, r
        want = _census(r)
        assert r["pages_absent"] == want["pages_absent"], (
            f"co-batched slot census is wrong\n  want {want}\n  got  {r}")
        assert r["pages_finite"] == want["pages_finite"], (
            f"co-batched slot census is wrong\n  want {want}\n  got  {r}")

    # The requests are independent programs with independent KV, so co-batching
    # must not be observable at all. Scores are compared with a tolerance
    # because a different decode batch shape changes the attention kernel's
    # split/reduction plan and moves the last bits of the query (see the
    # coherence note in test_curated.py); a wrong page slice, by contrast,
    # reads entirely different keys and moves the score by orders of magnitude.
    assert conc_long["text"] == solo["text"], (
        "sharing a fire changed the long request's continuation\n"
        f"  solo {solo['text']!r}\n  conc {conc_long['text']!r}"
    )
    a = solo["page_scores"]
    b = conc_long["page_scores"]
    assert len(a) == len(b), (len(a), len(b))
    worst = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        if x in ("inf", "-inf", "nan") or y in ("inf", "-inf", "nan"):
            assert x == y, (
                f"page {i} changed class between solo and co-batched: "
                f"{x} -> {y}\n  solo {a}\n  conc {b}")
            continue
        fx, fy = float(x), float(y)
        rel = abs(fx - fy) / max(abs(fx), abs(fy), 1e-6)
        worst = max(worst, rel)
    assert worst < 5e-2, (
        f"co-batching moved a page score by {worst:.3%} -- the fire's page "
        f"list was sliced at the wrong offset\n  solo {a}\n  conc {b}"
    )
    print(f"  [ ok ] co-batched scores match solo (worst rel {worst:.2e})")

    # Last, and only now that the comparison has passed: was the R>1 path even
    # reached? Co-batching is the scheduler's decision, so this cannot be
    # asserted before the fact -- but a green run that never batched two
    # requests into one fire has not tested the offset half of the bug at all,
    # and saying so is better than a false negative.
    assert r_max >= 2, (
        "the engine never co-batched the two requests into one fire, so the "
        "cross-request page-slice offset was not exercised. R_max="
        f"{r_max}. Raise max_tokens or lower the prompt lengths."
    )
    print(f"  [ ok ] R>1 fires observed (R_max={r_max})")


def tests():
    return [test_quest_page_slots, test_quest_two_requests]


if __name__ == "__main__":
    run_tests(tests(), "envelope_dot page-slice resolution")
