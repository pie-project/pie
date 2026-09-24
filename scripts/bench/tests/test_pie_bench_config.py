"""`--max-model-len` reaches the engine on every backend the bench boots.

The cuda_native branch of `build_config` once forwarded every knob but this
one, so `pie serve` declared its own 4096-token context whatever the shape
asked (pie-evals nightly 35993026892: `--max-model-len 34816`, engine at
`256 seat(s) x 4096 context`, lc-8k refused at page 257). The config is
checked as the server reads it — the rendered TOML — not as the dict is
built.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import pie_bench  # noqa: E402


def _engine_table(argv: list[str]) -> dict:
    args = pie_bench.build_parser().parse_args(argv)
    cfg, _ = pie_bench.build_config(args)
    return tomllib.loads(cfg.to_toml())["engine"]


def _tput(snapshot: Path, engine: str, *extra: str) -> list[str]:
    # A snapshot directory, as pie-evals passes one; the config is built
    # from its path alone and no weight is read.
    (snapshot / "config.json").write_text("{}")
    return [
        "tput",
        "--model",
        str(snapshot),
        "--engine",
        engine,
        "--device",
        "cuda:0" if engine == "cuda_native" else "metal:0",
        "--inferlet-dir",
        str(BENCH),
        *extra,
    ]


def test_the_stated_context_reaches_a_cuda_engine(tmp_path):
    engine = _engine_table(_tput(tmp_path, "cuda_native", "--max-model-len", "34816"))
    assert engine["type"] == "cuda_native"
    assert engine["max_model_len"] == 34816


def test_the_benchs_default_context_is_stated_not_the_engines(tmp_path):
    # Every other engine bench serves at the shared 2048 default; a
    # cross-engine row is like for like only if this one does too.
    assert _engine_table(_tput(tmp_path, "cuda_native"))["max_model_len"] == 2048
    assert _engine_table(_tput(tmp_path, "metal"))["max_model_len"] == 2048


def test_the_context_rides_beside_the_other_engine_knobs(tmp_path):
    engine = _engine_table(
        _tput(tmp_path, "cuda_native", "--max-model-len", "16384", "--max-forward-requests", "64")
    )
    assert engine["max_model_len"] == 16384
    assert engine["max_forward_requests"] == 64
    assert engine["gpu_mem_utilization"] == 0.9
