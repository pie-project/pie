"""Drift check: Python `control_protocol` constants must match the C++
`control_socket.hpp` definitions.

Parses the C++ header for `inline constexpr std::uint32_t CTRL_METHOD_*`
declarations and compares each value to the corresponding Python
constant. If either side is edited without the other, this test fails
loudly with a per-name diff.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from pie_driver_cuda_native import control_protocol as proto

# Locate the C++ header relative to this test file. Layout:
#   pie/tests/pie_driver_cuda_native/test_control_protocol.py
#   driver/cuda/src/control_socket.hpp
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTROL_SOCKET_HPP = _REPO_ROOT / "driver" / "cuda" / "src" / "control_socket.hpp"

# `inline constexpr std::uint32_t CTRL_METHOD_COPY_D2H = 1;` ─ the leading
# qualifiers may vary slightly; pin on the prefix + name + integer literal.
_DECL_RE = re.compile(
    r"inline\s+constexpr\s+\S+\s+(?P<name>CTRL_METHOD_\w+)\s*=\s*(?P<value>\d+)\s*;"
)


def _parse_cpp_methods() -> dict[str, int]:
    text = _CONTROL_SOCKET_HPP.read_text()
    out: dict[str, int] = {}
    for m in _DECL_RE.finditer(text):
        out[m["name"]] = int(m["value"])
    return out


def test_header_present():
    assert _CONTROL_SOCKET_HPP.exists(), (
        f"expected canonical C++ header at {_CONTROL_SOCKET_HPP}; "
        "the test cannot validate drift if the file moved"
    )


@pytest.mark.parametrize("py_name,cpp_name", [
    ("METHOD_COPY_D2H", "CTRL_METHOD_COPY_D2H"),
    ("METHOD_COPY_H2D", "CTRL_METHOD_COPY_H2D"),
    ("METHOD_COPY_D2D", "CTRL_METHOD_COPY_D2D"),
    ("METHOD_COPY_H2H", "CTRL_METHOD_COPY_H2H"),
])
def test_method_constants_match(py_name: str, cpp_name: str):
    cpp = _parse_cpp_methods()
    assert cpp_name in cpp, (
        f"{cpp_name} not found in {_CONTROL_SOCKET_HPP}; "
        "C++ header may have renamed or removed the constant"
    )
    py_value = getattr(proto, py_name)
    assert py_value == cpp[cpp_name], (
        f"protocol drift: control_protocol.{py_name}={py_value} "
        f"but control_socket.hpp::{cpp_name}={cpp[cpp_name]}"
    )


def test_header_layout_size():
    # Sanity: the header docs claim a 16-byte header.
    assert proto.HEADER_SIZE == 16
    assert proto.RESPONSE_SIZE == 4
