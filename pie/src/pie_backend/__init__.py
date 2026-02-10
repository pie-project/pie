# Pie Backend Python Package

# _pie is the compiled Rust extension module (built by maturin)
try:
    from . import _pie

    __all__ = ["_pie"]
except ImportError:
    # _pie not built yet - this is fine for pure Python usage
    __all__ = []
