"""
Market / scheduling accessors.

Thin wrappers over ``pie:core/scheduling`` for power users writing
custom bid strategies. Most callers should leave bidding alone — the
:class:`Generator` auto-bids using a budget-exhausting strategy that
spreads the wallet over the horizon.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wit_world.imports import scheduling as _sched

if TYPE_CHECKING:
    from .context import Context
    from .model import Model


def balance(model: Model) -> float:
    """Current credit balance (global, usable on any device)."""
    return _sched.balance(model._handle)


def rent(ctx: Context) -> float:
    """Rent: clearing price from the last knapsack auction on this
    context's device."""
    return _sched.rent(ctx._handle)


def dividend(model: Model) -> float:
    """Dividend received last step (endowment-proportional share of revenue)."""
    return _sched.dividend(model._handle)


def latency(ctx: Context) -> float:
    """Per-tick latency of this context's device (seconds)."""
    return _sched.latency(ctx._handle)


def price() -> float:
    """Cost to produce one new KV page (constant = 1 credit)."""
    return _sched.price()
