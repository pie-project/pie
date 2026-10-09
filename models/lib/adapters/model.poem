# The adapter banks a layer's LoRA correction reads: `slots` adapters of
# `rank`, registered by the host rather than read from a checkpoint.

ADAPTERS = struct(slots = 8, rank = 16)

def banks(prefix, hidden, dense):
    """The `lora_a` and `lora_b` banks under `prefix`, over a `hidden`-wide
    residual, in `dense`."""
    return (
        weight(prefix + ".lora_a", [ADAPTERS.slots, ADAPTERS.rank, hidden], dense).registered(),
        weight(prefix + ".lora_b", [ADAPTERS.slots, hidden, ADAPTERS.rank], dense).registered(),
    )
