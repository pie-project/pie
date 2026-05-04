from __future__ import annotations
from typing import TYPE_CHECKING
import torch

# if TYPE_CHECKING:
#     import torchao
#     from torchao.quantization import (
#         Int4WeightOnlyConfig,
#         Int8WeightOnlyConfig,
#         Float8WeightOnlyConfig,
#     )


# as requested, using absolute imports with lazy loading


def quantize(
    x: torch.Tensor,
    config: (
        "torchao.quantization.Int4WeightOnlyConfig"
        | "torchao.quantization.Int8WeightOnlyConfig"
        | "torchao.quantization.Float8WeightOnlyConfig"
    ),
) -> torch.Tensor:
    import torchao

    if isinstance(config, torchao.quantization.Int4WeightOnlyConfig):
        return quantize_int4(x, config)
    elif isinstance(config, torchao.quantization.Int8WeightOnlyConfig):
        return quantize_int8(x, config)
    elif isinstance(config, torchao.quantization.Float8WeightOnlyConfig):
        return quantize_float8(x, config)
    else:
        raise TypeError(
            f"Unsupported weight-only quantization config type: {type(config)}"
        )


def quantize_int4(
    x: torch.Tensor,
    config: "torchao.quantization.Int4WeightOnlyConfig",
) -> torch.Tensor:
    import torchao

    # Essential imports that are too deep or internal to access via top-level easily without being verbose
    # But user asked to use absolute modules.
    # Let's check if we can access them via torchao.quantization... usually workflows are hidden.
    # We will use the full path for imports if needed, but try to minimize "from ... import ..."

    # Actually, for these specific internal enums, it's safer to import them locally
    # but maybe we can just import the module?
    from torchao.quantization.quantize_.workflows import (
        Int4PackingFormat,
        Int4ChooseQParamsAlgorithm,
    )

    # Note: torchao.quantization exports tensor subclasses usually.

    if x.shape[-1] % config.group_size != 0:
        raise ValueError(
            f"Cannot int4-quantize weight with shape {x.shape}: "
            f"last dim must be divisible by group_size={config.group_size}"
        )

    block_size = [1] * (x.ndim - 1) + [config.group_size]

    if (
        config.int4_choose_qparams_algorithm is Int4ChooseQParamsAlgorithm.HQQ
        and config.int4_packing_format
        not in {Int4PackingFormat.TILE_PACKED_TO_4D, Int4PackingFormat.OPAQUE}
    ):
        raise ValueError(
            "Int4ChooseQParamsAlgorithm.HQQ is only supported by "
            "Int4PackingFormat.TILE_PACKED_TO_4D and Int4PackingFormat.OPAQUE"
        )

    match config.int4_packing_format:
        case Int4PackingFormat.PRESHUFFLED:
            return torchao.quantization.Int4PreshuffledTensor.from_hp(
                x,
                block_size,
                activation_dtype=torch.bfloat16,
            )
        case Int4PackingFormat.PLAIN:
            return torchao.quantization.Int4Tensor.from_hp(x, block_size)
        case Int4PackingFormat.PLAIN_INT32:
            return torchao.quantization.Int4PlainInt32Tensor.from_hp(x, block_size)
        case Int4PackingFormat.MARLIN_SPARSE:
            return torchao.quantization.Int4MarlinSparseTensor.from_hp(x, block_size)
        case Int4PackingFormat.TILE_PACKED_TO_4D:
            return torchao.quantization.Int4TilePackedTo4dTensor.from_hp(
                x,
                block_size,
                int4_choose_qparams_algorithm=config.int4_choose_qparams_algorithm,
            )
        case _:
            raise ValueError(
                f"Unsupported int4 packing format: {config.int4_packing_format}"
            )


def quantize_int8(
    x: torch.Tensor,
    config: "torchao.quantization.Int8WeightOnlyConfig",
) -> torch.Tensor:
    import torchao

    if config.group_size is None:
        group_size = x.shape[-1]
    else:
        group_size = config.group_size

    block_size = (1,) * (x.dim() - 1) + (group_size,)

    return torchao.dtypes.AffineQuantizedTensor.from_hp_to_intx(
        x,
        torchao.quantization.MappingType.SYMMETRIC,
        block_size,
        torch.int8,
        eps=torch.finfo(torch.float32).eps,
        zero_point_dtype=torch.int64,
    )


def quantize_float8(
    x: torch.Tensor, config: "torchao.quantization.Float8WeightOnlyConfig"
) -> torch.Tensor:
    import torchao

    return torchao.quantization.Float8Tensor.from_hp(
        x,
        float8_dtype=x.dtype,
        granularity=torchao.quantization.PerRow(),
    )
