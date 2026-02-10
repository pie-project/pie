def linear_hook(
    weight_name: str,
    adapter_indptr: list[int],
    adapters: list[Adapter],
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:

    # check if there exists an adapter for this weight

    # do the base linear
    base = torch.linear(x, w, bias)

    for i, adapter in enumerate(adapters):
        if adapter.contains(weight_name):
            x_start = adapter_indptr[i]
            x_end = adapter_indptr[i + 1]
            adapter.apply(weight_name, x[x_start:x_end], base[x_start:x_end])

    return base


class Adapter:

    def __init__(self):
        pass

    def init(self):
        pass

    def contains(self, weight_name: str) -> bool:
        return False

    def apply(self, weight_name: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        pass


class Trainer(Adapter):

    def __init__(self):
        super().__init__()
        pass

    def init(self):
        pass

    def set_seeds(self, seeds: torch.LongTensor):
        pass

    # only matmul
    def apply(self, weight_name: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        # low-rank perturbations are default.

        pass

    def update(self, seeds: torch.LongTensor, scores: torch.Tensor):
        pass


class CmaesTrainer(Trainer):
    def __init__(self):
        super().__init__()
        pass
