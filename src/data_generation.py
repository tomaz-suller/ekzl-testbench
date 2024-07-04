from typing import Literal

import torch


def torch_tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return (tensor.detach().numpy()).reshape([-1]).tolist()


def random_input_data(
    samples: int, *shape: int, scale: int = 1
) -> dict[Literal["input_data"], list[list[float]]]:
    data_list = torch_tensor_to_list(
        torch.randn(samples, *shape, requires_grad=True) * scale
    )
    return {"input_data": [data_list]}
