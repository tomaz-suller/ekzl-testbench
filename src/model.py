from dataclasses import dataclass, field, InitVar
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).absolute().parent.parent


class ModelAttributes(Enum):
    FNN = ("fnn", (50,))
    SMALL_CNN = ("small_cnn", (1, 10, 10))
    MNIST = ("mnist", (1, 28, 28))
    LENET5 = ("lenet5", (1, 32, 32))
    VGG11 = ("vgg11", (3, 224, 224))

    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        self.model_name = name
        self.input_shape = shape


@dataclass
class ModelPaths:
    name: InitVar[str]
    root: InitVar[Union[Path, None]] = None

    onnx: Path = field(init=False)
    calibration_data: Path = field(init=False)
    inference_data: Path = field(init=False)
    settings: Path = field(init=False)
    compiled_circuit: Path = field(init=False)
    witness: Path = field(init=False)
    verifier_key: Path = field(init=False)
    prover_key: Path = field(init=False)
    proof: Path = field(init=False)
    metrics: Path = field(init=False)

    def __post_init__(self, name: str, root: Union[Path, None]):
        root = root or REPO_ROOT
        output_dir = root / "output" / name
        data_dir = root / "data"

        self.onnx = root / "models" / f"{name}.onnx"

        self.calibration_data = data_dir / "2-calibration" / f"{name}.json"
        self.inference_data = data_dir / "3-inference" / f"{name}.json"

        self.settings = output_dir / "settings.json"
        self.compiled_circuit = output_dir / "compiled"
        self.witness = output_dir / "witness.json"
        self.verifier_key = output_dir / "vk"
        self.prover_key = output_dir / "pk"
        self.proof = output_dir / "proof"
        self.metrics = output_dir / "metrics.json"

    def __getitem__(self, key: str) -> Path:
        return getattr(self, key)


@dataclass
class Model:
    name: str
    input_shape: tuple[int, ...]
    polynomial: bool = True
    _model: Union[nn.Module, None] = field(default=None, init=False)
    paths: ModelPaths = field(init=False)
    root: InitVar[Union[Path, None]] = None

    def __post_init__(self, root: Union[Path, None]):
        self.paths = ModelPaths(self.name, root)

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            if self.name == ModelAttributes.FNN.model_name:
                self._model = Fnn()
            elif self.name == ModelAttributes.SMALL_CNN.model_name:
                self._model = SmallCnn()
            elif self.name == ModelAttributes.MNIST.model_name:
                self._model = Mnist()
            elif self.name == ModelAttributes.LENET5.model_name:
                self._model = Lenet5(self.polynomial)
            elif self.name == ModelAttributes.VGG11.model_name:
                from torchvision.models import vgg11

                self._model = vgg11()
            else:
                raise ValueError(f"model '{self.name}' not supported")
        return self._model


class Fnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=ModelAttributes.FNN.input_shape[0], out_features=25),
            nn.Linear(in_features=25, out_features=2),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class SmallCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=ModelAttributes.SMALL_CNN.input_shape[0],
            out_channels=6,
            kernel_size=3,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def polynomial_activation(a: float, x):
    """https://arxiv.org/abs/2011.05530"""
    return x * x + a * x


class Mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=ModelAttributes.MNIST.input_shape[0],
            out_channels=4,
            kernel_size=3,
        )
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dense = nn.Linear(200, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = polynomial_activation(1, x)
        # SumPool: AvgPool multiplied by number of elements (4)
        x = self.pool(x) * 4

        x = self.conv2(x)
        x = polynomial_activation(1, x)
        # SumPool: AvgPool multiplied by number of elements (4)
        x = self.pool(x) * 4

        x = x.flatten(start_dim=1)
        x = self.dense(x)

        return x


class Lenet5(nn.Module):
    def __init__(self, polynomial=True):
        super().__init__()

        if polynomial:
            self.activation = partial(polynomial_activation, 1)
        else:
            self.activation = nn.functional.relu

        self.conv1 = nn.Conv2d(
            in_channels=ModelAttributes.LENET5.input_shape[0],
            out_channels=6,
            kernel_size=5,
        )
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)

        x = x.flatten(start_dim=1)

        x = self.classifier(x)

        return x


def export(model: Model, path: Optional[Path] = None):
    input_ = torch.rand(1, *model.input_shape)
    torch.onnx.export(
        model.model,  # Actual nn.Module object
        input_,
        str(path or model.paths.onnx),
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
