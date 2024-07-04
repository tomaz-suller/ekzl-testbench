import asyncio
from dataclasses import dataclass, asdict, field
import json
import logging
from typing import Union

import hydra
from hydra.core.config_store import ConfigStore

from model import Model, ModelAttributes, export
from proof import calibrate_settings, generate_proof, verify_proof
from data_generation import random_input_data


@dataclass
class Visibility:
    input: str = "public"
    output: str = "public"
    parameter: str = "fixed"


@dataclass
class EzklConfig:
    models: Union[list[str], None] = None
    visibility: Visibility = field(default_factory=Visibility)
    generate_calibration_data: bool = False
    generate_inference_data: bool = False
    export: bool = False
    calibrate: bool = False
    calibration_samples: int = 20
    generate: bool = True
    inference_samples: int = 1
    verify: bool = False


config_store = ConfigStore.instance()
config_store.store(name="config", node=EzklConfig)

log = logging.getLogger(__name__)


async def main(cfg: EzklConfig) -> None:
    models = [
        Model(attributes.model_name, attributes.input_shape)
        for attributes in ModelAttributes
        if cfg.models is None or attributes.model_name in cfg.models
    ]
    for model in models:
        log.info(f"Processing model {model.name}")

        if cfg.generate_calibration_data:
            with model.paths.calibration_data.open("w") as f:
                json.dump(
                    random_input_data(cfg.calibration_samples, *model.input_shape), f
                )
        if cfg.generate_inference_data:
            with model.paths.inference_data.open("w") as f:
                json.dump(
                    random_input_data(cfg.inference_samples, *model.input_shape), f
                )

        if cfg.export:
            export(model)

        if cfg.calibrate:
            await calibrate_settings(model.paths, asdict(cfg.visibility))
        if cfg.generate:
            await generate_proof(model.paths)
        if cfg.verify:
            verify_proof(model.paths)


@hydra.main(version_base=None, config_name="config")
def entrypoint(cfg: EzklConfig) -> None:
    asyncio.run(main(cfg))


if __name__ == "__main__":
    entrypoint()
