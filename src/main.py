import asyncio
import json
from dataclasses import dataclass, asdict, field

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
    models: list[str]
    visibility: Visibility = field(default_factory=Visibility)
    generate_calibration_data: bool = False
    generate_inference_data: bool = False
    export: bool = False
    calibrate: bool = False
    generate: bool = True
    verify: bool = False


config_store = ConfigStore.instance()
config_store.store(name="config", node=EzklConfig)


async def main(cfg: EzklConfig) -> None:
    models = [
        Model(attributes.model_name, attributes.input_shape)
        for attributes in ModelAttributes
        if attributes.model_name in cfg.models
    ]
    for model in models:
        if cfg.generate_calibration_data:
            with model.paths.calibration_data.open("w") as f:
                json.dump(random_input_data(20, *model.input_shape), f)
        if cfg.generate_inference_data:
            with model.paths.inference_data.open("w") as f:
                json.dump(random_input_data(1, *model.input_shape), f)

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
