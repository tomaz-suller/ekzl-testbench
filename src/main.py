import asyncio
from dataclasses import dataclass, field
import json
import logging
from time import process_time_ns
from datetime import datetime
from typing import Any, Union

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

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
    polynomial: bool = True
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
        Model(attributes.model_name, attributes.input_shape, polynomial=cfg.polynomial)
        for attributes in ModelAttributes
        if cfg.models is None or attributes.model_name in cfg.models
    ]
    for model in models:
        log.info(f"Processing model {model.name}")
        metrics: dict[str, Any] = {"timestamp": str(datetime.now())}

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
            start = process_time_ns()
            export(model)
            export_time = process_time_ns() - start
            metrics["export_time"] = export_time

        if cfg.calibrate:
            start = process_time_ns()
            await calibrate_settings(
                model.paths, OmegaConf.to_container(cfg.visibility)
            )
            calibration_time = process_time_ns() - start
            metrics["calibration_time"] = calibration_time
        if cfg.generate:
            start = process_time_ns()
            await generate_proof(model.paths)
            generation_time = process_time_ns() - start
            metrics["generation_time"] = generation_time
        if cfg.verify:
            start = process_time_ns()
            verify_proof(model.paths)
            verification_time = process_time_ns() - start
            metrics["verification_time"] = verification_time

        metrics_path = model.paths.metrics
        old_metrics = []
        if metrics_path.exists():
            with metrics_path.open("r") as f:
                old_metrics = json.load(f)
        old_metrics.append(metrics)
        with metrics_path.open("w") as f:
            json.dump(old_metrics, f)


@hydra.main(version_base=None, config_name="config")
def entrypoint(cfg: EzklConfig) -> None:
    asyncio.run(main(cfg))


if __name__ == "__main__":
    entrypoint()
