import asyncio
from dataclasses import dataclass, field
import gc
import json
import logging
from time import process_time_ns, time_ns
from datetime import datetime
from typing import Any, Union

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
import torch

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
    run_name: str = MISSING
    seed: int = 0
    models: Union[list[str], None] = None
    polynomial: bool = True
    visibility: Visibility = field(default_factory=Visibility)
    generate_calibration_data: bool = True
    generate_inference_data: bool = True
    export: bool = True
    calibrate: bool = True
    calibration_samples: int = 20
    generate: bool = True
    inference_samples: int = 1
    verify: bool = True


config_store = ConfigStore.instance()
config_store.store(name="config", node=EzklConfig)

log = logging.getLogger(__name__)


async def main(cfg: EzklConfig) -> None:
    selected_model_attributes = [
        attributes
        for attributes in ModelAttributes
        if cfg.models is None or attributes.model_name in cfg.models
    ]
    for attributes in selected_model_attributes:
        torch.manual_seed(cfg.seed)
        model = Model(
            attributes.model_name, attributes.input_shape, polynomial=cfg.polynomial
        )
        log.info(f"Processing model {model.name}")
        metrics: dict[str, Any] = {
            "run": cfg.run_name,
            "seed": cfg.seed,
            "timestamp": datetime.now().isoformat(),
        }

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

        for task_name in ("export", "calibrate", "generate", "verify"):
            if cfg[task_name]:
                wall_start = time_ns()
                process_start = process_time_ns()
                await execute_task(
                    task_name, model, OmegaConf.to_container(cfg.visibility)
                )
                metrics[f"{task_name}_wall_time"] = time_ns() - wall_start
                metrics[f"{task_name}_process_time"] = process_time_ns() - process_start

        if cfg.export:
            metrics["export_size"] = model.paths.onnx.stat().st_size
        if cfg.generate:
            metrics["proofer_key_size"] = model.paths.proofer_key.stat().st_size
            metrics["verifier_key_size"] = model.paths.verifier_key.stat().st_size
            metrics["witness_size"] = model.paths.witness.stat().st_size
            metrics["proof_size"] = model.paths.proof.stat().st_size

        metrics_path = model.paths.metrics
        old_metrics = []
        if metrics_path.exists():
            with metrics_path.open("r") as f:
                old_metrics = json.load(f)
        old_metrics.append(metrics)
        with metrics_path.open("w") as f:
            json.dump(old_metrics, f)

        del model
        gc.collect()


async def execute_task(name: str, model: Model, visibility: dict[str, str]):
    if name == "export":
        export(model)
    if name == "calibrate":
        await calibrate_settings(model.paths, visibility)
    if name == "generate":
        await generate_proof(model.paths)
    if name == "verify":
        verify_proof(model.paths)


@hydra.main(version_base=None, config_name="config")
def entrypoint(cfg: EzklConfig) -> None:
    asyncio.run(main(cfg))


if __name__ == "__main__":
    entrypoint()
