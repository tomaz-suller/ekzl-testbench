from dataclasses import dataclass

import ezkl

from model import ModelPaths


@dataclass
class RunArguments:
    input_visibility: str
    output_visibility: str
    parameter_visibility: str

    def to_ezkl(self):
        args = ezkl.PyRunArgs()
        args.input_visibility = self.input_visibility
        args.output_visibility = self.output_visibility
        args.param_visibility = self.parameter_visibility
        return args


async def calibrate_settings(paths: ModelPaths, visibility: dict[str, str]):
    visibility = {f"{key}_visibility": value for key, value in visibility.items()}
    run_args = RunArguments(**visibility).to_ezkl()

    ezkl.gen_settings(paths.onnx, paths.settings, py_run_args=run_args)

    await ezkl.calibrate_settings(
        paths.calibration_data,
        paths.onnx,
        paths.settings,
        "accuracy",
    )


async def generate_proof(paths: ModelPaths):
    ezkl.compile_circuit(
        paths.onnx,
        paths.compiled_circuit,
        paths.settings,
    )

    await ezkl.get_srs(paths.settings)

    await ezkl.gen_witness(
        paths.inference_data,
        paths.compiled_circuit,
        paths.witness,
    )
    ezkl.setup(
        paths.compiled_circuit,
        paths.verifier_key,
        paths.proofer_key,
    )

    ezkl.prove(
        paths.witness,
        paths.compiled_circuit,
        paths.proofer_key,
        paths.proof,
        "single",
    )


def verify_proof(paths: ModelPaths):
    return ezkl.verify(
        paths.proof,
        paths.settings,
        paths.verifier_key,
    )
