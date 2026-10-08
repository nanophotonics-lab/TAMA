"""Optimize two spectral bands through TAMA's optional FDTDX backend."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import tama
from native_scene import make_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-offload", choices=("on", "off"), default="off")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("fdtdx_optimization.json"))
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    jax.config.update("jax_enable_x64", True)
    tm = tama.get_backend("fdtdx")
    scene = make_scene()
    spec, design, rho = scene["spec"], scene["design"], scene["rho"]
    f0 = scene["frequency"]
    bands = [
        tm.Band(0, tm.bandpass_kernel(a * f0, b * f0, spec.dt, 41), weight=1 / spec.dt)
        for a, b in ((0.65, 1.05), (0.95, 1.35))
    ]
    objective = tm.MultiTDAObjective(
        simulation=spec,
        design=design,
        targets=[tm.PointTarget(scene["target"], "Ez")],
        bands=bands,
        scalarization_fn=lambda values: -jnp.sum(values),
        cpu_offload=args.cpu_offload == "on",
    )
    x = np.asarray(rho).ravel()
    validation = objective.validate(x)
    initial = objective.fom(x)
    result = minimize(
        objective.fom_and_grad,
        x,
        jac=True,
        bounds=[(0.0, 1.0)] * x.size,
        method="L-BFGS-B",
        options={"maxiter": args.iterations},
    )
    output = dict(
        initial=initial,
        final=float(result.fun),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        stop_message=str(result.message),
        validation=validation,
        parameter_shape=list(design.parameter_shape),
        rho=result.x.reshape(design.parameter_shape).tolist(),
        cpu_offload=args.cpu_offload,
        final_info=objective.last_info,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(saved=str(args.output), initial=initial, final=float(result.fun))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
