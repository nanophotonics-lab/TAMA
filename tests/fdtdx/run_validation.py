"""Run installed-backend checks serially, saving numerical evidence and logs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "main",
            "periodic",
            "gaussian",
            "dipole",
            "float32",
            "fabrication",
            "sparse",
            "pec",
            "pmc",
            "bloch",
            "bloch32",
        ],
    )
    args = parser.parse_args()
    root, out = Path(__file__).resolve().parent, args.output
    out.mkdir(parents=True, exist_ok=True)
    for name in ("specs", "functionals", "api", "adapter"):
        result = subprocess.run(
            [sys.executable, str(root / f"check_{name}.py")],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "JAX_PLATFORMS": "cpu"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        (out / f"{name}.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            print(result.stdout + result.stderr, flush=True)
            raise SystemExit(result.returncode)
        evidence = json.loads(result.stdout)
        (out / f"{name}.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(dict(passed=name)), flush=True)
    if args.cpu_only:
        return
    for case in [*args.cases, "integration"]:
        path = out / f"backend_{case}.json"
        command = (
            [sys.executable, str(root / "check_integration.py"), "--output", str(path)]
            if case == "integration"
            else [
                sys.executable,
                str(root / "check_backend.py"),
                "--case",
                case,
                "--output",
                str(path),
            ]
        )
        print(json.dumps(dict(starting=case)), flush=True)
        result = subprocess.run(
            command,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "JAX_PLATFORMS": "cuda",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (out / f"backend_{case}.log").write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            print(result.stdout, flush=True)
            raise SystemExit(result.returncode)
        print(json.dumps(dict(passed=case)), flush=True)


if __name__ == "__main__":
    main()
