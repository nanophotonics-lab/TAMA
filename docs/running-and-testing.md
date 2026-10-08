# Running and testing

Install the built distribution into the test environment first. Tests must not
prepend `src` or the former standalone FDTDX package to PYTHONPATH. Test scenes
are loaded from examples, while implementation imports come from the installed
package. Common tests use subprocesses to check import isolation.

## Meep

```bash
python examples/tda_3d_projected_gradient.py --iterations 1
python examples/tda_3d_gamma_rgb_router.py --dry-run
python examples/tda_cylindrical_mode.py --target point
python -m pytest -q tests/common tests/meep
mpirun -np 2 python -m pytest -q -p no:cacheprovider tests/meep -m mpi2
mpirun -np 4 python -m pytest -q -p no:cacheprovider tests/meep -m mpi4
mpirun -np 8 python -m pytest -q -p no:cacheprovider tests/meep -m mpi8
```

The existing Meep CI builds the core source distribution, then builds and
installs the pure core and native sampler from that unpacked source release.
It retains the Python and NumPy compatibility lanes and the weekly MPI-8 job.

## FDTDX

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/common tests/fdtdx -m 'not fdtdx_gpu'
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m pytest -q tests/fdtdx -m fdtdx_gpu
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python tests/fdtdx/run_validation.py --output verification/fdtdx
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python examples/fdtdx/optimize_broadband.py --cpu-offload on --iterations 3
```

Ordinary pytest excludes the marked GPU tests by default. Selecting `fdtdx_gpu`
requires an actual CUDA device and fails if none is available. CPU tests cannot
certify offload or GPU execution. Backend tests are selected explicitly in CI;
do not run Meep tests in a Meep-free environment.

GPU release checks run manually on a maintainer-provisioned CUDA runner.
The maintainer selects the revision to test. The workflow does not
automatically run untrusted pull-request code.

## Release checks

Releases require clean source-distribution builds, installed-artifact import
checks, legacy module identities, Meep regression including MPI/cylindrical,
FDTDX CPU/GPU AD and finite differences, sparse sampling and offload parity.
Updating FDTDX dependencies requires rerunning the supported backend matrix.
Benchmark claims require warmed repeated timing and actual memory measurements;
diagnostic solver timings do not suffice.
