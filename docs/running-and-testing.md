# Running and testing

Install the built distribution into the test environment first. Tests must not
prepend `src` to PYTHONPATH. Test scenes are loaded from examples, while
implementation imports come from the installed package. Common tests use
subprocesses to check import isolation.

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

The Meep CI builds the core source distribution, then builds and installs the
Python core and native sampler from that unpacked source release. It includes
the Python and NumPy compatibility lanes and the weekly MPI-8 job.

## Release checks

Releases require clean source-distribution builds, installed-artifact import
checks, compatibility module identities, and Meep regression tests including
MPI, cylindrical gradients, finite differences, and sparse reconstruction.
Benchmark claims require warmed repeated timing and actual memory measurements;
diagnostic solver timings do not suffice.
