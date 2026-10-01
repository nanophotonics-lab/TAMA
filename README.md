# TAMA: Time-domain Adjoint with Memory-efficient Acceleration

TAMA provides time-domain adjoint optimization with an explicit **Meep or
FDTDX** backend. Version `0.6.0a1` is the first integrated alpha with both engines.

The Python package contains both backend implementations. The selected engine
supplies the FDTD calculations. FDTDX engine sources are not patched, and the
Meep native sampler is built separately inside the active Meep/MPI environment.

```python
import tama
tm = tama.get_backend("fdtdx")  # or "meep"
```

Use the complete [FDTDX broadband example](examples/fdtdx/optimize_broadband.py)
or [Meep projected-gradient example](examples/tda_3d_projected_gradient.py)
to construct and optimize a scene.

`import tama` does not import either engine. Existing root exports such as
`tama.TDAObjective` and old submodule imports continue to refer to the Meep API.
The two engines share the optimizer call contract, while scene definitions,
coordinates, units, band weights and mode normalization remain backend-specific.

## Install from this repository

The commands below run from the repository root or an unpacked source
distribution and do not require these packages to be published on PyPI.

```bash
git clone https://github.com/nanophotonics-lab/TAMA.git
cd TAMA
```

For FDTDX in a Linux/WSL Python 3.12 or 3.13 environment:

```bash
python -m pip install '.[fdtdx]'
# For the tested CUDA 12 lane, when CUDA JAX is not already configured:
python -m pip install 'jax[cuda12]==0.11.0'
JAX_PLATFORMS=cuda python -c "import jax; print(jax.devices())"
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python examples/fdtdx/optimize_broadband.py --cpu-offload off --iterations 3
```

For Meep, keep Python, PyMeep, MPI and the compiler in the same conda environment:

```bash
conda env create -f environment.yml
conda activate tama
python -m pip install .
python -m pip install --no-build-isolation ./native/meep
python -c "import tama; print(tama.get_backend('meep').native_sampler_available())"
python examples/tda_3d_projected_gradient.py --iterations 1
```

The pure Python package never probes Meep, MPI or a C++ compiler. The optional
`meep` extra names the matching native distribution; build that distribution
from `native/meep` as shown above.
There is no separate `tama-fdtdx` runtime package to install.

## Documentation and checks

- [Installation and supported environments](docs/installation.md)
- [Backend API and migration](docs/backends.md)
- [FDTDX configuration, numerical conventions and limits](docs/fdtdx.md)
- [Meep design gradients](docs/design-gradients.md), [single target](docs/tda-objective.md), [multiple targets](docs/multi-tda-objective.md)
- [Tests and release checks](docs/running-and-testing.md)
- [Project layout](docs/project-layout.md)

FDTDX defaults to `sampling_interval=1` and **CPU offload OFF**
(`cpu_offload=False`; example CLI: `--cpu-offload off`). Each gradient evaluation
runs only the requested sampling interval. Sparse histories with
`sampling_interval>1` give approximate gradients whose error depends on sampling
and reconstruction. CPU offload changes history placement, not the FDTD engine.

## License and provenance

The integrated TAMA distribution is licensed under **GPL-3.0-or-later**;
see [LICENSE](LICENSE). The original TAMA component notices are retained, and
the separate `tama-meep-native` distribution remains `GPL-2.0-or-later`.
FDTDX (MIT), Meep (GPL-2.0-or-later), JAX and Equinox (Apache-2.0) remain
separate dependencies with their own licenses. See
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) for component provenance.
Citation metadata is in [CITATION.cff](CITATION.cff).
