# Project Layout

## Files

The principal source-release files are:

```text
TAMA/
├── README.md
├── LICENSE
├── CITATION.cff
├── CMakeLists.txt
├── pyproject.toml
├── environment.yml
├── build_native_sampler.sh
├── docs/
│   ├── conf.py
│   ├── index.md
│   ├── installation.md
│   ├── project-layout.md
│   ├── design-gradients.md
│   ├── fabrication.md
│   ├── targets.md
│   ├── tda-objective.md
│   ├── multi-tda-objective.md
│   ├── mpi.md
│   ├── running-and-testing.md
│   └── troubleshooting.md
├── examples/
│   ├── tda_3d_projected_gradient.py
│   ├── tda_3d_gamma_rgb_router.py
│   └── tda_cylindrical_mode.py
├── src/
│   └── tama/
│       ├── __init__.py
│       ├── adaptive_chunking.py
│       ├── chunking.py
│       ├── coords.py
│       ├── eigenmode.py
│       ├── fabrication.py
│       ├── flux.py
│       ├── fastmeep_grid.py  # compatibility shim
│       ├── native_design.py
│       ├── native_sampler.cpp
│       ├── native_sampler.py  # missing-extension error marker
│       ├── native_sampler.pyi
│       ├── nyquist.py
│       ├── sampling_grid.py
│       ├── specs.py
│       ├── multi_tda_objective.py
│       ├── objectives.py
│       ├── py.typed
│       └── tda_objective.py
└── tests/
```

GitHub source releases contain the Python sources and native-sampler source
code. Build the native sampler after cloning the repository or unpacking a
source release.
