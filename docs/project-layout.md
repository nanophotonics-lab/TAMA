# Project layout

```text
TAMA-M/
├── pyproject.toml                 # pure Python tama distribution
├── native/meep/
│   ├── pyproject.toml             # tama-meep-native distribution
│   ├── CMakeLists.txt
│   ├── native_sampler.cpp         # C++ sampler
│   └── src/tama_meep_native/
├── src/tama/
│   ├── __init__.py                # lazy public API exports
│   ├── nyquist.py                 # reconstruction coefficients
│   ├── specs.py                   # scene and design specifications
│   ├── tda_objective.py           # single-target objective
│   ├── multi_tda_objective.py     # multiple-target objective
│   └── ...                       # sampling, sources, gradients, and MPI support
├── tests/common/
├── tests/meep/
├── examples/                     # Meep examples
└── docs/
```

Meep is installed separately. The Python package and native sampler use its
Python and C++ APIs. Numerical implementations reside directly in `src/tama`.
