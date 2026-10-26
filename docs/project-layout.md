# Project layout

```text
TAMA/
├── pyproject.toml                 # pure Python tama distribution
├── native/meep/
│   ├── pyproject.toml             # tama-meep-native distribution
│   ├── CMakeLists.txt
│   ├── native_sampler.cpp         # C++ sampler
│   └── src/tama_meep_native/
├── src/tama/
│   ├── __init__.py                # lazy compatibility exports and get_backend
│   ├── nyquist.py                 # reconstruction coefficients
│   ├── backends/meep/             # Meep backend implementation
│   └── ...                       # compatibility aliases for Meep module paths
├── tests/common/
├── tests/meep/
├── examples/                     # Meep examples
└── docs/
```

Meep is installed separately. The Python package and native sampler use its
Python and C++ APIs. Compatibility modules are aliases to the Meep backend
modules, not copies of the algorithms.
