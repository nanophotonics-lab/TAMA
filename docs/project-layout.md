# Project layout

```text
TAMA/
├── pyproject.toml                 # pure Python tama distribution
├── native/meep/
│   ├── pyproject.toml             # tama-meep-native distribution
│   ├── CMakeLists.txt
│   ├── native_sampler.cpp         # original C++ calculation code
│   └── src/tama_meep_native/
├── src/tama/
│   ├── __init__.py                # lazy legacy exports and get_backend
│   ├── nyquist.py                 # shared reconstruction coefficients
│   ├── backends/meep/             # original Meep implementation
│   ├── backends/fdtdx/            # embedded FDTDX implementation and adapter
│   └── ...                       # compatibility aliases for old module paths
├── tests/common/
├── tests/meep/
├── tests/fdtdx/
├── examples/fdtdx/
├── examples/                     # existing Meep examples retain their paths
└── docs/
```

FDTDX and Meep engine implementations are separately installed dependencies.
TAMA does not vendor or patch their engine sources. There is one implementation
of each backend and one shared Nyquist helper; legacy modules are aliases, not
copies of the Meep algorithms. Backend-specific filters keep their native
NumPy/JAX differentiation semantics.
