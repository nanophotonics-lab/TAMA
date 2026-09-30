# Backend selection and migration

```python
import tama
tm = tama.get_backend("fdtdx")  # or "meep"
```

The result is a module; selection does not change a global engine. An unavailable
engine fails explicitly. Plain `import tama` imports neither Meep nor JAX.
Existing `tama.TDAObjective`, `tama.DesignGrid`, etc. remain Meep classes.
Legacy Python submodule paths are true aliases to the relocated Meep modules,
preserving class identity and monkeypatch behavior.

## Shared evaluation contract

```python
value, gradient = objective.evaluate(x, need_gradient=True)
value, no_gradient = objective.evaluate(x, need_gradient=False)
value = objective.fom(x)
value, gradient = objective.fom_and_grad(x)
value, gradient = objective(x)
```

The common input is a flat real floating design vector. Outputs are a Python
float and flat NumPy gradient, or `None` for a forward-only evaluation.
The FDTDX adapter also accepts the exact native parameter shape, while returning
a flat gradient. Input validation, sparse accuracy errors and the actual engine's
numerical operations remain active. No alternate engine is tried on failure.

## Configuration boundaries

| Setting | Meep | FDTDX |
|---|---|---|
| Scene | Meep geometry, sources and Simulation | Placed native arrays, objects and config |
| Coordinates | Existing Meep coordinate conventions | Native Cartesian integer grid indices |
| Frequency/time | Meep unit system | Hz and seconds |
| Design | Existing MaterialGrid interface | Scalar epsilon density on one rectangular native region |
| Bands | Existing wavelength/filter-bank API | Explicit Band objects and FIR kernels |
| Weights | Existing filter amplitude weights | Objective priority weights |
| Callbacks | Existing target-specific traces | JAX scalar from history shaped (Nt, N, 6) and dt |
| Mode overlap | Existing Meep mode convention | Supplied fixed native profiles; no mode solver |
| Parallelism | Existing MPI paths | Single CUDA device |

Scene inputs are not converted between engines. Same-named targets do not imply
identical power normalization. Unsupported constructor options are rejected,
not silently ignored. FDTDX-native support of a forward physical model does
not establish support of its TAMA gradient.

## FDTDX objective construction

After constructing the native scene, design, targets and bands as described in
the [FDTDX guide](fdtdx.md), use the following constructors:

```python
objective = tm.TDAObjective(
    simulation=spec, design=design, target=target,
    sampling_interval=1, cpu_offload=False,
)
multi = tm.MultiTDAObjective(
    simulation=spec, design=design, targets=targets, bands=bands,
    scalarization_fn=scalarize, cpu_offload=False,
)
value, gradient = multi.evaluate(x, cpu_offload=True)
diagnostics = multi.last_info
validation = multi.validate(x, directions=3)
```

CPU offload defaults to `False`. Set it on the constructor or override it for a
gradient evaluation as shown above. On CUDA, the dense reference used to verify
`sampling_interval>1` always offloads its history, independently of that setting.

`sampling_interval` maps to the native history stride and
`reconstruction_half_width` maps to the sinc half-width. `native_objective`
exposes a read-only reference to the internal implementation for diagnostics;
it is not the cross-backend API. `last_info` clears on forward-only evaluations.
`reference_value_and_grad(x)` uses full-state AD and returns the same flat
gradient contract. Do not wrap the whole objective in outer `jax.jit` or
`jax.grad`; Python controls histories, validation and offload.

## Migrating source and install commands

Meep source modules moved to `tama.backends.meep` but legacy aliases remain.
The original examples still run from their original paths. The compiled sampler
now belongs to `tama_meep_native`; build it from `native/meep` after installing
the core package. The former root CMake build is replaced by this subproject.

FDTDX code is now part of TAMA; replace `import tama_fdtdx as tm` with explicit
backend selection and use the keyword constructors above. The common adapter
returns flat NumPy gradients rather than the standalone implementation's shaped
JAX arrays. See [installation](installation.md) and [FDTDX conventions](fdtdx.md).
