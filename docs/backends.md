# Meep backend API

```python
import tama
tm = tama.get_backend("meep")
```

The result is the Meep backend module. Selection does not change global state.
Plain `import tama` does not import Meep. Root exports such as
`tama.TDAObjective` and `tama.DesignGrid` refer to Meep classes.
Compatibility Python submodule paths are aliases to the backend modules,
preserving class identity and monkeypatch behavior.

## Evaluation contract

```python
value, gradient = objective.evaluate(x, need_gradient=True)
value, no_gradient = objective.evaluate(x, need_gradient=False)
value = objective.fom(x)
value, gradient = objective.fom_and_grad(x)
value, gradient = objective(x)
```

The input is a flat real floating design vector. Outputs are a Python float and
flat NumPy gradient, or `None` for a forward-only evaluation. Scene definitions,
coordinates and units follow Meep conventions. See the
[design-gradient guide](design-gradients.md), [target guide](targets.md),
[single-target objective](tda-objective.md), and
[multiple-target objective](multi-tda-objective.md).

## Package layout

Meep source modules are under `tama.backends.meep`, with the aliases described
above. The compiled sampler is packaged as `tama_meep_native`. Build it from
`native/meep` after installing the core package. See [installation](installation.md).
