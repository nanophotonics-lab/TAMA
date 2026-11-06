# Python API

```python
import tama as tm
```

TAMA-M uses the Python package `tama`. Importing the package loads a lightweight
interface. Accessing exports such as `tm.TDAObjective` and `tm.DesignGrid` loads
their Meep modules. Direct module imports are also available:

```python
from tama.tda_objective import TDAObjective
from tama.specs import DesignGrid, SimulationSpec
```

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

The numerical modules are under `tama`, including `tama.tda_objective`,
`tama.multi_tda_objective`, and `tama.specs`.
The compiled sampler is packaged as `tama_meep_native`. Build it from
`native/meep` after installing the core package. See [installation](installation.md).
