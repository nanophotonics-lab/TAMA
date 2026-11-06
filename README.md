# TAMA-M: Time-domain Adjoint with Memory-efficient Acceleration for Meep

TAMA-M provides time-domain adjoint optimization with **Meep**. Version `0.7.0a1`
is an alpha release.

Meep supplies the FDTD calculations. The Meep native sampler is built separately
inside the active Meep/MPI environment.

```python
import tama as tm
```

Use the complete [projected-gradient example](examples/tda_3d_projected_gradient.py)
to construct and optimize a scene.

The Python package is named `tama`. `import tama` loads a lightweight interface.
Accessing classes such as `tm.TDAObjective` and `tm.DesignGrid` loads their Meep modules.

## Install from this repository

Clone the repository, then run the installation commands from its root:

```bash
git clone https://github.com/nanophotonics-lab/TAMA.git
cd TAMA
```

For Meep, keep Python, PyMeep, MPI and the compiler in the same conda environment:

```bash
conda env create -f environment.yml
conda activate tama
python -m pip install .
python -m pip install --no-build-isolation ./native/meep
python -c "import tama; print(tama.native_sampler_available())"
python examples/tda_3d_projected_gradient.py --iterations 1
```

The pure Python package never probes Meep, MPI or a C++ compiler. The optional
`meep` extra names the matching native distribution; build that distribution
from `native/meep` as shown above.

## Documentation and checks

- [Installation and supported environments](https://tama-adjoint.readthedocs.io/en/latest/installation.html)
- [Python API](https://tama-adjoint.readthedocs.io/en/latest/api.html)
- [Meep design gradients](https://tama-adjoint.readthedocs.io/en/latest/design-gradients.html), [single target](https://tama-adjoint.readthedocs.io/en/latest/tda-objective.html), [multiple targets](https://tama-adjoint.readthedocs.io/en/latest/multi-tda-objective.html)
- [Tests and release checks](https://tama-adjoint.readthedocs.io/en/latest/running-and-testing.html)
- [Project layout](https://tama-adjoint.readthedocs.io/en/latest/project-layout.html)

## License and provenance

The TAMA-M Python package and `tama-meep-native` distribution are licensed under
**GPL-3.0-only**; see [LICENSE](LICENSE).
Meep (GPL-2.0-or-later) is a separate dependency. See
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) for component provenance.
Citation metadata is in [CITATION.cff](CITATION.cff).
