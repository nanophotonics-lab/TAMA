# Troubleshooting

This guide describes the Meep backend. See [backend selection](backends.md) and [FDTDX](fdtdx.md) for the other engine.

If the Meep native sampler is unavailable or incompatible, rebuild its separate
package inside the active Meep/MPI environment:

```bash
cd <tama-source-directory>
python -m pip install --no-deps --no-build-isolation --no-cache-dir --force-reinstall ./native/meep
python -c "import tama; print(tama.__file__); print(tama.native_sampler_available())"
```

If `mpic++` is missing:

```bash
which mpic++
conda install -c conda-forge mpi4py mpich cxx-compiler
```

If MPI execution differs from serial, verify that regularizable adjoint sources
come only from `PointTarget`, additional non-source points are listed in
`protected_points`, and `SimulationSpec.chunk_layout` was left unset for the
automatic Meep-native topology path. Forward sources in
`SimulationSpec.sources` are protected automatically; list only external or
otherwise undiscoverable source supports in `source_volumes`. Inspect
`objective.last_source_boundary_decision.method` and
`objective.last_source_boundary_decision.reason`, and inspect
`objective.last_chunk_balance` after gradient evaluations.
