# Changelog

## 0.6.0a1 — integrated Meep and FDTDX alpha

This alpha introduces the combined backend package following TAMA 0.5.0a3.

- Add explicit Meep/FDTDX selection with a shared flat-vector evaluation API.
- Package both Python backends in the pure Python TAMA distribution.
- Split the unchanged Meep native sampler into the tama-meep-native subproject.
- Preserve old Meep exports and Python submodules through lazy aliases.
- Integrate the TAMA FDTDX backend, examples and numerical checks; remove the separate
  tama-fdtdx runtime dependency and duplicate Nyquist implementation.
- Retain strict engine pins/source checks, M=1 default, and CPU history offload
  on/off with OFF as the default.
- Run sparse-history gradients only at the requested stride; remove automatic
  per-design dense comparisons.
- Separate common, Meep and FDTDX tests and provide installed-artifact CPU,
  MPI and manual CUDA workflow paths.
- License the integrated TAMA distribution under GPL-3.0-or-later, retain
  existing component notices, and keep the separate Meep native sampler
  distribution under GPL-2.0-or-later.

Install migration: build native/meep explicitly in Meep environments. FDTDX
users install the root fdtdx extra and configure CUDA JAX separately as needed.
Physical models and engine-specific unit/normalization conventions are unchanged.
