# Backend validation

This historical report records validation of TAMA `0.6.0a1` at revision
`0b66b8a`. Its preservation statements and test counts describe that revision.
The numerical results below were collected on 2026-09-30. The baseline revision is
`587c3afd063cf26ca3547607d3939c91cfa78c75`. The change reorganizes TAMA around
explicit Meep and FDTDX backends, preserves the existing numerical code, and
separates the Meep native build from the pure Python distribution.

The release installation checks were repeated on 2026-10-01 after updating the
license metadata and CI failure handling. Meep/common tests passed 864 cases
with 60 skips; MPI tests passed 68, 11 and 4 cases per rank at two, four and
eight ranks respectively. The combined-engine installation passed 11
common/FDTDX CPU tests and 14 selected Meep numerical tests. The numerical
Python implementations and native C++ sampler were unchanged.

## Preservation and installation

The 13 relocated Meep Python implementations preserve their numerical AST;
three modules only change the relative import of the shared Nyquist helper.
The native C++ sampler is unchanged. Legacy Python modules alias the relocated
module objects, preserving class identity and monkeypatch behavior.

FDTDX `specs.py`, `targets.py`, `filters.py` and the engine hash manifest match
the preceding standalone backend byte for byte. The numerical AST of
`backend.py` differs only in its Nyquist import. The shared Nyquist helper has
the same text and AST after normalizing line endings. The public adapter retains
the preceding integration's behavior, with local package imports.

The core source distribution was unpacked and built into a
`tama-0.6.0a1-py3-none-any.whl` with deliberately invalid compiler and Meep
prefix settings. The wheel contains both backend implementations, type markers,
the FDTDX source manifest and component license notices, and no native binary.
It does not depend on the former standalone `tama-fdtdx` distribution.
The native source distribution separately produced a CPython 3.13 Linux x86-64
wheel with relative RPATH and the original sampler implementation. That native
wheel is specific to the tested Meep/MPI environment.

Package tests check that plain `import tama` loads neither engine, selection
reports a missing engine, backend objects remain distinct, and old Meep imports
retain their identities. Selecting FDTDX does not change the global JAX precision
setting. Users configure precision before constructing a scene.

## Meep results

Installed core and native wheels were tested with the tests shipped in the
unpacked core source distribution. The environment used Python 3.13.15,
PyMeep 1.34.0, MPICH 4.3.2, mpi4py 4.1.2 and NumPy 2.5.2 under WSL Ubuntu.

| Run | Result |
|---|---|
| Serial Meep and common package tests | 863 passed, 60 skipped |
| MPI, two ranks | 68 passed per rank |
| MPI, four ranks | 11 passed per rank |
| MPI, eight ranks | 4 passed per rank |

The serial suite includes Cartesian and cylindrical design-gradient regression
checks. Existing tolerances were retained. The serial skips remain skips and are
not counted as successful numerical checks. MPI groups were executed separately;
per-rank counts do not represent additional distinct test cases.

After this full run, engine selection gained an explicit FDTDX dependency check
and an additional import-isolation regression. The Meep numerical implementation
and native binary did not change. The updated selector was tested again in the
combined installation described below; the 863 count above is the actual earlier
run, not an extrapolated count for the additional test.

## FDTDX GPU results

The installed pure wheel was tested with Python 3.12.13, FDTDX 0.6.2,
JAX/JAXlib 0.11.0, Equinox 0.13.8 and NumPy 2.4.6 on an NVIDIA GeForce RTX 4080
under WSL. The CUDA 12 plugin/PJRT versions were 0.11.0. All eleven GPU cases
passed: main, periodic, Gaussian source, dipole source, float32, fabrication
transform, sparse history, PEC, PMC, complex Bloch and complex64 Bloch.
The engine's 100-file source-hash check passed in every case.

Errors below are the largest recorded relative errors across the indicated
cases. Native AD includes both the initial and changed density. Full-state step
AD is a separate reference. Finite differences are directional centered checks,
not a componentwise proof for the full parameter space.

| Gradient comparison | float64/complex128, nine cases | float32/complex64, two cases |
|---|---:|---:|
| Native AD, initial and changed design | 2.365e-16 | 1.209e-7 |
| Full-state step AD | 2.258e-16 | 1.227e-7 |
| Centered finite difference | 9.808e-10 | 7.579e-4 |
| CPU offload ON versus OFF | 2.000e-16 | 9.009e-8 |

Objective values matched their native references and offload counterparts in
these cases. Gradient offload agreement is at the errors shown; it is not a
general claim of bitwise equality.

The public adapter's installed-wheel GPU test also passed. Direct-backend and
single-target comparisons had zero recorded value/gradient error. The
full-state AD gradient relative error was 1.772e-16; the maximum directional
finite-difference error was 8.218e-10. This public test was repeated after the
final selector correction and again passed, without importing Meep.

At `0b66b8a`, sparse histories retained a per-design dense check with `rtol=1e-3`
and `atol=0`. The public test accepted M=2 at relative gradient error 3.986e-5
and rejected M=8 at 0.106314. The internal test accepted M=2 and M=4, and
rejected M=8 at 0.111648. That automatic comparison and rejection have since
been removed; current evaluations run only the requested stride. The recorded
errors remain results for those scenes, not generally safe stride recommendations
or validation of the removal. M=1 remains the default.

Four CPU checks (specifications, functionals, API and adapter) passed. They
include rejection of non-finite derivatives and a deliberately incorrect
stop-gradient callback whose directional finite-difference error was 1.0.
The internal optimization example reduced its objective from
9.004230645495918 to 0.7440091155118383 in three iterations. It stopped at the
iteration limit; this does not establish convergence to an optimum.

The public two-band example was also run on the GPU with CPU offload enabled.
Its minimized objective is the negative summed band response. It changed from
-6.942303044079483 to -17.25695487823588 in three L-BFGS-B iterations and five
evaluations, again stopping at the iteration limit. Its initial validation had
AD relative gradient error 1.391e-16 and maximum directional finite-difference
error 1.207e-9. The scene, optimizer call and JSON output are available through
`examples/fdtdx/optimize_broadband.py`.

## Combined installation and documentation

A separate Python 3.13 environment installed the core wheel with its FDTDX
extra and the native Meep wheel through the normal dependency resolver. It used
the same engine versions above and NumPy 2.4.6. Both backends loaded in the same
process; their objective classes remained distinct, legacy Meep class identity
was preserved, the native sampler was available, and the JAX precision setting
was unchanged by selection. The former standalone backend was not imported.

The combined environment passed 11 common/FDTDX CPU tests (12 GPU tests
deselected) and 14 selected Meep filtered-objective and full-vector 3D
finite-difference tests. This also exercised the final missing-engine
regression added after the full Meep run. Sphinx built the documentation with
warnings treated as errors.

## Scope of the evidence

These checks support preservation of the tested discrete objective and gradient
calculations. They do not establish physical equivalence between Meep and FDTDX,
mesh/time/domain convergence, or correctness for every possible geometry and
user-supplied objective. Units, source conventions and mode normalization must
be reconciled for any cross-engine comparison.

GPU history offload is implemented and numerically checked. The recorded
`history_bytes` is not peak process VRAM, and the collected ON/OFF timings have
different compilation/cache histories. This report makes no new speed or peak
VRAM claim. A dedicated performance measurement needs separate initialized
processes, warm repeats and GPU/host peak measurements.

FDTDX remains limited to its documented native Cartesian configurations and a
single rectangular scalar, nondispersive, lossless epsilon design region.
Unsupported physics, coordinates and execution modes are listed in
[the FDTDX guide](fdtdx.md). No cylindrical emulation, multi-GPU support or new
material model is introduced by this refactor.

GitHub workflows cover the configured Python and NumPy matrix. Per-revision
results are available in the repository's Actions tab. The manual GPU job
requires a provisioned self-hosted CUDA runner. The local results above do not
substitute for a successful workflow run on a particular release revision.

The FDTDX validation runner writes JSON and logs to the directory selected with
`--output`; GitHub's GPU workflow retains its test evidence as an artifact.
Source preservation, installed Meep runs, FDTDX cases and API integration checks
have separate scopes and should be interpreted accordingly.
