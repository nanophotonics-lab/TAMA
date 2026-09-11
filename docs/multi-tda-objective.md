# MultiTDAObjective

## Example

`MultiTDAObjective` is a multi-band temporal-convolution objective. It
evaluates several wavelength bands from one broadband time-domain run and
combines the resulting per-band FoMs through a user-defined scalarization
function.

Wavelengths use Meep length units with `c=1`, so a band
`(lambda_min, lambda_max)` maps to frequencies
`(1 / lambda_max, 1 / lambda_min)` in cycles per Meep time unit. TAMA uses
the Fourier convention `sum_n h[n] * exp(-2*pi*i*f*tau[n]) * dt` and NumPy's
`sinc(x) = sin(pi*x)/(pi*x)`. In the default independent mode, each real,
centered windowed-sinc FIR is normalized to unit response magnitude at the
band center. Floating histories are filtered by zero-padded FFT/IFFT linear
convolution. TAMA always retains all `N + K - 1`
samples for an `N`-sample raw history and a `K`-tap FIR, including both tails.
This is offline linear filtering, not circular convolution or a causal source
filter. The full output sample times are
`(j - (K - 1) / 2) * dt`, relative to the first raw sample. Odd kernels have
an integer-sample center; even kernels have a half-sample center. The filter
transpose returns exactly `N` raw-history covectors before the adjoint sources
are constructed. Flux and modal histories additionally apply the existing
electric/magnetic time centering, reducing the filtered history by one sample.

`t_final` specifies the physical FDTD recording duration, independent of the
FIR length. Filtering assumes zero field outside this recorded interval, so
the duration must still capture the source and device ringdown. Retaining
the FIR tails does not simulate missing physical fields or establish that a
shorter recording is converged. Sparse design-history alignment may extend
the actual endpoint to the next stored sample.

Reference normalization values must use the same recording duration and
filters as the device calculation. Custom callbacks receive the full history;
any explicit temporal gate must use the full output grid.

For ordered, contiguous wavelength bands, `filter_bank="power_complementary"`
instead constructs coordinated complementary target responses and fits each
odd-length FIR on a shared frequency grid. Adjacent filters use complementary
`sin^2`/`cos^2` target powers across each crossover, while the two outer tapers
lie outside the requested wavelength union. Because the FIRs are finite, this
is an approximately power-complementary bank rather than an exact ideal
spectral partition. `transition_width` is the total crossover width in
frequency units; its default is `2 / (kernel_length * dt)`. Check convergence
against a longer kernel or a dense-frequency reference for production use.

```python
import autograd.numpy as npa


def scalarization_fn(band_objectives):
    return npa.sum(band_objectives)


multi_tda = tm.MultiTDAObjective(
    design=design,
    simulation=simulation,
    targets=[
        tm.PointTarget(
            position=pos,
            component=mp.Ez,
            adjoint_source_size=adjoint_source_size,
            adjoint_source_amplitude=1.0,
        )
        for pos in monitor_positions
    ],
    t_final=T_f,
    wavelength_bands=[(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)],
    weights=band_weights,
    kernel_length=2001,
    kernel_window="hamming",
    scalarization_fn=scalarization_fn,
    reconstruction_window="kaiser",
    reconstruction_window_params={"beta": 6.0},
    reconstruction_half_width=64,
)
```

For a coordinated bank, keep filter amplitudes at one and put unequal band
priorities in the scalarization:

```python
multi_tda = tm.MultiTDAObjective(
    design=design,
    simulation=simulation,
    targets=targets,
    t_final=T_f,
    wavelength_bands=[(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)],
    weights=[1.0] * 4,
    kernel_length=2001,
    filter_bank="power_complementary",
    transition_width=0.08,
    scalarization_fn=scalarization_fn,
)
```

`scalarization_fn` defines how per-band FoMs become one scalar FoM. An
autograd-compatible function may return that scalar directly:

```python
total_fom = scalarization_fn(band_objectives)
```

When a design gradient is requested, `MultiTDAObjective` differentiates this
function with respect to `band_objectives` and uses the resulting coefficients
to scale each band contribution to the adjoint source. Use
`autograd.numpy` operations and do not cast the returned value to `float`.

For a non-autograd scalarization or to retain explicit derivative control,
return the legacy manual form:

```python
return total_fom, band_coeffs
```

Here, `band_coeffs` must equal the derivative of `total_fom` with respect to
each band FoM. Optional diagnostics may be included as a third item:

```python
return total_fom, band_coeffs, {"name": value}
```

`MultiTDAObjective` returns `(total_fom, d total_fom / d rho)` with the
maximization sign, matching `TDAObjective` and Meep adjoint's
`OptimizationProblem`. Negate both in the optimizer callback when driving a
minimizer such as nlopt or `scipy.optimize.minimize`.

`MultiTDAObjective` applies the exact transpose of each finite temporal filter
to its corresponding band adjoint source. This is the discrete adjoint of the
filtered objective and avoids cross-band terms. Point and regional targets
retain one adjoint FDTD run. A gradient evaluation containing a flux or
eigenmode target also uses one adjoint run by causally time-centering its
magnetic-source waveforms before combining them with the electric sources.

By default, each point-target band FoM is:

```text
0.5 * integral |u_filtered(t)|^2 dt
```

For a `FieldRegionTarget`, the same expression is summed over its configured
spatial weights. For a `FluxTarget`, the constituent E/H histories are filtered
independently before forming the real length-`N-1` power trace, and the default
band FoM is `sample_dt * sum(filtered_power)`. For an
`EigenmodeCoefficientTarget`, the filtered E/H histories are projected onto
its fixed mode and the default is `sample_dt * sum(|a_filtered|^2)`. Pass
`fom_fn(filtered_history, sample_dt)` to replace the applicable default
objective. Pass one callback for all bands or a band-length sequence whose
entries are callbacks or `None`; `None` keeps that band's default. Each
sequence entry follows the corresponding `wavelength_bands` and `targets`
entry. Each callback is applied before `scalarization_fn` combines the values.
Point histories are one-dimensional. Regional histories use time as the first
axis and `sample_shape` as the remaining axes. Flux and eigenmode histories are
one-dimensional and one sample shorter than their raw E/H histories. The
callback must use autograd-compatible operations unless
`adjoint_signal_fn(filtered_history, sample_dt)` supplies the corresponding
continuous-time adjoint signal manually. It accepts the same shared-callback
or band-length-sequence form. Every non-`None` entry requires the matching
`fom_fn`; these callbacks are band-local;
cross-band coupling is performed through `scalarization_fn`.
For complex histories, the manual signal must satisfy
`delta_FoM = sample_dt * real(sum(adjoint_signal * delta_filtered_history))`.

## Sparse Design-History Reconstruction

For both objective classes, set `max_frequency=f_max` to choose the design-grid
sampling interval from

```text
floor(1 / (2 * f_max * dt))
```

`f_max` must be strictly below the full-rate temporal Nyquist frequency
`1 / (2 * dt)`; otherwise no positive storage interval satisfies the stated
sampling contract and TAMA raises `ValueError`.

For `MultiTDAObjective`, `max_frequency` must be at least
`max(1 / lambda_min)` over all configured wavelength bands. The constructor
enforces this necessary lower bound.

The public helper `tm.nyquist_sampling_interval(f_max, dt)` returns the same
interval without constructing an objective.

`sampling_interval=K` is a dimensionless stride in completed FDTD steps. The
stored design-history spacing is therefore `K * sim.fields.dt`. Target-field
histories remain on `sim.fields.dt`, and value-only evaluation creates no
design history, so this setting affects only gradient evaluations.

Point, regional, and constituent flux/eigenmode-target fields, and MultiTDA
temporal filters, remain sampled at every Meep time step; only the stored
design-grid forward history is sparse.
During the adjoint run, TAMA reconstructs the forward field at every fine
time step with configurable finite-windowed sinc interpolation. The native
`DesignGrid` path then applies the same one-sided/centered time derivative used by its
full-rate path and contracts at the native Yee sites. For magnetic monitor
polarizations, adjacent adjoint electric fields are midpoint-centered on the
corresponding time grid. The default reconstruction is a Kaiser window with
beta 6 and a 64-sample half-width. Reconstructed rows are streamed rather than
materialized as a second full-rate history.

Nyquist's criterion determines a sufficient ideal sampling rate for a
band-limited history but does not select a finite interpolation kernel. The
finite sinc window used here is therefore a configurable numerical
approximation. The supported reconstruction windows match the
temporal-convolution window names, but the two settings are independent.
Their canonical names are `rectangular`, `hamming`, `hann`, `blackman`,
`bartlett`, and `kaiser`. Python `None`, `"none"`, and `"boxcar"` select
`rectangular`; `"hanning"` is an alias for `"hann"`.
Samples outside the recorded interval are treated as zero; the history is not
wrapped periodically. The forward run is extended by less than one sparse
interval when necessary so that its terminal sample lies on the sparse grid.
Fields should be negligible at the beginning and end of that extended record.
Include the source, objective, adjoint response, generated frequencies,
finite-window leakage, and a suitable safety margin in `f_max`.
`max_frequency` applies the interval formula but cannot verify these
assumptions; compare against `sampling_interval=1` before production runs.
`last_sampling_interval` reports the selected interval.

The same path is available through an explicit `sampling_interval > 1`, but
`max_frequency` is preferred because it states the physical bandwidth. Sparse
sampling uses the same exact per-band finite-FIR transpose as the full-rate
path.

## MPI Execution

With an MPI Meep build, `MultiTDAObjective` distributes independent band FIR
work when histories are non-deduplicated or when deduplicated histories use
the built-in per-band FoMs. Deduplicated histories with custom per-band FoM
callbacks instead evaluate every band on every rank. For supported
nonsymmetric 2D Cartesian, 3D Cartesian, and cylindrical simulations, the
native path records rank-local target-point histories and reduces the complete
history once per distinct target component after the forward run. Native
`DesignGrid` histories remain rank-local, and only the completed design
gradient is reduced. Adjoint source splines are always evaluated by native
piecewise-cubic callables. Exactly real source channels share one batched real
coefficient bank; channels containing any nonzero complex phase retain the
per-channel complex path.

Capability checks, reductions, and distributed band work use the active Meep
process group, including simulations created after
`mp.divide_parallel_processes(...)`; they do not require every world rank to
enter an objective evaluation.

`target_history_block_size=N` opts into a bounded spatial-column path for
direct Cartesian `SimulationSpec` runs whose targets are either all built-in
`FieldRegionTarget` entries or all built-in `FluxTarget` entries and whose
`fom_fn` and `adjoint_signal_fn` entries are all `None`. A monitor channel
supported by one rank is stored only on that rank. Chunk-boundary channels are
replicated; all E/H channels at a flux point are also replicated when their
unique owners do not agree. Globally unallocated, decoupled field components
are represented as implicit zeros and are neither stored nor sourced. Thus
rank `r` stores `n_time * (U_r + B)` target samples instead of `n_time * M`,
where `U_r` is its uniquely owned channel count, `B` is the replicated channel
count, and `M` is the global deduplicated channel count.
Filtering and filter pullbacks process at most `N` spatial positions at once.
The same bound applies independently to each of the four flux components.
Replicated channels are stored on every active rank. If every flux point lies
on a chunk boundary or has components with different owners, then `B = M` and
this option does not reduce the raw target-history allocation. Inspect
`last_target_history_layout` when selecting a production chunk layout.

This option does not change custom callbacks, point/eigenmode targets,
cylindrical runs, arbitrary factories, or mixed target sets; those cases use
the legacy full-history path. `last_target_history_mode` reports
`"distributed"` or `"legacy"`. For the distributed path,
`last_target_history_layout` reports `local_unique_channels`,
`replicated_channels`, `implicit_zero_channels`, `stored_channels`,
`total_channels`, `block_size`, and `peak_work_columns` (spatial positions,
not total E/H arrays), together with the exact initial `allocated_rows`,
`history_itemsize`, `allocated_history_bytes`, and legacy-path comparison
`legacy_history_bytes`. The compact raw and compact adjoint histories coexist
during the pullback; the raw history is released before source construction.
Adjoint source spline construction is block-staged, but the completed source
bank must still retain coefficients proportional to `n_time` times the
rank-local active source-channel count for the duration of the adjoint run.

## Diagnostic Helpers and Simulation Reuse

`MultiTDAObjective.filter_monitor_signals(signals)` is a low-level diagnostic
helper that applies the configured band filters after `dt` has been resolved.
It accepts a flat history with shape
`(N, total_raw_target_channels)` and returns `N + K - 1` rows.
The standalone `temporal_convolve_signal` helper also returns full convolution;
`temporal_convolve_signal_transpose` maps its covectors back to the original
`N` rows. The public
`auto_pixel_chunk` helper only preserves legacy pixel-block sizing behavior;
it does not control current objective-gradient memory.

`AdaptiveSourceBoundaryDecision` is the immutable result record available as
`objective.last_source_boundary_decision`. It reports the selected method,
layout, effective source sizes, amplitudes, and changed axes; users do not
normally construct it directly.

`reuse_simulation=True` is an opt-in for both objective classes that requires
`simulation=SimulationSpec(...)` with its direct `make` factory. It calls
`restart_fields()`, clears DFT monitors, and replaces the sources before the
adjoint run.
For a cylindrical objective, it first changes the angular mode from `m` to
`-m`.
Reuse cannot be enabled for arbitrary `sim_factory` inputs; those inputs use
separate Simulation instances.
Enable reuse only when geometry, materials, boundaries, and other relevant
simulation state are unchanged and restart-safe.

## Inputs

| Input | Meaning |
| --- | --- |
| `design` | `DesignGrid` describing the optimized MaterialGrid. Required when computing a gradient; optional for value-only evaluation. |
| `simulation` | Optional `SimulationSpec`. Fills the simulation factory and resolution; its native `DesignGrid` use requires `eps_averaging=False`. Its direct `make` path enables exact indexed target transposes and is required for regional, flux, and eigenmode gradients. It is also required for cylindrical gradients and supplies the one forward mode `m` shared by all bands. |
| `reuse_simulation` | Reuse the forward `Simulation` for the adjoint run when explicitly set to `True`. Default is `False`; enable only for restart-safe, time-invariant media. A cylindrical reuse changes the mode from `m` to `-m`. |
| `targets` | Optional list containing one `PointTarget`, `FieldRegionTarget`, `FluxTarget`, or `EigenmodeCoefficientTarget` per wavelength band. Exact targets may contain different point counts. Regional output shapes may also differ. Their gradients require direct `SimulationSpec` use. Flux and eigenmode targets combine electric and time-centered magnetic sources in one adjoint run. Cylindrical surface targets use radial or axial normals and explicit `2*pi*r` physical quadrature; cylindrical eigenmode targets additionally require fixed reference fields and a matching `reference_m`. Other targets may mix electric components or mix magnetic components, but not both groups in one adjoint run. |
| `update_design` | Same role as in `TDAObjective`: writes the design vector into the active Meep design object. |
| `sim_factory` | Same role as in `TDAObjective`: returns forward or adjoint `mp.Simulation` objects. Arbitrary factories use ordinary Cartesian `mp.Source` injection and do not support cylindrical gradient evaluation. |
| `t_final` | Physical FDTD recording duration. Choose it from source completion and response convergence, independently of filter length. |
| `monitor_positions` | Point-monitor locations, one per wavelength band. They may override bundled `PointTarget` positions, but cannot be combined with a regional, flux, or eigenmode target. The indexed path injects the exact transpose of the native Yee-grid interpolation. Cylindrical targets use `(r, 0, z)` with `r >= 0`, may lie on or near `r=0`, and include modal axis parity in the transpose. |
| `component` | Shared Meep monitor/source component. It may override bundled `PointTarget` components, but cannot be combined with a regional, flux, or eigenmode target. Otherwise non-surface targets may independently select components from the same electric or magnetic group. The native path maps 2D TMz to `Ez`, 2D TEz to `Ex`/`Ey`, 3D scalar-isotropic designs to `Ex`/`Ey`/`Ez`, and cylindrical targets according to the [design-gradient component table](design-gradients.md#supported-components). |
| `wavelength_bands` | List of `(lambda_min, lambda_max)` intervals. Each interval creates one bandpass temporal-convolution kernel. |
| `weights` | Per-band amplitude weights applied to monitor filtering and the matching adjoint filter. The power-complementary bank requires unit values; use `scalarization_fn` for unequal objective priorities. |
| `kernel_length` | Number of FIR taps. Larger values generally sharpen the spectral response and increase filtering work and temporary storage, but do not extend FDTD duration. |
| `kernel_window` | Window used to taper each independent bandpass kernel. Canonical names and accepted aliases are listed above. Default is `hamming`; it is not applied to the power-complementary bank. |
| `kernel_window_params` | Required as `{"beta": value}` for `kernel_window="kaiser"`; other windows accept no parameters. Applies only to independent filters. |
| `filter_bank` | `"independent"` (default) creates the existing per-band windowed-sinc FIRs. `"power_complementary"` constructs coordinated complementary target responses and fits each FIR on a shared frequency grid for at least two ordered, contiguous bands; it requires an odd kernel length and unit `weights`. |
| `transition_width` | Optional total crossover width in frequency units for `filter_bank="power_complementary"`. `None` selects `2 / (kernel_length * dt)` after the authoritative Meep time step is known. |
| `pixel_chunk` | Legacy compatibility option. Dense forward-field derivatives are now computed one time row at a time, so this value no longer controls gradient memory. |
| `target_chunks_per_rank` | Legacy compatibility option used only to resolve `pixel_chunk="auto"`. |
| `min_pixel_chunk`, `max_pixel_chunk` | Legacy compatibility bounds for automatic `pixel_chunk`. |
| `adjoint_source_size` | Shared Meep source size. It may override bundled `PointTarget` sizes, but cannot be combined with a regional, flux, or eigenmode target. Otherwise each point target may use its own size. The default `mp.Vector3()` selects exact indexed injection for direct `SimulationSpec` use. Explicit nonzero Cartesian sizes or `source_boundary_mode="finite"` use ordinary sources; eligible axes may then be regularized independently. Cylindrical gradients require zero size. |
| `adjoint_source_amplitude` | Shared logical source amplitude. It may override bundled `PointTarget` amplitudes, but cannot be combined with a regional, flux, or eigenmode target. Gradient evaluation requires every point/regional amplitude to be `1.0`; Cartesian adaptive finite-source fallback applies effective density normalization internally. Other finite values are accepted only for value-only evaluation. |
| `dt` | Legacy expected Meep time step. The initialized simulation's `fields.dt` is authoritative; an explicit value must match it and is otherwise only a fallback for simulation doubles without `fields.dt`. |
| `resolution` | Used to initialize filter metadata before evaluation when `dt` is omitted. The initialized simulation's `fields.dt` reconfigures the temporal filters and is authoritative. |
| `sampling_interval` | Dimensionless stride in Meep steps between stored design-grid forward-field samples. Target monitors, filters, and adjoint accumulation remain full rate. Default is `1`. |
| `max_frequency` | Optional maximum relevant frequency in Meep inverse-time units. It must cover the highest wavelength-band frequency, selects `sampling_interval` from the Nyquist criterion after `dt` is known, and cannot be combined with `sampling_interval > 1`; include a safety margin above all relevant content. |
| `reconstruction_window` | Window applied to the finite sinc reconstruction. Canonical names and accepted aliases are listed above. Default is `kaiser`. |
| `reconstruction_window_params` | Optional reconstruction-window parameters. Only `kaiser` accepts `{"beta": value}`; its default beta is `6`. |
| `reconstruction_half_width` | Number of sparse samples on each side of the finite sinc support. Default is `64`. |
| `fom_fn` | Optional scalar FoM applied as `fom_fn(filtered_history, sample_dt)`. Supply one callback for all bands or a band-length sequence of callbacks/`None`; `None` selects that target's default. Point history has shape `(time,)`; regional history uses time as the first axis and `sample_shape` as the remaining axes; flux history is a real `(time - 1,)` power trace; and eigenmode history is a complex `(time - 1,)` coefficient. |
| `adjoint_signal_fn` | Optional manual continuous-time bilinear covector in the same shared-callback or band-length-sequence form. Every supplied entry requires the matching `fom_fn` and must return the same shape as its callback history. A flux-history covector must be real. |
| `scalarization_fn` | User-defined scalarization function. An autograd-compatible function may return a scalar directly; its band coefficients are differentiated automatically. The legacy `(total_fom, band_coeffs)` and `(total_fom, band_coeffs, info)` forms provide manual derivatives. Default is the sum of the band objectives. |
| `history_dtype` | Real or complex floating dtype requested for temporary histories. Default `np.complex128` resolves to `np.float64` when Meep uses real forward fields; complex forward fields require a complex dtype. Native design histories are rank-local disk-backed memory maps, while target histories remain in memory. |
| `target_history_block_size` | Optional positive number of spatial positions processed per block. `None` (default) keeps the legacy target-history path. A value opts eligible direct Cartesian built-in regional/flux objectives into unique-owner MPI storage and blockwise filter/pullback processing; unsupported target or callback combinations fall back to the legacy path. |
| `chunk_balancer` | Default `"auto"` creates an `AdaptiveAdjointChunkBalancer` for a direct `SimulationSpec` with `chunk_layout=None`. It automatically protects ordinary forward sources, preserves Meep's Cartesian or cylindrical R-Z topology, and calibrates over three to eight gradient evaluations before freezing the best measured safe layout. Pass `None` to opt out or an instance for custom settings. Each indexed source is excluded from geometric constraints. Regional, flux, eigenmode, and cylindrical indexed sources do not support `source_boundary_mode="finite"`. |

Normalized per-band target settings are available as `target_components`,
`target_positions`, `target_sample_shapes`, `target_spatial_weights`,
`target_normals`,
`adjoint_source_sizes`, and `adjoint_source_amplitudes`. The legacy singular
attributes contain the common value when all bands share it and otherwise are
`None`.
