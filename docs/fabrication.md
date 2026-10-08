# Periodic Fabrication Filter

This guide describes the Meep backend. See [backend selection](backends.md) and [FDTDX](fdtdx.md) for the other engine.

TAMA exposes a normalized conic density filter and its discrete transpose for
full-cell periodic `MaterialGrid` designs. Array axes follow `DesignGrid.shape`:
`(x, y)` in 2D and `(x, y, z)` in 3D. Reshape a flat optimizer vector in NumPy
C order before filtering.

```python
raw_grid = raw_design.reshape(design.shape)
filtered_grid = tm.periodic_conic_filter(
    raw_grid,
    radius=filter_radius,
    spacing=design.spacing,
    periodic_axes=(0, 1),
)
material_weights = tm.tanh_projection(
    filtered_grid,
    beta=projection_beta,
    eta=0.5,
)
value, material_gradient = objective.fom_and_grad(
    material_weights.ravel()
)
```

The `periodic_axes` are both filtered and wrapped. Axes not listed are neither
filtered nor wrapped. Consequently, `periodic_axes=(0, 1)` applies the same
x-y filter independently to each z layer of a 3D design. Use the same periodic
axes as the physical unit cell; normal-incidence x-y periodic simulations use
`SimulationSpec(k_point=mp.Vector3())` and z-directed PML.

For grid index offset `k` on a periodic axis of length `N` and spacing `h`, the
kernel uses the minimum-image, or toroidal, distance
`h * min(k, N - k)`. Its unnormalized weight is
`max(0, 1 - distance / radius)`, including the multidimensional Euclidean
distance when more than one axis is filtered. The complete discrete kernel is
normalized to sum to one. Circular convolution therefore preserves a constant
field and couples variables continuously across each unit-cell seam. This is a
discrete periodic filter; it does not zero-pad or renormalize an edge stencil.
The radius controls the smoothing support. By itself it does not certify a
minimum printable linewidth or spacing; that claim requires a chosen
projection/robust formulation and separate geometry checks.

## Gradient Chain Rule

`MultiTDAObjective` and `TDAObjective` return a gradient with respect to the
weights written into `MaterialGrid`. Pull it back in the reverse order of the
forward transformations:

```python
filtered_gradient = tm.tanh_projection_vjp(
    filtered_grid,
    material_gradient.reshape(design.shape),
    beta=projection_beta,
    eta=0.5,
)
raw_gradient = tm.periodic_conic_filter_vjp(
    filtered_gradient,
    radius=filter_radius,
    spacing=design.spacing,
    periodic_axes=(0, 1),
).ravel()
```

Thus the forward path is
`raw -> periodic conic filter -> tanh projection -> MaterialGrid`, and the
reverse path is
`MaterialGrid gradient -> projection VJP -> conic-filter VJP -> raw gradient`.
The conic filter is a real centrosymmetric circular convolution, but the VJP
still applies the conjugated Fourier transfer explicitly. These VJP functions
are the supported differentiation path; the NumPy FFT implementation is not an
`autograd.numpy` primitive.

The RGB example clips the projected weights to `[0, 1]` immediately before
the `MaterialGrid` update. The analytic filter/projection chain already maps a
bounded raw design into this interval; the clip only guards FFT roundoff at an
exact endpoint, so the documented VJP is unchanged in the interior.

`beta=0` (and numerically indistinguishable positive values below the
double-precision stability threshold) makes the projection and its VJP exact
identities. In this example, use `MaterialGrid(beta=0, do_averaging=False)`
because the external projection is applied before `DesignGrid.update_weights`.
The Cartesian native gradient also supports MaterialGrid's internal projection
and averaging. When using those options, TAMA includes their derivatives;
apply only the VJPs of mappings that you perform outside the MaterialGrid.

## RGB Router Example

[`examples/tda_3d_gamma_rgb_router.py`](../examples/tda_3d_gamma_rgb_router.py)
is a reduced normal-incidence 3D example. It performs a homogeneous no-device
forward calibration with the same source, time grid, temporal filters, and
full-cell flux quadrature. The resulting incident flux for each band is fixed
for all optimization iterations. The device objective contains all nine
band-detector pairs, not only the three desired diagonal terms, and reports
the incident-normalized 3-by-3 efficiency matrix through its scalarization
metadata. Its off-diagonal terms are signed transmitted-flux ratios. Their
squares enter the crosstalk penalty, so a negative local flux cannot improve
the objective.

Build the calibration and device objectives without running FDTD:

```bash
python examples/tda_3d_gamma_rgb_router.py --dry-run
```

Run the deliberately small optimization in serial or MPI:

```bash
python examples/tda_3d_gamma_rgb_router.py --iterations 1
mpirun -np 8 python examples/tda_3d_gamma_rgb_router.py --iterations 5
```

Pass `--output design.npz` to perform one final value-only evaluation and save
the matching raw, filtered, and projected densities together with the fixed
incident fluxes and 3-by-3 efficiency metrics. `--dry-run` never writes this
file.

The example uses coarse detector quadrature and a short run for API and
regression tests. Production results require convergence sweeps over the time
horizon, FIR length, spatial resolution, detector quadrature, filter radius,
and optimization schedule.
