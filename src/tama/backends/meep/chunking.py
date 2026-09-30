from typing import Optional, Sequence, Tuple

import meep as mp
import numpy as np


_AXIS_NAMES = ("x", "y", "z")


def _axis_name(axis) -> str:
    if isinstance(axis, str):
        axis = axis.lower()
        if axis in _AXIS_NAMES:
            return axis
    else:
        if axis == 0 or axis == mp.X:
            return "x"
        if axis == 1 or axis == mp.Y:
            return "y"
        if axis == 2 or axis == mp.Z:
            return "z"
    raise ValueError("source axes must be x/y/z, 0/1/2, or mp.X/mp.Y/mp.Z")


def _normalize_axis_names(axes: Sequence, dimensions: int) -> Tuple[str, ...]:
    axis_names = []
    for axis in axes:
        axis_name = _axis_name(axis)
        axis_index = _AXIS_NAMES.index(axis_name)
        if axis_index >= dimensions:
            raise ValueError("source axis is outside the simulated dimensions")
        if axis_name not in axis_names:
            axis_names.append(axis_name)
    return tuple(axis_names)


def _vector_from_coords(coords) -> mp.Vector3:
    return mp.Vector3(float(coords[0]), float(coords[1]), float(coords[2]))


def _meep_grid_axis_boundaries(
    *,
    length: float,
    center: float,
    resolution: float,
) -> np.ndarray:
    """Return the actual Meep grid-cell boundaries for one Cartesian axis."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    axis_grid = mp.vol1d(length, resolution)
    axis_grid.center_origin()
    axis_grid.shift_origin(mp.vec(center))
    n_cells = axis_grid.num_direction(mp.Z)
    grid_low = axis_grid.surroundings().get_min_corner().z()
    return grid_low + np.arange(n_cells + 1, dtype=float) / resolution


def regularize_source_size_and_amplitude(
    source_size: mp.Vector3,
    source_amplitude: complex,
    *,
    dimensions: int,
    resolution: float,
    width_cells: float = 2.0,
    axes: Optional[Sequence] = None,
) -> Tuple[mp.Vector3, complex, Tuple[str, ...]]:
    """Replace zero-size source axes by finite width and renormalize amplitude.

    Args:
        source_size: Original Meep source size.
        source_amplitude: Original source amplitude.
        dimensions: Simulated dimensionality, 1, 2, or 3.
        resolution: Meep spatial resolution.
        width_cells: Finite source width in grid cells.
        axes: Axes to regularize. If omitted, all simulated axes are checked.

    Returns:
        `(new_size, new_amplitude, changed_axes)`. Amplitude is divided by
        the product of newly added finite widths so that the integrated source
        weight is preserved for the changed axes.
    """
    if dimensions not in (1, 2, 3):
        raise ValueError("dimensions must be 1, 2, or 3")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if width_cells <= 0:
        raise ValueError("width_cells must be positive")

    width = width_cells / resolution
    coords = [float(source_size.x), float(source_size.y), float(source_size.z)]
    target_axes = (
        tuple(_AXIS_NAMES[:dimensions])
        if axes is None
        else _normalize_axis_names(axes, dimensions)
    )
    scale = 1.0
    changed = []
    for axis_name in target_axes:
        axis_index = _AXIS_NAMES.index(axis_name)
        if coords[axis_index] == 0.0:
            coords[axis_index] = width
            scale *= width
            changed.append(axis_name)

    if scale != 1.0:
        source_amplitude = source_amplitude / scale
    return _vector_from_coords(coords), source_amplitude, tuple(changed)
