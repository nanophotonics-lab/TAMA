from typing import List, Tuple

import meep as mp


def centered_grid_coords(
    center: mp.Vector3,
    shape: Tuple[int, int],
    spacing: Tuple[float, float],
) -> Tuple[List[float], List[float]]:
    """Return regular 2D sampling-grid coordinates centered on a Meep `Vector3`.

    Args:
        center: Meep `Vector3` used as the physical center of the grid.
        shape: Number of grid points as `(nx, ny)`.
        spacing: Grid spacing as `(dx, dy)` in Meep length units.

    Returns:
        `(coords_x, coords_y)` where each item is a list of physical sample
        coordinates. These arrays can be passed directly to `FastFieldGrid`
        or `FastGradientGrid`.
    """
    nx, ny = shape
    dx, dy = spacing
    coords_x = [center.x + (i - (nx - 1) / 2) * dx for i in range(nx)]
    coords_y = [center.y + (j - (ny - 1) / 2) * dy for j in range(ny)]
    return coords_x, coords_y
