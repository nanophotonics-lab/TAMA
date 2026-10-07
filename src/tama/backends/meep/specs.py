from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
from typing import Callable, Optional, Sequence, Tuple, Union

import meep as mp
import numpy as np

from .coords import centered_grid_coords
from .flux import _normalize_axis_normal
from .objectives import _epsilon_from_medium


@dataclass
class DesignGrid:
    """Bundle a Meep MaterialGrid with TAMA design-grid metadata.

    Args:
        material_grid: Meep `MaterialGrid` whose weights are optimized.
        center: Physical center of the design region.
        size: Physical size of the design region.
        shape: Number of design variables as `(nx, ny)` for 2D Cartesian,
            `(nx, ny, nz)` for 3D Cartesian, or `(nr, nz)` for
            cylindrical coordinates.
        background: Background medium used to infer `d epsilon / d rho`.
        design_material: Design medium used to infer `d epsilon / d rho`.
        material_factor: Explicit scalar `d epsilon / d rho`. If omitted,
            `background` and `design_material` are required. The Cartesian
            tensor/projection/averaging path instead derives its material
            Jacobian directly from the MaterialGrid endpoints.
        coordinate_system: `cartesian` (default) or `cylindrical`.
            Cylindrical grids use public shape `(nr, nz)` and require a Meep
            MaterialGrid with grid size `(nr, 1, nz)`.

    Attributes:
        spacing: Nominal MaterialGrid variable spacing as `(dx, dy)` or
            `(dx, dy, dz)` for Cartesian grids, and `(dr, dz)` for
            cylindrical grids.
        coords_x: Nominal MaterialGrid variable-center x coordinates.
        coords_y: Nominal MaterialGrid variable-center y coordinates.
        coords_z: Nominal MaterialGrid variable-center z coordinates for a 3D
            Cartesian or cylindrical design, otherwise `None`.
        coords_r: Nominal radial variable-center coordinates for a cylindrical
            design, otherwise `None`.
        cell_area: Nominal 2D coordinate-plane area represented by one design
            variable. For cylindrical grids, this is `dr * dz` rather than
            the physical volume measure.
        cell_volume: Nominal 3D volume represented by one design variable,
            otherwise `None`.
    """

    material_grid: mp.MaterialGrid
    center: mp.Vector3
    size: mp.Vector3
    shape: Tuple[int, ...]
    background: Optional[mp.Medium] = None
    design_material: Optional[mp.Medium] = None
    material_factor: Optional[float] = None
    coordinate_system: str = "cartesian"

    @property
    def is_cylindrical(self) -> bool:
        """Whether this design grid uses cylindrical `(r, z)` coordinates."""
        return self.coordinate_system == "cylindrical"

    def __post_init__(self) -> None:
        if self.coordinate_system not in ("cartesian", "cylindrical"):
            raise ValueError(
                "DesignGrid coordinate_system must be 'cartesian' or 'cylindrical'"
            )
        self.shape = tuple(self.shape)
        if self.is_cylindrical and len(self.shape) != 2:
            raise ValueError("cylindrical DesignGrid shape must be (nr, nz)")
        if not self.is_cylindrical and len(self.shape) not in (2, 3):
            raise ValueError("DesignGrid shape must be (nx, ny) or (nx, ny, nz)")
        if any(
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or count <= 0
            for count in self.shape
        ):
            raise ValueError("DesignGrid shape entries must be positive integers")
        grid_size = self.material_grid.grid_size
        if self.is_cylindrical:
            nr, nz = self.shape
            if (
                int(grid_size.x) != nr
                or int(grid_size.y) != 1
                or int(grid_size.z) != nz
            ):
                raise ValueError(
                    "cylindrical DesignGrid shape must match "
                    "MaterialGrid.grid_size=(nr, 1, nz)"
                )
            if self.size.x <= 0 or self.size.z <= 0:
                raise ValueError(
                    "cylindrical DesignGrid sizes in r and z must be positive"
                )
            if self.center.x - 0.5 * self.size.x < -1.0e-12:
                raise ValueError(
                    "cylindrical DesignGrid must not extend below r=0"
                )
            self.dimensions = 2
            self.spacing = (self.size.x / nr, self.size.z / nz)
            self.coords_r = [
                self.center.x + (i - (nr - 1) / 2) * self.spacing[0]
                for i in range(nr)
            ]
            self.coords_x = self.coords_r
            self.coords_y = None
            self.coords_z = [
                self.center.z + (k - (nz - 1) / 2) * self.spacing[1]
                for k in range(nz)
            ]
        else:
            nx, ny = self.shape[:2]
            nz = self.shape[2] if len(self.shape) == 3 else 1
            if (
                int(grid_size.x) != nx
                or int(grid_size.y) != ny
                or int(grid_size.z) != nz
            ):
                raise ValueError("DesignGrid shape must match MaterialGrid.grid_size")
            if self.size.x <= 0 or self.size.y <= 0 or (
                len(self.shape) == 3 and self.size.z <= 0
            ):
                raise ValueError(
                    "DesignGrid sizes in active dimensions must be positive"
                )
            self.dimensions = len(self.shape)
            self.spacing = (self.size.x / nx, self.size.y / ny)
            if self.dimensions == 3:
                self.spacing += (self.size.z / nz,)
            self.coords_x, self.coords_y = centered_grid_coords(
                center=self.center,
                shape=(nx, ny),
                spacing=self.spacing[:2],
            )
            self.coords_z = (
                [
                    self.center.z + (k - (nz - 1) / 2) * self.spacing[2]
                    for k in range(nz)
                ]
                if self.dimensions == 3
                else None
            )
            self.coords_r = None
        self.cell_area = self.spacing[0] * self.spacing[1]
        self.cell_volume = (
            self.cell_area * self.spacing[2]
            if self.dimensions == 3
            else None
        )

        if self.material_factor is None:
            if self.background is None or self.design_material is None:
                raise ValueError("DesignGrid requires material_factor or both background and design_material")
            self.material_factor = (
                _epsilon_from_medium(self.design_material)
                - _epsilon_from_medium(self.background)
            )

    def update_weights(self, x: np.ndarray) -> None:
        """Write a flat design vector into the bundled Meep MaterialGrid.

        Args:
            x: Flat design vector with `numpy.prod(shape)` entries.
        """
        shape = (
            (self.shape[0], 1, self.shape[1])
            if self.is_cylindrical
            else self.shape
        )
        self.material_grid.update_weights(np.asarray(x).reshape(shape))


@dataclass
class SimulationSpec:
    """Reusable Meep `Simulation` construction arguments.

    Args:
        cell_size: Meep simulation cell size.
        boundary_layers: Boundary layers passed to `mp.Simulation`.
        geometry: Geometry objects passed to `mp.Simulation`.
        sources: Default source list, or a zero-argument callable returning one.
        resolution: Meep spatial resolution.
        courant: Meep Courant factor. Defaults to Meep's default `0.5`.
        geometry_center: Optional Meep `geometry_center`.
        chunk_layout: Optional MPI chunk layout passed to Meep. Leaving this
            unset enables the objectives' default automatic mixed-topology
            chunk balancing; an explicit layout remains user-managed.
        dimensions: Optional simulation dimensionality, including
            `mp.CYLINDRICAL`.
        eps_averaging: Optional Meep subpixel averaging flag.
        k_point: Meep boundary-condition selector. The default `False` keeps
            Meep's perfect-metal boundary behavior. Use `mp.Vector3()` for
            normal-incidence Gamma-point periodic boundaries. Nonzero Bloch
            wavevectors are not yet supported by the high-level adjoint path.
        m: Integer angular mode number for cylindrical simulations.
            `TDAObjective` and `MultiTDAObjective` gradients currently support
            `m=-1`, `0`, and `+1`.
        symmetries: Meep `Mirror` objects. Geometry, design updates, and sources
            must obey the specified parity.
    """

    cell_size: mp.Vector3
    boundary_layers: Sequence = ()
    geometry: Sequence = ()
    sources: Union[Sequence, Callable[[], Sequence]] = ()
    resolution: float = 10
    geometry_center: Optional[mp.Vector3] = None
    chunk_layout: Optional[mp.BinaryPartition] = None
    dimensions: Optional[int] = None
    eps_averaging: Optional[bool] = None
    m: int = 0
    courant: float = 0.5
    k_point: Union[bool, mp.Vector3] = False
    symmetries: Sequence = ()

    def __post_init__(self) -> None:
        self.symmetries = tuple(self.symmetries)
        for symmetry in self.symmetries:
            if not isinstance(symmetry, mp.Mirror):
                raise ValueError("symmetries currently support mp.Mirror only")
            if symmetry.phase not in (-1, 1):
                raise ValueError("Mirror phase must be +1 or -1")
        if isinstance(self.resolution, (bool, np.bool_)):
            raise ValueError("resolution must be a positive finite number")
        self.resolution = float(self.resolution)
        if not np.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("resolution must be a positive finite number")
        if isinstance(self.dimensions, (bool, np.bool_)) or self.dimensions not in (
            None,
            1,
            2,
            3,
            mp.CYLINDRICAL,
        ):
            raise ValueError(
                "dimensions must be None, 1, 2, 3, or mp.CYLINDRICAL"
            )
        if isinstance(self.courant, (bool, np.bool_)):
            raise ValueError("courant must be a positive finite number")
        self.courant = float(self.courant)
        if not np.isfinite(self.courant) or self.courant <= 0:
            raise ValueError("courant must be a positive finite number")
        if self.k_point is not False:
            if not isinstance(self.k_point, mp.Vector3):
                raise TypeError("k_point must be False or a Meep Vector3")
            coordinates = np.asarray(
                (self.k_point.x, self.k_point.y, self.k_point.z),
                dtype=np.complex128,
            )
            if not np.all(np.isfinite(coordinates)):
                raise ValueError("k_point coordinates must be finite")
            if np.any(coordinates != 0.0):
                raise ValueError(
                    "SimulationSpec currently supports only k_point=False or "
                    "Gamma-point k_point=mp.Vector3()"
                )
        self.m = self._normalize_m(self.m)
        if self.m != 0 and not self.is_cylindrical:
            raise ValueError("nonzero m requires dimensions=mp.CYLINDRICAL")

    @property
    def is_cylindrical(self) -> bool:
        """Whether this specification constructs a cylindrical simulation."""
        return self.dimensions == mp.CYLINDRICAL

    @staticmethod
    def _normalize_m(value) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("m must be an integer")
        return int(value)

    def resolve_sources(self, sources=None):
        """Resolve the configured or supplied Meep source sequence."""
        selected = self.sources if sources is None else sources
        return selected() if callable(selected) else selected

    def make(
        self,
        sources: Optional[Union[Sequence, Callable[[], Sequence]]] = None,
        *,
        m: Optional[int] = None,
    ) -> mp.Simulation:
        """Create a Meep `Simulation`.

        Args:
            sources: Optional replacement source list. When omitted, the
                bundled `sources` value is used.
            m: Optional angular-mode override used to construct an adjoint
                cylindrical simulation without mutating this specification.

        Returns:
            Newly constructed Meep simulation.
        """
        kwargs = {
            "cell_size": self.cell_size,
            "boundary_layers": list(self.boundary_layers),
            "geometry": list(self.geometry),
            "sources": self.resolve_sources(sources),
            "resolution": self.resolution,
            "Courant": self.courant,
            "symmetries": list(self.symmetries),
        }
        selected_m = self.m if m is None else self._normalize_m(m)
        if selected_m != 0 and not self.is_cylindrical:
            raise ValueError("nonzero m requires dimensions=mp.CYLINDRICAL")
        if self.geometry_center is not None:
            kwargs["geometry_center"] = self.geometry_center
        if self.chunk_layout is not None:
            kwargs["chunk_layout"] = self.chunk_layout
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        if self.eps_averaging is not None:
            kwargs["eps_averaging"] = self.eps_averaging
        kwargs["k_point"] = self.k_point
        if self.is_cylindrical:
            kwargs["m"] = selected_m
        return mp.Simulation(**kwargs)


@dataclass(frozen=True)
class Near2FarTarget:
    """Discrete-frequency far fields from Meep near-surface monitors.

    The objective receives complex fields of shape `(points, frequencies, 6)`,
    ordered Ex, Ey, Ez, Hx, Hy, Hz. Frequencies and points retain input order.
    The near surfaces and propagation region must share a homogeneous,
    isotropic, lossless medium. The initial implementation uses Cartesian
    coordinates and accumulates every time step (DFT decimation factor 1).
    """

    near_regions: Sequence[mp.Near2FarRegion]
    frequencies: Sequence[float]
    far_points: Sequence[mp.Vector3]

    def __post_init__(self) -> None:
        regions = tuple(self.near_regions)
        if not regions or any(not isinstance(r, mp.Near2FarRegion) for r in regions):
            raise ValueError("near_regions must contain Meep Near2FarRegion objects")
        if not np.isrealobj(self.frequencies):
            raise ValueError("frequencies must be real")
        frequencies = np.asarray(self.frequencies, dtype=float)
        if (
            frequencies.ndim != 1 or not frequencies.size
            or not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0)
        ):
            raise ValueError("frequencies must be a nonempty sequence of positive finite values")
        points = tuple(self.far_points)
        if not points or any(
            not isinstance(p, mp.Vector3) or not np.isrealobj(tuple(p))
            or not np.all(np.isfinite(tuple(p)))
            for p in points
        ):
            raise ValueError("far_points must contain finite real Meep Vector3 positions")
        object.__setattr__(self, "near_regions", regions)
        object.__setattr__(self, "frequencies", tuple(frequencies.tolist()))
        object.__setattr__(self, "far_points", points)


@dataclass(frozen=True)
class PointTarget:
    """Point monitor and matching adjoint-source settings.

    Args:
        position: Physical point where the forward field is monitored and the
            adjoint source is placed. Cylindrical targets may lie on or near
            `r=0`, but must be expressed as `(r, 0, z)` with `r >= 0`;
            on-axis components must satisfy modal regularity.
        component: Meep field component, e.g. `mp.Ez` or `mp.Er`.
        adjoint_source_size: Meep source size for adjoint injection.
            With direct `SimulationSpec.make`, the default zero size selects
            the exact indexed point-monitor transpose. Explicit nonzero
            Cartesian sizes use ordinary `mp.Source` injection. Cylindrical
            gradients require zero size.
        adjoint_source_amplitude: Logical source amplitude. Gradient
            evaluation requires the default value `1.0`; finite-source density
            normalization is applied internally. Other finite values are
            accepted only for value-only evaluation, where no adjoint source
            is created.
    """

    position: mp.Vector3
    component: int
    adjoint_source_size: Optional[mp.Vector3] = None
    adjoint_source_amplitude: complex = 1.0

    def __post_init__(self) -> None:
        if not all(hasattr(self.position, axis) for axis in ("x", "y", "z")):
            raise TypeError("PointTarget position must be a Meep Vector3 value")
        raw_position = np.asarray(
            (self.position.x, self.position.y, self.position.z)
        )
        if np.iscomplexobj(raw_position):
            raise TypeError("PointTarget position must have real coordinates")
        try:
            position = np.asarray(raw_position, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "PointTarget position must have numeric coordinates"
            ) from exc
        if not np.all(np.isfinite(position)):
            raise ValueError("PointTarget position must have finite coordinates")

        if self.adjoint_source_size is not None:
            if not all(
                hasattr(self.adjoint_source_size, axis)
                for axis in ("x", "y", "z")
            ):
                raise TypeError(
                    "PointTarget adjoint_source_size must be a Meep Vector3 value"
                )
            raw_size = np.asarray(
                (
                    self.adjoint_source_size.x,
                    self.adjoint_source_size.y,
                    self.adjoint_source_size.z,
                )
            )
            if np.iscomplexobj(raw_size):
                raise TypeError(
                    "PointTarget adjoint_source_size must have real coordinates"
                )
            try:
                source_size = np.asarray(raw_size, dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "PointTarget adjoint_source_size must have numeric coordinates"
                ) from exc
            if not np.all(np.isfinite(source_size)) or np.any(source_size < 0.0):
                raise ValueError(
                    "PointTarget adjoint_source_size must be finite and nonnegative"
                )

        try:
            amplitude = complex(self.adjoint_source_amplitude)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "PointTarget adjoint_source_amplitude must be numeric"
            ) from exc
        if not np.isfinite(amplitude.real) or not np.isfinite(amplitude.imag):
            raise ValueError(
                "PointTarget adjoint_source_amplitude must be finite"
            )
        object.__setattr__(self, "adjoint_source_amplitude", amplitude)


@dataclass(frozen=True)
class FieldRegionTarget:
    """Field samples at explicit physical points in a region.

    Args:
        positions: Ordered physical points sampled during the forward run.
            Their order is preserved when the monitor history is reshaped.
        component: Meep field component, e.g. `mp.Ez` or `mp.Er`.
        sample_shape: Optional spatial output shape. Its product must equal the
            number of positions. The default is `(len(positions),)`.
        spatial_weights: Optional nonnegative quadrature weights with one entry
            per position. They are used by the default regional-intensity FoM;
            custom FoM callbacks receive the unweighted field history.
        adjoint_source_amplitude: Logical source amplitude shared by the
            region points. Gradient evaluation requires the default value
            `1.0`; finite-source density normalization is applied internally.
            Other finite values are accepted only for value-only evaluation,
            where no adjoint source is created.
    """

    positions: Sequence[mp.Vector3]
    component: int
    sample_shape: Optional[Tuple[int, ...]] = None
    spatial_weights: Optional[Sequence[float]] = None
    adjoint_source_amplitude: complex = 1.0

    def __post_init__(self) -> None:
        positions = tuple(self.positions)
        if not positions:
            raise ValueError("FieldRegionTarget positions must not be empty")
        if any(
            not all(hasattr(position, axis) for axis in ("x", "y", "z"))
            for position in positions
        ):
            raise TypeError(
                "FieldRegionTarget positions must contain Meep Vector3 values"
            )
        for position in positions:
            raw_coordinates = np.asarray(
                (position.x, position.y, position.z)
            )
            if np.iscomplexobj(raw_coordinates):
                raise TypeError(
                    "FieldRegionTarget positions must have real coordinates"
                )
            try:
                coordinates = np.asarray(raw_coordinates, dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "FieldRegionTarget positions must have numeric coordinates"
                ) from exc
            if not np.all(np.isfinite(coordinates)):
                raise ValueError(
                    "FieldRegionTarget positions must have finite coordinates"
                )

        if self.sample_shape is None:
            sample_shape = (len(positions),)
        else:
            sample_shape = tuple(self.sample_shape)
            if any(
                isinstance(count, bool)
                or not isinstance(count, (int, np.integer))
                or count <= 0
                for count in sample_shape
            ):
                raise ValueError(
                    "FieldRegionTarget sample_shape entries must be "
                    "positive integers"
                )
            if int(np.prod(sample_shape, dtype=np.int64)) != len(positions):
                raise ValueError(
                    "FieldRegionTarget sample_shape must match positions"
                )

        if self.spatial_weights is None:
            spatial_weights = np.ones(len(positions), dtype=float)
        else:
            raw_spatial_weights = np.asarray(self.spatial_weights)
            if np.iscomplexobj(raw_spatial_weights):
                raise TypeError(
                    "FieldRegionTarget spatial_weights must be real"
                )
            spatial_weights = np.asarray(
                raw_spatial_weights,
                dtype=float,
            ).reshape(-1)
            if spatial_weights.size != len(positions):
                raise ValueError(
                    "FieldRegionTarget spatial_weights must match positions"
                )
            if (
                not np.all(np.isfinite(spatial_weights))
                or np.any(spatial_weights < 0.0)
                or not np.any(spatial_weights > 0.0)
            ):
                raise ValueError(
                    "FieldRegionTarget spatial_weights must be finite, "
                    "nonnegative, and contain a positive value"
                )

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "sample_shape", sample_shape)
        object.__setattr__(
            self,
            "spatial_weights",
            tuple(float(value) for value in spatial_weights),
        )
        try:
            amplitude = complex(self.adjoint_source_amplitude)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "FieldRegionTarget adjoint_source_amplitude must be numeric"
            ) from exc
        if not np.isfinite(amplitude.real) or not np.isfinite(amplitude.imag):
            raise ValueError(
                "FieldRegionTarget adjoint_source_amplitude must be finite"
            )
        object.__setattr__(self, "adjoint_source_amplitude", amplitude)


@dataclass(frozen=True)
class FluxTarget:
    """Signed Poynting functional through explicit surface points.

    Args:
        positions: Ordered physical points sampled during the forward run.
        normal: Signed axis unit vector defining positive power flow. A
            cylindrical target accepts only radial or axial normals.
        spatial_weights: Optional nonnegative surface-quadrature weights with
            one entry per position. The default assigns unit weight to every
            point. Cylindrical weights must include the full physical
            `2*pi*r` measure. Complex and single-mode cylindrical fields yield
            a modal functional rather than automatically normalized total
            physical power.
    """

    positions: Sequence[mp.Vector3]
    normal: mp.Vector3
    spatial_weights: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        positions = tuple(self.positions)
        if not positions:
            raise ValueError("FluxTarget positions must not be empty")
        if any(
            not all(hasattr(position, axis) for axis in ("x", "y", "z"))
            for position in positions
        ):
            raise TypeError(
                "FluxTarget positions must contain Meep Vector3 values"
            )
        for position in positions:
            raw_coordinates = np.asarray(
                (position.x, position.y, position.z)
            )
            if np.iscomplexobj(raw_coordinates):
                raise TypeError(
                    "FluxTarget positions must have real coordinates"
                )
            try:
                coordinates = np.asarray(raw_coordinates, dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "FluxTarget positions must have numeric coordinates"
                ) from exc
            if not np.all(np.isfinite(coordinates)):
                raise ValueError(
                    "FluxTarget positions must have finite coordinates"
                )

        axis, orientation = _normalize_axis_normal(self.normal)
        normal_coordinates = [0.0, 0.0, 0.0]
        normal_coordinates[axis] = orientation

        if self.spatial_weights is None:
            spatial_weights = np.ones(len(positions), dtype=float)
        else:
            raw_spatial_weights = np.asarray(self.spatial_weights)
            if np.iscomplexobj(raw_spatial_weights):
                raise TypeError("FluxTarget spatial_weights must be real")
            spatial_weights = np.asarray(
                raw_spatial_weights,
                dtype=float,
            ).reshape(-1)
            if spatial_weights.size != len(positions):
                raise ValueError(
                    "FluxTarget spatial_weights must match positions"
                )
            if (
                not np.all(np.isfinite(spatial_weights))
                or np.any(spatial_weights < 0.0)
                or not np.any(spatial_weights > 0.0)
            ):
                raise ValueError(
                    "FluxTarget spatial_weights must be finite, "
                    "nonnegative, and contain a positive value"
                )

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "normal", mp.Vector3(*normal_coordinates))
        object.__setattr__(
            self,
            "spatial_weights",
            tuple(float(value) for value in spatial_weights),
        )


@dataclass(frozen=True)
class EigenmodeCoefficientTarget:
    """Time-domain overlap with a fixed reference eigenmode.

    Args:
        positions: Ordered physical points used for the modal quadrature.
        normal: Signed axis unit vector selecting the propagation direction.
            A cylindrical target accepts only radial or axial normals.
        mode_region: Optional Cartesian line or plane supplied to Meep's eigenmode
            solver. It must be normal to `normal` and contain every position.
            It must not intersect design-dependent material because the solved
            reference profile is fixed after its first evaluation. The port
            medium must be reciprocal. Required with `frequency` when
            `reference_mode_fields` is omitted.
        frequency: Optional positive reference frequency for the fixed
            eigenmode. Required with `mode_region` when
            `reference_mode_fields` is omitted.
        mode: Positive one-based Meep eigenmode band index.
        spatial_weights: Optional nonnegative surface-quadrature weights with
            one entry per position. The default assigns unit weight to every
            point. Cylindrical weights must include the full physical
            `2*pi*r` measure.
        eig_parity: Meep eigenmode parity bit mask.
        eig_resolution: Nonnegative Meep eigenmode-solver resolution. Zero
            selects Meep's default.
        eig_tolerance: Positive Meep eigenmode-solver tolerance.
        reference_mode_fields: Optional fixed reference-mode samples keyed by
            Meep field component. Each one-dimensional array must follow
            `positions`. Cylindrical simulations require this input because
            Meep's eigenmode solver does not support cylindrical coordinates.
            Both tangential E/H pairs must be supplied; zero arrays may
            represent a vanishing pair.
        reference_m: Angular mode represented by cylindrical
            `reference_mode_fields`. It must match the forward simulation's
            `m` and requires `reference_mode_fields`.
    """

    positions: Sequence[mp.Vector3]
    normal: mp.Vector3
    mode_region: Optional[mp.Volume] = None
    frequency: Optional[float] = None
    mode: int = 1
    spatial_weights: Optional[Sequence[float]] = None
    eig_parity: int = mp.NO_PARITY
    eig_resolution: float = 0
    eig_tolerance: float = 1e-12
    reference_mode_fields: Optional[Mapping[int, Sequence[complex]]] = None
    reference_m: Optional[int] = None

    def __post_init__(self) -> None:
        try:
            surface = FluxTarget(
                self.positions,
                self.normal,
                self.spatial_weights,
            )
        except (TypeError, ValueError) as exc:
            message = str(exc).replace(
                "FluxTarget",
                "EigenmodeCoefficientTarget",
            )
            raise type(exc)(message) from exc
        positions = surface.positions
        normal = surface.normal
        spatial_weights = surface.spatial_weights
        axis, _ = _normalize_axis_normal(normal)
        position_coordinates = np.asarray(
            [
                (position.x, position.y, position.z)
                for position in positions
            ],
            dtype=float,
        )

        reference_mode_fields = self.reference_mode_fields
        if reference_mode_fields is None and (
            self.mode_region is None or self.frequency is None
        ):
            raise ValueError(
                "EigenmodeCoefficientTarget requires mode_region and "
                "frequency or reference_mode_fields"
            )

        if self.mode_region is not None:
            if not isinstance(self.mode_region, mp.Volume):
                raise TypeError(
                    "EigenmodeCoefficientTarget mode_region must be a Meep Volume"
                )
            region_center = np.asarray(
                (
                    self.mode_region.center.x,
                    self.mode_region.center.y,
                    self.mode_region.center.z,
                ),
                dtype=float,
            )
            region_size = np.asarray(
                (
                    self.mode_region.size.x,
                    self.mode_region.size.y,
                    self.mode_region.size.z,
                ),
                dtype=float,
            )
            if (
                not np.all(np.isfinite(region_center))
                or not np.all(np.isfinite(region_size))
                or np.any(region_size < 0.0)
            ):
                raise ValueError(
                    "EigenmodeCoefficientTarget mode_region center and size "
                    "must be finite with nonnegative size"
                )
            tolerance = 1.0e-12
            if abs(region_size[axis]) > tolerance:
                raise ValueError(
                    "EigenmodeCoefficientTarget mode_region must be normal to "
                    "normal"
                )
            tangential_axes = tuple(
                index for index in range(3) if index != axis
            )
            if not any(
                region_size[index] > tolerance
                for index in tangential_axes
            ):
                raise ValueError(
                    "EigenmodeCoefficientTarget mode_region must have positive "
                    "tangential extent"
                )
            region_low = region_center - 0.5 * region_size - tolerance
            region_high = region_center + 0.5 * region_size + tolerance
            if (
                np.any(position_coordinates < region_low)
                or np.any(position_coordinates > region_high)
            ):
                raise ValueError(
                    "EigenmodeCoefficientTarget positions must lie in mode_region"
                )

        def real_solver_setting(name, value, *, allow_zero):
            if np.iscomplexobj(np.asarray(value)):
                raise TypeError(
                    f"EigenmodeCoefficientTarget {name} must be real"
                )
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"EigenmodeCoefficientTarget {name} must be a real scalar"
                ) from exc
            invalid_sign = normalized < 0.0 if allow_zero else normalized <= 0.0
            if not np.isfinite(normalized) or invalid_sign:
                bound = "nonnegative" if allow_zero else "positive"
                raise ValueError(
                    f"EigenmodeCoefficientTarget {name} must be finite and "
                    f"{bound}"
                )
            return normalized

        frequency = (
            None
            if self.frequency is None
            else real_solver_setting(
                "frequency",
                self.frequency,
                allow_zero=False,
            )
        )

        if (
            isinstance(self.mode, (bool, np.bool_))
            or not isinstance(self.mode, (int, np.integer))
            or self.mode <= 0
        ):
            raise ValueError(
                "EigenmodeCoefficientTarget mode must be a positive integer"
            )
        if (
            isinstance(self.eig_parity, (bool, np.bool_))
            or not isinstance(self.eig_parity, (int, np.integer))
            or self.eig_parity < 0
        ):
            raise ValueError(
                "EigenmodeCoefficientTarget eig_parity must be a "
                "nonnegative integer"
            )

        eig_resolution = real_solver_setting(
            "eig_resolution",
            self.eig_resolution,
            allow_zero=True,
        )
        eig_tolerance = real_solver_setting(
            "eig_tolerance",
            self.eig_tolerance,
            allow_zero=False,
        )

        normalized_reference_fields = None
        if reference_mode_fields is not None:
            if not isinstance(reference_mode_fields, Mapping):
                raise TypeError(
                    "EigenmodeCoefficientTarget reference_mode_fields must "
                    "map Meep components to arrays"
                )
            if not reference_mode_fields:
                raise ValueError(
                    "EigenmodeCoefficientTarget reference_mode_fields must "
                    "not be empty"
                )
            normalized_reference_fields = {}
            for component, values in reference_mode_fields.items():
                if (
                    isinstance(component, (bool, np.bool_))
                    or not isinstance(component, (int, np.integer))
                ):
                    raise TypeError(
                        "EigenmodeCoefficientTarget reference_mode_fields "
                        "keys must be Meep components"
                    )
                try:
                    field_values = np.asarray(
                        values,
                        dtype=np.complex128,
                    )
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "EigenmodeCoefficientTarget reference_mode_fields "
                        "values must be numeric"
                    ) from exc
                if field_values.ndim != 1 or field_values.size != len(positions):
                    raise ValueError(
                        "EigenmodeCoefficientTarget reference_mode_fields "
                        "must match positions"
                    )
                if not np.all(np.isfinite(field_values)):
                    raise ValueError(
                        "EigenmodeCoefficientTarget reference_mode_fields "
                        "must be finite"
                    )
                normalized_reference_fields[int(component)] = tuple(
                    complex(value) for value in field_values
                )
            normalized_reference_fields = MappingProxyType(
                normalized_reference_fields
            )

        reference_m = self.reference_m
        if reference_m is not None:
            if (
                isinstance(reference_m, (bool, np.bool_))
                or not isinstance(reference_m, (int, np.integer))
            ):
                raise ValueError(
                    "EigenmodeCoefficientTarget reference_m must be an integer"
                )
            reference_m = int(reference_m)
            if normalized_reference_fields is None:
                raise ValueError(
                    "EigenmodeCoefficientTarget reference_m requires "
                    "reference_mode_fields"
                )

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "mode", int(self.mode))
        object.__setattr__(self, "eig_parity", int(self.eig_parity))
        object.__setattr__(self, "eig_resolution", eig_resolution)
        object.__setattr__(self, "eig_tolerance", eig_tolerance)
        object.__setattr__(
            self,
            "reference_mode_fields",
            normalized_reference_fields,
        )
        object.__setattr__(self, "reference_m", reference_m)
        object.__setattr__(
            self,
            "spatial_weights",
            spatial_weights,
        )
