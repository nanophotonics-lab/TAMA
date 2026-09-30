import gc
import tempfile
import time
from typing import Callable, Optional, Tuple, Union

from autograd import grad
import autograd.numpy as npa
import meep as mp
import numpy as np

from .sampling_grid import (
    FastPointMonitor,
    _require_no_meep_symmetries,
    history_storage_dtype,
)
from .native_design import (
    _NativeDesignAccumulator,
    _NativeDesignHistorySet,
    _run_native_adjoint_loop,
    _run_native_forward_loop,
    _native_forward_step_count,
)
from .objectives import (
    _FieldRegionTarget,
    _PointTarget,
    _electric_components,
    _is_cartesian_2d_simulation,
    _is_cylindrical_simulation,
    _is_magnetic_component,
    _is_zero_source_size,
    _sample_time_from_simulation,
    _simulation_time_step,
    _validate_adjoint_source_amplitudes,
    _validate_cylindrical_axis_target,
    _validate_cylindrical_target_position,
    _validate_indexed_monitor_support,
    _validate_logical_adjoint_source_amplitudes,
)
from .flux import (
    _validate_cylindrical_surface,
    flux_component_pairs,
    power_history,
    power_history_pullback,
)
from .eigenmode import (
    coefficient_history,
    coefficient_history_pullback,
    sample_reference_mode,
)
from .specs import (
    DesignGrid,
    EigenmodeCoefficientTarget,
    FieldRegionTarget,
    FluxTarget,
    PointTarget,
    SimulationSpec,
)
from .adaptive_chunking import AdaptiveAdjointChunkBalancer
from ...nyquist import (
    _aligned_run_until,
    _monitor_history_stop,
    _positive_finite_float,
    _validate_reconstruction_settings,
    nyquist_sampling_interval,
)


def _default_intensity_fom(monitor_history: np.ndarray, sample_dt: float):
    """Default point-signal objective: 0.5 * integral |E(t)|^2 dt."""
    return 0.5 * npa.sum(npa.abs(monitor_history) ** 2) * sample_dt


class TDAObjective:
    """Meep-style callable optimization problem for time-domain field objectives.

    Users provide Meep simulation construction through `sim_factory`, design
    updates through `update_design`, and a point, regional-field, flux, or
    eigenmode-coefficient target. The scalar FoM can be customized with
    `fom_fn`; when no adjoint signal is supplied, autograd differentiates it
    with respect to the sampled target history.

    Sign convention: `evaluate` / `fom_and_grad` return
    `(fom, d fom / d rho)` with the maximization sign, matching
    `MultiTDAObjective` and Meep adjoint's `OptimizationProblem`. Negate
    both in the optimizer callback when driving a minimizer such as nlopt or
    `scipy.optimize.minimize`.
    """

    def __init__(
        self,
        *,
        update_design: Optional[Callable[[np.ndarray], None]] = None,
        t_final: Optional[float] = None,
        sim_factory: Optional[Callable[..., mp.Simulation]] = None,
        monitor_position: Optional[mp.Vector3] = None,
        component: Optional[int] = None,
        adjoint_source_size: Optional[mp.Vector3] = None,
        adjoint_source_amplitude: Optional[complex] = None,
        fom_fn: Optional[Callable[[np.ndarray, float], float]] = None,
        adjoint_signal_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        dt: Optional[float] = None,
        resolution: Optional[float] = None,
        sampling_interval: int = 1,
        design: Optional[DesignGrid] = None,
        simulation: Optional[SimulationSpec] = None,
        target: Optional[
            Union[
                PointTarget,
                FieldRegionTarget,
                FluxTarget,
                EigenmodeCoefficientTarget,
            ]
        ] = None,
        history_dtype=np.complex128,
        chunk_balancer: Union[AdaptiveAdjointChunkBalancer, str, None] = "auto",
        max_frequency: Optional[float] = None,
        reconstruction_window: Optional[str] = "kaiser",
        reconstruction_window_params=None,
        reconstruction_half_width: int = 64,
        reuse_simulation: bool = False,
    ) -> None:
        """Create a callable time-domain adjoint optimization problem.

        Args:
            update_design: Function that writes the design vector `x` into
                the Meep geometry or material grid.
            t_final: Positive finite forward simulation end time.
            sim_factory: Function returning a Meep `Simulation`. It is called
                with no arguments for the forward run and with a source list for
                the adjoint run. Arbitrary factories use ordinary Cartesian
                `mp.Source` injection and do not support cylindrical gradient
                evaluation.
            monitor_position: Physical point at which the forward field is
                sampled and the adjoint source is placed. For cylindrical
                gradients, targets use `(r, 0, z)` with `r >= 0` and may lie
                on or near `r=0`; on-axis components must satisfy modal
                regularity.
            component: Meep field component, e.g. `mp.Ez` or `mp.Er`.
                Cylindrical gradients accept `Er/Ep/Ez/Hr/Hp/Hz`.
            adjoint_source_size: Meep source size for adjoint injection.
                With direct `SimulationSpec.make`, the default zero size uses
                the exact indexed transpose of the point monitor. Explicit
                nonzero Cartesian sizes use ordinary `mp.Source` injection.
                Cylindrical gradients require zero size.
            adjoint_source_amplitude: Logical source amplitude. Gradient
                evaluation requires `1.0`; finite-source density
                normalization is applied internally. Other finite values are
                accepted only for value-only evaluation.
            fom_fn: Optional scalar objective `fom_fn(E_t, dt)`. It must use
                autograd-compatible operations if `adjoint_signal_fn` is not
                supplied.
            adjoint_signal_fn: Optional manual continuous-time bilinear
                covector provider. For monitor history `u` and returned `q`,
                the convention is
                `delta FoM = dt * Re(sum(q * delta u))`. It requires `fom_fn`
                and must return the same shape as the monitor history.
            dt: Optional legacy expected time step. The created simulation's
                `fields.dt` is authoritative; when supplied, `dt` must
                match it. It is used only as a fallback for simulation doubles
                that do not expose `fields.dt`.
            resolution: Optional Meep resolution used with the simulation's
                Courant factor to infer `dt` only when the created simulation
                does not expose `fields.dt`.
            sampling_interval: Dimensionless number of Meep time steps between
                design-grid field samples. Values above one may extend the
                forward run by less than one sparse interval to align its
                closing sample.
            max_frequency: Optional maximum relevant frequency `f_max` used
                to choose `sampling_interval` from the Nyquist criterion.
                Include the source, objective, adjoint response, and any safety
                margin. This does not verify the band-limit or temporal-boundary
                assumptions. Cannot be combined with `sampling_interval > 1`.
            reconstruction_window: Window applied to the finite sinc used to
                reconstruct sparse forward fields. Supported values are
                `rectangular`/`None`, `hamming`, `hann`, `blackman`,
                `bartlett`, and `kaiser`; `none`, `boxcar`, and `hanning` are
                accepted aliases.
            reconstruction_window_params: Optional reconstruction-window
                parameters. Only `kaiser` accepts `{"beta": value}`;
                its default beta is 6.
            reconstruction_half_width: Number of sparse samples on each side
                of the finite sinc support. Default is 64.
            design: Optional `DesignGrid` bundle. It is required for gradient
                evaluation and fills `update_design` when that callback is
                omitted. Value-only evaluation may omit it.
            simulation: Optional `SimulationSpec` bundle. When supplied, it
                fills `sim_factory` and `resolution` unless those are
                explicitly supplied. Its direct `make` path enables exact
                indexed injection for zero-size Cartesian targets. It is
                required for cylindrical gradients and supplies their forward
                angular mode `m`.
            target: Optional `PointTarget`, `FieldRegionTarget`,
                `FluxTarget`, or `EigenmodeCoefficientTarget` bundle.
                A regional target exposes a history whose first axis is time
                and whose remaining axes follow `sample_shape`. It samples
                explicitly ordered physical positions and requires direct
                `SimulationSpec` use for gradient evaluation. A flux target
                exposes a length-`N-1` signed-power history evaluated from
                four tangential E/H histories at explicit surface points;
                its electric and time-centered magnetic gradient sources are
                combined in one adjoint simulation. An eigenmode target
                exposes the complex time-domain overlap with one fixed,
                power-normalized reference mode and uses the same
                combined-adjoint structure.
            history_dtype: Real or complex floating dtype requested for
                temporary forward field histories. Its real counterpart is
                used automatically when Meep initializes real forward fields;
                complex forward fields require a complex dtype.
            chunk_balancer: Adaptive balancer that updates the supplied
                `SimulationSpec` chunk layout after each gradient evaluation.
                The default `"auto"` creates one for a direct, concrete
                `SimulationSpec` whose `chunk_layout` is unset. Pass `None` to
                opt out, or supply a custom balancer. Ordinary forward sources
                are protected automatically. Workload calibration uses three
                to eight gradient evaluations and freezes the best measured
                layout after the critical-rank work converges.
                Exact indexed targets do not constrain chunk boundaries.
                `source_boundary_mode="finite"` selects ordinary finite
                Cartesian sources for eligible `PointTarget` entries and is
                incompatible with exact regional, flux, and eigenmode targets.
            reuse_simulation: Reuse the initialized forward `Simulation` for
                the adjoint run. This requires
                `simulation=SimulationSpec(...)`
                and should only be enabled for restart-safe, time-invariant
                simulations. Cylindrical reuse changes the mode from `m` to
                `-m` before the adjoint run.
        """
        uses_simulation_spec = simulation is not None and (
            sim_factory is None or getattr(sim_factory, "__self__", None) is simulation
        )
        uses_concrete_simulation_spec = uses_simulation_spec and isinstance(
            simulation,
            SimulationSpec,
        )
        if isinstance(chunk_balancer, str):
            if chunk_balancer != "auto":
                raise ValueError("chunk_balancer must be 'auto', None, or a balancer")
            chunk_balancer = (
                AdaptiveAdjointChunkBalancer()
                if uses_concrete_simulation_spec
                and simulation.chunk_layout is None
                else None
            )
        if design is not None:
            update_design = update_design if update_design is not None else design.update_weights
        if simulation is not None:
            sim_factory = sim_factory if sim_factory is not None else simulation.make
            resolution = resolution if resolution is not None else simulation.resolution
        regional_target = isinstance(target, FieldRegionTarget)
        flux_target = isinstance(target, FluxTarget)
        eigenmode_target = isinstance(target, EigenmodeCoefficientTarget)
        mixed_surface_target = flux_target or eigenmode_target
        if regional_target or mixed_surface_target:
            conflicting = [
                name
                for name, value in (
                    ("monitor_position", monitor_position),
                    ("component", component),
                    ("adjoint_source_size", adjoint_source_size),
                    ("adjoint_source_amplitude", adjoint_source_amplitude),
                )
                if value is not None
            ]
            if conflicting:
                raise ValueError(
                    f"{type(target).__name__} cannot be combined with legacy target "
                    "arguments: " + ", ".join(conflicting)
                )
            if regional_target:
                component = target.component
                adjoint_source_amplitude = target.adjoint_source_amplitude
        elif target is not None:
            monitor_position = monitor_position if monitor_position is not None else target.position
            component = component if component is not None else target.component
            adjoint_source_size = (
                adjoint_source_size
                if adjoint_source_size is not None
                else target.adjoint_source_size
            )
            adjoint_source_amplitude = (
                adjoint_source_amplitude
                if adjoint_source_amplitude is not None
                else target.adjoint_source_amplitude
            )
        if adjoint_source_amplitude is None:
            adjoint_source_amplitude = 1.0

        missing = [
            name for name, value in (
                ("update_design", update_design),
                ("t_final", t_final),
                ("sim_factory", sim_factory),
                (
                    "monitor_position",
                    (
                        target.positions
                        if regional_target or mixed_surface_target
                        else monitor_position
                    ),
                ),
                (
                    "component",
                    target.normal if mixed_surface_target else component,
                ),
            )
            if value is None
        ]
        if missing:
            raise ValueError("TDAObjective missing required inputs: " + ", ".join(missing))
        if adjoint_signal_fn is not None and fom_fn is None:
            raise ValueError(
                "adjoint_signal_fn requires fom_fn so the objective value "
                "and derivative remain consistent"
            )
        t_final = _positive_finite_float(t_final, "t_final")
        if (
            isinstance(sampling_interval, (bool, np.bool_))
            or not isinstance(sampling_interval, (int, np.integer))
            or sampling_interval < 1
        ):
            raise ValueError("sampling_interval must be a positive integer")
        sampling_interval = int(sampling_interval)
        if max_frequency is not None:
            max_frequency = _positive_finite_float(
                max_frequency,
                "max_frequency",
            )
            if sampling_interval != 1:
                raise ValueError(
                    "max_frequency cannot be combined with sampling_interval > 1"
                )
        (
            reconstruction_window,
            reconstruction_window_params,
            reconstruction_half_width,
        ) = _validate_reconstruction_settings(
            reconstruction_window,
            reconstruction_window_params,
            reconstruction_half_width,
        )
        if not isinstance(reuse_simulation, (bool, np.bool_)):
            raise ValueError("reuse_simulation must be a boolean")
        reuse_simulation = bool(reuse_simulation)
        if reuse_simulation and not uses_simulation_spec:
            raise ValueError("reuse_simulation=True requires SimulationSpec.make")
        if chunk_balancer is not None and not uses_simulation_spec:
            raise ValueError("adaptive chunk balancing requires SimulationSpec.make")
        has_coordinate_system_metadata = (
            simulation is not None or design is not None
        )
        is_cylindrical = (
            _is_cylindrical_simulation(simulation)
            if simulation is not None
            else bool(getattr(design, "is_cylindrical", False))
        )
        if mixed_surface_target and has_coordinate_system_metadata:
            flux_component_pairs(
                target.normal,
                cylindrical=is_cylindrical,
            )
            if is_cylindrical:
                _validate_cylindrical_surface(
                    target.positions,
                    target.normal,
                    target.spatial_weights,
                )
        if (
            mixed_surface_target
            and not is_cylindrical
            and design is not None
            and design.dimensions == 2
            and abs(float(target.normal.z)) > 0.0
        ):
            raise ValueError(
                f"2D {type(target).__name__} normals must lie in the x-y plane"
            )
        if (
            mixed_surface_target
            and chunk_balancer is not None
            and getattr(chunk_balancer, "source_boundary_mode", "auto") == "finite"
        ):
            raise ValueError(
                f"{type(target).__name__} requires exact indexed adjoint "
                "sources and cannot "
                "use source_boundary_mode='finite'"
            )

        self.update_design = update_design
        self.sim_factory = sim_factory
        self.t_final = t_final
        self.dt = dt
        self.resolution = resolution
        self._fallback_courant = (
            float(getattr(simulation, "courant", 0.5))
            if simulation is not None
            else 0.5
        )
        self.sampling_interval = sampling_interval
        self.max_frequency = max_frequency
        self.last_sampling_interval = None
        self.reconstruction_window = reconstruction_window
        self.reconstruction_window_params = reconstruction_window_params
        self.reconstruction_half_width = reconstruction_half_width
        self.history_dtype = history_dtype
        self.chunk_balancer = chunk_balancer
        self._simulation_spec = simulation
        self._uses_simulation_spec = uses_simulation_spec
        self._uses_concrete_simulation_spec = uses_concrete_simulation_spec
        self._configured_simulation_m = (
            int(simulation.m) if uses_simulation_spec else None
        )
        self._has_coordinate_system_metadata = has_coordinate_system_metadata
        self._inferred_is_cylindrical = None
        self._is_cylindrical = is_cylindrical
        if simulation is not None and design is not None and (
            bool(getattr(design, "is_cylindrical", False))
            != self._is_cylindrical
        ):
            raise ValueError(
                "DesignGrid coordinate system must match SimulationSpec dimensions"
            )
        self.reuse_simulation = reuse_simulation
        self._reuse_simulation_for_adjoint = (
            self.reuse_simulation and uses_simulation_spec
        )
        self.last_chunk_balance = None
        self.last_source_boundary_decision = None
        self._is_regional_target = regional_target
        self._is_flux_target = flux_target
        self._is_eigenmode_target = eigenmode_target
        self._reference_mode_fields = None
        self.objective = (
            _FieldRegionTarget(
                positions=target.positions,
                component=component,
                sample_shape=target.sample_shape,
                spatial_weights=target.spatial_weights,
                adjoint_source_amplitude=adjoint_source_amplitude,
            )
            if regional_target
            else target
            if mixed_surface_target
            else _PointTarget(
                monitor_position=monitor_position,
                component=component,
                adjoint_source_size=adjoint_source_size,
                adjoint_source_amplitude=adjoint_source_amplitude,
            )
        )
        self._uses_default_fom = fom_fn is None
        if fom_fn is None and regional_target:
            spatial_weights = npa.asarray(self.objective.spatial_weights)

            def fom_fn(monitor_history, sample_dt):
                return (
                    0.5
                    * npa.sum(
                        npa.abs(monitor_history) ** 2
                        * spatial_weights
                    )
                    * sample_dt
                )
        elif fom_fn is None and flux_target:
            def fom_fn(power, sample_dt):
                return npa.sum(power) * sample_dt
        elif fom_fn is None and eigenmode_target:
            def fom_fn(coefficient, sample_dt):
                return npa.sum(npa.abs(coefficient) ** 2) * sample_dt
        self.fom_fn = fom_fn if fom_fn is not None else _default_intensity_fom
        self.adjoint_signal_fn = adjoint_signal_fn
        self.design = design
        if design is None:
            self.gradient_components = ()
        elif mixed_surface_target:
            required = set()
            for electric, magnetic, _ in flux_component_pairs(
                target.normal,
                cylindrical=self._is_cylindrical,
            ):
                required.update(
                    _electric_components(
                        electric,
                        design.dimensions,
                        cylindrical=self._is_cylindrical,
                        m=getattr(simulation, "m", 0),
                    )
                )
                required.update(
                    _electric_components(
                        magnetic,
                        design.dimensions,
                        cylindrical=self._is_cylindrical,
                        m=getattr(simulation, "m", 0),
                    )
                )
            self.gradient_components = tuple(
                component
                for component in (
                    (mp.Er, mp.Ep, mp.Ez)
                    if self._is_cylindrical
                    else (mp.Ex, mp.Ey, mp.Ez)
                )
                if component in required
            )
        else:
            self.gradient_components = _electric_components(
                component,
                design.dimensions,
                cylindrical=self._is_cylindrical,
                m=getattr(simulation, "m", 0),
            )
        self._native_adjoint_midpoint = (
            False if mixed_surface_target else _is_magnetic_component(component)
        )

    def _make_history_memmap(self, shape, *, dtype=None):
        tmp = tempfile.NamedTemporaryFile(prefix="tama_history_", suffix=".dat", delete=False)
        path = tmp.name
        tmp.close()
        return np.memmap(
            path,
            dtype=self.history_dtype if dtype is None else dtype,
            mode="w+",
            shape=shape,
        ), path

    def time_step(self, sim: mp.Simulation) -> float:
        """Return the time step used by TAMA sampling.

        Args:
            sim: Forward Meep simulation created by `sim_factory`.

        Returns:
            The Meep simulation time step when available, otherwise explicit
            `dt` when supplied, otherwise `Courant / resolution`.
        """
        sim_dt = _simulation_time_step(sim, self.dt)
        if sim_dt is not None:
            return sim_dt
        resolution = self.resolution if self.resolution is not None else getattr(sim, "resolution", None)
        if resolution is None:
            raise ValueError("TDAObjective requires dt, resolution, or a Simulation with a resolution attribute")
        courant = float(getattr(sim, "Courant", self._fallback_courant))
        return courant / resolution

    def _resolve_sampling_interval(self, dt: float) -> int:
        if self.max_frequency is None:
            return self.sampling_interval
        return nyquist_sampling_interval(self.max_frequency, dt)

    def _forward_sources_for_evaluation(self):
        if (
            self.chunk_balancer is None
            or not self._uses_concrete_simulation_spec
        ):
            return ()
        return tuple(self._simulation_spec.resolve_sources())

    def _make_forward_simulation(self, forward_sources=()):
        if self.chunk_balancer is not None and self._uses_concrete_simulation_spec:
            return self._simulation_spec.make(forward_sources)
        return self.sim_factory()

    def __call__(
        self,
        x: np.ndarray,
        need_gradient: bool = True,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Evaluate the objective and optionally its design gradient.

        Args:
            x: Flat design vector.
            need_gradient: If `True`, run the adjoint simulation and return a
                gradient. If `False`, run only the forward simulation.

        Returns:
            `(fom, gradient)`. `gradient` is `None` when
            `need_gradient=False`.
        """
        return self.evaluate(x, need_gradient=need_gradient)

    def fom(self, x: np.ndarray) -> float:
        """Evaluate only the scalar objective value.

        Args:
            x: Flat design vector passed to `update_design`.

        Returns:
            Scalar FoM returned by the forward-only path.
        """
        value, _ = self.evaluate(x, need_gradient=False)
        return value

    def fom_and_grad(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate both the scalar objective and the flat design gradient.

        Args:
            x: Flat design vector passed to `update_design`.

        Returns:
            `(objective_value, gradient)` where `gradient` is a flat real
            array over the design variables.
        """
        return self.evaluate(x, need_gradient=True)

    def _fom_value_and_adjoint_signal(
        self,
        monitor_history: np.ndarray,
        sample_dt: float,
    ) -> Tuple[float, np.ndarray]:
        """Compute the FoM and its continuous-time bilinear covector.

        Args:
            monitor_history: Forward point or regional field samples.
            sample_dt: Time step between adjacent monitor samples.

        Returns:
            Scalar FoM and sampled continuous-time bilinear covector `q`, with
            `delta FoM = sample_dt * Re(sum(q * delta monitor_history))`.
        """
        monitor_history = np.asarray(monitor_history)
        objective_value = self.fom_fn(monitor_history, sample_dt)

        if self.adjoint_signal_fn is not None:
            adjoint_signal = np.asarray(
                self.adjoint_signal_fn(monitor_history, sample_dt)
            )
            if adjoint_signal.shape != monitor_history.shape:
                raise ValueError(
                    "adjoint_signal_fn must match the monitor history shape"
                )
        else:
            # Autograd differentiates the Riemann-sum objective with respect to
            # sampled values; divide by dt to recover the continuous covector.
            d_fom_d_samples = grad(self.fom_fn, 0)(monitor_history, sample_dt)
            adjoint_signal = d_fom_d_samples / sample_dt

        return float(objective_value), np.asarray(adjoint_signal)

    def _evaluate_mixed_surface(
        self,
        x: np.ndarray,
        need_gradient: bool,
        forward_sources,
    ):
        """Evaluate a Cartesian or cylindrical mixed-field surface target."""
        target_name = type(self.objective).__name__
        self.update_design(x)
        target_positions = tuple(self.objective.positions)
        native_history = None
        native_accumulator = None
        sim_fwd = None
        sim_adj = None
        monitor_objects = {}
        monitor_stencils = {}
        try:
            if self._has_coordinate_system_metadata:
                is_cylindrical = self._is_cylindrical
            else:
                sim_fwd = self._make_forward_simulation(forward_sources)
                _require_no_meep_symmetries(sim_fwd)
                is_cylindrical = _is_cylindrical_simulation(sim_fwd)
                if (
                    self._inferred_is_cylindrical is not None
                    and is_cylindrical != self._inferred_is_cylindrical
                ):
                    raise ValueError(
                        "target coordinate system does not match the forward "
                        "simulation"
                    )

            component_pairs = flux_component_pairs(
                self.objective.normal,
                cylindrical=is_cylindrical,
            )
            if is_cylindrical and not self._has_coordinate_system_metadata:
                _validate_cylindrical_surface(
                    target_positions,
                    self.objective.normal,
                    self.objective.spatial_weights,
                )
            if not self._has_coordinate_system_metadata:
                self._inferred_is_cylindrical = is_cylindrical
            monitor_components = tuple(
                dict.fromkeys(
                    component
                    for electric, magnetic, _ in component_pairs
                    for component in (electric, magnetic)
                )
            )
            source_positions = tuple(
                position
                for _ in monitor_components
                for position in target_positions
            )
            source_amplitudes = tuple(1.0 for _ in source_positions)
            indexed_source_mask = tuple(
                self._uses_simulation_spec for _ in source_positions
            )

            if self.chunk_balancer is not None:
                source_boundary_decision = (
                    self.chunk_balancer.prepare_initial_layout(
                        self._simulation_spec,
                        forward_sources=forward_sources,
                        adjoint_source_positions=source_positions,
                        adjoint_source_sizes=tuple(
                            mp.Vector3() for _ in source_positions
                        ),
                        adjoint_source_amplitudes=source_amplitudes,
                        indexed_source_mask=indexed_source_mask,
                    )
                )
                self.last_source_boundary_decision = source_boundary_decision
                if any(
                    not _is_zero_source_size(size)
                    for size in source_boundary_decision.source_sizes
                ):
                    raise RuntimeError(
                        f"{target_name} indexed adjoint source acquired a "
                        "nonzero effective source size"
                    )
                source_amplitudes = tuple(
                    source_boundary_decision.source_amplitudes
                )

            amplitudes_by_component = {}
            point_count = len(target_positions)
            for component_index, component in enumerate(monitor_components):
                start = component_index * point_count
                amplitudes_by_component[component] = source_amplitudes[
                    start : start + point_count
                ]

            balance_enabled = bool(
                need_gradient
                and self.chunk_balancer is not None
                and getattr(
                    self.chunk_balancer,
                    "needs_rebalance",
                    not getattr(self.chunk_balancer, "is_frozen", False),
                )
            )
            balance_wall_start = (
                time.perf_counter() if balance_enabled else None
            )
            balance_timing = {"extra_seconds": 0.0}
            if sim_fwd is None:
                sim_fwd = self._make_forward_simulation(forward_sources)
                _require_no_meep_symmetries(sim_fwd)
            actual_is_cylindrical = _is_cylindrical_simulation(sim_fwd)
            if (
                need_gradient
                and actual_is_cylindrical
                and not self._uses_simulation_spec
            ):
                raise ValueError(
                    "cylindrical gradient evaluation requires "
                    "simulation=SimulationSpec(...)"
                )
            if (
                self._has_coordinate_system_metadata
                and actual_is_cylindrical != self._is_cylindrical
            ):
                raise ValueError(
                    "target coordinate system does not match the forward "
                    "simulation"
                )
            is_cylindrical = actual_is_cylindrical
            forward_mode = getattr(sim_fwd, "m", 0)
            forward_mode = 0 if forward_mode is None else int(forward_mode)
            if need_gradient and is_cylindrical:
                if forward_mode not in (-1, 0, 1):
                    raise ValueError(
                        "cylindrical gradient evaluation currently supports "
                        "m=-1, 0, or +1"
                    )
                for component in monitor_components:
                    for position, weight in zip(
                        target_positions,
                        self.objective.spatial_weights,
                    ):
                        if weight != 0.0:
                            _validate_cylindrical_axis_target(
                                component,
                                forward_mode,
                                position,
                            )
            if (
                _is_cartesian_2d_simulation(sim_fwd)
                and abs(float(self.objective.normal.z)) > 0.0
            ):
                raise ValueError(
                    f"2D {target_name} normals must lie in the x-y plane"
                )
            dt = self.time_step(sim_fwd)
            run_history_dtype = history_storage_dtype(
                sim_fwd,
                self.history_dtype,
            )
            if need_gradient:
                native_history = _NativeDesignHistorySet(
                    self.design,
                    self.gradient_components,
                    run_history_dtype,
                    lambda shape: self._make_history_memmap(
                        shape,
                        dtype=run_history_dtype,
                    ),
                )
            if (
                self._is_eigenmode_target
                and self._reference_mode_fields is None
            ):
                self._reference_mode_fields = sample_reference_mode(
                    sim_fwd,
                    self.objective,
                )
            sampling_interval = self._resolve_sampling_interval(dt)
            self.last_sampling_interval = sampling_interval
            dt_eff = dt * sampling_interval
            run_until = _aligned_run_until(
                self.t_final,
                dt=dt,
                sampling_interval=(sampling_interval if need_gradient else 1),
            )
            monitor_times = []
            sample_count = {"count": 0}
            fwd_count = {"value": 0}
            forward_sampling_interval = (
                sampling_interval if need_gradient else 1
            )
            use_native_forward = hasattr(
                getattr(sim_fwd, "fields", None),
                "this",
            )
            if use_native_forward:
                native_fine_step_count = _native_forward_step_count(
                    sim_fwd,
                    run_until,
                    forward_sampling_interval,
                )
                n_expected = (
                    native_fine_step_count // forward_sampling_interval + 1
                )
                n_monitor_expected = native_fine_step_count + 1
            else:
                n_expected = int(np.ceil(run_until / dt_eff)) + 8
                n_monitor_expected = int(np.ceil(run_until / dt)) + 8
            monitor_history = np.empty(
                (
                    n_monitor_expected,
                    len(monitor_components),
                    point_count,
                ),
                dtype=run_history_dtype,
            )

            if use_native_forward:
                native_setup_start = (
                    time.perf_counter() if balance_enabled else None
                )
                for component in monitor_components:
                    monitor = FastPointMonitor(
                        sim_fwd,
                        component,
                        target_positions,
                    )
                    indexed_stencil = monitor.indexed_transpose_stencil()
                    _validate_indexed_monitor_support(
                        sim_fwd,
                        (component,) * point_count,
                        indexed_stencil[0],
                    )
                    monitor_objects[component] = monitor
                    monitor_stencils[component] = indexed_stencil
                native_setup_seconds = (
                    time.perf_counter() - native_setup_start
                    if balance_enabled
                    else 0.0
                )
                (
                    sample_count["count"],
                    fwd_count["value"],
                    monitor_times,
                    native_forward_seconds,
                ) = _run_native_forward_loop(
                    sim_fwd,
                    tuple(
                        (
                            monitor_objects[component],
                            monitor_history[:, component_index, :],
                        )
                        for component_index, component in enumerate(
                            monitor_components
                        )
                    ),
                    native_history if need_gradient else None,
                    run_until=run_until,
                    sampling_interval=forward_sampling_interval,
                    fine_step_count=native_fine_step_count,
                )
                if balance_enabled:
                    balance_timing["extra_seconds"] += (
                        native_setup_seconds + native_forward_seconds
                    )
            else:
                def record_fwd_impl(sim):
                    nonlocal monitor_history
                    sample_index = sample_count["count"]
                    if sample_index >= monitor_history.shape[0]:
                        grown_history = np.empty(
                            (
                                2 * monitor_history.shape[0],
                                monitor_history.shape[1],
                                monitor_history.shape[2],
                            ),
                            dtype=monitor_history.dtype,
                        )
                        grown_history[:sample_index] = monitor_history
                        monitor_history = grown_history
                    for component_index, component in enumerate(
                        monitor_components
                    ):
                        monitor = monitor_objects.get(component)
                        if monitor is None:
                            monitor = FastPointMonitor(
                                sim,
                                component,
                                target_positions,
                            )
                            indexed_stencil = monitor.indexed_transpose_stencil()
                            _validate_indexed_monitor_support(
                                sim,
                                (component,) * point_count,
                                indexed_stencil[0],
                            )
                            monitor_stencils[component] = indexed_stencil
                        monitor_objects[component] = monitor
                        monitor.sample_history_into(
                            monitor_history[sample_index, component_index]
                        )
                    monitor_times.append(
                        _sample_time_from_simulation(
                            sim,
                            sample_index,
                            dt,
                        )
                    )
                    if (
                        need_gradient
                        and sample_index % sampling_interval == 0
                    ):
                        native_history.sample_forward(
                            sim,
                            fwd_count["value"],
                            n_expected,
                        )
                        fwd_count["value"] += 1
                    sample_count["count"] += 1

                record_fwd = record_fwd_impl
                if balance_enabled:
                    def record_fwd(sim):
                        start = time.perf_counter()
                        try:
                            record_fwd_impl(sim)
                        finally:
                            balance_timing["extra_seconds"] += (
                                time.perf_counter() - start
                            )

                sim_fwd.run(record_fwd, until=run_until)
            actual_time = sim_fwd.round_time()
            monitor_history = monitor_history[:sample_count["count"]]
            monitor_history_shape = monitor_history.shape
            monitor_history = monitor_objects[
                monitor_components[0]
            ].reduce_history(
                monitor_history.reshape(
                    monitor_history.shape[0],
                    -1,
                )
            ).reshape(monitor_history_shape)
            full_monitor_times = np.asarray(monitor_times, dtype=float)
            monitor_stop = _monitor_history_stop(
                full_monitor_times,
                self.t_final,
            )
            raw_histories = {}
            indexed_stencils = {}
            for component_index, component in enumerate(
                monitor_components
            ):
                monitor = monitor_objects[component]
                raw_histories[component] = np.asarray(
                    monitor_history[
                        :monitor_stop,
                        component_index,
                    ],
                    dtype=run_history_dtype,
                )
                if need_gradient:
                    offsets, source_data, weights = monitor_stencils[component]
                    if offsets.shape != (point_count + 1,):
                        raise RuntimeError(
                            f"{target_name} indexed monitor stencil count does not "
                            "match its target positions"
                        )
                    component_stencils = []
                    for point_index in range(point_count):
                        point_slice = slice(
                            int(offsets[point_index]),
                            int(offsets[point_index + 1]),
                        )
                        component_stencils.append(
                            (
                                source_data[point_slice],
                                weights[point_slice],
                            )
                        )
                    indexed_stencils[component] = component_stencils
            del monitor
            monitor_times = full_monitor_times[:monitor_stop]
            monitor_objects.clear()
            monitor_stencils.clear()

            if need_gradient:
                field_histories = native_history.finish_forward(
                    fwd_count["value"]
                )
                fine_step_count = sample_count["count"] - 1
                expected_history_rows = (
                    fine_step_count // sampling_interval + 1
                )
                if (
                    fine_step_count % sampling_interval != 0
                    or any(
                        history.shape[0] != expected_history_rows
                        for history in field_histories.values()
                    )
                ):
                    raise RuntimeError(
                        "forward callback count is not aligned with the "
                        "Nyquist sampling interval"
                    )
                native_history.release_forward()

            if self._is_eigenmode_target:
                target_history = coefficient_history(
                    raw_histories,
                    self._reference_mode_fields,
                    self.objective.normal,
                    self.objective.spatial_weights,
                    cylindrical=is_cylindrical,
                )
            else:
                target_history = power_history(
                    raw_histories,
                    self.objective.normal,
                    self.objective.spatial_weights,
                    cylindrical=is_cylindrical,
                )
            fwd_timing = (
                self.chunk_balancer.capture_timing(sim_fwd)
                if balance_enabled
                else None
            )
            if not (need_gradient and self._reuse_simulation_for_adjoint):
                sim_fwd.reset_meep()
                sim_fwd = None
            gc.collect()

            if need_gradient:
                objective_value, target_covector = (
                    self._fom_value_and_adjoint_signal(target_history, dt)
                )
            else:
                objective_value = float(self.fom_fn(target_history, dt))
                return objective_value, None

            if any(
                history.shape[0] < 4
                for history in raw_histories.values()
            ):
                raise RuntimeError(
                    f"{target_name} gradient requires at least four monitor "
                    "samples for cubic adjoint-source interpolation; "
                    "increase t_final"
                )
            if any(
                history.shape[0] < 2
                for history in field_histories.values()
            ):
                raise RuntimeError(
                    "TDAObjective gradient requires at least two field "
                    "samples; increase t_final or decrease sampling_interval"
                )

            if self._is_eigenmode_target:
                raw_covectors = coefficient_history_pullback(
                    raw_histories,
                    self._reference_mode_fields,
                    self.objective.normal,
                    self.objective.spatial_weights,
                    target_covector,
                    cylindrical=is_cylindrical,
                )
            else:
                raw_covectors = power_history_pullback(
                    raw_histories,
                    self.objective.normal,
                    self.objective.spatial_weights,
                    target_covector,
                    cylindrical=is_cylindrical,
                )
            del (
                monitor_history,
                raw_histories,
                target_covector,
                target_history,
            )
            target_adapters = {
                component: _FieldRegionTarget(
                    positions=target_positions,
                    component=component,
                    sample_shape=(point_count,),
                    spatial_weights=np.ones(point_count),
                )
                for component in monitor_components
            }

            previous_measurements = [fwd_timing] if balance_enabled else []
            adjoint_sources = []
            for component in monitor_components:
                component_covector = raw_covectors.pop(component)
                component_amplitudes = amplitudes_by_component[component]
                source_end_times = (actual_time,)
                if _is_magnetic_component(component):
                    # Causal two-tap filtering makes ordinary accumulation
                    # equal the previous midpoint-adjoint accumulation.
                    component_amplitudes = tuple(
                        0.5 * amplitude
                        for amplitude in component_amplitudes
                    )
                    source_end_times = (actual_time, actual_time + dt)
                for source_end_time in source_end_times:
                    adjoint_sources.extend(
                        target_adapters[component].adjoint_sources(
                            component_covector,
                            monitor_times,
                            source_end_time,
                            source_amplitudes=component_amplitudes,
                            indexed_stencils=indexed_stencils[component],
                        )
                    )

            if self._reuse_simulation_for_adjoint:
                if is_cylindrical:
                    sim_fwd.change_m(-forward_mode)
                sim_fwd.restart_fields()
                sim_fwd.clear_dft_monitors()
                if balance_enabled:
                    sim_fwd.fields.reset_timers()
                sim_fwd.change_sources(adjoint_sources)
                sim_adj = sim_fwd
                sim_fwd = None
            elif is_cylindrical:
                sim_adj = self._simulation_spec.make(
                    adjoint_sources,
                    m=-forward_mode,
                )
            else:
                sim_adj = self._simulation_spec.make(adjoint_sources)
            _require_no_meep_symmetries(sim_adj)
            native_accumulator = _NativeDesignAccumulator(
                self.design,
                self.gradient_components,
                native_history.signatures,
                False,
            )
            balance_timing["extra_seconds"] += _run_native_adjoint_loop(
                sim_adj,
                native_accumulator,
                field_histories,
                fine_step_count=fine_step_count,
                actual_time=actual_time,
                dt=dt,
                sampling_interval=sampling_interval,
                reconstruction_half_width=self.reconstruction_half_width,
                reconstruction_window=self.reconstruction_window,
                reconstruction_window_params=(
                    self.reconstruction_window_params
                ),
            )
            gradient_grid = native_accumulator.reduce()
            native_accumulator.release()
            native_accumulator = None

            if balance_enabled:
                self._simulation_spec.chunk_layout = (
                    self.chunk_balancer.rebalance(
                        sim_adj,
                        local_extra_work_seconds=(
                            balance_timing["extra_seconds"]
                        ),
                        local_wall_seconds=(
                            time.perf_counter() - balance_wall_start
                        ),
                        previous_measurements=tuple(
                            previous_measurements
                        ),
                    )
                )
                self.last_chunk_balance = self.chunk_balancer.last_report
            sim_adj.reset_meep()
            sim_adj = None
            if self._reuse_simulation_for_adjoint:
                sim_fwd = None
            gc.collect()

            gradient = (
                gradient_grid.real.flatten()
                * dt
                * self.design.material_factor
            )
            return objective_value, gradient
        finally:
            if native_accumulator is not None:
                native_accumulator.release()
            if native_history is not None:
                native_history.release_forward()
            monitor_objects.clear()
            gc.collect()
            try:
                try:
                    if sim_adj is not None and sim_adj is not sim_fwd:
                        sim_adj.reset_meep()
                finally:
                    if sim_fwd is not None:
                        sim_fwd.reset_meep()
            finally:
                if native_history is not None:
                    native_history.cleanup_memmaps()
                gc.collect()

    def evaluate(
        self,
        x: np.ndarray,
        need_gradient: bool = True,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Run forward/adjoint Meep simulations for one design vector.

        Args:
            x: Flat design vector passed to `update_design`.
            need_gradient: Whether to run the adjoint simulation.

        Returns:
            `(objective_value, gradient)`. The gradient is a flat real array
            from the objective model, or `None` for value-only evaluation.
        """
        if need_gradient and self.design is None:
            raise ValueError(
                "TDAObjective gradient evaluation requires design=DesignGrid(...)"
            )
        if (
            need_gradient
            and self._uses_simulation_spec
            and self._simulation_spec.m != self._configured_simulation_m
        ):
            raise ValueError(
                "SimulationSpec.m must not change after TDAObjective construction"
            )
        if need_gradient and not (
            self._is_flux_target or self._is_eigenmode_target
        ):
            _validate_logical_adjoint_source_amplitudes(
                (self.objective.adjoint_source_amplitude,)
            )
        if (
            need_gradient
            and self.adjoint_signal_fn is not None
            and (self._uses_default_fom or self.fom_fn is None)
        ):
            raise ValueError(
                "adjoint_signal_fn requires fom_fn so the objective value "
                "and derivative remain consistent"
            )
        if (
            need_gradient
            and self._is_regional_target
            and not self._uses_simulation_spec
        ):
            raise ValueError(
                "FieldRegionTarget gradient evaluation requires "
                "simulation=SimulationSpec(...)"
            )
        forward_sources = self._forward_sources_for_evaluation()
        if self._is_flux_target or self._is_eigenmode_target:
            if need_gradient and not self._uses_simulation_spec:
                raise ValueError(
                    f"{type(self.objective).__name__} gradient evaluation requires "
                    "simulation=SimulationSpec(...)"
                )
            return self._evaluate_mixed_surface(
                x,
                need_gradient=need_gradient,
                forward_sources=forward_sources,
            )
        self.update_design(x)
        target_positions = self.objective.positions
        if self._is_regional_target:
            effective_adjoint_source_sizes = tuple(
                mp.Vector3() for _ in target_positions
            )
            effective_adjoint_source_amplitudes = tuple(
                self.objective.adjoint_source_amplitude
                for _ in target_positions
            )
            indexed_source_mask = tuple(
                bool(self._uses_simulation_spec)
                for _ in target_positions
            )
        else:
            effective_adjoint_source_sizes = (
                self.objective.adjoint_source_size,
            )
            effective_adjoint_source_amplitudes = (
                self.objective.adjoint_source_amplitude,
            )
            indexed_source_mask = (
                bool(
                    self._uses_simulation_spec
                    and (
                        self._is_cylindrical
                        or (
                            _is_zero_source_size(
                                self.objective.adjoint_source_size
                            )
                            and getattr(
                                self.chunk_balancer,
                                "source_boundary_mode",
                                "auto",
                            )
                            != "finite"
                        )
                    )
                ),
            )
        if self.chunk_balancer is not None:
            if self._is_regional_target:
                source_boundary_decision = (
                    self.chunk_balancer.prepare_initial_layout(
                        self._simulation_spec,
                        forward_sources=forward_sources,
                        adjoint_source_positions=target_positions,
                        adjoint_source_sizes=effective_adjoint_source_sizes,
                        adjoint_source_amplitudes=(
                            effective_adjoint_source_amplitudes
                        ),
                        indexed_source_mask=indexed_source_mask,
                    )
                )
            else:
                source_boundary_decision = (
                    self.chunk_balancer.prepare_initial_layout(
                        self._simulation_spec,
                        forward_sources=forward_sources,
                        adjoint_source_positions=target_positions,
                        adjoint_source_size=self.objective.adjoint_source_size,
                        adjoint_source_amplitude=(
                            self.objective.adjoint_source_amplitude
                        ),
                        indexed_source_mask=indexed_source_mask,
                    )
                )
            self.last_source_boundary_decision = source_boundary_decision
            effective_adjoint_source_sizes = (
                source_boundary_decision.source_sizes
            )
            effective_adjoint_source_amplitudes = (
                source_boundary_decision.source_amplitudes
            )
        if any(
            indexed
            and not _is_zero_source_size(source_size)
            for indexed, source_size in zip(
                indexed_source_mask,
                effective_adjoint_source_sizes,
            )
        ):
            raise RuntimeError(
                "indexed adjoint target acquired a nonzero effective source size"
            )

        native_history = None
        native_accumulator = None
        sim_fwd = None
        sim_adj = None
        fwd_monitor = None
        adjoint_indexed_stencils = None
        try:
            balance_enabled = bool(
                need_gradient
                and self.chunk_balancer is not None
                and getattr(
                    self.chunk_balancer,
                    "needs_rebalance",
                    not getattr(self.chunk_balancer, "is_frozen", False),
                )
            )
            balance_wall_start = time.perf_counter() if balance_enabled else None
            sim_fwd = self._make_forward_simulation(forward_sources)
            _require_no_meep_symmetries(sim_fwd)
            is_cylindrical = _is_cylindrical_simulation(sim_fwd)
            forward_mode = getattr(sim_fwd, "m", 0)
            forward_mode = 0 if forward_mode is None else forward_mode
            if is_cylindrical:
                for target_position in target_positions:
                    _validate_cylindrical_target_position(target_position)
            if need_gradient and is_cylindrical:
                if not self._uses_simulation_spec:
                    raise ValueError(
                        "cylindrical gradient evaluation requires "
                        "simulation=SimulationSpec(...)"
                    )
                if forward_mode not in (-1, 0, 1):
                    raise ValueError(
                        "cylindrical gradient evaluation currently supports "
                        "m=-1, 0, or +1"
                    )
                for target_position in target_positions:
                    _validate_cylindrical_axis_target(
                        self.objective.component,
                        forward_mode,
                        target_position,
                    )
            dt = self.time_step(sim_fwd)
            run_history_dtype = history_storage_dtype(
                sim_fwd,
                self.history_dtype,
            )
            if need_gradient:
                native_history = _NativeDesignHistorySet(
                    self.design,
                    self.gradient_components,
                    run_history_dtype,
                    lambda shape: self._make_history_memmap(
                        shape,
                        dtype=run_history_dtype,
                    ),
                )
            sampling_interval = self._resolve_sampling_interval(dt)
            self.last_sampling_interval = sampling_interval
            dt_eff = dt * sampling_interval
            run_until = _aligned_run_until(
                self.t_final,
                dt=dt,
                sampling_interval=(sampling_interval if need_gradient else 1),
            )
            monitor_times = []
            forward_sampling_interval = (
                sampling_interval if need_gradient else 1
            )
            use_native_forward = hasattr(
                getattr(sim_fwd, "fields", None),
                "this",
            )
            if use_native_forward:
                native_fine_step_count = _native_forward_step_count(
                    sim_fwd,
                    run_until,
                    forward_sampling_interval,
                )
                n_expected = (
                    native_fine_step_count // forward_sampling_interval + 1
                )
                n_monitor_expected = native_fine_step_count + 1
            else:
                n_expected = int(np.ceil(run_until / dt_eff)) + 8
                n_monitor_expected = int(np.ceil(run_until / dt)) + 8
            monitor_history = np.empty(
                (n_monitor_expected, len(target_positions)),
                dtype=run_history_dtype,
            )

            sample_count = {"count": 0}
            fwd_count = {"value": 0}
            fwd_monitor = {"obj": None, "indexed_stencil": None}
            balance_timing = {"extra_seconds": 0.0}

            if use_native_forward:
                native_setup_start = (
                    time.perf_counter() if balance_enabled else None
                )
                fwd_monitor["obj"] = FastPointMonitor(
                    sim_fwd,
                    self.objective.component,
                    target_positions,
                )
                fwd_monitor["indexed_stencil"] = (
                    fwd_monitor["obj"].indexed_transpose_stencil()
                )
                _validate_indexed_monitor_support(
                    sim_fwd,
                    (self.objective.component,) * len(target_positions),
                    fwd_monitor["indexed_stencil"][0],
                )
                native_setup_seconds = (
                    time.perf_counter() - native_setup_start
                    if balance_enabled
                    else 0.0
                )
                (
                    sample_count["count"],
                    fwd_count["value"],
                    monitor_times,
                    native_forward_seconds,
                ) = _run_native_forward_loop(
                    sim_fwd,
                    ((fwd_monitor["obj"], monitor_history),),
                    native_history if need_gradient else None,
                    run_until=run_until,
                    sampling_interval=forward_sampling_interval,
                    fine_step_count=native_fine_step_count,
                )
                if balance_enabled:
                    balance_timing["extra_seconds"] += (
                        native_setup_seconds + native_forward_seconds
                    )
            else:
                def record_fwd_impl(s):
                    nonlocal monitor_history
                    sample_index = sample_count["count"]
                    if sample_index >= monitor_history.shape[0]:
                        grown_history = np.empty(
                            (
                                2 * monitor_history.shape[0],
                                monitor_history.shape[1],
                            ),
                            dtype=monitor_history.dtype,
                        )
                        grown_history[:sample_index] = monitor_history
                        monitor_history = grown_history
                    if fwd_monitor["obj"] is None:
                        fwd_monitor["obj"] = FastPointMonitor(
                            s,
                            self.objective.component,
                            target_positions,
                        )
                        fwd_monitor["indexed_stencil"] = (
                            fwd_monitor["obj"].indexed_transpose_stencil()
                        )
                        _validate_indexed_monitor_support(
                            s,
                            (self.objective.component,) * len(target_positions),
                            fwd_monitor["indexed_stencil"][0],
                        )
                    fwd_monitor["obj"].sample_history_into(
                        monitor_history[sample_index]
                    )
                    monitor_times.append(
                        _sample_time_from_simulation(
                            s,
                            sample_index,
                            dt,
                        )
                    )
                    if need_gradient and sample_index % sampling_interval == 0:
                        native_history.sample_forward(
                            s,
                            fwd_count["value"],
                            n_expected,
                        )
                        fwd_count["value"] += 1
                    sample_count["count"] += 1

                record_fwd = record_fwd_impl
                if balance_enabled:
                    def record_fwd(s):
                        start = time.perf_counter()
                        try:
                            record_fwd_impl(s)
                        finally:
                            balance_timing["extra_seconds"] += time.perf_counter() - start

                sim_fwd.run(record_fwd, until=run_until)

            actual_time = sim_fwd.round_time()
            monitor_history = monitor_history[:sample_count["count"]]
            monitor_history = fwd_monitor["obj"].reduce_history(
                monitor_history
            )
            full_monitor_times = np.asarray(monitor_times, dtype=float)
            if need_gradient:
                field_histories = native_history.finish_forward(fwd_count["value"])
                fine_step_count = sample_count["count"] - 1
                expected_history_rows = fine_step_count // sampling_interval + 1
                if (
                    fine_step_count % sampling_interval != 0
                    or any(
                        history.shape[0] != expected_history_rows
                        for history in field_histories.values()
                    )
                ):
                    raise RuntimeError(
                        "forward callback count is not aligned with the Nyquist "
                        "sampling interval"
                    )
            monitor_stop = _monitor_history_stop(
                full_monitor_times,
                self.t_final,
            )
            monitor_history = np.asarray(
                monitor_history[:monitor_stop],
                dtype=run_history_dtype,
            )
            if self._is_regional_target:
                monitor_history = self.objective.reshape_history(
                    monitor_history
                )
            else:
                monitor_history = monitor_history[:, 0]
            monitor_times = full_monitor_times[:monitor_stop]
            if any(indexed_source_mask):
                offsets, source_data, weights = fwd_monitor["indexed_stencil"]
                if offsets.shape != (len(target_positions) + 1,):
                    raise RuntimeError(
                        "TDAObjective indexed monitor stencil count does not "
                        "match its target positions"
                    )
                adjoint_indexed_stencils = []
                for point_index, indexed in enumerate(
                    indexed_source_mask
                ):
                    if not indexed:
                        adjoint_indexed_stencils.append(None)
                        continue
                    point_slice = slice(
                        int(offsets[point_index]),
                        int(offsets[point_index + 1]),
                    )
                    adjoint_indexed_stencils.append(
                        (
                            source_data[point_slice],
                            weights[point_slice],
                        )
                    )
            fwd_timing = self.chunk_balancer.capture_timing(sim_fwd) if balance_enabled else None
            fwd_monitor["obj"] = None
            if native_history is not None:
                native_history.release_forward()
            if not (need_gradient and self._reuse_simulation_for_adjoint):
                sim_fwd.reset_meep()
                sim_fwd = None
            gc.collect()

            objective_value = None
            adjoint_signal = None
            if need_gradient:
                objective_value, adjoint_signal = self._fom_value_and_adjoint_signal(monitor_history, dt)
            else:
                objective_value = float(self.fom_fn(monitor_history, dt))

            gradient = None
            if need_gradient:
                histories_too_short = any(
                    history.shape[0] < 2 for history in field_histories.values()
                )
                if histories_too_short:
                    raise RuntimeError(
                        "TDAObjective gradient requires at least two field samples; "
                        "increase t_final or decrease sampling_interval"
                    )
                if monitor_history.shape[0] < 4:
                    raise RuntimeError(
                        "TDAObjective gradient requires at least four monitor samples "
                        "for cubic adjoint-source interpolation; increase t_final"
                    )

                if self._is_regional_target:
                    adjoint_sources = self.objective.adjoint_sources(
                        adjoint_signal,
                        monitor_times,
                        actual_time,
                        source_amplitudes=(
                            effective_adjoint_source_amplitudes
                        ),
                        indexed_stencils=adjoint_indexed_stencils,
                    )
                else:
                    adjoint_sources = self.objective.adjoint_sources(
                        adjoint_signal,
                        monitor_times,
                        actual_time,
                        source_size=effective_adjoint_source_sizes[0],
                        source_amplitude=(
                            effective_adjoint_source_amplitudes[0]
                        ),
                        indexed_stencil=(
                            None
                            if adjoint_indexed_stencils is None
                            else adjoint_indexed_stencils[0]
                        ),
                    )
                del monitor_history
                gc.collect()

                if self._reuse_simulation_for_adjoint:
                    if is_cylindrical:
                        sim_fwd.change_m(-forward_mode)
                    sim_fwd.restart_fields()
                    sim_fwd.clear_dft_monitors()
                    if balance_enabled:
                        # Meep timers are cumulative across field restarts.
                        sim_fwd.fields.reset_timers()
                    sim_fwd.change_sources(adjoint_sources)
                    sim_adj = sim_fwd
                    sim_fwd = None
                else:
                    if is_cylindrical:
                        sim_adj = self._simulation_spec.make(
                            adjoint_sources,
                            m=-forward_mode,
                        )
                    else:
                        sim_adj = self.sim_factory(adjoint_sources)
                _require_no_meep_symmetries(sim_adj)
                _validate_adjoint_source_amplitudes(
                    sim_adj,
                    effective_adjoint_source_amplitudes,
                )
                native_accumulator = _NativeDesignAccumulator(
                    self.design,
                    self.gradient_components,
                    native_history.signatures,
                    self._native_adjoint_midpoint,
                )
                balance_timing["extra_seconds"] += _run_native_adjoint_loop(
                    sim_adj,
                    native_accumulator,
                    field_histories,
                    fine_step_count=fine_step_count,
                    actual_time=actual_time,
                    dt=dt,
                    sampling_interval=sampling_interval,
                    reconstruction_half_width=(
                        self.reconstruction_half_width
                    ),
                    reconstruction_window=self.reconstruction_window,
                    reconstruction_window_params=(
                        self.reconstruction_window_params
                    ),
                )
                grad_grid = native_accumulator.reduce()
                native_accumulator.release()
                gc.collect()
                if balance_enabled:
                    self._simulation_spec.chunk_layout = self.chunk_balancer.rebalance(
                        sim_adj,
                        local_extra_work_seconds=balance_timing["extra_seconds"],
                        local_wall_seconds=time.perf_counter() - balance_wall_start,
                        previous_measurements=(fwd_timing,),
                    )
                    self.last_chunk_balance = self.chunk_balancer.last_report
                sim_adj.reset_meep()
                sim_adj = None
                del adjoint_sources
                del field_histories
                gc.collect()

                gradient = (
                    grad_grid.real.flatten()
                    * dt
                    * self.design.material_factor
                )
                del grad_grid
                gc.collect()
            else:
                del monitor_history
                gc.collect()

            return objective_value, gradient
        finally:
            if native_accumulator is not None:
                native_accumulator.release()
            if native_history is not None:
                native_history.release_forward()
            if fwd_monitor is not None:
                fwd_monitor["obj"] = None
            gc.collect()
            try:
                try:
                    if sim_adj is not None:
                        sim_adj.reset_meep()
                finally:
                    if sim_fwd is not None:
                        sim_fwd.reset_meep()
            finally:
                if native_history is not None:
                    native_history.cleanup_memmaps()
                gc.collect()
