"""External discrete adjoint using unchanged, version-pinned FDTDX kernels.

Only electric design histories are retained. FDTD, reconstruction and gradient
contraction remain on the selected device; CPU offload changes history storage.
"""
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .targets import Band, make_loss
from tama.nyquist import _windowed_sinc_stencils


class AccuracyError(RuntimeError):
    """An explicit gradient validation failed its reference check."""


@lru_cache(maxsize=1)
def verify_fdtdx_sources():
    """Refuse unreviewed engine versions or modified Python kernels."""
    import fdtdx
    if importlib.metadata.version('fdtdx') != '0.6.2':
        raise RuntimeError('This backend is validated only for FDTDX 0.6.2')
    root = Path(fdtdx.__file__).parent
    expected = json.loads(Path(__file__).with_name('fdtdx_source_sha256.json').read_text())
    mismatches = [name for name, digest in expected.items()
                  if not (root / name).is_file()
                  or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise RuntimeError(f'FDTDX sources differ from the official wheel: {mismatches}')
    return True


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


class MultiTDAObjective:
    """Evaluate a scalarized multi-band objective at the supplied current density.

    Native FDTDX geometry/source/boundary definitions enter through SimulationSpec.
    Targets use native integer grid indices, not Meep coordinates. A call returns
    (value, gradient), or (value, gradient, info) with return_info=True.

    stride=1 is the exact discrete adjoint. Larger strides reconstruct the
    electric history from sparse samples without an additional dense run.
    """

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False) and name != 'last_info':
            raise AttributeError('Objective configuration is immutable; construct a new objective for changed settings')
        object.__setattr__(self, name, value)

    def __init__(self, simulation_spec, design_grid, targets, bands=None,
                 scalarization_fn=None, *, stride=1, block_steps=64,
                 half_width=64, cpu_offload=False):
        verify_fdtdx_sources()
        from .specs import SimulationSpec, DesignGrid
        from .targets import PointTarget, FieldRegionTarget, FluxTarget, EigenmodeCoefficientTarget
        if type(simulation_spec) is not SimulationSpec or type(design_grid) is not DesignGrid:
            raise TypeError('Use the validated SimulationSpec and DesignGrid classes')
        self.spec, self.design = simulation_spec, design_grid
        self.design.validate(self.spec)
        self.targets = tuple(targets)
        if not self.targets:
            raise ValueError('At least one target is required')
        for target in self.targets:
            if type(target) not in (PointTarget, FieldRegionTarget, FluxTarget, EigenmodeCoefficientTarget):
                raise TypeError('Custom target sampling is not validated; use Band.fom_fn for nonlinear objectives')
            target.validate_shape(self.spec.shape)
        self.bands = tuple(bands) if bands is not None else tuple(Band(i) for i in range(len(self.targets)))
        self.stride = _positive_int(stride, 'stride')
        self.block_steps = _positive_int(block_steps, 'block_steps')
        self.half_width = _positive_int(half_width, 'half_width')
        if self.block_steps % self.stride or self.spec.time_steps % self.stride:
            raise ValueError('block_steps and simulation time steps must be divisible by stride')
        if type(cpu_offload) is not bool:
            raise TypeError('cpu_offload must be a Python bool')
        self.cpu_offload = cpu_offload
        self._loss = make_loss(self.targets, self.bands, self.spec.dt, scalarization_fn)
        self._loss_grad = jax.jit(jax.value_and_grad(self._loss))
        self._kernels = {}
        self._material = jax.jit(self.design.inverse_permittivity)
        self._pullback = jax.jit(lambda rho, g: jax.vjp(self.design.inverse_permittivity, rho)[1](g)[0])
        self.last_info = None
        self._sealed = True

    def _density(self, rho):
        value = np.asarray(rho)
        if value.shape != tuple(self.design.parameter_shape):
            raise ValueError(f'rho must have shape {self.design.parameter_shape}, got {value.shape}')
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError('rho must be a finite real floating array')
        if np.any((value < 0) | (value > 1)):
            raise ValueError('rho must lie in [0, 1]')
        device = next(iter(self.spec.arrays.fields.E.devices()))
        rho = jax.device_put(jnp.asarray(value, dtype=self.spec.arrays.inv_permittivities.dtype), device)
        q = self._material(rho)
        host_q = np.asarray(q)
        if host_q.shape != (1, *self.design.shape) or not np.isfinite(host_q).all() or np.any(host_q <= 0):
            raise ValueError('The design map must return positive finite scalar inverse permittivity')
        tolerance = 16 * np.finfo(host_q.dtype).eps
        if host_q.min() < (1 / self.design.epsilon_high) * (1 - tolerance) or host_q.max() > (1 / self.design.epsilon_low) * (1 + tolerance):
            raise ValueError('The design transform must keep density within [0, 1]')
        self.spec._validate_courant(max(float(host_q.max()), float(np.asarray(self.spec.arrays.inv_permittivities).max())),
                                  float(np.asarray(self.spec.arrays.inv_permeabilities).max()))
        return rho, self.spec.arrays.inv_permittivities.at[self.design.region].set(q)

    def _make_kernels(self, stride):
        from fdtdx.core.physics.curl import interpolate_fields
        from fdtdx.fdtd.forward import forward
        from fdtdx.fdtd.update import pad_fields_for_boundaries, update_E, update_H

        base, placed, config, key = self.spec.arrays, self.spec.objects, self.spec.config, self.spec.key
        region, chunk = self.design.region, self.block_steps
        zero = jax.tree.map(jnp.zeros_like, base.fields)
        local_shape = zero.E[region].shape
        coarse = chunk // stride
        first, weights = _windowed_sinc_stencils(stride, self.half_width, 'kaiser', {'beta': 6.0})
        offsets = first + jnp.arange(weights.shape[1])
        weights = jnp.asarray(weights, dtype=zero.E.dtype)
        left, right = (0, 0) if stride == 1 else (self.half_width - 1, self.half_width)

        def measurement(e, hp, hn):
            ee, hh = interpolate_fields(pad_fields_for_boundaries(e, placed, config),
                pad_fields_for_boundaries((hp + hn) / 2, placed, config))
            return tuple(target.sample(ee, hh) for target in self.targets)

        prototype, observe_pullback = jax.vjp(measurement, zero.E, zero.H, zero.H)

        @jax.jit
        def forward_chunk(state, inv, start, length):
            saved = jnp.zeros((coarse + 1, *local_shape), zero.E.dtype).at[0].set(state.E[region])
            traces = tuple(jnp.zeros((chunk, *y.shape), y.dtype) for y in prototype)

            def step(j, carry):
                state, saved, traces = carry
                current = base.aset('fields', state).aset('inv_permittivities', inv)
                _, nxt = forward((start + j, current), config, placed, key,
                    record_detectors=False, record_boundaries=False, simulate_boundaries=True)
                observed = measurement(nxt.fields.E, state.H, nxt.fields.H)
                traces = tuple(y.at[j].set(v) for y, v in zip(traces, observed))
                saved = jax.lax.cond((j + 1) % stride == 0,
                    lambda a: a.at[(j + 1) // stride].set(nxt.fields.E[region]), lambda a: a, saved)
                return nxt.fields, saved, traces

            return jax.lax.fori_loop(0, length, step, (state, saved, traces))

        @jax.jit
        def reverse_chunk(cot, gq, inv, saved, dy, start, length):
            lower = start // stride - left
            q = inv[region]

            def reconstructed(t):
                k, phase = t // stride - lower, t % stride
                if stride == 1:
                    return saved[k]
                return jax.lax.cond(phase == 0, lambda: saved[k],
                    lambda: jnp.tensordot(weights[phase], saved[k + offsets], axes=1))

            def step(j, carry):
                cot, gq = carry
                local_t = length - 1 - j
                t = start + local_t
                qe, qhp, qhn = observe_pullback(tuple(y[local_t] for y in dy))
                cot = cot.aset('E', cot.E + qe).aset('H', cot.H + qhn)
                current = base.aset('inv_permittivities', inv)
                h_step = lambda fields: update_H(t, current.aset('fields', fields), placed, config, True).fields
                e_step = lambda fields: update_E(t, current.aset('fields', fields), placed, config, True).fields
                mid = jax.vjp(h_step, zero)[1](cot)[0]
                delta = reconstructed(t + 1) - reconstructed(t)
                gq = gq + jnp.sum(jnp.real(mid.E[region] * delta / q), axis=0, keepdims=True)
                prev = jax.vjp(e_step, zero)[1](mid)[0]
                return prev.aset('H', prev.H + qhp), gq

            return jax.lax.fori_loop(0, length, step, (cot, gq))

        # This independent AD reference differentiates the full native step state,
        # including PML, instead of using the E-increment contraction above.
        @jax.jit
        def reference_value(rho):
            import equinox.internal as eqxi
            inv = base.inv_permittivities.at[region].set(self.design.inverse_permittivity(rho))
            traces = tuple(jnp.zeros((self.spec.time_steps, *y.shape), y.dtype) for y in prototype)

            def step(carry):
                t, state, traces = carry
                current = base.aset('fields', state).aset('inv_permittivities', inv)
                _, nxt = forward((t, current), config, placed, key,
                    record_detectors=False, record_boundaries=False, simulate_boundaries=True)
                observed = measurement(nxt.fields.E, state.H, nxt.fields.H)
                traces = tuple(y.at[t].set(v) for y, v in zip(traces, observed))
                return t + 1, nxt.fields, traces

            _, _, traces = eqxi.while_loop(lambda carry: carry[0] < self.spec.time_steps,
                step, (jnp.int32(0), zero, traces), max_steps=self.spec.time_steps,
                kind='checkpointed', checkpoints=8)
            return self._loss(traces)

        return dict(forward=forward_chunk, reverse=reverse_chunk, reference=reference_value,
                    reference_grad=jax.jit(jax.value_and_grad(reference_value)),
                    zero=zero, coarse=coarse, left=left, right=right,
                    window_rows=coarse + left + right + 1, local_shape=local_shape)

    def _get_kernels(self, stride):
        if stride not in self._kernels:
            self._kernels[stride] = self._make_kernels(stride)
        return self._kernels[stride]

    def _run(self, rho, inv, stride, cpu_offload):
        kernels = self._get_kernels(stride)
        steps, chunk = self.spec.time_steps, self.block_steps
        nblocks, b = (steps + chunk - 1) // chunk, kernels['coarse']
        device = next(iter(inv.devices()))
        dev = jax.sharding.SingleDeviceSharding(device, memory_kind='device')
        if cpu_offload and device.platform != 'gpu':
            raise ValueError('CPU offload requires a CUDA GPU execution device')
        host = dev.with_memory_kind('pinned_host') if cpu_offload else None
        state = kernels['zero']
        blocks, traces = [], []
        d2h_bytes = h2d_bytes = 0
        started = time.perf_counter()
        for i in range(nblocks):
            length = min(chunk, steps - i * chunk)
            state, saved, trace = kernels['forward'](state, inv, jnp.int32(i * chunk), jnp.int32(length))
            stored = jax.device_put(saved, host) if cpu_offload else saved
            if cpu_offload:
                d2h_bytes += saved.nbytes
            jax.block_until_ready((state, stored, trace))
            blocks.append(stored)
            traces.append(trace)
            del saved, stored
        forward_s = time.perf_counter() - started
        del state
        y = tuple(jnp.concatenate([block[t] for block in traces], axis=0)[:steps]
                  for t in range(len(self.targets)))
        value, dy = self._loss_grad(y)
        jax.block_until_ready((value, dy))
        loss_end = time.perf_counter()
        cot = kernels['zero']
        gq = jnp.zeros_like(inv[self.design.region])
        cache, max_cache_bytes = {}, 0
        maximum_index = steps // stride
        for i in reversed(range(nblocks)):
            lo, hi = i * b - kernels['left'], i * b + b + kernels['right'] + 1
            v_lo, v_hi = max(0, lo), min(maximum_index + 1, hi)
            first_block = min(v_lo // b, nblocks - 1)
            last_block = min((v_hi - 1) // b, nblocks - 1)
            needed = set(range(first_block, last_block + 1))
            for old in set(cache) - needed:
                del cache[old]
            for idx in sorted(needed):
                if idx not in cache:
                    cache[idx] = jax.device_put(blocks[idx], dev) if cpu_offload else blocks[idx]
                    if cpu_offload:
                        h2d_bytes += blocks[idx].nbytes
            max_cache_bytes = max(max_cache_bytes, sum(x.nbytes for x in cache.values()))
            pieces = []
            if lo < 0:
                pieces.append(jnp.zeros((-lo, *kernels['local_shape']), kernels['zero'].E.dtype))
            for idx in sorted(needed):
                start = max(v_lo, idx * b) - idx * b
                end = min(v_hi, (idx + 1) * b if idx < nblocks - 1 else maximum_index + 1) - idx * b
                pieces.append(cache[idx][start:end])
            if hi > maximum_index + 1:
                pieces.append(jnp.zeros((hi - maximum_index - 1, *kernels['local_shape']), kernels['zero'].E.dtype))
            window = jnp.concatenate(pieces, axis=0)
            if window.shape[0] != kernels['window_rows']:
                raise RuntimeError('Internal history window length mismatch')
            length = min(chunk, steps - i * chunk)
            cot, gq = kernels['reverse'](cot, gq, inv, window,
                tuple(v[i * chunk:(i + 1) * chunk] for v in dy), jnp.int32(i * chunk), jnp.int32(length))
            jax.block_until_ready((cot, gq))
            del pieces, window
        gradient = self._pullback(rho, gq)
        jax.block_until_ready((value, gradient))
        if not np.isfinite(float(value)) or not np.isfinite(np.asarray(gradient)).all():
            raise FloatingPointError('Nonfinite objective/gradient; check stability, source and objective scaling')
        info = dict(stride=stride, cpu_offload=cpu_offload,
            history_memory_kind=blocks[0].sharding.memory_kind,
            history_bytes=sum(x.nbytes for x in blocks), reverse_cache_max_bytes=max_cache_bytes,
            d2h_bytes=d2h_bytes, h2d_bytes=h2d_bytes, forward_s=forward_s,
            loss_s=loss_end - started - forward_s, total_s=time.perf_counter() - started,
            fdtdx_sources_verified=True, target_history_bytes=sum(v.nbytes for v in y))
        return value, gradient, info

    def evaluate(self, rho, *, cpu_offload=None, return_info=False):
        """Return the objective and gradient for this rho, without stale material captures."""
        offload = self.cpu_offload if cpu_offload is None else cpu_offload
        if type(offload) is not bool:
            raise TypeError('cpu_offload must be a Python bool')
        rho, inv = self._density(rho)
        value, gradient, info = self._run(rho, inv, self.stride, offload)
        self.last_info = info
        return (value, gradient, info) if return_info else (value, gradient)

    def value(self, rho):
        """Forward value for line searches or independent finite differences."""
        rho, _ = self._density(rho)
        return self._get_kernels(1)['reference'](rho)

    def reference_value_and_grad(self, rho):
        """Expensive checkpointed AD of complete native steps, for validation."""
        rho, _ = self._density(rho)
        return self._get_kernels(1)['reference_grad'](rho)

    def validate(self, rho, *, directions=3, step=None):
        """Check this design against full-state AD and centered finite differences.

        This checks the actual callback and density transform, including cases
        where a user-supplied custom derivative differs from its forward value.
        A successful check is evidence at this design, not a global error bound.
        Use an interior density for two-sided finite differences.
        """
        directions = _positive_int(directions, 'directions')
        rho, inv = self._density(rho)
        single = inv.dtype == jnp.float32
        step = (2e-3 if single else 2e-5) if step is None else float(step)
        if not np.isfinite(step) or step <= 0:
            raise ValueError('finite-difference step must be positive and finite')
        value, gradient, _ = self._run(rho, inv, 1, self.cpu_offload)
        native_value, native_gradient = self.reference_value_and_grad(rho)
        if not np.isfinite(float(native_value)) or not np.isfinite(np.asarray(native_gradient)).all():
            raise AccuracyError('Native reference returned nonfinite values')
        norm = float(np.linalg.norm(np.asarray(native_gradient, dtype=np.float64)))
        difference = float(np.linalg.norm(np.asarray(gradient, dtype=np.float64) - np.asarray(native_gradient, dtype=np.float64)))
        relative = difference / norm if norm else (0. if difference == 0 else float('inf'))
        rtol = 3e-5 if single else 2e-10
        if not np.isfinite(relative) or relative > rtol or not np.isclose(float(value), float(native_value), rtol=rtol, atol=0):
            raise AccuracyError(f'Dense adjoint disagrees with complete-state AD: relative gradient error {relative:g}')
        random = np.random.default_rng(19)
        fd_errors = []
        for _ in range(directions):
            direction = random.normal(size=rho.shape)
            direction = jnp.asarray(direction / np.linalg.norm(direction), dtype=rho.dtype)
            if np.any(np.asarray(rho - step * jnp.abs(direction)) < 0) or np.any(np.asarray(rho + step * jnp.abs(direction)) > 1):
                raise ValueError('Use an interior rho or a smaller step for two-sided finite differences')
            fd = float((self.value(rho + step * direction) - self.value(rho - step * direction)) / (2 * step))
            prediction = float(jnp.sum(gradient * direction))
            if not np.isfinite(fd) or not np.isfinite(prediction):
                raise AccuracyError('Finite-difference reference returned nonfinite values')
            scale = max(abs(fd), abs(prediction), np.finfo(np.dtype(rho.dtype)).eps * norm)
            error = abs(fd - prediction) / scale if scale else 0.
            fd_errors.append(error)
        if not np.isfinite(fd_errors).all() or max(fd_errors) > (1e-2 if single else 3e-5):
            raise AccuracyError(f'Centered finite differences failed: relative errors {fd_errors}')
        return dict(passed=True, dense_relative_gradient_error=relative,
                    finite_difference_relative_errors=fd_errors, finite_difference_step=step)

    def scipy_fun(self, flat_rho):
        """Adapter for scipy.optimize.minimize(..., jac=True, bounds=...)."""
        rho = np.asarray(flat_rho).reshape(self.design.parameter_shape)
        value, gradient = self.evaluate(rho)
        return float(value), np.asarray(gradient).ravel().astype(float)


class TDAObjective(MultiTDAObjective):
    """Single-target convenience interface; the same validated execution path."""

    def __init__(self, simulation_spec, design_grid, target, *, kernel=None,
                 fom_fn=None, **kwargs):
        super().__init__(simulation_spec, design_grid, [target],
            [Band(0, kernel=kernel, fom_fn=fom_fn)], **kwargs)
