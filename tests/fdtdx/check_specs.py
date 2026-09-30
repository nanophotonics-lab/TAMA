"""Run native scene adaptation, material-map and rejection checks on CPU."""

from dataclasses import replace
import json
from pathlib import Path
import sys


import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from _scene import make_scene
from tama.backends.fdtdx.specs import DesignGrid, SimulationSpec


def rejected(fn, phrase):
    try:
        fn()
    except (TypeError, ValueError) as error:
        assert phrase in str(error), str(error)
        return
    raise AssertionError(f"expected rejection containing {phrase!r}")


def main():
    import fdtdx

    scene = make_scene(steps=64, backend="cpu")
    spec, design, rho = (scene[k] for k in ("spec", "design", "rho"))
    assert spec.time_steps == 64 and spec.dt == spec.config.time_step_duration
    assert spec.shape == (40, 32, 12)
    assert isinstance(spec.objects.object_list, tuple)
    assert design.parameter_shape == rho.shape == (4, 4, 2)
    assert design.inverse_permittivity(rho).shape == (1, 16, 16, 8)
    expected = 1 / (1 + 3 * np.asarray(rho))
    expected = expected.repeat(4, axis=0).repeat(4, axis=1).repeat(4, axis=2)
    np.testing.assert_array_equal(design.inverse_permittivity(rho)[0], expected)
    grad = jax.grad(lambda x: jnp.sum(design.inverse_permittivity(x)))(rho)
    np.testing.assert_allclose(grad, -64 * 3 / (1 + 3 * np.asarray(rho)) ** 2, rtol=1e-14)
    transformed = replace(design, transform=lambda x: x ** 2)
    grad_transform = jax.grad(lambda x: jnp.sum(transformed.inverse_permittivity(x)))(rho)
    np.testing.assert_allclose(grad_transform, -64 * 6 * np.asarray(rho) / (1 + 3 * np.asarray(rho) ** 2) ** 2,
                               rtol=1e-14)
    rejected(lambda: design.inverse_permittivity(rho.reshape(-1)), "rho must have shape")
    rejected(lambda: design.inverse_permittivity(jnp.ones(rho.shape, dtype=jnp.int32)), "real floating-point")
    rejected(lambda: replace(design, transform=lambda x: x.reshape(-1)).inverse_permittivity(rho), "preserve")
    rejected(lambda: replace(design, shape=(17, 16, 8)), "divisible")
    rejected(lambda: replace(design, start=(-1, 8, 2)), "nonnegative")
    rejected(lambda: replace(design, epsilon_low=0), "epsilon endpoints")
    rejected(lambda: replace(design, start=(30, 8, 2)).validate(spec), "outside")
    rejected(lambda: replace(design, start=(0, 8, 2)).validate(spec), "PML")
    rejected(lambda: replace(design, start=(9, 8, 2)).validate(spec), "source")
    rejected(lambda: replace(design, epsilon_low=.1).validate(spec), "CFL")
    rejected(lambda: replace(spec, arrays=spec.arrays.aset("electric_conductivity", jnp.zeros_like(spec.arrays.inv_permittivities))),
             "electric_conductivity")
    rejected(lambda: replace(spec, arrays=spec.arrays.aset("dispersive_c1", jnp.zeros((1,)))), "dispersive_c1")
    rejected(lambda: replace(spec, arrays=spec.arrays.aset("inv_permittivities", -spec.arrays.inv_permittivities)), "strictly positive")
    rejected(lambda: replace(spec, arrays=spec.arrays.aset("inv_permittivities", jnp.repeat(spec.arrays.inv_permittivities, 3, axis=0))),
             "one-component isotropic inverse permittivity")
    rejected(lambda: replace(spec, arrays=spec.arrays.aset("fields->E", spec.arrays.fields.E.astype(jnp.float32))), "float precision")
    rejected(lambda: replace(spec, arrays=spec.arrays.aset("alpha", spec.arrays.alpha.astype(jnp.float32))), "alpha dtype")
    rejected(lambda: replace(spec, key=jnp.ones((3,), dtype=jnp.uint32)), "PRNG key")
    rejected(lambda: replace(spec, arrays=spec.arrays.aset("inv_permeabilities", jnp.ones((3, *spec.shape)))), "permeability must")
    badsource = fdtdx.ModePlaneSource(name="mode", partial_grid_shape=(1, 20, 8), direction="+",
                                    wave_character=fdtdx.WaveCharacter(frequency=scene["frequency"]))
    rejected(lambda: replace(spec, objects=spec.objects.replace_sources([badsource])), "only native")
    class CustomUniform(fdtdx.UniformPlaneSource):
        pass
    badsource = CustomUniform(name="custom", partial_grid_shape=(1, 20, 8), direction="+",
                              wave_character=fdtdx.WaveCharacter(frequency=scene["frequency"]))
    rejected(lambda: replace(spec, objects=spec.objects.replace_sources([badsource])), "only native")
    cases = {}
    for kind in ("uniform", "gaussian", "dipole"):
        thin = make_scene(steps=32, periodic_z=True, backend="cpu", source_kind=kind)
        assert thin["spec"].shape == (40, 32, 2)
        assert thin["design"].shape == (16, 16, 2)
        assert thin["detector_shape"] == (1, 4, 2)
        cases[kind] = True
    for kind in ("pec", "pmc"):
        wall = make_scene(steps=32, backend="cpu", boundary_kind=kind)
        rejected(lambda: replace(wall["design"], start=(0, 8, 2)).validate(wall["spec"]), "clamped boundary")
    complex_scene = make_scene(steps=32, periodic_z=True, backend="cpu", bloch_vector=(0., 0., 1e6))
    assert complex_scene["spec"].arrays.fields.E.dtype == jnp.complex128
    assert complex_scene["spec"].arrays.detector_states["reference"]["fields"].dtype == jnp.complex128
    assert complex_scene["spec"].arrays.inv_permittivities.dtype == jnp.float64
    rejected(lambda: make_scene(dtype=jnp.float32, steps=32, backend="cpu"), "source _E cache precision")
    result = dict(material_map=True, transformed_gradient=True, invalid_scenes_rejected=True,
                  native_cartesian_3d=True, native_thin_periodic_sources=cases,
                  native_pec_pmc_initialization=True, native_complex_bloch_initialization=True)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
