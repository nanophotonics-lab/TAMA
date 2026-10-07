"""Flat design-vector metadata for independent, nonoverlapping regions."""

import numpy as np

from .sampling_grid import (
    _native_design_components,
    _native_design_gradient_scale,
    _uses_material_jacobian,
)


def _resolve_designs(design, designs):
    if design is not None and designs is not None:
        raise ValueError("provide either design or designs, not both")
    regions = (design,) if design is not None else tuple(designs) if designs is not None else ()
    if designs is not None and not regions:
        raise ValueError("designs must contain at least one DesignGrid")
    if len(regions) < 2:
        return regions
    if len({id(region.material_grid) for region in regions}) != len(regions):
        raise ValueError("independent designs must use distinct MaterialGrid objects")
    if any((region.dimensions, region.is_cylindrical) !=
           (regions[0].dimensions, regions[0].is_cylindrical) for region in regions):
        raise ValueError("all designs must use the same dimensions and coordinate system")
    axes = (0, 2) if regions[0].is_cylindrical else tuple(range(regions[0].dimensions))
    for i, first in enumerate(regions):
        for second in regions[i + 1:]:
            if all(abs(first.center[axis] - second.center[axis]) <
                   (first.size[axis] + second.size[axis]) / 2 for axis in axes):
                raise ValueError("independent design regions must not overlap")
    return regions


class _DesignCollection:
    def __init__(self, designs):
        self.designs = tuple(designs)
        self.sizes = tuple(int(np.prod(region.shape)) for region in self.designs)
        self.offsets = tuple(np.cumsum((0,) + self.sizes[:-1]).tolist())
        self.shape = (sum(self.sizes),)
        self.dimensions = self.designs[0].dimensions
        self.is_cylindrical = self.designs[0].is_cylindrical

    def update_weights(self, values):
        values = np.asarray(values)
        if values.size != self.shape[0]:
            raise ValueError(f"design vector must contain {self.shape[0]} values")
        values = values.reshape(-1)
        for region, offset, size in zip(self.designs, self.offsets, self.sizes):
            region.update_weights(values[offset:offset + size])


def _collection_components(sim, design, components):
    if not isinstance(design, _DesignCollection):
        return _native_design_components(sim, design, components)
    return tuple((index, component)
                 for index, region in enumerate(design.designs)
                 for component in _native_design_components(sim, region, components))


def _collection_gradient_scale(sim, design):
    if not isinstance(design, _DesignCollection):
        return _native_design_gradient_scale(sim, design)
    return np.concatenate([
        np.full(size, _native_design_gradient_scale(sim, region))
        for region, size in zip(design.designs, design.sizes)
    ])


def _collection_uses_material_jacobian(sim, design):
    if isinstance(design, _DesignCollection):
        return any(_uses_material_jacobian(sim, region) for region in design.designs)
    return _uses_material_jacobian(sim, design)


def _region_component(design, key):
    if isinstance(design, _DesignCollection):
        index, component = key
        return design.designs[index], component, design.offsets[index]
    return design, key, 0
