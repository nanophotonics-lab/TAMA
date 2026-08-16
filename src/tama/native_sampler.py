"""Import guard used when the required native sampler extension is absent."""

raise ImportError(
    "TAMA native sampler extension is not built; reinstall TAMA from "
    "the active Meep/MPI conda environment"
)
