# Running and Testing

## Running

Install TAMA in the active environment, then run scripts from any working
directory in that environment.

Single process:

```bash
python <path-to-your-script.py>
```

MPI:

```bash
mpirun -np 4 python <path-to-your-script.py>
```

## Testing

Run the test suite from a source checkout in the same Meep environment:

```bash
cd <tama-source-directory>
python -m pytest -q
```

Run the MPI marker groups with their matching rank counts:

```bash
mpirun -np 2 python -m pytest -q -p no:cacheprovider -m mpi2
mpirun -np 4 python -m pytest -q -p no:cacheprovider -m mpi4
mpirun -np 8 python -m pytest -q -p no:cacheprovider -m mpi8
```

The `mpi2` and `mpi4` groups run in the main CI workflow. The more expensive
`mpi8` group has a separate weekly or manually dispatched workflow.
