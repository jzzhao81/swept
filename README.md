# swept
Strongly correlated WanniEr Python Tools

## Testing

Run the serial core test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

To exercise MPI collectives as well:

```bash
mpiexec -n 2 python -m unittest discover -s tests -v
```

## Input-file driver

```bash
mpirun -np N python /path/to/swept/run.py input.toml
```

The pipeline is fixed: read hr -> add spin -> eshift -> magmom axis ->
rotate_spin -> SOC -> tasks (`udmft`, `bands`, `dos`, run in the order
written). `[shell] start` is the 0-based orbital index in the hr file.
Every run writes `swept_run.log` with the swept version, the resolved
orbital indices and a copy of the input. See `examples/co3sn2s2.toml`.
