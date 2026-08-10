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
