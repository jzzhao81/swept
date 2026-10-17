"""MPI fault-injection worker, launched by test_mpi_faults.py as `mpiexec -n N python -B -u
mpi_fault_worker.py CASE` (not a test module itself).

Every rank prints one line `MARK {json}` with its rank and outcome ("returned", the name of
the exception it caught, or "status S" for run.main) and exits 0, so the parent can check
each rank.  Cases "cli_abort" (RuntimeError) and "cli_abort_fpe" (a plain FloatingPointError)
instead run the production `python -m swept.run` entry with an uncaught error on rank 0 and
are expected to end through MPI Abort."""
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # parent of the swept package
import numpy as np
from swept import fermisurface as fs
from swept import lindhard as lh
from swept import model as model_module
from swept import mpiutils as mpi
from swept import run
from swept.model import Model

HR = "toy\n2\n1\n1\n" + "".join(f"0 0 0 {m} {n} {v} 0.0\n"
                                 for (m, n), v in {(1, 1): 0.0, (2, 1): 0.1, (1, 2): 0.1, (2, 2): 1.0}.items())
TOML = """[model]
hr_file = "toy_hr.dat"
spin = "{spin}"
[lattice]
rvec = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
[task.chi0]
mesh = [2, 1, 1]
kT = 0.1
window = 5.0
vertex = "cell"
"""
LAST = mpi.size - 1


def mark(**kw):
    # one write() per marker, so the launcher does not interleave it with other ranks
    sys.stdout.flush()
    os.write(1, ("\nMARK " + json.dumps(dict(rank=mpi.rank, **kw)) + "\n").encode())


def toy():
    ham = {(0, 0, 0): np.diag([0.0, 1.0]).astype(complex),
           (1, 0, 0): 0.5*np.eye(2, dtype=complex), (-1, 0, 0): 0.5*np.eye(2, dtype=complex)}
    return Model(2, 3, [1]*3, ham)


def inject(obj, name, ranks=(0,), calls=None):
    """obj.name raises LinAlgError on the given ranks (on the given 1-based calls only)."""
    real, count = getattr(obj, name), [0]

    def broken(*args, **kwargs):
        count[0] += 1
        if mpi.rank in ranks and (calls is None or count[0] in calls):
            raise np.linalg.LinAlgError(f"injected on rank {mpi.rank}")
        return real(*args, **kwargs)
    setattr(obj, name, broken)


def input_dir(spin="none"):
    d = mpi.bcast(tempfile.mkdtemp() if mpi.rank0 else None, root=0)
    os.chdir(d)
    if mpi.rank0:
        Path("toy_hr.dat").write_text(HR)
        Path("in.toml").write_text(TOML.format(spin=spin))
    mpi.barrier()


class FailingZeros:
    """numpy stand-in whose zeros() fails: a receive buffer that cannot be allocated."""
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def zeros(*args, **kwargs):
        raise MemoryError("injected receive-buffer allocation failure")


def main(case, extra):
    m = toy()
    one_k = np.zeros((1, 3))                     # rank 0 owns all work, the others none
    if case == "none":
        fs.plane_eigens(m, np.zeros((4, 3)), [[0]], chunk=1)
        m.get_bulk_eigvals(one_k)
        lh.mesh_eigens(m, (1, 1, 1), 0.0, 5.0)
    elif case == "plane":
        inject(m, "get_bulk_Hks")
        fs.plane_eigens(m, np.zeros((4, 3)), [[0]], chunk=1)
    elif case in ("mesh_e", "mesh_v"):            # mesh (1,1,1): one plane, owned by rank 0
        inject(m, "get_bulk_Hk_slab", calls=(1,) if case == "mesh_e" else (2,))
        lh.mesh_eigens(m, (1, 1, 1), 0.0, 5.0)
    elif case == "empty_eigvals":
        inject(m, "get_bulk_eigval")
        m.get_bulk_eigvals(one_k)
    elif case == "eprojs_multi":
        inject(m, "get_bulk_evec")
        m.get_bulk_eprojs_multi(one_k, [[0], [1]])
    elif case == "proj_dos":
        inject(m, "get_bulk_Hk")
        m.get_proj_dos(one_k, proj=[[0]], ne=5)
    elif case == "nonzero_rank":
        inject(m, "get_bulk_eigval", ranks=(LAST,))
        m.get_bulk_eigvals(np.zeros((mpi.size, 3)))
    elif case == "recv_alloc":
        calls, real = [0], mpi.Allgatherv

        def counting(*args, **kwargs):
            calls[0] += 1
            return real(*args, **kwargs)
        mpi.Allgatherv = counting
        if mpi.rank == LAST:
            model_module.np = FailingZeros()
        try:
            m.get_bulk_eigvals(one_k)
        finally:
            model_module.np, mpi.Allgatherv = np, real
            extra["allgatherv_calls"] = calls[0]
    elif case == "devide_prep":
        real = np.ascontiguousarray
        if mpi.rank == LAST:
            def failing(*args, **kwargs):
                raise MemoryError("injected in devide_array preparation")
            np.ascontiguousarray = failing
        try:
            m.get_bulk_eigvals(np.zeros((mpi.size, 3)))
        finally:
            np.ascontiguousarray = real
    elif case == "log_open":
        input_dir()
        if mpi.rank0:
            os.mkdir("swept_run.log")
        mpi.barrier()
        calls, real = [0], run.load_input

        def counting(path):
            calls[0] += 1
            return real(path)
        run.load_input = counting
        status = run.main(["in.toml"])
        return dict(outcome=f"status {status}", load_input_calls=calls[0])
    elif case == "nonfinite_main":              # identical non-finite reduced data on all ranks
        input_dir()
        real = mpi.Allreduce

        def nan_first(sendobj, *args, **kwargs):
            out = real(sendobj, *args, **kwargs)
            if not nan_first.done:
                out = np.full_like(out, np.nan)
                nan_first.done = True
            return out
        nan_first.done = False
        mpi.Allreduce = nan_first
        status = run.main(["in.toml"])
        mpi.Allreduce = real
        return dict(outcome=f"status {status}")
    elif case in ("cli_abort", "cli_abort_fpe"):
        input_dir(spin="doubled")
        real = Model.addspinbasis
        error = RuntimeError if case == "cli_abort" else FloatingPointError

        def failing(self, *args, **kwargs):          # outside any collective guard
            if mpi.rank == 0:
                raise error("injected uncaught error on rank 0")
            return real(self, *args, **kwargs)
        Model.addspinbasis = failing
        mark(outcome="started")
        sys.argv = ["run.py", "in.toml"]
        runpy.run_module("swept.run", run_name="__main__")   # the production entry point
        return dict(outcome="not aborted")
    else:
        raise SystemExit(f"unknown case {case!r}")
    return dict(outcome="returned", **extra)


if __name__ == "__main__":
    case = sys.argv[1]
    extra = {}                                   # filled by main also when it raises
    try:
        result = main(case, extra)
    except Exception as exc:
        result = dict(outcome=type(exc).__name__, msg=str(exc), **extra)
    mark(**result)
