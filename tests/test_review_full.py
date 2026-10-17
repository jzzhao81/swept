"""Regression tests for REVIEW_FULL_2026-10-16 (items 1-7 and maintenance notes)."""
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # parent of the swept package
from swept import lindhard as lh
from swept import mpiutils as mpi
from swept import run
from swept.model import Model

PKG = Path(__file__).resolve().parents[1]

BASE = """
[model]
hr_file = "x_hr.dat"
spin = "none"

[lattice]
rvec = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

[shell]
a = { start = 0, l = 0, pos = [0.0, 0.0, 0.0] }
b = { start = 1, l = 0, pos = [0.5, 0.5, 0.0] }
"""


def chain():
    """H(k) = 2 cos(2 pi kx) on orbital 0, onsite 0.3 on orbital 1."""
    ham = {(0, 0, 0): np.diag([0.0, 0.3]).astype(complex),
           (1, 0, 0): np.diag([1.0, 0.0]).astype(complex), (-1, 0, 0): np.diag([1.0, 0.0]).astype(complex)}
    return Model(nwan=2, nrpt=3, deg=[1]*3, ham=ham)


class InTempDir:
    """All ranks share rank 0's temporary directory."""
    def __enter__(self):
        self.cwd = os.getcwd()
        self.dir = mpi.bcast(tempfile.mkdtemp() if mpi.rank0 else None, root=0)
        os.chdir(self.dir)
        return Path(self.dir)

    def __exit__(self, *exc):
        os.chdir(self.cwd)
        mpi.barrier()
        if mpi.rank0:
            shutil.rmtree(self.dir)


class MpiDtypeTests(unittest.TestCase):          # review item 1
    def test_float32_int_and_strided_k_points(self):
        model = chain()
        k64 = np.array([[0.25, 0, 0], [0.1, 0, 0], [0.5, 0, 0]])
        ref = model.get_bulk_eigvals(k64)
        for k in (k64.astype(np.float32), np.asfortranarray(k64), np.hstack([k64, k64])[:, ::2][:, :3]):
            np.testing.assert_allclose(model.get_bulk_eigvals(k), ref, atol=1e-7)
        kint = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.int64)
        np.testing.assert_allclose(model.get_bulk_eigvals(kint), model.get_bulk_eigvals(kint.astype(float)))
        recv, count, _ = mpi.devide_array(np.arange(6, dtype=np.float32), 0)
        self.assertEqual(recv.dtype, np.float64)
        self.assertEqual(sum(count), 6)


class CollectiveErrorTests(unittest.TestCase):   # review item 2
    def run_text(self, text, model=None):
        with InTempDir():
            (model or chain()).to_hr("x_hr.dat")
            if mpi.rank0:
                Path("in.toml").write_text(text)
            mpi.barrier()
            return run.main(["in.toml"])

    def test_wrong_table_types_are_input_errors(self):
        for bad in ["projection = []\n", "task = 1\n", "shell = 'a'\n"]:
            self.assertEqual(self.run_text(bad + BASE.replace("\n[shell]", "\n[shellx]").replace(
                "[shellx]\na = { start = 0, l = 0, pos = [0.0, 0.0, 0.0] }\n"
                "b = { start = 1, l = 0, pos = [0.5, 0.5, 0.0] }\n", "")), 1)

    def test_unexpected_validation_failure_reaches_every_rank(self):
        with InTempDir():
            if mpi.rank0:
                Path("in.toml").write_text(BASE)
            mpi.barrier()
            with mock.patch.object(run, "validate", side_effect=KeyError("boom")):
                with self.assertRaises(RuntimeError):      # on every rank, nobody left in bcast
                    run.load_input("in.toml")

    def test_output_failure_reaches_every_rank(self):
        text = BASE + textwrap.dedent("""
        [task.chi0]
        mesh = [2, 1, 1]
        kT = 0.1
        window = 10.0
        output = "no_such_dir/chi0.npz"
        """)
        self.assertEqual(self.run_text(text), 1)


class DuplicateOrbitalTests(unittest.TestCase):  # review item 3
    def test_projection_sets_must_not_repeat(self):
        for proj in ["P = { index = [0, 0] }", "P = { index = [0.0] }", "P = { index = [] }",
                     'P = { shells = ["a", "a"] }', 'P = { shells = ["a"], orbitals = ["s", "s"] }']:
            with self.assertRaises(run.InputError, msg=proj):
                run.validate(tomllib.loads(BASE + "[projection]\n" + proj + "\n"))
        cfg = run.validate(tomllib.loads(BASE + '[projection]\nP = { shells = ["a", "b"] }\n'))
        shells = run.resolve_shells(cfg, 2)
        np.testing.assert_array_equal(run.projection_indices(cfg["projection"]["P"], shells, 2), [0, 1])
        with self.assertRaises(run.InputError):                      # bypassing validate
            run.projection_indices({"index": [1, 1]}, shells, 2)

    def test_channel_rejects_repeated_orbitals(self):
        with self.assertRaises(ValueError):
            lh.make_channel([0, 0], [0, 0], np.zeros((2, 3)), [[0, 0, 0]])
        with self.assertRaises(ValueError):
            lh.cell_channel([1, 1])


class NumericValidationTests(unittest.TestCase):  # review item 4
    def test_bad_numbers_are_rejected(self):
        dos = "[task.dos]\nmesh = [1, 1, 1]\nemin = -1.0\nemax = 1.0\nne = 3\nsigma = 0.1\n"
        run.validate(tomllib.loads(BASE + dos))
        for old, new in [("sigma = 0.1", "sigma = -0.1"), ("sigma = 0.1", "sigma = nan"),
                         ("sigma = 0.1", "sigma = inf"), ("ne = 3", "ne = 1"), ("ne = 3", "ne = 3.0"),
                         ("emax = 1.0", "emax = -2.0"), ("mesh = [1, 1, 1]", "mesh = [1.5, 1, 1]"),
                         ("mesh = [1, 1, 1]", "mesh = [true, 1, 1]"), ("emin = -1.0", "emin = nan")]:
            with self.assertRaises(run.InputError, msg=new):
                run.validate(tomllib.loads(BASE + dos.replace(old, new)))
        chi0 = "[task.chi0]\nmesh = [1, 1, 1]\nkT = 0.1\n"
        for extra in ["kT = nan\n", "mu = inf\n", "window = -1.0\n"]:
            text = BASE + (chi0.replace("kT = 0.1\n", extra) if extra.startswith("kT") else chi0 + extra)
            with self.assertRaises(run.InputError, msg=extra):
                run.validate(tomllib.loads(text))
        for bad in [BASE.replace("start = 1,", "start = 1.0,"), BASE.replace("l = 0, pos = [0.5", "l = true, pos = [0.5"),
                    BASE.replace("[0.5, 0.5, 0.0]", "[0.5, nan, 0.0]"),
                    BASE.replace("[0.0, 0.0, 1.0]]", "[0.0, 0.0, inf]]")]:
            with self.assertRaises(run.InputError):
                run.validate(tomllib.loads(bad))


class PackageTests(unittest.TestCase):           # review item 5
    def child(self, code):
        # mpi4py disabled in the child: it must not join this job's MPI process manager
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        with tempfile.TemporaryDirectory() as tmp:
            return subprocess.run([sys.executable, "-c", "import sys; sys.modules['mpi4py'] = None\n" + code],
                                  cwd=tmp, env=env, capture_output=True, text=True, timeout=120)

    def test_import_as_package_from_a_clean_directory(self):
        if not mpi.rank0:
            return
        out = self.child(f"sys.path.insert(0, {str(PKG.parent)!r})\nimport swept\n"
                         "from swept import run, lindhard, fermisurface, symmetry, udmft, bands, dos\n"
                         "print(swept.Model.__module__, swept.Kpoint.__module__)")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("swept.model swept.kpoint", out.stdout)

    def test_run_py_as_a_script(self):
        if not mpi.rank0:
            return
        out = self.child(f"import runpy\nsys.argv = ['run.py']\n"
                         f"runpy.run_path({str(PKG / 'run.py')!r}, run_name='__main__')")
        self.assertIn("usage:", out.stdout, out.stderr)


HR_HEADER = "test\n2\n1\n1\n"


class HrReaderTests(unittest.TestCase):          # review item 6
    def read(self, body):
        with InTempDir():
            if mpi.rank0:
                Path("t_hr.dat").write_text(HR_HEADER + body)
            mpi.barrier()
            return Model.from_hr("t_hr.dat")

    def test_lines_placed_by_their_labels(self):
        # same matrix [[1, 2+1j], [2-1j, 3]], lines in a non-standard order
        body = ("0 0 0 2 2 3 0\n0 0 0 1 2 2 1\n0 0 0 1 1 1 0\n0 0 0 2 1 2 -1\n")
        np.testing.assert_allclose(self.read(body).ham[(0, 0, 0)], [[1, 2+1j], [2-1j, 3]])

    def test_corrupt_files_are_rejected(self):
        good = ["0 0 0 1 1 1 0", "0 0 0 2 1 0 0", "0 0 0 1 2 0 0", "0 0 0 2 2 2 0"]
        bad = {"repeated (m,n)": ["0 0 0 1 1 1 0", "0 0 0 1 1 0 0", "0 0 0 1 1 0 0", "0 0 0 1 1 2 0"],
               "label out of range": good[:3] + ["0 0 0 3 2 2 0"],
               "R changes in block": good[:3] + ["1 0 0 2 2 2 0"],
               "non-integer label": good[:3] + ["0 0 0 2 2.5 2 0"],
               "bad token": good[:3] + ["0 0 0 2 2 x 0"],
               "too few numbers": good[:3] + ["0 0 0 2 2 2"]}
        self.read("\n".join(good) + "\n")
        for name, lines in bad.items():
            with self.assertRaises(ValueError, msg=name):
                self.read("\n".join(lines) + "\n")

    def test_repeated_r_block(self):
        lines = [f"{r} 0 0 {m} {n} 0.1 0" for r in (0, 0) for n in (1, 2) for m in (1, 2)]
        with InTempDir():
            if mpi.rank0:
                Path("t_hr.dat").write_text("test\n2\n2\n1 1\n" + "\n".join(lines) + "\n")
            mpi.barrier()
            with self.assertRaises(ValueError):
                Model.from_hr("t_hr.dat")


class PlotAndMaintenanceTests(unittest.TestCase):  # review item 7 + maintenance notes
    def test_plotbands_without_ticks_and_dos_ef(self):
        from swept.bands import plotbands
        from swept.dos import plot_total_dos
        import matplotlib
        matplotlib.use("Agg")
        with tempfile.TemporaryDirectory() as tmp:
            plotbands([0, 1], [[0, 1]], file=str(Path(tmp)/"b.pdf"))
            ax = plot_total_dos([0, 1], [1, 2], file=str(Path(tmp)/"d.pdf"), ef=0.5)
            self.assertEqual(list(ax.lines[-1].get_xdata()), [0.5, 0.5])

    def test_surfiso_not_implemented_and_cache_reset(self):
        model = chain()
        with self.assertRaises(NotImplementedError):
            model.get_slab_surfiso(np.zeros((1, 3)), 2)
        h0 = model.get_bulk_Hk([0, 0, 0]).copy()
        model.ham[(0, 0, 0)][1, 1] = 5.0          # in-place change: cache is stale until reset
        model.reset_cache()
        self.assertAlmostEqual((model.get_bulk_Hk([0, 0, 0]) - h0)[1, 1].real, 4.7)


if __name__ == "__main__":
    unittest.main()
