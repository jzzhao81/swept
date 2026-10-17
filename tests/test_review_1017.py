"""Regression tests for REVIEW_2026-10-17 (PROPOSAL_v2_2026-10-17), serial part.

The multi-rank error paths are in test_mpi_faults.py (separate mpiexec jobs)."""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # parent of the swept package
from swept import mpiutils as mpi
from swept import run
from swept.model import Model, hermiticity_residual

BASE = """
[model]
hr_file = "x_hr.dat"
spin = "none"
{extra}
[lattice]
rvec = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

[task.chi0]
mesh = [2, 1, 1]
kT = 0.1
window = 10.0
vertex = "cell"
"""


def chain():
    ham = {(0, 0, 0): np.diag([0.0, 0.3]).astype(complex),
           (1, 0, 0): np.diag([1.0, 0.0]).astype(complex), (-1, 0, 0): np.diag([1.0, 0.0]).astype(complex)}
    return Model(nwan=2, nrpt=3, deg=[1]*3, ham=ham)


def hr_text(value, R=(1, 0, 0), mn=(1, 2), imag=False):
    """2-orbital hr with R = 0 and R; element (mn) of block R replaced by value."""
    lines = ["test", "2", "2", "1 1"]
    for r in [(0, 0, 0), R]:
        for n in (1, 2):
            for m in (1, 2):
                re, im = "0.1", "0.0"
                if r == R and (m, n) == mn:
                    re, im = ("0.1", value) if imag else (value, "0.0")
                lines.append(f"{r[0]} {r[1]} {r[2]} {m} {n} {re} {im}")
    return "\n".join(lines) + "\n"


@contextlib.contextmanager
def temp_dir():
    cwd, d = os.getcwd(), tempfile.mkdtemp()
    os.chdir(d)
    try:
        yield Path(d)
    finally:
        os.chdir(cwd)
        shutil.rmtree(d)


@unittest.skipIf(mpi.size > 1, "serial semantics; the MPI paths are in test_mpi_faults.py")
class NonFiniteHrTests(unittest.TestCase):            # item 1
    def test_read_rejects_nan_inf_with_location(self):
        for value in ("nan", "inf", "-inf"):
            for imag in (False, True):
                with self.subTest(value=value, imag=imag), temp_dir():
                    Path("x_hr.dat").write_text(hr_text(value, imag=imag))
                    with self.assertRaisesRegex(ValueError, r"non-finite matrix element at "
                                                r"R = \[1, 0, 0\], \(m, n\) = \(1, 2\)"):
                        Model.from_hr("x_hr.dat")

    def test_finite_file_still_reads(self):
        with temp_dir():
            Path("x_hr.dat").write_text(hr_text("0.2"))
            self.assertEqual(Model.from_hr("x_hr.dat").nwan, 2)

    def test_cache_is_published_only_when_finite(self):
        model = chain()
        model.get_bulk_Hk([0.1, 0, 0])
        model.ham[(1, 0, 0)][0, 1] = np.nan
        model.reset_cache()
        for _ in range(2):                    # the failed build must not leave a cache behind
            with self.assertRaisesRegex(ValueError, r"R = \(1, 0, 0\)"):
                model.get_bulk_Hk([0.1, 0, 0])
            self.assertIsNone(model._hstack)
        model.ham[(1, 0, 0)][0, 1] = 0.0
        model.reset_cache()
        np.testing.assert_allclose(np.linalg.eigvalsh(model.get_bulk_Hk([0.25, 0, 0])), [0.0, 0.3], atol=1e-12)


@unittest.skipIf(mpi.size > 1, "serial semantics")
class HermiticityTests(unittest.TestCase):            # item 1c
    def test_values_by_hand(self):
        a = np.array([[0.0, 0.2], [0.1, 0.0]], dtype=complex)
        ham = {(0, 0, 0): np.array([[0.0, 1e-3], [0.0, 0.0]], dtype=complex),
               (1, 0, 0): a, (-1, 0, 0): a.conj().T + 0.05, (0, 2, 0): np.full((2, 2), 0.01)}
        mx, sb = hermiticity_residual(ham)
        # R = 0: 1e-3; R = +-1: 0.05 each; (0, +-2, 0) (partner missing): 0.01 each
        self.assertAlmostEqual(mx, 0.05, places=15)
        self.assertAlmostEqual(sb, 1e-3 + 2*0.05 + 2*0.01, places=15)

    def test_sum_bound_holds_where_max_does_not(self):
        t = 1e-6
        ham = {(n, 0, 0): np.array([[t]], dtype=complex) for n in range(1, 101)}   # one-way hoppings
        ham[(0, 0, 0)] = np.zeros((1, 1), complex)
        model = Model(nwan=1, nrpt=len(ham), deg=[1]*len(ham), ham=ham)
        mx, sb = hermiticity_residual(ham)
        self.assertAlmostEqual(sb, 200*t, places=15)
        hk = np.array([model.get_bulk_Hk([k, 0, 0])[0, 0] for k in np.linspace(0, 1, 4001)])
        worst = np.abs(hk - hk.conj()).max()
        self.assertLessEqual(worst, sb*(1 + 1e-12))
        self.assertGreater(worst, 10*mx)

    def test_from_hr_attribute_log_and_warning(self):
        with temp_dir():
            Path("x_hr.dat").write_text(hr_text("0.3"))          # H(R)_12 = 0.3, H(-R) missing
            m = Model.from_hr("x_hr.dat")
            self.assertEqual(m.hermiticity["at"], "load")
            self.assertAlmostEqual(m.hermiticity["max"], 0.3)
            for extra, warns in [("", True), ("hermiticity_warn = 0.5", False)]:
                Path("in.toml").write_text(BASE.format(extra=extra))
                with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(io.StringIO()):
                    warnings.simplefilter("always")
                    self.assertEqual(run.main(["in.toml"]), 0)
                hits = [x for x in w if "hermiticity_warn" in str(x.message)]
                self.assertEqual(bool(hits), warns, extra)
                self.assertIn("hermiticity as read: max_R max_mn |H(R) - H(-R)^+| = 3.0e-01 eV",
                              Path("swept_run.log").read_text())

    def test_bad_threshold_is_input_error(self):
        for bad in ("-1.0", "0", "true", '"x"', "nan"):
            with self.subTest(bad=bad), temp_dir():
                chain().to_hr("x_hr.dat")
                Path("in.toml").write_text(BASE.format(extra=f"hermiticity_warn = {bad}"))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run.main(["in.toml"]), 1)
                self.assertIn("hermiticity_warn", Path("swept_run.log").read_text())


@unittest.skipIf(mpi.size > 1, "serial semantics")
class SerialCollectiveTests(unittest.TestCase):       # item 2, single rank
    def test_exception_passes_unchanged(self):
        for exc in (ValueError("v"), np.linalg.LinAlgError("l"), MemoryError("m")):
            with self.subTest(exc=type(exc).__name__):
                with self.assertRaises(type(exc)) as cm:
                    with mpi.collective("test"):
                        raise exc
                self.assertIs(cm.exception, exc)

    def test_methods_keep_original_type(self):
        model = chain()
        with mock.patch.object(model, "get_bulk_eigval", side_effect=np.linalg.LinAlgError("x")):
            with self.assertRaises(np.linalg.LinAlgError):
                model.get_bulk_eigvals(np.zeros((2, 3)))

    def test_main_returns_1_only_for_synchronised_errors(self):
        # NonFiniteResult (check on reduced data) is reported; a plain FloatingPointError
        # may come from one rank only, so main must not swallow it
        for exc, returns in [(mpi.NonFiniteResult("reduced"), True), (FloatingPointError("local"), False)]:
            with self.subTest(exc=type(exc).__name__), temp_dir():
                chain().to_hr("x_hr.dat")
                Path("in.toml").write_text(BASE.format(extra=""))
                with mock.patch("swept.lindhard.mesh_eigens", side_effect=exc), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if returns:
                        self.assertEqual(run.main(["in.toml"]), 1)
                    else:
                        with self.assertRaises(FloatingPointError):
                            run.main(["in.toml"])

    def test_empty_k_list(self):
        self.assertEqual(chain().get_bulk_eigvals(np.zeros((0, 3))).shape, (2, 0))


class FailingFile:
    """File object whose write, flush or close raises ENOSPC."""
    def __init__(self, fail):
        self.fail, self.text, self.closed = fail, [], False

    def write(self, text):
        if self.fail == "write" and self.text:
            raise OSError(28, "No space left on device")
        self.text.append(text)

    def flush(self):
        if self.fail == "flush" and len(self.text) > 1:
            raise OSError(28, "No space left on device")

    def close(self):
        self.closed = True
        if self.fail in ("flush", "close"):
            raise OSError(28, "No space left on device")


@unittest.skipIf(mpi.size > 1, "serial semantics; the MPI log case is in test_mpi_faults.py")
class RunLogTests(unittest.TestCase):                 # item 3
    def test_log_open_failure_stops_before_input(self):
        with temp_dir():
            chain().to_hr("x_hr.dat")
            Path("in.toml").write_text(BASE.format(extra=""))
            os.mkdir("swept_run.log")
            err = io.StringIO()
            with mock.patch.object(run, "load_input") as load, contextlib.redirect_stderr(err):
                self.assertEqual(run.main(["in.toml"]), 1)
            load.assert_not_called()
            self.assertIn("cannot open swept_run.log", err.getvalue())

    def test_write_flush_close_failures_degrade_to_stdout(self):
        for fail in ("write", "flush", "close"):
            with self.subTest(fail=fail), temp_dir():
                chain().to_hr("x_hr.dat")
                Path("in.toml").write_text(BASE.format(extra=""))
                fake = FailingFile(fail)

                def fake_open(self):
                    self.file = fake

                out, err = io.StringIO(), io.StringIO()
                with mock.patch.object(run.RunLog, "open", fake_open), \
                        contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    self.assertEqual(run.main(["in.toml"]), 0)
                self.assertTrue(fake.closed)
                self.assertIn("done", out.getvalue())
                text = err.getvalue()
                self.assertEqual(text.count("swept: warning"), 1, text)     # once, also when close fails again
                self.assertIn("may be incomplete" if fail == "close" else "the log continues on stdout only", text)
                self.assertTrue(Path("chi0.npz").exists())

    def test_input_copy_goes_to_the_log(self):
        with temp_dir():
            chain().to_hr("x_hr.dat")
            Path("in.toml").write_text(BASE.format(extra=""))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(["in.toml"]), 0)
            text = Path("swept_run.log").read_text()
            self.assertIn("----- input -----\n" + BASE.format(extra=""), text)


if __name__ == "__main__":
    unittest.main()
