"""Regression tests for swept_scientific_review_20261017 (PROPOSAL_science_2026-10-17 as
corrected by REVIEW_PROPOSAL_science_2026-10-17): chi0 band closure, Mz log, zx_to_rmat."""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # parent of the swept package
from swept import lindhard as lh
from swept import mpiutils as mpi
from swept import run, spinor, udmft
from swept.model import Model


def degenerate_pair_model():
    """Two orbitals, mesh (2,1,1): k = 0 -> diag(-0.1, 1), k = 1/2 -> 1 (degenerate)."""
    ham = {(0, 0, 0): np.diag([0.45, 1.0]).astype(complex),
           (1, 0, 0): np.diag([-0.275, 0.0]).astype(complex), (-1, 0, 0): np.diag([-0.275, 0.0]).astype(complex)}
    return Model(2, 3, [1]*3, ham)


def rotate_pair(v, theta):
    c, s = np.cos(theta), np.sin(theta)
    return v @ np.array([[c, -s], [s, c]])


class SelectBandsTests(unittest.TestCase):
    def test_chain_across_k_on_both_sides(self):
        eall = np.array([[-1.0, -1.0, 0.0, 1.0, 1.0],
                         [-2.0, -1.0, 0.5, 0.5, 3.0],
                         [-3.0, -0.5, -0.5, 2.0, 4.0]])
        bands, info = lh.select_bands(eall, 0.2)
        self.assertEqual(info["initial"], [2, 2])
        self.assertEqual(bands.tolist(), [0, 1, 2, 3, 4])                    # all bands kept
        self.assertEqual(info["added"], [("lower", 1, 2), ("lower", 0, 0), ("upper", 3, 1), ("upper", 4, 0)])
        self.assertEqual(info["boundary_gap"], [np.inf, np.inf])

    def test_tolerance_and_gaps(self):
        eall = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.5, 0.5 + 0.00005]])        # band 2 outside
        bands, info = lh.select_bands(eall, 0.2)                              # gap 5e-5 < 1e-4
        self.assertEqual(bands.tolist(), [1, 2])
        bands, info = lh.select_bands(eall, 0.2, degeneracy_tol=1e-5)
        self.assertEqual(bands.tolist(), [1])
        np.testing.assert_allclose(info["boundary_gap"], [1.0, 5e-5], rtol=1e-9)

    def test_bad_parameters_rejected_by_the_library(self):
        eall = np.array([[-1.0, 0.0, 1.0]])
        for kw in ({"degeneracy_tol": -1e-4}, {"degeneracy_tol": 0.0}, {"degeneracy_tol": np.nan},
                   {"degeneracy_tol": True}, {"degeneracy_tol": [1e-4]}, {"window": np.inf}, {"window": -0.2}):
            with self.subTest(kw=kw), self.assertRaisesRegex(ValueError, "positive finite"):
                lh.select_bands(eall, kw.get("window", 0.2), kw.get("degeneracy_tol", 1e-4))
        m = degenerate_pair_model()
        with self.assertRaisesRegex(ValueError, "degeneracy_tol"):     # same rejection on every rank
            lh.mesh_eigens(m, (2, 1, 1), 0.0, 0.2, degeneracy_tol=float("nan"))

    def test_empty_window(self):
        with self.assertRaisesRegex(ValueError, "no band within"):
            lh.select_bands(np.array([[1.0, 2.0]]), 0.5)


class Chi0GaugeTests(unittest.TestCase):
    def test_review_counterexample(self):
        m, mesh, chan = degenerate_pair_model(), (2, 1, 1), [lh.cell_channel([0, 1])]
        points = lh.mesh_points(mesh)
        full = None
        for window in (0.2, 0.4, 5.0):
            info = {}
            eigs, vecs, bands = lh.mesh_eigens(m, mesh, 0.0, window, info=info)
            self.assertEqual(bands.tolist(), [0, 1])
            vals = []
            for theta in (0.0, np.pi/4, np.pi/6):
                v = vecs.copy()
                v[1] = rotate_pair(vecs[1], theta)
                vals.append(lh.site_matrices(eigs, v, points, mesh, 1, chan, 0.01)[0][0, 0].real)
            full = vals[0] if full is None else full
            np.testing.assert_allclose(vals, full, rtol=1e-12)
        self.assertAlmostEqual(full, 0.9090496383011792, places=12)
        r = lh.lindhard(m, mesh, 0.0, 0.01, 0.2, chan, log=lambda s: None)
        self.assertEqual(r["band_selection"]["initial"], [0, 0])
        self.assertAlmostEqual(r["chi"][0][1, 0, 0].real, full, places=12)

    def test_random_unitary_inside_exact_degeneracies(self):
        rng = np.random.default_rng(7)
        nw = 3
        h0 = rng.normal(size=(nw, nw)) + 1j*rng.normal(size=(nw, nw))
        ham = {(0, 0, 0): 0.3*(h0 + h0.conj().T)}
        for R in [(1, 0, 0), (0, 1, 0), (1, 1, 0)]:
            h = (rng.normal(size=(nw, nw)) + 1j*rng.normal(size=(nw, nw)))*0.3
            ham[R], ham[tuple(-x for x in R)] = h, h.conj().T
        m = Model(nw, len(ham), [1]*len(ham), ham)
        m.addspinbasis()                                       # every band exactly doubly degenerate
        mesh = (3, 3, 1)
        chans = [lh.cell_channel(list(range(6))), lh.cell_channel([0, 1, 2])]
        eigs, vecs, bands = lh.mesh_eigens(m, mesh, 0.0, 0.6)
        self.assertEqual(len(bands) % 2, 0)
        points = lh.mesh_points(mesh)
        rot = vecs.copy()
        for ik in range(len(points)):
            for j in range(0, len(bands), 2):
                np.testing.assert_allclose(eigs[ik, j], eigs[ik, j+1], atol=1e-12)
                a = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))
                u, _ = np.linalg.qr(a)
                rot[ik, :, j:j+2] = vecs[ik, :, j:j+2] @ u
        for iq in range(len(points)):
            ref = lh.site_matrices(eigs, vecs, points, mesh, iq, chans, 0.05)
            new = lh.site_matrices(eigs, rot, points, mesh, iq, chans, 0.05)
            for a, b in zip(ref, new):
                np.testing.assert_allclose(a, b, atol=1e-12)


BASE = """
[model]
hr_file = "x_hr.dat"
spin = "none"
[lattice]
rvec = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
[task.chi0]
mesh = [2, 1, 1]
kT = 0.01
window = 0.2
window_check = 2.0
vertex = "cell"
{extra}
"""


@contextlib.contextmanager
def in_temp_dir():
    cwd = os.getcwd()
    d = mpi.bcast(tempfile.mkdtemp() if mpi.rank0 else None, root=0)
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(cwd)
        mpi.barrier()
        if mpi.rank0:
            shutil.rmtree(d)


class Chi0DriverTests(unittest.TestCase):
    def run_text(self, extra=""):
        degenerate_pair_model().to_hr("x_hr.dat")
        if mpi.rank0:
            Path("in.toml").write_text(BASE.format(extra=extra))
        mpi.barrier()
        with contextlib.redirect_stdout(io.StringIO()):
            return run.main(["in.toml"])

    def test_npz_and_log(self):
        with in_temp_dir():
            self.assertEqual(self.run_text(), 0)
            d = np.load("chi0.npz")
            self.assertEqual(d["bands"].tolist(), [1, 2])
            self.assertEqual(d["bands_before_closure"].tolist(), [1])
            self.assertEqual(float(d["degeneracy_tol"]), 1e-4)
            self.assertEqual(d["boundary_gap"].tolist(), [np.inf, np.inf])
            text = Path("swept_run.log").read_text() if mpi.rank0 else ""
            if mpi.rank0:
                self.assertIn("degeneracy closure (tol 0.0001 eV) extends bands 1-1 by 1: 2 (upper", text)
                self.assertEqual(text.count("smallest gap on the mesh"), 2)      # main + window_check

    def test_bad_tolerance(self):
        for bad in ("-1e-4", "0", "nan", "true"):
            with self.subTest(bad=bad), in_temp_dir():
                self.assertEqual(self.run_text(f"degeneracy_tol = {bad}"), 1)


class MzLogTests(unittest.TestCase):
    def test_offdiagonal_counterexample(self):
        _, _, _, mz = udmft.local_eigenbasis(0.5*np.array([[0, 1], [1, 0]], dtype=complex), ll=0)
        self.assertIn("max |off-diagonal Mz| = 1.000e+00", run.mz_offdiagonal_summary(mz, False))

    def test_kramers_pairs_diagonal_inside(self):
        rng = np.random.default_rng(3)
        cf = rng.normal(size=(5, 5))
        cf = 0.2*(cf + cf.T)
        ls = Model(nwan=10, nrpt=1, deg=[1], ham={(0, 0, 0): np.zeros((10, 10), complex)})
        ls.soc(asoc=0.1, lnum=2, sidx=0)
        h0 = np.kron(cf, np.eye(2)) + ls.ham[(0, 0, 0)]
        _, _, _, mz = udmft.local_eigenbasis(h0, ll=2, kramers=True)
        text = run.mz_offdiagonal_summary(mz, True)
        inside = float(text.split("inside Kramers pairs ")[1].split(",")[0])
        between = float(text.split("between pairs ")[1].split(")")[0])
        self.assertLess(inside, 1e-12)
        self.assertGreater(between, 1e-3)                      # not diagonal between pairs


class ZxToRmatTests(unittest.TestCase):
    def test_rejects_bad_axes(self):
        for z, x in [([1, 1, 1], [1, 0, 0]), ([0, 0, 0], [1, 0, 0]), ([0, 0, 1], [0, 0, 0]),
                     ([np.nan, 0, 1], [1, 0, 0]), ([0, 0, 1e200], [1e200, 0, 0]), ([0, 1], [1, 0])]:
            with self.subTest(z=z, x=x), self.assertRaises(ValueError):
                spinor.zx_to_rmat(z, x)

    def test_valid_axes_unchanged(self):
        z, x = np.array([1.0, 1.0, 0.0]), np.array([1.0, -1.0, 0.0])
        r = spinor.zx_to_rmat(z, x)
        zz, xx = z/np.sqrt(z @ z), x/np.sqrt(x @ x)
        np.testing.assert_array_equal(r, np.column_stack([xx, np.cross(zz, xx), zz]))
        m = Model(nwan=2, nrpt=1, deg=[1], ham={(0, 0, 0): np.eye(2, dtype=complex)})
        m.set_spin_axis([0.3, -0.4, 0.5])                      # builds its own orthogonal x


if __name__ == "__main__":
    unittest.main()
