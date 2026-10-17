import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # parent of the swept package
from swept import mpiutils as mpi
from swept.model import Model, read_wsvec

# Chain along a1 (cubic, a = 1): A at 0, B at 1/2, k mesh 4 x 1 x 1.  The
# wannier90 R set is {-2(deg 2), -1, 0, 1, 2(deg 2)}; the true hoppings below
# reach |R| = 2, so the 4-cell Fourier transform folds them.
NK = 4
TAU = [0.0, 0.5]
RSET = [-2, -1, 0, 1, 2]
DEG = [2, 1, 1, 1, 2]
TRUE = {                                    # (m, n, R) -> <m,0|H|n,R>
    (0, 0, 0): 0.3, (1, 1, 0): -0.2,
    (0, 1, 0): -1.0, (0, 1, -1): -1.0,      # A-B at d = +-0.5
    (0, 1, 1): 0.15, (0, 1, -2): 0.15,      # A-B at d = +-1.5: R = -2 sits on the WS boundary
    (0, 0, 1): 0.25, (0, 0, -1): 0.25, (0, 0, 2): 0.05, (0, 0, -2): 0.05,   # d = +-2: a tie
    (1, 1, 1): -0.1, (1, 1, -1): -0.1,
}
for (m, n, r), v in list(TRUE.items()):
    TRUE[(n, m, -r)] = v


def true_hk(k):
    hk = np.zeros((2, 2), dtype=complex)
    for (m, n, r), v in TRUE.items():
        hk[m, n] += v * np.exp(2j*np.pi*r*k[0])
    return hk


def write_inputs(directory, ws=True):
    """hr = folded hoppings on the R set (as wannier90 writes them),
    wsvec = minimal-distance images of every (R, m, n)."""
    ham = {}
    for r, deg in zip(RSET, DEG):
        hmat = np.zeros((2, 2), dtype=complex)
        for (m, n, rt), v in TRUE.items():
            if (rt - r) % NK == 0:
                hmat[m, n] += v
        ham[(r, 0, 0)] = hmat / deg
    Model(nwan=2, nrpt=len(RSET), deg=list(DEG), ham=ham).to_hr(str(Path(directory) / "x_hr.dat"))

    lines = ["## written on 15Oct2026 at 12:00:00 with use_ws_distance=" + (".true." if ws else ".false.")]
    for r in RSET:
        for m in range(2):
            for n in range(2):
                lines.append("".join(f"{x:5d}" for x in (r, 0, 0, m + 1, n + 1)))
                if ws:
                    dist = {t: abs(r + t + TAU[n] - TAU[m]) for t in (-NK, 0, NK)}
                    best = [t for t in sorted(dist) if dist[t] < min(dist.values()) + 1e-8]
                else:
                    best = [0]
                lines.append(f"{len(best):5d}")
                lines.extend(f"{t:5d}{0:5d}{0:5d}" for t in best)
    (Path(directory) / "x_wsvec.dat").write_text("\n".join(lines) + "\n")


class WsvecTests(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.dir = mpi.bcast(tempfile.mkdtemp() if mpi.rank0 else None, root=0)

    def tearDown(self):
        mpi.barrier()           # every rank out of the directory before rmtree
        if mpi.rank0:
            shutil.rmtree(self.dir)

    def path(self, name):
        return str(Path(self.dir) / name)

    def write(self, ws=True):
        if mpi.rank0:
            write_inputs(self.dir, ws=ws)
        mpi.bcast(None, root=0)

    def test_wsvec_recovers_true_hoppings(self):
        self.write()
        plain = Model.from_hr(self.path("x_hr.dat"))
        model = Model.from_hr(self.path("x_hr.dat"), wsvec_file=self.path("x_wsvec.dat"))
        ks = np.random.default_rng(0).random((6, 3))
        # without wsvec half of the d = -1.5 A-B hopping sits at d = +2.5
        self.assertGreater(max(np.abs(plain.get_bulk_Hk(k) - true_hk(k)).max() for k in ks), 0.05)
        for k in ks:
            np.testing.assert_allclose(model.get_bulk_Hk(k), true_hk(k), atol=1e-6)
        self.assertEqual(model.deg, [1]*model.nrpt)
        np.testing.assert_allclose(model.ham[(-2, 0, 0)][0, 1], 0.15, atol=1e-6)
        self.assertNotIn((2, 0, 0), [r for r, h in model.ham.items() if abs(h[0, 1]) > 1e-9])
        # ties are split: A-A at d = +-2 keeps 0.05 on both sides
        np.testing.assert_allclose([model.ham[(2, 0, 0)][0, 0], model.ham[(-2, 0, 0)][0, 0]], 0.05, atol=1e-6)

    def test_round_trip_through_to_hr(self):
        self.write()
        model = Model.from_hr(self.path("x_hr.dat"), wsvec_file=self.path("x_wsvec.dat"))
        model.to_hr(self.path("y_hr.dat"))
        again = Model.from_hr(self.path("y_hr.dat"))
        k = np.array([0.37, 0.0, 0.0])
        np.testing.assert_allclose(again.get_bulk_Hk(k), model.get_bulk_Hk(k), atol=1e-6)

    def test_use_ws_distance_false_file_changes_nothing(self):
        self.write(ws=False)
        plain = Model.from_hr(self.path("x_hr.dat"))
        model = Model.from_hr(self.path("x_hr.dat"), wsvec_file=self.path("x_wsvec.dat"))
        k = np.array([0.21, 0.0, 0.0])
        np.testing.assert_allclose(model.get_bulk_Hk(k), plain.get_bulk_Hk(k), atol=1e-12)

    def test_read_wsvec_layout(self):
        self.write()
        if not mpi.rank0:
            return
        rvec, m, n, ndeg, shifts = read_wsvec(self.path("x_wsvec.dat"))
        self.assertEqual(len(rvec), len(RSET)*4)
        self.assertEqual(len(shifts), ndeg.sum())
        # R = 2, A -> A: tie between d = +2 and d = -2
        ie = np.where((rvec[:, 0] == 2) & (m == 0) & (n == 0))[0][0]
        self.assertEqual(ndeg[ie], 2)
        offset = np.cumsum(ndeg)[ie] - ndeg[ie]
        self.assertEqual(sorted(shifts[offset:offset + 2, 0].tolist()), [-4, 0])

    def test_mismatched_wsvec_raises_on_all_ranks(self):
        self.write()
        if mpi.rank0:
            text = Path(self.path("x_wsvec.dat")).read_text().splitlines()
            Path(self.path("bad_wsvec.dat")).write_text("\n".join(text[:-3]) + "\n")
        mpi.bcast(None, root=0)
        with self.assertRaises(ValueError):
            Model.from_hr(self.path("x_hr.dat"), wsvec_file=self.path("bad_wsvec.dat"))


    def test_missing_hr_raises_on_all_ranks(self):
        with self.assertRaises(ValueError):
            Model.from_hr(self.path("missing_hr.dat"))

    def test_truncated_hr_raises_on_all_ranks(self):
        self.write()
        if mpi.rank0:
            text = Path(self.path("x_hr.dat")).read_text().splitlines()
            Path(self.path("cut_hr.dat")).write_text("\n".join(text[:-5]) + "\n")
            Path(self.path("head_hr.dat")).write_text("\n".join(text[:2]) + "\n")
        mpi.bcast(None, root=0)
        for name in ("cut_hr.dat", "head_hr.dat"):
            with self.assertRaises(ValueError):
                Model.from_hr(self.path(name))

    def test_zero_degeneracy_raises_on_all_ranks(self):
        # used to be a ZeroDivisionError on rank 0 only -> the others hung in bcast
        if mpi.rank0:
            Path(self.path("z_hr.dat")).write_text("test\n1\n1\n0\n0 0 0 1 1 1.0 0.0\n")
        mpi.bcast(None, root=0)
        with self.assertRaisesRegex(ValueError, "degeneracies must be >= 1"):
            Model.from_hr(self.path("z_hr.dat"))

    def test_nonpositive_wsvec_images_raise(self):
        # N_mnR = -2 is a zero cursor step: an endless loop before the fix
        if mpi.rank0:
            Path(self.path("one_hr.dat")).write_text("test\n1\n1\n1\n0 0 0 1 1 1.0 0.0\n")
            for nimg in (-2, -1, 0):
                Path(self.path(f"w{nimg}_wsvec.dat")).write_text(f"## written on test\n0 0 0 1 1\n{nimg}\n")
        mpi.bcast(None, root=0)
        for nimg in (-2, -1, 0):
            with self.assertRaisesRegex(ValueError, "N_mnR"):
                Model.from_hr(self.path("one_hr.dat"), wsvec_file=self.path(f"w{nimg}_wsvec.dat"))

    def test_unexpected_error_raises_on_all_ranks(self):
        # an exception type the reader does not anticipate is still broadcast,
        # as RuntimeError (a bug, not an input error)
        from unittest import mock
        with mock.patch.object(Model, "_read_hr", side_effect=TypeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "TypeError: boom"):
                Model.from_hr(self.path("any_hr.dat"))


class DriverKeyTests(unittest.TestCase):
    def cfg(self, spin, extra):
        import tomllib
        from swept import run
        text = f"""
        [model]
        hr_file = "x_hr.dat"
        spin = "{spin}"
        {extra}
        [lattice]
        rvec = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        """
        return run.validate(tomllib.loads("\n".join(line.strip() for line in text.splitlines())))

    def test_wsvec_keys(self):
        from swept import run
        self.cfg("spinor", 'wsvec = "x_wsvec.dat"')
        self.cfg("collinear", 'hr_dn = "d_hr.dat"\nwsvec = "u.dat"\nwsvec_dn = "d.dat"')
        with self.assertRaises(run.InputError):
            self.cfg("collinear", 'hr_dn = "d_hr.dat"\nwsvec = "u.dat"')
        with self.assertRaises(run.InputError):
            self.cfg("spinor", 'wsvec_dn = "d.dat"')


if __name__ == "__main__":
    unittest.main()
