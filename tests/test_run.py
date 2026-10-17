import os
import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # parent of the swept package
from swept.bands import dump_projs, dump_projs_multi
from swept.model import Model
from swept import mpiutils as mpi
from swept import run
BASE = """
[model]
hr_file = "x_hr.dat"
spin = "{spin}"
{extra_model}

[lattice]
rvec = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
"""


def make_cfg(spin="spinor", extra_model="", body=""):
    text = BASE.format(spin=spin, extra_model=extra_model) + textwrap.dedent(body)
    return run.validate(tomllib.loads(text))


class ValidateTests(unittest.TestCase):
    def test_unknown_key_is_rejected(self):
        with self.assertRaises(run.InputError):
            make_cfg(extra_model="eshfit = 1.0")

    def test_collinear_requires_hr_dn(self):
        with self.assertRaises(run.InputError):
            make_cfg(spin="collinear")
        make_cfg(spin="collinear", extra_model='hr_dn = "dn_hr.dat"')

    def test_soc_on_spinor_hr_needs_explicit_permission(self):
        body = """
        [shell.d]
        start = 0
        l = 2
        [soc]
        d = 0.1
        """
        with self.assertRaises(run.InputError):
            make_cfg(spin="spinor", body=body)
        make_cfg(spin="spinor", extra_model="allow_extra_soc = true", body=body)

    def test_soc_needs_spinful_model(self):
        body = """
        [shell.d]
        start = 0
        l = 2
        [soc]
        d = 0.1
        """
        with self.assertRaises(run.InputError):
            make_cfg(spin="none", body=body)

    def test_spin_frame_axes_must_be_orthogonal(self):
        body = """
        [shell.d]
        start = 0
        l = 2
        z = [0, 0, 1]
        x = [1, 0, 1]
        """
        with self.assertRaises(run.InputError):
            make_cfg(body=body)


class IndexTests(unittest.TestCase):
    co_shells = """
    [shell.Co1_d]
    start = 0
    l = 2
    [shell.Co2_d]
    start = 10
    l = 2
    [shell.Co3_d]
    start = 20
    l = 2
    """

    def test_spinor_dx2y2_dn_matches_original_plist(self):
        cfg = make_cfg(spin="spinor", body=self.co_shells)
        shells = run.resolve_shells(cfg, nwan_hr=58)
        proj = {"shells": ["Co1_d", "Co2_d", "Co3_d"], "orbitals": ["dx2-y2"], "spin": "dn"}
        np.testing.assert_array_equal(run.projection_indices(proj, shells, 58), [7, 17, 27])

    def test_doubled_start_is_mapped_to_spinful_basis(self):
        cfg = make_cfg(spin="doubled", body="""
        [shell.p]
        start = 5
        l = 1
        """)
        shells = run.resolve_shells(cfg, nwan_hr=8)
        self.assertEqual((shells["p"]["start"], shells["p"]["size"]), (10, 6))
        np.testing.assert_array_equal(
            run.projection_indices({"shells": ["p"], "orbitals": ["px"]}, shells, 16), [12, 13])

    def test_spinless_indices(self):
        cfg = make_cfg(spin="none", body="""
        [shell.d]
        start = 3
        l = 2
        """)
        shells = run.resolve_shells(cfg, nwan_hr=8)
        np.testing.assert_array_equal(
            run.projection_indices({"shells": ["d"], "orbitals": ["dxy", "dz2"]}, shells, 8), [3, 7])

    def test_shell_out_of_range_or_overlapping(self):
        cfg = make_cfg(spin="doubled", body="""
        [shell.d]
        start = 4
        l = 2
        """)
        with self.assertRaises(run.InputError):
            run.resolve_shells(cfg, nwan_hr=8)
        cfg = make_cfg(spin="spinor", body="""
        [shell.a]
        start = 0
        l = 1
        [shell.b]
        start = 4
        l = 1
        """)
        with self.assertRaises(run.InputError):
            run.resolve_shells(cfg, nwan_hr=20)


class ModelExtensionTests(unittest.TestCase):
    def test_set_spin_axis_aligns_majority_spin(self):
        # one orbital, collinear: up (along m) lower by 2
        sigma = {
            "x": np.array([[0, 1], [1, 0]], dtype=complex),
            "y": np.array([[0, -1j], [1j, 0]]),
            "z": np.diag([1.0, -1.0]).astype(complex),
        }
        for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, -1.0]):
            model = Model(nwan=2, nrpt=1, deg=[1], ham={(0, 0, 0): np.diag([-1.0, 1.0]).astype(complex)})
            model.set_spin_axis(axis)
            _, vec = np.linalg.eigh(model.ham[(0, 0, 0)])
            ground = vec[:, 0]
            spin = np.array([np.real(ground.conj() @ sigma[c] @ ground) for c in "xyz"])
            np.testing.assert_allclose(spin, np.array(axis) / np.linalg.norm(axis), atol=1e-12)

    def test_eprojs_multi_matches_single(self):
        rng = np.random.default_rng(5)
        hop = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
        h0 = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
        model = Model(nwan=6, nrpt=3, deg=[1, 1, 1],
                      ham={(0, 0, 0): h0 + h0.conj().T, (1, 0, 0): hop, (-1, 0, 0): hop.conj().T})
        klist = np.array([[0.0, 0.0, 0.0], [0.13, 0.0, 0.0], [0.4, 0.0, 0.0]])
        plists = [np.array([0, 3]), np.array([1, 2, 5])]

        eigs, projs = model.get_bulk_eprojs_multi(klist, plists)
        for ipr, plist in enumerate(plists):
            eref, pref = model.get_bulk_eprojs(klist, plist)
            np.testing.assert_allclose(eigs, eref, atol=1e-12)
            np.testing.assert_allclose(projs[ipr], pref, atol=1e-12)

    def test_dump_projs_multi_single_column_matches_dump_projs(self):
        kpth = np.array([0.0, 0.5])
        eigs = np.array([[1.0, 2.0], [3.0, 4.0]])
        proj = np.array([[0.1, 0.2], [0.3, 0.4]])
        with tempfile.TemporaryDirectory() as directory:
            dump_projs(kpth, eigs, proj, fname=Path(directory) / "a.dat", mu=0.5)
            dump_projs_multi(kpth, eigs, proj[None], ["w"], fname=Path(directory) / "b.dat", mu=0.5)
            a = (Path(directory) / "a.dat").read_text()
            b = (Path(directory) / "b.dat").read_text()
        self.assertEqual(b, "# k  E-mu  w\n" + a)


class EndToEndTests(unittest.TestCase):
    def test_doubled_d_shell_with_soc(self):
        # isolated d shell, zero crystal field: zeta L.S gives
        # j=5/2 at +zeta (6 states), j=3/2 at -3/2 zeta (4 states)
        zeta = 0.1
        toml = textwrap.dedent(f"""
        [model]
        hr_file = "d_hr.dat"
        spin = "doubled"

        [lattice]
        rvec = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]

        [shell.d]
        start = 0
        l = 2

        [soc]
        d = {zeta}

        [projection]
        d_up = {{ shells = ["d"], spin = "up" }}

        [kpath]
        points = [[0, 0, 0], [0.5, 0, 0]]
        labels = ["G", "X"]
        nk = 3

        [task.bands]
        output = "bands.dat"
        project = ["d_up"]
        """)
        # all ranks share rank 0's directory; only rank 0 writes and checks files
        cwd = os.getcwd()
        directory = mpi.bcast(tempfile.mkdtemp() if mpi.rank0 else None, root=0)
        os.chdir(directory)
        try:
            Model(nwan=5, nrpt=1, deg=[1], ham={(0, 0, 0): np.zeros((5, 5), dtype=complex)}).to_hr("d_hr.dat")
            if mpi.rank0:
                Path("in.toml").write_text(toml)
            self.assertEqual(run.main(["in.toml"]), 0)
            if mpi.rank0:
                data = np.loadtxt("bands.dat")
                ticks = Path("bands_ticks.dat").read_text().split()
        finally:
            os.chdir(cwd)
            mpi.barrier()           # every rank out of the directory before rmtree
            if mpi.rank0:
                shutil.rmtree(directory)
        if not mpi.rank0:
            return

        energies = np.sort(np.unique(np.round(data[:, 1], 9)))
        np.testing.assert_allclose(energies, [-1.5 * zeta, zeta], atol=1e-6)
        self.assertEqual(np.sum(np.isclose(data[:, 1], zeta, atol=1e-6)), 6 * len(data) // 10)
        # spin up carries half of the weight summed over all bands at each k
        np.testing.assert_allclose(data[:, 2].reshape(10, -1).sum(axis=0), 5.0, atol=1e-6)
        self.assertEqual(ticks[0], "G")


if __name__ == "__main__":
    unittest.main()
