import os
import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # parent of the swept package
from swept import mpiutils as mpi
from swept import run
from swept import symmetry as sym
from swept.model import Model
from swept.spinor import rotate_spinor


# Slater-Koster s/p model: symmetric under every operation of the structure
# by construction, independent of the representation code under test.
PIDX = [2, 0, 1]                      # wannier90 p order (z, x, y) -> Cartesian index


def sk_model(lattice, sites, onsite, cutoff=1.3):
    """sites: list of (species, frac pos, [l, ...]); spinless, one shell per l."""
    lattice = np.asarray(lattice, dtype=float)
    orbs = []                         # (site, l, m)
    for isite, (_, _, ls) in enumerate(sites):
        for l in ls:
            orbs.extend((isite, l, m) for m in range(2*l + 1))
    vss, vsp, vpps, vppp = -1.0, 0.8, 1.2, -0.4
    ham = {}
    for rvec in np.ndindex(5, 5, 5):
        rvec = tuple(np.array(rvec) - 2)
        hmat = np.zeros((len(orbs), len(orbs)), dtype=complex)
        for i, (si, li, mi) in enumerate(orbs):
            for j, (sj, lj, mj) in enumerate(orbs):
                d = (np.array(rvec) + sites[sj][1] - sites[si][1]) @ lattice
                dist = np.linalg.norm(d)
                if dist < 1e-8:
                    if i == j:
                        hmat[i, j] = onsite[(sites[si][0], li)]
                    continue
                if dist > cutoff:
                    continue
                n, f = d / dist, np.exp(-dist)
                if li == 0 and lj == 0:
                    hmat[i, j] = vss * f
                elif li == 0:
                    hmat[i, j] = n[PIDX[mj]] * vsp * f
                elif lj == 0:
                    hmat[i, j] = -n[PIDX[mi]] * vsp * f
                else:
                    a, b = PIDX[mi], PIDX[mj]
                    hmat[i, j] = (n[a]*n[b]*(vpps - vppp) + (a == b)*vppp) * f
        if np.any(hmat):
            ham[rvec] = hmat
    return Model(nwan=len(orbs), nrpt=len(ham), deg=[1]*len(ham), ham=ham)


def shells_of(sites, nspin):
    shells, start = [], 0
    for isite, (species, pos, ls) in enumerate(sites):
        for l in ls:
            shells.append(dict(name=f"{species}{isite}_{l}", start=start, l=l, nspin=nspin,
                               pos=pos, species=species, frame=None))
            start += nspin*(2*l + 1)
    return shells


TET = np.diag([1.0, 1.0, 1.3])
CSCL = [("A", np.zeros(3), [0, 1]), ("B", np.full(3, 0.5), [1])]
CSCL_ONSITE = {("A", 0): -2.0, ("A", 1): 1.0, ("B", 1): 0.3}


def run_check(model, lattice, shells, magmoms=None, time_reversal=True, **kw):
    sites, _ = sym.group_sites(shells, lattice)
    species = sorted({s["species"] for s in sites})
    numbers = [species.index(s["species"]) for s in sites]
    ops, _ = sym.find_operations(lattice, [s["pos"] for s in sites], numbers, magmoms=magmoms)
    if time_reversal:
        ops = ops + [(w, t, True) for w, t, tr in ops if not tr]
    return sym.check_model(model, lattice, shells, ops, sym.random_kpoints(4), **kw)


class RepresentationTests(unittest.TestCase):
    def test_orbital_dmat_is_an_orthogonal_representation(self):
        rng = np.random.default_rng(1)
        for l in range(4):
            for _ in range(3):
                a = Rotation.random(random_state=rng).as_matrix() * rng.choice([-1, 1])
                b = Rotation.random(random_state=rng).as_matrix()
                da = sym.orbital_dmat(l, a)
                np.testing.assert_allclose(da @ da.T, np.eye(2*l + 1), atol=1e-10)
                np.testing.assert_allclose(sym.orbital_dmat(l, a @ b), da @ sym.orbital_dmat(l, b), atol=1e-10)
            np.testing.assert_allclose(sym.orbital_dmat(l, -np.eye(3)), (-1)**l*np.eye(2*l + 1), atol=1e-10)

    def test_c4z_sends_px_to_py(self):
        c4 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        np.testing.assert_allclose(sym.orbital_dmat(1, c4)[:, 1], [0, 0, 1], atol=1e-12)

    def test_spin_dmat_rotates_sigma_and_matches_rotate_spinor(self):
        rng = np.random.default_rng(2)
        sigma = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]
        for _ in range(5):
            rot = Rotation.random(random_state=rng).as_matrix()
            dmat = sym.spin_dmat(rot)
            for i in range(3):
                np.testing.assert_allclose(dmat.conj().T @ sigma[i] @ dmat,
                                           sum(rot[i, j]*sigma[j] for j in range(3)), atol=1e-12)
            rs = rotate_spinor(rot[:, 2], rot[:, 0])
            self.assertTrue(np.allclose(rs, dmat) or np.allclose(rs, -dmat))


class ModelCheckTests(unittest.TestCase):
    def test_spinless_sk_model_is_symmetric(self):
        model = sk_model(TET, CSCL, CSCL_ONSITE)
        res = run_check(model, TET, shells_of(CSCL, 1))
        self.assertEqual(len(res), 32)                    # P4/mmm x {E, T}
        self.assertLess(max(r["mat_err"] for r in res), 1e-10)

    def test_pure_translation_maps_equivalent_atoms(self):
        # bcc in its conventional cell: ops with t = (1/2, 1/2, 1/2) swap the two atoms
        cubic = np.eye(3)
        bcc = [("A", np.zeros(3), [0, 1]), ("A", np.full(3, 0.5), [0, 1])]
        model = sk_model(cubic, bcc, {("A", 0): -1.0, ("A", 1): 0.5}, cutoff=1.01)
        res = run_check(model, cubic, shells_of(bcc, 1), time_reversal=False)
        self.assertEqual(len(res), 96)
        self.assertLess(max(r["mat_err"] for r in res), 1e-10)

    def spinful_soc(self):
        model = sk_model(TET, CSCL, CSCL_ONSITE)
        model.addspinbasis()
        model.soc(asoc=0.2, lnum=1, sidx=2)
        model.soc(asoc=0.1, lnum=1, sidx=8)
        return model

    def test_soc_model_and_broken_onsite(self):
        model = self.spinful_soc()
        shells = shells_of(CSCL, 2)
        self.assertLess(max(r["mat_err"] for r in run_check(model, TET, shells)), 1e-10)
        # px != py on B breaks C4: eigenvalues change too
        model.ham[(0, 0, 0)][10:12, 10:12] += 0.05*np.eye(2)
        model._hstack = None
        res = run_check(model, TET, shells)
        c4 = [r for r in res if r["name"] == "4"]
        self.assertTrue(c4 and all(r["eig_err"] > 1e-3 for r in c4))
        self.assertEqual({r["worst"][:2] for r in c4}, {("B1_1", "B1_1")})
        self.assertLess(min(r["mat_err"] for r in res if r["name"] == "1"), 1e-10)

    def test_magnetic_group(self):
        model = self.spinful_soc()
        model.ham[(0, 0, 0)][8:14, 8:14] += np.kron(np.eye(3), np.diag([-0.3, 0.3]))   # m || z on B
        model._hstack = None
        shells = shells_of(CSCL, 2)
        res = run_check(model, TET, shells, magmoms=[[0, 0, 0], [0, 0, 1.0]], time_reversal=False)
        self.assertEqual(len(res), 16)                    # 4/mm'm'
        self.assertEqual(sum(r["op"][2] for r in res), 8)
        self.assertLess(max(r["mat_err"] for r in res), 1e-10)
        # pure T is broken; with inversion the spectrum still looks symmetric
        res = run_check(model, TET, shells)
        tr_only = [r for r in res if r["name"] == "1" and r["op"][2]][0]
        self.assertGreater(tr_only["mat_err"], 0.1)
        self.assertLess(tr_only["eig_err"], 1e-10)

    def test_local_axes(self):
        # B p shell expressed in local axes for orbitals and (rotate_spin) spin
        model = self.spinful_soc()
        z, x = np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        frame = sym.frame_matrix(z, x)
        smat = np.eye(model.nwan, dtype=complex)
        smat[8:14, 8:14] = np.kron(sym.orbital_dmat(1, frame), np.eye(2))
        model.rotate_basis(smat.conj().T)          # H -> S^+ H S
        model.rotate_spin(lnum=[1], zxmat=np.array([[z, x]]), index=[8])
        shells = shells_of(CSCL, 2)
        shells[2]["frame"] = (z, x)
        self.assertLess(max(r["mat_err"] for r in run_check(model, TET, shells)), 1e-10)
        res = run_check(model, TET, shells, orbital_axes=False)
        self.assertGreater(max(r["mat_err"] for r in res), 1e-2)
        self.assertLess(max(r["eig_err"] for r in res), 1e-10)

    def test_missing_partner_site_is_reported(self):
        model = sk_model(TET, CSCL, CSCL_ONSITE)
        shells = shells_of(CSCL, 1)
        sites, site_of = sym.group_sites(shells, TET)
        # mirror x -> -x + 1/4 does not map B onto a site
        op = (np.diag([-1, 1, 1]), np.array([0.25, 0, 0]), False)
        with self.assertRaises(ValueError):
            sym.op_representation(op, TET, shells, sites, site_of, model.nwan, np.eye(model.nwan))


    def test_shell_positions_outside_the_unit_cell(self):
        # hr built from the raw positions: the cell offset must reach the phase
        for bpos in ([1.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, -1.5]):
            sites = [("A", np.zeros(3), [0, 1]), ("B", np.array(bpos), [1])]
            model = sk_model(TET, sites, CSCL_ONSITE)
            res = run_check(model, TET, shells_of(sites, 1))
            self.assertEqual(len(res), 32)
            self.assertLess(max(r["mat_err"] for r in res), 1e-10, bpos)

    def test_shells_of_one_site_in_different_cells(self):
        # A s at x = 0, A p at x = 1: one site, two cell offsets
        sites = [("A", np.zeros(3), [0, 1]), ("B", np.full(3, 0.5), [1])]
        model = sk_model(TET, sites, CSCL_ONSITE)
        shells = shells_of(sites, 1)
        shells[1]["pos"] = np.array([1.0, 0.0, 0.0])
        # same Hamiltonian, A p now taken from cell +a1: H'(R)_ij = H(R + s_j - s_i)_ij
        ham = {}
        shift = np.zeros(model.nwan, dtype=int)
        shift[1:4] = 1
        for rvec, hmat in model.ham.items():
            for i in range(model.nwan):
                for j in range(model.nwan):
                    rnew = tuple(np.array(rvec) - (shift[j] - shift[i])*np.array([1, 0, 0]))
                    ham.setdefault(rnew, np.zeros_like(hmat))[i, j] = hmat[i, j]
        moved = Model(nwan=model.nwan, nrpt=len(ham), deg=[1]*len(ham), ham=ham)
        sites_, _ = sym.group_sites(shells, TET)
        self.assertEqual(len(sites_), 2)
        self.assertEqual(sites_[0]["cells"][1].tolist(), [1, 0, 0])
        self.assertLess(max(r["mat_err"] for r in run_check(moved, TET, shells)), 1e-10)

    def test_site_tolerance_follows_spglib_symprec(self):
        # B off-centre by 2e-4 (Cartesian): spglib (symprec 1e-3) keeps P4/mmm,
        # the site matching must accept the same operations
        sites = [("A", np.zeros(3), [0, 1]), ("B", np.array([0.5002, 0.5, 0.5]), [1])]
        model = sk_model(TET, sites, CSCL_ONSITE)
        res = run_check(model, TET, shells_of(sites, 1))
        self.assertEqual(len(res), 32)
        with self.assertRaises(ValueError):
            sym.check_model(model, TET, shells_of(sites, 1),
                            [(np.diag([-1, 1, 1]), np.zeros(3), False)],
                            sym.random_kpoints(2), symprec=1e-4)

    def test_empty_kpoints_are_rejected(self):
        model = sk_model(TET, CSCL, CSCL_ONSITE)
        op = [(np.eye(3, dtype=int), np.zeros(3), False)]
        with self.assertRaises(ValueError):
            sym.check_model(model, TET, shells_of(CSCL, 1), op, sym.random_kpoints(0))


class SymmetryTaskTests(unittest.TestCase):
    def test_task_end_to_end(self):
        toml = textwrap.dedent("""
        [model]
        hr_file = "x_hr.dat"
        spin = "doubled"

        [lattice]
        rvec = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.3]]

        [shell]
        A1_s = { start = 0, l = 0, pos = [0, 0, 0] }
        A1_p = { start = 1, l = 1, pos = [0, 0, 0] }
        B1_p = { start = 4, l = 1, pos = [0.5, 0.5, 0.5], z = [1, 1, 0], x = [0, 0, 1] }

        [soc]
        A1_p = 0.2

        [task.symmetry]
        nk = 3
        """)
        cwd = os.getcwd()
        directory = mpi.bcast(tempfile.mkdtemp() if mpi.rank0 else None, root=0)
        os.chdir(directory)
        try:
            # hr whose B p orbitals are given in the local axes, as wannier90 does with z=/x=
            model = sk_model(TET, CSCL, CSCL_ONSITE)
            smat = np.eye(model.nwan)
            smat[4:7, 4:7] = sym.orbital_dmat(1, sym.frame_matrix([1, 1, 0], [0, 0, 1]))
            model.rotate_basis(smat.T)
            model.to_hr("x_hr.dat")
            if mpi.rank0:
                Path("in.toml").write_text(toml)
            self.assertEqual(run.main(["in.toml"]), 0)
            log = Path("swept_run.log").read_text() if mpi.rank0 else ""
        finally:
            os.chdir(cwd)
            mpi.barrier()           # every rank out of the directory before rmtree
            if mpi.rank0:
                shutil.rmtree(directory)
        if mpi.rank0:
            self.assertIn("P4/mmm (123), 16 operations", log)
            self.assertIn("32/32 operations satisfied", log)

    def test_validate_requires_positions(self):
        base = """
        [model]
        hr_file = "x_hr.dat"
        spin = "none"
        [lattice]
        rvec = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        [shell]
        d = {{ start = 0, l = 2 {pos} }}
        [task.symmetry]
        """
        import tomllib
        with self.assertRaises(run.InputError):
            run.validate(tomllib.loads(textwrap.dedent(base.format(pos=""))))
        run.validate(tomllib.loads(textwrap.dedent(base.format(pos=", pos = [0, 0, 0]"))))

    def test_validate_rejects_nonpositive_nk_and_tolerances(self):
        import tomllib
        base = """
        [model]
        hr_file = "x_hr.dat"
        spin = "none"
        [lattice]
        rvec = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        [shell]
        d = {{ start = 0, l = 2, pos = [0, 0, 0] }}
        [task.symmetry]
        {extra}
        """
        for extra in ("nk = 0", "nk = -3", "nk = 2.5", "nk = true", "tol = 0", "symprec = -1e-3"):
            with self.assertRaises(run.InputError, msg=extra):
                run.validate(tomllib.loads(textwrap.dedent(base.format(extra=extra))))
        run.validate(tomllib.loads(textwrap.dedent(base.format(extra="nk = 1"))))

    def test_species_from_shell_name(self):
        self.assertEqual([run.shell_species(n) for n in ("Co1_d", "S2_p", "Sn_s", "O")],
                         ["Co", "S", "Sn", "O"])


if __name__ == "__main__":
    unittest.main()
