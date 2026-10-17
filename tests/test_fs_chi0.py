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
from swept import fermisurface as fs
from swept import lindhard as lh
from swept import mpiutils as mpi
from swept import run
from swept.model import Model


def random_model(nwan=3, rmax=3, seed=1):
    """Hermitian model with hoppings up to |R_i| = rmax (beyond small meshes)."""
    rng = np.random.default_rng(seed)
    ham = {}
    for r in np.ndindex(2*rmax+1, 2*rmax+1, 2*rmax+1):
        rv = tuple(int(x) - rmax for x in r)
        if rv in ham:
            continue
        h = (rng.normal(size=(nwan, nwan)) + 1j*rng.normal(size=(nwan, nwan)))*0.1/(1 + sum(map(abs, rv)))
        if rv == (0, 0, 0):
            h = h + h.conj().T
        ham[rv] = h
        ham[tuple(-x for x in rv)] = h.conj().T
    return Model(nwan=nwan, nrpt=len(ham), deg=[1]*len(ham), ham=ham)


def square_model(t=1.0, tp=0.3, tz=0.2, eps=(0.0, 0.1)):
    """Two orbitals on a tetragonal lattice, D4h-like: orbital 0 hops along x
    more strongly, orbital 1 along y (an xz/yz pair), mixed by tp on diagonals."""
    ham = {}
    def add(r, m):
        ham[r] = ham.get(r, 0) + m
    add((0, 0, 0), np.diag(eps).astype(complex))
    for s in (1, -1):
        add((s, 0, 0), np.diag([-t, -0.3*t]).astype(complex))
        add((0, s, 0), np.diag([-0.3*t, -t]).astype(complex))
        add((0, 0, s), np.diag([-tz, -tz]).astype(complex))
    for sx in (1, -1):
        for sy in (1, -1):
            add((sx, sy, 0), np.array([[0, sx*sy*tp], [sx*sy*tp, 0]], dtype=complex))
    return Model(nwan=2, nrpt=len(ham), deg=[1]*len(ham), ham=ham)


class HkGridTests(unittest.TestCase):
    def test_slab_fft_matches_direct_sum(self):
        model = random_model()
        mesh = (2, 3, 4)
        points = lh.mesh_points(mesh)
        for i0 in range(mesh[0]):
            sel = points[:, 0] == i0
            direct = model.get_bulk_Hks(points[sel]/np.array(mesh))
            np.testing.assert_allclose(model.get_bulk_Hk_slab(mesh, i0), direct, atol=1e-12)

    def test_batch_matches_single(self):
        model = random_model(seed=2)
        ks = np.random.default_rng(0).random((5, 3))
        for k, hk in zip(ks, model.get_bulk_Hks(ks)):
            np.testing.assert_allclose(hk, model.get_bulk_Hk(k), atol=1e-12)


def shifted_model(model, orb, L):
    """Same crystal, orbital `orb` represented one cell over: H'_ab(R) = H_ab(R + L_b - L_a)."""
    Ls = np.zeros((model.nwan, 3), int)
    Ls[orb] = L
    new = {}
    for R, h in model.ham.items():
        for a in range(model.nwan):
            for b in range(model.nwan):
                Rn = tuple(int(x) for x in np.array(R) - Ls[b] + Ls[a])
                new.setdefault(Rn, np.zeros((model.nwan, model.nwan), complex))[a, b] += h[a, b]
    return Model(nwan=model.nwan, nrpt=len(new), deg=[1]*len(new), ham=new)


def hop_model(nwan, hops):
    """Hermitian model from (R, a, b, t) entries; the conjugate partner is added."""
    ham = {}
    for r, a, b, t in hops:
        ham.setdefault(tuple(r), np.zeros((nwan, nwan), complex))[a, b] += t
        ham.setdefault(tuple(-x for x in r), np.zeros((nwan, nwan), complex))[b, a] += np.conj(t)
    return Model(nwan=nwan, nrpt=len(ham), deg=[1]*len(ham), ham=ham)


def brute_point(model, mesh, mu, kT, Q, tau, orbitals):
    """Independent reference: point-vertex response at Q (any cell) with explicit
    per-orbital positions tau_a; sums over the mesh with H(k+Q) evaluated directly."""
    kf = lh.mesh_points(mesh)/np.array(mesh, float)
    e1, v1 = np.linalg.eigh(model.get_bulk_Hks(kf))
    e2, v2 = np.linalg.eigh(model.get_bulk_Hks(kf + np.asarray(Q)))
    e1, e2 = e1 - mu, e2 - mu
    ph = np.exp(-2j*np.pi*np.asarray(tau)[orbitals] @ np.asarray(Q, float))
    o = np.einsum("a,kan,kam->knm", ph, v1[:, orbitals, :].conj(), v2[:, orbitals, :])
    f1, f2 = lh.fermi(e1, kT)[:, :, None], lh.fermi(e2, kT)[:, None, :]
    de = e1[:, :, None] - e2[:, None, :]
    deg = np.abs(de) < 1e-12
    F = np.where(deg, f1*(1 - f1)/kT, -(f1 - f2)/np.where(deg, 1.0, de))
    return float(np.sum(np.abs(o)**2*F)/len(kf))


QUIET = dict(log=lambda m: None)


class FermiDifferenceTests(unittest.TestCase):
    def test_symmetric_accurate_and_finite(self):
        rng = np.random.default_rng(5)
        for kT in (0.05, 0.001):
            a = rng.normal(0, 5*kT, 2000)
            b = np.concatenate([a[:1000] + rng.normal(0, 1e-9, 1000), rng.normal(0, 5*kT, 1000)])
            np.testing.assert_array_equal(lh.fermi_difference(a, b, kT), lh.fermi_difference(b, a, kT))
            far = np.abs(a - b) > 1e-3
            direct = -(lh.fermi(a, kT) - lh.fermi(b, kT))[far]/(a - b)[far]
            np.testing.assert_allclose(lh.fermi_difference(a, b, kT)[far], direct, rtol=1e-7, atol=1e-300)
            same = 1/(4*kT*np.cosh(a/(2*kT))**2)          # = f(1-f)/kT without the 1-f cancellation
            np.testing.assert_allclose(lh.fermi_difference(a, a, kT), same, rtol=1e-12, atol=1e-300)
        big = lh.fermi_difference(np.array([40.0, -40.0, 5.0]), np.array([41.0, -40.0, -5.0]), 1e-3)
        self.assertTrue(np.all(np.isfinite(big)) and big[0] == 0.0 and big[1] == 0.0)
        self.assertAlmostEqual(big[2], 0.1, places=12)


    def test_low_temperature_against_high_precision(self):
        # implementation review 2026-10-16: |u-v| - |u| - |v| cancelled for kT -> 0
        import mpmath as mp
        mp.mp.dps = 50

        def reference(a, b, kT):          # same identity, 50 digits: sinhc(d)/(4 kT cosh u cosh v)
            u, v = mp.mpf(a)/(2*mp.mpf(kT)), mp.mpf(b)/(2*mp.mpf(kT))
            d = u - v
            sinhc = mp.sinh(d)/d if d != 0 else mp.mpf(1)
            return sinhc/(4*mp.mpf(kT)*mp.cosh(u)*mp.cosh(v))

        rng = np.random.default_rng(11)
        worst = 0.0
        for kT in [1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-20]:
            pairs = [(1.0, -1.0), (0.3, -0.7), (0.0, 0.0), (2*kT, -3*kT), (kT, kT*(1 + 1e-6))]
            pairs += [tuple(x) for x in rng.normal(0, 3*kT, (20, 2))]
            for a, b in pairs:
                got = float(lh.fermi_difference(a, b, kT))
                self.assertEqual(got, float(lh.fermi_difference(b, a, kT)))
                ref = float(reference(a, b, kT))
                if ref > 1e-290:
                    worst = max(worst, abs(got - ref)/ref)
                else:
                    self.assertLess(got, 1e-280)
        self.assertLess(worst, 1e-12)
        self.assertAlmostEqual(float(lh.fermi_difference(1.0, -1.0, 1e-20)), 0.5, places=14)


class LindhardTests(unittest.TestCase):
    def setUp(self):
        # random complex model; orbitals 0, 1 on site 0 (orbital 1 represented one cell over
        # in the shifted copy), orbital 2 on site 1
        self.model = random_model(nwan=3, rmax=1, seed=7)
        self.tau = np.array([[0.1, 0.2, 0.0], [0.1, 0.2, 0.0], [0.6, 0.5, 0.3]])
        self.mesh, self.mu, self.kT = (3, 4, 2), 0.05, 0.05

    def channel(self, offsets):
        return lh.make_channel([0, 1, 2], [0, 0, 1], offsets, self.tau[[0, 2]])

    def test_matches_brute_force_with_positions_and_offsets(self):
        L = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
        moved = shifted_model(self.model, 1, L[1])
        res = lh.lindhard(moved, self.mesh, self.mu, self.kT, 100.0, [self.channel(L)], **QUIET)
        res0 = lh.lindhard(self.model, self.mesh, self.mu, self.kT, 100.0,
                           [self.channel(np.zeros((3, 3), int))], **QUIET)
        np.testing.assert_allclose(res["chi"][0], res0["chi"][0], atol=1e-13)   # representation invariance
        for n in lh.mesh_points(self.mesh):
            for G in [(0, 0, 0), (1, -1, 2), (-2, 0, 1)]:
                Q = n/np.array(self.mesh) + np.array(G)
                ref = brute_point(self.model, self.mesh, self.mu, self.kT, Q, self.tau, [0, 1, 2])
                got = lh.chi_physical(res["chi"][0], self.tau[[0, 2]], self.mesh, Q)[0]
                self.assertAlmostEqual(got, ref, places=12)

    def test_naive_grouping_without_offsets_differs(self):
        # follow-up review: same atom, second orbital represented one cell over
        h = np.array([[0., 1.], [1., 0.]], complex)
        m0 = Model(nwan=2, nrpt=1, deg=[1], ham={(0, 0, 0): h})
        m1 = shifted_model(m0, 1, (1, 0, 0))
        mesh, Q = (16, 1, 1), [0.25, 0, 0]
        ref = brute_point(m0, mesh, 0.0, 0.05, Q, np.zeros((2, 3)), [0, 1])
        ok = lh.lindhard(m1, mesh, 0.0, 0.05, 100.0,
                         [lh.make_channel([0, 1], [0, 0], [[0, 0, 0], [1, 0, 0]], [[0, 0, 0]])], **QUIET)
        self.assertAlmostEqual(lh.chi_physical(ok["chi"][0], [[0, 0, 0]], mesh, Q)[0], ref, places=13)
        bad = lh.lindhard(m1, mesh, 0.0, 0.05, 100.0, [lh.cell_channel([0, 1])], **QUIET)
        self.assertGreater(abs(lh.chi_physical(bad["chi"][0], [[0, 0, 0]], mesh, Q)[0] - ref), 0.49)

    def test_q_pairing_matches_full_and_matrices_are_psd(self):
        chan = [self.channel(np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])), lh.cell_channel([1, 2])]
        mesh = (4, 4, 2)                                    # several self-paired points
        paired = lh.lindhard(self.model, mesh, self.mu, self.kT, 100.0, chan, **QUIET)
        full = lh.lindhard(self.model, mesh, self.mu, self.kT, 100.0, chan, use_pairs=False, **QUIET)
        for a, b in zip(paired["chi"], full["chi"]):
            np.testing.assert_allclose(a, b, atol=1e-14)
            np.testing.assert_allclose(a, a.conj().transpose(0, 2, 1), atol=1e-15)
            self.assertGreater(min(np.linalg.eigvalsh(m).min() for m in a), -1e-14)

    def test_cell_vertex_is_lattice_gauge_sum(self):
        res = lh.lindhard(self.model, self.mesh, self.mu, self.kT, 100.0, [lh.cell_channel([0, 2])], **QUIET)
        for n in lh.mesh_points(self.mesh):
            Q = n/np.array(self.mesh)
            ref = brute_point(self.model, self.mesh, self.mu, self.kT, Q, np.zeros((3, 3)), [0, 2])
            self.assertAlmostEqual(lh.chi_physical(res["chi"][0], [[0, 0, 0]], self.mesh, Q)[0], ref, places=12)

    def test_checkerboard_c4_and_mirrors(self):
        # A at (0,0), B at (1/2,1/2); C4 about A maps B to B + (-1, 0)
        hops = [(r, 0, 1, -1.0) for r in [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0)]]
        hops += [(r, a, a, -0.3) for r in [(1, 0, 0), (0, 1, 0)] for a in (0, 1)]
        hops += [((0, 0, 0), 1, 1, 0.2)]                    # onsite 0.4 after the conjugate partner
        model = hop_model(2, hops)
        mesh = (8, 8, 1)
        tau0 = [[0, 0, 0], [0.5, 0.5, 0]]
        res = lh.lindhard(model, mesh, 0.0, 0.05, 100.0, [lh.make_channel([0, 1], [0, 1], np.zeros((2, 3)), tau0)],
                          **QUIET)
        Q = np.array([(i/8, j/8, 0) for i in range(-8, 16) for j in range(-8, 16)])
        ref = lh.chi_physical(res["chi"][0], tau0, mesh, Q)
        for g in [[[0, -1, 0], [1, 0, 0], [0, 0, 1]], [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  [[0, 1, 0], [1, 0, 0], [0, 0, 1]]]:
            np.testing.assert_allclose(lh.chi_physical(res["chi"][0], tau0, mesh, Q @ np.array(g).T), ref,
                                       atol=1e-12)
        cell = lh.lindhard(model, mesh, 0.0, 0.05, 100.0, [lh.cell_channel([0, 1])], **QUIET)
        c4 = lh.chi_physical(cell["chi"][0], [[0, 0, 0]], mesh, Q @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T)
        self.assertGreater(np.abs(c4 - lh.chi_physical(cell["chi"][0], [[0, 0, 0]], mesh, Q)).max(), 1e-3)

    def test_doubled_chain_equals_monatomic_chain(self):
        mu = -0.3
        m2 = hop_model(2, [((0, 0, 0), 0, 1, -1.0), ((-1, 0, 0), 0, 1, -1.0), ((0, 0, 0), 0, 0, -mu/2),
                           ((0, 0, 0), 1, 1, -mu/2)])
        m1 = hop_model(1, [((1, 0, 0), 0, 0, -1.0), ((0, 0, 0), 0, 0, -mu/2)])
        n2 = 16
        r2 = lh.lindhard(m2, (n2, 1, 1), 0.0, 0.05, 100.0,
                         [lh.make_channel([0, 1], [0, 1], np.zeros((2, 3)), [[0, 0, 0], [0.5, 0, 0]])], **QUIET)
        r1 = lh.lindhard(m1, (2*n2, 1, 1), 0.0, 0.05, 100.0, [lh.cell_channel([0])], **QUIET)
        m = np.arange(0, 4*n2)
        chi2 = lh.chi_physical(r2["chi"][0], [[0, 0, 0], [0.5, 0, 0]], (n2, 1, 1),
                               np.stack([m/n2, 0*m, 0*m], 1))
        chi1 = lh.chi_physical(r1["chi"][0], [[0, 0, 0]], (2*n2, 1, 1), np.stack([m/(2*n2), 0*m, 0*m], 1))
        np.testing.assert_allclose(chi2, 2*chi1, atol=1e-12)        # per doubled cell = 2 x per small cell

    def test_near_degenerate_gamma_is_accepted(self):
        # follow-up review v2: levels 8e-10 eV apart, complex eigenvectors, self-paired Gamma
        u = np.exp(2j*np.pi*np.arange(3)[:, None]*np.arange(3)[None, :]/3)/np.sqrt(3)
        for kT in (0.05, 0.001):
            e = 1.3*kT + np.array([-4e-10, 4e-10, kT])
            model = Model(nwan=3, nrpt=1, deg=[1], ham={(0, 0, 0): u @ np.diag(e) @ u.conj().T})
            res = lh.lindhard(model, (1, 1, 1), 0.0, kT, 100.0,
                              [lh.make_channel([0, 1, 2], [0, 1, 2], np.zeros((3, 3)), np.zeros((3, 3)))], **QUIET)
            m = res["chi"][0][0]
            self.assertEqual(np.abs(m.imag).max(), 0.0)
            self.assertGreater(np.linalg.eigvalsh(m).min(), 0)

    def test_chi_physical_rejects_off_mesh_q(self):
        with self.assertRaises(ValueError):
            lh.chi_physical(np.zeros((8, 1, 1)), [[0, 0, 0]], (2, 2, 2), [0.25, 0, 0])


class FermiSurfaceTests(unittest.TestCase):
    def test_plane_energies_and_weights(self):
        model = square_model()
        bvec = 2*np.pi*np.eye(3)
        s, t, kfrac, kcart = fs.plane_kpoints([0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], (5, 7), bvec, "frac")
        self.assertEqual(kfrac.shape, (35, 3))
        np.testing.assert_allclose(kfrac[0], [-0.5, -0.5, 0])
        np.testing.assert_allclose(kfrac[1], [-0.5, -0.5 + 1/6, 0])       # t runs fastest
        eigs, projs = fs.plane_eigens(model, kfrac, [np.array([0]), np.array([1])], chunk=4)
        np.testing.assert_allclose(eigs, model.get_bulk_eigvals(kfrac).T, atol=1e-12)
        np.testing.assert_allclose(projs.sum(axis=0), 1.0, atol=1e-12)
        # cart basis gives the same points
        _, _, kf2, _ = fs.plane_kpoints([0, 0, 0], [np.pi, 0, 0], [0, np.pi, 0], (5, 7), bvec, "cart")
        np.testing.assert_allclose(kf2, kfrac, atol=1e-12)

    def test_bz_function(self):
        bvec = 2*np.pi*np.eye(3)
        val = fs.bz_function(np.array([[0, 0, 0], [np.pi, 0, 0], [1.2*np.pi, 0, 0]]), bvec)
        self.assertLess(val[0], 0)
        self.assertAlmostEqual(val[1], 0.0, places=10)
        self.assertGreater(val[2], 0)

    def test_bz_function_matches_brute_force(self):
        def inside_brute(k, b, n=14):
            r = np.arange(-n, n+1)
            nn = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
            g = nn[np.any(nn != 0, 1)] @ b
            return np.max(2*k @ g.T - (g**2).sum(1), 1) <= 1e-12
        a, b, c = 4.123, 6.086, 13.812
        rvec = np.array([[-a/2, b/2, c/2], [a/2, -b/2, c/2], [a/2, b/2, -c/2]])
        cases = [np.eye(3), np.array([[1., 0, 0], [3., 1, 0], [0, 0, 1.]]),    # review #4 skew basis
                 np.diag([1., 1., 100.]), np.array([[1., 0, 0], [0.3, 1, 0], [7, 2, 60.]]),
                 2*np.pi*np.linalg.inv(rvec).T] + \
                [np.random.default_rng(10 + s).normal(size=(3, 3)) for s in range(3)]
        rng = np.random.default_rng(3)
        for bv in cases:
            g = fs.voronoi_vectors(bv)
            self.assertEqual(len(g), 14)
            k = rng.uniform(-2, 2, (2000, 3))*np.linalg.norm(bv, axis=1).max()
            k = np.vstack([k, 0.5*g*(1 - 1e-9), 0.5*g*(1 + 1e-9)])          # just inside / outside
            np.testing.assert_array_equal(fs.bz_function(k, bv, chunk=333) <= 1e-12, inside_brute(k, bv))
        # implementation review: anisotropic and strongly sheared bases of known lattices
        square_xy = np.array([[1.0, 0, 0], [3.0, 1.0, 0], [0, 0, 1e6]])     # x-y square, c* huge
        self.assertGreater(fs.bz_function(np.array([[0, 0.6, 0]]), square_xy)[0], 0)
        self.assertLess(fs.bz_function(np.array([[0, 0.4, 0], [0.45, -0.45, 3e5]]), square_xy).max(), 0)
        kc = rng.uniform(-0.8, 0.8, (2000, 3))
        cube = np.max(np.abs(kc), axis=1) <= 0.5
        for shear in ([[1, 0, 0], [3, 1, 0], [10000, 0, 1]], [[1, 0, 0], [0, 1, 0], [123456, -98765, 1]]):
            sheared = np.array(shear, float)                      # unimodular: same cubic lattice
            np.testing.assert_array_equal(fs.bz_function(kc, sheared) <= 0, cube)
        bte = cases[4]
        for seed in range(3):
            m = np.eye(3, dtype=int)
            r = np.random.default_rng(seed)
            for _ in range(6):                                    # random unimodular shear, large entries
                i, j = r.choice(3, 2, replace=False)
                m[i] += r.integers(-60, 60)*m[j]
            k = rng.uniform(-2, 2, (2000, 3))*np.linalg.norm(bte, axis=1).max()
            np.testing.assert_array_equal(fs.bz_function(k, m @ bte) <= 1e-12, fs.bz_function(k, bte) <= 1e-12)
        # follow-up review: far point of an extended cut no longer aborts
        v = fs.bz_function(np.array([[6., 0, 0], [0, 0.6, 0]]), np.array([[1., 0, 0], [3., 1, 0], [0, 0, 1.]]))
        self.assertTrue(np.all(v > 0))

    def test_plane_axes_non_orthogonal(self):
        u, v = np.array([1.0, 0, 0]), np.array([0.5, 1.0, 0])
        e1, e2, st2xy = fs.plane_axes(u, v)
        np.testing.assert_allclose(np.array([1, 1]) @ st2xy, [1.5, 1.0])


INPUT = """
[model]
hr_file = "x_hr.dat"
spin = "none"

[lattice]
rvec = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.5]]

[shell]
a = { start = 0, l = 0, pos = [0.0, 0.0, 0.0] }
b = { start = 1, l = 0, pos = [0.5, 0.5, 0.0] }

[projection]
A = { shells = ["a"] }
B = { shells = ["b"] }
"""


class TaskTests(unittest.TestCase):
    def test_fermi_surface_without_projections_or_bands(self):
        # review 2026-10-16, #1: project omitted, and a plane where no band is kept
        cfg = dict(lattice=dict(rvec=np.eye(3).tolist()))
        model = Model(nwan=2, nrpt=1, deg=[1], ham={(0, 0, 0): np.diag([-0.1, 5.0]).astype(complex)})
        cwd = os.getcwd()
        directory = mpi.bcast(tempfile.mkdtemp() if mpi.rank0 else None, root=0)
        os.chdir(directory)
        try:
            for mu, nkept in [(0.0, 1), (2.0, 0)]:
                task = dict(nk=[5, 5], window=0.5, mu=mu, output="fs.npz",
                            plane=[dict(name="kz0", origin=[0, 0, 0], u=[.5, 0, 0], v=[0, .5, 0])])
                run.task_fermi_surface(cfg, task, model, {}, lambda m: None)
                mpi.barrier()
                if mpi.rank0:
                    data = dict(np.load("fs.npz"))
                    self.assertEqual(data["kz0_E"].shape, (5, 5, nkept))
                    self.assertEqual(data["kz0_W"].shape, (0, 5, 5, nkept))
                    from swept import plot_fermi_surface
                    plot_fermi_surface.main(["fs.npz", "-o", "fs.pdf"])
                mpi.barrier()
        finally:
            os.chdir(cwd)
            mpi.barrier()
            if mpi.rank0:
                shutil.rmtree(directory)

    def test_validation(self):
        good_fs = """
        [task.fermi_surface]
        nk = [11, 11]
        [[task.fermi_surface.plane]]
        name = "kz0"
        origin = [0, 0, 0]
        u = [0.5, 0, 0]
        v = [0, 0.5, 0]
        """
        run.validate(tomllib.loads(INPUT + textwrap.dedent(good_fs)))
        for bad in [good_fs.replace("v = [0, 0.5, 0]", "v = [1.0, 0, 0]"),     # parallel
                    good_fs.replace("nk = [11, 11]", "nk = [11]"),
                    good_fs.replace('name = "kz0"', 'name = "kz0"\nw = 1')]:
            with self.assertRaises(run.InputError):
                run.validate(tomllib.loads(INPUT + textwrap.dedent(bad)))
        chi0 = "[task.chi0]\nmesh = [4, 4, 4]\nkT = 0.1\n"
        run.validate(tomllib.loads(INPUT + chi0 + "vertex = 'cell'\nwindow_check = 2.0\n"))
        for extra in ["kT = -0.1\n", "project = ['C']\n", "symmetry = false\n", "sym_tol = 1e-3\n",
                      "vertex = 'site'\n", "total = false\n", "window_check = 1.0\n", "kblock = 0\n",
                      "project = ['A', 'A']\n"]:
            text = INPUT + (chi0.replace("kT = 0.1\n", extra) if extra.startswith("kT") else chi0 + extra)
            with self.assertRaises(run.InputError, msg=extra):
                run.validate(tomllib.loads(text))

    def run_input(self, text, model):
        cwd = os.getcwd()
        directory = mpi.bcast(tempfile.mkdtemp() if mpi.rank0 else None, root=0)
        os.chdir(directory)
        out = {}
        try:
            model.to_hr("x_hr.dat")
            if mpi.rank0:
                Path("in.toml").write_text(text)
            mpi.barrier()
            out["status"] = run.main(["in.toml"])
            if mpi.rank0:
                for name in ("fs.npz", "chi0.npz", "cell.npz"):
                    if Path(name).exists():
                        out[name] = dict(np.load(name))
                if "fs.npz" in out:
                    from swept import plot_fermi_surface
                    plot_fermi_surface.main(["fs.npz", "-o", "fs.pdf"])
                    out["pdf"] = Path("fs.pdf").exists()
        finally:
            os.chdir(cwd)
            mpi.barrier()
            if mpi.rank0:
                shutil.rmtree(directory)
        return out

    def test_channel_name_collision(self):
        # implementation review: a projection called "total" overwrote the all-orbital channel
        text = INPUT + 'total = { index = [0] }\n[task.chi0]\nmesh = [1, 1, 1]\nkT = 0.1\nproject = ["total"]\n'
        with self.assertRaises(run.InputError):
            run.validate(tomllib.loads(text))
        run.validate(tomllib.loads(text + "total = false\n"))

    def test_end_to_end(self):
        tasks = """
        [task.fermi_surface]
        nk = [21, 21]
        window = 10.0
        project = ["A", "B"]
        output = "fs.npz"
        [[task.fermi_surface.plane]]
        name = "kz0"
        origin = [0, 0, 0]
        u = [0.5, 0, 0]
        v = [0, 0.5, 0]

        [task.chi0]
        mesh = [6, 6, 4]
        kT = 0.05
        window = 1.0
        window_check = 20.0
        kblock = 50
        project = ["A"]
        output = "chi0.npz"
        """
        out = self.run_input(INPUT + textwrap.dedent(tasks), square_model())
        self.assertEqual(out["status"], 0)
        if not mpi.rank0:
            return
        fsd, chi = out["fs.npz"], out["chi0.npz"]
        self.assertTrue(out["pdf"])
        self.assertEqual(fsd["kz0_E"].shape, (21, 21, 2))
        np.testing.assert_allclose(fsd["kz0_W"].sum(axis=0), 1.0, atol=1e-5)    # hr has 6 decimals
        self.assertEqual((str(chi["format"]), int(chi["version"]), str(chi["vertex_model"])),
                         ("swept-chi0", 2, "point_site"))
        self.assertEqual(list(chi["channels"]), ["total", "A"])
        self.assertEqual(chi["chi_total"].shape, (144, 2, 2))
        self.assertEqual(chi["chi_A"].shape, (144, 1, 1))
        np.testing.assert_allclose(chi["tau0_total"], [[0, 0, 0], [0.5, 0.5, 0]])
        q = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1 + 1/6, -0.5, 0.25]])
        vals = lh.chi_physical_npz(chi, "total", q)
        self.assertTrue(np.all(vals > 0))
        self.assertEqual(chi["window_check"].shape, (4, 6))
        self.assertIn("B/2", str(chi["normalization"]))

    def test_point_site_needs_positions(self):
        no_pos = INPUT.replace(", pos = [0.5, 0.5, 0.0]", "")
        text = no_pos + "[task.chi0]\nmesh = [2, 2, 2]\nkT = 0.1\noutput = 'chi0.npz'\n"
        self.assertEqual(self.run_input(text, square_model())["status"], 1)
        # projection A only uses the shell with pos: fine; cell vertex never needs pos
        text = no_pos + "[task.chi0]\nmesh = [2, 2, 2]\nkT = 0.1\ntotal = false\nproject = ['A']\n"
        self.assertEqual(self.run_input(text, square_model())["status"], 0)
        text = no_pos + "[task.chi0]\nmesh = [2, 2, 2]\nkT = 0.1\nvertex = 'cell'\noutput = 'cell.npz'\n"
        out = self.run_input(text, square_model())
        self.assertEqual(out["status"], 0)
        if mpi.rank0:
            self.assertIn("tau = 0", str(out["cell.npz"]["position_source"]))


if __name__ == "__main__":
    unittest.main()
