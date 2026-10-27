"""Unified DMFT export (UTe2/docs/PROPOSAL_interleaved_flavor_v5_2026-10-25.md, section 3):
Kramers clusters (TK), orbital_spin bases (TO), the real-harmonic reference (TP1, T8p), the
E_mesh definition (T1E) and a write/read-back round trip of eig.dat, udmft.dat, Trans.dat
and dmft_basis.json."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from swept import udmft
from swept.basis import complex2real


def ls_real(ll, zeta):
    """zeta L.S in the spin-interleaved real-harmonic basis."""
    m = np.arange(-ll, ll+1, dtype=float)
    lp = np.diag(np.sqrt(ll*(ll+1) - m[:-1]*(m[:-1]+1)), -1)
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    ls = np.kron(np.diag(m), np.diag([0.5, -0.5])) + 0.5*(np.kron(lp, sp.T) + np.kron(lp.T, sp))
    c2r = complex2real(ll, 2)
    return zeta*c2r.conj().T @ ls @ c2r


def ops(ll):
    o = udmft.moment_operators(ll)
    return [(k, o[k]) for k in ("Mz", "Mx", "My")]


def check_basis(tc, E, von, ll):
    U = udmft.time_reversal_matrix(2*ll+1)
    n = len(E)
    np.testing.assert_allclose(von.conj().T @ von, np.eye(n), atol=1e-12)
    np.testing.assert_allclose(von[:, 1::2], U @ von[:, 0::2].conj(), atol=1e-10)
    H = von.conj().T @ E @ von
    tc.assertLessEqual(np.abs(H - np.diag(np.diag(H))).max(), 1e-10)


class KramersClusterTests(unittest.TestCase):                     # TK
    def test_pure_soc_f_shell(self):
        E = ls_real(3, 0.3)
        von, rules = udmft.kramers_basis(E, udmft.time_reversal_matrix(7), ops(3))
        check_basis(self, E, von, 3)
        self.assertEqual(set(rules), {"Mz"})
        mz = von.conj().T @ udmft.moment_operators(3)["Mz"] @ von
        for blk in (slice(0, 6), slice(6, 14)):                       # j = 5/2 and 7/2 multiplets: Jz eigenstates
            m = mz[blk, blk]
            self.assertLess(np.abs(m - np.diag(np.diag(m))).max(), 1e-10)
        self.assertGreater(np.abs(mz[:6, 6:]).max(), 0.1)             # 2Sz couples the two multiplets

    def test_cubic_quartet(self):
        cf = np.diag([0.6, 0.0, 0.0, 0.6, 0.0])                      # eg (z2, x2-y2) / t2g
        E = np.kron(cf, np.eye(2)) + ls_real(2, 0.08)
        e = np.linalg.eigvalsh(E)
        self.assertLess(e[3] - e[0], 1e-12)                           # a Gamma8 quartet exists
        von, rules = udmft.kramers_basis(E, udmft.time_reversal_matrix(5), ops(2))
        check_basis(self, E, von, 2)

    def test_zero_mz_subspace_uses_mx(self):
        E = ls_real(2, 0.1)
        o = udmft.moment_operators(2)
        von, rules = udmft.kramers_basis(E, udmft.time_reversal_matrix(5),
                                         [("Mz", 0*o["Mz"]), ("Mx", o["Mx"]), ("My", o["My"])])
        check_basis(self, E, von, 2)
        self.assertEqual(set(rules), {"Mx"})

    def test_degenerate_top_eigenvalue_uses_mx(self):
        E = np.zeros((10, 10))                                        # fully degenerate d shell
        o = udmft.moment_operators(2)
        two_sz = np.kron(np.eye(5), np.diag([1.0, -1.0]))
        von, rules = udmft.kramers_basis(E, udmft.time_reversal_matrix(5),
                                         [("2Sz", two_sz), ("Mx", o["Mx"]), ("My", o["My"])])
        check_basis(self, E, von, 2)
        self.assertEqual(rules[0], "Mx")

    def test_near_degenerate_cluster_is_split(self):
        # review R12: two Kramers doublets 5e-9 eV apart, Mz mixing them
        E = np.diag([0.0, 0.0, 5e-9, 5e-9])
        mz = np.kron(np.array([[0, 1], [1, 0]]), np.diag([1.0, -1.0]))
        von, _ = udmft.kramers_basis(E, udmft.time_reversal_matrix(2), [("Mz", mz)])
        H = von.conj().T @ E @ von
        self.assertLessEqual(np.abs(H - np.diag(np.diag(H))).max(), 1e-10)
        np.testing.assert_allclose(von[:, 1::2], udmft.time_reversal_matrix(2) @ von[:, 0::2].conj(), atol=1e-10)

    def test_rejects_magnetic(self):
        E = ls_real(2, 0.1) + 0.05*np.kron(np.eye(5), np.diag([1.0, -1.0]))
        with self.assertRaises(ValueError):
            udmft.kramers_basis(E, udmft.time_reversal_matrix(5), ops(2))


class OrbitalSpinTests(unittest.TestCase):                         # TO
    def embed(self, hu, hd):
        n = len(hu)
        E = np.zeros((2*n, 2*n))
        E[0::2, 0::2], E[1::2, 1::2] = hu, hd
        return E

    def check(self, E, von):
        np.testing.assert_allclose(von.conj().T @ von, np.eye(len(E)), atol=1e-12)
        H = von.conj().T @ E @ von
        self.assertLess(np.abs(H - np.diag(np.diag(H))).max(), 1e-10)
        V = von[0::2, 0::2]
        np.testing.assert_allclose(von, np.kron(V, np.eye(2)), atol=1e-14)

    def test_average_degenerate(self):
        sx = np.array([[0, 1], [1, 0]], float)
        hu = np.zeros((5, 5)); hd = np.zeros((5, 5))
        hu[:2, :2], hd[:2, :2] = sx, -sx
        hu[2:, 2:] = hd[2:, 2:] = np.diag([1.0, 2.0, 3.0])
        E = self.embed(hu, hd)
        self.check(E, udmft.orbital_spin_basis(E))

    def test_common_rotation(self):
        q = np.linalg.qr(np.random.default_rng(1).normal(size=(5, 5)))[0]
        E = self.embed(q @ np.diag([0, 1, 2, 3, 4.0]) @ q.T, q @ np.diag([0.3, 1.1, 2.5, 3.2, 4.4]) @ q.T)
        self.check(E, udmft.orbital_spin_basis(E))

    def test_rejects_different_rotations(self):
        rng = np.random.default_rng(2)
        a, b = rng.normal(size=(5, 5)), rng.normal(size=(5, 5))
        with self.assertRaises(ValueError):
            udmft.orbital_spin_basis(self.embed(a + a.T, b + b.T))

    def test_rejects_spin_mixing(self):
        with self.assertRaises(ValueError):
            udmft.orbital_spin_basis(ls_real(2, 0.1))


class RealHarmonicReferenceTests(unittest.TestCase):               # TP1, T8p
    def test_reference_is_unitary(self):
        for ll in range(4):
            C = udmft.real_harmonic_reference(ll)
            self.assertLess(np.abs(C @ C.conj().T - np.eye(2*ll+1)).max(), 1e-13)

    def test_complex2real_matches_reference(self):
        for ll in range(4):                                           # [m, r] = C_ref.T, plain transpose
            self.assertLess(udmft.check_complex2real(ll), 1e-12, f"l={ll}")

    def test_sign_or_phase_error_is_detected(self):
        orig = udmft.complex2real
        for factor in (-1.0, 1j):
            def bad(ll, ns, factor=factor):
                t = orig(ll, 1).copy()
                t[:, 2] *= factor                                     # one real orbital
                return t if ns == 1 else orig(ll, ns)
            with mock.patch.object(udmft, "complex2real", bad):
                self.assertGreater(udmft.check_complex2real(2), 0.5)


def synthetic_tb(ll, natm, extra, seed, H_shell=None):
    """One-k or few-k model: Wannier basis = natm shells (2(2l+1) each) + extra orbitals."""
    rng = np.random.default_rng(seed)
    nf = 2*(2*ll+1)
    nwan = natm*nf + extra
    if H_shell is None:
        H_shell = np.kron(np.diag(rng.uniform(0, 1, 2*ll+1)), np.eye(2)) + ls_real(ll, 0.05)
    H = np.zeros((nwan, nwan), complex)
    for a in range(natm):
        H[a*nf:(a+1)*nf, a*nf:(a+1)*nf] = H_shell
    H[natm*nf:, natm*nf:] = np.diag(rng.uniform(-2, 2, extra))
    return H


class ExportTests(unittest.TestCase):                               # T1E and round trip
    def test_r12_counterexample_not_rotated_twice(self):
        c, s = np.cos(0.4), np.sin(0.4)
        V = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        h = V @ np.diag([1.0, 3.0, 5.0]) @ V.T
        H = np.kron(h, np.eye(2))                                     # l = 1, no SOC
        e, v = np.linalg.eigh(H)
        data = udmft.build_export(e[None, :], v[None, :, :], H, [dict(indices=list(range(6)))], 1,
                                  "orbital_spin", 6.0, 0.0, log=lambda *_: None)
        np.testing.assert_allclose(np.sort(data["eimp_abs"]), [1, 1, 3, 3, 5, 5], atol=1e-12)

    def read_files(self, d, nf, natm):
        lines = (Path(d)/"eig.dat").read_text().splitlines()
        nk, _, norb, _, _ = (int(x) for x in lines[0].split()[:5])
        t = lines[1].split()
        nb, mu = int(t[2]), float(t[4])
        eps, w, p = [], [], 3
        for _ in range(nk):
            w.append(float(lines[p].split()[1])); p += 1
            eps.append([float(lines[p+b].split()[1]) for b in range(nb)]); p += nb + 2
        rows = [ln.split() for ln in (Path(d)/"udmft.dat").read_text().splitlines()[3:]
                if ln.strip() and "#" not in ln]
        data = np.array(rows, float)
        P = (data[:, 2] + 1j*data[:, 3]).reshape(nk, natm*norb, nb).transpose(0, 2, 1)
        return np.array(eps), np.array(w), P, mu

    def test_round_trip_kramers(self):
        ll, natm = 2, 2
        H = synthetic_tb(ll, natm, 4, 3)
        # a few k points: H(k) = H + k-dependent hopping between shells and extra orbitals
        rng = np.random.default_rng(4)
        T = rng.normal(size=(len(H)//2, len(H)//2)) * 0.05
        T = np.kron(T + T.T, np.eye(2))                               # real, spin diagonal: time-reversal even
        ks = [0.0, 0.3, 0.6]
        eigs, vecs = [], []
        for k in ks:
            Hk = H + np.cos(2*np.pi*k)*T
            # keep time reversal: T is real and spin-diagonal in the interleaved real basis
            e, v = np.linalg.eigh(Hk)
            eigs.append(e); vecs.append(v)
        eigs, vecs = np.array(eigs), np.array(vecs)
        atoms = [dict(indices=list(range(a*10, a*10+10)), frame=None) for a in range(natm)]
        data = udmft.build_export(eigs, vecs, H, atoms, ll, "kramers", 20.0, 0.1, tol=dict(check_tol=1.0),
                                  log=lambda *_: None)
        with tempfile.TemporaryDirectory() as d:
            m = udmft.write_export(data, d, dict(hr="0"*64))
            eps, w, P, mu = self.read_files(d, 10, natm)
            self.assertLessEqual(abs(w.sum() - 1.0), 1e-12)
            E1 = sum(w[k]*P[k][:, :10].conj().T @ np.diag(eps[k] - mu) @ P[k][:, :10] for k in range(len(w))) \
                + mu*np.eye(10)
            EI = np.array([[complex(*z) for z in row] for row in m["E_mesh_I"][0]])
            self.assertLessEqual(np.abs(E1 - EI).max(), 1e-10)
            self.assertLessEqual(np.abs(np.real(np.diag(EI)) - m["eimp_abs"]).max(), 1e-14)
            trans = (Path(d)/"Trans.dat").read_text().splitlines()
            self.assertIn("flavor_order=interleaved", trans[1])
            cf = np.array([[float(x) for x in ln.split()] for ln in trans[13:23]])
            cf = cf[:, 0::2] + 1j*cf[:, 1::2]
            T2C = np.array([[complex(*z) for z in row] for row in m["T2C"]])
            self.assertLessEqual(np.abs(cf - T2C.T).max(), 1e-14)
            c2r = np.array([[complex(*z) for z in row] for row in m["c2r"]])
            von = np.array([[complex(*z) for z in row] for row in m["von"]])
            self.assertLessEqual(np.abs(c2r[m["perm_rows"], :] @ von - T2C).max(), 1e-14)
            self.assertEqual(m["basis_id"], udmft.basis_id(m))
            self.assertEqual(len(m["basis_id"]), 16)
            self.assertEqual(m["pairs"][1], [2, 3])

    def test_old_weights_are_not_one(self):
        self.assertGreater(abs(1728*float(f"{1/1728:15.7f}") - 1), 1e-6)   # why 17 digits are needed


if __name__ == "__main__":
    unittest.main()
