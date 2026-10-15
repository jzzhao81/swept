import tempfile
import unittest
from pathlib import Path

import numpy as np

import mpiutils as mpi
from bands import dump_akws
from model import Model
from spinor import euler_to_rmat, rmat_to_euler, rotate_spinor
from udmft import dump_dmft, local_eigenbasis, spin_block_perm


class ModelGeometryTests(unittest.TestCase):
    def test_slab_can_be_finite_along_first_lattice_vector(self):
        model = Model(
            nwan=1,
            nrpt=3,
            deg=[1, 1, 1],
            ham={
                (0, 0, 0): np.array([[1.0]]),
                (1, 0, 0): np.array([[2.0]]),
                (-1, 0, 0): np.array([[2.0]]),
            },
        )

        actual = model.get_slab_Hk([0.0, 0.0, 0.0], nlay=3, direction=0)
        expected = np.array(
            [[1.0, 2.0, 0.0], [2.0, 1.0, 2.0], [0.0, 2.0, 1.0]],
            dtype=complex,
        )
        np.testing.assert_allclose(actual, expected)

    def test_surface_projection_order_is_backward_compatible(self):
        onsite = np.diag([0.0, 1.0])
        forward = np.array([[0.2, 0.3 + 0.1j], [-0.2j, 0.1]])
        model = Model(
            nwan=2,
            nrpt=3,
            deg=[1, 1, 1],
            ham={
                (0, 0, 0): onsite,
                (0, 0, 1): forward,
                (0, 0, -1): forward.conj().T,
            },
        )
        kpoint = np.zeros((1, 3))
        ham = model.get_slab_Hk(kpoint[0], nlay=2)
        expected_eigs, vectors = np.linalg.eigh(ham)
        expected_top = np.sum(np.abs(vectors[:2, :]) ** 2, axis=0)
        expected_bottom = np.sum(np.abs(vectors[-2:, :]) ** 2, axis=0)

        eigs, ptop, pbot = model.get_slab_eprojs(kpoint, nlay=2)

        np.testing.assert_allclose(eigs[:, 0], expected_eigs)
        np.testing.assert_allclose(ptop[:, 0], expected_top)
        np.testing.assert_allclose(pbot[:, 0], expected_bottom)

    def test_wire_hamiltonian_is_hermitian(self):
        forward_y = np.array([[0.3 + 0.2j]])
        forward_z = np.array([[-0.1 + 0.4j]])
        model = Model(
            nwan=1,
            nrpt=5,
            deg=[1] * 5,
            ham={
                (0, 0, 0): np.array([[0.5]]),
                (0, 1, 0): forward_y,
                (0, -1, 0): forward_y.conj().T,
                (0, 0, 1): forward_z,
                (0, 0, -1): forward_z.conj().T,
            },
        )

        ham = model.get_wire_Hk([0.25, 0.0, 0.0], nl1=2, nl2=3, axis=0)

        self.assertEqual(ham.shape, (6, 6))
        np.testing.assert_allclose(ham, ham.conj().T)


class SpectrumTests(unittest.TestCase):
    def test_projected_kz_is_an_average(self):
        sigma = 0.2
        model = Model(
            nwan=1,
            nrpt=1,
            deg=[1],
            ham={(0, 0, 0): np.array([[0.0]])},
        )

        _, akw_one = model.get_bulk_projected_kz(
            [[0.0, 0.0, 0.0]], omin=0.0, omax=0.0,
            nomg=1, nkz=1, sigma=sigma,
        )
        _, akw_five = model.get_bulk_projected_kz(
            [[0.0, 0.0, 0.0]], omin=0.0, omax=0.0,
            nomg=1, nkz=5, sigma=sigma,
        )
        expected = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

        np.testing.assert_allclose(akw_one, [[expected]])
        np.testing.assert_allclose(akw_five, akw_one)

    def test_dump_akws_column_order(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "akws.dat"
            dump_akws(
                np.array([0.0, 0.5]),
                np.array([-1.0, 1.0]),
                np.array([[2.0, 3.0], [4.0, 5.0]]),
                fname=output,
            )
            rows = np.loadtxt(output)

        np.testing.assert_allclose(
            rows,
            [[-1.0, 0.0, 2.0], [-1.0, 0.5, 3.0],
             [1.0, 0.0, 4.0], [1.0, 0.5, 5.0]],
        )


def _random_hermitian(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return a + a.conj().T


def _random_model(nwan, seed=0):
    rng = np.random.default_rng(seed)
    hop = rng.normal(size=(nwan, nwan)) + 1j * rng.normal(size=(nwan, nwan))
    return Model(
        nwan=nwan,
        nrpt=3,
        deg=[1, 1, 1],
        ham={
            (0, 0, 0): _random_hermitian(nwan, seed + 1),
            (0, 0, 1): hop,
            (0, 0, -1): hop.conj().T,
        },
    )


class RotateSpinTests(unittest.TestCase):
    zx = np.array([
        [[0.0, 0.0, 1.0], [0.8660254038, 0.5, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
    ])

    def test_default_index_matches_contiguous_blocks(self):
        lnum = 1
        norb = 2 * (2 * lnum + 1)
        model = _random_model(2 * norb)
        h0 = model.ham[(0, 0, 0)].copy()

        smat = np.zeros((2 * norb, 2 * norb), dtype=complex)
        for iatm in range(2):
            smat[iatm * norb:(iatm + 1) * norb, iatm * norb:(iatm + 1) * norb] = \
                np.kron(np.eye(2 * lnum + 1), rotate_spinor(*self.zx[iatm]))

        model.rotate_spin(lnum=lnum, zxmat=self.zx)

        np.testing.assert_allclose(model.ham[(0, 0, 0)], smat.conj().T @ h0 @ smat, atol=1e-12)

    def test_mixed_lnum_with_index_is_unitary(self):
        # d block at 2, p block at 14, orbitals 0-1 and 12-13 untouched
        nwan = 20
        model = _random_model(nwan, seed=3)
        kpoint = np.array([[0.1, 0.2, 0.3]])
        eigs_before, _ = model.get_bulk_eigens(kpoint)
        h0 = model.ham[(0, 0, 0)].copy()

        model.rotate_spin(lnum=[2, 1], zxmat=self.zx, index=[2, 14])
        eigs_after, _ = model.get_bulk_eigens(kpoint)

        np.testing.assert_allclose(eigs_after, eigs_before, atol=1e-10)
        np.testing.assert_allclose(model.ham[(0, 0, 0)], model.ham[(0, 0, 0)].conj().T, atol=1e-12)
        for idx in (0, 1, 12, 13):
            self.assertAlmostEqual(model.ham[(0, 0, 0)][idx, idx], h0[idx, idx])

    def test_overlapping_or_out_of_range_blocks_raise(self):
        model = _random_model(16)
        with self.assertRaises(ValueError):
            model.rotate_spin(lnum=[2, 1], zxmat=self.zx, index=[0, 8])
        with self.assertRaises(ValueError):
            model.rotate_spin(lnum=[2, 1], zxmat=self.zx, index=[0, 12])


class UdmftTests(unittest.TestCase):
    def test_spin_block_perm(self):
        np.testing.assert_array_equal(spin_block_perm(3), [0, 2, 4, 1, 3, 5])

    def test_local_eigenbasis_diagonalises_h0(self):
        h0 = _random_hermitian(10, seed=7)
        eon, von, htmat, mz = local_eigenbasis(h0, ll=2)

        np.testing.assert_allclose(von.conj().T @ von, np.eye(10), atol=1e-12)
        np.testing.assert_allclose(von.conj().T @ h0 @ von, np.diag(eon), atol=1e-10)
        np.testing.assert_allclose(htmat @ htmat.conj().T, np.eye(10), atol=1e-12)
        np.testing.assert_allclose(mz, mz.conj().T, atol=1e-12)
        # trace of Lz + 2Sz over a full shell vanishes
        self.assertAlmostEqual(np.trace(mz).real, 0.0)

    def test_local_eigenbasis_mz_of_pure_spin_states(self):
        # Non-degenerate diagonal h0: eigenstates are single real harmonics with
        # <Lz> = 0, so Mz = 2Sz = +1 (up, even index) or -1 (down, odd index).
        h0 = np.diag(np.arange(10, dtype=float))
        _, von, _, mz = local_eigenbasis(h0, ll=2)
        spin = np.where(np.argmax(np.abs(von), axis=0) % 2 == 0, 1.0, -1.0)

        np.testing.assert_allclose(np.diag(mz).real, spin[spin_block_perm(5)], atol=1e-12)

    def test_dump_dmft_projection(self):
        nkpt, nwan, norb, natm = 2, 6, 2, 2
        rng = np.random.default_rng(11)
        vec = np.stack([np.linalg.qr(rng.normal(size=(nwan, nwan)) + 1j * rng.normal(size=(nwan, nwan)))[0]
                        for _ in range(nkpt)])
        eig = rng.normal(size=(nkpt, nwan))
        von = np.linalg.qr(rng.normal(size=(norb, norb)) + 1j * rng.normal(size=(norb, norb)))[0]

        def read_proj(directory):
            rows = [line.split() for line in (Path(directory) / "udmft.dat").read_text().splitlines()[3:]
                    if line.strip() and "#" not in line]
            data = np.array(rows, dtype=float)
            return (data[:, 2] + 1j * data[:, 3]).reshape(nkpt, norb * natm, nwan)

        for v in (None, von):
            with tempfile.TemporaryDirectory() as directory:
                dump_dmft(eig, vec, 0.0, ntot=3.0, norb=norb, natm=natm, lnum=0, von=v, outdir=directory)
                proj = read_proj(directory)
            tmat = np.eye(norb * natm) if v is None else np.kron(np.eye(natm), v)
            for ik in range(nkpt):
                expected = vec[ik, :norb * natm, :].conj().T @ tmat
                np.testing.assert_allclose(proj[ik], expected.T, atol=1e-11)


class UtilityTests(unittest.TestCase):
    def test_allreduce(self):
        values = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(mpi.Allreduce(values), values * mpi.size)

    def test_euler_gimbal_lock_round_trip(self):
        for beta in (0.0, np.pi):
            for alpha in (0.3, 4.7):
                rotation = euler_to_rmat(alpha, beta, 0.0)
                reconstructed = euler_to_rmat(*rmat_to_euler(rotation))
                np.testing.assert_allclose(reconstructed, rotation, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
