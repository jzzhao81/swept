import tempfile
import unittest
from pathlib import Path

import numpy as np

import mpiutils as mpi
from bands import dump_akws
from model import Model
from spinor import euler_to_rmat, rmat_to_euler


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
