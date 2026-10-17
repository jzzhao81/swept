'''
Author: Jianzhou Zhao
Date: 2025-05-24 17:09:48
LastEditors: your name
LastEditTime: 2026-10-15
Description: dump Wannier eigenvalues / projectors for the ksum + DMFT interface
'''

import os
import numpy as np
import scipy.linalg as la
from .basis import complex2real
from .kpoint import Kpoint


def spin_block_perm(nbnd):
    """Permutation from spin-interleaved (m1u, m1d, m2u, ...) to spin-blocked
    (m1u, m2u, ..., m1d, m2d, ...) ordering of 2*nbnd spin-orbitals."""
    return np.r_[0:2*nbnd:2, 1:2*nbnd:2]


def time_reversal_matrix(nbnd):
    """Unitary part of T = i sigma_y K in the spin-interleaved real-harmonic
    basis (real orbitals are T-even): T|m up> = -|m dn>, T|m dn> = |m up>."""
    return np.kron(np.eye(nbnd), np.array([[0.0, 1.0], [-1.0, 0.0]]))


def time_reversal_error(h0):
    """max |T h0 T^-1 - h0|; zero for a nonmagnetic on-site block."""
    h0 = np.asarray(h0)
    theta = time_reversal_matrix(h0.shape[0]//2)
    return np.max(np.abs(theta @ h0.conj() @ theta.T - h0))


def kramers_eigenbasis(h0, mz_r, tr_tol=1e-3):
    """Eigenbasis of a T-invariant h0 with every Kramers pair in a fixed gauge.

    eigh returns an arbitrary U(2) mixture inside each degenerate pair.  Here
    the first state of a pair maximises <Mz> and the second is its exact time
    reversed partner, so solver flavours (1+, 2+, ..., 1-, 2-, ...) are
    reproducible and Mz is diagonal in every pair.  h0 is symmetrised,
    (h0 + T h0 T^-1)/2, after checking that its T-odd part is below tr_tol (eV):
    a magnetic h0 must not be forced into pairs."""
    h0 = np.asarray(h0, dtype=complex)
    n = h0.shape[0]
    theta = time_reversal_matrix(n//2)
    h0_tr = theta @ h0.conj() @ theta.T
    err = np.max(np.abs(h0_tr - h0))
    if err > tr_tol:
        raise ValueError(f"on-site block breaks time reversal by {err:.3e} eV > {tr_tol:.1e}; "
                         "Kramers gauge only applies to a nonmagnetic H_loc")
    h0 = 0.5*(h0 + h0_tr)
    h0 = 0.5*(h0 + h0.conj().T)

    eon, raw = la.eigh(h0)
    von = np.empty_like(raw)
    for i in range(0, n, 2):
        if eon[i+1] - eon[i] > 1e-8:
            raise ValueError(f"states {i}, {i+1} are not a Kramers pair (splitting {eon[i+1] - eon[i]:.3e} eV)")
        pair = raw[:, i:i+2]
        mval, mvec = la.eigh(pair.conj().T @ mz_r @ pair)
        if np.max(np.abs(mval)) < 1e-10:
            raise ValueError(f"Mz vanishes in Kramers pair {i//2}; cannot fix its gauge")
        plus = pair @ mvec[:, np.argmax(mval)]
        # phase reference: first component within 1e-6 of the largest
        # modulus; a bare argmax flips between symmetry-equal components
        # under roundoff
        amp = np.abs(plus)
        ipiv = np.flatnonzero(amp > amp.max() - 1e-6)[0]
        plus *= np.exp(-1j*np.angle(plus[ipiv]))
        minus_exact = theta @ plus.conj()
        minus = pair @ (pair.conj().T @ minus_exact)
        if la.norm(minus_exact - minus) > 1e-6:
            raise ValueError(f"Kramers pair {i//2} is not closed under T "
                             "(degeneracy beyond Kramers, e.g. a cubic quartet)")
        minus -= plus * np.vdot(plus, minus)
        von[:, i], von[:, i+1] = plus, minus / la.norm(minus)
    return eon, von


def local_eigenbasis(h0, ll=2, kramers=False, tr_tol=1e-3):
    """Diagonalise the on-site Hamiltonian of one correlated shell.

    h0 is the (2*(2l+1))^2 block in the spin-interleaved real-harmonic basis.
    With SOC the spin off-diagonal block is finite, so the full matrix must be
    diagonalised; splitting it into up/up and dn/dn blocks leaves sizeable
    off-diagonal terms in Eimp which ksum silently drops.

    kramers=True (nonmagnetic h0 only): fix the gauge of every Kramers pair
    with kramers_eigenbasis; otherwise a plain eigh.

    Returns
    -------
    eon   : eigenvalues of h0 (ascending)
    von   : eigenvectors of h0, columns in interleaved flavour order
    htmat : T2C[a,i] = <complex_a | Eimp_i> for atom_d (which reads CF = T2C.T),
            i.e. T2C = c2r @ von with c2r[a,b] = <complex_a | real_b> and
            von[b,i] = <real_b | Eimp_i>; no complex conjugation of von.
            Both axes are permuted to spin-blocked order:
              * complex-harmonic axis : interleaved -> spin blocked (atom_d)
              * flavour axis          : ksum stores Eimp interleaved and writes
                Ed as emat(k)=eimp(2k-1), emat(k+nbnd)=eimp(2k), so solver
                flavour j corresponds to von column perm[j].
    mz    : the matrix of Lz + 2 Sz (hbar = 1) in the Eimp basis, solver
            (spin-blocked) order; z is the shell's local z axis when the shell
            has z/x.  Under Q3d=False ctqmc's chiS0 / <Sz> are meaningless.
            In general m_z = Tr(Mz rho) = sum_ab Mz_ab rho_ba, with
            rho_ab = <c_b^+ c_a> in the same basis; the physical moment is
            <mu_z> = -mu_B Tr(Mz rho) (operator form mu = -(mu_B/hbar)(L + 2S)
            for L, S in units of hbar).  Only when the off-diagonal
            contraction vanishes -- Mz or rho diagonal in this basis is
            sufficient -- does it reduce to sum_f Mz_ff <n_f>.  Using flavour
            occupations alone is an approximation (diagonal density matrix)
            that does not follow from Q3d or from a diagonal hybridization:
            interactions can make rho off-diagonal; which components are
            available is set by Sigind, cix and the solver's measurements.
            kramers=True makes Mz diagonal inside each Kramers pair (solver
            flavours i and i + 2l+1), not between pairs.  A global direction
            needs the full moment vector rotated from the local frame; only
            Mz is provided here.
    """
    nbnd = 2*ll + 1
    h0 = np.asarray(h0)
    if h0.shape != (2*nbnd, 2*nbnd):
        raise ValueError(f"h0 must be {2*nbnd}x{2*nbnd} for l={ll}, got {h0.shape}")

    perm = spin_block_perm(nbnd)
    c2r = complex2real(ll, 2)
    lz = np.kron(np.diag(np.arange(-ll, ll+1, dtype=float)), np.eye(2))
    sz = np.kron(np.eye(nbnd), np.diag([0.5, -0.5]))
    mz_r = c2r.conj().T @ lz @ c2r + 2.0*sz

    if kramers:
        eon, von = kramers_eigenbasis(h0, mz_r, tr_tol=tr_tol)
    else:
        eon, von = la.eigh(h0)
    htmat = (c2r[perm, :] @ von[:, perm]).T
    mz = von[:, perm].conj().T @ mz_r @ von[:, perm]

    return eon, von, htmat, mz


def onsite_block_mismatch(h00, norb, natm, start=0):
    """Max |H_loc(atom i) - H_loc(atom 0)| over the natm correlated blocks.

    dump_dmft applies the same von to every block, which is only valid when
    the sites are equivalent in their local frames (mismatch ~ 0)."""
    blocks = [h00[start+i*norb:start+(i+1)*norb, start+i*norb:start+(i+1)*norb]
              for i in range(natm)]
    return max(np.max(np.abs(b - blocks[0])) for b in blocks)


def dump_local_basis(eon, von, htmat, outdir='.'):
    """Write eon.dat / von.dat / htmat.dat for the local eigenbasis."""
    norb = len(eon)

    with open(os.path.join(outdir, "eon.dat"), "w") as feon:
        output_format = "{:5d}{:5d}{:20.10f}{:20.10f}\n"
        for ibnd in range(norb):
            feon.write(output_format.format(ibnd+1, ibnd+1, eon[ibnd], eon[ibnd]-eon[0]))

    with open(os.path.join(outdir, "von.dat"), "w") as fvon:
        output_format = "{:5d}{:5d}{:20.10f}{:20.10f}\n"
        for ibnd in range(norb):
            for iorb in range(norb):
                fvon.write(output_format.format(ibnd+1, iorb+1, von[ibnd, iorb].real, von[ibnd, iorb].imag))
            fvon.write('\n')
            fvon.write('\n')

    with open(os.path.join(outdir, "htmat.dat"), "w") as fhtmat:
        output_format = "{:12.8f} {:12.8f}  "*norb + "\n"
        for ibnd in range(norb):
            fhtmat.write(output_format.format(
                *(v for iorb in range(norb)
                  for v in (htmat[ibnd, iorb].real, htmat[ibnd, iorb].imag))))


def dump_dmft(eig, vec, mu, ntot, norb=10, natm=3, lnum=2, nspin=2, von=None, start=0, outdir='.'):
    """Write eig.dat and udmft.dat for ksum.

    Wannier orbitals start:start+norb*natm are the correlated shells.  If von is
    given, the projectors are rotated to the local eigenbasis of every shell,
    vmat = <psi_nk | w> von  (same von for all natm shells)."""

    nkpt = vec.shape[0]
    nbnd = vec.shape[2]

    tmat = np.eye(norb*natm, dtype=complex)
    if von is not None:
        if von.shape != (norb, norb):
            raise ValueError(f"von must be {norb}x{norb}, got {von.shape}")
        tmat = np.kron(np.eye(natm), von)

    head_format = "{:5d}{:5d}{:5d}{:5d}{:5d}{:50.35s}\n"
    info_format = "{:5d}{:5d}{:5d}{:10.5f}{:10.5f}{:50.40s}\n"
    kw_format = "{:5d}{:15.7f}{:31.20s}\n"

    # Print eigen values
    with open(os.path.join(outdir, "eig.dat"), "w") as feig:
        feig.write(head_format.format(nkpt, natm, norb, lnum, nspin, "     # nkpt, natm, norb, L, nspin"))
        feig.write(info_format.format(1, nbnd, nbnd, ntot, mu, "     # nemin, nemax, nbands, ntotal, mu"))
        feig.write('\n')
        output_format = "{:5d}{:20.12f}\n"
        for ikpt in range(nkpt):
            feig.write(kw_format.format(ikpt+1, 1.0/float(nkpt), " # ikpt, k-weight"))
            for ibnd in range(nbnd):
                feig.write(output_format.format(ibnd+1, eig[ikpt, ibnd]))
            feig.write('\n')
            feig.write('\n')

    with open(os.path.join(outdir, "udmft.dat"), "w") as fprj:
        fprj.write(head_format.format(nkpt, natm, norb, lnum, nspin, "     # nkpt, natm, norb, L, nspin"))
        fprj.write(info_format.format(1, nbnd, nbnd, ntot, mu, "     # nemin, nemax, nbands, ntotal, mu"))
        fprj.write('\n')
        output_format = "{:5d}{:5d}{:20.12f}{:20.12f}\n"
        for ikpt in range(nkpt):
            fprj.write(kw_format.format(ikpt+1, 1.0/float(nkpt), " # ikpt, k-weight"))
            vmat = np.dot(vec[ikpt, start:start+norb*natm, :].conj().T, tmat)
            for iorb in range(norb*natm):
                for ibnd in range(nbnd):
                    fprj.write(output_format.format(ibnd+1, iorb+1, vmat[ibnd, iorb].real, vmat[ibnd, iorb].imag))
            fprj.write('\n')
            fprj.write('\n')


def dump_kpoints(kpts, fname='kpoints.dat'):
    """
    Dump k-points to a file.
    """
    kreal = Kpoint.k_length_accumulate(kpts.kreal)
    with open(fname, "w") as f:
        for ikpt, kpt in enumerate(kpts.kfrac):
            f.write("{:5d} {:15.9f} {:15.9f} {:15.9f} {:15.9f}\n".format(ikpt + 1, kpt[0], kpt[1], kpt[2], kreal[ikpt]))
