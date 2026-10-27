'''
Author: Jianzhou Zhao
Date: 2025-05-24 17:09:48
LastEditors: your name
LastEditTime: 2026-10-26
Description: dump Wannier eigenvalues / projectors for the ksum + DMFT interface

Impurity flavors are spin interleaved everywhere outside atom_d: flavor 2k and
2k+1 are the two members of pair k (orbital k up/down, or a Kramers pair).
Only the complex-harmonic rows of T2C (= columns of CF in Trans.dat) stay spin
blocked, because that is atom_d's internal basis.  See
UTe2/docs/PROPOSAL_interleaved_flavor_v5_2026-10-25.md.
'''

import datetime
import hashlib
import json
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
    basis (real orbitals are T-even): T|m up> = -|m dn>, T|m dn> = |m up>.
    T acts as  psi -> U psi.conj()."""
    return np.kron(np.eye(nbnd), np.array([[0.0, 1.0], [-1.0, 0.0]]))


def time_reversal_error(h0):
    """max |T h0 T^-1 - h0|; zero for a nonmagnetic on-site block."""
    h0 = np.asarray(h0)
    theta = time_reversal_matrix(h0.shape[0]//2)
    return np.max(np.abs(theta @ h0.conj() @ theta.T - h0))


def moment_operators(ll):
    """M_a = L_a + 2 S_a (hbar = 1, dimensionless) in the spin-interleaved
    real-harmonic basis, a = x, y, z.  L is built from Condon-Shortley ladder
    operators in the complex harmonics m = -l..l and rotated with complex2real."""
    m = np.arange(-ll, ll+1, dtype=float)
    lp = np.diag(np.sqrt(ll*(ll+1) - m[:-1]*(m[:-1]+1)), -1)      # <m+1|L+|m>
    lx, ly, lz = 0.5*(lp + lp.T), -0.5j*(lp - lp.T), np.diag(m)
    sx = 0.5*np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5*np.array([[0, -1j], [1j, 0]])
    sz = 0.5*np.diag([1.0, -1.0])
    c2r = complex2real(ll, 2)
    out = {}
    for name, L, S in (("Mz", lz, sz), ("Mx", lx, sx), ("My", ly, sy)):
        M = np.kron(L, np.eye(2)) + 2*np.kron(np.eye(2*ll+1), S)
        out[name] = c2r.conj().T @ M @ c2r
    return out


# --------------------------------------------------------------------------
# independent reference of the real harmonics (PROPOSAL v5, TP1)
# --------------------------------------------------------------------------

def _real_harmonic_polynomials(ll, x, y, z):
    """Wannier90 real harmonics (user guide, projections), unnormalised, r = 1."""
    if ll == 0:
        return [np.ones_like(z)]
    if ll == 1:
        return [z, x, y]
    if ll == 2:
        return [3*z*z - 1, x*z, y*z, x*x - y*y, x*y]
    if ll == 3:
        return [z*(5*z*z - 3), x*(5*z*z - 1), y*(5*z*z - 1), z*(x*x - y*y), x*y*z,
                x*(x*x - 3*y*y), y*(3*x*x - y*y)]
    raise ValueError(f"no real harmonics tabulated for l={ll}")


def real_harmonic_reference(ll, nth=32, nph=64):
    """C_ref[r, m] = <Y_lm | R_r> by quadrature (Gauss-Legendre in cos(theta) x uniform phi),
    Y_lm from scipy (Condon-Shortley phase), R_r the normalised Wannier90 polynomials.

    Independent of complex2real: swept's complex2real(l, 1) must equal C_ref.T and
    atom_d's Complex2RealHarmonics(l, 1) must equal C_ref."""
    from scipy import special
    ct, wt = np.polynomial.legendre.leggauss(nth)
    ph = 2*np.pi*np.arange(nph)/nph
    th = np.arccos(ct)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    W = np.outer(wt, np.full(nph, 2*np.pi/nph))
    x, y, z = np.sin(TH)*np.cos(PH), np.sin(TH)*np.sin(PH), np.cos(TH)
    R = _real_harmonic_polynomials(ll, x, y, z)
    if hasattr(special, "sph_harm_y"):
        Y = [special.sph_harm_y(ll, m, TH, PH) for m in range(-ll, ll+1)]
    else:                                                        # scipy < 1.15
        Y = [special.sph_harm(m, ll, PH, TH) for m in range(-ll, ll+1)]
    C = np.zeros((2*ll+1, 2*ll+1), dtype=complex)
    for r, Rr in enumerate(R):
        Rr = Rr/np.sqrt(np.sum(W*Rr*Rr))
        for im, Ym in enumerate(Y):
            C[r, im] = np.sum(W*Ym.conj()*Rr)
    return C


def check_complex2real(ll):
    """max |complex2real(l,1) - C_ref.T| (TP1); the two must agree element by element."""
    return float(np.abs(complex2real(ll, 1) - real_harmonic_reference(ll).T).max())


# --------------------------------------------------------------------------
# local bases
# --------------------------------------------------------------------------

def _span_clusters(e, tol):
    """Groups of consecutive sorted eigenvalues whose total span is <= tol (no chaining)."""
    groups, start = [], 0
    for i in range(1, len(e)+1):
        if i == len(e) or e[i] - e[start] > tol:
            groups.append(list(range(start, i)))
            start = i
    return groups


def _theta(U, X):
    return U @ np.asarray(X).conj()


def _theta_leak(W, U):
    """|| (1 - W W^+) Theta W ||_F: zero when span(W) is closed under time reversal."""
    if W.shape[1] == 0:
        return 0.0
    TW = _theta(U, W)
    return float(np.linalg.norm(TW - W @ (W.conj().T @ TW)))


def _orth_complement(R, vecs):
    """Orthonormal basis of span(R) minus span(vecs) (vecs orthonormal, inside span(R))."""
    X = R - sum(np.outer(v, v.conj() @ R) for v in vecs)
    u, _, _ = np.linalg.svd(X, full_matrices=False)
    return u[:, :R.shape[1]-len(vecs)]


def _fix_phase(v):
    amp = np.abs(v)
    ipiv = np.flatnonzero(amp > amp.max() - 1e-6)[0]
    return v*np.exp(-1j*np.angle(v[ipiv]))


def _pick_vector(S, ops, m_tol):
    """Narrows S with the operators in turn (largest eigenvalue, only inside the part not yet
    resolved), then falls back to the projection of the standard basis.  Returns (v, rule)."""
    rule = None
    for name, A in ops:
        if S.shape[1] == 1:
            break
        a, Q = la.eigh(S.conj().T @ A @ S)
        if a[-1] > m_tol:
            S = S @ Q[:, a >= a[-1] - m_tol]
            rule = name
    if S.shape[1] == 1:
        return S[:, 0], rule or ops[0][0]
    for i in range(S.shape[0]):
        p = S @ S[i, :].conj()                                   # S S^+ e_i
        nrm = np.linalg.norm(p)
        if nrm >= 1e-6:
            return p/nrm, "basis"
    raise ValueError("cannot pick a vector from a degenerate subspace")


def _kramers_cluster(W, U, ops, m_tol):
    R = W.copy()
    vecs, rules = [], []
    while R.shape[1] > 0:
        v, rule = _pick_vector(R, ops, m_tol)
        v = _fix_phase(v/np.linalg.norm(v))
        u = _theta(U, v)
        if np.linalg.norm(u - R @ (R.conj().T @ u)) > 1e-8:
            raise ValueError("Kramers partner leaves the cluster (not closed under time reversal)")
        if abs(np.vdot(v, u)) > 1e-10:
            raise ValueError("Theta v is not orthogonal to v")
        vecs += [v, u]
        rules.append(rule)
        R = _orth_complement(R, [v, u])
    return vecs, rules


def kramers_basis(E, U, ops, deg_tol=1e-8, m_tol=1e-8, res_tol=1e-10):
    """Eigenbasis of a time-reversal symmetric E (spin-interleaved basis) with every
    Kramers pair (v, Theta v) on neighbouring columns (PROPOSAL v5 section 3.3).

    ops: [(name, M), ...] T-odd Hermitian operators used in turn to fix the gauge
    (Mz first).  Clusters of eigenvalues with span <= deg_tol are treated as one
    degenerate space; a cluster whose rotated H keeps off-diagonal elements above
    res_tol is split at the largest gap that leaves both parts even-dimensional and
    closed under Theta.  Returns (von, rules) with one rule per pair."""
    E = 0.5*(E + E.conj().T)
    e, V = la.eigh(E)
    groups = _span_clusters(e, deg_tol)
    # a group boundary must not cut a Kramers pair: merge with the next group until closed
    merged, cur = [], []
    for g in groups:
        cur = cur + g
        if len(cur) % 2 == 0 and _theta_leak(V[:, cur], U) <= 1e-8:
            merged.append(cur)
            cur = []
    if cur:
        raise ValueError(f"eigenvalues {e[cur[0]]:.6f}..{e[cur[-1]]:.6f}: no time-reversal closed "
                         f"grouping (odd dimension or Theta leak {_theta_leak(V[:, cur], U):.2e})")

    def build(idx):
        vecs, rules = _kramers_cluster(V[:, idx], U, ops, m_tol)
        Vc = np.column_stack(vecs)
        Hc = Vc.conj().T @ E @ Vc
        res = float(np.abs(Hc - np.diag(np.diag(Hc))).max()) if len(idx) > 2 else 0.0
        if res <= res_tol:
            return vecs, rules
        cands = []
        for b in range(2, len(idx), 2):
            if _theta_leak(V[:, idx[:b]], U) <= 1e-8 and _theta_leak(V[:, idx[b:]], U) <= 1e-8:
                cands.append((e[idx[b]] - e[idx[b-1]], b))
        if not cands:
            raise ValueError(f"Kramers cluster {e[idx[0]]:.6f}..{e[idx[-1]]:.6f} eV (span "
                             f"{e[idx[-1]]-e[idx[0]]:.2e}): residual {res:.2e} > res_tol and no "
                             f"even, time-reversal closed split")
        b = max(cands)[1]
        v1, r1 = build(idx[:b])
        v2, r2 = build(idx[b:])
        return v1 + v2, r1 + r2

    vecs, rules = [], []
    for g in merged:
        v, r = build(g)
        vecs += v
        rules += r
    # Pairwise Gram-Schmidt in the final order: v_k is orthogonalised against the
    # previous pairs (a time-reversal closed set), then u_k = Theta v_k is exactly
    # orthogonal to them as well and to v_k (Theta^2 = -1).  Removes the ~1e-13
    # drift of the cluster construction without touching the pairing.
    fixed = []
    for k in range(0, len(vecs), 2):
        v = vecs[k]
        for _ in range(2):
            for x in fixed:
                v = v - x*np.vdot(x, v)
        v = v/np.linalg.norm(v)
        fixed += [v, _theta(U, v)]
    von = np.column_stack(fixed)
    err_u = np.abs(von.conj().T @ von - np.eye(len(E))).max()
    err_t = np.abs(von[:, 1::2] - _theta(U, von[:, 0::2])).max()
    Hd = von.conj().T @ E @ von
    err_h = np.abs(Hd - np.diag(np.diag(Hd))).max()
    if err_u > 1e-12 or err_t > 1e-10 or err_h > res_tol:
        raise ValueError(f"Kramers basis checks failed: unitarity {err_u:.2e}, "
                         f"pairing {err_t:.2e}, off-diagonal H {err_h:.2e}")
    return von, rules


def orbital_spin_basis(E, deg_tol=1e-8, spin_mix_tol=1e-6, offdiag_tol=1e-6):
    """von = kron(V_orb, I_2) diagonalising both spin blocks of E (PROPOSAL v5 section 3.4).
    Raises ValueError when E mixes the spins or the two spins need different orbitals."""
    hu, hd, hud = E[0::2, 0::2], E[1::2, 1::2], E[0::2, 1::2]
    mix = float(np.abs(hud).max())
    if mix > spin_mix_tol:
        raise ValueError(f"orbital_spin: spin-mixing block {mix:.2e} eV > {spin_mix_tol:.1e}; "
                         f"use flavor_kind = 'kramers'")
    hbar, delta = 0.5*(hu + hd), 0.5*(hu - hd)
    hbar, delta = 0.5*(hbar + hbar.conj().T), 0.5*(delta + delta.conj().T)
    e, Q = la.eigh(hbar)
    cols = []
    for g in _span_clusters(e, deg_tol):
        Qg = Q[:, g]
        _, R = la.eigh(Qg.conj().T @ delta @ Qg)
        cols.append(Qg @ R)
    V = np.column_stack(cols)
    res = max(float(np.abs(X - np.diag(np.diag(X))).max())
              for X in (V.conj().T @ hu @ V, V.conj().T @ hd @ V))
    if res > offdiag_tol:
        comm = float(np.linalg.norm(hu @ hd - hd @ hu))
        raise ValueError(f"orbital_spin: no common orbital basis for both spins (off-diagonal "
                         f"{res:.2e} eV > {offdiag_tol:.1e}; ||[h_up, h_dn]||_F = {comm:.2e})")
    return np.kron(V, np.eye(2))


def jj_basis(ll):
    """|j, m_j> of an l shell in the spin-interleaved real-harmonic basis, Kramers pairs
    (v, Theta v) on neighbouring columns.

    Pair order: j = l - 1/2 then l + 1/2, |m_j| descending inside each j (for l = 3:
    (5/2,5/2), (5/2,3/2), (5/2,1/2), (7/2,7/2), ..., (7/2,1/2), the eDMFT component order).
    v = |j, +|m_j|> from Condon-Shortley Clebsch-Gordan coefficients in the complex harmonics,
    u = Theta v (proportional to |j, -|m_j|>).  The basis is fixed by symmetry and does not
    diagonalise H_loc.  Returns (von, labels) with one label per pair."""
    if ll < 1:
        raise ValueError("jj basis needs l >= 1")
    n = 2*ll + 1
    c2r = complex2real(ll, 2)                     # rows: complex (m, spin) interleaved; columns: real
    U = time_reversal_matrix(n)
    cols, labels = [], []
    for j2 in (2*ll - 1, 2*ll + 1):               # 2j
        for mj2 in range(j2, 0, -2):              # 2 m_j > 0
            a = np.zeros(2*n, dtype=complex)
            mu, md = (mj2 - 1)//2, (mj2 + 1)//2   # m of the up and of the down component
            if j2 == 2*ll + 1:
                cu, cd = np.sqrt((2*ll + 1 + mj2)/(2*(2*ll + 1))), np.sqrt((2*ll + 1 - mj2)/(2*(2*ll + 1)))
            else:
                cu, cd = -np.sqrt((2*ll + 1 - mj2)/(2*(2*ll + 1))), np.sqrt((2*ll + 1 + mj2)/(2*(2*ll + 1)))
            if abs(mu) <= ll:
                a[2*(mu + ll)] = cu
            if abs(md) <= ll:
                a[2*(md + ll) + 1] = cd
            v = c2r.conj().T @ a
            cols += [v, _theta(U, v)]
            labels.append(f"j={j2}/2,mj=+{mj2}/2")
    von = np.column_stack(cols)
    err = np.abs(von.conj().T @ von - np.eye(2*n)).max()
    if err > 1e-12:
        raise ValueError(f"jj basis is not unitary: {err:.2e}")
    return von, labels


def kramers_eigenbasis(h0, mz_r, tr_tol=1e-3):
    """Backwards-compatible wrapper: symmetrise h0 under T and call kramers_basis
    (Mz only).  Columns are interleaved pairs (v, Theta v)."""
    h0 = np.asarray(h0, dtype=complex)
    theta = time_reversal_matrix(h0.shape[0]//2)
    h0_tr = theta @ h0.conj() @ theta.T
    err = np.max(np.abs(h0_tr - h0))
    if err > tr_tol:
        raise ValueError(f"on-site block breaks time reversal by {err:.3e} eV > {tr_tol:.1e}; "
                         "Kramers gauge only applies to a nonmagnetic H_loc")
    h0 = 0.5*(h0 + h0_tr)
    von, _ = kramers_basis(h0, theta, [("Mz", mz_r)])
    eon = np.real(np.diag(von.conj().T @ h0 @ von))
    return eon, von


def local_eigenbasis(h0, ll=2, kramers=False, tr_tol=1e-3):
    """Diagonalise the on-site Hamiltonian of one correlated shell.

    h0 is the (2*(2l+1))^2 block in the spin-interleaved real-harmonic basis.

    kramers=True (nonmagnetic h0 only): time-reversal symmetrise h0 and build the
    basis with kramers_basis (Mz, then Mx, My fix the gauge); otherwise plain eigh.

    Returns
    -------
    eon   : diagonal of von^+ h0 von
    von   : eigenvectors, columns are the impurity flavors (interleaved; with
            kramers=True flavor 2k+1 = Theta flavor 2k)
    htmat : CF for Trans.dat, = T2C.T with T2C[a,i] = <complex_a | flavor_i>,
            T2C = complex2real[perm,:] @ von: only the complex-harmonic rows are
            permuted to atom_d's spin-blocked order; the flavor columns stay
            interleaved.  No complex conjugation of von.
    mz    : L_z + 2 S_z (dimensionless, hbar = 1) in the flavor basis.  The
            electron moment is -mu_B Tr[mz rho] with rho_ab = <c_b^+ c_a>; it
            reduces to -mu_B sum_f mz_ff n_f only when mz or rho is diagonal in
            this basis.  kramers=True makes mz diagonal inside every pair.
    """
    nbnd = 2*ll + 1
    h0 = np.asarray(h0)
    if h0.shape != (2*nbnd, 2*nbnd):
        raise ValueError(f"h0 must be {2*nbnd}x{2*nbnd} for l={ll}, got {h0.shape}")

    ops = moment_operators(ll)
    if kramers:
        theta = time_reversal_matrix(nbnd)
        h0c = np.asarray(h0, dtype=complex)
        err = np.max(np.abs(theta @ h0c.conj() @ theta.T - h0c))
        if err > tr_tol:
            raise ValueError(f"on-site block breaks time reversal by {err:.3e} eV > {tr_tol:.1e}; "
                             "Kramers gauge only applies to a nonmagnetic H_loc")
        hs = 0.5*(h0c + theta @ h0c.conj() @ theta.T)
        von, _ = kramers_basis(hs, theta, [(k, ops[k]) for k in ("Mz", "Mx", "My")])
        eon = np.real(np.diag(von.conj().T @ hs @ von))
    else:
        eon, von = la.eigh(h0)
    perm = spin_block_perm(nbnd)
    htmat = (complex2real(ll, 2)[perm, :] @ von).T
    mz = von.conj().T @ ops["Mz"] @ von
    return eon, von, htmat, mz


def onsite_block_mismatch(h00, norb, natm, start=0):
    """Max |H_loc(atom i) - H_loc(atom 0)| over the natm correlated blocks.

    dump_dmft applies the same von to every block, which is only valid when
    the sites are equivalent in their local frames (mismatch ~ 0)."""
    blocks = [h00[start+i*norb:start+(i+1)*norb, start+i*norb:start+(i+1)*norb]
              for i in range(natm)]
    return max(np.max(np.abs(b - blocks[0])) for b in blocks)


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------

def _f(x):
    return f" {x:.17e}"


def dump_local_basis(eon, von, htmat, outdir='.'):
    """Write eon.dat / von.dat / htmat.dat for the local eigenbasis (full precision)."""
    norb = len(eon)
    with open(os.path.join(outdir, "eon.dat"), "w") as feon:
        for ibnd in range(norb):
            feon.write(f"{ibnd+1:5d}{ibnd+1:5d}{_f(eon[ibnd])}{_f(eon[ibnd]-eon[0])}\n")
    with open(os.path.join(outdir, "von.dat"), "w") as fvon:
        for ibnd in range(norb):
            for iorb in range(norb):
                fvon.write(f"{ibnd+1:5d}{iorb+1:5d}{_f(von[ibnd, iorb].real)}{_f(von[ibnd, iorb].imag)}\n")
            fvon.write('\n\n')
    with open(os.path.join(outdir, "htmat.dat"), "w") as fhtmat:
        for ibnd in range(norb):
            fhtmat.write("".join(_f(v) for iorb in range(norb)
                                 for v in (htmat[ibnd, iorb].real, htmat[ibnd, iorb].imag)) + "\n")


def write_eig_udmft(outdir, eig, proj, natm, norb, lnum, nspin, ntot, mu):
    """eig.dat and udmft.dat for ksum.  proj[k] is (nbnd, natm*norb) = <psi_nk | flavor>.
    Floats are written with 17 significant digits (lossless), k weights are 1/nk.
    Returns the sum of the k weights as written."""
    nkpt, nbnd = eig.shape
    w = 1.0/nkpt
    head = (f"{nkpt:5d}{natm:5d}{norb:5d}{lnum:5d}{nspin:5d}     # nkpt, natm, norb, L, nspin\n"
            f"{1:5d}{nbnd:5d}{nbnd:5d}{_f(float(ntot))}{_f(float(mu))}     # nemin, nemax, nbands, ntotal, mu\n\n")
    with open(os.path.join(outdir, "eig.dat"), "w") as feig:
        feig.write(head)
        for k in range(nkpt):
            feig.write(f"{k+1:5d}{_f(w)}  # ikpt, k-weight\n")
            for b in range(nbnd):
                feig.write(f"{b+1:5d}{_f(eig[k, b])}\n")
            feig.write('\n\n')
    with open(os.path.join(outdir, "udmft.dat"), "w") as fprj:
        fprj.write(head)
        for k in range(nkpt):
            fprj.write(f"{k+1:5d}{_f(w)}  # ikpt, k-weight\n")
            P = proj[k]
            for o in range(P.shape[1]):
                for b in range(nbnd):
                    fprj.write(f"{b+1:5d}{o+1:5d}{_f(P[b, o].real)}{_f(P[b, o].imag)}\n")
            fprj.write('\n\n')
    return float(sum(float(f"{w:.17e}") for _ in range(nkpt)))


def dump_dmft(eig, vec, mu, ntot, norb=10, natm=3, lnum=2, nspin=2, von=None, start=0, outdir='.',
              indices=None):
    """Write eig.dat and udmft.dat for ksum.

    The correlated shells are the Wannier orbitals indices[a] (one list of norb
    indices per atom), or start:start+norb*natm when indices is None.  If von is
    given, the projectors are rotated to the local basis of every shell,
    vmat = <psi_nk | w> von  (same von for all natm shells)."""
    if indices is None:
        indices = [list(range(start + a*norb, start + (a+1)*norb)) for a in range(natm)]
    idx = np.concatenate([np.asarray(i, dtype=int) for i in indices])
    tmat = np.eye(norb*natm, dtype=complex)
    if von is not None:
        if von.shape != (norb, norb):
            raise ValueError(f"von must be {norb}x{norb}, got {von.shape}")
        tmat = np.kron(np.eye(natm), von)
    proj = [vec[k, idx, :].conj().T @ tmat for k in range(vec.shape[0])]
    return write_eig_udmft(outdir, eig, proj, natm, norb, lnum, nspin, ntot, mu)


def dump_kpoints(kpts, fname='kpoints.dat'):
    """
    Dump k-points to a file.
    """
    kreal = Kpoint.k_length_accumulate(kpts.kreal)
    with open(fname, "w") as f:
        for ikpt, kpt in enumerate(kpts.kfrac):
            f.write("{:5d} {:15.9f} {:15.9f} {:15.9f} {:15.9f}\n".format(ikpt + 1, kpt[0], kpt[1], kpt[2], kreal[ikpt]))


# --------------------------------------------------------------------------
# unified export (PROPOSAL v5 section 3)
# --------------------------------------------------------------------------

DEFAULT_TOL = dict(tr_tol=1e-3, check_tol=1e-3, deg_tol=1e-8, m_tol=1e-8, res_tol=1e-10,
                   spin_mix_tol=1e-6, offdiag_tol=1e-6, offdiag_warn=1e-3)


def _cplx(a):
    a = np.asarray(a)
    return np.stack((a.real, a.imag), axis=-1).tolist()


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def basis_id(manifest):
    """sha256 of the canonical JSON of (l, flavor_kind, atoms' indices and frames, c2r, von, hr sha256)."""
    key = dict(l=manifest["l"], flavor_kind=manifest["flavor_kind"],
               atoms=[dict(wannier_indices=a["wannier_indices"], frame=a["frame"]) for a in manifest["atoms"]],
               c2r=manifest["c2r"], von=manifest["von"], hr_sha256=manifest["sources"].get("hr"))
    text = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def build_export(eigs, vecs, h00, atoms, ll, flavor_kind, ntot, mu, tol=None, log=print):
    """Everything of the export that does not touch the disk (PROPOSAL v5 section 3.2).

    atoms: list of dict(indices=[2n Wannier indices], frame=dict(z, x) or None,
    spin_rotated=bool).  eigs (nk, nbnd), vecs (nk, nwan, nbnd); h00 = H(R=0)."""
    tol = dict(DEFAULT_TOL, **(tol or {}))
    n = 2*ll + 1
    nf = 2*n
    nk = len(eigs)
    w = 1.0/nk
    idx = [np.asarray(a["indices"], dtype=int) for a in atoms]
    for i in idx:
        if len(i) != nf:
            raise ValueError(f"every correlated shell needs {nf} Wannier indices for l={ll}; got {len(i)}")
    Q = [[vecs[k][i, :].conj().T for i in idx] for k in range(nk)]
    H_W = [sum(w*Q[k][a].conj().T @ np.diag(eigs[k]) @ Q[k][a] for k in range(nk)) for a in range(len(idx))]
    H_W = [0.5*(h + h.conj().T) for h in H_W]
    H0 = [h00[np.ix_(i, i)] for i in idx]
    U = time_reversal_matrix(n)
    ops = moment_operators(ll)
    tr_W = float(np.linalg.norm(U @ H_W[0].conj() @ U.T - H_W[0]))

    if flavor_kind in ("kramers", "jj"):
        tr_max = float(np.abs(U @ H_W[0].conj() @ U.T - H_W[0]).max())
        if tr_max > tol["tr_tol"]:
            raise ValueError(f"H_mesh breaks time reversal by {tr_max:.3e} eV > tr_tol = {tol['tr_tol']}")
        E_sel = 0.5*(H_W[0] + U @ H_W[0].conj() @ U.T)
        if flavor_kind == "kramers":
            von, rules = kramers_basis(E_sel, U, [(k, ops[k]) for k in ("Mz", "Mx", "My")],
                                       tol["deg_tol"], tol["m_tol"], tol["res_tol"])
        else:
            von, rules = jj_basis(ll)
    elif flavor_kind == "orbital_spin":
        E_sel = H_W[0]
        von = orbital_spin_basis(E_sel, tol["deg_tol"], tol["spin_mix_tol"], tol["offdiag_tol"])
        rules = None
    else:
        raise ValueError(f"flavor_kind must be 'kramers', 'jj' or 'orbital_spin'; got {flavor_kind!r}")

    E_I = [von.conj().T @ h @ von for h in H_W]
    proj = [np.hstack([Q[k][a] @ von for a in range(len(idx))]) for k in range(nk)]
    direct = sum(w*proj[k][:, :nf].conj().T @ np.diag(eigs[k]) @ proj[k][:, :nf] for k in range(nk))
    if np.abs(direct - E_I[0]).max() > 1e-12*max(1.0, np.abs(E_I[0]).max()):
        raise ValueError("E_mesh_I != von^+ H_mesh_W von")
    eimp_abs = np.real(np.diag(E_I[0]))
    off = float(np.abs(E_I[0] - np.diag(np.diag(E_I[0]))).max())
    mismatch = max(float(np.abs(e - E_I[0]).max()) for e in E_I)
    if mismatch > tol["check_tol"]:
        raise ValueError(f"correlated shells differ by {mismatch:.3e} eV > check_tol; one von cannot serve all")
    if off > tol["offdiag_warn"]:
        log(f"udmft: WARNING off-diagonal Eimp in the impurity basis {off:.3e} eV (dropped by ksum)")
    ortho = None
    if vecs.shape[1] == vecs.shape[2]:
        ortho = max(float(np.abs(p.conj().T @ p - np.eye(p.shape[1])).max()) for p in proj)
        if ortho > 1e-12:
            raise ValueError(f"projectors are not orthonormal: {ortho:.2e}")
    c2r = complex2real(ll, 2)
    perm = spin_block_perm(n)
    T2C = c2r[perm, :] @ von
    c2r_err = check_complex2real(ll)
    if c2r_err > 1e-12:
        raise ValueError(f"complex2real({ll}) differs from the independent real-harmonic reference by {c2r_err:.2e}")
    mz = von.conj().T @ ops["Mz"] @ von
    res = dict(weight_sum=None, projector_orthonormality=ortho, tr_residual_W=tr_W,
               atom_mismatch=mismatch, eimp_offdiag=off,
               mesh_vs_R0=float(np.abs(H_W[0] - H0[0]).max()))
    if flavor_kind in ("kramers", "jj"):
        res["kramers_pair_splitting"] = float(np.abs(eimp_abs[1::2] - eimp_abs[0::2]).max())
    else:
        res["orbital_spin_offdiag"] = off
    return dict(ll=ll, n=n, nf=nf, natm=len(idx), atoms=atoms, eigs=eigs, proj=proj, H_W=H_W, E_I=E_I,
                von=von, rules=rules, c2r=c2r, perm=perm, T2C=T2C, mz=mz, eimp_abs=eimp_abs,
                eon=np.real(la.eigvalsh(E_sel)), flavor_kind=flavor_kind, ntot=ntot, mu=mu, tol=tol,
                residuals=res, c2r_check=dict(max_abs=c2r_err, ok=True))


def write_trans(path, T2C):
    """Trans.dat: Sigind = diag(1..2n), CF = T2C.T, with the flavor-order marker."""
    nf = len(T2C)
    with open(path, "w") as f:
        f.write(f"{nf} {nf}\n#---- Sigind follows  flavor_order=interleaved\n")
        for i in range(nf):
            f.write(" ".join(str(i+1 if j == i else 0) for j in range(nf)) + "\n")
        f.write("#---- CF follows\n")
        for row in T2C.T:
            f.write("".join(f"{_f(z.real)}{_f(z.imag)}" for z in row) + "\n")


def write_seed_sigma(path, mz_diag, delta, beta, max_freq, edc, bid):
    """Static seed Sigma = -delta*Mz_ff + edc (interleaved flavors), one '#' header line."""
    num_freq = int(max_freq*beta/2.0/np.pi + 0.5)
    sigma = -delta*np.asarray(mz_diag) + edc
    with open(path, "w") as fh:
        fh.write(f"# seeded Sigma = -delta*Mz + edc  delta={delta!r} edc={edc!r} beta={beta!r} "
                 f"max_freq={max_freq!r} nfreq={num_freq} flavor_order=interleaved basis_id={bid}\n")
        for k in range(num_freq):
            om = (2.0*k + 1.0)*np.pi/beta
            fh.write(_f(om) + "".join(f"{_f(s)}{_f(0.0)}" for s in sigma) + "\n")
    return num_freq


def write_export(data, outdir, sources, seed=None):
    """Writes eig.dat, udmft.dat, Trans.dat, eon.dat, von.dat, htmat.dat, (Sig.out) and dmft_basis.json."""
    os.makedirs(outdir, exist_ok=True)
    wsum = write_eig_udmft(outdir, data["eigs"], data["proj"], data["natm"], data["nf"], data["ll"], 2,
                           data["ntot"], data["mu"])
    if abs(wsum - 1.0) > 1e-12:
        raise ValueError(f"k weights add up to {wsum!r}")
    write_trans(os.path.join(outdir, "Trans.dat"), data["T2C"])
    dump_local_basis(data["eimp_abs"], data["von"], data["T2C"].T, outdir)
    res = dict(data["residuals"], weight_sum=wsum)
    m = dict(format_version=1, created=datetime.datetime.now().isoformat(timespec="seconds"),
             status="complete", flavor_order="interleaved", spherical_spin_order="blocked",
             flavor_kind=data["flavor_kind"], l=data["ll"], n_flavor=data["nf"], natm=data["natm"],
             atoms=[dict(wannier_indices=[int(i) for i in a["indices"]], frame=a.get("frame"),
                         spin_rotated_to_local_frame=bool(a.get("spin_rotated", False))) for a in data["atoms"]],
             c2r=_cplx(data["c2r"]), perm_rows=[int(i) for i in data["perm"]], von=_cplx(data["von"]),
             T2C=_cplx(data["T2C"]), H_mesh_W=[_cplx(h) for h in data["H_W"]],
             E_mesh_I=[_cplx(h) for h in data["E_I"]],
             eimp_abs=[float(x) for x in data["eimp_abs"]],
             eimp_rel=[float(x - data["eimp_abs"][0]) for x in data["eimp_abs"]],
             mu_lda=float(data["mu"]), ntotal=float(data["ntot"]), eon=[float(x) for x in data["eon"]],
             mz=_cplx(data["mz"]), mz_note="L_z + 2 S_z (dimensionless, hbar = 1); mu_z = -mu_B Tr[mz rho]",
             pairs=[[2*k, 2*k+1] for k in range(data["n"])], pair_rule=data["rules"],
             residuals=res, c2r_check=data["c2r_check"],
             basis_model=("Wannier orbitals interpreted as the standard real harmonics of "
                          "swept.basis.complex2real in each shell's local frame"),
             tolerances=data["tol"], sources=dict(sources))
    m["basis_id"] = basis_id(m)
    files = ["eig.dat", "udmft.dat", "Trans.dat"]
    if seed is not None:
        write_seed_sigma(os.path.join(outdir, "Sig.out"), np.real(np.diag(data["mz"])), seed["delta"],
                         seed["beta"], seed["max_freq"], seed["edc"], m["basis_id"])
        files.append("Sig.out")
    m["files"] = {f: dict(path=f, sha256=_sha256(os.path.join(outdir, f))) for f in files}
    tmp = os.path.join(outdir, "dmft_basis.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(m, fh, indent=1)
    os.replace(tmp, os.path.join(outdir, "dmft_basis.json"))
    return m
