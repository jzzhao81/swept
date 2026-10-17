#!/usr/bin/env python
'''
Fermi-surface cuts: band energies and orbital weights on 2D planes in k space.

A plane is k = origin + s*u + t*v with s, t in [-1, 1] (u, v = half extents),
given in fractional reciprocal coordinates (basis = "frac", the rows of
2 pi inv(rvec).T) or Cartesian 1/Angstrom (basis = "cart", 2 pi included).
Only the bands with energies inside mu +- window somewhere on the plane are
kept.  Plot with plot_fermi_surface.py.
'''

import numpy as np
from . import mpiutils as mpi


def plane_kpoints(origin, u, v, nk, bvec, basis="frac"):
    """Fractional and Cartesian k of an nk[0] x nk[1] plane, s index slowest.
    Returns s (nu,), t (nv,), kfrac (nu*nv, 3), kcart (nu*nv, 3)."""
    origin, u, v = (np.asarray(x, dtype=float) for x in (origin, u, v))
    if basis == "frac":
        origin, u, v = origin @ bvec, u @ bvec, v @ bvec
    s = np.linspace(-1.0, 1.0, nk[0])
    t = np.linspace(-1.0, 1.0, nk[1])
    kcart = origin + s[:, None, None]*u + t[None, :, None]*v
    kcart = kcart.reshape(-1, 3)
    return s, t, kcart @ np.linalg.inv(bvec), kcart


def plane_axes(u, v):
    """Orthonormal in-plane axes (e1 along u) and the (s, t) -> (x, y) map
    (Cartesian lengths along e1, e2), so non-orthogonal u, v plot true to shape."""
    e1 = u/np.linalg.norm(u)
    w = v - np.dot(v, e1)*e1
    e2 = w/np.linalg.norm(w)
    return e1, e2, np.array([[np.dot(u, e1), np.dot(u, e2)], [np.dot(v, e1), np.dot(v, e2)]])


def lll_reduce(bvec, delta=0.99, max_steps=10000):
    """LLL-reduced rows of the same lattice (short, nearly orthogonal)."""
    b = np.array(bvec, dtype=float)

    def gram_schmidt(b):
        bs, mu = b.copy(), np.zeros((3, 3))
        for i in range(3):
            for j in range(i):
                mu[i, j] = np.dot(b[i], bs[j])/np.dot(bs[j], bs[j])
                bs[i] = bs[i] - mu[i, j]*bs[j]
        return bs, mu

    bs, mu = gram_schmidt(b)
    k = 1
    for _ in range(max_steps):
        if k >= 3:
            return b
        for j in range(k-1, -1, -1):
            q = np.rint(mu[k, j])
            if q != 0:
                b[k] = b[k] - q*b[j]
                bs, mu = gram_schmidt(b)
        if np.dot(bs[k], bs[k]) >= (delta - mu[k, k-1]**2)*np.dot(bs[k-1], bs[k-1]):
            k += 1
        else:
            b[[k-1, k]] = b[[k, k-1]]
            bs, mu = gram_schmidt(b)
            k = max(k-1, 1)
    raise ValueError("lll_reduce: no convergence")


def voronoi_vectors(bvec, max_steps=10000):
    """The 14 candidate Voronoi-relevant vectors of the lattice with rows bvec.
    The basis is LLL-reduced first (so any shear converges in a few steps),
    then Selling-reduced: start from the superbase v0 = -(b1+b2+b3), v1..v3 = b_i
    and, while some v_i.v_j > 0, flip v_i and add it to the two other vectors
    (sum stays 0, sum |v|^2 drops by 2 v_i.v_j).  For the resulting obtuse
    superbase every Voronoi-relevant vector is one of +-v_i, +-(v_i + v_j)
    (Voronoi / Conway-Sloane).  The obtuseness test is relative per pair,
    v_i.v_j > 1e-12 |v_i||v_j|, so strongly anisotropic lattices are handled."""
    b = lll_reduce(bvec)
    v = [-(b[0] + b[1] + b[2]), b[0].copy(), b[1].copy(), b[2].copy()]
    for _ in range(max_steps):
        hit = [(i, j) for i in range(4) for j in range(i+1, 4)
               if np.dot(v[i], v[j]) > 1e-12*np.linalg.norm(v[i])*np.linalg.norm(v[j])]
        if not hit:
            break
        i, j = hit[0]
        vi = v[i].copy()
        for k in range(4):
            if k not in (i, j):
                v[k] = v[k] + vi
        v[i] = -vi
    else:
        raise ValueError("voronoi_vectors: Selling reduction did not terminate")
    # v_i + v_j = -(v_k + v_l) for complementary pairs: the pairs with v0 suffice
    g = np.array(v + [v[0] + v[j] for j in (1, 2, 3)])
    return np.vstack([g, -g])


def bz_function(kcart, bvec, chunk=20000):
    """f(k) = max_G (2 k.G - |G|^2) over the Voronoi-relevant G (voronoi_vectors):
    f <= 0 exactly inside the first Brillouin zone, f = 0 on its boundary, f > 0
    outside, for any k.  Away from the zone f is not the maximum over all G;
    only its sign and zero level are meaningful (enough for the BZ contour)."""
    kcart = np.asarray(kcart, dtype=float)
    gvec = voronoi_vectors(bvec)
    g2 = (gvec**2).sum(1)
    out = np.empty(len(kcart))
    for s in range(0, len(kcart), chunk):
        out[s:s+chunk] = np.max(2*kcart[s:s+chunk] @ gvec.T - g2, axis=1)
    return out


def plane_eigens(model, kfrac, plists, chunk=1000):
    """Energies (nk, nwan) and projection weights (nproj, nk, nwan) at kfrac.
    Chunks are dealt round-robin over MPI ranks; every rank gets the result."""
    with mpi.collective("plane_eigens"):
        nk = len(kfrac)
        eigs = np.zeros((nk, model.nwan))
        projs = np.zeros((len(plists), nk, model.nwan))
        starts = list(range(0, nk, chunk))
        for s in starts[mpi.rank::mpi.size]:
            e, vec = np.linalg.eigh(model.get_bulk_Hks(kfrac[s:s+chunk]))
            eigs[s:s+chunk] = e
            weight = np.abs(vec)**2                       # (k, orbital, band)
            for ip, plist in enumerate(plists):
                projs[ip, s:s+chunk] = weight[:, plist, :].sum(axis=1)
    eigs = mpi.Allreduce(eigs)
    projs = mpi.Allreduce(projs)
    # same reduced data on every rank, so the verdict agrees; the check itself is guarded
    with mpi.collective("plane_eigens (output check)"):
        finite = bool(np.isfinite(eigs).all() and np.isfinite(projs).all())
    if not finite:
        raise FloatingPointError("plane_eigens: non-finite energies or projection weights")
    return eigs, projs
