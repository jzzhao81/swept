#!/usr/bin/env python
'''
Bare static Lindhard response, resolved by atomic site, on a Gamma-centred mesh.

For an orbital set P with point-site vertices (orbital a at tau_a = tau0_s + L_a,
fractional, L_a integer) and a mesh representative q = n/N in [0,1)^3:

    Ot^s_nm(k,q) = sum_{a in P, a on site s} e^{-2 pi i q.L_a} U*_an(k) U_am(k+q)
    chi_st(q)    = 1/Nk sum_k sum_nm Ot^s_nm Ot^t_nm* F_nm,
    F_nm         = -(f(E_n(k)) - f(E_m(k+q))) / (E_n(k) - E_m(k+q))  >= 0
    chi(Q)       = sum_st e^{-2 pi i Q.tau0_s} chi_st(q) e^{+2 pi i Q.tau0_t},   Q = q + G

chi_st is Hermitian positive semi-definite and chi(Q) a non-negative real
number; chi(Q) is not periodic in G (intra-cell interference).  The "cell"
vertex is the special case of one site holding all of P with tau = 0 (the
lattice-gauge cell-density correlation, in general not covariant under the
crystal's space-group operations).  No spatial symmetry is used; only the
exact static relation chi_st(-q) = conj(chi_st(q)) halves the q work.

Units: 1/eV per model unit cell.  U are eigenvectors of Model.get_bulk_Hk
(H(k) = sum_R e^{2 pi i k.R} H(R)), so the vertex phases above are what turn
the lattice-gauge overlaps into point-site densities.

Band window: only the bands with energies within mu +- window somewhere on the
mesh are kept, extended until no kept band is within degeneracy_tol of a dropped
one at any k (select_bands).  Exactly degenerate subspaces are therefore never cut
and chi0 does not depend on the eigenvector basis inside them; dropping the bands
outside the window remains an approximation (truncation error), to be checked by
widening the window.

Mesh index convention everywhere: ik = (i0*N1 + i1)*N2 + i2.
'''

import numpy as np
from . import mpiutils as mpi

LOG2 = np.log(2.0)


def fermi(e, kT):
    return 0.5*(1.0 - np.tanh(e/(2.0*kT)))


def fermi_difference(a, b, kT):
    """F(a, b) = -(f(a) - f(b))/(a - b), with the limit f(1-f)/kT at a = b.
    Exact identity F = sinhc(d)/(4 kT cosh u cosh v), u = a/2kT, v = b/2kT,
    d = u - v, evaluated in log space:
        log sinhc(d) - log cosh u - log cosh v
          = (|d| - |u| - |v|) + log1p(-e^{-2|d|}) - log|d| + log 2
            - log1p(e^{-2|u|}) - log1p(e^{-2|v|})                     (|d| >= 1e-3)
    with |d| - |u| - |v| taken exactly (0 for opposite signs, -2 min(|u|,|v|)
    otherwise), so no large numbers cancel even for kT -> 0; for |d| < 1e-3
    a series for log sinhc.  No degeneracy threshold, no overflow, and
    F(a, b) == F(b, a) bit for bit."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    u, v = a/(2.0*kT), b/(2.0*kT)
    au, av = np.abs(u), np.abs(v)
    d = np.abs(u - v)
    tails = np.log1p(np.exp(-2.0*au)) + np.log1p(np.exp(-2.0*av))
    small = d < 1e-3
    expo = np.empty_like(d)
    ds = d[small]
    expo[small] = np.log1p(ds*ds/6.0 + ds**4/120.0) - (au[small] + av[small] + tails[small] - 2*LOG2)
    dl = d[~small]
    lead = np.where(u*v > 0, -2.0*np.minimum(au, av), 0.0)[~small]            # = |d| - |u| - |v|
    expo[~small] = lead + np.log1p(-np.exp(-2.0*dl)) - np.log(dl) + LOG2 - tails[~small]
    return np.exp(expo)/(4.0*kT)


def mesh_index(n, mesh):
    n0, n1, n2 = mesh
    return ((n[..., 0] % n0)*n1 + n[..., 1] % n1)*n2 + n[..., 2] % n2


def mesh_points(mesh):
    """Integer mesh coordinates (nk, 3) in mesh_index order."""
    return np.array(np.meshgrid(*[np.arange(n) for n in mesh], indexing="ij")).reshape(3, -1).T


DEGENERACY_TOL = 1e-4      # eV, default for select_bands / mesh_eigens / lindhard


def select_bands(eall, window, degeneracy_tol=DEGENERACY_TOL):
    """Bands kept for chi0, from the energies E - mu (nk, nb) on the whole mesh (sorted per k).

    First the bands inside +- window somewhere on the mesh; that set is a contiguous range,
    because the per-band minimum and maximum over the mesh both grow with the band index.
    Then the range is closed under near degeneracy: while, at any k, the band just outside
    an end lies within degeneracy_tol (eV) of the end band, it is added.  Steps along
    different k chain, so no kept band is within the tolerance of a dropped one anywhere.
    Exactly degenerate subspaces are thus kept or dropped whole, and chi0 does not depend on
    the eigenvector basis eigh picks inside them; for split levels (numerical noise, near
    degeneracy) the tolerance only keeps the cut away from them -- energies are not changed.
    Returns (bands 0-based, info) with info = dict(initial=[lo, hi] 0-based,
    boundary_gap=[lower, upper] in eV after the closure (inf without an outer band),
    added=[(side, band, k index), ...])."""
    for name, value in (("window", window), ("degeneracy_tol", degeneracy_tol)):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) \
                or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number (eV), got {value!r}")
    nb = eall.shape[1]
    sel = np.flatnonzero((eall.min(axis=0) < window) & (eall.max(axis=0) > -window))
    if sel.size == 0:
        raise ValueError(f"no band within mu +- {window} eV on the mesh")
    lo, hi = int(sel[0]), int(sel[-1])
    assert sel.size == hi - lo + 1, "band window is not contiguous"
    info = dict(initial=[lo, hi], added=[])
    while lo > 0:
        near = np.flatnonzero(eall[:, lo] - eall[:, lo-1] <= degeneracy_tol)
        if near.size == 0:
            break
        lo -= 1
        info["added"].append(("lower", lo, int(near[0])))
    while hi < nb - 1:
        near = np.flatnonzero(eall[:, hi+1] - eall[:, hi] <= degeneracy_tol)
        if near.size == 0:
            break
        hi += 1
        info["added"].append(("upper", hi, int(near[0])))
    info["boundary_gap"] = [float((eall[:, lo] - eall[:, lo-1]).min()) if lo > 0 else np.inf,
                            float((eall[:, hi+1] - eall[:, hi]).min()) if hi < nb - 1 else np.inf]
    return np.arange(lo, hi + 1), info


def mesh_eigens(model, mesh, mu, window, degeneracy_tol=DEGENERACY_TOL, info=None):
    """Energies E - mu (nk, nb) and eigenvectors (nk, nwan, nb) of the bands
    chosen by select_bands (inside mu +- window somewhere on the mesh, closed under
    degeneracy).  Planes i0 are dealt over MPI ranks.  Returns (eigs, vecs, bands)
    with 0-based bands; a dict passed as info receives select_bands' info."""
    # every local stage is guarded (mpi.collective): a failure on one rank is
    # raised on all of them; the checks after each Allreduce see identical data
    with mpi.collective("mesh_eigens (energies)"):
        n0, n1, n2 = mesh
        npl = n1*n2
        eall = np.zeros((n0*npl, model.nwan))
        mine = list(range(n0))[mpi.rank::mpi.size]
        for i0 in mine:
            eall[i0*npl:(i0+1)*npl] = np.linalg.eigvalsh(model.get_bulk_Hk_slab(mesh, i0))
    eall = mpi.Allreduce(eall)
    with mpi.collective("mesh_eigens (energy check, band window)"):
        finite = bool(np.isfinite(eall).all())
        if finite:
            eall = eall - mu
            try:
                bands, sel_info = select_bands(eall, window, degeneracy_tol)
                err = None
            except ValueError as exc:            # empty window: same data, same verdict everywhere
                err = exc
    if not finite:
        raise mpi.NonFiniteResult("mesh_eigens: non-finite band energies")
    if err is not None:
        raise err
    if info is not None:
        info.update(sel_info)
    with mpi.collective("mesh_eigens (eigenvectors)"):
        vecs = np.zeros((n0*npl, model.nwan, len(bands)), dtype=complex)
        for i0 in mine:
            _, v = np.linalg.eigh(model.get_bulk_Hk_slab(mesh, i0))
            vecs[i0*npl:(i0+1)*npl] = v[:, :, bands]
    vecs = mpi.Allreduce(vecs)
    with mpi.collective("mesh_eigens (eigenvector check)"):
        finite = bool(np.isfinite(vecs).all())
        eigs = eall[:, bands]
    if not finite:
        raise mpi.NonFiniteResult("mesh_eigens: non-finite eigenvectors")
    return eigs, vecs, bands


def make_channel(orbitals, site_of_orbital, offsets, tau0):
    """Channel = orbital set P with its point-site data.
    orbitals (n,) model indices; site_of_orbital (n,) in 0..ns-1; offsets (n, 3)
    integer L_a; tau0 (ns, 3) fractional site positions."""
    orbitals = np.asarray(orbitals, dtype=int)
    site = np.asarray(site_of_orbital, dtype=int)
    offsets = np.asarray(offsets, dtype=int).reshape(len(orbitals), 3)
    tau0 = np.asarray(tau0, dtype=float).reshape(-1, 3)
    if orbitals.ndim != 1 or orbitals.size == 0 or len(np.unique(orbitals)) != len(orbitals):
        raise ValueError("a channel is a non-empty set of distinct orbitals")
    if site.shape != orbitals.shape or site.min() < 0 or site.max() >= len(tau0):
        raise ValueError("site index of every orbital must lie in 0..len(tau0)-1")
    return dict(orbitals=orbitals, site=site, offsets=offsets, tau0=tau0)


def cell_channel(orbitals):
    """'cell' vertex: one site, all tau = 0."""
    n = len(orbitals)
    return make_channel(orbitals, np.zeros(n, int), np.zeros((n, 3), int), np.zeros((1, 3)))


def site_matrices(eigs, vecs, points, mesh, iq, channels, kT, kblock=4096):
    """chi_st(q) (complex, ns x ns) of every channel at mesh point iq."""
    qf = points[iq]/np.asarray(mesh, dtype=float)
    kq_all = mesh_index(points + points[iq], mesh)
    out = [np.zeros((len(c["tau0"]),)*2, dtype=complex) for c in channels]
    phases = [np.exp(-2j*np.pi*(c["offsets"] @ qf)) for c in channels]
    for s in range(0, len(points), kblock):
        kq = kq_all[s:s+kblock]
        vk, vq = vecs[s:s+kblock], vecs[kq]
        sqf = np.sqrt(fermi_difference(eigs[s:s+kblock, :, None], eigs[kq][:, None, :], kT))
        for ic, (c, ph) in enumerate(zip(channels, phases)):
            ns = len(c["tau0"])
            ot = np.zeros((ns,) + sqf.shape, dtype=complex)
            for isite in range(ns):
                sel = c["site"] == isite
                orb = c["orbitals"][sel]
                left = vk[:, orb, :].conj()*ph[sel][None, :, None]        # (k, a, n)
                ot[isite] = left.transpose(0, 2, 1) @ vq[:, orb, :]       # (k, n, m)
            w = (ot*sqf).reshape(ns, -1)
            out[ic] += w @ w.conj().T                                     # sum Ot^s Ot^t* F
    return [m/len(points) for m in out]


def lindhard(model, mesh, mu, kT, window, channels, kblock=4096, use_pairs=True,
             imag_rtol=1e-10, imag_atol=1e-14, log=print, degeneracy_tol=DEGENERACY_TOL):
    """Site-resolved chi for every channel on the whole mesh.
    Returns dict(chi=[(nk, ns, ns) complex per channel], points, bands,
    window_occupation)."""
    sel = {}
    eigs, vecs, bands = mesh_eigens(model, mesh, mu, window, degeneracy_tol, info=sel)
    errors = []
    try:                                    # all local work up to the next collective
        points = mesh_points(mesh)
        nk = len(points)
        log(f"chi0: {nk} k-points, {len(bands)} bands within mu +- {window} eV: {(bands+1).tolist()}")
        log_band_selection(sel, degeneracy_tol, log)
        minus = mesh_index(-points, mesh)
        todo = np.flatnonzero(np.arange(nk) <= minus) if use_pairs else np.arange(nk)
        log(f"chi0: {len(todo)} q computed of {nk}" + (" (q, -q paired)" if use_pairs else ""))

        chi = [np.zeros((nk,) + (len(c["tau0"]),)*2, dtype=complex) for c in channels]
        mine = todo[mpi.rank::mpi.size]
        for count, iq in enumerate(mine):
            mats = site_matrices(eigs, vecs, points, mesh, iq, channels, kT, kblock)
            for ic, m in enumerate(mats):
                im = minus[iq]
                if use_pairs and im == iq:
                    # self-paired (2q = 0 mod G): chi_st = conj(chi_st), i.e. real symmetric
                    imag = np.abs(m.imag).max()
                    if imag > imag_atol + imag_rtol*np.abs(m).max():
                        errors.append(f"q index {iq} channel {ic}: |Im chi| = {imag:.3e} at a "
                                      "self-paired q (expected real)")
                    m = m.real.astype(complex)
                chi[ic][iq] = m
                if use_pairs and im != iq:
                    chi[ic][im] = m.conj()
            if mpi.rank0 and (count % 200 == 0 or count == len(mine) - 1):
                log(f"  rank 0: {count+1}/{len(mine)} q")
    except Exception as exc:                # collective: no rank may stay in Allreduce alone
        errors.append(f"rank {mpi.rank}: {type(exc).__name__}: {exc}")
    errors = mpi.allreduce(errors)
    if errors:
        raise ValueError("; ".join(errors[:5]))
    chi = [mpi.Allreduce(c) for c in chi]
    with mpi.collective("lindhard (window occupation)"):
        occupation = float(fermi(eigs, kT).sum())/nk
    return dict(chi=chi, points=points, bands=bands, window_occupation=occupation, band_selection=sel)


def log_band_selection(info, degeneracy_tol, log):
    lo, hi = info["initial"]
    if info["added"]:
        log(f"chi0: degeneracy closure (tol {degeneracy_tol:g} eV) extends bands {lo+1}-{hi+1} by "
            f"{len(info['added'])}: " + ", ".join(f"{b+1} ({side}, first at k index {k})"
                                                 for side, b, k in info["added"]))
    gap = ", ".join("none" if np.isinf(g) else f"{g*1e3:.3f} meV" for g in info["boundary_gap"])
    log(f"chi0: smallest gap on the mesh between kept and dropped bands (below, above): {gap}")


def chi_of_matrix(m, tau0, qfrac):
    """chi(Q) = sum_st e^{-2 pi i Q.tau0_s} m_st e^{+2 pi i Q.tau0_t} for one site matrix."""
    ph = np.exp(2j*np.pi*np.asarray(tau0) @ np.asarray(qfrac, dtype=float))
    return float(np.real(ph.conj() @ m @ ph))


def chi_physical(chi_st, tau0, mesh, qfrac_list):
    """chi(Q) for Q (fractional, any reciprocal cell) on mesh + integer G.
    chi_st: (nk, ns, ns) in mesh_index order; returns real (nQ,)."""
    mesh = np.asarray(mesh)
    qfrac_list = np.atleast_2d(np.asarray(qfrac_list, dtype=float))
    n = qfrac_list*mesh
    nint = np.rint(n).astype(int)
    if np.abs(n - nint).max() > 1e-8:
        raise ValueError("Q must lie on the k mesh plus an integer reciprocal-lattice vector")
    ph = np.exp(2j*np.pi*qfrac_list @ np.asarray(tau0).T)                 # (nQ, ns)
    mats = chi_st[mesh_index(nint, mesh)]                                  # (nQ, ns, ns)
    val = np.einsum("qs,qst,qt->q", ph.conj(), mats, ph)
    return val.real


def chi_physical_npz(data, channel, qfrac_list):
    """chi(Q) of one channel from a loaded chi0.npz (format version 2)."""
    if int(data["version"]) != 2:
        raise ValueError("chi_physical_npz needs a version-2 chi0.npz")
    return chi_physical(data[f"chi_{channel}"], data[f"tau0_{channel}"], data["mesh"], qfrac_list)
