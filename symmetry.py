#!/usr/bin/env python
'''
Symmetry check of a Wannier tight-binding model.

H(k) = sum_R exp(2 pi i k.R) H(R)  (wannier90 hr convention, no orbital
positions in the phase).  An operation g = {W|t} (fractional coordinates,
x -> W x + t), optionally combined with time reversal T, is a symmetry when

    unitary       g :  H(k')  = U(k) H(k)   U(k)^+ ,  k' =  W^-T k
    antiunitary  gT :  H(k')  = U(k) H(k)^* U(k)^+ ,  k' = -W^-T k

g moves site a to site b:  W tau_a + t = tau_b + L_a  (L_a a lattice vector),
where tau are the positions of the Wannier functions of the home cell R = 0
(the shell pos, not reduced mod 1: tau = 1.5 and 0.5 differ by one cell in
every H(R)).  Sites are matched within symprec, a Cartesian distance as in
spglib.  In the global frame

    U_{(b m' s'), (a m s)} = D^l_{m'm}(R) D^1/2_{s's}(R) exp(-2 pi i k'.L_a)

with R the Cartesian rotation, D^l on real harmonics in wannier90 order and
D^1/2 the SU(2) matrix of the proper part of R.  For gT the spin part is
followed by T = i sigma_y K.  Shells with local axes (z/x) are handled by
U -> S^+ U S (S^+ U S^* for gT), S the block-diagonal local-frame matrix.

Two numbers per operation, maximum over random k-points:
    eig : max_n |E_n(k') - E_n(k)|       independent of the orbital basis
    mat : max_ij |H(k') - U H U^+|_ij     tests orbital order / axes / spin too
'''

import warnings

import numpy as np
from scipy.spatial.transform import Rotation


# ----------------------------------------------------------------------------
# representation matrices
# ----------------------------------------------------------------------------

def real_harmonics(l, r):
    """Real spherical harmonics (wannier90 order and signs) at unit vectors r,
    shape (npt, 2l+1).  The common factor 1/sqrt(pi) is dropped."""
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    if l == 0:
        cols = [0.5*np.ones_like(x)]
    elif l == 1:
        c = np.sqrt(3.0)/2.0
        cols = [c*z, c*x, c*y]
    elif l == 2:
        cols = [np.sqrt(5.0)/4.0*(3*z*z - 1),
                np.sqrt(15.0)/2.0*x*z,
                np.sqrt(15.0)/2.0*y*z,
                np.sqrt(15.0)/4.0*(x*x - y*y),
                np.sqrt(15.0)/2.0*x*y]
    elif l == 3:
        cols = [np.sqrt(7.0)/4.0*z*(5*z*z - 3),
                np.sqrt(21.0)/(4*np.sqrt(2.0))*x*(5*z*z - 1),
                np.sqrt(21.0)/(4*np.sqrt(2.0))*y*(5*z*z - 1),
                np.sqrt(105.0)/4.0*z*(x*x - y*y),
                np.sqrt(105.0)/2.0*x*y*z,
                np.sqrt(35.0)/(4*np.sqrt(2.0))*x*(x*x - 3*y*y),
                np.sqrt(35.0)/(4*np.sqrt(2.0))*y*(3*x*x - y*y)]
    else:
        raise ValueError(f"l = {l} not supported (0-3)")
    return np.column_stack(cols)


_SPHERE = np.random.default_rng(12345).normal(size=(64, 3))
_SPHERE /= np.linalg.norm(_SPHERE, axis=1)[:, None]


def orbital_dmat(l, rot):
    """D^l(R) for a Cartesian (proper or improper) rotation R:
    Y_m(R^-1 r) = sum_m' Y_m'(r) D_m'm.  Real orthogonal."""
    rot = np.asarray(rot, dtype=float)
    yr = real_harmonics(l, _SPHERE)
    yrot = real_harmonics(l, _SPHERE @ rot)          # rows: (R^T r)^T = r^T R
    dmat, *_ = np.linalg.lstsq(yr, yrot, rcond=None)
    return dmat


def spin_dmat(rot):
    """SU(2) matrix exp(-i theta n.sigma/2) of the proper part of R, in the
    (up, dn) basis along global z.  Defined up to an overall sign."""
    rot = np.asarray(rot, dtype=float)
    proper = rot * np.sign(np.linalg.det(rot))
    rotvec = Rotation.from_matrix(proper).as_rotvec()
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(2, dtype=complex)
    nx, ny, nz = rotvec / theta
    nsig = np.array([[nz, nx - 1j*ny], [nx + 1j*ny, -nz]])
    return np.cos(theta/2)*np.eye(2) - 1j*np.sin(theta/2)*nsig


# T = i sigma_y K:  T|up> = |dn>,  T|dn> = -|up>
SPIN_T = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)


def frame_matrix(z, x):
    """Columns = local x, y, z axes in Cartesian coordinates."""
    from .spinor import zx_to_rmat
    return zx_to_rmat(z, x)


# ----------------------------------------------------------------------------
# operations
# ----------------------------------------------------------------------------

def cart_rotation(wmat, lattice):
    """Cartesian rotation of the fractional W (lattice rows = a1, a2, a3)."""
    at = np.asarray(lattice, dtype=float).T
    return at @ np.asarray(wmat, dtype=float) @ np.linalg.inv(at)


def op_symbol(rot):
    """Short Hermann-Mauguin-like name and axis of a Cartesian rotation."""
    det = int(round(np.linalg.det(rot)))
    proper = rot * det
    order = {3: 1, -1: 2, 0: 3, 1: 4, 2: 6}[int(round(np.trace(proper)))]
    if det > 0:
        name = str(order)
    else:
        name = {1: "-1", 2: "m", 3: "-3", 4: "-4", 6: "-6"}[order]
    if order == 1:
        return name, None
    w, v = np.linalg.eig(proper)
    axis = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    axis /= np.linalg.norm(axis)
    first = axis[np.argmax(np.abs(axis) > 1e-6)]
    return name, axis * np.sign(first)


def find_operations(lattice, positions, numbers, magmoms=None, symprec=1e-3):
    """Space-group (or, with magmoms, magnetic space-group) operations of the
    site set via spglib.  Returns (ops, label); ops = [(W, t, time_reversal)]."""
    try:
        import spglib
    except ImportError as exc:
        raise ImportError("the symmetry check needs spglib (pip install spglib)") from exc

    lattice = np.asarray(lattice, dtype=float)
    positions = np.asarray(positions, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if magmoms is None:
            cell = (lattice, positions, numbers)
            sym = spglib.get_symmetry(cell, symprec=symprec)
            if sym is None:
                raise ValueError("spglib could not determine the symmetry of the sites")
            label = spglib.get_spacegroup(cell, symprec=symprec) or "?"
            trs = np.zeros(len(sym["rotations"]), dtype=bool)
        else:
            cell = (lattice, positions, numbers, np.asarray(magmoms, dtype=float))
            sym = spglib.get_magnetic_symmetry(cell, symprec=symprec)
            if sym is None:
                raise ValueError("spglib could not determine the magnetic symmetry of the sites")
            label = "?"
            dataset = spglib.get_magnetic_symmetry_dataset(cell, symprec=symprec)
            if dataset is not None:
                mtype = spglib.get_magnetic_spacegroup_type(dataset.uni_number)
                if mtype is not None:
                    label = f"BNS {mtype.bns_number} (UNI {dataset.uni_number})"
            trs = np.asarray(sym["time_reversals"], dtype=bool)
    ops = [(np.array(w, dtype=int), np.array(t, dtype=float), bool(tr))
           for w, t, tr in zip(sym["rotations"], sym["translations"], trs)]
    return ops, label


# ----------------------------------------------------------------------------
# sites / representation in the model basis
# ----------------------------------------------------------------------------

def lattice_offset(dfrac, lattice, symprec):
    """Integer lattice vector L if dfrac (fractional) lies within symprec
    (Cartesian distance, spglib's tolerance) of L, else None."""
    lvec = np.round(dfrac)
    if np.linalg.norm((dfrac - lvec) @ np.asarray(lattice, dtype=float)) < symprec:
        return lvec.astype(int)
    return None


def group_sites(shells, lattice, symprec=1e-3):
    """Group shells into atomic sites by (species, position mod lattice).
    shells: list of dict(name, start, l, nspin, pos, species, frame).
    Returns site list [dict(species, pos, shells=[shell index], cells=[L])]
    and site index of every shell.  pos is reduced to [0, 1) for spglib; the
    shell pos is pos + L, L kept per shell for the Bloch phase."""
    sites, site_of = [], []
    for ish, sh in enumerate(shells):
        raw = np.asarray(sh["pos"], dtype=float)
        for isite, site in enumerate(sites):
            cell = lattice_offset(raw - site["pos"], lattice, symprec)
            if site["species"] == sh["species"] and cell is not None:
                site["shells"].append(ish)
                site["cells"].append(cell)
                site_of.append(isite)
                break
        else:
            pos = raw % 1.0
            sites.append(dict(species=sh["species"], pos=pos, shells=[ish],
                              cells=[np.round(raw - pos).astype(int)]))
            site_of.append(len(sites) - 1)
    return sites, site_of


def check_basis(shells, nwan):
    """Every orbital must belong to exactly one shell, all with the same nspin."""
    used = np.zeros(nwan, dtype=int)
    for sh in shells:
        size = sh["nspin"]*(2*sh["l"] + 1)
        used[sh["start"]:sh["start"] + size] += 1
    if (used != 1).any():
        missing = np.where(used == 0)[0]
        raise ValueError("every orbital must belong to a shell with a position; "
                         f"uncovered orbitals: {missing.tolist()}")
    if len({sh["nspin"] for sh in shells}) != 1:
        raise ValueError("shells mix spinless and spinful blocks")


def frame_basis(shells, nwan, orbital_axes=True):
    """S with columns = model basis states in the global frame
    (orbital: D^l(Q); spin: rotate_spinor(z, x), as applied by rotate_spin)."""
    from .spinor import rotate_spinor
    smat = np.eye(nwan, dtype=complex)
    for sh in shells:
        if sh.get("frame") is None:
            continue
        z, x = sh["frame"]
        norb = 2*sh["l"] + 1
        dorb = orbital_dmat(sh["l"], frame_matrix(z, x)) if orbital_axes else np.eye(norb)
        block = np.kron(dorb, rotate_spinor(z, x)) if sh["nspin"] == 2 else dorb
        ist = sh["start"]
        smat[ist:ist + block.shape[0], ist:ist + block.shape[0]] = block
    return smat


def op_representation(op, lattice, shells, sites, site_of, nwan, smat, symprec=1e-3):
    """U (model basis, without the k phase) and the lattice vector L of the
    source orbital of every column.  Raises ValueError if the basis is not closed."""
    wmat, tvec, trev = op
    rot = cart_rotation(wmat, lattice)
    nspin = shells[0]["nspin"]
    dspin = spin_dmat(rot) if nspin == 2 else None
    if trev and nspin == 2:
        dspin = dspin @ SPIN_T

    umat = np.zeros((nwan, nwan), dtype=complex)
    lcol = np.zeros((nwan, 3), dtype=int)
    dcache = {}
    cell = {ish: c for site in sites for ish, c in zip(site["shells"], site["cells"])}
    for isite, site in enumerate(sites):
        new = wmat @ site["pos"] + tvec
        target = None
        for jsite, other in enumerate(sites):
            lvec = lattice_offset(new - other["pos"], lattice, symprec)
            if other["species"] == site["species"] and lvec is not None:
                target = jsite
                break
        if target is None:
            raise ValueError(f"site {site['species']} {np.round(site['pos'], 5).tolist()} is mapped "
                             f"to {np.round(new % 1.0, 5).tolist()}, which has no shell")
        # pair shells by (l, order among shells of that l on the site)
        dst = {}
        for ish in sites[target]["shells"]:
            key = shells[ish]["l"]
            dst.setdefault(key, []).append(ish)
        seen = {}
        for ish in site["shells"]:
            sh = shells[ish]
            iorder = seen.get(sh["l"], 0)
            seen[sh["l"]] = iorder + 1
            if iorder >= len(dst.get(sh["l"], [])):
                raise ValueError(f"shell {sh['name']} (l={sh['l']}) has no partner on the image site")
            th = shells[dst[sh["l"]][iorder]]
            if sh["l"] not in dcache:
                dcache[sh["l"]] = orbital_dmat(sh["l"], rot)
            block = np.kron(dcache[sh["l"]], dspin) if nspin == 2 else dcache[sh["l"]]
            size = block.shape[0]
            umat[th["start"]:th["start"] + size, sh["start"]:sh["start"] + size] = block
            # W (s_a + c_a) + t = (s_b + c_b) + L + W c_a - c_b
            lcol[sh["start"]:sh["start"] + size] = lvec + wmat @ cell[ish] - cell[dst[sh["l"]][iorder]]

    if trev:
        umat = smat.conj().T @ umat @ smat.conj()
    else:
        umat = smat.conj().T @ umat @ smat
    if not np.allclose(umat @ umat.conj().T, np.eye(nwan), atol=1e-8):
        raise ValueError("internal error: representation matrix is not unitary")
    return umat, lcol


def check_model(model, lattice, shells, ops, kpts, orbital_axes=True, symprec=1e-3):
    """Run the check for every operation.  Returns one dict per op with
    eig_err, mat_err (eV, max over kpts) and the worst shell-pair block.
    symprec: Cartesian site-matching tolerance, the one given to spglib."""
    nwan = model.nwan
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    if len(kpts) == 0:
        raise ValueError("no k-points: an empty sample would report every operation as satisfied")
    check_basis(shells, nwan)
    sites, site_of = group_sites(shells, lattice, symprec=symprec)
    smat = frame_basis(shells, nwan, orbital_axes=orbital_axes)

    owner = np.empty(nwan, dtype=int)
    for ish, sh in enumerate(shells):
        owner[sh["start"]:sh["start"] + sh["nspin"]*(2*sh["l"] + 1)] = ish

    hks = [model.get_bulk_Hk(k) for k in kpts]
    eks = [np.linalg.eigvalsh(h) for h in hks]

    results = []
    for op in ops:
        wmat, tvec, trev = op
        umat, lcol = op_representation(op, lattice, shells, sites, site_of, nwan, smat, symprec=symprec)
        kmap = np.linalg.inv(wmat).T * (-1 if trev else 1)
        eig_err, mat_err = 0.0, 0.0
        block_err = np.zeros((len(shells), len(shells)))
        for hk, ek, kvec in zip(hks, eks, kpts):
            kp = kmap @ kvec
            hp = model.get_bulk_Hk(kp)
            uk = umat * np.exp(-2j*np.pi*(lcol @ kp))[None, :]
            diff = np.abs(hp - uk @ (hk.conj() if trev else hk) @ uk.conj().T)
            eig_err = max(eig_err, np.max(np.abs(np.linalg.eigvalsh(hp) - ek)))
            mat_err = max(mat_err, diff.max())
            np.maximum.at(block_err, (owner[:, None], owner[None, :]), diff)
        iw, jw = np.unravel_index(np.argmax(block_err), block_err.shape)
        name, axis = op_symbol(cart_rotation(wmat, lattice))
        results.append(dict(op=op, name=name, axis=axis, eig_err=eig_err, mat_err=mat_err,
                            worst=(shells[iw]["name"], shells[jw]["name"], block_err[iw, jw])))
    return results


def random_kpoints(nk, seed=0):
    return np.random.default_rng(seed).random((nk, 3))
