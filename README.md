# swept
Strongly correlated WanniEr Python Tools

## Testing

Run the serial core test suite from the repository root:

```bash
python -m unittest discover -s tests -v      # from the swept directory
```

To exercise MPI collectives as well (always with an outer timeout: a
regression in an error path can otherwise hang the job):

```bash
timeout -k 10 900 mpiexec -n 2 python -m unittest discover -s tests -v
```

The multi-rank error paths (`tests/test_mpi_faults.py`) run only from the
serial suite: each case is its own `mpiexec -n 2/3` job of
`tests/mpi_fault_worker.py`, killed as a process group after 60 s.  They are
skipped when there is no mpiexec or mpi4py, or the launcher cannot start a
job (e.g. sockets refused in a sandbox); a control job that starts but fails
is a test failure.

## Package and entry points

swept is a regular Python package (relative imports inside).  Put the
directory that *contains* `swept/` on the path to use it as a library:

```python
import sys; sys.path.insert(0, "/path/to")      # parent of swept/
from swept import run, lindhard
from swept.model import Model
```

The scripts `run.py` and `plot_fermi_surface.py` also run directly
(`python /path/to/swept/run.py ...`, `runpy.run_path`, or
`python -m swept.run ...` with the parent on PYTHONPATH): they load
themselves as part of the package.  Do not put the swept directory itself
on sys.path and import `model`, `run`, ... as top-level modules.

`Model` caches the stacked H(R); its own methods keep the cache valid, but
after changing `model.ham` in place call `model.reset_cache()`.

With mpi4py on Intel MPI on a single node, set `I_MPI_FABRICS=shm`.

## Input-file driver

```bash
mpirun -np N python /path/to/swept/run.py input.toml
```

Input is validated on rank 0 (types, finiteness, integer meshes and indices,
projections as sets without repeated orbitals) and any failure, including an
unexpected one, is raised on every rank.  Failures and the log:

- a result file that cannot be written stops every rank (`output error`);
- `swept_run.log` that cannot be opened stops every rank before the input is
  read (message on stderr, exit status 1);
- a later write/flush/close failure of the log only warns once on stderr; the
  log continues on stdout and the run goes on.

The hr file must contain finite numbers (NaN/inf is an input error naming R
and (m, n)); `Model` also refuses to build H(k) from a non-finite H(R).  As a
real-space diagnostic the log reports, for the model as read (after the WS
expansion), `max_R max_mn |H(R) - H(-R)^+|` and the bound
`sum_R max_mn |H(R) - H(-R)^+|` (R over the set closed under R -> -R) on
`max_mn |H(k) - H(k)^+|`; a warning is issued when the first exceeds
`[model] hermiticity_warn` (default 1e-4 eV).  It is not a test of H(k) or of
the Wannier fit, and nothing is symmetrised.

MPI: the distributed methods (bands, projections, DOS, Fermi-surface planes,
chi0 mesh) run the local work between two collective calls inside
`mpiutils.collective(...)`: a failure on one rank is raised on every rank as
`mpiutils.CollectiveError` (with a single rank the original exception
propagates unchanged), and `run.main` reports it and returns 1.  When run as
a script with several ranks, an exception that escapes `main` is printed and
the job is ended with `MPI_Abort`.  Not covered: a rank killed by a signal
(e.g. the OOM killer), a native crash, a failure inside an MPI call.

The pipeline is fixed: read hr -> add spin -> eshift -> magmom axis ->
rotate_spin -> SOC -> tasks (`udmft`, `bands`, `dos`, `symmetry`,
`fermi_surface`, `chi0`, run in the order written). `[shell] start` is the 0-based orbital index in the hr file.
Every run writes `swept_run.log` with the swept version, the resolved
orbital indices and a copy of the input. See `examples/co3sn2s2.toml`.

## DMFT local basis

`[task.udmft] local_basis = "eigen"` diagonalises the on-site block of the
correlated shell and writes `htmat.dat` = T2C^T for atom_d, with
T2C[a,i] = <complex_a | Eimp_i> = (c2r @ von)[a,i] (both axes spin-blocked).
For a nonmagnetic SOC start set `kramers = true`: H_loc is checked to be
time-reversal invariant within `kramers_tol` (eV), symmetrised, and every
Kramers pair is put in the gauge that diagonalises Mz = Lz + 2Sz (first
state +Mz, second its exact T partner), so solver flavours are reproducible.
Leave it off for a magnetic H_loc.

## Symmetry check

`[task.symmetry]` checks the final model against the space group of the
Wannier sites (spglib; with `magmom`, the magnetic space group), plus time
reversal for non-magnetic models.  Every shell needs `pos`, the Wannier
centre of the home cell as in the hr file (not reduced to [0, 1): the cell
offset enters the Bloch phase); shell `z`/`x` are taken as the local axes
of the wannier90 projection.  Sites are matched within `symprec`, a
Cartesian distance, the same tolerance spglib uses.  For each
operation g it reports, over random k-points,

- `eig_err` = max |E_n(gk) - E_n(k)|: basis independent;
- `mat_err` = max |H(gk) - U H(k) U^+|, with U built from the real-harmonic
  (wannier90 order) and spinor representation matrices.

`H-only` (spectrum symmetric, matrix not) means either the operation is
broken invisibly in the spectrum (T in a magnet with inversion), or the
basis description (shell order, pos, z/x, spin ordering) is wrong.  Hoppings
whose symmetry partner falls outside the hr R-vector set also show up
here unless the model is read with its `wsvec.dat` (below).

## Wigner-Seitz images (wsvec.dat)

`hr.dat` alone gives the `use_ws_distance = .false.` interpolation: every
H_mn(R) sits at the R chosen around the origin, not at the image R+T with
the shortest |R + T + tau_n - tau_m|, which breaks symmetries for pairs
whose bond centre is not at the origin.  With

```toml
[model]
wsvec = "seedname_wsvec.dat"     # collinear: also wsvec_dn
```

(or `Model.from_hr(hr, wsvec_file=...)`) each H_mn(R)/deg_R is split
equally over its N_mnR minimal-distance images R+T_j, as wannier90 does in
its own band interpolation.  The R set grows accordingly and `deg` becomes
1, so `to_hr` writes a plain hr that needs no wsvec any more.

## Fermi-surface cuts

```toml
[task.fermi_surface]
mu = 11.1062
nk = [241, 241]            # points along u, v
basis = "cart"             # plane vectors in 1/Angstrom (2 pi included); "frac" = reciprocal-lattice units
window = 0.2               # keep bands inside mu +- window somewhere on the plane
project = ["U_5f", "Te_5p"]
output = "fs.npz"

[[task.fermi_surface.plane]]   # one table per plane
name = "kz0"
origin = [0, 0, 0]
u = [1.524, 0, 0]          # k = origin + s u + t v, s, t in [-1, 1]
v = [0, 1.032, 0]
```

`fs.npz` holds, per plane `<name>`, the grid `_s`, `_t`, the kept bands
`_bands` (1-based), energies `_E` (E - mu; [s, t, band]), projection weights
`_W` ([proj, s, t, band]), the Cartesian `_origin/_u/_v` and `_bz`
(<= 0 exactly inside the first Brillouin zone: the max of 2k.G - |G|^2 over
the 14 Voronoi-relevant G from a Selling reduction, so only its sign and
zero level are meaningful).  Plot with
`python plot_fermi_surface.py fs.npz -o fs.pdf [-e E]`: contours coloured by
the first three projections (red, blue, green), dashed BZ boundary, axes in
1/Angstrom along u and perpendicular to it.

## Bare susceptibility chi0, site resolved

```toml
[task.chi0]
mesh = [24, 24, 24]        # Gamma-centred; q runs over the same mesh
kT = 0.01                  # eV, Fermi smearing
mu = 11.1062
window = 0.5               # keep bands inside mu +- window somewhere on the mesh
vertex = "point_site"      # or "cell"
total = true               # channel of all orbitals
project = ["U_5f"]         # further channels
window_check = 2.0         # optional: redo Gamma and the peak with window*2.0, log the change
kblock = 4096              # k points per block (memory)
output = "chi0.npz"
```

For orbital set P (orbital a at tau_a = tau0_s + L_a, site s, integer cell
offset L_a) and the mesh representative q = n/N in [0,1)^3:

    Ot^s_nm(k,q) = sum_{a in P on s} e^{-2 pi i q.L_a} U*_an(k) U_am(k+q)
    chi_st(q)    = 1/Nk sum_k sum_nm Ot^s_nm Ot^t_nm* F_nm,   F = -(f(E_n(k)) - f(E_m(k+q)))/(E_n(k) - E_m(k+q))
    chi(Q)       = sum_st e^{-2 pi i Q.tau0_s} chi_st(q) e^{+2 pi i Q.tau0_t},   Q = q + G

`chi_st` is Hermitian positive semi-definite, chi(Q) is real, >= 0 and not
periodic in G (intra-cell interference).  Sites and offsets come from the
shell `pos` grouped by `symmetry.group_sites`, so `vertex = "point_site"`
needs a `pos` on every shell a channel touches (an error otherwise).
`vertex = "cell"` puts all of P on one site at tau = 0: the lattice-gauge
cell-density correlation, which in general is not covariant under the
space-group operations of the crystal.

- No spatial symmetry is used (the earlier reduction was unreliable,
  REVIEW_2026-10-16 #2/#3; the options `symmetry`, `sym_tol` are rejected).
  Only the exact static relation chi_st(-q) = conj(chi_st(q)) halves the
  work; at self-paired q the matrix must be real (checked).
- F is evaluated as sinhc(u - v)/(4 kT cosh u cosh v), u = E/2kT: exactly
  symmetric in its two energies, no degeneracy threshold, no overflow.
- Approximations: point-site density vertex (no Wannier form factors
  <w_a|e^{-iQ.r}|w_b>); bands outside mu +- window dropped (check with
  `window_check`, a diagnostic at Gamma and the peak only).  This is a static
  density response, not an SOC magnetic-moment tensor.
- Units 1/eV per model unit cell.  spin = "none": B of one spin copy;
  spin-degenerate non-interacting chi_nn = 2B, chi_SzSz = B/2 (hbar = 1).
  Spinful bases: response of the chosen spin-orbital set, no factor applied.

`chi0.npz` (format "swept-chi0", version 2) holds the metadata above and,
per channel c, `chi_<c>` [iq, s, t] (complex), `tau0_<c>`, `sites_<c>`,
`orbitals_<c>`, `siteidx_<c>`, `offsets_<c>`; `qfrac[iq]` are the
representatives.  `lindhard.chi_physical_npz(np.load("chi0.npz"), "total", Q)`
returns chi(Q) for fractional Q on the mesh plus any integer G.
