#!/usr/bin/env python
'''
SWEPT input-file driver.

    mpirun -np N python /path/to/swept/run.py input.toml

Pipeline (fixed order):
    read hr -> add spin -> eshift -> magmom axis -> rotate_spin -> SOC -> tasks

Index convention:
    [shell] start counts orbitals in the hr file, 0-based.  For
    spin = "doubled"/"collinear" it is mapped to 2*start in the spinful
    (interleaved up/dn) basis; for "spinor"/"none" it is used as is.
See examples/co3sn2s2.toml for a complete input.
'''

import sys
import datetime
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import tomllib
except ModuleNotFoundError:          # Python < 3.11
    import tomli as tomllib

import numpy as np
import mpiutils as mpi
from model import Model
from kpoint import Kpoint


ORBITAL_NAMES = {
    0: ["s"],
    1: ["pz", "px", "py"],
    2: ["dz2", "dxz", "dyz", "dx2-y2", "dxy"],
    3: ["fz3", "fxz2", "fyz2", "fz(x2-y2)", "fxyz", "fx(x2-3y2)", "fy(3x2-y2)"],
}
SPIN_MODES = ("none", "doubled", "collinear", "spinor")
TASKS = ("udmft", "bands", "dos")


class InputError(Exception):
    pass


# ----------------------------------------------------------------------------
# input parsing / validation (pure, no model needed)
# ----------------------------------------------------------------------------

def _check_keys(table, where, required=(), optional=()):
    if not isinstance(table, dict):
        raise InputError(f"{where} must be a table")
    unknown = set(table) - set(required) - set(optional)
    if unknown:
        raise InputError(f"unknown key(s) in {where}: {sorted(unknown)}")
    missing = [k for k in required if k not in table]
    if missing:
        raise InputError(f"missing key(s) in {where}: {missing}")


def _vec3(value, where):
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise InputError(f"{where} must be a 3-vector, got {value!r}")
    if np.linalg.norm(arr) < 1e-12:
        raise InputError(f"{where} must be non-zero")
    return arr


def validate(cfg):
    """Check the structure of the parsed TOML; returns cfg unchanged."""
    _check_keys(cfg, "input", required=("model", "lattice"),
                optional=("shell", "soc", "projection", "kpath", "task"))

    model = cfg["model"]
    _check_keys(model, "[model]", required=("hr_file", "spin"),
                optional=("hr_dn", "eshift", "magmom_axis", "allow_extra_soc"))
    spin = model["spin"]
    if spin not in SPIN_MODES:
        raise InputError(f"[model] spin must be one of {SPIN_MODES}, got {spin!r}")
    if (spin == "collinear") != ("hr_dn" in model):
        raise InputError("[model] hr_dn is required for spin = 'collinear' and only allowed there")
    if "magmom_axis" in model:
        if spin != "collinear":
            raise InputError("[model] magmom_axis is only meaningful for spin = 'collinear'")
        _vec3(model["magmom_axis"], "[model] magmom_axis")
    if "allow_extra_soc" in model and spin != "spinor":
        raise InputError("[model] allow_extra_soc is only meaningful for spin = 'spinor'")

    _check_keys(cfg["lattice"], "[lattice]", required=("rvec",))
    rvec = np.asarray(cfg["lattice"]["rvec"], dtype=float)
    if rvec.shape != (3, 3) or abs(np.linalg.det(rvec)) < 1e-8:
        raise InputError("[lattice] rvec must be a non-singular 3x3 matrix (rows = a1, a2, a3)")

    shells_cfg = cfg.get("shell", {})
    if not isinstance(shells_cfg, dict):
        raise InputError("[shell] must be a table: name = { start = ..., l = ... }")
    for sname, shell in shells_cfg.items():
        where = f"[shell] {sname}"
        _check_keys(shell, where, required=("start", "l"), optional=("z", "x"))
        if shell["l"] not in ORBITAL_NAMES:
            raise InputError(f"{where}: l must be 0-3")
        if ("z" in shell) != ("x" in shell):
            raise InputError(f"{where}: give both z and x, or neither")
        if "z" in shell:
            if spin == "none":
                raise InputError(f"{where}: z/x (spin rotation) needs a spinful model")
            z = _vec3(shell["z"], where + " z")
            x = _vec3(shell["x"], where + " x")
            if abs(np.dot(z, x)) > 1e-6 * np.linalg.norm(z) * np.linalg.norm(x):
                raise InputError(f"{where}: z and x must be orthogonal")

    soc = cfg.get("soc", {})
    if soc:
        if spin == "none":
            raise InputError("[soc] needs a spinful model (spin != 'none')")
        if spin == "spinor" and not model.get("allow_extra_soc", False):
            raise InputError("[soc] with spin = 'spinor': the hr already contains SOC; "
                             "set [model] allow_extra_soc = true to add more on purpose")
        shells = cfg.get("shell", {})
        for name, zeta in soc.items():
            if name not in shells:
                raise InputError(f"[soc] unknown shell {name!r}")
            if shells[name]["l"] == 0:
                raise InputError(f"[soc] shell {name!r} has l = 0, L.S vanishes")
            if not isinstance(zeta, (int, float)):
                raise InputError(f"[soc] {name} must be a number (zeta in eV)")

    for pname, proj in cfg.get("projection", {}).items():
        where = f"[projection] {pname}"
        if "index" in proj:
            _check_keys(proj, where, required=("index",))
        else:
            _check_keys(proj, where, required=("shells",), optional=("orbitals", "spin"))
            if "spin" in proj and proj["spin"] not in ("up", "dn"):
                raise InputError(f"{where}: spin must be 'up' or 'dn'")
            if "spin" in proj and spin == "none":
                raise InputError(f"{where}: spin selection needs a spinful model")

    if "kpath" in cfg:
        kpath = cfg["kpath"]
        _check_keys(kpath, "[kpath]", required=("points", "labels", "nk"))
        if np.asarray(kpath["points"], dtype=float).shape[1:] != (3,) or len(kpath["points"]) < 2:
            raise InputError("[kpath] points must be a list of >= 2 fractional 3-vectors")
        if len(kpath["labels"]) != len(kpath["points"]):
            raise InputError("[kpath] labels and points must have the same length")

    for tname, task in cfg.get("task", {}).items():
        if tname not in TASKS:
            raise InputError(f"unknown task [task.{tname}], available: {TASKS}")
        where = f"[task.{tname}]"
        if tname == "udmft":
            _check_keys(task, where, required=("mesh", "ntot", "shells"),
                        optional=("mu", "local_basis", "save_mz", "check_tol"))
            if task.get("local_basis", "eigen") not in ("eigen", "none"):
                raise InputError(f"{where} local_basis must be 'eigen' or 'none'")
        elif tname == "bands":
            _check_keys(task, where, optional=("output", "mu", "project"))
            if "kpath" not in cfg:
                raise InputError(f"{where} needs a [kpath] section")
        elif tname == "dos":
            _check_keys(task, where, required=("mesh", "emin", "emax", "ne", "sigma"),
                        optional=("mu", "project", "output"))
        if "mesh" in task and (len(task["mesh"]) != 3 or min(task["mesh"]) < 1):
            raise InputError(f"{where} mesh must be three positive integers")
        for pname in task.get("project", []):
            if pname not in cfg.get("projection", {}):
                raise InputError(f"{where} unknown projection {pname!r}")
    return cfg


def load_input(path):
    """Read and validate on rank 0, broadcast to all ranks."""
    cfg, err = None, None
    if mpi.rank0:
        try:
            with open(path, "rb") as f:
                cfg = validate(tomllib.load(f))
        except (OSError, tomllib.TOMLDecodeError, InputError, ValueError, TypeError) as exc:
            err = f"{type(exc).__name__}: {exc}"
    cfg, err = mpi.bcast((cfg, err), root=0)
    if err is not None:
        raise InputError(err)
    return cfg


# ----------------------------------------------------------------------------
# index resolution (needs nwan of the hr file)
# ----------------------------------------------------------------------------

def resolve_shells(cfg, nwan_hr):
    """Map every [shell] entry to the final (possibly spin-doubled) basis.

    Returns {name: dict(start, size, l, nspin, z, x)} with start/size in the
    final basis."""
    spin = cfg["model"]["spin"]
    factor = 2 if spin in ("doubled", "collinear") else 1
    nspin = 1 if spin == "none" else 2
    nwan = nwan_hr * factor

    shells = {}
    used = np.zeros(nwan, dtype=bool)
    for name, shell in cfg.get("shell", {}).items():
        start = factor * shell["start"]
        size = nspin * (2*shell["l"] + 1)
        if shell["start"] < 0 or start + size > nwan:
            raise InputError(f"shell {name!r}: orbitals [{start}:{start+size}] "
                             f"exceed the model size {nwan}")
        if spin == "spinor" and start % 2 != 0:
            raise InputError(f"shell {name!r}: start must be even for an interleaved spinor hr")
        if used[start:start+size].any():
            raise InputError(f"shell {name!r} overlaps another shell")
        used[start:start+size] = True
        shells[name] = dict(start=start, size=size, l=shell["l"], nspin=nspin,
                                     z=shell.get("z"), x=shell.get("x"))
    return shells


def projection_indices(proj, shells, nwan):
    """Orbital indices (final basis) of one [projection] entry."""
    if "index" in proj:
        idx = np.asarray(proj["index"], dtype=int)
        if idx.size == 0 or idx.min() < 0 or idx.max() >= nwan:
            raise InputError(f"projection index {proj['index']} out of range [0, {nwan})")
        return idx

    idx = []
    for sname in proj["shells"]:
        if sname not in shells:
            raise InputError(f"projection: unknown shell {sname!r}")
        sh = shells[sname]
        names = ORBITAL_NAMES[sh["l"]]
        orbs = proj.get("orbitals", names)
        for orb in orbs:
            if orb not in names:
                raise InputError(f"projection: orbital {orb!r} not in shell {sname!r} (l={sh['l']}: {names})")
            iorb = names.index(orb)
            if sh["nspin"] == 1:
                idx.append(sh["start"] + iorb)
            else:
                spins = {"up": [0], "dn": [1]}.get(proj.get("spin"), [0, 1])
                idx.extend(sh["start"] + 2*iorb + s for s in spins)
    return np.array(sorted(idx), dtype=int)


# ----------------------------------------------------------------------------
# model construction
# ----------------------------------------------------------------------------

def build_model(cfg, log):
    mcfg = cfg["model"]
    spin = mcfg["spin"]

    tbmodel = Model.from_hr(mcfg["hr_file"])
    nwan_hr = tbmodel.nwan
    log(f"read {mcfg['hr_file']}: nwan = {nwan_hr}, nrpt = {tbmodel.nrpt}")

    if spin == "doubled":
        tbmodel.addspinbasis()
        log("spin doubled (non-magnetic): nwan -> %d" % tbmodel.nwan)
    elif spin == "collinear":
        model_dn = Model.from_hr(mcfg["hr_dn"])
        if model_dn.nwan != nwan_hr or set(model_dn.ham) != set(tbmodel.ham):
            raise InputError("hr_file and hr_dn differ in nwan or R vectors")
        tbmodel.addspinbasis(hown=model_dn.ham)
        log(f"collinear up/dn from {mcfg['hr_file']} / {mcfg['hr_dn']}: nwan -> {tbmodel.nwan}")

    shells = resolve_shells(cfg, nwan_hr)

    if "eshift" in mcfg:
        tbmodel.shift_onsite(eshift=mcfg["eshift"])
        log(f"on-site shift {mcfg['eshift']} eV")

    if "magmom_axis" in mcfg:
        tbmodel.set_spin_axis(mcfg["magmom_axis"])
        log(f"collinear spin axis {mcfg['magmom_axis']} -> quantised along global z")

    rot = [(name, sh) for name, sh in shells.items() if sh["z"] is not None]
    if rot:
        tbmodel.rotate_spin(lnum=[sh["l"] for _, sh in rot],
                            zxmat=np.array([[sh["z"], sh["x"]] for _, sh in rot], dtype=float),
                            index=[sh["start"] for _, sh in rot])
        log("spin rotated to local frames: " + ", ".join(name for name, _ in rot))

    for name, zeta in cfg.get("soc", {}).items():
        sh = shells[name]
        tbmodel.soc(asoc=zeta, lnum=sh["l"], sidx=sh["start"])
        frame = "local" if sh["z"] is not None else "global"
        log(f"SOC zeta = {zeta} eV on {name} ({frame} frame)")
        if sh["z"] is None:
            log(f"  note: {name} has no z/x; SOC assumes its orbitals are projected on global axes")

    log("shells (final basis):")
    for name, sh in shells.items():
        log(f"  {name:12s} l={sh['l']}  orbitals [{sh['start']}:{sh['start']+sh['size']}]")
    return tbmodel, shells


def build_projections(cfg, names, shells, nwan, log):
    plists = []
    for pname in names:
        idx = projection_indices(cfg["projection"][pname], shells, nwan)
        log(f"  projection {pname}: {idx.tolist()}")
        plists.append(idx)
    return plists


def reciprocal(cfg):
    rvec = np.asarray(cfg["lattice"]["rvec"], dtype=float)
    return 2*np.pi*np.linalg.inv(rvec).T


# ----------------------------------------------------------------------------
# tasks
# ----------------------------------------------------------------------------

def task_udmft(cfg, task, tbmodel, shells, log):
    from udmft import local_eigenbasis, onsite_block_mismatch, dump_local_basis, dump_dmft

    group = [shells[name] if name in shells else None for name in task["shells"]]
    if None in group:
        raise InputError(f"[task.udmft] unknown shell in {task['shells']}")
    l, norb, start = group[0]["l"], group[0]["size"], group[0]["start"]
    for ish, sh in enumerate(group):
        if sh["l"] != l or sh["start"] != start + ish*norb:
            raise InputError("[task.udmft] shells must have the same l and be consecutive in the basis")
    natm = len(group)
    nspin = group[0]["nspin"]
    mu = task.get("mu", 0.0)

    von = None
    if task.get("local_basis", "eigen") == "eigen":
        if nspin != 2:
            raise InputError("[task.udmft] local_basis = 'eigen' needs a spinful model")
        h00 = tbmodel.ham[(0, 0, 0)]
        mismatch = onsite_block_mismatch(h00, norb, natm, start=start)
        log(f"udmft: max |H_loc(i) - H_loc(1)| = {mismatch:.3e} eV")
        if mismatch > task.get("check_tol", 1e-3):
            raise InputError(f"[task.udmft] correlated shells are not equivalent "
                             f"(mismatch {mismatch:.3e} eV > check_tol); one von cannot serve all")
        eon, von, htmat, mz = local_eigenbasis(h00[start:start+norb, start:start+norb], ll=l)
        log("udmft: Eimp = " + np.array2string(eon, precision=5, max_line_width=200))
        log("udmft: Mz per solver flavour = " + np.array2string(np.diag(mz).real, precision=4, max_line_width=200))

    kpts = Kpoint.gen_uniform_kpoints(task["mesh"])
    log(f"udmft: diagonalising {kpts.nktot} k-points")
    eigs, vecs = tbmodel.get_bulk_eigens(kpts.kfrac)

    if mpi.rank0:
        if von is not None:
            dump_local_basis(eon, von, htmat)
            if task.get("save_mz", False):
                np.save("Mz_eimp_blocked.npy", mz)
        dump_dmft(eigs, vecs, mu, ntot=task["ntot"], norb=norb, natm=natm,
                  lnum=l, nspin=nspin, von=von, start=start)
    log("udmft: wrote eig.dat, udmft.dat" + (", eon.dat, von.dat, htmat.dat" if von is not None else ""))


def task_bands(cfg, task, tbmodel, shells, log):
    from bands import dump_bands, dump_projs_multi, dump_ticks

    kpath = cfg["kpath"]
    bvec = reciprocal(cfg)
    kpts = Kpoint.from_high_symmetry_points(khsym_frac=kpath["points"], bzvec=bvec, kwant=kpath["nk"])
    kpth = Kpoint.k_length_accumulate(kpts.kreal)
    ticks = Kpoint.k_length_accumulate(np.array([np.dot(k, bvec) for k in kpath["points"]]))
    output = task.get("output", "bands.dat")
    mu = task.get("mu", 0.0)
    names = task.get("project", [])
    log(f"bands: {kpts.nktot} k-points along {'-'.join(kpath['labels'])}")

    if names:
        plists = build_projections(cfg, names, shells, tbmodel.nwan, log)
        eigs, projs = tbmodel.get_bulk_eprojs_multi(kpts.kfrac, plists)
    else:
        eigs = tbmodel.get_bulk_eigvals(kpts.kfrac)

    tick_file = str(Path(output).with_suffix("")) + "_ticks.dat"
    if mpi.rank0:
        if names:
            dump_projs_multi(kpth, eigs, projs, names, fname=output, mu=mu)
        else:
            dump_bands(kpth, eigs, fname=output, mu=mu)
        dump_ticks(ticks, kpath["labels"], fname=tick_file)
    log(f"bands: wrote {output}, {tick_file}")


def task_dos(cfg, task, tbmodel, shells, log):
    from dos import dump_total_dos

    names = task.get("project", [])
    plists = build_projections(cfg, names, shells, tbmodel.nwan, log) if names else []
    kpts = Kpoint.gen_uniform_kpoints(task["mesh"])
    output = task.get("output", "dos.dat")
    log(f"dos: {kpts.nktot} k-points")

    omega, tdos, pdos = tbmodel.get_proj_dos(kpts.kfrac, proj=plists, emin=task["emin"], emax=task["emax"],
                                             ne=task["ne"], mu=task.get("mu", 0.0), sigma=task["sigma"])
    if mpi.rank0:
        dump_total_dos(omega, tdos, pdos, output, names=names)
    log(f"dos: wrote {output}")


TASK_FUNCS = {"udmft": task_udmft, "bands": task_bands, "dos": task_dos}


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def _git_version():
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + (" (with uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: [mpirun -np N] python run.py input.toml")
        return 2
    path = argv[0]

    flog = open("swept_run.log", "w") if mpi.rank0 else None

    def log(msg):
        if mpi.rank0:
            print(msg); sys.stdout.flush()
            flog.write(msg + "\n"); flog.flush()

    try:
        log(f"SWEPT run  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  "
            f"swept {_git_version()}  nproc = {mpi.size}")
        log(f"input: {Path(path).resolve()}")
        cfg = load_input(path)
        if mpi.rank0:
            flog.write("----- input -----\n" + Path(path).read_text() + "\n----- end input -----\n")

        tbmodel, shells = build_model(cfg, log)
        for tname, task in cfg.get("task", {}).items():
            log(f"=== task {tname} ===")
            TASK_FUNCS[tname](cfg, task, tbmodel, shells, log)
        log("done")
        return 0
    except InputError as exc:
        log(f"input error: {exc}")
        return 1
    finally:
        if flog is not None:
            flog.close()


if __name__ == "__main__":
    sys.exit(main())
