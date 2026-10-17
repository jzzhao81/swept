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
import math
import datetime
import subprocess
import warnings
from pathlib import Path

if __package__ in (None, ""):
    # run as a script (python /path/to/swept/run.py, runpy.run_path, mpirun ... run.py):
    # import it as part of the package so the relative imports below work
    import importlib
    _pkgdir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_pkgdir.parent))
    __package__ = _pkgdir.name
    importlib.import_module(__package__)

try:
    import tomllib
except ModuleNotFoundError:          # Python < 3.11
    import tomli as tomllib

import numpy as np
from . import mpiutils as mpi
from .model import Model
from .kpoint import Kpoint


ORBITAL_NAMES = {
    0: ["s"],
    1: ["pz", "px", "py"],
    2: ["dz2", "dxz", "dyz", "dx2-y2", "dxy"],
    3: ["fz3", "fxz2", "fyz2", "fz(x2-y2)", "fxyz", "fx(x2-3y2)", "fy(3x2-y2)"],
}
SPIN_MODES = ("none", "doubled", "collinear", "spinor")
TASKS = ("udmft", "bands", "dos", "symmetry", "fermi_surface", "chi0")
HERMITICITY_WARN = 1e-4         # eV, default of [model] hermiticity_warn


class InputError(Exception):
    pass


class OutputError(Exception):
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


def _is_int(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _is_number(x):
    """Finite real number (bool and NaN/inf rejected)."""
    return (isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)
            and math.isfinite(x))


def _has_bool(v):
    return isinstance(v, bool) or (isinstance(v, (list, tuple)) and any(_has_bool(x) for x in v))


def _finite_array(value, where, shape=None):
    try:
        if _has_bool(value) or isinstance(value, (str, dict)):
            raise TypeError
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise InputError(f"{where} must be numeric, got {value!r}") from None
    if shape is not None and arr.shape != shape:
        raise InputError(f"{where} must have shape {shape}, got {value!r}")
    if not np.all(np.isfinite(arr)):
        raise InputError(f"{where} must be finite, got {value!r}")
    return arr


def _int_list(value, where, length=None, minimum=None):
    if not isinstance(value, list) or not all(_is_int(v) for v in value):
        raise InputError(f"{where} must be a list of integers, got {value!r}")
    if length is not None and len(value) != length:
        raise InputError(f"{where} must have {length} entries, got {value!r}")
    if minimum is not None and any(v < minimum for v in value):
        raise InputError(f"{where} entries must be >= {minimum}, got {value!r}")
    return value


def _unique_list(value, where, kind=str, allow_empty=False):
    if (not isinstance(value, list) or (not value and not allow_empty)
            or not all(isinstance(v, kind) and not isinstance(v, bool) for v in value)):
        raise InputError(f"{where} must be a {'' if allow_empty else 'non-empty '}list of {kind.__name__}, "
                         f"got {value!r}")
    dup = sorted({v for v in value if value.count(v) > 1})
    if dup:
        raise InputError(f"{where} has duplicate entries {dup}")
    return value


def _vec3(value, where):
    arr = _finite_array(value, where, (3,))
    if np.linalg.norm(arr) < 1e-12:
        raise InputError(f"{where} must be non-zero")
    return arr


def validate(cfg):
    """Check the structure of the parsed TOML; returns cfg unchanged."""
    _check_keys(cfg, "input", required=("model", "lattice"),
                optional=("shell", "soc", "projection", "kpath", "task"))

    for key in ("shell", "soc", "projection", "kpath", "task"):
        if key in cfg and not isinstance(cfg[key], dict):
            raise InputError(f"[{key}] must be a table, got {type(cfg[key]).__name__}")

    model = cfg["model"]
    _check_keys(model, "[model]", required=("hr_file", "spin"),
                optional=("hr_dn", "wsvec", "wsvec_dn", "eshift", "magmom_axis", "allow_extra_soc",
                          "hermiticity_warn"))
    spin = model["spin"]
    if spin not in SPIN_MODES:
        raise InputError(f"[model] spin must be one of {SPIN_MODES}, got {spin!r}")
    if (spin == "collinear") != ("hr_dn" in model):
        raise InputError("[model] hr_dn is required for spin = 'collinear' and only allowed there")
    if "wsvec_dn" in model and spin != "collinear":
        raise InputError("[model] wsvec_dn is only meaningful for spin = 'collinear'")
    if spin == "collinear" and ("wsvec" in model) != ("wsvec_dn" in model):
        raise InputError("[model] give both wsvec and wsvec_dn, or neither")
    if "magmom_axis" in model:
        if spin != "collinear":
            raise InputError("[model] magmom_axis is only meaningful for spin = 'collinear'")
        _vec3(model["magmom_axis"], "[model] magmom_axis")
    if "allow_extra_soc" in model and spin != "spinor":
        raise InputError("[model] allow_extra_soc is only meaningful for spin = 'spinor'")
    for key in ("hr_file", "hr_dn", "wsvec", "wsvec_dn"):
        if key in model and not isinstance(model[key], str):
            raise InputError(f"[model] {key} must be a file name")
    if "eshift" in model:
        _finite_array(model["eshift"], "[model] eshift")
    if "hermiticity_warn" in model and not (_is_number(model["hermiticity_warn"])
                                           and model["hermiticity_warn"] > 0):
        raise InputError(f"[model] hermiticity_warn must be a positive number (eV), "
                         f"got {model['hermiticity_warn']!r}")

    _check_keys(cfg["lattice"], "[lattice]", required=("rvec",))
    rvec = _finite_array(cfg["lattice"]["rvec"], "[lattice] rvec", (3, 3))
    if abs(np.linalg.det(rvec)) < 1e-8:
        raise InputError("[lattice] rvec must be a non-singular 3x3 matrix (rows = a1, a2, a3)")

    shells_cfg = cfg.get("shell", {})
    if not isinstance(shells_cfg, dict):
        raise InputError("[shell] must be a table: name = { start = ..., l = ... }")
    for sname, shell in shells_cfg.items():
        where = f"[shell] {sname}"
        _check_keys(shell, where, required=("start", "l"), optional=("z", "x", "pos"))
        if not _is_int(shell["l"]) or shell["l"] not in ORBITAL_NAMES:
            raise InputError(f"{where}: l must be an integer 0-3")
        if not _is_int(shell["start"]) or shell["start"] < 0:
            raise InputError(f"{where}: start must be a non-negative integer")
        if ("z" in shell) != ("x" in shell):
            raise InputError(f"{where}: give both z and x, or neither")
        if "pos" in shell:
            _finite_array(shell["pos"], f"{where}: pos (fractional)", (3,))
        if "z" in shell:
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
            if not _is_number(zeta):
                raise InputError(f"[soc] {name} must be a number (zeta in eV)")

    for pname, proj in cfg.get("projection", {}).items():
        where = f"[projection] {pname}"
        if not isinstance(proj, dict):
            raise InputError(f"{where} must be a table")
        if "index" in proj:
            _check_keys(proj, where, required=("index",))
            # an orbital set: integers (no silent float truncation), no repeats
            _unique_list(proj["index"], f"{where} index", kind=int)
        else:
            _check_keys(proj, where, required=("shells",), optional=("orbitals", "spin"))
            _unique_list(proj["shells"], f"{where} shells")
            if "orbitals" in proj:
                _unique_list(proj["orbitals"], f"{where} orbitals")
            if "spin" in proj and proj["spin"] not in ("up", "dn"):
                raise InputError(f"{where}: spin must be 'up' or 'dn'")
            if "spin" in proj and spin == "none":
                raise InputError(f"{where}: spin selection needs a spinful model")

    if "kpath" in cfg:
        kpath = cfg["kpath"]
        _check_keys(kpath, "[kpath]", required=("points", "labels", "nk"))
        points = _finite_array(kpath["points"], "[kpath] points")
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
            raise InputError("[kpath] points must be a list of >= 2 fractional 3-vectors")
        if not _is_int(kpath["nk"]) or kpath["nk"] < 1:
            raise InputError("[kpath] nk must be a positive integer")
        if not isinstance(kpath["labels"], list) or not all(isinstance(x, str) for x in kpath["labels"]):
            raise InputError("[kpath] labels must be a list of strings")
        if len(kpath["labels"]) != len(kpath["points"]):
            raise InputError("[kpath] labels and points must have the same length")

    for tname, task in cfg.get("task", {}).items():
        if tname not in TASKS:
            raise InputError(f"unknown task [task.{tname}], available: {TASKS}")
        where = f"[task.{tname}]"
        if tname == "udmft":
            _check_keys(task, where, required=("mesh", "ntot", "shells"),
                        optional=("mu", "local_basis", "save_mz", "check_tol", "kramers", "kramers_tol"))
            if not _is_number(task["ntot"]) or task["ntot"] < 0:
                raise InputError(f"{where} ntot must be a non-negative number")
            if task.get("local_basis", "eigen") not in ("eigen", "none"):
                raise InputError(f"{where} local_basis must be 'eigen' or 'none'")
            if not isinstance(task.get("kramers", False), bool):
                raise InputError(f"{where} kramers must be true or false")
        elif tname == "bands":
            _check_keys(task, where, optional=("output", "mu", "project"))
            if "kpath" not in cfg:
                raise InputError(f"{where} needs a [kpath] section")
        elif tname == "dos":
            _check_keys(task, where, required=("mesh", "emin", "emax", "ne", "sigma"),
                        optional=("mu", "project", "output"))
            if not (_is_number(task["emin"]) and _is_number(task["emax"])) or task["emin"] >= task["emax"]:
                raise InputError(f"{where} emin < emax must be finite numbers")
            if not _is_int(task["ne"]) or task["ne"] < 2:
                raise InputError(f"{where} ne must be an integer >= 2")
        elif tname == "symmetry":
            _check_keys(task, where, optional=("nk", "tol", "symprec", "magmom",
                                               "time_reversal", "orbital_axes"))
            nopos = [name for name, sh in shells_cfg.items() if "pos" not in sh]
            if not shells_cfg or nopos:
                raise InputError(f"{where} needs pos = [x, y, z] (fractional) on every shell; missing: {nopos}")
            if task.get("orbital_axes", "shell") not in ("shell", "global"):
                raise InputError(f"{where} orbital_axes must be 'shell' or 'global'")
            nk = task.get("nk", 8)
            if isinstance(nk, bool) or not isinstance(nk, int) or nk < 1:
                raise InputError(f"{where} nk must be a positive integer")
            for name, mom in task.get("magmom", {}).items():
                if name not in shells_cfg:
                    raise InputError(f"{where} magmom: unknown shell {name!r}")
                _finite_array(mom, f"{where} magmom {name} (Cartesian)", (3,))
        elif tname == "fermi_surface":
            _check_keys(task, where, required=("plane",),
                        optional=("mu", "nk", "basis", "window", "project", "output"))
            if task.get("basis", "frac") not in ("frac", "cart"):
                raise InputError(f"{where} basis must be 'frac' or 'cart'")
            nk = task.get("nk", [201, 201])
            _int_list(nk, f"{where} nk", length=2, minimum=2)
            planes = task["plane"]
            if not isinstance(planes, list) or not planes:
                raise InputError(f"{where} needs at least one [[task.fermi_surface.plane]]")
            names = []
            for plane in planes:
                _check_keys(plane, f"{where} plane", required=("name", "origin", "u", "v"))
                names.append(plane["name"])
                _finite_array(plane["origin"], f"{where} plane {plane['name']!r} origin", (3,))
                u = _vec3(plane["u"], f"{where} plane {plane['name']!r} u")
                v = _vec3(plane["v"], f"{where} plane {plane['name']!r} v")
                if np.linalg.norm(np.cross(u, v)) < 1e-8*np.linalg.norm(u)*np.linalg.norm(v):
                    raise InputError(f"{where} plane {plane['name']!r}: u and v must not be parallel")
            if len(set(names)) != len(names):
                raise InputError(f"{where} plane names must be unique")
        elif tname == "chi0":
            removed = [k for k in ("symmetry", "sym_tol") if k in task]
            if removed:
                raise InputError(f"{where} {removed} removed: the spatial q reduction was unreliable "
                                 "(REVIEW_2026-10-16 #2/#3); only the exact q/-q pairing is used")
            _check_keys(task, where, required=("mesh", "kT"),
                        optional=("mu", "window", "vertex", "total", "project", "symprec",
                                  "window_check", "kblock", "output"))
            if task.get("vertex", "point_site") not in ("point_site", "cell"):
                raise InputError(f"{where} vertex must be 'point_site' or 'cell'")
            if not isinstance(task.get("total", True), bool):
                raise InputError(f"{where} total must be true or false")
            if not task.get("total", True) and not task.get("project"):
                raise InputError(f"{where} no channel: total = false and project is empty")
            if "window_check" in task and (not _is_number(task["window_check"]) or task["window_check"] <= 1):
                raise InputError(f"{where} window_check must be a factor > 1")
            kblock = task.get("kblock", 4096)
            if not _is_int(kblock) or kblock < 1:
                raise InputError(f"{where} kblock must be a positive integer")
            # channel names become npz keys (chi_<name>, ...): they must be unique
            chans = (["total"] if task.get("total", True) else []) + list(task.get("project", []))
            dup = sorted({c for c in chans if chans.count(c) > 1})
            if dup:
                raise InputError(f"{where} duplicate channel name(s) {dup}: 'total' is the all-orbital "
                                 "channel while total = true, and project entries must be distinct")
        for pname in ("tol", "symprec", "check_tol", "kramers_tol", "kT", "window", "sigma"):
            if pname in task and (not _is_number(task[pname]) or task[pname] <= 0):
                raise InputError(f"{where} {pname} must be a positive finite number")
        if "mu" in task and not _is_number(task["mu"]):
            raise InputError(f"{where} mu must be a finite number")
        if "mesh" in task:
            _int_list(task["mesh"], f"{where} mesh", length=3, minimum=1)
        if "project" in task:
            _unique_list(task["project"], f"{where} project", allow_empty=True)
        for pname in task.get("project", []):
            if pname not in cfg.get("projection", {}):
                raise InputError(f"{where} unknown projection {pname!r}")
    return cfg


def load_input(path):
    """Read and validate on rank 0, broadcast to all ranks."""
    cfg, err = None, None
    if mpi.rank0:
        # everything rank 0 does is inside the try: an exception escaping here
        # would leave the other ranks waiting in bcast
        try:
            with open(path, "rb") as f:
                cfg = validate(tomllib.load(f))
        except (OSError, tomllib.TOMLDecodeError, InputError, ValueError, TypeError) as exc:
            err = ("input", f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            import traceback
            err = ("internal", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        if err is not None:
            cfg = None
    cfg, err = mpi.bcast((cfg, err), root=0)
    if err is not None:
        kind, msg = err
        if kind == "input":
            raise InputError(msg)
        raise RuntimeError(f"reading {path} (rank 0): {msg}")
    return cfg


def on_rank0(action, what):
    """Run action() on rank 0 only (file output) and share its failure with
    every rank, so no rank continues into a collective alone."""
    err = None
    if mpi.rank0:
        try:
            action()
        except Exception as exc:
            err = f"{what}: {type(exc).__name__}: {exc}"
    err = mpi.bcast(err, root=0)
    if err is not None:
        raise OutputError(err)


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
                            z=shell.get("z"), x=shell.get("x"), pos=shell.get("pos"))
    return shells


def projection_indices(proj, shells, nwan):
    """Orbital indices (final basis) of one [projection] entry."""
    if "index" in proj:
        if not isinstance(proj["index"], list) or not all(_is_int(i) for i in proj["index"]):
            raise InputError(f"projection index {proj['index']!r} must be a list of integers")
        idx = np.asarray(proj["index"], dtype=int)
        if idx.size == 0 or idx.min() < 0 or idx.max() >= nwan:
            raise InputError(f"projection index {proj['index']} out of range [0, {nwan})")
        if len(np.unique(idx)) != len(idx):
            raise InputError(f"projection index {proj['index']}: an orbital set must not repeat orbitals")
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
    if len(set(idx)) != len(idx):
        raise InputError(f"projection {proj}: shells/orbitals select some orbitals more than once")
    return np.array(sorted(idx), dtype=int)


# ----------------------------------------------------------------------------
# model construction
# ----------------------------------------------------------------------------

def build_model(cfg, log):
    mcfg = cfg["model"]
    spin = mcfg["spin"]

    def read_hr(hr_file, wsvec):
        try:
            model = Model.from_hr(hr_file, wsvec_file=wsvec)
        except ValueError as exc:
            raise InputError(str(exc))
        ws = f" (+ {wsvec}: R set expanded to minimal-distance images)" if wsvec else ""
        log(f"read {hr_file}: nwan = {model.nwan}, nrpt = {model.nrpt}{ws}")
        # real-space diagnostic only: not a bound on H(k) (sum_bound is), not a fit quality
        herm, warn = model.hermiticity, mcfg.get("hermiticity_warn", HERMITICITY_WARN)
        log(f"  hermiticity as read: max_R max_mn |H(R) - H(-R)^+| = {herm['max']:.1e} eV, "
            f"bound on max_k |H(k) - H(k)^+| = {herm['sum_bound']:.1e} eV")
        if herm["max"] > warn and mpi.rank0:
            warnings.warn(f"{hr_file}: max_R max_mn |H(R) - H(-R)^+| = {herm['max']:.1e} eV "
                          f"exceeds hermiticity_warn = {warn:g} eV")
        return model

    tbmodel = read_hr(mcfg["hr_file"], mcfg.get("wsvec"))
    nwan_hr = tbmodel.nwan

    if spin == "doubled":
        tbmodel.addspinbasis()
        log("spin doubled (non-magnetic): nwan -> %d" % tbmodel.nwan)
    elif spin == "collinear":
        model_dn = read_hr(mcfg["hr_dn"], mcfg.get("wsvec_dn"))
        if model_dn.nwan != nwan_hr:
            raise InputError("hr_file and hr_dn differ in nwan")
        if set(model_dn.ham) != set(tbmodel.ham):
            if "wsvec" not in mcfg:
                raise InputError("hr_file and hr_dn differ in R vectors")
            # the two spin channels have their own Wannier centres -> own images
            rvecs = set(model_dn.ham) | set(tbmodel.ham)
            tbmodel.pad_rvecs(rvecs)
            model_dn.pad_rvecs(rvecs)
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
    if rot and spin != "none":
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
    from .udmft import (local_eigenbasis, onsite_block_mismatch, dump_local_basis, dump_dmft,
                       time_reversal_error)

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
        hloc = h00[start:start+norb, start:start+norb]
        log(f"udmft: max |T H_loc T^-1 - H_loc| = {time_reversal_error(hloc):.3e} eV")
        try:
            eon, von, htmat, mz = local_eigenbasis(hloc, ll=l, kramers=task.get("kramers", False),
                                                   tr_tol=task.get("kramers_tol", 1e-3))
        except ValueError as exc:
            raise InputError(f"[task.udmft] {exc}")
        if task.get("kramers", False):
            log("udmft: Kramers gauge fixed by Mz; H_loc symmetrised under T")
        log("udmft: Eimp = " + np.array2string(eon, precision=5, max_line_width=200))
        log("udmft: Mz per solver flavour = " + np.array2string(np.diag(mz).real, precision=4, max_line_width=200))

    kpts = Kpoint.gen_uniform_kpoints(task["mesh"])
    log(f"udmft: diagonalising {kpts.nktot} k-points")
    eigs, vecs = tbmodel.get_bulk_eigens(kpts.kfrac)

    def write():
        if von is not None:
            dump_local_basis(eon, von, htmat)
            if task.get("save_mz", False):
                np.save("Mz_eimp_blocked.npy", mz)
        dump_dmft(eigs, vecs, mu, ntot=task["ntot"], norb=norb, natm=natm,
                  lnum=l, nspin=nspin, von=von, start=start)
    on_rank0(write, "udmft output")
    log("udmft: wrote eig.dat, udmft.dat" + (", eon.dat, von.dat, htmat.dat" if von is not None else ""))


def task_bands(cfg, task, tbmodel, shells, log):
    from .bands import dump_bands, dump_projs_multi, dump_ticks

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
    def write():
        if names:
            dump_projs_multi(kpth, eigs, projs, names, fname=output, mu=mu)
        else:
            dump_bands(kpth, eigs, fname=output, mu=mu)
        dump_ticks(ticks, kpath["labels"], fname=tick_file)
    on_rank0(write, f"bands output {output}")
    log(f"bands: wrote {output}, {tick_file}")


def task_dos(cfg, task, tbmodel, shells, log):
    from .dos import dump_total_dos

    names = task.get("project", [])
    plists = build_projections(cfg, names, shells, tbmodel.nwan, log) if names else []
    kpts = Kpoint.gen_uniform_kpoints(task["mesh"])
    output = task.get("output", "dos.dat")
    log(f"dos: {kpts.nktot} k-points")

    omega, tdos, pdos = tbmodel.get_proj_dos(kpts.kfrac, proj=plists, emin=task["emin"], emax=task["emax"],
                                             ne=task["ne"], mu=task.get("mu", 0.0), sigma=task["sigma"])
    on_rank0(lambda: dump_total_dos(omega, tdos, pdos, output, names=names), f"dos output {output}")
    log(f"dos: wrote {output}")


def shell_species(name):
    """Element of a shell from its name: Co1_d -> Co, S2_p -> S."""
    return name.split("_")[0].rstrip("0123456789") or name


def task_symmetry(cfg, task, tbmodel, shells, log):
    from . import symmetry as sym

    spin = cfg["model"]["spin"]
    lattice = np.asarray(cfg["lattice"]["rvec"], dtype=float)
    tol = task.get("tol", 1e-3)
    symprec = task.get("symprec", 1e-3)
    shl = [dict(name=name, start=sh["start"], l=sh["l"], nspin=sh["nspin"], pos=sh["pos"],
                species=shell_species(name), frame=None if sh["z"] is None else (sh["z"], sh["x"]))
           for name, sh in shells.items()]
    try:
        sym.check_basis(shl, tbmodel.nwan)
    except ValueError as exc:
        raise InputError(f"[task.symmetry] {exc}")
    sites, site_of = sym.group_sites(shl, lattice, symprec=symprec)
    log(f"symmetry: {len(sites)} sites from the shells (symprec = {symprec:.1e}, Cartesian)")
    for site in sites:
        log(f"  {site['species']:3s} {np.array2string(site['pos'], precision=5, floatmode='fixed')}  "
            + " ".join(shl[i]["name"] + ("" if not c.any() else f"{c.tolist()}")
                       for i, c in zip(site["shells"], site["cells"])))

    moms = None
    if "magmom" in task:
        moms = np.zeros((len(sites), 3))
        names = [sh["name"] for sh in shl]
        for name, mom in task["magmom"].items():
            isite = site_of[names.index(name)]
            if np.any(moms[isite]) and not np.allclose(moms[isite], mom):
                raise InputError(f"[task.symmetry] magmom: conflicting moments on the site of {name}")
            moms[isite] = mom
    species = sorted({site["species"] for site in sites})
    numbers = [species.index(site["species"]) + 1 for site in sites]
    try:
        ops, label = sym.find_operations(lattice, [site["pos"] for site in sites], numbers,
                                         magmoms=moms, symprec=symprec)
    except (ImportError, ValueError) as exc:
        raise InputError(f"[task.symmetry] {exc}")
    kind = "magnetic space group" if moms is not None else "space group"
    log(f"symmetry: {kind} of the sites {label}, {len(ops)} operations")
    if spin == "none" and moms is not None:
        # one spin channel: the moment-flipping (primed) operations swap up and dn
        ops = [op for op in ops if not op[2]]
        log(f"symmetry: spinless model, keeping the {len(ops)} unprimed operations")
    if task.get("time_reversal", moms is None or spin == "none"):
        ops = ops + [(w, t, True) for w, t, tr in ops if not tr]
        log(f"symmetry: + time reversal ({'K' if spin == 'none' else 'i sigma_y K'}) -> {len(ops)} operations")

    kpts = sym.random_kpoints(task.get("nk", 8))
    orbital_axes = task.get("orbital_axes", "shell") == "shell"
    log(f"symmetry: {len(kpts)} random k-points, tol = {tol:.1e} eV, "
        f"shell z/x are {'orbital + spin' if orbital_axes else 'spin-only'} frames")
    try:
        results = sym.check_model(tbmodel, lattice, shl, ops, kpts, orbital_axes=orbital_axes,
                                  symprec=symprec)
    except ValueError as exc:
        raise InputError(f"[task.symmetry] {exc}")

    log(f"  {'#':>3s}  {'op':4s} {'axis (cart)':22s} {'t (frac)':22s} {'TR':2s}  "
        f"{'eig_err':>9s}  {'mat_err':>9s}  status   worst block")
    nok, nbasis = 0, 0
    for iop, res in enumerate(results):
        wmat, tvec, trev = res["op"]
        axis = "" if res["axis"] is None else "[" + ", ".join(f"{a:6.3f}" for a in res["axis"]) + "]"
        tstr = "[" + ", ".join(f"{t:6.3f}" for t in tvec) + "]"
        if res["mat_err"] < tol:
            status, nok = "ok", nok + 1
        elif res["eig_err"] < tol:
            status, nbasis = "H-only", nbasis + 1
        else:
            status = "BROKEN"
        iw, jw, err = res["worst"]
        worst = f"{iw}-{jw} {err:.1e}" if status != "ok" else ""
        log(f"  {iop+1:3d}  {res['name']:4s} {axis:22s} {tstr:22s} {'T' if trev else ' ':2s}  "
            f"{res['eig_err']:9.2e}  {res['mat_err']:9.2e}  {status:7s}  {worst}")
    log(f"symmetry: {nok}/{len(results)} operations satisfied within {tol:.1e} eV")
    if nbasis:
        log("symmetry: 'H-only' = spectrum symmetric but H(k) not.  Either the operation is "
            "broken invisibly in the spectrum (e.g. T in a magnet with inversion), or the basis "
            "is described wrongly: check shell order / l, pos, z/x (orbital_axes), spin ordering")


def task_fermi_surface(cfg, task, tbmodel, shells, log):
    from . import fermisurface as fs

    bvec = reciprocal(cfg)
    names = task.get("project", [])
    plists = build_projections(cfg, names, shells, tbmodel.nwan, log) if names else []
    nk = task.get("nk", [201, 201])
    mu = task.get("mu", 0.0)
    window = task.get("window", 0.2)
    basis = task.get("basis", "frac")
    output = task.get("output", "fermi_surface.npz")
    data = dict(mu=mu, window=window, basis=basis, bvec=bvec, project=np.array(names, dtype=str),
                planes=np.array([p["name"] for p in task["plane"]], dtype=str))
    for plane in task["plane"]:
        name = plane["name"]
        s, t, kfrac, kcart = fs.plane_kpoints(plane["origin"], plane["u"], plane["v"], nk, bvec, basis)
        log(f"fermi_surface: plane {name}: {nk[0]} x {nk[1]} k-points")
        eigs, projs = fs.plane_eigens(tbmodel, kfrac, plists)
        eigs = eigs - mu
        keep = np.flatnonzero((eigs.min(axis=0) < window) & (eigs.max(axis=0) > -window))
        crossing = [int(b)+1 for b in keep if eigs[:, b].min() < 0 < eigs[:, b].max()]
        log(f"  bands within mu +- {window} eV: {(keep+1).tolist()}; crossing mu: {crossing}")
        to_cart = (lambda x: np.asarray(x, dtype=float) @ bvec) if basis == "frac" else \
                  (lambda x: np.asarray(x, dtype=float))
        data.update({f"{name}_s": s, f"{name}_t": t, f"{name}_bands": keep+1,
                     f"{name}_origin": to_cart(plane["origin"]), f"{name}_u": to_cart(plane["u"]),
                     f"{name}_v": to_cart(plane["v"]),
                     # explicit band count: -1 cannot be inferred for empty project or keep
                     f"{name}_E": eigs[:, keep].reshape(nk[0], nk[1], len(keep)),
                     f"{name}_W": projs[:, :, keep].reshape(len(plists), nk[0], nk[1], len(keep)),
                     f"{name}_bz": fs.bz_function(kcart, bvec).reshape(nk[0], nk[1])})
    on_rank0(lambda: np.savez_compressed(output, **data), f"fermi_surface output {output}")
    log(f"fermi_surface: wrote {output}")


def chi0_channels(cfg, task, tbmodel, shells, log):
    """[(name, channel, site names)] for [task.chi0]; see lindhard.make_channel."""
    from . import lindhard as lh
    from . import symmetry as sym

    lists = [("total", np.arange(tbmodel.nwan))] if task.get("total", True) else []
    names = task.get("project", [])
    if names:
        lists += list(zip(names, build_projections(cfg, names, shells, tbmodel.nwan, log)))
    if task.get("vertex", "point_site") == "cell":
        return [(name, lh.cell_channel(idx), ["cell"]) for name, idx in lists]

    # point_site: orbital -> (site, integer cell offset L) from the shells with pos
    lattice = np.asarray(cfg["lattice"]["rvec"], dtype=float)
    shl = [dict(name=name, start=sh["start"], size=sh["size"], pos=sh["pos"], species=shell_species(name))
           for name, sh in shells.items() if sh["pos"] is not None]
    sites, site_of = sym.group_sites(shl, lattice, symprec=task.get("symprec", 1e-3))
    orb_site = -np.ones(tbmodel.nwan, dtype=int)
    orb_cell = np.zeros((tbmodel.nwan, 3), dtype=int)
    for ish, sh in enumerate(shl):
        site = sites[site_of[ish]]
        orb_site[sh["start"]:sh["start"]+sh["size"]] = site_of[ish]
        orb_cell[sh["start"]:sh["start"]+sh["size"]] = site["cells"][site["shells"].index(ish)]
    site_names = [shl[site["shells"][0]]["name"].split("_")[0] for site in sites]

    out = []
    for name, idx in lists:
        missing = idx[orb_site[idx] < 0]
        if missing.size:
            raise InputError(f"[task.chi0] vertex = 'point_site': channel {name!r} has orbitals without "
                             f"a shell pos: {missing.tolist()} (give every shell pos, or vertex = 'cell')")
        used = sorted(set(orb_site[idx].tolist()))
        local = {s: i for i, s in enumerate(used)}
        out.append((name, lh.make_channel(idx, [local[s] for s in orb_site[idx]], orb_cell[idx],
                                          [sites[s]["pos"] for s in used]),
                    [site_names[s] for s in used]))
    return out


def task_chi0(cfg, task, tbmodel, shells, log):
    from . import lindhard as lh

    bvec = reciprocal(cfg)
    mesh = task["mesh"]
    mu, kT, window = task.get("mu", 0.0), task["kT"], task.get("window", 0.5)
    vertex = task.get("vertex", "point_site")
    kblock = task.get("kblock", 4096)
    output = task.get("output", "chi0.npz")
    chans = chi0_channels(cfg, task, tbmodel, shells, log)
    for name, c, snames in chans:
        log(f"chi0 channel {name}: {len(c['orbitals'])} orbitals on sites {snames}, vertex {vertex}")
    try:
        res = lh.lindhard(tbmodel, mesh, mu, kT, window, [c for _, c, _ in chans], kblock=kblock, log=log)
    except ValueError as exc:
        raise InputError(f"[task.chi0] {exc}")

    qfrac = res["points"]/np.asarray(mesh, dtype=float)
    # extended zone for the log: Q = q + G, G in {0,1}^3
    gshift = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij")).reshape(3, -1).T
    qext = (qfrac[None, :, :] + gshift[:, None, :]).reshape(-1, 3)
    peaks = {}
    for name, c, _ in chans:
        chi_st = res["chi"][[n for n, _, _ in chans].index(name)]
        vals = lh.chi_physical(chi_st, c["tau0"], mesh, qext)
        imax = int(np.argmax(vals))
        peaks[name] = qext[imax]
        log(f"chi0 {name:>10s}: Q=0 {vals[0]:.4f}, max over Q in [0,2)^3 {vals[imax]:.4f} at Q_frac = "
            + np.array2string(qext[imax], precision=4) + "  (1/eV per cell)")

    check = []
    if "window_check" in task:
        wide = window*task["window_check"]
        eig2, vec2, bands2 = lh.mesh_eigens(tbmodel, mesh, mu, wide)
        log(f"chi0 window_check: window {window} -> {wide} eV ({len(res['bands'])} -> {len(bands2)} bands)")
        for ic, (name, c, _) in enumerate(chans):
            for Q in (np.zeros(3), peaks[name]):
                iq = lh.mesh_index(np.rint(Q*np.asarray(mesh)).astype(int), mesh)
                new_st = lh.site_matrices(eig2, vec2, res["points"], mesh, iq, [c], kT, kblock)[0]
                old = lh.chi_physical(res["chi"][ic], c["tau0"], mesh, Q)[0]
                new = lh.chi_of_matrix(new_st, c["tau0"], Q)
                floor = 1e-8*max(abs(old), abs(new)) + 1e-14
                rel = abs(new - old)/max(abs(old), floor)
                check.append([ic, *Q, old, new])
                log(f"  {name:>10s} Q_frac {np.array2string(Q, precision=4)}: {old:.6f} -> {new:.6f} "
                    f"(rel. change {rel:.2e})")
        log("  (diagnostic at Gamma and the peak only; not a convergence proof for the whole map)")

    spin = cfg["model"]["spin"]
    meta = dict(
        format="swept-chi0", version=2, vertex_model=vertex, spin_mode=spin,
        units="1/eV per model unit cell",
        position_source=("shell pos grouped into sites by symmetry.group_sites "
                         f"(symprec {task.get('symprec', 1e-3)} Angstrom)") if vertex == "point_site"
        else "none: one site per channel, tau = 0 (lattice-gauge cell density, no position phases)",
        normalization=("density bubble B of one spin copy; spin-degenerate non-interacting: "
                       "chi_nn = 2B, chi_SzSz = B/2 (hbar = 1)") if spin == "none"
        else "density response of the selected spin-orbital set; no spin factor applied",
        index_convention=("chi_<c>[iq, s, t] = chi_st(q), iq = (i0*N1 + i1)*N2 + i2, qfrac[iq] = (i0, i1, i2)/N "
                          "in [0,1); orbitals_<c>: 0-based model indices; siteidx_<c>[a]: row of tau0_<c>; "
                          "tau_a = tau0_<c>[siteidx_<c>[a]] + offsets_<c>[a] (fractional, integer offsets); "
                          "chi(Q) = sum_st e^{-2 pi i Q.tau0_s} chi_st e^{+2 pi i Q.tau0_t}"),
        approximation=("point-site density vertex (no Wannier form factors)" if vertex == "point_site"
                       else "cell density vertex (no position phases, no form factors)")
        + "; bands outside mu +- window dropped")
    data = dict(meta, mesh=np.asarray(mesh), kT=kT, mu=mu, window=window, bands=res["bands"]+1,
                channels=np.array([n for n, _, _ in chans], dtype=str), qfrac=qfrac, bvec=bvec,
                window_occupation=res["window_occupation"])
    for ic, (name, c, snames) in enumerate(chans):
        data.update({f"chi_{name}": res["chi"][ic], f"tau0_{name}": c["tau0"],
                     f"sites_{name}": np.array(snames, dtype=str), f"orbitals_{name}": c["orbitals"],
                     f"siteidx_{name}": c["site"], f"offsets_{name}": c["offsets"]})
    if check:
        data["window_check"] = np.array(check)           # rows: channel, Q_frac(3), old, new
        data["window_check_factor"] = task["window_check"]
    on_rank0(lambda: np.savez_compressed(output, **data), f"chi0 output {output}")
    log(f"chi0: wrote {output}")


TASK_FUNCS = {"udmft": task_udmft, "bands": task_bands, "dos": task_dos, "symmetry": task_symmetry,
              "fermi_surface": task_fermi_surface, "chi0": task_chi0}


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


class RunLog:
    """swept_run.log, written on rank 0; call it with a message to log one line.

    Opening is checked collectively by main before any other collective call.
    A later write, flush or close failure warns once on stderr and the log
    continues on stdout only (the computation goes on).  Contains no
    collective call: not every rank logs the same number of times."""

    def __init__(self, path="swept_run.log"):
        self.path = path
        self.file = None

    def open(self):
        """Open the file (rank 0 only); returns None or the error message."""
        try:
            self.file = open(self.path, "w")
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        return None

    def _write(self, text):
        if self.file is None:
            return
        try:
            self.file.write(text)
            self.file.flush()
        except OSError as exc:
            print(f"swept: warning: writing {self.path} failed ({type(exc).__name__}: {exc}); "
                  "the log continues on stdout only", file=sys.stderr, flush=True)
            self.close(warn=False)

    def __call__(self, msg):
        if mpi.rank0:
            print(msg); sys.stdout.flush()
            self._write(msg + "\n")

    def write_input(self, path):
        """Copy of the input file into the log (not to stdout)."""
        if mpi.rank0 and self.file is not None:
            try:
                text = Path(path).read_text()
            except (OSError, ValueError) as exc:
                text = f"(input not copied: {type(exc).__name__}: {exc})"
            self._write("----- input -----\n" + text + "\n----- end input -----\n")

    def close(self, warn=True):
        f, self.file = self.file, None
        if f is not None:
            try:
                f.close()                    # may flush again and fail once more
            except OSError as exc:
                if warn:
                    print(f"swept: warning: closing {self.path} failed ({type(exc).__name__}: {exc}); "
                          "the log file may be incomplete", file=sys.stderr, flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: [mpirun -np N] python run.py input.toml")
        return 2
    path = argv[0]

    log = RunLog()
    # every rank passes here before any other collective: a log that cannot be
    # opened stops all ranks before the run starts
    err = mpi.bcast(log.open() if mpi.rank0 else None, root=0)
    if err is not None:
        if mpi.rank0:
            print(f"swept: cannot open {log.path}: {err}", file=sys.stderr, flush=True)
        return 1

    try:
        log(f"SWEPT run  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  "
            f"swept {_git_version()}  nproc = {mpi.size}")
        log(f"input: {Path(path).resolve()}")
        cfg = load_input(path)
        log.write_input(path)

        tbmodel, shells = build_model(cfg, log)
        for tname, task in cfg.get("task", {}).items():
            log(f"=== task {tname} ===")
            TASK_FUNCS[tname](cfg, task, tbmodel, shells, log)
        log("done")
        return 0
    except InputError as exc:
        log(f"input error: {exc}")
        return 1
    except OutputError as exc:
        log(f"output error: {exc}")
        return 1
    except (mpi.CollectiveError, FloatingPointError) as exc:
        # raised on every rank alike (collective failure, or a check on reduced data)
        log(f"error: {type(exc).__name__}: {exc}")
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    try:
        status = main()
    except Exception:
        if mpi.size > 1:
            # an exception on some ranks only would leave the others waiting in a
            # collective forever: report it and end the whole job
            import traceback
            traceback.print_exc()
            sys.stderr.flush()
            mpi.comm.Abort(1)
        raise
    sys.exit(status)
