#!/usr/bin/env python
'''
Plot the planes written by [task.fermi_surface].

    python plot_fermi_surface.py fermi_surface.npz [-o fs.pdf] [-e 0.0]

One panel per plane, axes in 1/Angstrom along u and its orthogonal
complement (true shape also for non-orthogonal u, v).  Contours at E - mu = e
are coloured by the first three projections (red, blue, green, mixed by
weight); without projections every band gets its own colour.  The dashed line
is the first Brillouin-zone boundary.
'''

import argparse

if __package__ in (None, ""):
    # run as a script (python /path/to/swept/x.py): import it as part of the package
    import importlib
    import sys as _sys
    from pathlib import Path as _Path
    _pkgdir = _Path(__file__).resolve().parent
    _sys.path.insert(0, str(_pkgdir.parent))
    __package__ = _pkgdir.name
    importlib.import_module(__package__)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from contourpy import contour_generator
from scipy.interpolate import RegularGridInterpolator

from .fermisurface import plane_axes

RGB = np.array([[1.0, 0.2, 0.2], [0.255, 0.412, 0.882], [0.1, 0.65, 0.2]])


def plot_plane(ax, data, name, names, energy):
    s, t = data[f"{name}_s"], data[f"{name}_t"]
    E, W, bands = data[f"{name}_E"], data[f"{name}_W"], data[f"{name}_bands"]
    _, _, st2xy = plane_axes(data[f"{name}_u"], data[f"{name}_v"])
    nproj = min(len(names), 3)
    for ib in range(E.shape[-1]):
        if not E[..., ib].min() < energy < E[..., ib].max():
            continue
        # lines in (s, t): contourpy takes z indexed [y, x]
        segs = contour_generator(s, t, E[..., ib].T).lines(energy)
        interp = [RegularGridInterpolator((s, t), W[ip, ..., ib]) for ip in range(nproj)]
        for path in segs:
            if len(path) < 2:
                continue
            xy = path @ st2xy
            if nproj:
                w = np.stack([f(path) for f in interp], axis=1)
                col = np.clip(w/np.maximum(w.sum(axis=1, keepdims=True), 1e-9) @ RGB[:nproj], 0, 1)
                ax.add_collection(LineCollection(np.stack([xy[:-1], xy[1:]], axis=1),
                                                 colors=col[:-1], lw=1.5))
            else:
                ax.plot(xy[:, 0], xy[:, 1], lw=1.5, color=f"C{ib % 10}", label=f"band {bands[ib]}")
    S, T = np.meshgrid(s, t, indexing="ij")
    XY = np.stack([S, T], axis=-1) @ st2xy
    ax.contour(XY[..., 0], XY[..., 1], data[f"{name}_bz"], levels=[0.0],
               colors="k", linewidths=0.8, linestyles="--")
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) @ st2xy
    ax.set_xlim(corners[:, 0].min(), corners[:, 0].max())
    ax.set_ylim(corners[:, 1].min(), corners[:, 1].max())
    ax.set_aspect("equal")
    ax.set_title(name)
    ax.set_xlabel(r"$k_{\parallel u}$ (1/$\AA$)")
    ax.set_ylabel(r"$k_{\perp u}$ (1/$\AA$)")
    ax.tick_params(direction="in", top=True, right=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz")
    parser.add_argument("-o", "--output", default="fermi_surface.pdf")
    parser.add_argument("-e", "--energy", type=float, default=0.0, help="E - mu of the contours (eV)")
    args = parser.parse_args(argv)
    data = dict(np.load(args.npz))
    planes, names = list(data["planes"]), list(data["project"])
    fig, axes = plt.subplots(1, len(planes), figsize=(5*len(planes), 5.2), squeeze=False,
                             layout="constrained")
    for ax, name in zip(axes[0], planes):
        plot_plane(ax, data, name, names, args.energy)
    handles = [plt.Line2D([], [], color=RGB[i], lw=2, label=n) for i, n in enumerate(names[:3])]
    if handles:
        axes[0, 0].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.95)
    fig.savefig(args.output, dpi=150)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
