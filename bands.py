#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jianzhou Zhao
@Date: 2019-05-11 21:40:31
LastEditors: your name
LastEditTime: 2025-05-26 10:30:48
@Description: File content
'''


def plotbands(kpth, eigs, file='bands.pdf', dpi=100, ylim=None, ticks=None, labels=None, linewidth=1.0,
              color='red', figsize=(6, 6), ef=0.0):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for band in eigs:
        ax.plot(kpth, band, linewidth=linewidth, color=color)
    ax.set_xlim(kpth[0], kpth[-1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if ticks is not None:
        ax.set_xticks(ticks)
    if ticks is not None and labels is not None:
        ax.set_xticklabels(labels)
    ax.axhline(ef, linestyle='--', color='grey')
    if ticks is not None:
        for tick in ticks[1:-1]:
            ax.axvline(tick, linestyle='--', color='grey')
    plt.savefig(file, dpi=dpi)
    return ax

def dump_bands(kpth, eigs, fname='bands.dat', mu=0):
    with open(fname,"w") as fdos:
        for band in eigs:
            for ikpt, kpt in enumerate(kpth):
                fdos.write("{:15.9f}{:15.9f}\n".format(kpt, band[ikpt]-mu))
            fdos.write("\n")
            fdos.write("\n")

def dump_projs(kpth, eigs, projs, fname='bands.dat', mu=0):
    output_format = "{:15.9f}{:15.9f}{:15.9f}"+"\n"
    with open(fname,"w") as fdos:
        for ibnd, band in enumerate(eigs):
            for ikpt, kpt in enumerate(kpth):
                fdos.write(output_format.format(kpt, band[ikpt]-mu, projs[ibnd,ikpt]))
            fdos.write("\n")
            fdos.write("\n")

def dump_projs_multi(kpth, eigs, projs, names, fname='bands.dat', mu=0):
    # eigs (nbnd, nk), projs (nproj, nbnd, nk); 列: k, E-mu, w_1 ... w_nproj
    # 只有一组投影时数据格式与 dump_projs 相同
    output_format = "{:15.9f}{:15.9f}" + "{:15.9f}"*len(names) + "\n"
    with open(fname,"w") as fdos:
        fdos.write("# k  E-mu  " + "  ".join(names) + "\n")
        for ibnd, band in enumerate(eigs):
            for ikpt, kpt in enumerate(kpth):
                fdos.write(output_format.format(kpt, band[ikpt]-mu, *projs[:, ibnd, ikpt]))
            fdos.write("\n")
            fdos.write("\n")

def dump_ticks(ticks, labels, fname='ticks.dat'):
    with open(fname,"w") as f:
        for label, tick in zip(labels, ticks):
            f.write("{:>10s}{:15.9f}\n".format(label, tick))

def dump_surfbands(kpth, eigs, ptop, pbot, fname='bands.dat', mu=0):
    with open(fname,"w") as fdos:
        for ibnd, band in enumerate(eigs):
            for ikpt, kpt in enumerate(kpth):
                fdos.write("{:15.9f}{:15.9f}{:15.9f}{:15.9f}\n".format(kpt, band[ikpt]-mu, ptop[ibnd,ikpt], pbot[ibnd,ikpt]))
            fdos.write("\n")
            fdos.write("\n")

def dump_akws(kpth, omgs, akws, fname='akws.dat'):
    with open(fname,"w") as fdos:
        for iomg, omg in enumerate(omgs):
            for ikpt, kpt in enumerate(kpth):
                fdos.write("{:10.4f}{:10.4f}{:20.10f}\n".format(omg, kpt, akws[iomg, ikpt]))
            fdos.write("\n")
            fdos.write("\n")

if __name__ == "__main__":
    if __package__ in (None, ""):
        # run as a script (python /path/to/swept/x.py): import it as part of the package
        import importlib
        import sys as _sys
        from pathlib import Path as _Path
        _pkgdir = _Path(__file__).resolve().parent
        _sys.path.insert(0, str(_pkgdir.parent))
        __package__ = _pkgdir.name
        importlib.import_module(__package__)


    from .kpoint import Kpoint
    from .model import Model
    from . import mpiutils as mpi
    import numpy as np
    import sys
    # from basis import p2t2g, complex2real, clebsch_gordan
    # from scipy import constants

    np.set_printoptions(precision=3,suppress=True,linewidth=180)
    
    a = 5.378080; c = 4.391283
    rvec = np.array([
        [-a,   0.0,  0.0],
        [ a/2, a*np.sqrt(3.0)/2,  0.0],
        [ 0.0, a*np.sqrt(3.0)/3,  c  ]
    ])
    bvec = 2*np.pi*np.linalg.inv(rvec).T

    local_zx = np.array([
       [[+0.0,+0.0,+1.0],[+0.0000000000,-1.0000000000,+0.0000000000]],
       [[+0.0,+0.0,+1.0],[-0.8660254038,+0.5000000000,+0.0000000000]],
       [[+0.0,+0.0,+1.0],[+0.8660254038,+0.5000000000,+0.0000000000]]
    ])

    if mpi.rank0 : print(rvec, '\n')
    if mpi.rank0 : print(bvec, '\n'); sys.stdout.flush()

    tbmodel = Model.from_hr('co3sn2s2_hr.dat')

    # tbmodel.to_hr('co3sn2s2_eshift_hr.dat')
    # exit()

    # tbmodel = Model.from_hr('co3sn2s2_eshift_hr.dat')

    # tmat = np.dot(complex2real(1,2), p2t2g())
    # tmat = np.dot(clebsch_gordan(1), tmat)
    # prmat = np.eye(tbmodel.nwan, dtype=complex)
    # for iatm in range(12):
    #     prmat[iatm*6:iatm*6+6,iatm*6:iatm*6+6] = tmat

    # tbmodel.rotate_spin(lnum=1, zxmat=local_zx)
    # if mpi.rank0 : print(np.diag(tbmodel.ham[(0,0,0)][0:6,0:6].real), '\n')
    
    # tbmodel.rotate_basis(rmat=prmat)
    # if mpi.rank0 : print(np.diag(tbmodel.ham[(0,0,0)][0:6,0:6].real), '\n')

    # emat = [-0.289844030566405600, 0.63357407252051150,
    #         -0.370305182458046900, 0.32093712472656166,
    #         -0.026002969928052178, 0.40071887422586450,
    #         -0.133815801158777110, 0.56212364119113570,
    #         -0.301479651137959400, 0.54621096954821250]
    
    # tbmodel.shift_onsite(emat, index=range(10))
    # tbmodel.shift_onsite(emat, index=range(10,20))
    # tbmodel.shift_onsite(emat, index=range(20,30))

    # tbmodel.to_hr('co3sn2s2_eshift_hr.dat')
    # exit()

    # plist = np.array([0,10,20]) # Ni 3dz2
    # plist = np.array([2,12,22]) # Ni 3dxz
    # plist = np.array([4,14,24]) # Ni 3dyz
    # plist = np.array([6,16,26]) # Ni 3dx2-y2
    # plist = np.array([8,18,28]) # Ni 3dxy

    # plist = np.array([1,11,21]) # Ni 3dz2
    # plist = np.array([3,13,23]) # Ni 3dxz
    # plist = np.array([5,15,25]) # Ni 3dyz
    # plist = np.array([7,17,27]) # Ni 3dx2-y2
    # plist = np.array([9,19,29]) # Ni 3dxy

    hskp = [
           [ 0.5,      0.0,     0.0],
           [ 1.0/3.0,  1.0/3.0, 0.0],
           [ 0.0,      0.0,     0.0],
           [ 0.5,      0.0,     0.0]]
    kwant = 1000

    kpts = Kpoint.from_high_symmetry_points(khsym_frac=hskp, bzvec=bvec, kwant=kwant)

    # eigs = tbmodel.get_bulk_eigvals(kpts.kfrac)
    omgs, akws = tbmodel.get_bulk_projected_kz(kpts.kfrac, omin=-0.5, omax=0.5, nomg=1000, nkz=1000, sigma=0.01, mu=12.5800)
    # eigs, projs = tbmodel.get_bulk_eprojs(kpts.kfrac, plist)
    # eigs, ptop, pbot = tbmodel.get_slab_eprojs(kpts.kfrac,100,surf_pot=0.0)

    hkreal = np.array([np.dot(kpt, bvec) for kpt in kpts.kfrac])
    ticks = Kpoint.k_length_accumulate((np.array(hskp)))
    # label = [r'K', r'$\Gamma$', r'T']

    if mpi.rank0: print(ticks)

    if mpi.rank0:
        # dump_bands(Kpoint.k_length_accumulate(kpts.kreal), eigs, fname='ggawan_eshift.dat', mu=12.5800)
        # dump_projs(Kpoint.k_length_accumulate(kpts.kreal), eigs, projs, fname='ggawan_dx2-y2_dn.dat', mu=12.5647)
        dump_akws(Kpoint.k_length_accumulate(kpts.kreal), omgs, akws, fname='akws.dat')
        # plotbands(Kpoint.k_length_accumulate(kpts.kreal), eigs, ticks=ticks, labels=label)
