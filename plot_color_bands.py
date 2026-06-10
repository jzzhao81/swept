'''
Author: Jianzhou Zhao
Date: 2025-02-02 15:23:04
LastEditors: your name
LastEditTime: 2025-02-02 16:31:27
Description: file content
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import itertools
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

if __name__ == '__main__':

    ry2ev = constants.value('Rydberg constant times hc in eV')

    band_color = (0.75,0.75,0.75,0.5)

    mycmap = []

    colors = [(1.0,1.0,1.0,0.0), (1.0,0.7,0.7,0.5), (1.0,0.2,0.2,0.9)]
    nodes =  [0.0, 0.4, 1.0]
    mycmap.append( LinearSegmentedColormap.from_list("mycmap1", list(zip(nodes, colors))) )

    colors = [(1.0,1.0,1.0,0.0), (0.4,0.60,0.9,0.5), (0.0,0.4,0.8,0.9)]
    nodes =  [0.0, 0.4, 1.0]
    mycmap.append( LinearSegmentedColormap.from_list("mycmap2", list(zip(nodes, colors))) )

    colors = [(1.0,1.0,1.0,0.0), (0.4,0.9,0.4,0.5), (0.1,0.8,0.1,0.9)]
    nodes =  [0.0, 0.4, 1.0]
    mycmap.append( LinearSegmentedColormap.from_list("mycmap3", list(zip(nodes, colors))) )

    colors = [(1.0,1.0,1.0,0.0), (1.0,0.8,0.4,0.5), (1.0,0.6,0.2,0.9)]
    nodes =  [0.0, 0.4, 1.0]
    mycmap.append( LinearSegmentedColormap.from_list("mycmap4", list(zip(nodes, colors))) )

    # qtldata = [ line.split()  for line in open('soc.qtl', 'r')]
    # efermi = float( qtldata[2][-1] )*ry2ev

    # qtldata = qtldata[8:]

    # qtl = [ qtldata[ibnd*(nkpt*(natm+1)+1)+ikpt*(natm+1)+iatm+1][3:] for ibnd in range(nbnd) for ikpt in range(nkpt) for iatm in range(natm)]
    # eigs = np.array([float(qtldata[ibnd*(nkpt*(natm+1)+1)+ikpt*(natm+1)+1][0])*ry2ev for ibnd in range(nbnd) for ikpt in range(nkpt)]).reshape(nbnd,nkpt) - efermi

    # projection = []
    # projection.append( [float(qtl[ibnd*nkpt*natm+ikpt*natm+1][4]) for ibnd in range(nbnd) for ikpt in range(nkpt)] )
    # projection.append( [float(qtl[ibnd*nkpt*natm+ikpt*natm+2][1])+float(qtl[ibnd*nkpt*natm+ikpt*natm+3][1]) for ibnd in range(nbnd) for ikpt in range(nkpt)] )
    # projection.append( [float(qtl[ibnd*nkpt*natm+ikpt*natm+0][4]) for ibnd in range(nbnd) for ikpt in range(nkpt)] )
    # projection.append( [float(qtl[ibnd*nkpt*natm+ikpt*natm+0][7]) for ibnd in range(nbnd) for ikpt in range(nkpt)] )

    # projection = np.array(projection).reshape(-1,nbnd,nkpt)
    # for ip in projection:
    #     print(ip.min(), ip.max())
    # proj_norm = [ plt.Normalize(ip.min(), ip.max()) for ip in projection ]

    # kpts = np.loadtxt('soc.spaghetti_ene', usecols=(3), comments="bandindex")[:nkpt]

    nkpt   = 201

    kpts, eigs, pj12 = np.loadtxt('socwan_dx2-y2_up.dat', usecols=(0,1,2), unpack=True)
    pj32 = np.loadtxt('socwan_dx2-y2_dn.dat', usecols=(2), unpack=True)

    kpts = kpts[:nkpt]
    eigs = eigs.reshape(-1,nkpt)
    pj12 = pj12.reshape(-1,nkpt)
    pj32 = pj32.reshape(-1,nkpt)

    pn12 = plt.Normalize(pj12.min(), pj12.max())
    pn32 = plt.Normalize(pj32.min(), pj32.max())

    fig, ax = plt.subplots(figsize=(5,5))

    for ibnd, eig in enumerate(eigs):
        points = np.array([kpts, eig]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        ax.plot(kpts, eig, ls='-', color=band_color, lw=1.2, zorder=0)
        lc = LineCollection(segments, array=pj32[ibnd], cmap=mycmap[1], norm=pn32, linewidth=1.5, linestyles='-')
        ax.add_collection(lc)
        lc = LineCollection(segments, array=pj12[ibnd], cmap=mycmap[0], norm=pn12, linewidth=1.5, linestyles='-')
        ax.add_collection(lc)

    ax.set_ylim(-2, 2)
    ax.set_yticks(np.linspace(ax.get_ylim()[0],ax.get_ylim()[1], num=5))
    ax.set_yticks(np.linspace(ax.get_ylim()[0],ax.get_ylim()[1], num=21), minor=True)
    ax.set_xlim(kpts.min(), kpts.max())
    ax.set_xticks([0.00000,0.778863499,1.557726997])
    ax.set_xticklabels([r"K", r"$\Gamma$", r"K"])

    ax.axhline(0.0, ls='--', color='gray', lw=1)
    for itick in ax.get_xticks()[1:-1]:
        ax.plot([itick, itick], [ax.get_ylim()[0], ax.get_ylim()[1]],
            linestyle='--', linewidth=1, color='grey')

    plt.tight_layout()

    plt.savefig('socwan_dx2-y2.pdf')

