'''
Author: Jianzhou Zhao
Date: 2024-01-11 10:31:45
LastEditors: your name
LastEditTime: 2024-01-17 14:48:36
Description: file content
'''

import sys

def plot_total_dos(omega, tdos, file='dos.pdf', dpi=100, xlim=None, ylim=None, linewidth=1.0, color='red', figsize=(6, 5), ef=0.0):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(omega, tdos, linewidth=linewidth, color=color)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.axvline( 0, linestyle='--', color='grey')
    plt.savefig(file, dpi=dpi)

def dump_total_dos(omega, tdos, pdos, fname, names=None):
    output_format = "{:15.9f}{:15.9f}"+"{:15.9f}"*len(pdos[0])+"\n"
    with open(fname,"w") as fdos:
        if names is not None:
            fdos.write("# E-mu  total  " + "  ".join(names) + "\n")
        for img, ene in enumerate(omega):
            fdos.write(output_format.format(ene, tdos[img], *pdos[img]))

if __name__ == "__main__":

    from kpoint import Kpoint
    from model import Model
    import mpiutils as mpi
    import numpy as np
    from basis import p2t2g, complex2real, clebsch_gordan

    proj_list = [
        [0,1,6,7,12,13,18,19,24,25,30,31,36,37,42,43,48,49,54,55,60,61,66,67], # Ir5d J1/2
        [2,3,4,5, 8,9,10,11, 14,15,16,17, 20,21,22,23, 26,27,28,29, 32,33,34,35, 38,39,40,41, 44,45,46,47, 50,51,52,53, 56,57,58,59, 62,63,64,65, 68,69,70,71] # Ir5d J3/2
    ]
    # if mpi.rank0 : print(proj_list, '\n')

    local_zx = np.array([
       [[+0.0321610403,-0.0134404347,+0.9993923265],[-0.7506551577,+0.6598680157,+0.0330308357]],
       [[-0.0321610403,-0.0134404347,+0.9993923265],[+0.7509038507,-0.6602346301,+0.0152852970]],
       [[-0.0321610403,+0.0134404347,+0.9993923265],[+0.7506551577,-0.6598680157,+0.0330308357]],
       [[+0.0321610403,+0.0134404347,+0.9993923265],[-0.7509038507,+0.6602346301,+0.0152852970]],
       [[+0.9993923265,+0.0321610403,-0.0134404347],[+0.0330308357,-0.7506551577,+0.6598680157]],
       [[+0.9993923265,-0.0321610403,-0.0134404347],[+0.0152852970,+0.7509038507,-0.6602346301]],
       [[+0.9993923265,-0.0321610403,+0.0134404347],[+0.0330308357,+0.7506551577,-0.6598680157]],
       [[+0.9993923265,+0.0321610403,+0.0134404347],[+0.0152852970,-0.7509038507,+0.6602346301]],
       [[-0.0134404347,+0.9993923265,+0.0321610403],[+0.6598680157,+0.0330308357,-0.7506551577]],
       [[-0.0134404347,+0.9993923265,-0.0321610403],[-0.6602346301,+0.0152852970,+0.7509038507]],
       [[+0.0134404347,+0.9993923265,-0.0321610403],[-0.6598680157,+0.0330308357,+0.7506551577]],
       [[+0.0134404347,+0.9993923265,+0.0321610403],[+0.6602346301,+0.0152852970,-0.7509038507]]
    ])

    tbmodel = Model.from_hr('la3ir3o11_hr.dat')
    kpts = Kpoint.gen_uniform_kpoints((8, 8, 8))

    tmat = np.dot(complex2real(1,2), p2t2g())
    tmat = np.dot(clebsch_gordan(1), tmat)
    prmat = np.eye(tbmodel.nwan, dtype=complex)
    for iatm in range(12):
        prmat[iatm*6:iatm*6+6,iatm*6:iatm*6+6] = tmat

    tbmodel.rotate_spin(lnum=1, zxmat=local_zx)
    if mpi.rank0 : print(np.diag(tbmodel.ham[(0,0,0)][0:6,0:6].real), '\n')
    
    tbmodel.rotate_basis(rmat=prmat)
    if mpi.rank0 : print(np.diag(tbmodel.ham[(0,0,0)][0:6,0:6].real), '\n')

    omega, tdos, pdos = tbmodel.get_proj_dos(kpts.kfrac, proj=proj_list,  emin=-3.0, emax=1.0, ne=1001, mu=11.8928, sigma=0.02)

    if mpi.rank0:
        dump_total_dos(omega, tdos, pdos, 'socpdos.dat')

