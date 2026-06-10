'''
Author: Jianzhou Zhao
Date: 2025-05-24 17:09:48
LastEditors: your name
LastEditTime: 2025-05-26 10:50:46
Description: file content
'''

def dump_dmft(eig,vec,mu):

    nkpt = vec.shape[0]
    nbnd = vec.shape[2]
    norb = 10
    natm = 3
    ntot = 43.0
    # mu   = 0.00

    # Print eigen values
    feig = open("eig.dat","w")
    output_format="{:5d}{:5d}{:5d}{:5d}{:5d}{:50.35s}\n"
    feig.write(output_format.format(nkpt,natm,norb,2,2,"     # nkpt, natm, norb, L, nspin"))
    output_format="{:5d}{:5d}{:5d}{:10.5f}{:10.5f}{:50.40s}\n"
    feig.write(output_format.format(1,nbnd,nbnd,ntot,mu,"     # nemin, nemax, nbands, ntotal, mu"))
    feig.write('\n')
    kw_format="{:5d}{:15.7f}{:31.20s}\n"
    output_format="{:5d}{:15.8f}\n"
    for ikpt in range(nkpt):
        feig.write(kw_format.format(ikpt+1,1.0/float(nkpt)," # ikpt, k-weight"))
        for ibnd in range(nbnd):
            feig.write(output_format.format(ibnd+1,eig[ikpt,ibnd]))
        feig.write('\n')
        feig.write('\n')
    feig.close()

    fprj = open("udmft.dat","w")
    output_format="{:5d}{:5d}{:5d}{:5d}{:5d}{:50.35s}\n"
    fprj.write(output_format.format(nkpt,natm,norb,2,2,"     # nkpt, natm, norb, L, nspin"))
    output_format="{:5d}{:5d}{:5d}{:10.5f}{:10.5f}{:50.40s}\n"
    fprj.write(output_format.format(1,nbnd,nbnd,ntot,mu,"     # nemin, nemax, nbands, ntotal, mu"))
    fprj.write('\n')
    kw_format="{:5d}{:15.7f}{:31.20s}\n"
    output_format="{:5d}{:5d}{:15.8f}{:15.8f}\n"
    for ikpt in range(nkpt):
        fprj.write(kw_format.format(ikpt+1,1.0/float(nkpt)," # ikpt, k-weight"))
        vmat = vec[ikpt,:norb*natm,:].conj().T
        for iorb in range(norb*natm):
            for ibnd in range(nbnd):
                fprj.write(output_format.format(ibnd+1,iorb+1,vmat[ibnd,iorb].real, vmat[ibnd,iorb].imag))
        fprj.write('\n')
        fprj.write('\n')
    fprj.close()

def dump_kpoints(kpts, fname='kpoints.dat'):
    """
    Dump k-points to a file.
    """
    kreal = Kpoint.k_length_accumulate(kpts.kreal)
    with open(fname, "w") as f:
        for ikpt, kpt in enumerate(kpts.kfrac):
            f.write("{:5d} {:15.9f} {:15.9f} {:15.9f} {:15.9f}\n".format(ikpt + 1, kpt[0], kpt[1], kpt[2], kreal[ikpt]))

if __name__ == "__main__":

    from kpoint import Kpoint
    from model import Model
    import mpiutils as mpi
    import numpy as np

    a = 5.378080; c = 4.391283
    rvec = np.array([
        [-a,   0.0,  0.0],
        [ a/2, a*np.sqrt(3.0)/2,  0.0],
        [ 0.0, a*np.sqrt(3.0)/3,  c  ]
    ])
    bvec = 2*np.pi*np.linalg.inv(rvec).T

    tbmodel = Model.from_hr('co3sn2s2_up_hr.dat')
    tbmodel_dn = Model.from_hr('co3sn2s2_dn_hr.dat')
    tbmodel.addspinbasis(hown=tbmodel_dn.ham)

    local_zx = np.array([
       [[+0.0,+0.0,+1.0],[+0.0000000000,-1.0000000000,+0.0000000000]],
       [[+0.0,+0.0,+1.0],[-0.8660254038,+0.5000000000,+0.0000000000]],
       [[+0.0,+0.0,+1.0],[+0.8660254038,+0.5000000000,+0.0000000000]]
    ])
    tbmodel.rotate_spin(lnum=2, zxmat=local_zx)

    # tmat = np.zeros((tbmodel.nwan,tbmodel.nwan), dtype=complex)
    # for iwan in range(tbmodel.nwan//2):
    #     tmat[ 2*iwan,   iwan ] = 1.0
    #     tmat[ 2*iwan+1, iwan + tbmodel.nwan//2 ] = 1.0

    # tbmodel.rotate_basis(rmat=tmat)

    kpts = Kpoint.gen_uniform_kpoints((7, 8, 7))

    # hskp = [
    #        [-1.0/3.0,  2.0/3.0, 1.0/3.0],
    #        [ 0.0,      0.0,     0.0],
    #        [ 0.0,      0.0,    -0.5]]
    # kwant = 2000
    # kpts = Kpoint.from_high_symmetry_points(khsym_frac=hskp, bzvec=bvec, kwant=kwant)

    eigs, vecs = tbmodel.get_bulk_eigens(kpts.kfrac)

    if mpi.rank0:
        # dump_kpoints(kpts, fname='kpoints.dat')
        # print(eigs.shape, vecs.shape)
        dump_dmft(eigs, vecs, 12.5800)
