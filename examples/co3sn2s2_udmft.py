'''
Co3Sn2S2 SOC-FM Wannier TB -> ksum/DMFT input (eig.dat, udmft.dat, eon/von/htmat.dat)

Ported from Co3Sn2S2/202606/qe/socfm_try2revocer/udmft.py.
Wannier order (co3sn2s2_hr.dat, spin interleaved, 58 orbitals):
    Co1/Co2/Co3 3d : 0-9 / 10-19 / 20-29
    Sn1 s+p : 30-37,  Sn2 s+p : 38-45,  S1 p : 46-51,  S2 p : 52-57

Run in the directory holding co3sn2s2_hr.dat:
    mpirun -np N python /path/to/swept/examples/co3sn2s2_udmft.py
'''

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # parent of the swept package

import numpy as np
from swept import mpiutils as mpi
from swept.kpoint import Kpoint
from swept.model import Model
from swept.udmft import local_eigenbasis, onsite_block_mismatch, dump_local_basis, dump_dmft, dump_kpoints

np.set_printoptions(precision=3, suppress=True, linewidth=240)

a = 5.3689; c = 13.176
rvec = np.array([
    [ a/2, a*np.sqrt(3.0)/6,  c/3],
    [-a/2, a*np.sqrt(3.0)/6,  c/3],
    [ 0.0,-a*np.sqrt(3.0)/3,  c/3]
])
bvec = 2*np.pi*np.linalg.inv(rvec).T

tbmodel = Model.from_hr('co3sn2s2_hr.dat')
tbmodel.shift_onsite(eshift=-(12.5776))

local_zx = np.array([
   [[+0.0,+0.0,+1.0],[+0.8660254038,+0.5000000000,+0.0000000000]],
   [[+0.0,+0.0,+1.0],[-0.8660254038,+0.5000000000,+0.0000000000]],
   [[+0.0,+0.0,+1.0],[+0.0000000000,-1.0000000000,+0.0000000000]],
   [[+0.0,+0.0,+1.0],[+1.0000000000,+0.0000000000,+0.0000000000]],
   [[+0.0,+0.0,+1.0],[+1.0000000000,+0.0000000000,+0.0000000000]],
   [[+0.0,+0.0,-1.0],[+0.8660254038,-0.5000000000,+0.0000000000]],
   [[+0.0,+0.0,+1.0],[-0.8660254038,+0.5000000000,+0.0000000000]]
])
idxs = np.array([0, 10, 20, 32, 40, 46, 52])
lls = [2, 2, 2, 1, 1, 1, 1]
tbmodel.rotate_spin(lnum=lls, zxmat=local_zx, index=idxs)

# Local eigenbasis of Co1; the same von is applied to Co2/Co3 in dump_dmft.
norb = 10; natm = 3
h00 = tbmodel.ham[(0, 0, 0)]
eon, von, htmat, mz = local_eigenbasis(h00[:norb, :norb], ll=2)

if mpi.rank0:
    print(" max |H_loc(Co_i) - H_loc(Co1)| = %.3e eV" % onsite_block_mismatch(h00, norb, natm))
    print(" max |H_up,dn| in H_loc        = %.6f eV" % np.max(np.abs(h00[:norb:2, 1:norb:2])))
    print(" CF spin off-diagonal block    = %.6f" % np.max(np.abs(htmat[5:, :5])))
    print(" Mz per solver flavour         =", np.diag(mz).real)

kpts = Kpoint.gen_uniform_kpoints((12, 12, 12))

# hskp = [
#         [0.50000000,  0.24952000,  0.75048000],
#         [0.37476000,  0.37476000,  0.75048000],
#         [0.00000000,  0.00000000,  0.50000000],
#         [0.00000000,  0.00000000,  0.00000000],
#         [0.50000000,  0.50000000,  0.50000000]]
# kwant = 1000
# kpts = Kpoint.from_high_symmetry_points(khsym_frac=hskp, bzvec=bvec, kwant=kwant)

eigs, vecs = tbmodel.get_bulk_eigens(kpts.kfrac)

if mpi.rank0:
    # dump_kpoints(kpts, fname='kpoints.dat')
    dump_local_basis(eon, von, htmat)
    dump_dmft(eigs, vecs, 0.0, ntot=43.0, norb=norb, natm=natm, lnum=2, von=von)
    # np.save("Mz_eimp_blocked.npy", mz)
