#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jianzhou Zhao
@Date: 2018-11-21 19:05:50
LastEditors: your name
LastEditTime: 2025-05-26 10:47:09
@Description: File content
'''

import mpiutils as mpi
import numpy as np
import scipy.linalg as la
import sys
from collections import defaultdict

def extract_block_diag(A,M,k=0):
    """Extracts blocks of size M from the kth diagonal
    of square matrix A, whose size must be a multiple of M."""

    # Check that the matrix can be block divided
    if A.shape[0] != A.shape[1] or A.shape[0] % M != 0:
        raise ValueError('Matrix must be square and a multiple of block size')

    # Assign indices for offset from main diagonal
    if abs(k) > M - 1:
        raise ValueError('kth diagonal does not exist in matrix')
    elif k > 0:
        ro = 0
        co = abs(k)*M 
    elif k < 0:
        ro = abs(k)*M
        co = 0
    else:
        ro = 0
        co = 0

    blocks = np.array([A[i+ro:i+ro+M,i+co:i+co+M] 
                       for i in range(0,len(A)-abs(k)*M,M)])
    return blocks

class Model():

    def __init__(self, nwan=None, nrpt=None, deg=None, ham=None):
        self.nwan = nwan
        self.nrpt = nrpt
        self.deg = deg
        self.ham = ham
        self._rstack = None
        self._hstack = None
        return

    @classmethod
    def from_hr(cls, hr_file):

        if mpi.rank == 0:
            with open(hr_file, 'r') as f:
                next(f)  # skip title line
                nwan = int(next(f))
                nrpt = int(next(f))

                # read degeneration of R points
                deg = []
                for _, line in zip(range(int(np.ceil(nrpt / 15))), f):
                    deg.extend(int(i) for i in line.split())

                assert len(deg) == nrpt

                # read rpt and hamr
                raw_list = [line.split() for line in f]

                nwan_square = nwan**2
                hr_list = defaultdict(list)
                for num, line in enumerate(raw_list):
                    Rvec = tuple(map(int, line[:3]))
                    hr_list[Rvec].append(
                        (float(line[5]) + 1j * float(line[6])) / deg[num // nwan_square])

            for key, array in hr_list.items():
                hr_list[key] = np.array([array[iwan::nwan] for iwan in range(nwan)])
        else:
            nwan = 0
            nrpt = 0
            deg = None
            hr_list = None

        nwan = mpi.bcast(nwan, root=0)
        nrpt = mpi.bcast(nrpt, root=0)
        deg = mpi.bcast(deg, root=0)
        hr_list = mpi.bcast(hr_list, root=0)

        return cls(nwan=nwan, nrpt=nrpt, deg=deg, ham=hr_list)
    
    def to_hr(self, hr_file):
        import datetime
        wtitle = " Wannier HR file written by SWEPT @ "+str(datetime.datetime.now())+"\n"
        if mpi.rank == 0:
            with open(hr_file, 'w') as f:
                f.write(wtitle)
                f.write("{:12d}\n".format(self.nwan))
                f.write("{:12d}\n".format(self.nrpt))
                
                for irpt in range(self.nrpt):
                    if irpt % 15 == 14 or irpt == self.nrpt-1:
                        f.write("{:5d}\n".format(self.deg[irpt]))
                    else:
                        f.write("{:5d}".format(self.deg[irpt]))
                
                for irpt, (Rvec, Hvec) in enumerate(self.ham.items()):
                    for jwan in range(self.nwan):
                        for iwan in range(self.nwan):
                            rpart = Hvec[iwan][jwan].real * self.deg[irpt]
                            ipart = Hvec[iwan][jwan].imag * self.deg[irpt]
                            f.write("{:5d}{:5d}{:5d}{:5d}{:5d}{:12.6f}{:12.6f}\n".format(*Rvec,iwan+1,jwan+1,rpart,ipart))
        return
    
    def addspinbasis(self, hown=None):
        ns = 2
        self.nwan = self.nwan * ns
        for Rvec, Hvec in self.ham.items():
            hmat = np.zeros((self.nwan,self.nwan),dtype=complex)
            hmat[0::2,0::2] = Hvec
            hmat[1::2,1::2] = hown[Rvec] if hown is not None else Hvec
            self.ham[Rvec]  = hmat
        self._hstack = None

    def rotate_spin(self, lnum, zxmat):
        from spinor import rotate_spinor

        # np.set_printoptions(linewidth=180, precision=3)
        natm = len(zxmat)
        ns = 2; nbnd = 2*lnum+1; norb = ns*nbnd

        smat = np.eye(self.nwan, dtype=complex)
        for iatm in range(natm):
            dmat = np.kron(np.eye(nbnd), rotate_spinor(*zxmat[iatm]))
            smat[iatm*norb:(iatm+1)*norb,iatm*norb:(iatm+1)*norb] = dmat

        for Rvec,Hvec in self.ham.items():
            self.ham[Rvec] = np.dot(smat.conj().T, np.dot(Hvec,smat))
        self._hstack = None

        return

    def soc(self, asoc, lnum, sidx, rot_soc=None):

        import atom_soc
        import basis
        
        ns = 2; nbnd = 2*lnum+1; norb = ns*nbnd

        # Complex to real basis
        wbas = basis.complex2real(lnum,ns)
        # SOC in complex basis
        cmat = atom_soc.getsoc(lnum,asoc)

        # SOC in real basis
        cmat = np.dot(la.inv(wbas),np.dot(cmat,wbas))
        # CG basis to real basis
        # tbas = np.dot(basis.clebsch_gordan(lnum), wbas)
        # if there are external rotation basis
        if rot_soc is not None:
            # tbas = np.dot( tbas, rot_soc )
            cmat = np.dot( la.inv(rot_soc), np.dot(cmat, rot_soc) )

        # if mpi.rank0:
        #     h0 = self.ham[(0,0,0)][sidx:sidx+norb,sidx:sidx+norb]
        #     print(h0.real, '\n')

        # Add SOC
        self.ham[(0,0,0)][sidx:sidx+norb,sidx:sidx+norb] += cmat
        self._hstack = None

        # Test H0
        # if mpi.rank0:
        #     test = np.dot( np.dot(tbas, self.ham[(0,0,0)][sidx:sidx+norb,sidx:sidx+norb]), la.inv(tbas))
        #     print(test.real, '\n')

        return

    def shift_onsite(self, eshift, index=None):
        if index is None:
            self.ham[(0,0,0)] += np.diag(np.repeat(eshift,self.nwan))
        else :
            hmat = np.zeros(self.nwan)
            hmat[index] = eshift
            self.ham[(0,0,0)] += np.diag(hmat)
        self._hstack = None
        return
    
    def involve_zmat(self, zmat):
        from scipy.linalg import block_diag
        for Rvec, Hvec in self.ham.items():
            if Rvec == (0,0,0) :
                dmat = extract_block_diag(np.array(Hvec), M=6)
                dmat = block_diag( *dmat )
                hmat = Hvec - dmat
                self.ham[Rvec] = np.dot(zmat, np.dot(hmat, zmat)) + dmat
            else :
                self.ham[Rvec] = np.dot(zmat, np.dot(Hvec, zmat))
        self._hstack = None
        return
    
    def rotate_basis(self, rmat):
        from numpy.linalg import inv
        for Rvec, Hvec in self.ham.items():
            self.ham[Rvec] = np.dot(np.dot(rmat, Hvec), inv(rmat))
        self._hstack = None
        return
    
    def _stack_ham(self):
        # cached (nrpt,3) R vectors and (nrpt,nwan,nwan) hamiltonian stack;
        # every method that modifies self.ham must reset self._hstack to None
        if self._hstack is None:
            self._rstack = np.array(list(self.ham.keys()), dtype=float)
            self._hstack = np.array([np.asarray(Hvec, dtype=complex) for Hvec in self.ham.values()])
        return self._rstack, self._hstack

    def get_bulk_Hk(self, Kvec):
        rstack, hstack = self._stack_ham()
        phase = np.exp(2j * np.pi * np.dot(rstack, Kvec))
        return np.tensordot(phase, hstack, axes=(0, 0))

    def get_slab_Hk(self, Kvec, nlay, surf_pot=0.0):
        hamk = np.zeros((self.nwan*nlay,self.nwan*nlay),dtype=complex)
        for Rvec, Hvec in self.ham.items():
            ratio = np.exp(2j * np.pi * np.dot(Rvec[:2], Kvec[:2]))
            for ilay in range(nlay):
                jlay = ilay + Rvec[2]
                if (jlay >=  0 ) and (jlay < nlay) :
                    hamk[ilay*self.nwan:(ilay+1)*self.nwan,jlay*self.nwan:(jlay+1)*self.nwan] += ratio * np.asarray(Hvec,dtype=complex)
        for iwan in range(self.nwan):
            hamk[ iwan  , iwan  ] += surf_pot
            hamk[-iwan-1,-iwan-1] += surf_pot
        return hamk

    def get_bulk_eigval(self, Kvec):
        return la.eigvalsh(self.get_bulk_Hk(Kvec))
    
    def get_bulk_evec(self, Kvec):
        return la.eigh(self.get_bulk_Hk(Kvec))

    def get_bulk_eigvals(self, klist):
        kmpi,kcount,_ = mpi.devide_array(klist, root=0)
        empi = np.array([self.get_bulk_eigval(kvec) for kvec in kmpi])
        eigs = np.zeros((len(klist), self.nwan), dtype=float)
        ecount = np.array(kcount)*self.nwan
        edisp  = np.insert(np.cumsum(ecount),0,0)[0:-1]
        mpi.Allgatherv(empi,eigs,ecount,edisp)
        return eigs.T
    
    def get_bulk_eigens(self, klist):
        kmpi,kcount,_ = mpi.devide_array(klist, root=0)
        empi = np.zeros((len(kmpi), self.nwan), dtype=float)
        vmpi = np.zeros((len(kmpi), self.nwan, self.nwan), dtype=complex)
        for ikp, kvec in enumerate(kmpi):
            empi[ikp], vmpi[ikp] = self.get_bulk_evec(kvec)

        eigs = np.zeros((len(klist), self.nwan), dtype=float)
        vecs = np.zeros((len(klist), self.nwan, self.nwan), dtype=complex)
        ecount = np.array(kcount)*self.nwan
        edisp  = np.insert(np.cumsum(ecount),0,0)[0:-1]
        # complex128 = 2 MPI.DOUBLE, gather eigenvectors as float views
        vcount = ecount*self.nwan*2
        vdisp  = np.insert(np.cumsum(vcount),0,0)[0:-1]
        mpi.Allgatherv(empi,eigs,ecount,edisp)
        mpi.Allgatherv(vmpi.view(float),vecs.view(float),vcount,vdisp)
        return eigs, vecs

    def get_bulk_eprojs(self, klist, plist):
        kmpi,kcount,_ = mpi.devide_array(klist, root=0)
        empi = np.zeros((len(kmpi), self.nwan), dtype=float)
        pmpi = np.zeros((len(kmpi), self.nwan), dtype=float)
        for ikp, kvec in enumerate(kmpi):
            eig, vec = self.get_bulk_evec(kvec)
            empi[ikp] = eig
            pmpi[ikp] = np.sum(np.abs(vec[plist,:])**2, axis=0)

        eigs  = np.zeros((len(klist), self.nwan), dtype=float)
        projs = np.zeros((len(klist), self.nwan), dtype=float)
        ecount = np.array(kcount)*self.nwan
        edisp  = np.insert(np.cumsum(ecount),0,0)[0:-1]
        mpi.Allgatherv(empi,eigs,ecount,edisp)
        mpi.Allgatherv(pmpi,projs,ecount,edisp)
        return eigs.T, projs.T

    def get_slab_eigvals(self, klist, nlay, surf_pot=0.0):
        kmpi,kcount,_ = mpi.devide_array(klist, root=0)
        empi = np.array([la.eigvalsh(self.get_slab_Hk(kvec,nlay,surf_pot=surf_pot)) for kvec in kmpi])
        eigs = np.zeros((len(klist),self.nwan*nlay))
        ecount = np.array(kcount)*self.nwan*nlay
        edisp  = np.insert(np.cumsum(ecount),0,0)[0:-1]
        mpi.Allgatherv(empi,eigs,ecount,edisp)
        return eigs.T
    
    def get_slab_eprojs(self, klist, nlay, surf_pot=0.0):
        kmpi,kcount,_ = mpi.devide_array(klist, root=0)
        empi = []; tmpi = []; bmpi = []
        for kvec in kmpi:
            eig, vec = la.eigh(self.get_slab_Hk(kvec,nlay,surf_pot=surf_pot))
            empi.append(eig)
            tmpi.append( np.diag(np.dot(vec[:self.nwan,:].conj().T,vec[:self.nwan,:])).real )
            bmpi.append( np.diag(np.dot(vec[-self.nwan:,:].conj().T,vec[-self.nwan:,:])).real )
        
        eigs = np.zeros((len(klist),self.nwan*nlay))
        ptop = np.zeros((len(klist),self.nwan*nlay))
        pbot = np.zeros((len(klist),self.nwan*nlay))
        ecount = np.array(kcount)*self.nwan*nlay
        edisp  = np.insert(np.cumsum(ecount),0,0)[0:-1]
        mpi.Allgatherv(np.array(empi),eigs,ecount,edisp)
        mpi.Allgatherv(np.array(tmpi),ptop,ecount,edisp)
        mpi.Allgatherv(np.array(bmpi),pbot,ecount,edisp)
        return eigs.T, ptop.T, pbot.T

    def get_total_dos(self, klist, emin=-10.0, emax=10.0, ne=101, mu=0.0, sigma=0.1):

        from scipy.stats import norm

        omega = np.linspace(emin, emax, num=ne)
        kmpi,_,_ = mpi.devide_array(klist, root=0)

        tdos_mpi = np.zeros_like(omega)
        for ikp, kpt in enumerate(kmpi):
            if mpi.rank0 : print(ikp, len(kmpi))
            sys.stdout.flush()
            eig = la.eigvalsh(self.get_bulk_Hk(kpt))
            for img, ene in enumerate(omega):
                tdos_mpi[img] += np.sum(norm.pdf(eig,loc=mu+ene,scale=sigma)) / float(len(klist))
        tdos = mpi.allreduce(tdos_mpi)
        return omega, tdos
    
    def get_proj_dos(self, klist, proj=None, emin=-10.0, emax=10.0, ne=101, mu=0.0, sigma=0.1):

        from scipy.stats import norm

        omega = np.linspace(emin, emax, num=ne)
        kmpi,_,_ = mpi.devide_array(klist, root=0)

        tdos_mpi = np.zeros_like(omega)
        pdos_mpi = np.zeros((len(omega),len(proj)))
        for ikp, kpt in enumerate(kmpi):
            # if mpi.rank0 : print(ikp, len(kmpi))
            sys.stdout.flush()
            eig,vec = la.eigh(self.get_bulk_Hk(kpt))
            pmat = []
            for ipr in proj:
                pmat.append( np.diag(np.dot(vec[ipr,:].conj().T, vec[ipr,:]).real) )
            for img, ene in enumerate(omega):
                dmat = norm.pdf(eig,loc=mu+ene,scale=sigma)
                tdos_mpi[img] += np.sum(dmat) / float(len(klist))
                for ipr, prj in enumerate(pmat):
                    pdos_mpi[img, ipr] += np.sum(np.multiply(dmat,prj)) / float(len(klist))
        tdos = mpi.allreduce(tdos_mpi)
        pdos = mpi.allreduce(pdos_mpi)
        return omega, tdos, pdos

    def get_slab_dos(self, klist, nlay, surf_pot=0.0, emin=-10.0, emax=10.0, ne=101, mu=0.0, sigma=0.1):

        from scipy.stats import norm

        omega = np.linspace(emin, emax, num=ne)
        kmpi,_,_ = mpi.devide_array(klist, root=0)

        tdos_mpi = np.zeros_like(omega)
        for ikp, kpt in enumerate(kmpi):
            if mpi.rank0 : print(ikp, len(kmpi))
            sys.stdout.flush()
            eig = la.eigvalsh(self.get_slab_Hk(kpt,nlay,surf_pot=surf_pot))
            for img, ene in enumerate(omega):
                tdos_mpi[img] += np.sum(norm.pdf(eig,loc=mu+ene,scale=sigma)) / float(len(klist))
        tdos = mpi.allreduce(tdos_mpi)
        return omega, tdos
    
    def get_slab_surfdos(self, klist, nlay, proj, surf_pot=0.0, emin=-10.0, emax=10.0, ne=101, mu=0.0, sigma=0.1):

        from scipy.stats import norm

        omega = np.linspace(emin, emax, num=ne)
        kmpi,_,_ = mpi.devide_array(klist, root=0)

        tdos_mpi = np.zeros_like(omega)
        pdos_mpi = np.zeros((len(omega),len(proj)))
        for ikp, kpt in enumerate(kmpi):
            if mpi.rank0 : print(ikp, len(kmpi))
            sys.stdout.flush()
            eig,vec = la.eigh(self.get_slab_Hk(kpt,nlay,surf_pot=surf_pot))
            sys.stdout.flush()
            for img, ene in enumerate(omega):
                dos = norm.pdf(eig,loc=mu+ene,scale=sigma)
                tdos_mpi[img] += np.sum(np.diag(dos).real) / float(len(klist))
                for ipr,pr in enumerate(proj):
                    ptmp = np.dot( np.dot(vec[pr,:], np.diag(dos)), vec[pr,:].T.conj() )
                    pdos_mpi[img,ipr] += np.sum(np.diag(ptmp).real) / float(len(klist))
        tdos = mpi.allreduce(tdos_mpi)
        pdos = mpi.allreduce(pdos_mpi)
        return omega, tdos, pdos
    
    def get_slab_surfiso(self, klist, nlay, ene=0.0, surf_pot=0.0, mu=0.0, sigma=0.05):

        from scipy.stats import norm
        kmpi,_,_ = mpi.devide_array(klist, root=0)
        
        pass

# 待实现：现版本是坏的——devide_array 返回三元组只接了一个变量、
# self.get_eigval 不存在（应为 get_bulk_eigval）、mpi.Gatherv 在 mpiutils 里也不存在。
# 真正的 JDOS 还需要在能量网格上对 (E_n(k)-E_m(k)) 做直方图/展宽，以后再做。
#    def get_jdos(self, klist):
#        kmpi = mpi.devide_array(klist, root=0)
#        empi = np.array([self.get_eigval(kvec) for kvec in kmpi])
#        eigs = np.zeros((len(klist), self.nwan), dtype=float)
#        mpi.Gatherv(empi, [eigs, len(empi.flatten()),float], root=0)
#        return eigs.T

if __name__ == '__main__':

    pass
    # tbm = Model().from_hr('euzn2as2_hr.dat')
    # tbm.to_hr("test.dat")
    # print(np.diag(tbm.ham[(0,0,0)]).real)
    # tbm.soc(asoc=0.0, lnum=1, sidx=0, addspin=True,spin=(0.0,0.0,1.0))
    # print(np.diag(tbm.ham[(0,0,0)]).real)

