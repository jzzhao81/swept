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

    def rotate_spin(self, lnum, zxmat, index=None):
        # lnum : 每个块的角动量, 整数(所有块相同)或列表
        # index: 每个块的起始轨道序号; None 时从 0 开始连续排列(旧接口)
        # 块内轨道按 (m, spin) 自旋交错排列, 大小 2*(2l+1)
        from spinor import rotate_spinor

        ns = 2
        natm = len(zxmat)
        lnum = np.broadcast_to(np.asarray(lnum, dtype=int), (natm,)) if np.ndim(lnum) == 0 \
            else np.asarray(lnum, dtype=int)
        if len(lnum) != natm:
            raise ValueError("lnum and zxmat must have the same length")
        sizes = ns*(2*lnum+1)
        if index is None:
            index = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        index = np.asarray(index, dtype=int)
        if len(index) != natm:
            raise ValueError("index and zxmat must have the same length")

        used = np.zeros(self.nwan, dtype=bool)
        smat = np.eye(self.nwan, dtype=complex)
        for iatm in range(natm):
            ist, ied = index[iatm], index[iatm] + sizes[iatm]
            if ist < 0 or ied > self.nwan:
                raise ValueError(f"spin block {iatm} [{ist}:{ied}] out of range (nwan={self.nwan})")
            if used[ist:ied].any():
                raise ValueError(f"spin block {iatm} [{ist}:{ied}] overlaps another block")
            used[ist:ied] = True
            dmat = np.kron(np.eye(2*lnum[iatm]+1), rotate_spinor(*zxmat[iatm]))
            smat[ist:ied, ist:ied] = dmat

        for Rvec,Hvec in self.ham.items():
            self.ham[Rvec] = np.dot(smat.conj().T, np.dot(Hvec,smat))
        self._hstack = None

        return

    def set_spin_axis(self, axis):
        # 当前基(自旋交错)的 ↑/↓ 沿全局笛卡尔方向 axis 量子化(如共线磁计算的磁矩方向),
        # 变换到沿全局 z 量子化的基, 之后 soc / rotate_spin 均以全局 z 为参考。
        # rotate_spinor(m, x) 的列是 m 方向的 |↑>,|↓> 在 z 基中的表示 D,
        # H_z = D H_m D^†, 即 S^† H S, S = 1 ⊗ D^†。x 只影响自旋的规范相位。
        from spinor import rotate_spinor

        if self.nwan % 2 != 0:
            raise ValueError("set_spin_axis needs a spinful (interleaved) model")
        m = np.asarray(axis, dtype=float)
        m = m / np.linalg.norm(m)
        ref = np.array([0.0, 0.0, 1.0]) if abs(m[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x = ref - np.dot(ref, m) * m
        x = x / np.linalg.norm(x)

        smat = np.kron(np.eye(self.nwan//2), rotate_spinor(m, x).conj().T)
        for Rvec, Hvec in self.ham.items():
            self.ham[Rvec] = np.dot(smat.conj().T, np.dot(Hvec, smat))
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

    def get_slab_Hk(self, Kvec, nlay, surf_pot=0.0, direction=2):
        # Slab finite along lattice vector `direction` (0/1/2 = a/b/c), periodic
        # in the other two. Stacked block index = layer*nwan, layers ordered along
        # +direction. direction=2 reproduces the original c-axis slab.
        # (replaces the Fortran trans_hams_xlay/ylay/zlay subroutines)
        nw = self.nwan
        inplane = [i for i in range(3) if i != direction]
        hamk = np.zeros((nw*nlay, nw*nlay), dtype=complex)
        for Rvec, Hvec in self.ham.items():
            ratio = np.exp(2j * np.pi * sum(Rvec[i] * Kvec[i] for i in inplane))
            Hmat = ratio * np.asarray(Hvec, dtype=complex)
            for ilay in range(nlay):
                jlay = ilay + Rvec[direction]
                if (jlay >= 0) and (jlay < nlay):
                    hamk[ilay*nw:(ilay+1)*nw, jlay*nw:(jlay+1)*nw] += Hmat
        for iwan in range(nw):
            hamk[ iwan  , iwan  ] += surf_pot
            hamk[-iwan-1,-iwan-1] += surf_pot
        return hamk

    def get_wire_Hk(self, Kvec, nl1, nl2, axis=0):
        # 1D wire: periodic only along lattice vector `axis`, finite along the
        # other two (the lower index -> nl1, the higher -> nl2; ascending order).
        # cell index = i1*nl2 + i2, block = cell*nwan. axis=0 == Fortran trans_qshe.
        nw = self.nwan
        d1, d2 = [i for i in range(3) if i != axis]
        ncell = nl1 * nl2
        hamk = np.zeros((nw*ncell, nw*ncell), dtype=complex)
        for Rvec, Hvec in self.ham.items():
            ratio = np.exp(2j * np.pi * Rvec[axis] * Kvec[axis])
            Hmat = ratio * np.asarray(Hvec, dtype=complex)
            s1, s2 = Rvec[d1], Rvec[d2]
            for i1 in range(nl1):
                j1 = i1 + s1
                if (j1 < 0) or (j1 >= nl1):
                    continue
                for i2 in range(nl2):
                    j2 = i2 + s2
                    if (j2 < 0) or (j2 >= nl2):
                        continue
                    ii = (i1*nl2 + i2) * nw
                    jj = (j1*nl2 + j2) * nw
                    hamk[ii:ii+nw, jj:jj+nw] += Hmat
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

    def get_bulk_projected_kz(self, kxy, omin=-10, omax=10, nomg=100, nkz=100, sigma=0.1, mu=0.0):
        # kz 方向积分的谱函数: A(kx,ky,w) = (1/nkz) * sum_{kz,n} Gauss(E_n(kx,ky,kz) - mu - w)
        # 按 kz 并行: 每个 rank 处理一部分 kz, 内层对所有 kxy 批量对角化, omega 维度一次性向量化。
        # 注意: 这里必须用串行/批量本征值, 不能调用集合操作 get_bulk_eigvals,
        #       否则两层 MPI 嵌套会导致 Allgatherv 数据错位 / 进程数除不尽时死锁。
        kxy = np.asarray(kxy, dtype=float)
        nk  = len(kxy)
        kzlist = np.linspace(0, 1, num=nkz, endpoint=False)
        omgs   = np.linspace(omin, omax, num=nomg)
        norm_factor = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

        rstack, hstack = self._stack_ham()            # (nrpt,3), (nrpt,nwan,nwan)
        akws = np.zeros((nomg, nk), dtype=float)

        my_kz = kzlist[mpi.rank::mpi.size]
        for ikz, kz in enumerate(my_kz):

            if mpi.rank0:
                print("calculating kz={:.3f}  ({:6.1f}%)".format(kz, 100.0 * (ikz + 1) / len(my_kz)))
                sys.stdout.flush()

            kvec  = np.column_stack([kxy[:, 0], kxy[:, 1], np.full(nk, kz)])   # (nk,3)
            phase = np.exp(2j * np.pi * (rstack @ kvec.T))                     # (nrpt,nk)
            hamk  = np.einsum('rk,rij->kij', phase, hstack)                    # (nk,nwan,nwan)
            eigs  = np.linalg.eigvalsh(hamk)                                   # (nk,nwan), 升序

            # omega 与 kxy 一次性向量化; 对能带(数量小)累加以把临时数组控制在 (nomg,nk)
            for n in range(eigs.shape[1]):
                arg  = (eigs[:, n][None, :] - (mu + omgs[:, None])) / sigma    # (nomg,nk)
                akws += norm_factor * np.exp(-0.5 * arg * arg)

        akws = mpi.Allreduce(akws) / float(nkz)
        return omgs, akws


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

    def get_bulk_eprojs_multi(self, klist, plists):
        # 一次对角化同时计算多组投影权重 sum_{i in plist} |<i|nk>|^2
        # 返回 eigs (nwan, nk), projs (nproj, nwan, nk)
        kmpi,kcount,_ = mpi.devide_array(klist, root=0)
        nproj = len(plists)
        empi = np.zeros((len(kmpi), self.nwan), dtype=float)
        pmpi = np.zeros((len(kmpi), nproj, self.nwan), dtype=float)
        for ikp, kvec in enumerate(kmpi):
            eig, vec = self.get_bulk_evec(kvec)
            empi[ikp] = eig
            weight = np.abs(vec)**2
            for ipr, plist in enumerate(plists):
                pmpi[ikp, ipr] = np.sum(weight[plist, :], axis=0)

        eigs  = np.zeros((len(klist), self.nwan), dtype=float)
        projs = np.zeros((len(klist), nproj, self.nwan), dtype=float)
        ecount = np.array(kcount)*self.nwan
        edisp  = np.insert(np.cumsum(ecount),0,0)[0:-1]
        pcount = ecount*nproj
        pdisp  = np.insert(np.cumsum(pcount),0,0)[0:-1]
        mpi.Allgatherv(empi,eigs,ecount,edisp)
        mpi.Allgatherv(pmpi,projs,pcount,pdisp)
        return eigs.T, projs.transpose(1, 2, 0)

    def get_slab_eigvals(self, klist, nlay, surf_pot=0.0, direction=2):
        kmpi,kcount,_ = mpi.devide_array(klist, root=0)
        empi = np.array([la.eigvalsh(self.get_slab_Hk(kvec,nlay,surf_pot=surf_pot,direction=direction)) for kvec in kmpi])
        eigs = np.zeros((len(klist),self.nwan*nlay))
        ecount = np.array(kcount)*self.nwan*nlay
        edisp  = np.insert(np.cumsum(ecount),0,0)[0:-1]
        mpi.Allgatherv(empi,eigs,ecount,edisp)
        return eigs.T

    def get_wire_eigvals(self, klist, nl1, nl2, axis=0):
        kmpi,kcount,_ = mpi.devide_array(klist, root=0)
        ncell = nl1*nl2
        empi = np.array([la.eigvalsh(self.get_wire_Hk(kvec,nl1,nl2,axis=axis)) for kvec in kmpi])
        eigs = np.zeros((len(klist),self.nwan*ncell))
        ecount = np.array(kcount)*self.nwan*ncell
        edisp  = np.insert(np.cumsum(ecount),0,0)[0:-1]
        mpi.Allgatherv(empi,eigs,ecount,edisp)
        return eigs.T

    def get_slab_eprojs(self, klist, nlay, surf_pot=0.0, direction=2, drop_idx=None):
        # drop_idx: 需要从 slab 哈密顿量中删除的行/列索引 (0-based, 取值 0..nlay*nwan-1),
        #           可传 list/array, 默认 None 表示不删. 例: drop_idx=range(30,38)
        #           删除最底层原子的 8 个轨道. 删除时会取保留轨道的子矩阵,
        #           保留所有 off-diagonal 耦合 (而不是只拷对角块).
        kmpi,kcount,_ = mpi.devide_array(klist, root=0)
        nbase = self.nwan*nlay
        if drop_idx is None:
            keep = np.arange(nbase)
        else:
            keep = np.setdiff1d(np.arange(nbase), np.asarray(drop_idx, dtype=int))
        nred = len(keep)                                          # 删除后剩余轨道数
        # 投影到首/末层时, 在 reduced basis 中选出仍属于该层的行。
        # 保持旧接口语义: ptop 对应首层，pbot 对应末层。
        top_rows = np.where(keep < self.nwan)[0]
        bot_rows = np.where(keep >= (nlay-1)*self.nwan)[0]

        empi = []; tmpi = []; bmpi = []
        for ikvec, kvec in enumerate(kmpi):

            if mpi.rank0:
                print("Progress: ({:6.1f}%)".format(100.0 * (ikvec + 1) / len(kmpi)))
                sys.stdout.flush()

            ham_full = self.get_slab_Hk(kvec,nlay,surf_pot=surf_pot,direction=direction)
            hamk = ham_full[np.ix_(keep, keep)]                   # 子矩阵, 保留全部耦合
            eig, vec = la.eigh(hamk)

            empi.append(eig)
            tmpi.append( np.diag(np.dot(vec[top_rows,:].conj().T,vec[top_rows,:])).real )
            bmpi.append( np.diag(np.dot(vec[bot_rows,:].conj().T,vec[bot_rows,:])).real )

        eigs = np.zeros((len(klist),nred))
        ptop = np.zeros((len(klist),nred))
        pbot = np.zeros((len(klist),nred))
        ecount = np.array(kcount)*nred
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

    def get_slab_dos(self, klist, nlay, surf_pot=0.0, emin=-10.0, emax=10.0, ne=101, mu=0.0, sigma=0.1, direction=2):

        from scipy.stats import norm

        omega = np.linspace(emin, emax, num=ne)
        kmpi,_,_ = mpi.devide_array(klist, root=0)

        tdos_mpi = np.zeros_like(omega)
        for ikp, kpt in enumerate(kmpi):
            if mpi.rank0 : print(ikp, len(kmpi))
            sys.stdout.flush()
            eig = la.eigvalsh(self.get_slab_Hk(kpt,nlay,surf_pot=surf_pot,direction=direction))
            for img, ene in enumerate(omega):
                tdos_mpi[img] += np.sum(norm.pdf(eig,loc=mu+ene,scale=sigma)) / float(len(klist))
        tdos = mpi.allreduce(tdos_mpi)
        return omega, tdos
    
    def get_slab_surfdos(self, klist, nlay, proj, surf_pot=0.0, emin=-10.0, emax=10.0, ne=101, mu=0.0, sigma=0.1, direction=2):

        from scipy.stats import norm

        omega = np.linspace(emin, emax, num=ne)
        kmpi,_,_ = mpi.devide_array(klist, root=0)

        tdos_mpi = np.zeros_like(omega)
        pdos_mpi = np.zeros((len(omega),len(proj)))
        for ikp, kpt in enumerate(kmpi):
            if mpi.rank0 : print(ikp, len(kmpi))
            sys.stdout.flush()
            eig,vec = la.eigh(self.get_slab_Hk(kpt,nlay,surf_pot=surf_pot,direction=direction))
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
