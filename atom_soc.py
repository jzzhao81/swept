'''
Author: Jianzhou Zhao
Date: 2018-10-26 17:37:29
LastEditors: your name
LastEditTime: 2024-01-25 17:33:34
Description: file content
'''
# -*- coding: utf-8 -*-
import numpy as np
# import basis

def genladder(ll, ns):

    # print
    # print " Generate ladder operator !"
    # print " For f orbitals, L=3"
    # print " The basis :"
    # print " |-3,±>, |-2,±>, |-1,±>, |0,±>"
    # print " | 1,±>, | 2,±>, | 3,±>"
    # print

    nm = 2 * ll + 1

    Lpls = np.zeros((nm * ns, nm * ns), dtype=float)
    Lmin = np.zeros((nm * ns, nm * ns), dtype=float)

    for mm in range(nm - 1):
        mj = -1.0 * ll + float(mm)

        mi = mj + 1.0
        for ss in range(ns):
            Lpls[ns * (mm + 1) + ss, ns * mm +
                 ss] = np.sqrt((ll + mi) * (ll - mj))

    for mm in range(1, nm):
        mj = -1.0 * ll + float(mm)
        mi = mj - 1.0
        for ss in range(ns):
            Lmin[ns * (mm - 1) + ss, ns * mm +
                 ss] = np.sqrt((ll - mi) * (ll + mj))

    return Lpls, Lmin


def genLxyz(ll, ns, Lpls, Lmin):

    nm = 2 * ll + 1

    LL = np.zeros((3, nm * ns, nm * ns), dtype=complex)

    LL[0, :, :] = (Lpls + Lmin) / (2.0 + 1j * 0.0)
    LL[1, :, :] = (Lpls - Lmin) / (1j * 2.0)

    for mm in range(nm):
        mi = -1.0 * ll + float(mm)
        for ss in range(ns):
            LL[2, ns * mm + ss, ns * mm + ss] = mi

    return LL


def genSOC(LL, Lpls, Lmin, lsoc):

    norb = LL.shape[1] * 2

    soc = np.zeros((norb, norb), dtype=complex)

    soc[0:norb:2, 0:norb:2] = +LL[2, :, :] * lsoc / 2.0
    soc[1:norb:2, 1:norb:2] = -LL[2, :, :] * lsoc / 2.0
    soc[0:norb:2, 1:norb:2] = Lmin * lsoc / 2.0
    soc[1:norb:2, 0:norb:2] = Lpls * lsoc / 2.0

    return soc


def getsoc(lmom, lsoc):

    # gen Ladder operator
    Lpls, Lmin = genladder(lmom, 1)

    # gen Lx, Ly, Lz
    LL = genLxyz(lmom, 1, Lpls, Lmin)

    # gen SOC from LL & Lpls & Lmin
    soc = genSOC(LL, Lpls, Lmin, lsoc)

    return soc

if __name__ == "__main__":

    lsoc = 1
    lmom = 1
    # ns = 2

    soc = getsoc(lmom, lsoc)

    tmat = basis.clebsch_gordan(lmom)

    print(soc)
    print()
    print(np.dot(np.dot(tmat,soc),tmat.T.conj()))
    print()

