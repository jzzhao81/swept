#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jianzhou Zhao
@Date: 2018-11-21 15:58:18
@LastEditors: Jianzhou Zhao
@LastEditTime: 2019-07-12 20:33:46
@Description: File content
'''
# @Author: Jianzhou Zhao <jzzhao>
# @Date:   2018-11-21T15:58:18+01:00
# @Email:  jianzhou.zhao@gmail.com
# @Last modified by:   jzzhao
# @Last modified time: 2018-11-22T10:07:05+01:00


import numpy as np


class Kpoint():

    def __init__(self, nktot=None, kfrac=None, kreal=None):
        self.nktot = nktot
        self.kfrac = kfrac
        self.kreal = kreal
        return

    @staticmethod
    def k_length(K_list):
        """
        Return k point distance from previous one
        Exclude the first point
        """
        return np.array([np.linalg.norm(K1 - K0) for K0, K1 in zip(K_list, K_list[1:])])

    @classmethod
    def k_length_accumulate(cls, K_list):
        """
        Return k point distance from first one
        Include the first pointself.

        This code should be written beautifully!

        """
        dist = [0.0]
        dist.extend(cls.k_length(K_list))
        return [sum(dist[:ii])+dist[ii] for ii in range(len(dist))]

    @classmethod
    def from_high_symmetry_points(cls, khsym_frac, bzvec=None, kwant=None):
        """
        Return a K list along high symmetry lines given.
        """
        if bzvec is None:
            bzvec = np.eye(3)

        khsym_real = np.array([np.dot(kvec, bzvec) for kvec in khsym_frac])
        klen = np.sum(cls.k_length(khsym_real))

        if kwant is None or kwant < len(khsym_frac):
            kwant = len(khsym_frac) - 1
        kstep = klen / kwant

        kline_real = []
        for k0, k1 in zip(khsym_real, khsym_real[1:]):
            nk = int(np.ceil(np.linalg.norm(k1 - k0) / kstep))
            kfrac = (np.array(k1) - np.array(k0)) / nk
            kline_real.extend(list(k0 + ik * kfrac) for ik in range(nk))
        kline_real.append(list(khsym_real[-1]))

        kline_frac = []
        for kpt in kline_real:
            kline_frac.append(list(np.dot(kpt, np.linalg.inv(bzvec))))

        return cls(nktot=len(kline_frac), kfrac=np.array(kline_frac), kreal=np.array(kline_real))

    @classmethod
    def gen_uniform_kpoints(cls, nk, bzvec=None):
        """
        Return a uniform K point grid.
        """
        if bzvec is None:
            bzvec = np.eye(3)
        nktot = np.prod(nk)
        kfrac = [[ikx/nk[0], iky/nk[1], ikz/nk[2]] for ikz in range(nk[2]) for iky in range(nk[1]) for ikx in range(nk[0])]
        kreal = [np.dot(kpt, bzvec) for kpt in kfrac]
        return cls(nktot=nktot, kfrac=np.array(kfrac), kreal=np.array(kreal))


if __name__ == '__main__':

    # K_list = [[0, 0, 0], [1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0.5]]
    # kpts = Kpoint().from_high_symmetry_points(khsym_frac=K_list)
    # kpth = kpts.k_length_accumulate(kpts.kreal)
    # print(kpts.kfrac)
    # print(kpts.kreal)
    # print(kpth)

    umat = np.array([
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ])

    kpts = Kpoint.gen_uniform_kpoints(nk=(4, 3, 2), bzvec=umat)
    print(kpts.kfrac)
    print(kpts.kreal)
