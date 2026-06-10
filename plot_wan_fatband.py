'''
Author: Jianzhou Zhao
Date: 2021-01-13 10:28:53
LastEditors: your name
LastEditTime: 2025-04-07 17:06:52
Description: file content
'''

import numpy as np
import matplotlib.pyplot as plt

nkpt   = 103
wscale = 150
ef     = 0

kpts, eigs, proj_ird = np.loadtxt('socwan_j32_2.dat',  unpack=True)
# proj_op = np.loadtxt('cuo_o_2p_fatbands', usecols=(2), unpack=True)
# proj_cus = np.loadtxt('cuo_cu_4s_fatbands', usecols=(2), unpack=True)

kpts = kpts[:nkpt]
eigs = eigs.reshape(-1,nkpt) - ef
proj_ird = proj_ird.reshape(-1,nkpt) * wscale
# proj_cus = proj_cus.reshape(-1,nkpt) * wscale
# proj_op = proj_op.reshape(-1,nkpt) * wscale

fig, ax = plt.subplots(figsize=(5,6))

for ibnd, band in enumerate(eigs):
    ax.scatter( kpts, band, s=proj_ird[ibnd], color='salmon',     marker='o', linewidths=0.1, zorder=2)
    # ax.scatter( kpts, band, s=proj_op[ibnd], color='royalblue',  marker='o', linewidths=0.1, zorder=1)
    # ax.scatter( kpts, band, s=proj_cus[ibnd], color='springgreen',  marker='o', linewidths=0.1, zorder=1)
    ax.plot(kpts, band, ls='-', color=('gray'), lw=1, zorder=2)

ax.set_ylim(-2, 1)
ax.set_yticks(np.linspace(ax.get_ylim()[0],ax.get_ylim()[1], num=4))
ax.set_yticks(np.linspace(ax.get_ylim()[0],ax.get_ylim()[1], num=7), minor=True)
ax.set_xlim(kpts.min(), kpts.max())
ax.set_xticks([0.0000, 0.33087855894233553, 0.6617571178846711, 1.1296900634393876, 1.7027885386626862])
ax.set_xticklabels([r"$\Gamma$", r"X", r"M", r"$\Gamma$", r"R"])
# ax.set_ylabel('Energy (eV)')

ax.axhline(0.0, ls='--', lw=1, color='gray')
for itk in ax.get_xticks()[1:-1]:
    ax.axvline( itk, ls='--', color='gray', lw=1 )

plt.tight_layout()
plt.savefig('socwan_j32_2.pdf', dpi=150)
plt.close()

