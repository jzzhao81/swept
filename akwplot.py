

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

afile = 'dmft.akw.dat'
ofile = 'co3sn2s2_akw_cbar.pdf'

# Load data from files
kpts = np.loadtxt('kpoints.dat', usecols=(4), unpack=True)
omg, akw = np.loadtxt(afile, usecols=(1,3), unpack=True)
nkpt = kpts.shape[0]
akw = akw.reshape(nkpt, -1)
nomg = akw.shape[1]
omg = omg[:nomg]
print('nomg, nkpt :', nomg, nkpt)
print(akw.min(), akw.max())

amin=0.258597177
amax=73.0142107

fig, ax = plt.subplots(figsize=(6,5))

ticklist = [0.0, 0.778863499, 1.494279165]
ticklabel = ["K", r"$\Gamma$", "T"]
ax.set_xlim(kpts.min(), kpts.max())
ax.set_xticks(ticklist)
ax.set_xticklabels(ticklabel)
ax.set_ylim(-1.0, 1.0)
ax.set_yticks(np.linspace(-1.0,1.0, num=5))
ax.set_yticks(np.linspace(-1.0,1.0, num=21), minor=True)

# ax.set_xlabel("Momentum")
# ax.set_ylabel("Energy (eV)")

apcm = ax.pcolormesh(kpts, omg, akw.T, norm=colors.PowerNorm(gamma=1.0,vmin=amin,vmax=amax), cmap='bwr', rasterized=True, zorder=0)

# for eig in eigs_dft:
    # ax.plot( kpt_dft, eig, ls='-', lw=1.0, color='red', zorder=5)

# ax.axhline(0.0, c='black', ls='--', lw=1.0)
# for ik in ax.get_xticks()[1:-1]:
#     ax.plot((ik, ik), (ax.get_ylim()[0],
#                        ax.get_ylim()[1]), '--', c='black', lw=1.0)

cbar = plt.colorbar(apcm, ax=ax, ticks=[amin, amax])
cbar.ax.set_yticklabels(['Low', 'High']) # For vertical colorbar

plt.tight_layout()

plt.savefig(ofile, dpi=150)

