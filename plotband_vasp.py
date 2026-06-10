'''
@Author: Jianzhou Zhao
@Data: Do not edit
@Github: https://github.com/jzzhao81
LastEditors: your name
LastEditTime: 2025-05-26 10:43:45
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

file_w2k_up = 'co3sn2s2_bandsup.gnu'
file_w2k_dn = 'co3sn2s2_bandsdn.gnu'
file_wan    = 'ggawan_eshift.dat'
nkpt = 101;   nbnd = 56
nkwn = 202;   nwan = 58
mu = 12.5647

yrange = [-2.00, 2.00]
ymajorLocator = MultipleLocator(1.0)
yminorLocator = MultipleLocator(0.5)
ymajorFormatter = FormatStrFormatter('%4.1f')

xticks = [0.0000, 0.81649658, 1.31649658]
xlabel = [r'K', r'$\Gamma$', r'T']

# Read data
# kpth, eigs_up = np.loadtxt(file_w2k_up, unpack=True)
# kpth, eigs_dn = np.loadtxt(file_w2k_dn, unpack=True)
# eigs_up = eigs_up.reshape(nbnd, nkpt) - mu
# eigs_dn = eigs_dn.reshape(nbnd, nkpt) - mu
# kpth = kpth[:nkpt]

kwan, ewan = np.loadtxt(file_wan, usecols=(0,1), unpack=True)
kwan = kwan[:nkwn]
ewan = ewan.reshape(nwan, nkwn)

# frac = kpth[-1]/kwan[-1]

# Initialize figs
figs, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(kwan.min(), kwan.max())
ax.set_ylim(yrange[0], yrange[1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabel)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.yaxis.set_major_formatter(ymajorFormatter)
ax.set_ylabel("Energy (eV)")

# Plot Ef
ax.plot((kwan.min(), kwan.max()), (0.00, 0.00), color='grey', linestyle='--', lw=1.0)

# Plot high symmetry lines
for ik in ax.get_xticks()[1:-1]:
    ax.plot((ik, ik), (ax.get_ylim()[0],
                       ax.get_ylim()[1]), '--', c='grey', lw=0.8)

# Plot bands
# for band in eigs_up:
#     ax.plot(kpth, band, 'r-', lw=1.2, zorder=1)
# for band in eigs_dn:
#     ax.plot(kpth, band, 'r-', lw=1.2, zorder=1)

for band in ewan:
    ax.plot(kwan, band, ls='-', color='royalblue', lw=1.0, zorder=5)

plt.tight_layout()

# Save figs
plt.savefig('ggawan_eshift.pdf')

