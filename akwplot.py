

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

afile = 'akws.dat'
ofile = 'ggawan_kz.pdf'

# Load data from files
rawdata = np.loadtxt(afile, unpack=True)
if rawdata.ndim != 2 or rawdata.shape[0] != 3:
    raise ValueError(f"Expected three columns in {afile}")

omega_changes = np.flatnonzero(~np.isclose(rawdata[0], rawdata[0, 0]))
nkpt = int(omega_changes[0]) if len(omega_changes) else rawdata.shape[1]
if rawdata.shape[1] % nkpt:
    raise ValueError(f"Inconsistent omega/k-point grid in {afile}")
nomg = rawdata.shape[1] // nkpt

omega_grid = rawdata[0].reshape(nomg, nkpt)
kpoint_grid = rawdata[1].reshape(nomg, nkpt)
if not np.allclose(omega_grid, omega_grid[:, :1]):
    raise ValueError(f"Omega values are not grouped consistently in {afile}")
if not np.allclose(kpoint_grid, kpoint_grid[:1, :]):
    raise ValueError(f"K-point paths differ between omega slices in {afile}")

omgs = omega_grid[:, 0]
kpts = kpoint_grid[0]
akws = rawdata[2].reshape(nomg, nkpt)

# akw = akw.reshape(nkpt, -1)
# nomg = akw.shape[1]
# omg = omg[:nomg]
print('nomg, nkpt :', nomg, nkpt)
print(akws.min(), akws.max())

amin=akws.min()
amax=akws.max()

fig, ax = plt.subplots(figsize=(6,5))

# ticklist = [0.0, 0.778863499, 1.494279165]
# ticklabel = ["K", r"$\Gamma$", "T"]
# ax.set_xlim(kpts.min(), kpts.max())
# ax.set_xticks(ticklist)
# ax.set_xticklabels(ticklabel)
# ax.set_ylim(-1.0, 1.0)
# ax.set_yticks(np.linspace(-1.0,1.0, num=5))
# ax.set_yticks(np.linspace(-1.0,1.0, num=21), minor=True)

# ax.set_xlabel("Momentum")
# ax.set_ylabel("Energy (eV)")

apcm = ax.pcolormesh(kpts, omgs, akws, norm=colors.PowerNorm(gamma=1.0,vmin=amin,vmax=amax), cmap='Blues', rasterized=True, zorder=0)

# for eig in eigs_dft:
    # ax.plot( kpt_dft, eig, ls='-', lw=1.0, color='red', zorder=5)

# ax.axhline(0.0, c='black', ls='--', lw=1.0)
# for ik in ax.get_xticks()[1:-1]:
#     ax.plot((ik, ik), (ax.get_ylim()[0],
#                        ax.get_ylim()[1]), '--', c='black', lw=1.0)

# cbar = plt.colorbar(apcm, ax=ax, ticks=[amin, amax])
# cbar.ax.set_yticklabels(['Low', 'High']) # For vertical colorbar

plt.tight_layout()

plt.savefig(ofile, dpi=150)
