
import numpy as np
import matplotlib.pyplot as plt

ene, tdos, *pdos = np.loadtxt('socpdos.dat', unpack=True)

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(ene, tdos, linewidth=1.0, color='black', label='Total DOS', zorder=11)
ax.plot(ene, pdos[0], linewidth=1.0, color='red', label='Ir-5d J1/2', zorder=12)
ax.plot(ene, pdos[1], linewidth=1.0, color='blue', label='Ir-5d J3/2', zorder=13)

ax.fill_between(ene, tdos, facecolor='gray', alpha=0.2, zorder=1)
ax.fill_between(ene, pdos[0], facecolor='salmon', alpha=0.2, zorder=2)
ax.fill_between(ene, pdos[1], facecolor='lightskyblue', alpha=0.2, zorder=3)

ax.axvline(0, linestyle='--', color='grey', zorder=20)

ax.set_xlim(-2, 1)
ax.set_xticks(np.arange(-2, 2, 1))
ax.set_ylim(0, 70)
ax.set_yticks(np.arange(0, 71, 20))

plt.legend(loc='upper left')
plt.savefig('pdos.pdf', dpi=150)
