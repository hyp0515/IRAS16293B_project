import numpy as np
import matplotlib.pyplot as plt
import sys
from radmc3dPy import image
from radmc3dPy.analyze import * 

sys.path.append('../..')
from radmc.setup import *
from radmc.simulate import Simulation
from sed.plot_sed import HG_19_data, this_work_data

"""
Initialize observation (sed)
"""

freqs_HG19 = np.array([d["Freq"] for d in HG_19_data])
S_B_HG19   = np.array([d["S_B"] for d in HG_19_data])
sigma_HG19 = np.array([d["sigma"] for d in HG_19_data])
freqs_this_work = np.array([d["Freq"] for d in this_work_data])
S_B_this_work   = np.array([d["S_B"] for d in this_work_data])
sigma_this_work = np.array([d["sigma"] for d in this_work_data])

def plot_sed_diff_params(simulation, dir, fname):
    sed = simulation.spectrum
    lam = sed[:, 0]
    nu = (1e-2*cc)*1e-9/(1e-6*lam) # GHz
    fnu = sed[:, 1]*1e26/(140**2) # mJy

    fig, ax = plt.subplots(1, 2, figsize=(12, 10), sharey=True)

    ax[0].scatter(freqs_HG19, S_B_HG19, marker='x', 
                color='blue', s=30, label='Hernández-Gómez et al. 2019')
    ax[0].scatter(freqs_this_work, S_B_this_work, marker='o', 
                color='olive', s=100, label='This Work')
    ax[0].plot(nu, fnu, 'o-r', label=f'model', 
                    markersize=8)
    ax[0].set_xscale('log'); ax[0].set_xlim((5e-1, 1e3))
    ax[0].set_yscale('log'); ax[0].set_ylim((1e-2, 2e5))
    ax[0].set_xlabel('Frequency (GHz)', fontsize=14)
    ax[0].set_ylabel('Flux Density (mJy)', fontsize=14)
    ax[0].set_title('SED', fontsize=16)
    ax[0].grid(True, which='major', linestyle='--', linewidth=0.5)
    ax[0].legend()

    ax[1].scatter(freqs_HG19, S_B_HG19, marker='x', 
                color='blue', s=30, label='Hernández-Gómez et al. 2019')
    ax[1].scatter(freqs_this_work, S_B_this_work, marker='o', 
                color='olive', s=100, label='This Work')
    ax[1].plot(nu, fnu, 'o-r', label=f'model', 
                    markersize=8)
    ax[1].set_xscale('log'); ax[1].set_xlim((1e+1, 5e2))
    ax[1].set_yscale('log'); ax[1].set_ylim((1e-2, 2e5))
    ax[1].set_xlabel('Frequency (GHz)', fontsize=14)
    ax[1].set_ylabel('Flux Density (mJy)', fontsize=14)
    ax[1].set_title('SED (Zoomed)', fontsize=16)
    ax[1].grid(True, which='major', linestyle='--', linewidth=0.5)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(f'{dir}{fname}.pdf', transparent=True)
    plt.close()

# Fiducial model
simulation = Simulation(save_out=True, save_npz=False)
simulation.generate_sed(
    scat=True,
    load_simulation=True,
    fname='sed',
    dir=f'/home/hyp0515/simulation/ext_fig_2/fiducial/'
)
sed = simulation.spectrum
nu_fiducial = (1e-2*cc)*1e-9/(1e-6*sed[:, 0]) # GHz
fnu_fiducial = sed[:, 1]*1e26/(140**2) # mJy


# colors = ["#FF0000", "#FF7F00", "#FFFF00", 
#           "#00FF00", "#0000FF", "#4B0082", "#8B00FF"]

cmap = plt.get_cmap("plasma")
values = np.linspace(0, 1, 7)
colors = [cmap(v) for v in values]

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(18, 8))

for ax in axes:
    ax.scatter(freqs_HG19, S_B_HG19, marker='o', color='blue', s=50, label='HG19')
    ax.scatter(freqs_this_work, S_B_this_work, marker='o', color='k', s=50, label='Observation')
    ax.plot(nu_fiducial, fnu_fiducial, color='k', label=f'Fiducial Model', lw=3)
    ax.set_xscale('log'); ax.set_xlim((5e+0, 3e2))
    ax.set_yscale('log'); ax.set_ylim((1e-1, 1e4))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Frequency (GHz)', fontsize=16)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
axes[0].set_ylabel('Flux Density (mJy)', fontsize=16)


# Left panel
ax = axes[0]
incl = 45
ain = 1.2
amid = 0.8
aout = 0.08
mdot= 4.5e-5
rd = 35
lambda_to_simulate = np.array([8.7e2, 1.3e3, 2e3, 3e3, 6.8e3, 9e3, 1.3e4, 2e4, 3e4, 3.75e4, 5e4, 7.5e4])*1e-3
for i, Q in enumerate([0.3, 0.5, 0.8, 1.0, 1.5]):
    simulation = Simulation(save_out=True, save_npz=False)
    simulation.generate_sed(
        scat=True,
        load_simulation=True,
        fname='sed',
        dir=f'/home/hyp0515/simulation/ext_fig_2/ain_{ain}_amid_{amid}_aout_{aout}_Mdot_{mdot}_Q_{Q}_incl_{incl}_rd_{rd}/'
    )
    sed = simulation.spectrum
    nu = (1e-2*cc)*1e-9/(1e-6*sed[:, 0]) # GHz
    fnu = sed[:, 1]*1e26/(140**2) # mJy
    ax.plot(nu, fnu, label=f'Q={Q}', lw=3, alpha=0.75, color=colors[i])
ax.legend(fontsize=14)

# Middle panel
ax = axes[1]
incl = 45
ain = 1.2
amid = 0.8
aout = 0.08
Q = 0.4
rd = 35
for i, mdot in enumerate([5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]):
    simulation = Simulation(save_out=True, save_npz=False)
    simulation.generate_sed(
        scat=True,
        load_simulation=True,
        fname='sed',
        dir=f'/home/hyp0515/simulation/ext_fig_2/ain_{ain}_amid_{amid}_aout_{aout}_Mdot_{mdot}_Q_{Q}_incl_{incl}_rd_{rd}/'
    )
    sed = simulation.spectrum
    nu = (1e-2*cc)*1e-9/(1e-6*sed[:, 0]) # GHz
    fnu = sed[:, 1]*1e26/(140**2) # mJy
    ax.plot(nu, fnu, label=r'$\dot{M}=$'+f'{mdot:.0E}'+r'$M_{\odot} yr^{-1}$', lw=3, alpha=0.75, color=colors[i])
ax.legend(fontsize=14)

# Right panel
ax = axes[2]
incl = 45
mdot= 4.5e-5
Q = 0.4
rd = 35
for i, a in enumerate([10, 1, 0.5, 0.3, 0.1, 0.01]):
    ain = a
    amid = a
    aout = a
    simulation = Simulation(save_out=True, save_npz=False)
    simulation.generate_sed(
        scat=True,
        load_simulation=True,
        fname='sed',
        dir=f'/home/hyp0515/simulation/ext_fig_2/ain_{ain}_amid_{amid}_aout_{aout}_Mdot_{mdot}_Q_{Q}_incl_{incl}_rd_{rd}/'
    )
    sed = simulation.spectrum
    nu = (1e-2*cc)*1e-9/(1e-6*sed[:, 0]) # GHz
    fnu = sed[:, 1]*1e26/(140**2) # mJy
    ax.plot(nu, fnu, label=r'$a_{max}=$'+f'{a} mm', lw=3, alpha=0.75, color=colors[i])
ax.legend(fontsize=14)

plt.tight_layout()
plt.savefig('./ext_fig_2.pdf', transparent=True)
plt.show()