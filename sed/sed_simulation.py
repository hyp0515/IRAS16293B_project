import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from radmc3dPy import image
from radmc3dPy.analyze import * 

sys.path.append('..')
from radmc.setup import radmc3d_setup
from radmc.simulate import generate_simulation
from radmc.plot import generate_plot

from plot_sed import HG_19_data, this_work_data

amax        = .3 # maximum grain size in mm
mstar       = 0.5 # stellar mass in solar masses
mdot        = 5e-5 # accretion rate in solar masses per year
rd          = 50 # disk radius in AU
Toomre_Q    = 0.3 # Toomre Q parameter
l_star      = .1 # stellar luminosity in solar luminosities
heating     = 'accretion' # heating mechanism
incl        = 50

model = radmc3d_setup(silent=False)
model.get_mastercontrol(filename=None,
                        comment=None,
                        incl_dust=1,
                        incl_lines=0,
                        nphot=500000,
                        nphot_scat=5000000,
                        nphot_spec=100000,
                        scattering_mode_max=5,
                        istar_sphere=1,
                        num_cpu=None,
                        modified_random_walk = 1,
                        alignment_mode=-1, # 1 for grain alignment
                        )
model.get_continuumlambda(filename=None,
                        comment=None,
                        lambda_micron=None,
                        append=False)
model.get_diskcontrol(  d_to_g_ratio    = 0.01,
                        a_max           = amax, # mm
                        Mass_of_star    = mstar, # Msun
                        Accretion_rate  = mdot, # Msun/yr
                        Radius_of_disk  = rd,   # AU
                        Q               = Toomre_Q, # Toomre Q
                        NR    =200,
                        NTheta=200,
                        NPhi  =20,
                        )
model.get_heatcontrol(L_star=l_star, # Lsun
                      R_star=1,
                      heat=heating) # radiation/accretion

model.write_dust_opac(inputstyle=20, grain_align=True)
model.get_dustalignmentcontrol(alpha=1e-33, hourglass=True, uniform_z=True)


obs_lambda = np.array([
    1.3, 
    3,
    7,
    18
])

with open('camera_wavelength_micron.inp', 'w+') as f:
    f.write('%d\n'%(len(obs_lambda)))
    for value in obs_lambda:
        f.write('%13.6e\n'%(value*1e3))  # Convert to microns

os.system(f'radmc3d spectrum incl {incl} loadlambda noline')
s = readSpectrum('spectrum.out')
lam = s[:, 0]
nu = (1e-2*cc)*1e-9/(1e-6*lam) # GHz
fnu = s[:, 1]*1e26/(140**2) # mJy

freqs_this_work = np.array([d["Freq"] for d in this_work_data])
S_B_this_work   = np.array([d["S_B"] for d in this_work_data])
sigma_this_work = np.array([d["sigma"] for d in this_work_data])


plt.scatter(nu, fnu, s=5, c='black', label='RADMC3D')
plt.errorbar(freqs_this_work, S_B_this_work, yerr=sigma_this_work, fmt='o', 
             color='r', ecolor='r', elinewidth=2, capsize=4, label='This Work')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Flux Density (mJy)')
plt.show()
