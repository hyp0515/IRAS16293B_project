import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
from radmc.setup import radmc3d_setup
from radmc.simulate import generate_simulation
from radmc.plot import generate_plot
import os 

from radmc3dPy import image
from radmc3dPy.analyze import *


incl_list = [60] # inclination in degrees
a_list = [1, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01] # grain size in mm
Q_list = [0.3] # Toomre Q parameter
lstar_list = [10] # stellar luminosity in solar luminosities
heating_list = ['accretion'] # heating mechanism



# incl_list = [30, 40, 50, 60] # inclination in degrees
# a_list = [1, 0.5, 0.1, 0.05, 0.01] # grain size in mm
# Q_list = [1, 0.5, 0.3, 0.1] # Toomre Q parameter
# lstar_list = [0.1, 0.5, 1, 5, 10] # stellar luminosity in solar luminosities
# heating_list = ['combine', 'accretion', 'radiation'] # heating mechanism


for a in a_list:
    for Q in Q_list:
        for lstar in lstar_list:
            for heat in heating_list:
                amax        = a # maximum grain size in mm
                mstar       = .5 # stellar mass in solar masses
                mdot        = 4.5e-5 # accretion rate in solar masses per year
                rd          = 40 # disk radius in AU
                Toomre_Q    = Q # Toomre Q parameter
                l_star      = lstar # stellar luminosity in solar luminosities
                heating     = heat # heating mechanism


                model = radmc3d_setup(silent=False)
                model.get_mastercontrol(filename=None,
                                        comment=None,
                                        incl_dust=1,
                                        incl_lines=0,
                                        nphot=1000000,
                                        nphot_scat=10000000,
                                        scattering_mode_max=2,
                                        istar_sphere=1,
                                        num_cpu=None,
                                        modified_random_walk = 1
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

                for incl in incl_list:

                    print(f"incl: {incl}, a: {a}, Q: {Q}, lstar: {lstar}, heating: {heating}")
                    simulation = generate_simulation(save_out=True, save_npz=True)
                    simulate_mutual_parms = {
                        "incl"      : incl,
                        "npix"      : 118,
                        "sizeau"    : 100,
                        "posang"    : 100,
                        "dir"       : './syn_obs/',
                        "fname"     : f'{incl}_{a}_{Q}_{lstar}_{heating}',
                    }


                    simulation.generate_continuum(
                        scat=True,
                        wav=3000,
                        **simulate_mutual_parms
                    )



                    distance = 140 # distance in pc

                    sizeau = simulate_mutual_parms['sizeau']
                    npix = simulate_mutual_parms['npix']
                    pixel_area = (sizeau/npix/140)**2

                    beam_axis = [0.0478, 0.0441] # beam axis in arcsec
                    beam_area = beam_axis[0]*beam_axis[1]*np.pi/(4*np.log(2))

                    f_dir  = simulate_mutual_parms['dir']
                    f_name = simulate_mutual_parms['fname']



                    model_img = image.readImage(fname=f'./{f_dir}/outfile/conti_{f_name}_scat.out')
                    conv_image = model_img.imConv(dpc=distance, fwhm=beam_axis, pa=-79.32)
                    conv_image.imageJyppix *= beam_area/pixel_area/(distance**2)

                    plt.imshow(conv_image.imageJyppix[:, :, 0].T, origin='lower', cmap="seismic", vmin=0, vmax=0.008)
                    plt.colorbar()
                    plt.title(f"incl: {incl}, a: {a}, Q: {Q}, lstar: {lstar}, heating: {heating}")
                    plt.savefig(f'./synthetic_img/{f_name}.pdf', transparent=True)
                    plt.close()
                os.system('make cleanall')