import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import io
import contextlib
from radmc3dPy import image
from radmc3dPy.analyze import * 

sys.path.append('..')
from radmc.setup import radmc3d_setup
from radmc.simulate import generate_simulation

from plot_sed import HG_19_data, this_work_data

def write_log(fname='log.txt'):
    if not os.path.exists(fname):
        open(fname, 'w').close()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        read_data = readData(ddens=True)
        total_mass = read_data.getDustMass()
    with open(fname, 'r+') as f:
        f.write('  amax   Mstar      Mdot  Rd    Q      Md\n')
        f.write('    mm    Msun   Msun/yr  AU         Msun\n')
        f.write('=========================================\n')
        f.seek(0,2)
        f.write(f'{amax:.2e}  {mstar:.2f}  {mdot:.2e}  {rd}  {Toomre_Q:.1f}  {total_mass/ms*101:.4f}\n')

freqs_HG19 = np.array([d["Freq"] for d in HG_19_data])
S_B_HG19   = np.array([d["S_B"] for d in HG_19_data])
sigma_HG19 = np.array([d["sigma"] for d in HG_19_data])
freqs_this_work = np.array([d["Freq"] for d in this_work_data])
S_B_this_work   = np.array([d["S_B"] for d in this_work_data])
sigma_this_work = np.array([d["sigma"] for d in this_work_data])


distance = 140

amax_list = [ 1e1,  1e0, 1e-1, 1e-2] # maximum grain sizes in mm
Mdot_list = [1e-5, 1e-6, 1e-7, 1e-8] # accretion rates
Q_list    = [ 1.5,    1,  0.5,  0.3] # Toomre Q parameters


amax_fiducial = 1e-1
Mdot_fiducial = 1e-6
Q_fiducial    = 0.5


obs_wav = np.array([1300, 3000, 7000, 18000])

for i, amax in enumerate(amax_list):
    for j, mdot in enumerate(Mdot_list):
        for k, Q in enumerate(Q_list):
            amax        = amax # maximum grain size in mm
            mstar       = 0.5 # stellar mass in solar masses
            mdot        = mdot # accretion rate in solar masses per year
            rd          = 50 # disk radius in AU
            Toomre_Q    = Q # Toomre Q parameter
            l_star      = .1 # stellar luminosity in solar luminosities
            heating     = 'accretion' # heating mechanism


            # model = radmc3d_setup(silent=False)
            # model.get_mastercontrol(filename=None,
            #                         comment=None,
            #                         incl_dust=1,
            #                         incl_lines=0,
            #                         nphot=500000,
            #                         nphot_scat=5000000,
            #                         nphot_spec=500000,
            #                         scattering_mode_max=5,
            #                         istar_sphere=1,
            #                         num_cpu=None,
            #                         modified_random_walk = 1,
            #                         alignment_mode=-1, # 1 for grain alignment
            #                         )
            # model.get_continuumlambda(filename=None,
            #                         comment=None,
            #                         lambda_micron=None,
            #                         append=False)
            # model.write_dust_opac(inputstyle=20, grain_align=True)
            # model.get_diskcontrol(  d_to_g_ratio    = 0.01,
            #                         a_max           = amax, # mm
            #                         Mass_of_star    = mstar, # Msun
            #                         Accretion_rate  = mdot, # Msun/yr
            #                         Radius_of_disk  = rd,   # AU
            #                         Q               = Toomre_Q, # Toomre Q
            #                         NR    =200,
            #                         NTheta=200,
            #                         NPhi  =100,
            #                         )
            # model.get_heatcontrol(L_star=l_star, # Lsun
            #                     R_star=1,
            #                     heat=heating) # radiation/accretion

            # model.get_dustalignmentcontrol(alpha=1/(20*au*au), 
            #                             hourglass=True, 
            #                             uniform_z=True, 
            #                             uniform_x=False, 
            #                             uniform_y=False,
            #                             toroidal=False)
            # write_log()
            
            simulation = generate_simulation(save_out=True, save_npz=True)
            simulate_mutual_parms = {
                "incl"      : 50,
                "npix"      : 500,
                "sizeau"    : 200,
                "posang"    : 0,
                "phi"       : 0,
                "dir"       : f'./simulation/amax_{amax}_Mdot_{mdot}_Q_{Toomre_Q}/',
                "fname"     : f'sed',
            }
            simulation.generate_sed(
                scat=True,
                read_lambda=obs_wav*1e-3,
                load_simulation=True,
                **simulate_mutual_parms
            )
            # for lam in obs_wav:
            #     simulate_mutual_parms["fname"] = f'wav_{lam}'
            #     simulation.generate_continuum(
            #         scat=True,
            #         wav=lam,
            #         stokes=True,
            #         load_simulation=False,
            #         **simulate_mutual_parms
            #     )
            sed = simulation.spectrum
            lam = sed[:, 0]
            nu = (1e-2*cc)*1e-9/(1e-6*lam) # GHz
            fnu = sed[:, 1]*1e26/(140**2) # mJy

            
            # plt.figure(figsize=(6, 10))

            fig, ax = plt.subplots(1, 2, figsize=(12, 10), sharey=True)
            ax[0].scatter(freqs_HG19, S_B_HG19, marker='x', 
                        color='blue', s=30, label='Hernández-Gómez et al. 2019')
            ax[0].scatter(freqs_this_work, S_B_this_work, marker='o', 
                        color='olive', s=100, label='This Work')
            ax[0].plot(nu, fnu, 'o-r', label=f'amax = {amax:.0e} mm, Mdot = {mdot:.0e} Msun/yr, Q = {Toomre_Q:.1f}', 
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
            ax[1].plot(nu, fnu, 'o-r', label=f'amax = {amax:.0e} mm, Mdot = {mdot:.0e} Msun/yr, Q = {Toomre_Q:.1f}', 
                          markersize=8)
            ax[1].set_xscale('log'); ax[1].set_xlim((1e+1, 5e2))
            ax[1].set_yscale('log'); ax[1].set_ylim((1e-2, 2e5))
            ax[1].set_xlabel('Frequency (GHz)', fontsize=14)
            ax[1].set_ylabel('Flux Density (mJy)', fontsize=14)
            ax[1].set_title('SED (Zoomed)', fontsize=16)
            ax[1].grid(True, which='major', linestyle='--', linewidth=0.5)
            ax[1].legend()
            plt.tight_layout()
            plt.savefig(f'./simulation/amax_{amax}_Mdot_{mdot}_Q_{Toomre_Q}/sed.pdf', transparent=True)
            plt.close()