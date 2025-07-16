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


distance = 140

amax_list = [ 1e1, 1e-1, 1e-2, 1e-3] # maximum grain sizes in mm
Mdot_list = [1e-5, 1e-6, 1e-7, 1e-8] # accretion rates
Q_list    = [ 1.5,    1,  0.5,  0.3] # Toomre Q parameters


amax_fiducial = 1e-1
Mdot_fiducial = 1e-6
Q_fiducial    = 0.5


obs_wav = np.array([1300, 3000, 7000, 18000])
fig, ax = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)

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


            model = radmc3d_setup(silent=False)
            model.get_mastercontrol(filename=None,
                                    comment=None,
                                    incl_dust=1,
                                    incl_lines=0,
                                    nphot=500000,
                                    nphot_scat=5000000,
                                    nphot_spec=500000,
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
            model.write_dust_opac(inputstyle=20, grain_align=True)
            model.get_diskcontrol(  d_to_g_ratio    = 0.01,
                                    a_max           = amax, # mm
                                    Mass_of_star    = mstar, # Msun
                                    Accretion_rate  = mdot, # Msun/yr
                                    Radius_of_disk  = rd,   # AU
                                    Q               = Toomre_Q, # Toomre Q
                                    NR    =200,
                                    NTheta=200,
                                    NPhi  =100,
                                    )
            model.get_heatcontrol(L_star=l_star, # Lsun
                                R_star=1,
                                heat=heating) # radiation/accretion

            model.get_dustalignmentcontrol(alpha=1/(20*au*au), 
                                        hourglass=True, 
                                        uniform_z=True, 
                                        uniform_x=False, 
                                        uniform_y=False,
                                        toroidal=False)
            write_log()

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
                load_simulation=False,
                **simulate_mutual_parms
            )
            for lam in obs_wav:
                simulate_mutual_parms["fname"] = f'wav_{lam}'
                simulation.generate_continuum(
                    scat=True,
                    wav=lam,
                    stokes=True,
                    load_simulation=False,
                    **simulate_mutual_parms
                )
                

# sizeau = simulate_mutual_parms['sizeau']
# npix = simulate_mutual_parms['npix']
# pixel_area = (sizeau/npix/140)**2
# beam_axis = [0.277, 0.231] # beam axis in arcsec
# beam_area = beam_axis[0]*beam_axis[1]*np.pi/(4*np.log(2))


# f_dir  = simulate_mutual_parms['dir']
# f_name = simulate_mutual_parms['fname']

# model_img = image.readImage(fname=f'./{f_dir}/outfile/conti_{f_name}_scat_stokes.out')
# conv_image = model_img.imConv(dpc=distance, fwhm=beam_axis, pa=-65.60)
# conv_image.imageJyppix *= beam_area/pixel_area/(distance**2)

# mask = conv_image.imageJyppix[:, :, 0, 0] > 1e-8

# for j in range(4):
#     conv_image.imageJyppix[:, :, j, 0][~mask] = np.nan

# polang  = 0.5 * np.arctan2(conv_image.imageJyppix[:, :, 2, 0], conv_image.imageJyppix[:, :, 1, 0]) + np.pi/2
# polfrac = np.sqrt(conv_image.imageJyppix[:, :, 1, 0]**2 + conv_image.imageJyppix[:, :, 2, 0]**2) / conv_image.imageJyppix[:, :, 0, 0]

# y, x = np.mgrid[0:model_img.imageJyppix.shape[0], 0:model_img.imageJyppix.shape[1]]
# step = 25
# x_ds, y_ds = x[::step, ::step], y[::step, ::step]
# polfrac_ds = polfrac[::step, ::step]
# polang_ds = polang[::step, ::step]
# u = polfrac_ds * np.cos(polang_ds) / 0.01
# v = polfrac_ds * np.sin(polang_ds) / 0.01

# ax[i].imshow(conv_image.imageJyppix[:, :, 0, 0].T, origin='lower', cmap="magma")
# ax[i].quiver(x_ds, y_ds, u, v, color='white', scale=50, headlength=0, headaxislength=0, headwidth=0)
# ax[i].set_title(f'$a_{{max}} = {amax:.0e}$ mm')

# amax        = 0.1 # maximum grain size in mm
# mstar       = 0.3 # stellar mass in solar masses
# mdot        = 1e-6 # accretion rate in solar masses per year
# rd          = 50 # disk radius in AU
# Toomre_Q    = 0.5 # Toomre Q parameter
# l_star      = .1 # stellar luminosity in solar luminosities
# heating     = 'accretion' # heating mechanism


# model = radmc3d_setup(silent=False)
# model.get_mastercontrol(filename=None,
#                         comment=None,
#                         incl_dust=1,
#                         incl_lines=0,
#                         nphot=500000,
#                         nphot_scat=10000000,
#                         nphot_spec=100000,
#                         scattering_mode_max=5,
#                         istar_sphere=1,
#                         num_cpu=None,
#                         modified_random_walk = 1,
#                         alignment_mode=-1,
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

# simulation = generate_simulation(save_out=True, save_npz=True)

# simulate_mutual_parms = {
#     "incl"      : 50,
#     "npix"      : 500,
#     "sizeau"    : 200,
#     "posang"    : 0,
#     "phi"       : 0,
#     "dir"       : f'./test/{1300}/',
#     "fname"     : f'amax_{amax}_Mdot_{mdot}_Q_{Toomre_Q}',
# }

# simulation.generate_continuum(
#     scat=True,
#     wav=1300,
#     stokes=True,
#     load_simulation=False,
#     **simulate_mutual_parms
# )

# plt.imshow(simulation.conti.imageJyppix[:, :, 0, 0].T, origin='lower', cmap="magma")
# plt.show()
# plt.close()

# simulation.generate_sed(
#     scat=True,
#     read_lambda=[
#         1.3, 3, 7, 18
#     ],
#     load_simulation=False,
#     **simulate_mutual_parms
# )

# sed = simulation.spectrum
# lam = sed[:, 0]
# nu = (1e-2*cc)*1e-9/(1e-6*lam) # GHz
# fnu = sed[:, 1]*1e26/(140**2) # mJy
# plt.plot(nu, fnu, label=f'$a_{{max}} = {amax:.0e}$ mm')
# plt.show()
# plt.close()