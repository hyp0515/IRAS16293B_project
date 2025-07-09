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


amax        = 1 # maximum grain size in mm
mstar       = 0.5 # stellar mass in solar masses
mdot        = 5e-5 # accretion rate in solar masses per year
rd          = 50 # disk radius in AU
Toomre_Q    = 0.4 # Toomre Q parameter
l_star      = .1 # stellar luminosity in solar luminosities
heating     = 'accretion' # heating mechanism


model = radmc3d_setup(silent=False)
model.get_mastercontrol(filename=None,
                        comment=None,
                        incl_dust=1,
                        incl_lines=0,
                        nphot=500000,
                        nphot_scat=5000000,
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



simulation = generate_simulation(save_out=True, save_npz=True)

simulate_mutual_parms = {
    "incl"      : 90,
    "npix"      : 500,
    "sizeau"    : 150,
    "posang"    : 0,
    "dir"       : './test/',
    "fname"     : 'test',
}



simulation.generate_continuum(
   scat=True,
   wav=3000,
   stokes=True,
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

model_img = image.readImage(fname=f'./{f_dir}/outfile/conti_{f_name}_scat_stokes.out')

conv_image = model_img.imConv(dpc=distance, fwhm=beam_axis, pa=-79.32)
conv_image.imageJyppix *= beam_area/pixel_area/(distance**2)

mask = conv_image.imageJyppix[:, :, 0, 0] > 1e-19

for i in range(4):
    conv_image.imageJyppix[:, :, i, 0][~mask] = np.nan

polang  = 0.5 * np.arctan2(conv_image.imageJyppix[:, :, 2, 0], conv_image.imageJyppix[:, :, 1, 0]) + np.pi / 2
polfrac = np.sqrt(conv_image.imageJyppix[:, :, 1, 0]**2 + conv_image.imageJyppix[:, :, 2, 0]**2) / conv_image.imageJyppix[:, :, 0, 0]

fig, ax = plt.subplots(1, 6, figsize=(24, 4), sharex=True, sharey=True)
ax = ax.flatten()
for i in range(4):
    ax[i].imshow(conv_image.imageJyppix[:, :, i, 0].T, origin='lower', cmap="magma")
    ax[i].set_title(f'Stokes {["I", "Q", "U", "V"][i]}')
ax[4].imshow(polang.T, origin='lower', cmap="seismic", vmin=-np.pi+np.pi/2, vmax=np.pi+np.pi/2)
ax[4].set_title('Polarization Angle + 90 deg')

ax[5].imshow(polfrac.T, origin='lower', cmap="viridis", vmin=0, vmax=0.1)
ax[5].set_title('Polarization Fraction')

y, x = np.mgrid[0:conv_image.imageJyppix.shape[0], 0:conv_image.imageJyppix.shape[1]]
step = 20
x_ds, y_ds = x[::step, ::step], y[::step, ::step]
polfrac_ds = polfrac[::step, ::step]
polang_ds = polang[::step, ::step]

u = polfrac_ds * np.cos(polang_ds) / 0.01
v = polfrac_ds * np.sin(polang_ds) / 0.01

ax[0].quiver(x_ds, y_ds, u, v, color='white', scale=50, headlength=0, headaxislength=0, headwidth=0)


plt.savefig('test.pdf', transparent=True)