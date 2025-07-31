import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import sys
import os
import io
import contextlib
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from radmc3dPy import image
from radmc3dPy.analyze import * 

sys.path.append('..')
from X22_model.disk_model import generate_disk_property_table
from radmc.setup import *
from radmc.simulate import generate_simulation


SPH = Grid()
SPH.sph_grid(
    r_bound =[1e-2*au, 50*au],
    theta_bound = [np.pi/8, 7*np.pi/8],
    phi_bound = [ 0, 2*np.pi],
    nr=200,
    ntheta=200,
    nphi=20,
)


model = Model()
model.generate_opacity_optool(a_max=1.0, composition='X22') # a_max in mm
opacity_tables = []
opacity_dir = ['temp_regime_1', 'temp_regime_2', 'temp_regime_3', 'temp_regime_4']
for dir in opacity_dir:
    p = optool.particle('',
                        cache=f'./kappa/{dir}/',
                        silent=True)
    opacity_tables.append(p)

inner_opacity = model.combine_opacity_tables(opacity_tables,
                            T_crit=[150, 425, 680, 1200],
                            fraction=[0.2, 0.3966, 0.0743, 0.3291],)
x22model = model.X22(
    opacity_table=inner_opacity,
    Mass_of_star=0.5,
    Accretion_rate=5e-5,
    Radius_of_disk=40,
    Q=0.7,
)
rho = model.rho_dust



outer_model = Model()
outer_model.generate_opacity_optool(a_max=0.01, composition='dsharp')
p = optool.particle('',
                    cache='./kappa/dsharp/',
                    silent=True)
outer_opacity = outer_model.read_opacity_table(p,)
outer_layer = outer_model.X22(
    opacity_table=outer_opacity,
    Mass_of_star=0.5,
    Accretion_rate=5e-5,
    Radius_of_disk=45,
    Q=0.7,
)
rho[:, :, :, 0] = outer_model.rho_dust[:, :, :, 0]


model.rho_dust = rho

model.interp_model_to_grid(grid=SPH)


setup = Setup(model_class=model, grid_class=SPH, silent=True)
setup.get_mastercontrol(filename=None,
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
setup.get_continuumlambda(filename=None,
                        comment=None,
                        lambda_micron=None,
                        append=False,
                        silent=True)
setup.write_amr_grid()
setup.write_dust_opac(dust_type=['dsharp']+opacity_dir[:-1],)
setup.write_density_file()
setup.write_temperature_file()
setup.get_dustalignmentcontrol(alpha=1/(10*au*au), 
                                hourglass=False, 
                                uniform_z=True, 
                                uniform_x=False, 
                                uniform_y=False,
                                toroidal=False)


simulation = generate_simulation(save_out=True, save_npz=False)
simulate_mutual_parms = {
    "incl"      : 50,
    "npix"      : 500,
    "sizeau"    : 200,
    "posang"    : 0,
    "phi"       : 0,
    "dir"       : f'./test/',
}
obs_wav = np.array([
    1300, 
    3000, 
    7000, 
    18000
])
simulation.generate_sed(
    scat=True,
    read_lambda=obs_wav*1e-3,
    load_simulation=False,
    fname='sed',
    **simulate_mutual_parms
)
simulation.generate_continuum(
    scat=True,
    stokes=True,
    read_lambda=obs_wav*1e-3,
    load_simulation=False,
    fname=f'conti',
    **simulate_mutual_parms
)

im = image.readImage('./test/outfile/conti_conti_scat_stokes.out')

cim = im.imConv(fwhm=[0.23, 0.09], pa=-(5.19+90), dpc=140,)
pix_x_au = cim.sizepix_x/au
pix_y_au = cim.sizepix_y/au
pix_x_arcsec = pix_x_au / 140
pix_y_arcsec = pix_y_au / 140

pixel_area = pix_x_arcsec * pix_y_arcsec
beam_area = cim.fwhm[0] * cim.fwhm[1] * np.pi / (4 * np.log(2))
ratio = beam_area / pixel_area

data = cim.imageJyppix * ratio / (140 ** 2)
data = data * 1e-23
stokes = ['I', 'Q', 'U']

cim.image = data
ra_ref = '16h32m22.6090s'
dec_ref = '-24d28m32.675s'
for stoke in stokes:
    FITS_name = 'modelimage_conv_{}.fits'.format(stoke)
    os.system('rm -rf '+FITS_name)
    cim.writeFits(FITS_name, coord=ra_ref+' '+dec_ref, stokes=stoke,)


def get_info(fname):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with fits.open(fname) as hdul:
            data = hdul[0].data[0, :, :]
            header = hdul[0].header
        wcs = WCS(header=header)
        wcs = wcs.sub(['longitude', 'latitude'])
    return data, header, wcs

data_I, header_I, wcs = get_info('modelimage_conv_I.fits')
data_Q, header_Q, wcs = get_info('modelimage_conv_Q.fits')
data_U, header_U, wcs = get_info('modelimage_conv_U.fits')


mask = data_I > 1e-8
data_I[~mask] = np.nan
data_Q[~mask] = np.nan
data_U[~mask] = np.nan

print(f"Data shapes: I={data_I.shape}, Q={data_Q.shape}, U={data_U.shape}")

PA = np.arctan2(-data_U, -data_Q) / 2 
Per = np.sqrt(data_Q**2 + data_U**2) / data_I


def b_segment(PA, Per, step=15):
    y, x = np.mgrid[0:PA.shape[0], 0:PA.shape[1]]
    x_sampled = x[::step, ::step]
    y_sampled = y[::step, ::step]
    angle_sampled = PA[::step, ::step]
    intensity_sampled = Per[::step, ::step]
    u_segment = (intensity_sampled/0.02) * np.cos(angle_sampled)
    v_segment = (intensity_sampled/0.02) * np.sin(angle_sampled)
    return x_sampled, y_sampled, u_segment, v_segment
x_sampled, y_sampled, u_segment, v_segment = b_segment(PA, Per, step=15)

fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
i_img = ax[0].imshow(data_I, origin='lower', cmap='magma')
fig.colorbar(i_img, ax=ax[0], orientation='vertical', label='Stokes I')
ax[0].quiver(x_sampled, y_sampled, u_segment, v_segment,
             color='cyan', angles='xy', scale_units='xy', pivot='mid',
             scale=0.1, headlength=0, headaxislength=0, headwidth=0, width=.005)
ax[0].set_title('Stokes I with B-field segments')
ax[0].set_xlabel('X (pixels)')
ax[0].set_ylabel('Y (pixels)')

q_img = ax[1].imshow(-data_Q, origin='lower', cmap='magma')
fig.colorbar(q_img, ax=ax[1], orientation='vertical', label='Stokes Q')
ax[1].set_title('Stokes Q')

u_img = ax[2].imshow(-data_U, origin='lower', cmap='seismic')
fig.colorbar(u_img, ax=ax[2], orientation='vertical', label='Stokes U')
ax[2].set_title('Stokes U')

plt.savefig('stokes_images_with_bfield_segments.pdf', transparent=True)