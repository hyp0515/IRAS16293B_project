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
from radmc.simulate import Simulation
from sed.plot_sed import HG_19_data, this_work_data

"""
Define functions to analyze the synthetic and observed polarization data.
"""
def setup_model(amax_inner, amax_outer, mdot, rd, Toomre_Q, align=True):
    if align:
        inputstyle_index = 20
        align_model_index = -1
    else:
        inputstyle_index = 10
        align_model_index = 0
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
    opacity_dir_inner = ['temp_regime_1_inner', 'temp_regime_2_inner', 'temp_regime_3_inner', 'temp_regime_4_inner']
    model.generate_opacity_optool(a_max=amax_inner, composition='X22', 
                                  fnames=opacity_dir_inner, inputstyle=inputstyle_index) # a_max in mm
    opacity_tables_inner = []
    for dir in opacity_dir_inner:
        p = optool.particle('',
                            cache=f'./kappa/{dir}/',
                            silent=True)
        opacity_tables_inner.append(p)

    inner_opacity = model.combine_opacity_tables(opacity_tables_inner,
                                T_crit=[150, 425, 680, 1200],
                                fraction=[0.2, 0.3966, 0.0743, 0.3291],)
    x22model = model.X22(
        opacity_table=inner_opacity,
        Mass_of_star=0.5,
        Accretion_rate=mdot,
        Radius_of_disk=rd,
        Q=Toomre_Q,
    )

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
                            alignment_mode=align_model_index, # 1 for grain alignment
                            )
    setup.get_continuumlambda(filename=None,
                            comment=None,
                            lambda_micron=None,
                            append=False,
                            silent=True)
    setup.write_amr_grid()
    setup.write_dust_opac(dust_type=opacity_dir_inner, inputstyle=inputstyle_index, grain_align=align)
    setup.write_density_file()
    setup.write_temperature_file()
    setup.get_dustalignmentcontrol(alpha=1/(10*au*au), 
                                    hourglass=True, 
                                    uniform_z=True, 
                                    uniform_x=False, 
                                    uniform_y=False,
                                    toroidal=False)


def get_info(fname):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with fits.open(fname) as hdul:
            data = hdul[0].data
            header = hdul[0].header
        wcs = WCS(header=header)
        wcs = wcs.sub(['longitude', 'latitude'])
    return data, header, wcs

def get_beam_axis(header, distance_pc):
    au_per_pix = abs(header['CDELT1'])/180*np.pi*distance_pc*pc/au
    if 'BMAJ' in header and 'BMIN' in header and 'BPA' in header:
        beam_major_arcsec = header['BMAJ'] * 3600  # Major axis in arcseconds
        beam_minor_arcsec = header['BMIN'] * 3600  # Minor axis in arcseconds
        beam_major_pixels = (header['BMAJ']/180*np.pi)*distance_pc*pc/au / au_per_pix
        beam_minor_pixels = (header['BMIN']/180*np.pi)*distance_pc*pc/au / au_per_pix
        beam_pa = header['BPA']  # Position angle in degrees
    else:
        raise ValueError("Beam size (BMAJ and BMIN) not found in FITS header.")
    return (beam_major_arcsec, beam_minor_arcsec), (beam_major_pixels, beam_minor_pixels), beam_pa, au_per_pix

def crop_around_coord(data, wcs, center_coord, sizeau, au_per_pix):
    # Convert center coordinate to pixel position
    x_center, y_center = skycoord_to_pixel(center_coord, wcs)
    # Convert size in AU to size in pixels (half-width)
    half_size_pix = int(sizeau / (2 * au_per_pix))
    x_center = int(np.round(x_center))
    y_center = int(np.round(y_center))
    # Calculate crop boundaries
    x_min = max(x_center - half_size_pix, 0)
    x_max = min(x_center + half_size_pix, data.shape[1])
    y_min = max(y_center - half_size_pix, 0)
    y_max = min(y_center + half_size_pix, data.shape[0])
    # Crop data and WCS
    cropped_data = data[y_min:y_max, x_min:x_max]
    cropped_wcs = wcs.slice((slice(y_min, y_max), slice(x_min, x_max)))
    return cropped_data, cropped_wcs

def rotate_image(image, posang):
    rotated_image = np.zeros_like(image)
    rotated_image = ndimage.rotate(image, posang, reshape=False, axes=(1, 0))
    rotated_image = np.nan_to_num(rotated_image, nan=0)
    return rotated_image

def radial_intensity(image_array, center, width):
    if center is None:
        # peak_idx_x, peak_idx_y = np.unravel_index(np.argmax(image_array, axis=None), image_array.shape)
        # center = peak_idx_y
        center = image_array.shape[1] // 2
    if width != 1:
        radial_profile_major = np.mean(image_array[center-width//2:center+width//2, :], axis=0)
        radial_profile_minor = np.mean(image_array[:, center-width//2:center+width//2], axis=1)
    else:
        radial_profile_major = image_array[center, :]
        radial_profile_minor = image_array[:, center]
    return radial_profile_major, radial_profile_minor, center

def get_flux_density(image_array, rms, beam):
    beam_area_pixels = beam
    image_copy = image_array.copy()
    mask = image_array > 5 * rms
    image_copy[~mask] = np.nan
    total_flux = np.nansum(image_copy) / beam_area_pixels
    uncertainty = rms * mask.sum() / beam_area_pixels
    return total_flux, uncertainty


"""
Initialize observation (continuum and polarization)
"""
distance = 140

obs_wav = np.array([
    1300, 
    3000, 
    7000, 
    18000
])
obs_rms = np.array([
    104e-6, # 1300 micron
    17e-6, # 3000 micron
    47e-6, # 7000 micron
    47e-6, # 18000 micron
])
beam_info     = np.zeros((len(obs_wav), 3)) # beam major, beam minor, beam position angle
beam_plr_info = np.zeros((len(obs_wav), 3)) # beam major, beam minor, beam position angle

fname_band6_stokeI  = '~/project_data/IRAS16293/image_FITS/ALMA_Band6_Pol/c2d_1008_Stokes.I.regrid.fits'
fname_band6_stokeI_highres = '~/project_data/IRAS16293/image_FITS/ALMA_Band6_Pol/iras16293B_cont_b6_foralpha_pbcor.fits'
fname_band3_stokeI  = '~/project_data/IRAS16293/image_FITS/ALMA_Band3_StokesI/sourceB.image.dropdreg.fits'
fname_qband_stokeI  = '~/project_data/IRAS16293/image_FITS/JVLA_Qband_Pol/iras16293_Qband.rob0.I.regrid.fits'
fname_kuband_stokeI = '~/project_data/IRAS16293/image_FITS/JVLA_Kuband_Pol/rob0/I16293_Kuband_A.rob0.I.image.tt0.pbcor.subim.fits'

fname_band6_stokeQ  = '~/project_data/IRAS16293/image_FITS/ALMA_Band6_Pol/c2d_1008_Stokes.Q.regrid.fits'
fname_qband_stokeQ  = '~/project_data/IRAS16293/image_FITS/JVLA_Qband_Pol/iras16293_Qband.rob0.Q.regrid.fits'
fname_kuband_stokeQ = '~/project_data/IRAS16293/image_FITS/JVLA_Kuband_Pol/rob0/I16293_Kuband_A.rob0.Q.image.tt0.pbcor.subim.fits'

fname_band6_stokeU  = '~/project_data/IRAS16293/image_FITS/ALMA_Band6_Pol/c2d_1008_Stokes.U.regrid.fits'
fname_qband_stokeU  = '~/project_data/IRAS16293/image_FITS/JVLA_Qband_Pol/iras16293_Qband.rob0.U.regrid.fits'
fname_kuband_stokeU = '~/project_data/IRAS16293/image_FITS/JVLA_Kuband_Pol/rob0/I16293_Kuband_A.rob0.U.image.tt0.pbcor.subim.fits'

crop_sizeau = 150

# # Reference coords (Ku Band)
ra_ref = '16:32:22.6090'
dec_ref = '-24:28:32.675'
data, header, wcs = get_info(fname_kuband_stokeI)
# Reference coords (Q Band)
# ra_ref = '16:32:22.6145'
# dec_ref = '-24:28:32.555'
# data, header, wcs = get_info(fname_qband_stokeI)

t = Time(header['DATE-OBS'], format='isot', scale='utc')

pm_ra = -11.8 * u.mas / u.yr  # Proper motion in right ascension
pm_dec = -19.7 * u.mas / u.yr  # Proper motion in declination

# decimal_year = t.decimalyear
coord_ref = SkyCoord(ra=ra_ref, dec=dec_ref, pm_ra_cosdec=pm_ra, pm_dec=pm_dec,
                     frame='icrs', obstime=t, unit=('hourangle','deg'))


obs_stokeI_fnames = [
    fname_band6_stokeI_highres,
    fname_band3_stokeI,
    fname_qband_stokeI,
    fname_kuband_stokeI
]

obs_stokeQ_fnames = [
    fname_band6_stokeQ,
    None,
    fname_qband_stokeQ,
    fname_kuband_stokeQ
]
obs_stokeU_fnames = [
    fname_band6_stokeU,
    None,
    fname_qband_stokeU,
    fname_kuband_stokeU
]

obs_data_I_coords = []
obs_data_plr_coords = []
obs_data_I = []
obs_data_PA  = []
obs_data_Per = []
for i, (fname_I, fname_Q, fname_U) in enumerate(zip(obs_stokeI_fnames, obs_stokeQ_fnames, obs_stokeU_fnames)):
    data_I, header_I, wcs_I = get_info(fname_I)
    
    beam_arcsec_I, beam_pixel_I, beam_pa_I, au_per_pix_I = get_beam_axis(header=header_I, distance_pc=distance)
    beam_major_arcsec_I, beam_minor_arcsec_I = beam_arcsec_I
    beam_major_pixels_I, beam_minor_pixels_I = beam_pixel_I
    beam_info[i, :] = [beam_major_arcsec_I, beam_minor_arcsec_I, beam_pa_I]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        coords_shifted_I = coord_ref.apply_space_motion(Time(header_I['DATE-OBS'], format='isot', scale='utc'))
    if fname_I == fname_band6_stokeI_highres:
        data_crop_I, wcs_crop_I = crop_around_coord(data_I[0, 0, :, :], wcs_I, coords_shifted_I, crop_sizeau, au_per_pix_I)
    else:
        data_crop_I, wcs_crop_I = crop_around_coord(data_I, wcs_I, coords_shifted_I, crop_sizeau, au_per_pix_I)
    obs_data_I.append(data_crop_I)
    obs_data_I_coords.append(coords_shifted_I.to_string(style='hmsdms'))

    if (fname_Q is not None) or (fname_U is not None):
        data_Q, header_Q, wcs_Q = get_info(fname_Q)
        data_U, header_U, wcs_U = get_info(fname_U)

        beam_arcsec_plr, beam_pixel_plr, beam_pa_plr, au_per_pix_plr = get_beam_axis(header=header_Q, distance_pc=distance)
        beam_major_arcsec_plr, beam_minor_arcsec_plr = beam_arcsec_plr
        beam_major_pixels_plr, beam_minor_pixels_plr = beam_pixel_plr
        beam_plr_info[i, :] = [beam_major_arcsec_plr, beam_minor_arcsec_plr, beam_pa_plr]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            coords_shifted_plr = coord_ref.apply_space_motion(Time(header_Q['DATE-OBS'], format='isot', scale='utc'))
        data_crop_Q, wcs_crop_Q = crop_around_coord(data_Q, wcs_Q, coords_shifted_plr, crop_sizeau, au_per_pix_plr)
        data_crop_U, wcs_crop_U = crop_around_coord(data_U, wcs_U, coords_shifted_plr, crop_sizeau, au_per_pix_plr)

        if fname_Q == fname_band6_stokeQ:
            data_I, header_I, wcs_I = get_info(fname_band6_stokeI)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                coords_shifted_I = coord_ref.apply_space_motion(Time(header_I['DATE-OBS'], format='isot', scale='utc'))
            data_crop_I, wcs_crop_I = crop_around_coord(data_I, wcs_I, coords_shifted_I, crop_sizeau, au_per_pix_plr)
        
        PA = 0.5 * np.arctan2(-data_crop_U, -data_crop_Q)
        Per = np.sqrt(data_crop_Q**2 + data_crop_U**2) / data_crop_I

        obs_data_PA.append(PA)
        obs_data_Per.append(Per)
        obs_data_plr_coords.append(coords_shifted_plr.to_string(style='hmsdms'))
    else:
        obs_data_PA.append(None)
        obs_data_Per.append(None)
        obs_data_plr_coords.append(None)


def plot_conti_diff_params(simulation, obs, dir, fname):
    model_img = simulation.conti

    fig, axes = plt.subplots(4, len(obs_wav), figsize=(4*len(obs_wav), 4*len(obs_wav)))
    for i, lam in enumerate(obs_wav):
        
        conv_image = model_img.imConv(dpc=distance, fwhm=beam_info[i, :-1], pa=-(beam_info[i, -1]+90)) # Convolve with beam
        npix = len(model_img.x)
        pixel_area = (crop_sizeau/npix/140)**2
        beam_area = beam_info[i, 0]*beam_info[i, 0]*np.pi/(4*np.log(2))
        conv_image.imageJyppix *= beam_area/pixel_area/(distance**2) # Convert to Jy/pixel
        model = conv_image.imageJyppix[:, :, 0, i].T
        obs = obs_data_I[i]

        rotated_obs = rotate_image(obs, -90) # Rotate the image to match the model orientation (pa=0)
        radial_obs_major, radial_obs_minor, center = radial_intensity(rotated_obs, center=None, width=5)
        ax = axes[:, i]
        ax[0].imshow(rotated_obs, origin='lower', cmap="magma", vmin=0, vmax=np.nanmax(rotated_obs))
        ax[0].text(0.05, 0.95, f'{lam*1e-3:.1f} mm', transform=ax[0].transAxes, fontsize=12, color='white', va='top')
        ax[0].hlines(center, xmin=0, xmax=rotated_obs.shape[0]-1, color='violet', linestyle='--', linewidth=1)
        ax[0].vlines(center, ymin=0, ymax=rotated_obs.shape[1]-1, color='dodgerblue', linestyle='--', linewidth=1)
        ax[0].set_xticks([0, rotated_obs.shape[0]//2, rotated_obs.shape[0]-1])
        ax[0].set_xticklabels([-(crop_sizeau//2), 0, (crop_sizeau//2)])
        ax[0].set_xlabel('AU')
        ax[0].set_yticks([0, rotated_obs.shape[1]//2, rotated_obs.shape[1]-1])
        ax[0].set_yticklabels([-(crop_sizeau//2), 0, (crop_sizeau//2)])
        ax[0].set_ylabel('AU')

        radial_model_major, radial_model_minor, center = radial_intensity(model, center=None, width=5)
        
        ax[1].imshow(model, origin='lower', cmap="magma", vmin=0, vmax=np.nanmax(rotated_obs))
        ax[1].hlines(center, xmin=0, xmax=model.shape[0]-1, color='violet', linestyle='-', linewidth=1)
        ax[1].vlines(center, ymin=0, ymax=model.shape[1]-1, color='dodgerblue', linestyle='-', linewidth=1)
        ax[1].set_xticks([0, model.shape[0]//2, model.shape[0]-1])
        ax[1].set_xticklabels([-(crop_sizeau//2), 0, (crop_sizeau//2)])
        ax[1].set_xlabel('AU')
        ax[1].set_yticks([0, model.shape[1]//2, model.shape[1]-1])
        ax[1].set_yticklabels([-(crop_sizeau//2), 0, (crop_sizeau//2)])
        ax[1].set_ylabel('AU')

        interp_obs = ndimage.zoom(rotated_obs, zoom=model.shape[0] / rotated_obs.shape[0])
        radial_obs_major, radial_obs_minor, center = radial_intensity(interp_obs, center=None, width=5)

        residual = interp_obs - model

        ax[2].imshow(residual, origin='lower', cmap="seismic", vmin=-(np.nanmax(rotated_obs)/2), vmax=np.nanmax(rotated_obs)/2)
        ax[2].set_xticks([0, residual.shape[0]//2, residual.shape[0]-1])
        ax[2].set_xticklabels([-(crop_sizeau//2), 0, (crop_sizeau//2)])
        ax[2].set_xlabel('AU')
        ax[2].set_yticks([0, residual.shape[1]//2, residual.shape[1]-1])
        ax[2].set_yticklabels([-(crop_sizeau//2), 0, (crop_sizeau//2)])
        ax[2].set_ylabel('AU')

        ax[3].plot(np.linspace(-100, 100, num=radial_obs_major.size, endpoint=True), radial_obs_major, color='violet', linestyle='--')
        ax[3].plot(np.linspace(-100, 100, num=radial_obs_minor.size, endpoint=True), radial_obs_minor, color='dodgerblue', linestyle='--')
        ax[3].plot(np.linspace(-100, 100, num=radial_model_major.size, endpoint=True), radial_model_major, color='violet')
        ax[3].plot(np.linspace(-100, 100, num=radial_model_minor.size, endpoint=True), radial_model_minor, color='dodgerblue')
        ax[3].vlines(0, ymin=0, ymax=max(np.nanmax(radial_obs_major), np.nanmax(radial_model_major)), color='k', linestyle='--', linewidth=1)
        ax[3].text(0.05, 0.95, f'{lam*1e-3:.1f} mm', transform=ax[3].transAxes, fontsize=12, color='k', va='top')
        # ax[3].set_title(f'Radial profile at {lam} $\mu$m')
        ax[3].set_xlabel('AU')
        ax[3].set_ylabel('Intensity (Jy/pixel)')
        ax[3].set_xlim((-(crop_sizeau//2), crop_sizeau//2))
        ax[3].set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f'{dir}{fname}.pdf', transparent=True)
    plt.close()

def b_segment(PA, Per, step=8):
    y, x = np.mgrid[0:PA.shape[0], 0:PA.shape[1]]
    x_sampled = x[::step, ::step]
    y_sampled = y[::step, ::step]
    angle_sampled = PA[::step, ::step]
    intensity_sampled = Per[::step, ::step]
    u_segment = (intensity_sampled/0.02) * np.cos(angle_sampled)
    v_segment = (intensity_sampled/0.02) * np.sin(angle_sampled)
    return x_sampled, y_sampled, u_segment, v_segment

def plot_plr_diff_params(simulation, obs, dir, fname):
    
    obs_I, obs_PA, obs_Per = obs
    model_img = simulation.conti

    fig, axes = plt.subplots(2, len(obs_wav), figsize=(4*len(obs_wav), 2*len(obs_wav)))

    sampled_step = [9, None, 10, 2]
    for i, lam in enumerate(obs_wav):
        ax = axes[:, i]
    
        '''
        Plot observed continuum and polarization data
        '''
        rotated_obs_I = rotate_image(obs_I[i], -90) # Rotate the image to match the model orientation (pa=0)
        if obs_PA[i] is None or obs_Per[i] is None:
            ax[0].imshow(rotated_obs_I, origin='lower', cmap="magma", vmin=0, vmax=np.nanmax(rotated_obs_I), 
                    extent=[0, rotated_obs_I.shape[1], 0, rotated_obs_I.shape[0]])
        else:
            rotated_obs_PA = rotate_image(obs_PA[i], -90)
            rotated_obs_Per = rotate_image(obs_Per[i], -90)
            if i == 0: # Band 6
                interp_rotated_obs_I = ndimage.zoom(rotated_obs_I, zoom=rotated_obs_PA.shape[0] / rotated_obs_I.shape[0])
                mask_obs = interp_rotated_obs_I > 10 * obs_rms[i]
            else:
                mask_obs = rotated_obs_I > 10 * obs_rms[i]
            rotated_obs_PA[~mask_obs] = np.nan
            rotated_obs_Per[~mask_obs] = np.nan
            
            
            x_sampled_obs, y_sampled_obs, u_segment_obs, v_segment_obs = b_segment(rotated_obs_PA, rotated_obs_Per, step=sampled_step[i])
            
            ax[0].quiver(x_sampled_obs, y_sampled_obs, u_segment_obs, v_segment_obs, color='white', 
                        scale=50, headlength=0, headaxislength=0, headwidth=0)
            ax[0].imshow(rotated_obs_I, origin='lower', cmap="magma", vmin=0, vmax=np.nanmax(rotated_obs_I), 
                    extent=[0, rotated_obs_PA.shape[1], 0, rotated_obs_PA.shape[0]])
            ax[0].text(0.05, 0.95, f'{lam*1e-3:.1f} mm', transform=ax[0].transAxes, fontsize=12, color='white', va='top')
            ax[0].set_xticks([0, rotated_obs_I.shape[0]//2, rotated_obs_I.shape[0]-1])
            ax[0].set_xticklabels([-(crop_sizeau//2), 0, (crop_sizeau//2)])
            ax[0].set_xlabel('AU')
            ax[0].set_yticks([0, rotated_obs_I.shape[1]//2, rotated_obs_I.shape[1]-1])
            ax[0].set_yticklabels([-(crop_sizeau//2), 0, (crop_sizeau//2)])
            ax[0].set_ylabel('AU')
            
        '''
        Plot synthetic continuum and polarization
        '''
        # save_image_to_fits(model_img, beam_info[i, :], obs_data_I_coords[i], f'{dir}model_{lam}um', stokes=['I'])
        # model_I, header_I, wcs_I = get_info(f'{dir}model_{lam}um_conv_I.fits')
        # model_I = model_I[i, :, :]
        # mask_model = model_I > 10 * obs_rms[i]
        # ax[1].imshow(model_I, origin='lower', cmap="magma", vmin=0, vmax=np.nanmax(rotated_obs_I))

        # if obs_data_plr_coords[i] is None:
        #     pass
        # else:
        #     save_image_to_fits(model_img, beam_plr_info[i, :], obs_data_plr_coords[i], f'{dir}model_{lam}um', stokes=['Q', 'U'])
        #     model_Q, header_Q, wcs_Q = get_info(f'{dir}model_{lam}um_conv_Q.fits')
        #     model_U, header_U, wcs_U = get_info(f'{dir}model_{lam}um_conv_U.fits')
        #     model_Q = model_Q[i, :, :]
        #     model_U = model_U[i, :, :]
        #     model_PA = 0.5 * np.arctan2(-model_U, -model_Q)
        #     model_Per = np.sqrt(model_Q**2 + model_U**2) / model_I
        #     model_PA[~mask_model] = np.nan
        #     model_Per[~mask_model] = np.nan
        #     x_sampled_model, y_sampled_model, u_segment_model, v_segment_model = b_segment(model_PA, model_Per, step=12)
        #     ax[1].quiver(x_sampled_model, y_sampled_model, u_segment_model, v_segment_model, color='white', 
        #                 scale=50, headlength=0, headaxislength=0, headwidth=0)
    
        # ax[1].set_xticks([0, model_I.shape[0]//2, model_I.shape[0]-1])
        # ax[1].set_xticklabels([-(crop_sizeau//2), 0, (crop_sizeau//2)])
        # ax[1].set_xlabel('AU')
        # ax[1].set_yticks([0, model_I.shape[1]//2, model_I.shape[1]-1])
        # ax[1].set_yticklabels([-(crop_sizeau//2), 0, (crop_sizeau//2)])
        # ax[1].set_ylabel('AU')

        conv_image_I   = model_img.imConv(dpc=distance, fwhm=beam_info[i, :-1], pa=-(beam_info[i, -1]+90)) # Convolve with beam
        
        npix = len(model_img.x)
        pixel_area = (crop_sizeau/npix/140)**2
        beam_area_I = beam_info[i, 0]*beam_info[i, 0]*np.pi/(4*np.log(2))
        conv_image_I.imageJyppix *= beam_area_I/pixel_area/(distance**2) # Convert to Jy/pixel
        model_I = conv_image_I.imageJyppix[:, :, 0, i].T
        mask_model = model_I > 10 * obs_rms[i]

        conv_image_plr = model_img.imConv(dpc=distance, fwhm=beam_plr_info[i, :-1], pa=-(beam_plr_info[i, -1]+90)) 
        beam_area_plr = beam_plr_info[i, 0]*beam_plr_info[i, 0]*np.pi/(4*np.log(2))
        conv_image_plr.imageJyppix *= beam_area_plr/pixel_area/(distance**2) # Convert to Jy/pixel
        model_Q = conv_image_plr.imageJyppix[:, :, 1, i].T
        model_U = conv_image_plr.imageJyppix[:, :, 2, i].T
        model_PA = 0.5 * np.arctan2(-model_U, -model_Q)
        model_Per = np.sqrt(model_Q**2 + model_U**2) / model_I
        model_PA[~mask_model] = np.nan
        model_Per[~mask_model] = np.nan
        x_sampled_model, y_sampled_model, u_segment_model, v_segment_model = b_segment(model_PA, model_Per, step=15)
        ax[1].imshow(model_I, origin='lower', cmap="magma", vmin=0, vmax=np.nanmax(rotated_obs_I))
        ax[1].quiver(x_sampled_model, y_sampled_model, u_segment_model, v_segment_model, color='white', 
                     scale=50, headlength=0, headaxislength=0, headwidth=0)
        ax[1].set_xticks([0, model_I.shape[0]//2, model_I.shape[0]-1])
        ax[1].set_xticklabels([-(crop_sizeau//2), 0, (crop_sizeau//2)])
        ax[1].set_xlabel('AU')
        ax[1].set_yticks([0, model_I.shape[1]//2, model_I.shape[1]-1])
        ax[1].set_yticklabels([-(crop_sizeau//2), 0, (crop_sizeau//2)])
        ax[1].set_ylabel('AU')

    plt.tight_layout()
    plt.savefig(f'{dir}{fname}.pdf', transparent=True)
    plt.close()

def save_image_to_fits(model_image, beam_info, coords, fname, stokes=['I', 'Q', 'U']):
    cim = model_image.imConv(fwhm=[beam_info[0], beam_info[1]], pa=-(beam_info[2]+90), dpc=distance,)
    pix_x_au = cim.sizepix_x/au
    pix_y_au = cim.sizepix_y/au
    pix_x_arcsec = pix_x_au / distance
    pix_y_arcsec = pix_y_au / distance
    pixel_area = pix_x_arcsec * pix_y_arcsec
    beam_area = cim.fwhm[0] * cim.fwhm[1] * np.pi / (4 * np.log(2))
    ratio = beam_area / pixel_area

    data = cim.imageJyppix * ratio / (distance ** 2)
    data = data * 1e-23  # Convert to Jy/pixel
    cim.image = data
    for stoke in stokes:
        FITS_name = f'{fname}_conv_{stoke}.fits'
        os.system('rm -rf '+FITS_name)
        cim.writeFits(FITS_name, coord=coords, stokes=stoke,)


a_list = [10, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]

for ain in a_list:
    incl = 45
    ain = ain * 1e-3  # Convert um to mm
    mdot= 5e-5
    rd = 35
    Q = 0.5

    setup_model(ain, None, mdot, rd, Q, align=False)
    simulation = Simulation(save_out=True, save_npz=False)
    simulate_mutual_parms = {
        "incl"      : incl,
        "npix"      : 500,
        "sizeau"    : crop_sizeau,
        "posang"    : 0,
        "phi"       : 0,
        "dir"       : f'/run/media/hyp0515/storage/ext_fig_3_noscat/amax_{ain}/',
    }
    simulation.generate_continuum(
        scat=False,
        stokes=True,
        read_lambda=obs_wav*1e-3,
        load_simulation=False,
        fname=f'conti',
        **simulate_mutual_parms
    )

def b_T(wav, beam_info, data_jybeam):
    return 1.36*(wav*1e-1*1e-3)**2*data_jybeam*1e3 / (beam_info[0] * beam_info[1])

Tb_max = [300, 500, 150, 300]

fig, axes = plt.subplots(len(a_list)+1, 4, figsize=(4*4, 4*(len(a_list)+1)), constrained_layout=True)
# plt.subplots_adjust(wspace=0.0, hspace=0.0, top=0.9, bottom=0.1, left=0.1, right=0.9)

for i, lam in enumerate(obs_wav):
    ax = axes[0, :]
    '''
    Plot observed continuum and polarization data
    '''
    rotated_obs_I = rotate_image(obs_data_I[i], -90) # Rotate the image to match the model orientation (pa=0)
    rotated_obs_I_bT = b_T(lam, beam_info[i, :], rotated_obs_I)
    if obs_data_PA[i] is None or obs_data_Per[i] is None:
        ax[i].imshow(rotated_obs_I_bT, origin='lower', cmap="magma", vmin=0, vmax=Tb_max[i], 
                extent=[0, rotated_obs_I.shape[1], 0, rotated_obs_I.shape[0]])
        fig.colorbar(ax[i].images[0], ax=ax[i], orientation='horizontal', location='top', label='Brightness Temperature (K)',
                    pad=0.00, shrink=1)
    else:
        rotated_obs_PA = rotate_image(obs_data_PA[i], -90)
        rotated_obs_Per = rotate_image(obs_data_Per[i], -90)
        if i == 0: # Band 6
            interp_rotated_obs_I = ndimage.zoom(rotated_obs_I, zoom=rotated_obs_PA.shape[0] / rotated_obs_I.shape[0])
            mask_obs = interp_rotated_obs_I > 10 * obs_rms[i]
        else:
            mask_obs = rotated_obs_I > 10 * obs_rms[i]
        rotated_obs_PA[~mask_obs] = np.nan
        rotated_obs_Per[~mask_obs] = np.nan
        
        
        x_sampled_obs, y_sampled_obs, u_segment_obs, v_segment_obs = b_segment(rotated_obs_PA, rotated_obs_Per, step=[9, None, 10, 2][i])
        
        ax[i].quiver(x_sampled_obs, y_sampled_obs, u_segment_obs, v_segment_obs, color='cyan', 
                    scale=50, headlength=0, headaxislength=0, headwidth=0)
        ax[i].imshow(rotated_obs_I_bT, origin='lower', cmap="magma", vmin=0, vmax=Tb_max[i], 
                extent=[0, rotated_obs_PA.shape[1], 0, rotated_obs_PA.shape[0]])
        fig.colorbar(ax[i].images[0], ax=ax[i], orientation='horizontal', location='top', label='Brightness Temperature (K)',
                    pad=0.00, shrink=1)
        # ax[i].text(0.05, 0.95, f'{lam*1e-3:.1f} mm', transform=ax[i].transAxes, fontsize=12, color='white', va='top')
        # ax[i].set_xticks([0, rotated_obs_I.shape[0]//2, rotated_obs_I.shape[0]-1])
        # ax[0].set_xticklabels([-(crop_sizeau//2), 0, (crop_sizeau//2)])
        # ax[0].set_xlabel('AU')
        # ax[0].set_yticks([0, rotated_obs_I.shape[1]//2, rotated_obs_I.shape[1]-1])
        # ax[0].set_yticklabels([-(crop_sizeau//2), 0, (crop_sizeau//2)])
        # ax[0].set_ylabel('AU')
    ax[i].text(0.05, 0.95, r"$\lambda$="+f'{lam*1e-3:.1f}mm', transform=ax[i].transAxes, fontsize=14, color='white', va='top')
    ax[i].set_xticks([])
    ax[i].set_yticks([])

for j, ain in enumerate(np.array(a_list)*1e-3):
        
    simulation = Simulation(save_out=True, save_npz=False)
    simulate_mutual_parms = {
        "incl"      : 45,
        "npix"      : 500,
        "sizeau"    : crop_sizeau,
        "posang"    : 0,
        "phi"       : 0,
        "dir"       : f'/run/media/hyp0515/storage/ext_fig_3_noscat/amax_{ain}/',
    }
    simulation.generate_continuum(
        scat=False,
        stokes=True,
        read_lambda=obs_wav*1e-3,
        load_simulation=True,
        fname=f'conti',
        **simulate_mutual_parms
    )
    model_img = simulation.conti
    
    for i, lam in enumerate(obs_wav):
        ax = axes[j+1, :]
        conv_image_I   = model_img.imConv(dpc=distance, fwhm=beam_info[i, :-1], pa=-(beam_info[i, -1]+90)) # Convolve with beam
        
        npix = len(model_img.x)
        pixel_area = (crop_sizeau/npix/140)**2
        beam_area_I = beam_info[i, 0]*beam_info[i, 0]*np.pi/(4*np.log(2))
        conv_image_I.imageJyppix *= beam_area_I/pixel_area/(distance**2) # Convert to Jy/pixel
        model_I = conv_image_I.imageJyppix[:, :, 0, i].T
        mask_model = model_I > 10 * obs_rms[i]

        conv_image_plr = model_img.imConv(dpc=distance, fwhm=beam_plr_info[i, :-1], pa=-(beam_plr_info[i, -1]+90)) 
        beam_area_plr = beam_plr_info[i, 0]*beam_plr_info[i, 0]*np.pi/(4*np.log(2))
        conv_image_plr.imageJyppix *= beam_area_plr/pixel_area/(distance**2) # Convert to Jy/pixel
        model_Q = conv_image_plr.imageJyppix[:, :, 1, i].T
        model_U = conv_image_plr.imageJyppix[:, :, 2, i].T
        model_PA = 0.5 * np.arctan2(-model_U, -model_Q)
        model_Per = np.sqrt(model_Q**2 + model_U**2) / model_I
        model_PA[~mask_model] = np.nan
        model_Per[~mask_model] = np.nan
        x_sampled_model, y_sampled_model, u_segment_model, v_segment_model = b_segment(model_PA, model_Per, step=15)
        model_I_bT = b_T(lam, beam_info[i, :], model_I)
        ax[i].imshow(model_I_bT, origin='lower', cmap="magma", vmin=0, vmax=Tb_max[i])
        ax[i].quiver(x_sampled_model, y_sampled_model, u_segment_model, v_segment_model, color='cyan', pivot='middle',
                     scale=50, headlength=0, headaxislength=0, headwidth=0)
        # ax[i].text(0.05, 0.95, r"$\lambda$="+f'{lam*1e-3:.1f}mm', transform=ax[i].transAxes, fontsize=12, color='white', va='top')
        ax[i].text(0.55, 0.95, r"$a_{max}=$"+f'{ain:.2f} mm', transform=ax[i].transAxes, fontsize=14, color='white', va='top')
        ax[i].set_xticks([])
        ax[i].set_yticks([])

# plt.tight_layout()
plt.savefig('./ext_fig_3_noscat.pdf', transparent=True)
# plt.show()