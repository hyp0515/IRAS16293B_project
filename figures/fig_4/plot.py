import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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
def setup_model(amax_inner, amax_middle, amax_outer, mdot, rd, Toomre_Q, align=True):
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

    # inner layer
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
    rho = model.rho_dust

    # middle layer
    middle_model = Model()
    opacity_dir_mid = ['temp_regime_1_middle', 'temp_regime_2_middle', 'temp_regime_3_middle', 'temp_regime_4_middle']
    middle_model.generate_opacity_optool(a_max=amax_middle, composition='X22', 
                                        fnames=opacity_dir_mid, inputstyle=inputstyle_index)
    opacity_tables_middle = []
    for dir in opacity_dir_mid:
        p = optool.particle('',
                            cache=f'./kappa/{dir}/',
                            silent=True)
        opacity_tables_middle.append(p)
    middle_opacity = middle_model.combine_opacity_tables(opacity_tables_middle,
                                T_crit=[150, 425, 680, 1200],
                                fraction=[0.2, 0.3966, 0.0743, 0.3291],)
    middle_layer = middle_model.X22(
        opacity_table=middle_opacity,
        Mass_of_star=0.5,
        Accretion_rate=mdot,
        Radius_of_disk=rd,
        Q=Toomre_Q,
    )
    rho[:, :, :, 1] = middle_model.rho_dust[:, :, :, 1]

    # outer layer
    outer_model = Model()
    opacity_dir_outer = ['temp_regime_1_outer', 'temp_regime_2_outer', 'temp_regime_3_outer', 'temp_regime_4_outer']
    outer_model.generate_opacity_optool(a_max=amax_outer, composition='X22', 
                                        fnames=opacity_dir_outer, inputstyle=inputstyle_index)
    opacity_tables_outer = []
    for dir in opacity_dir_outer:
        p = optool.particle('',
                            cache=f'./kappa/{dir}/',
                            silent=True)
        opacity_tables_outer.append(p)
    outer_opacity = outer_model.combine_opacity_tables(opacity_tables_outer,
                                T_crit=[150, 425, 680, 1200],
                                fraction=[0.2, 0.3966, 0.0743, 0.3291],)
    outer_layer = outer_model.X22(
        opacity_table=outer_opacity,
        Mass_of_star=0.5,
        Accretion_rate=mdot,
        Radius_of_disk=rd,
        Q=Toomre_Q,
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
                            alignment_mode=align_model_index, # 1 for grain alignment
                            )
    setup.get_continuumlambda(filename=None,
                            comment=None,
                            lambda_micron=None,
                            append=False,
                            silent=True)
    setup.write_amr_grid()
    setup.write_dust_opac(dust_type=opacity_dir_outer[:1]+opacity_dir_inner[1:], inputstyle=inputstyle_index, grain_align=align)
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
distance = 140 # pc
crop_sizeau = 150 # AU

obs_wav = np.array([ # in microns
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
obs_data_wcs = []
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
        
        PA = 0.5 * np.arctan2(data_crop_U, data_crop_Q)
        Per = np.sqrt(data_crop_Q**2 + data_crop_U**2) / data_crop_I

        obs_data_PA.append(PA)
        obs_data_Per.append(Per)
        obs_data_plr_coords.append(coords_shifted_plr.to_string(style='hmsdms'))
        obs_data_wcs.append(wcs_crop_Q)
    else:
        obs_data_PA.append(None)
        obs_data_Per.append(None)
        obs_data_plr_coords.append(None)
        obs_data_wcs.append(wcs_crop_I)

def b_segment(PA, Per, step=8): # b-field segment
    y, x = np.mgrid[0:PA.shape[0], 0:PA.shape[1]]
    x_sampled = x[::step, ::step]
    y_sampled = y[::step, ::step]
    angle_sampled = PA[::step, ::step]
    intensity_sampled = Per[::step, ::step]
    u_segment = (intensity_sampled/0.02) * np.cos(angle_sampled)
    v_segment = (intensity_sampled/0.02) * np.sin(angle_sampled)
    return x_sampled, y_sampled, u_segment, v_segment

def b_T(wav, beam_info, data_jybeam): # Convert Jy/beam to brightness temperature
    return 1.36*(wav*1e-4)**2*data_jybeam*1e3 / (beam_info[0] * beam_info[1])

def save_image_to_fits(model_image, beam_info, coords, fname, stokes=['I', 'Q', 'U']):
    cim = model_image.imConv(fwhm=[beam_info[0], beam_info[1]], pa=-(beam_info[2]), dpc=distance,)
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

def plot_plr_diff_params(simulation, obs, dir, fname):
    
    obs_I, obs_PA, obs_Per = obs
    model_img = simulation.conti

    fig, axes = plt.subplots(2, len(obs_wav), figsize=(4*len(obs_wav), 2*len(obs_wav)), constrained_layout=True)

    for i, lam in enumerate(obs_wav):
        ax = axes[:, i]
    
        '''
        Plot observed continuum and polarization data
        '''

        if i != 1:
            interp_obs_I = ndimage.zoom(obs_I[i], zoom=obs_PA[i].shape[0] / obs_I[i].shape[0])
            mask_obs = interp_obs_I > [25, None, 25, 5][i] * obs_rms[i]
            obs_PA[i][~mask_obs] = np.nan
            obs_Per[i][~mask_obs] = np.nan
            x_sampled_obs, y_sampled_obs, u_segment_obs, v_segment_obs = b_segment(obs_PA[i], obs_Per[i], step=[10, None, 12, 4][i])

            ax[0].quiver(x_sampled_obs, y_sampled_obs, u_segment_obs, v_segment_obs, color=polar_color, 
                        scale=30, headlength=0, headaxislength=0, headwidth=0, pivot='middle', width=0.01)
            
            # plot quiver key
            ax[0].quiver(0.97, 0.18, 2.5, 0, color=polar_color, pivot='tip', transform=ax[0].transAxes,
                        scale=30, headlength=0, headaxislength=0, headwidth=0, width=0.01)
            ax[0].text(0.97, 0.26, r'5$\%$', transform=ax[0].transAxes, fontweight='bold', fontsize=16, color=polar_color, va='top', ha='right')

            # plot beam ellipse for polarization
            beam_major_arcsec, beam_minor_arcsec, beam_pa = beam_plr_info[i, :]
            if i == 0:
                beam = Ellipse((0.15, 0.18), transform=ax[0].transAxes,
                                width=beam_minor_arcsec*distance/crop_sizeau, height=beam_major_arcsec*distance/crop_sizeau,
                                angle=beam_pa, edgecolor=polar_color, facecolor='None', lw=5)
                ax[0].add_patch(beam)
        
        obs_I_bT = b_T(obs_wav[i], beam_info[i, :], obs_I[i])
        # plot observed continuum image
        if i == 0:
            ax[0].imshow(obs_I_bT, origin='lower', cmap=conti_cmap, vmin=0, vmax=Tb_max[i], 
                    extent=[0, obs_PA[i].shape[1], 0, obs_PA[i].shape[0]])
        else:
            ax[0].imshow(obs_I_bT, origin='lower', cmap=conti_cmap, vmin=0, vmax=Tb_max[i])
        # plot colorbar
        cbar = fig.colorbar(ax[0].images[0], ax=ax[0], orientation='horizontal', location='top',
                        pad=0.00, shrink=0.95)
        cbar.set_label(r'$T_b$ (K)', fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        # plot scale bar
        ax[0].plot([0.8-50/crop_sizeau/2, 0.8+50/crop_sizeau/2], [0.05, 0.05], transform=ax[0].transAxes,  # Scale bar in Axes coordinates
                    color=text_color, linewidth=4)
        ax[0].text(0.8, 0.13, '50AU', transform=ax[0].transAxes, fontweight='bold', fontsize=16, color=text_color, va='top', ha='left')

        # plot beam ellipse for stokes I
        beam_major_arcsec, beam_minor_arcsec, beam_pa = beam_info[i, :]

        if i == 2:
            beam = Ellipse((0.20, 0.18), transform=ax[0].transAxes,
                            width=beam_minor_arcsec*distance/crop_sizeau, height=beam_major_arcsec*distance/crop_sizeau,
                            angle=beam_pa, edgecolor=polar_color, facecolor=text_color, lw=5)
        elif i == 3:
            beam = Ellipse((0.15, 0.18), transform=ax[0].transAxes,
                            width=beam_minor_arcsec*distance/crop_sizeau, height=beam_major_arcsec*distance/crop_sizeau,
                            angle=beam_pa, edgecolor=polar_color, facecolor=text_color, lw=5)
        else:
            beam = Ellipse((0.15, 0.18), transform=ax[0].transAxes,
                            width=beam_minor_arcsec*distance/crop_sizeau, height=beam_major_arcsec*distance/crop_sizeau,
                            angle=beam_pa, edgecolor=text_color, facecolor=text_color, lw=5)
        ax[0].add_patch(beam)

        ax[0].text(0.05, 0.95, obs_title[i], fontweight='bold', transform=ax[0].transAxes, fontsize=16, color=text_color, va='top')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
            
        '''
        Plot synthetic continuum and polarization
        '''
        save_image_to_fits(model_img, beam_info[i, :], obs_data_I_coords[i], f'{dir}model_{lam}um', stokes=['I'])
        model_I, header_I, wcs_I = get_info(f'{dir}model_{lam}um_conv_I.fits')
        model_I = model_I[i, :, :]
        model_I_bT = b_T(obs_wav[i], beam_info[i, :], model_I)

        if i != 1:
            save_image_to_fits(model_img, beam_plr_info[i, :], obs_data_plr_coords[i], f'{dir}model_{lam}um', stokes=['Q', 'U'])
            model_Q, header_Q, wcs_Q = get_info(f'{dir}model_{lam}um_conv_Q.fits')
            model_U, header_U, wcs_U = get_info(f'{dir}model_{lam}um_conv_U.fits')
            model_Q = model_Q[i, :, :]
            model_U = model_U[i, :, :]
            model_PA = 0.5 * np.arctan2(-model_U, -model_Q)
            model_Per = np.sqrt(model_Q**2 + model_U**2) / model_I
            mask_model = model_I > [25, None, 25, 5][i] * obs_rms[i]
            model_PA[~mask_model] = np.nan
            model_Per[~mask_model] = np.nan
            x_sampled_model, y_sampled_model, u_segment_model, v_segment_model = b_segment(model_PA, model_Per, step=30)

            
            # plot model polarization segments
            ax[1].quiver(x_sampled_model, y_sampled_model, u_segment_model, v_segment_model, color=polar_color, pivot='middle',
                            scale=30, headlength=0, headaxislength=0, headwidth=0, width=0.01)

            # plot quiver key
            ax[1].quiver(0.97, 0.18, 2.5, 0, color=polar_color, pivot='tip', transform=ax[1].transAxes,
                        scale=30, headlength=0, headaxislength=0, headwidth=0, width=0.01)
            ax[1].text(0.97, 0.26, r'5$\%$', transform=ax[1].transAxes, fontweight='bold', fontsize=16, color=polar_color, va='top', ha='right')

            beam_major_arcsec, beam_minor_arcsec, beam_pa = beam_plr_info[i, :]
            if i == 2:
                beam = Ellipse((0.20, 0.18), transform=ax[1].transAxes,
                                width=beam_minor_arcsec*distance/crop_sizeau, height=beam_major_arcsec*distance/crop_sizeau,
                                angle=beam_pa, edgecolor=polar_color, facecolor=text_color, lw=5)
            elif i == 3:
                beam = Ellipse((0.15, 0.18), transform=ax[1].transAxes,
                                width=beam_minor_arcsec*distance/crop_sizeau, height=beam_major_arcsec*distance/crop_sizeau,
                                angle=beam_pa, edgecolor=polar_color, facecolor=text_color, lw=5)
            else:
                beam = Ellipse((0.15, 0.18), transform=ax[1].transAxes,
                                width=beam_minor_arcsec*distance/crop_sizeau, height=beam_major_arcsec*distance/crop_sizeau,
                                angle=beam_pa, edgecolor=polar_color, facecolor='None', lw=5)
            ax[1].add_patch(beam)

        ax[1].imshow(model_I_bT, origin='lower', cmap=conti_cmap, vmin=0, vmax=Tb_max[i])

        if i == 0 or i == 1:
            # plot beam ellipse
            beam_major_arcsec, beam_minor_arcsec, beam_pa = beam_info[i, :]
            beam = Ellipse((0.15, 0.18), transform=ax[1].transAxes,
                            width=beam_minor_arcsec*distance/crop_sizeau, height=beam_major_arcsec*distance/crop_sizeau,
                            angle=beam_pa, edgecolor=text_color, facecolor=text_color, lw=5)
            ax[1].add_patch(beam)


        # plot scale bar
        ax[1].plot([0.8-50/crop_sizeau/2, 0.8+50/crop_sizeau/2], 
                    [0.05, 0.05], transform=ax[1].transAxes,  # Scale bar in Axes coordinates
                    color=text_color, linewidth=4)
        ax[1].text(0.8, 0.13, '50AU', transform=ax[1].transAxes, fontweight='bold', fontsize=16, color=text_color, va='top', ha='left')

        ax[1].text(0.45, 0.95, 'Fiducial Model', fontweight='bold', transform=ax[1].transAxes, fontsize=16, color=text_color, va='top')
        ax[1].set_xticks([])
        ax[1].set_yticks([])

    plt.savefig(f'{dir}{fname}.pdf', transparent=True)
    plt.close()



"""
Generate synthetic models and plot them
"""

incl = 45
ain = 1.2
amid = 0.8
aout = 0.08
mdot= 4.5e-5
rd = 35
Q = 0.4

setup_model(ain, amid, aout, mdot, rd, Q, align=True)
simulation = Simulation(save_out=True, save_npz=False)
simulate_mutual_parms = {
    "incl"      : incl,
    "npix"      : 500,
    "sizeau"    : crop_sizeau,
    "posang"    : 90,
    "phi"       : 0,
    "dir"       : f'/home/hyp0515/simulation/fiducial_model/',
}
simulation.generate_continuum(
    scat=True,
    stokes=True,
    read_lambda=obs_wav*1e-3,
    load_simulation=False,
    fname=f'conti',
    **simulate_mutual_parms
)

simulation.generate_sed(
    scat=True,
    read_lambda=obs_wav*1e-3,
    load_simulation=False,
    fname='sed',
    **simulate_mutual_parms
)

conti_cmap = 'Oranges'
polar_color = 'dodgerblue'
text_color = 'k'
obs_title = ['ALMA Band 6 (1.3mm)', 'ALMA Band 3 (3mm)', 'JVLA Q Band (7mm)', 'JVLA Ku Band (18mm)']
Tb_max = [300, 450, 150, 300]
plot_plr_diff_params(simulation, obs=(obs_data_I, obs_data_PA, obs_data_Per),
                    dir='./', fname='fig_4')