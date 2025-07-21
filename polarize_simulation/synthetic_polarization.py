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
from radmc.setup import radmc3d_setup
from radmc.simulate import generate_simulation
from sed.plot_sed import HG_19_data, this_work_data

"""
Define functions to analyze the synthetic and observed polarization data.
"""
def setup_model(amax, mdot, rd, Toomre_Q):
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
    beam_area_pixels = np.pi * beam[0] * beam[0] / (4 * np.log(2))
    image_copy = image_array.copy()
    mask = image_array > 5 * rms
    image_copy[~mask] = np.nan
    total_flux = np.nansum(image_copy) / beam_area_pixels
    uncertainty = rms * mask.sum() / beam_area_pixels
    return total_flux, uncertainty

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

"""
Initialize observation (continuum)
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
beam_info = np.zeros((len(obs_wav), 3)) # beam major, beam minor, beam position angle

fname_band6_stokeI  = '~/project_data/IRAS16293/image_FITS/ALMA_Band6_Pol/c2d_1008_Stokes.I.regrid.fits'
fname_band6_stokeI_highres = '~/project_data/IRAS16293/image_FITS/ALMA_Band6_Pol/iras16293B_cont_b6_foralpha_pbcor.fits'
fname_band3_stokeI  = '~/project_data/IRAS16293/image_FITS/ALMA_Band3_StokesI/sourceB.image.dropdreg.fits'
fname_qband_stokeI  = '~/project_data/IRAS16293/image_FITS/JVLA_Qband_Pol/iras16293_Qband.rob0.I.regrid.fits'
fname_kuband_stokeI = '~/project_data/IRAS16293/image_FITS/JVLA_Kuband_Pol/rob0/I16293_Kuband_A.rob0.I.image.tt0.pbcor.subim.fits'

crop_sizeau = 200

# # Reference coords (Ku Band)
# ra_ref = '16:32:22.6126'
# dec_ref = '-24:28:32.678'
# data, header, wcs = get_info(fname_kuband_stokeI)
# date_obs = header['DATE-OBS']
# t = Time(date_obs, format='isot', scale='utc')
# Reference coords (Q Band)
ra_ref = '16:32:22.6150'
dec_ref = '-24:28:32.547'
data, header, wcs = get_info(fname_qband_stokeI)
date_obs = header['DATE-OBS']
t = Time(date_obs, format='isot', scale='utc')

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

obs_data = []
for i, fname in enumerate(obs_stokeI_fnames):
    data, header, wcs = get_info(fname)
    beam_arcsec, beam_pixel, beam_pa, au_per_pix = get_beam_axis(header=header, distance_pc=distance)
    beam_major_arcsec, beam_minor_arcsec = beam_arcsec
    beam_major_pixels, beam_minor_pixels = beam_pixel
    beam_info[i, :] = [beam_major_arcsec, beam_minor_arcsec, beam_pa]

    date_obs = header['DATE-OBS']
    t = Time(date_obs, format='isot', scale='utc')
    coords_shifted = coord_ref.apply_space_motion(t)

    if fname == fname_band6_stokeI_highres:
        data_crop, wcs_crop = crop_around_coord(data[0, 0, :, :], wcs, coords_shifted, crop_sizeau, au_per_pix)
    else:
        data_crop, wcs_crop = crop_around_coord(data, wcs, coords_shifted, crop_sizeau, au_per_pix)
    obs_data.append(data_crop)

def plot_conti_column(ax, obs, model, lam, sizeau):


    rotated_obs = rotate_image(obs, -90) # Rotate the image to match the model orientation (pa=0)
    radial_obs_major, radial_obs_minor, center = radial_intensity(rotated_obs, center=None, width=5)

    ax[0].imshow(rotated_obs, origin='lower', cmap="magma", vmin=0, vmax=np.nanmax(rotated_obs))
    ax[0].text(0.05, 0.95, f'{lam*1e-3:.1f} mm', transform=ax[0].transAxes, fontsize=12, color='white', va='top')
    ax[0].hlines(center, xmin=0, xmax=rotated_obs.shape[0]-1, color='violet', linestyle='--', linewidth=1)
    ax[0].vlines(center, ymin=0, ymax=rotated_obs.shape[1]-1, color='dodgerblue', linestyle='--', linewidth=1)
    ax[0].set_xticks([0, rotated_obs.shape[0]//2, rotated_obs.shape[0]-1])
    ax[0].set_xticklabels([-(sizeau//2), 0, (sizeau//2)])
    ax[0].set_xlabel('AU')
    ax[0].set_yticks([0, rotated_obs.shape[1]//2, rotated_obs.shape[1]-1])
    ax[0].set_yticklabels([-(sizeau//2), 0, (sizeau//2)])
    ax[0].set_ylabel('AU')

    radial_model_major, radial_model_minor, center = radial_intensity(model, center=None, width=5)
    
    ax[1].imshow(model, origin='lower', cmap="magma", vmin=0, vmax=np.nanmax(rotated_obs))
    ax[1].hlines(center, xmin=0, xmax=model.shape[0]-1, color='violet', linestyle='-', linewidth=1)
    ax[1].vlines(center, ymin=0, ymax=model.shape[1]-1, color='dodgerblue', linestyle='-', linewidth=1)
    ax[1].set_xticks([0, model.shape[0]//2, model.shape[0]-1])
    ax[1].set_xticklabels([-(sizeau//2), 0, (sizeau//2)])
    ax[1].set_xlabel('AU')
    ax[1].set_yticks([0, model.shape[1]//2, model.shape[1]-1])
    ax[1].set_yticklabels([-(sizeau//2), 0, (sizeau//2)])
    ax[1].set_ylabel('AU')

    interp_obs = ndimage.zoom(rotated_obs, zoom=model.shape[0] / rotated_obs.shape[0])
    radial_obs_major, radial_obs_minor, center = radial_intensity(interp_obs, center=None, width=5)

    residual = interp_obs - model

    ax[2].imshow(residual, origin='lower', cmap="seismic", vmin=-(np.nanmax(rotated_obs)/2), vmax=np.nanmax(rotated_obs)/2)
    ax[2].set_xticks([0, residual.shape[0]//2, residual.shape[0]-1])
    ax[2].set_xticklabels([-(sizeau//2), 0, (sizeau//2)])
    ax[2].set_xlabel('AU')
    ax[2].set_yticks([0, residual.shape[1]//2, residual.shape[1]-1])
    ax[2].set_yticklabels([-(sizeau//2), 0, (sizeau//2)])
    ax[2].set_ylabel('AU')



    ax[3].plot(np.linspace(-100, 100, num=radial_obs_major.size, endpoint=True), radial_obs_major, color='violet', linestyle='--')
    ax[3].plot(np.linspace(-100, 100, num=radial_obs_minor.size, endpoint=True), radial_obs_minor, color='dodgerblue', linestyle='--')
    ax[3].plot(np.linspace(-100, 100, num=radial_model_major.size, endpoint=True), radial_model_major, color='violet')
    ax[3].plot(np.linspace(-100, 100, num=radial_model_minor.size, endpoint=True), radial_model_minor, color='dodgerblue')
    ax[3].text(0.05, 0.95, f'{lam*1e-3:.1f} mm', transform=ax[3].transAxes, fontsize=12, color='k', va='top')
    # ax[3].set_title(f'Radial profile at {lam} $\mu$m')
    ax[3].set_xlabel('AU')
    ax[3].set_ylabel('Intensity (Jy/pixel)')
    ax[3].set_xlim((-(sizeau//2), sizeau//2))
    ax[3].set_ylim(bottom=0)

def plot_conti_diff_params(simulation, obs, dir, fname):

    model_img = simulation.conti
    conv_image = model_img.imConv(dpc=distance, fwhm=beam_info[i, :-1], pa=-(beam_info[i, -1]+90)) # Convolve with beam
    sizeau = model_img.x/au
    npix = len(model_img.x)
    pixel_area = (sizeau/npix/140)**2
    beam_area = beam_info[i, 0]*beam_info[i, 0]*np.pi/(4*np.log(2))
    conv_image.imageJyppix *= beam_area/pixel_area/(distance**2) # Convert to Jy/pixel

    fig, axes = plt.subplots(4, len(obs_wav), figsize=(4*len(obs_wav), 4*len(obs_wav)))

    for i, lam in enumerate(obs_wav):
        model = conv_image.imageJyppix[:, :, 0, i].T
        obs = obs_data[i]

        rotated_obs = rotate_image(obs, -90) # Rotate the image to match the model orientation (pa=0)
        radial_obs_major, radial_obs_minor, center = radial_intensity(rotated_obs, center=None, width=5)
        ax = axes[:, i]
        ax[0].imshow(rotated_obs, origin='lower', cmap="magma", vmin=0, vmax=np.nanmax(rotated_obs))
        ax[0].text(0.05, 0.95, f'{lam*1e-3:.1f} mm', transform=ax[0].transAxes, fontsize=12, color='white', va='top')
        ax[0].hlines(center, xmin=0, xmax=rotated_obs.shape[0]-1, color='violet', linestyle='--', linewidth=1)
        ax[0].vlines(center, ymin=0, ymax=rotated_obs.shape[1]-1, color='dodgerblue', linestyle='--', linewidth=1)
        ax[0].set_xticks([0, rotated_obs.shape[0]//2, rotated_obs.shape[0]-1])
        ax[0].set_xticklabels([-(sizeau//2), 0, (sizeau//2)])
        ax[0].set_xlabel('AU')
        ax[0].set_yticks([0, rotated_obs.shape[1]//2, rotated_obs.shape[1]-1])
        ax[0].set_yticklabels([-(sizeau//2), 0, (sizeau//2)])
        ax[0].set_ylabel('AU')

        radial_model_major, radial_model_minor, center = radial_intensity(model, center=None, width=5)
        
        ax[1].imshow(model, origin='lower', cmap="magma", vmin=0, vmax=np.nanmax(rotated_obs))
        ax[1].hlines(center, xmin=0, xmax=model.shape[0]-1, color='violet', linestyle='-', linewidth=1)
        ax[1].vlines(center, ymin=0, ymax=model.shape[1]-1, color='dodgerblue', linestyle='-', linewidth=1)
        ax[1].set_xticks([0, model.shape[0]//2, model.shape[0]-1])
        ax[1].set_xticklabels([-(sizeau//2), 0, (sizeau//2)])
        ax[1].set_xlabel('AU')
        ax[1].set_yticks([0, model.shape[1]//2, model.shape[1]-1])
        ax[1].set_yticklabels([-(sizeau//2), 0, (sizeau//2)])
        ax[1].set_ylabel('AU')

        interp_obs = ndimage.zoom(rotated_obs, zoom=model.shape[0] / rotated_obs.shape[0])
        radial_obs_major, radial_obs_minor, center = radial_intensity(interp_obs, center=None, width=5)

        residual = interp_obs - model

        ax[2].imshow(residual, origin='lower', cmap="seismic", vmin=-(np.nanmax(rotated_obs)/2), vmax=np.nanmax(rotated_obs)/2)
        ax[2].set_xticks([0, residual.shape[0]//2, residual.shape[0]-1])
        ax[2].set_xticklabels([-(sizeau//2), 0, (sizeau//2)])
        ax[2].set_xlabel('AU')
        ax[2].set_yticks([0, residual.shape[1]//2, residual.shape[1]-1])
        ax[2].set_yticklabels([-(sizeau//2), 0, (sizeau//2)])
        ax[2].set_ylabel('AU')



        ax[3].plot(np.linspace(-100, 100, num=radial_obs_major.size, endpoint=True), radial_obs_major, color='violet', linestyle='--')
        ax[3].plot(np.linspace(-100, 100, num=radial_obs_minor.size, endpoint=True), radial_obs_minor, color='dodgerblue', linestyle='--')
        ax[3].plot(np.linspace(-100, 100, num=radial_model_major.size, endpoint=True), radial_model_major, color='violet')
        ax[3].plot(np.linspace(-100, 100, num=radial_model_minor.size, endpoint=True), radial_model_minor, color='dodgerblue')
        ax[3].text(0.05, 0.95, f'{lam*1e-3:.1f} mm', transform=ax[3].transAxes, fontsize=12, color='k', va='top')
        # ax[3].set_title(f'Radial profile at {lam} $\mu$m')
        ax[3].set_xlabel('AU')
        ax[3].set_ylabel('Intensity (Jy/pixel)')
        ax[3].set_xlim((-(sizeau//2), sizeau//2))
        ax[3].set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f'{dir}{fname}.pdf', transparent=True)
    plt.close()

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
    plt.savefig(f'{dir}{fname}.pdf', transparent=True)
    plt.close()
    

"""
Generate synthetic models and plot them
"""

amax_list = [ 1e1,  1e0, 7e-1, 5e-1, 3e-1, 1e-1, 1e-2] # maximum grain sizes in mm
Mdot_list = [1e-5, 1e-6, 1e-7, 1e-8] # accretion rates
Q_list    = [ 1.5,    1,  0.5,  0.75, 0.3] # Toomre Q parameters
for i, amax in enumerate(amax_list):
    for j, mdot in enumerate(Mdot_list):
        for k, Q in enumerate(Q_list):
            amax        = amax # maximum grain size in mm
            mstar       = 0.5 # stellar mass in solar masses
            mdot        = mdot # accretion rate in solar masses per year
            rd          = 40 # disk radius in AU
            Toomre_Q    = Q # Toomre Q parameter
            l_star      = .1 # stellar luminosity in solar luminosities
            heating     = 'accretion' # heating mechanism

            setup_model(amax, mdot, rd, Toomre_Q)
            # write_log()
            
            simulation = generate_simulation(save_out=True, save_npz=True)
            simulate_mutual_parms = {
                "incl"      : 50,
                "npix"      : 500,
                "sizeau"    : 200,
                "posang"    : 0,
                "phi"       : 0,
                "dir"       : f'./simulation/amax_{amax}_Mdot_{mdot}_Q_{Toomre_Q}/',
            }
            simulation.generate_sed(
                scat=True,
                read_lambda=obs_wav*1e-3,
                load_simulation=True,
                fname='sed',
                **simulate_mutual_parms
            )
            simulation.generate_continuum(
                scat=True,
                wav=lam,
                stokes=True,
                read_lambda=obs_wav*1e-3,
                load_simulation=True,
                fname=f'conti',
                **simulate_mutual_parms
            )

            plot_sed_diff_params(simulation, dir=simulate_mutual_parms["dir"], fname='sed')
            plot_conti_diff_params(simulation, obs_data, dir=simulate_mutual_parms["dir"], fname='conti')

            fig, ax = plt.subplots(4, len(obs_wav), figsize=(4*len(obs_wav), 4*len(obs_wav)))

            for i, lam in enumerate(obs_wav):
                simulation.generate_continuum(
                    scat=True,
                    wav=lam,
                    stokes=True,
                    load_simulation=True,
                    fname=f'wav_{lam}',
                    **simulate_mutual_parms
                )
                
                model_img = simulation.conti
                conv_image = model_img.imConv(dpc=distance, fwhm=beam_info[i, :-1], pa=-(beam_info[i, -1]+90)) # Convolve with beam
                sizeau = model_img.x/au
                npix = len(model_img.x)
                pixel_area = (sizeau/npix/140)**2
                beam_area = beam_info[i, 0]*beam_info[i, 0]*np.pi/(4*np.log(2))
                conv_image.imageJyppix *= beam_area/pixel_area/(distance**2) # Convert to Jy/pixel

                plot_conti_column(ax[:, i], obs_data[i], conv_image.imageJyppix[:, :, 0, 0].T, lam, sizeau)
            plt.tight_layout()
            plt.savefig(f'{simulate_mutual_parms["dir"]}conti.pdf', transparent=True)
            plt.close()


            
             

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
