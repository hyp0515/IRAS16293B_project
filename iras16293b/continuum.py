import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib.colors import SymLogNorm
from scipy import ndimage
import warnings
import sys

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.constants as const
au = const.au.cgs.value
pc = const.pc.cgs.value



stokeI_color = 'inferno'




distance_pc = 140
# crop_sizeau = 250
# ra_min = '16:32:22.652'
# ra_max = '16:32:22.568'
# # dec_min = '-24:28:33.28'
# # dec_max = '-24:28:32.12'

# dec_min = '-24:28:33.18'
# dec_max = '-24:28:32.02'

# coord_min = SkyCoord(ra=ra_min, dec=dec_min, unit=('hourangle','deg'))
# coord_max = SkyCoord(ra=ra_max, dec=dec_max, unit=('hourangle','deg'))




# print(coord_min)
# print(coord_max)


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

def crop(data, wcs):
    """
    Crop the data to the specified coordinates.
    """
    # Convert sky coordinates to pixel coordinates
    min_pixel = skycoord_to_pixel(coord_min, wcs)
    max_pixel = skycoord_to_pixel(coord_max, wcs)

    # Calculate pixel indices
    x_min, y_min = int(min_pixel[0]), int(min_pixel[1])
    x_max, y_max = int(max_pixel[0]), int(max_pixel[1])

    # Crop the data
    cropped_data = data[y_min:y_max, x_min:x_max]
    cropped_wcs = wcs.slice(slice(y_min, y_max), slice(x_min, x_max))
    return cropped_data, cropped_wcs



# iras16293b_obs = {}
# obs_band = ['band6', 'band3', 'qband', 'kuband']
# stokes = ['I', 'Q', 'U', 'Per', 'PA']

# for band in obs_band:
#     iras16293b_obs[band] = {}
#     for stoke in stokes:
#         if band == 'band6' and stoke == 'I':
#         fname = f'fname_{band}_stoke{stoke}'
#         if fname in globals():
#             data, header, wcs = get_info(globals()[fname])
#             beam_axis, beam_pixels, beam_pa, au_per_pix = get_beam_axis(header, distance_pc)
#             cropped_data, cropped_wcs = crop(data, wcs)
#             iras16293b_obs[band][stoke] = {
#                 'data': cropped_data,
#                 'header': header,
#                 'wcs': cropped_wcs,
#                 'beam_axis': beam_axis,
#                 'beam_pixels': beam_pixels,
#                 'beam_pa': beam_pa,
#                 'au_per_pix': au_per_pix
#             }
#         else:
#             print(f"Warning: {fname} not defined.")


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


fname_band6_Per  = '~/project_data/IRAS16293/image_FITS/ALMA_Band6_Pol/c2d_1008_Stokes.Per.regrid.fits'
fname_qband_Per  = '~/project_data/IRAS16293/image_FITS/JVLA_Qband_Pol/iras16293_Qband.rob0.Per.regrid.fits'
fname_kuband_Per = '~/project_data/IRAS16293/image_FITS/JVLA_Kuband_Pol/rob0/I16293_Kuband_A.rob0.Per.image.tt0.miriad.fits'



fname_band6_PA  = '~/project_data/IRAS16293/image_FITS/ALMA_Band6_Pol/c2d_1008_Stokes.PA.regrid.fits'
fname_qband_PA  = '~/project_data/IRAS16293/image_FITS/JVLA_Qband_Pol/iras16293_Qband.rob0.PA.regrid.fits'
fname_kuband_PA = '~/project_data/IRAS16293/image_FITS/JVLA_Kuband_Pol/rob0/I16293_Kuband_A.rob0.PA.image.tt0.miriad.fits'



obs_stokeI_fnames = [
    fname_band6_stokeI_highres,
    fname_band3_stokeI,
    fname_qband_stokeI,
    fname_kuband_stokeI
]
obs_Per_fnames = [
    fname_band6_Per,
    None,
    fname_qband_Per,
    fname_kuband_Per
]
obs_PA_fnames = [
    fname_band6_PA,
    None,
    fname_qband_PA,
    fname_kuband_PA
]

obs_wav = [
    1.3,
    3,
    7, 
    18
]
obs_rms = [
    104e-6,
    17e-6,
    45e-6,
    47e-6
]


# Reference coords (Ku Band)
ra_center = '16:32:22.6126'
dec_center = '-24:28:32.678'
coord_center = SkyCoord(ra=ra_center, dec=dec_center, unit=('hourangle','deg'))
print(coord_center.ra.deg)
print(coord_center.dec.deg)



obs = ['Band6', 'Band3', 'Qband', 'Kuband']
for i, fname in enumerate(obs_stokeI_fnames):
    data, header, wcs = get_info(fname)
    date_obs = header['DATE-OBS']
    t = Time(date_obs, format='isot', scale='utc')
    decimal_year = t.decimalyear
    print(f"Observation: {obs[i]}")
    print(f"Decimal year: {decimal_year:.6f}")



def plot_subimage(fname, wav, rms):
    data, header, wcs = get_info(fname)
    beam_arcsec, beam_pixel, beam_pa, au_per_pix = get_beam_axis(header=header, distance_pc=distance_pc)
    beam_major_arcsec, beam_minor_arcsec = beam_arcsec
    beam_major_pixels, beam_minor_pixels = beam_pixel
    if fname == fname_band6_stokeI_highres:
        data_crop, wcs_crop = crop(data[0, 0, :, :], wcs)
    else:
        data_crop, wcs_crop = crop(data, wcs)

    mask = data_crop < (rms * 5)

    data_bT = 1.36*(wav*1e-1)**2*data_crop*1e3 / (beam_major_arcsec * beam_minor_arcsec)
    
    
    # ax.imshow(data_bT, origin='lower', cmap=stokeI_color, vmin=0, vmax=400)
    if fname == fname_qband_stokeI:
        beam = Ellipse((data_bT.shape[0]*0.18, data_bT.shape[1]*0.15), 
                        width=beam_minor_pixels, height=beam_major_pixels, 
                        angle=beam_pa, edgecolor='white', facecolor='none', lw=1.5)
    else:
        beam = Ellipse((data_bT.shape[0]*0.10, data_bT.shape[1]*0.12), 
                        width=beam_minor_pixels, height=beam_major_pixels, 
                        angle=beam_pa, edgecolor='white', facecolor='none', lw=1.5)
    # ax.add_patch(beam)

    return data_bT, wcs_crop, beam, mask

def plot_subimage_plr(fname_Per, fname_PA):
    data_Per, header, wcs = get_info(fname_Per)
    data_PA, _, _ = get_info(fname_PA)
    beam_arcsec, beam_pixel, beam_pa, au_per_pix = get_beam_axis(header=header, distance_pc=distance_pc)
    beam_major_arcsec, beam_minor_arcsec = beam_arcsec
    beam_major_pixels, beam_minor_pixels = beam_pixel

    if fname_Per == fname_kuband_Per and fname_PA == fname_kuband_PA:
        data_Per_crop, wcs_crop = crop(data_Per[0, :, :], wcs)
        data_PA_crop, _ = crop(data_PA[0, :, :], wcs)
    else:
        data_Per_crop, wcs_crop = crop(data_Per, wcs)
        data_PA_crop, _ = crop(data_PA, wcs)

    beam = Ellipse((data_Per_crop.shape[0]*0.80, data_Per_crop.shape[1]*0.12), 
                    width=beam_minor_pixels, height=beam_major_pixels, 
                    angle=beam_pa, edgecolor='cyan', facecolor='none', lw=1.5)


    return data_Per_crop, data_PA_crop, beam


fig = plt.figure(figsize=(20, 5))

for i, (fname, wav, rms) in enumerate(zip(obs_stokeI_fnames, obs_wav, obs_rms)):
    data_bT, wcs_crop, beam, conti_mask = plot_subimage(fname, wav, rms)
    if fname != fname_band3_stokeI:
        data_Per, data_PA, beam_plr = plot_subimage_plr(
            obs_Per_fnames[i], obs_PA_fnames[i])
        
        y, x = np.mgrid[0:data_Per.shape[0], 0:data_Per.shape[1]]
        step = 5
        x_ds = x[::step, ::step]
        y_ds = y[::step, ::step]
        angle_ds = np.deg2rad(data_PA[::step, ::step]) + np.pi / 2  
        intensity_ds = data_Per[::step, ::step]

        # E-vector components
        u = (intensity_ds/0.02) * np.cos(angle_ds)
        v = (intensity_ds/0.02) * np.sin(angle_ds)
    

    ax = plt.subplot(1, 4, i+1)
    im = ax.imshow(data_bT, origin='lower', cmap=stokeI_color, vmin=0, vmax=300)
    
    
    if fname != fname_band3_stokeI:
        im = ax.imshow(data_bT, origin='lower', cmap=stokeI_color, vmin=0, vmax=300, extent=[0, data_Per.shape[1], 0, data_Per.shape[0]])
        ax.quiver(x_ds, y_ds, u, v, color='cyan', scale=30, headlength=0, headwidth=0, headaxislength=0)
        ax.add_patch(beam_plr)

    ax.add_patch(beam)
    ax.set_yticks([])
    ax.set_xticks([])
    
cbar_ax = fig.add_axes([0.92, 0.05, 0.01, 0.9])
fig.colorbar(im, cax=cbar_ax, label=r'$T_b$ [K]')
plt.tight_layout(rect=[0, 0, 0.92, 1])



