import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import ndimage
import warnings
import sys
sys.path.append('..')
from iras16293b.data_list import data_dict

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
import astropy.constants as const
au = const.au.cgs.value
pc = const.pc.cgs.value

stokeI_color = 'hot'


distance_pc = 140
crop_sizeau = 250
ra_min = '16:32:22.652'
ra_max = '16:32:22.568'
# dec_min = '-24:28:33.28'
# dec_max = '-24:28:32.12'

dec_min = '-24:28:33.18'
dec_max = '-24:28:32.02'

coord_min = SkyCoord(ra=ra_min, dec=dec_min, unit=('hourangle','deg'))
coord_max = SkyCoord(ra=ra_max, dec=dec_max, unit=('hourangle','deg'))


def get_info(fname):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with fits.open(fname) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            wcs = WCS(header=header)
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

fname_band6_stokeI  = '/home/hyp0515/project_data/IRAS16293/image_FITS/ALMA_Band6_Pol/c2d_1008_Stokes.I.regrid.fits'
fname_band6_stokeI_highres = '/home/hyp0515/project_data/IRAS16293/image_FITS/ALMA_Band6_Pol/iras16293B_cont_b6_foralpha_pbcor.fits'
fname_band3_stokeI  = '/home/hyp0515/project_data/IRAS16293/image_FITS/ALMA_Band3_StokesI/sourceB.image.dropdreg.fits'
fname_qband_stokeI  = '/home/hyp0515/project_data/IRAS16293/image_FITS/JVLA_Qband_Pol/iras16293_Qband.rob0.I.regrid.fits'
fname_kuband_stokeI = '/home/hyp0515/project_data/IRAS16293/image_FITS/JVLA_Kuband_Pol/rob0/I16293_Kuband_A.rob0.I.image.tt0.pbcor.subim.fits'



data_band6, header_band6, wcs_band6 = get_info(fname_band6_stokeI_highres)
beam_arcsec, beam_pixel, beam_pa, au_per_pix = get_beam_axis(header=header_band6, distance_pc=distance_pc)
beam_major_arcsec, beam_minor_arcsec = beam_arcsec
beam_major_pixels, beam_minor_pixels = beam_pixel
data_band6_crop, wcs_band6_crop = crop(data_band6[0, 0, :, :], wcs_band6)
data_band6_crop_bT = 1.36*(0.13)**2*data_band6_crop*1e3 / (beam_major_arcsec * beam_minor_arcsec)



data_band3, header_band3, wcs_band3 = get_info(fname_band3_stokeI)
beam_arcsec, beam_pixel, beam_pa, au_per_pix = get_beam_axis(header=header_band3, distance_pc=distance_pc)
beam_major_arcsec, beam_minor_arcsec = beam_arcsec
beam_major_pixels, beam_minor_pixels = beam_pixel
data_band3_crop, wcs_band3_crop = crop(data_band3, wcs_band3)
data_band3_crop_bT = 1.36*(0.3)**2*data_band3_crop*1e3 / (beam_major_arcsec * beam_minor_arcsec)

data_qband, header_qband, wcs_qband = get_info(fname_qband_stokeI)
beam_arcsec, beam_pixel, beam_pa, au_per_pix = get_beam_axis(header=header_qband, distance_pc=distance_pc)
beam_major_arcsec, beam_minor_arcsec = beam_arcsec
beam_major_pixels, beam_minor_pixels = beam_pixel
data_qband_crop, wcs_qband_crop = crop(data_qband, wcs_qband)
data_qband_crop_bT = 1.36*(0.7)**2*data_qband_crop*1e3 / (beam_major_arcsec * beam_minor_arcsec)

data_kuband, header_kuband, wcs_kuband = get_info(fname_kuband_stokeI)
beam_arcsec, beam_pixel, beam_pa, au_per_pix = get_beam_axis(header=header_kuband, distance_pc=distance_pc)
beam_major_arcsec, beam_minor_arcsec = beam_arcsec
beam_major_pixels, beam_minor_pixels = beam_pixel
data_kuband_crop, wcs_kuband_crop = crop(data_kuband, wcs_kuband)
data_kuband_crop_bT = 1.36*(1.8)**2*data_kuband_crop*1e3 / (beam_major_arcsec * beam_minor_arcsec)


fig, ax = plt.subplots(3, 4, figsize=(20, 10))
fig.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1, wspace=0.0, hspace=0.0)

obs = ax[0, 0].imshow(data_band6_crop_bT, cmap=stokeI_color, origin='lower')
cbar_obs = fig.colorbar(obs, ax=ax[0, 0], pad=0.00, aspect=30, shrink=.98, location='left')
cbar_obs.set_label('Brightness Temperature (K)')
ax[0, 1].imshow(data_band3_crop_bT, cmap=stokeI_color, origin='lower')
ax[0, 2].imshow(data_qband_crop_bT, cmap=stokeI_color, origin='lower')
ax[0, 3].imshow(data_kuband_crop_bT, cmap=stokeI_color, origin='lower')

model = ax[1, 0].imshow(data_band6_crop_bT, cmap=stokeI_color, origin='lower')
cbar_model = fig.colorbar(model, ax=ax[1, 0], pad=0.00, aspect=30, shrink=.98, location='left')
cbar_model.set_label('Brightness Temperature (K)')
ax[1, 1].imshow(data_band3_crop_bT, cmap=stokeI_color, origin='lower')
ax[1, 2].imshow(data_qband_crop_bT, cmap=stokeI_color, origin='lower')
ax[1, 3].imshow(data_kuband_crop_bT, cmap=stokeI_color, origin='lower')


residual = ax[2, 0].imshow(data_band6_crop_bT, cmap=stokeI_color, origin='lower')
cbar_residual = fig.colorbar(residual, ax=ax[2, 0], pad=0.00, aspect=30, shrink=.98, location='left')
cbar_residual.set_label('Brightness Temperature (K)')
ax[2, 1].imshow(data_band3_crop_bT, cmap=stokeI_color, origin='lower')
ax[2, 2].imshow(data_qband_crop_bT, cmap=stokeI_color, origin='lower')
ax[2, 3].imshow(data_kuband_crop_bT, cmap=stokeI_color, origin='lower')

plt.show()


'''
Template for plotting residuals of ALMA and VLA observations
(similar to the Fig. 1 in Xu et al. 2023)
'''
# import matplotlib.pyplot as plt
# import numpy as np
# from astropy.io import fits
# from astropy.visualization import AsinhStretch, ImageNormalize
# from astropy.wcs import WCS
# from matplotlib.colors import SymLogNorm

# # ====== Configuration ======
# bands = ["VLA Ka (8.7 mm)", "VLA Q (6.8 mm)", "ALMA Band 6 (1.3 mm)", "ALMA Band 7 (0.9 mm)"]
# rms_vals = [0.32, 0.35, 0.014, 0.034]  # in Kelvin
# beam_texts = ["10μJy/beam", "13μJy/beam", "17μJy/beam", "143μJy/beam"]
# beam_positions = [(0.9, 0.1)] * 4  # location of beam ellipses (x, y) in normalized axes coords
# scale_bar_au = 100

# # Assume 3 arrays per band: [obs, model, residual]
# # You can replace with file paths or FITS reader
# obs_data_list = [np.load(f"obs_band{i}.npy") for i in range(4)]
# model_data_list = [np.load(f"model_band{i}.npy") for i in range(4)]
# resid_data_list = [np.load(f"resid_band{i}.npy") for i in range(4)]

# # For plotting
# vmin = 0.1
# vmax = 100  # adjust based on data range
# resid_norm = SymLogNorm(linthresh=0.1, linscale=0.5, vmin=-10, vmax=10)

# fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
# stretch = AsinhStretch()
# norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch)

# # ====== Plotting Loop ======
# for i in range(4):
#     # Top row: Observation
#     ax = axes[0, i]
#     im = ax.imshow(obs_data_list[i], origin='lower', norm=norm, cmap='plasma')
#     ax.set_title(bands[i], fontsize=12)
#     ax.text(0.05, 0.90, rf"$\sigma_{{\rm rms}} = {rms_vals[i]:.3f}$K\n({beam_texts[i]})",
#             transform=ax.transAxes, fontsize=10, color='white')

#     # Middle row: Model
#     ax = axes[1, i]
#     ax.imshow(model_data_list[i], origin='lower', norm=norm, cmap='plasma')

#     # Bottom row: Residual
#     ax = axes[2, i]
#     im_res = ax.imshow(resid_data_list[i], origin='lower', norm=resid_norm, cmap='RdBu_r')

#     # Add scale bar (simplified version, replace with real WCS if needed)
#     ax.plot([10, 10 + scale_bar_au], [10, 10], 'w-', lw=3)
#     ax.text(10, 15, f"{scale_bar_au} au", color='white', fontsize=10)

#     # Optional: Add beams (ellipses)
#     ax_beam = fig.add_axes([ax.get_position().x1 - 0.05, ax.get_position().y0, 0.02, 0.02])
#     ax_beam.set_facecolor('black')
#     beam = plt.Circle((0.5, 0.5), 0.45, color='white')
#     ax_beam.add_patch(beam)
#     ax_beam.axis('off')

# # ====== Shared Labels ======
# fig.text(0.07, 0.8, "Observation $T_b$ [K]", va='center', rotation='vertical', fontsize=13)
# fig.text(0.07, 0.52, "Model $T_b$ [K]", va='center', rotation='vertical', fontsize=13)
# fig.text(0.07, 0.25, "Residual $T_b$ [K]", va='center', rotation='vertical', fontsize=13)

# # ====== Colorbars ======
# cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
# fig.colorbar(im, cax=cbar_ax, label=r'$T_b$ [K]')

# cbar_resid = fig.add_axes([0.92, 0.18, 0.015, 0.25])
# fig.colorbar(im_res, cax=cbar_resid, label=r'Residual $T_b$ [K]')

# plt.tight_layout(rect=[0, 0, 0.9, 1])
# plt.show()