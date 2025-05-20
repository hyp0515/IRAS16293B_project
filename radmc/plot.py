import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from scipy.ndimage import gaussian_filter
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from radmc3dPy.data import *


class npz_to_class:
    def __init__(self, npz_data):
        for key in npz_data.keys():
            setattr(self, key, npz_data[key])


class generate_plot():
    
    def __init__(self, parms,
                 profile   = True,
                 channel   = True,
                 continuum = False,
                 pv        = True,
                 sed       = False,
                 spectrum  = False):
        
        if profile is True:
            self.plot_profile(parms.profile_parms)
        if channel is True:
            self.plot_channel_map(parms.channel_parms)
        if pv is True:
            self.plot_pv(parms.pv_parms)
        if continuum is True:
            self.plot_continuum(parms.continuum_parms)
        if sed is True:
            self.plot_sed(parms.sed_parms)
        if spectrum is True:
            self.plot_spectrum(parms.spectra_parms)

        pass

    def plot_profile(self, parms):
        
        read_data = readData(dtemp=True, ddens=True, gdens=True, ispec='ch3oh')
        grid = readGrid(wgrid=False)
        nch3oh    = read_data.ndens_mol
        dust      = read_data.rhodust
        t         = read_data.dusttemp
        R, Theta, Phi =  grid.x/au, grid.y, grid.z
        
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6),
                               subplot_kw={'projection': 'polar'})
        fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.05)
        cmaps = ['BuPu', 'OrRd', 'BuPu']
        titles = [r'$\rho_{dust}$', r'$T$', r'$n_{\mathregular{CH_3OH}}$']
        cbar = [r'log($\rho$) [g$cm^{-3}$]', r'log(T) [K]',r'log($n_{\mathregular{CH_3OH}}$) [$cm^{-3}$]']
        
        for idx_val, val in enumerate([np.sum(dust[:, :, 0, :], axis=2), t[:, :, 0, 1], nch3oh[:, :, 0, 0]]):
            c = ax[idx_val].pcolormesh(Theta-np.pi/2, R, np.log10(val), shading='auto', cmap=cmaps[idx_val])
            ax[idx_val].pcolormesh(Theta+np.pi/2, R, np.log10(val), shading='auto', cmap=cmaps[idx_val])
            if idx_val == 0:
                den = val
                levels = np.linspace(np.log10(den).min(), np.log10(den).max(), 3)
            ax[idx_val].contour(Theta-np.pi/2, R, np.log10(den), levels=levels, colors='k', linewidths=.7, linestyles='dashed')
            ax[idx_val].contour(Theta+np.pi/2, R, np.log10(den), levels=levels, colors='k', linewidths=.7, linestyles='dashed')
            ax[idx_val].set_xticks([])
            ax[idx_val].set_yticks([])
            fig.colorbar(c, ax=ax[idx_val], orientation='vertical', shrink=0.7).set_label(cbar[idx_val], fontsize=18)
            ax[idx_val].set_title(titles[idx_val], fontsize=26, color='k')
        
        
        scale_bar_ax = fig.add_axes([0.48, 0.12, 0.09, 0.02]) # [left, bottom, width, height]
        scale_bar = AnchoredSizeBar(scale_bar_ax.transData,
                                    1,  # Size of the scale bar in data coordinates
                                    f'{round(R[-1])} AU',  # Label for the scale bar
                                    'lower center',  # Location
                                    pad=0.1,
                                    color='black',
                                    frameon=False,
                                    size_vertical=0.01,
                                    fontproperties=fm.FontProperties(size=12))
        
        scale_bar_ax.add_artist(scale_bar)
        scale_bar_ax.set_axis_off()
        
        scale_bar_ax = fig.add_axes([0.15, 0.12, 0.09, 0.02]) # [left, bottom, width, height]
        scale_bar = AnchoredSizeBar(scale_bar_ax.transData,
                                    1,  # Size of the scale bar in data coordinates
                                    f'{round(R[-1])} AU',  # Label for the scale bar
                                    'lower center',  # Location
                                    pad=0.1,
                                    color='black',
                                    frameon=False,
                                    size_vertical=0.01,
                                    fontproperties=fm.FontProperties(size=12))
        
        scale_bar_ax.add_artist(scale_bar)
        scale_bar_ax.set_axis_off()
        
        scale_bar_ax = fig.add_axes([0.8, 0.12, 0.09, 0.02]) # [left, bottom, width, height]
        scale_bar = AnchoredSizeBar(scale_bar_ax.transData,
                                    1,  # Size of the scale bar in data coordinates
                                    f'{round(R[-1])} AU',  # Label for the scale bar
                                    'lower center',  # Location
                                    pad=0.1,
                                    color='black',
                                    frameon=False,
                                    size_vertical=0.01,
                                    fontproperties=fm.FontProperties(size=12))
        
        scale_bar_ax.add_artist(scale_bar)
        scale_bar_ax.set_axis_off()
        
        self.save_plot(fig=fig, parms=parms)

    def plot_channel_map(self, parms):
        
        if os.path.isdir(parms.cube_dir):
            try:
                if os.path.isfile(parms.cube_dir+parms.cube_fname):
                    if parms.cube_fname.endswith(".out"):
                        cube = readImage(parms.cube_dir+parms.cube_fname)
                        image = cube.imageJyppix
                    elif parms.cube_fname.endswith(".npz"):
                        raw_cube = np.load(parms.cube_dir+parms.cube_fname)
                        cube = npz_to_class(raw_cube)
                        image = cube.imageJyppix
                    else:
                        print('No correct cube file is given')
                        pass
                else:
                    print('No cube file is found')
                    pass 
            except:
                if isinstance(parms.cube_fname, list):
                    for fname in parms.cube_fname:
                        if fname.endswith("_scat.out"):
                            cube = readImage(parms.cube_dir+fname)
                        elif fname.endswith("_scat.npz"):
                            raw_cube_gas = np.load(parms.cube_dir+fname)
                            cube = npz_to_class(raw_cube_gas)
                        elif fname.endswith("_conti.out"):
                            cube_conti = readImage(parms.cube_dir+fname)
                        elif fname.endswith("_conti.npz"):
                            raw_cube_dust = np.load(parms.cube_dir+fname)
                            cube_conti = npz_to_class(raw_cube_dust)
                        else:
                            print("No correct cube file is given")
                            break
                    
                    image = cube.imageJyppix - np.tile(cube_conti.imageJyppix[:, :, np.newaxis], (1, 1, cube.nwav))
                else:
                    print('No correct cube file is given')
                    pass
        else:
            print('No correct cube directory is given')
            pass
        
        nlam = len(cube.wav)
        # if nlam > 20:
        #     print('Too much wavelength!!')
        #     return
        sizeau = int(round((cube.x/au)[-1]))*2
        npix = cube.nx
        if nlam%2 == 0:
            freq0 = (cube.freq[nlam//2] + cube.freq[(nlam//2)-1])/2
        else:
            freq0 = cube.freq[nlam//2]
        v = cc / 1e5 * (freq0 - cube.freq) / freq0
        vkms = parms.vkms
        
        image = image * 1e3/(parms.d**2) # mJy
        title = getattr(parms, 'title', '')
        fwhm  = getattr(parms,  'fwhm', 50)
        
        
        def convolve(image, fwhm):
            convolved_image = np.zeros(shape=image.shape)
            for i in range(nlam):
                sigma = fwhm * (npix/sizeau)/ (2*np.sqrt(2*np.log(2)))
                convolved_image[:, :, i] = gaussian_filter(image[:, :, i], sigma=sigma)
            return convolved_image*(np.pi/4*((fwhm*npix/sizeau)**2))
        
        def channel(image, conti=None, negative=True,
                    colormap='hot', neg_colormap='viridis_r',
                    text_color='w',convolved=False):
            
            fig, ax = plt.subplots(2, (nlam//2)+1, figsize=(3*((nlam//2)+1), 6), sharex=True, sharey=True,
                                   layout="constrained", gridspec_kw={'wspace': 0.0, 'hspace': 0.1})
            
            
            if convolved is True:
                fig.suptitle(title+' (convolved)', fontsize = 16)
            else:
                fig.suptitle(title, fontsize = 16)
            
            axes = ax.flatten()
            
            if negative is True:
                vmin = 0
            else:
                vmin = np.min(image)
            vmax = np.max(image)
            # vmin = 0
            # vmax = 10
            if conti is not None:
                x, y = np.linspace(0, npix, npix), np.linspace(0, npix, npix)
                X, Y = np.meshgrid(x, y)
                contour_level = np.linspace(0, np.max(conti), 5)
                
            if negative is True:
                negative_image = np.where(image<0, image, 0)
                
            extent = [0, npix, 0, npix]
            
            for idx in range(nlam+1):
                
                if idx <= nlam//2:
                    c = axes[idx].imshow(image[:, :, idx].T, cmap=colormap,
                                        vmin=vmin, vmax=vmax, extent=extent)
                    if conti is not None:
                        axes[idx].contour(Y, X, conti[:, ::-1, idx],
                                          levels=contour_level, colors='w', linewidths=1)  
                    if negative is True:
                        axes[idx].imshow(negative_image[:, :, idx].T, cmap=neg_colormap, alpha=0.5)
                        
                        
                    axes[idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]+vkms:.1f} $km/s$',
                                   ha='right', va='top', color=text_color, fontsize=16)
                    if idx == 0:
                        axes[idx].set_yticks([int(npix*0.1), int(npix*0.3), npix//2, int(npix*0.7), int(npix*0.9)])
                        axes[idx].set_yticklabels([f'{int((sizeau//2)*0.8)}',
                                                   f'{int((sizeau//2)*0.4)}',
                                                   '0',
                                                   f'-{int((sizeau//2)*0.4)}',
                                                   f'-{int((sizeau//2)*0.8)}'],
                                                  fontsize=14)
                        axes[idx].set_ylabel('AU',fontsize=16)
                elif idx > nlam//2:
                    axes[idx].imshow(image[:, :, -int(idx-(nlam//2))].T, cmap=colormap,
                                     vmin=vmin, vmax=vmax, extent=extent)
                    if conti is not None:
                        axes[idx].contour(Y, X, conti[:, ::-1, -int(idx-(nlam//2))],
                                          levels=contour_level, colors='w', linewidths=1)
                    if negative is True:
                        axes[idx].imshow(negative_image[:, :, -int(idx-(nlam//2))].T, cmap=neg_colormap, alpha=0.5)  
                    
                    
                    axes[idx].text(int(npix*0.9),int(npix*0.1),f'{v[-int(idx-(nlam//2))]+vkms:.1f} $km/s$',
                                   ha='right', va='top', color=text_color, fontsize=16)
                    axes[idx].set_xticks([int(npix*0.1), int(npix*0.3), npix//2, int(npix*0.7), int(npix*0.9)])
                    axes[idx].set_xticklabels([f'-{int((sizeau//2)*0.8)}',
                                               f'-{int((sizeau//2)*0.4)}',
                                               '0',
                                               f'{int((sizeau//2)*0.4)}',
                                               f'{int((sizeau//2)*0.8)}'],
                                              fontsize=14)
                    axes[idx].set_xlabel('AU',fontsize=16)
                    if idx == (nlam//2)+1:
                        axes[idx].set_yticks([int(npix*0.1), int(npix*0.3), npix//2, int(npix*0.7), int(npix*0.9)])
                        axes[idx].set_yticklabels([f'{int((sizeau//2)*0.8)}',
                                                   f'{int((sizeau//2)*0.4)}',
                                                   '0',
                                                   f'-{int((sizeau//2)*0.4)}',
                                                   f'-{int((sizeau//2)*0.8)}'],
                                                  fontsize=14)
                        axes[idx].set_ylabel('AU',fontsize=16)
                else:
                    pass
            # cbar_ax = fig.add_axes([0.95, 0.05, 0.01, 0.9])
            cbar = fig.colorbar(c, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)          
            return fig, cbar
        
        
        # image = np.ones(image.shape)
        try:
            fig, cbar = channel(image=image, conti=cube_conti.imageJyppix)
        except:
            fig, cbar = channel(image=image)
        cbar.set_label('Intensity (mJy/pixel)')
        self.save_plot(fig=fig, parms=parms)
        
        if parms.convolve is True:
            convolved_image = convolve(image, fwhm)
            try:
                convolved_conti = convolve(cube_conti.imageJyppix, fwhm)
                fig, cbar = channel(image=convolved_image, conti=convolved_conti, convolved=True)
            except:
                fig, cbar = channel(image=convolved_image, convolved=True)
            cbar.set_label('Intensity (mJy/beam)')
            self.save_plot(fig=fig, parms=parms, f='_convolved')
                
    def plot_pv(self, parms):
        
        if os.path.isdir(parms.cube_dir):
            try:
                if os.path.isfile(parms.cube_dir+parms.cube_fname):
                    if parms.cube_fname.endswith(".out"):
                        cube = readImage(parms.cube_dir+parms.cube_fname)
                        image = cube.imageJyppix
                    elif parms.cube_fname.endswith(".npz"):
                        raw_cube = np.load(parms.cube_dir+parms.cube_fname)
                        cube = npz_to_class(raw_cube)
                        image = cube.imageJyppix
                    else:
                        print('No correct cube file is given')
                        pass
                else:
                    print('No cube file is found')
                    pass 
            except:
                if isinstance(parms.cube_fname, list):
                    for fname in parms.cube_fname:
                        if fname.endswith("_scat.out"):
                            cube = readImage(parms.cube_dir+fname)
                        elif fname.endswith("_scat.npz"):
                            raw_cube_gas = np.load(parms.cube_dir+fname)
                            cube = npz_to_class(raw_cube_gas)
                        elif fname.endswith("_conti.out"):
                            cube_conti = readImage(parms.cube_dir+fname)
                        elif fname.endswith("_conti.npz"):
                            raw_cube_dust = np.load(parms.cube_dir+fname)
                            cube_conti = npz_to_class(raw_cube_dust)
                        else:
                            print("No correct cube file is given")
                            break
                    image = cube.imageJyppix - cube_conti.imageJyppix
                else:
                    print('No correct cube file is given')
                    pass
        else:
            print('No correct cube directory is given')
            pass
        
        nlam = len(cube.wav)
        sizeau = int(round((cube.x/au)[-1]))*2
        npix = cube.nx
        freq0 = (cube.freq[nlam//2] + cube.freq[(nlam//2)-1])/2
        v = cc / 1e5 * (freq0 - cube.freq) / freq0
        v_width = round(v[-1]-v[0])//2
        vkms = getattr(parms, 'vkms', 5)
        center = int(cube.ny//2)
        
        image = image * 1e3/(parms.d**2) 
        x_axis, v_axis = cube.x/au, v+vkms
        
        def convolve(image, fwhm):
            convolved_image = np.zeros(shape=image.shape)
            for i in range(nlam):
                sigma = fwhm * (npix/sizeau)/ (2*np.sqrt(2*np.log(2)))
                convolved_image[:, :, i] = gaussian_filter(image[:, :, i], sigma=sigma)
            return convolved_image*(np.pi/4*((fwhm*npix/sizeau)**2))
        
        def slice_pv(image, fwhm):
            pv_slice = np.sum(image[:, int(center-((5*npix/sizeau)//2)):int(center+((5*npix/sizeau)//2)), :], axis=1)
            # pv_slice = image[:, int(center), :]
            return pv_slice.T
        
        def pv(x_axis, v_axis, pv_slice, convolve=False):
            fig, ax = plt.subplots()
            c = ax.pcolormesh(x_axis, v_axis, pv_slice, shading="nearest", rasterized=True, cmap='gist_ncar', vmin=-5, vmax=50)
            cbar = fig.colorbar(c, ax=ax)
            if convolve is True:
                cbar.set_label('mJy/beam',fontsize = 16)
            else:
                cbar.set_label('mJy/pixel',fontsize = 16)
            ax.set_xlabel("Offset [au]",fontsize = 16)
            ax.set_ylabel("Velocity [km/s]",fontsize = 16)
            ax.plot([0, 0], [-v_width+vkms, v_width+vkms], 'w:')
            ax.plot([-(sizeau//2), (sizeau//2)], [vkms, vkms], 'w:')
            
            if parms.CB68 is True:
                CB68_PV = np.load('../../CB68/CB68_PV.npy')
                v_axis_cb68 = np.linspace(0, 10, 60, endpoint=True)
                offset = np.linspace(-150, 150, 50, endpoint=True) 
                O, V = np.meshgrid(offset, v_axis_cb68)
                # contour_levels = np.linspace(0.01, CB68_PV.max(), 4)
                contour_levels = np.array([0.01, 0.02, 0.03, 0.04])
                contour = ax.contour(O, V, CB68_PV[::-1, ::-1], levels=contour_levels, colors='w', linewidths=1)
                ax.set_xlim(max(np.min(x_axis), np.min(offset)), min(np.max(x_axis), np.max(offset)))
                ax.set_ylim(max(np.min(v_axis), np.min(v_axis_cb68)), min(np.max(v_axis), np.max(v_axis_cb68)))
            
            
            return fig, ax
        
        title = getattr(parms, 'title', '')
        fwhm  = getattr(parms, 'fwhm', 50)
        
        pv_slice = slice_pv(image, fwhm=fwhm)
        pv_plot, ax_plot = pv(x_axis, v_axis, pv_slice)
        ax_plot.set_title(title, fontsize = 16)
        self.save_plot(fig=pv_plot, parms=parms)
        
        if parms.convolve is True:
            convolved_image = convolve(image, fwhm=fwhm)
            pv_slice = slice_pv(convolved_image, fwhm=fwhm)
            pv_plot, ax_plot = pv(x_axis, v_axis, pv_slice, convolve=True)
            ax_plot.set_title(title+' (convolved)', fontsize = 16)
            self.save_plot(fig=pv_plot, parms=parms, f='_convolved')

    def plot_continuum(self, parms):
        
        if os.path.isdir(parms.conti_dir):
            if os.path.isfile(parms.conti_dir+parms.conti_fname):
                if parms.conti_fname.endswith(".out"):
                    conti = readImage(parms.conti_dir+parms.conti_fname)
                    image = conti.imageJyppix
                elif parms.conti_fname.endswith(".npz"):
                    raw_conti = np.load(parms.conti_dir+parms.conti_fname)
                    conti = npz_to_class(raw_conti)
                    image = conti.imageJyppix
                else:
                    print('No correct continuum file is given')
                    pass
            else:
                print('No continuum file is found')
                pass 
        else:
            print('No correct continuum directory is given')
            pass
        

        sizeau = int(round((conti.x/au)[-1]))*2
        npix = conti.nx
        image = image * 1e3/(parms.d**2) # mJy
        
        
        title = getattr(parms, 'title', '')
        fwhm  = getattr(parms,  'fwhm', 50)
        
        
        
        
        def convolve(image, fwhm):
            convolved_image = np.zeros(shape=image.shape)
            sigma = fwhm * (npix/sizeau)/ (2*np.sqrt(2*np.log(2)))
            convolved_image[:, :, 0] = gaussian_filter(image[:, :, 0], sigma=sigma)
            return convolved_image*(np.pi/4*((fwhm*npix/sizeau)**2))
        
        def plot_conti(image, convolved=True):
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
            
            
                
            vmin, vmax = np.min(image), np.max(image)
            x, y = np.linspace(0, npix, npix), np.linspace(0, npix, npix)
            X, Y = np.meshgrid(x, y)
            contour_level = np.linspace(0.2*vmax, 0.8*vmax, 4, endpoint=True)
            extent = [0, npix, 0, npix]
            
            c = ax.imshow(image[:, ::-1, 0].T, cmap='jet',
                            vmin=vmin, vmax=vmax, extent=extent)
            ax.contour(Y, X, image[:, :, 0], levels=contour_level, colors='w', linewidths=1)
            if convolved is True:
                ax.set_title(title+ ' (convolved)', fontsize = 16)
            else:
                ax.set_title(title, fontsize = 16)
            ax.set_title(title, fontsize = 16)
            ax.set_xticks([int(npix*0.1),
                        int(npix*0.3),
                        npix//2,
                        int(npix*0.7),
                        int(npix*0.9)])
            ax.set_xticklabels([f'-{int((sizeau//2)*0.8)}',
                                f'-{int((sizeau//2)*0.4)}',
                                '0',
                                f'{int((sizeau//2)*0.4)}',
                                f'{int((sizeau//2)*0.8)}'],
                            fontsize=14)
            ax.set_xlabel('AU',fontsize=16)
            ax.set_yticks([int(npix*0.1),
                        int(npix*0.3),
                        npix//2,
                        int(npix*0.7),
                        int(npix*0.9)])
            ax.set_yticklabels([f'-{int((sizeau//2)*0.8)}',
                                f'-{int((sizeau//2)*0.4)}',
                                '0',
                                f'{int((sizeau//2)*0.4)}',
                                f'{int((sizeau//2)*0.8)}'],
                            fontsize=14)
            ax.set_ylabel('AU',fontsize=16)
            
            divider = make_axes_locatable(ax)
            cax2 = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(c, cax=cax2, orientation='vertical', shrink=0.8)
            if convolved is True:
                cbar.set_label('Intensity (mJy/beam)')
            else:
                cbar.set_label('Intensity (mJy/pixel)')
            
            return fig
        
        conti_plot = plot_conti(image=image, convolved=False)
        self.save_plot(fig=conti_plot, parms=parms)
        if parms.convolve is True:
            convolved_image = convolve(image=image, fwhm=fwhm)
            conti_plot_convolved = plot_conti(image=convolved_image, convolved=True)
            self.save_plot(fig=conti_plot_convolved, parms=parms, f='_convolved')
    
    def plot_spectrum(self, parms):
        
        if os.path.isdir(parms.spectra_dir):
            try:
                if os.path.isfile(parms.spectra_dir+parms.spectra_fname):
                    if parms.spectra_fname.endswith(".out"):
                        spectrum = readSpectrum(parms.spectra_dir+parms.spectra_fname)
                        lam = spectrum[:, 0]
                        fnu = spectrum[:, 1]
                    elif parms.spectra_fname.endswith(".npz"):
                        raw_spectrum = np.load(parms.spectra_dir+parms.spectra_fname)
                        spectrum = npz_to_class(raw_spectrum)
                        lam = spectrum.lam
                        fnu = spectrum.fnu
                    else:
                        print('No correct spectrum file is given')
                        pass
                else:
                    print('No spectrum file is found')
                    pass 
            except:
                if isinstance(parms.spectra_fname, list):
                    for fname in parms.spectra_fname:
                        if fname.endswith("_scat.out"):
                            line = readSpectrum(parms.spectra_dir+fname)
                            lam = line[:, 0]
                            fnu = line[:, 1]
                        elif fname.endswith("_scat.npz"):
                            raw_line = np.load(parms.spectra_dir+fname)
                            line = npz_to_class(raw_line)
                            lam = line.lam
                            fnu = line.fnu
                        elif fname.endswith("_conti.out"):
                            conti = readSpectrum(parms.spectra_dir+fname)
                            fnu_conti = line[:, 1]
                        elif fname.endswith("_conti.npz"):
                            raw_conti = np.load(parms.spectra_dir+fname)
                            conti = npz_to_class(raw_conti)
                            fnu_conti = conti.fnu
                        else:
                            print("No correct spectrum file is given")
                            break
                    fnu = fnu - fnu_conti
                else:
                    print('No correct spectrum file is given')
                    pass
        else:
            print('No correct spectrum directory is given')
            pass
        
        def plot_specta(lam, fnu, d, vkms):
            fnu = fnu * 1e26 / d ** 2  # mJy

            freq = (cc*1e-2) / (lam*1e-6)
            freq0 = (freq[len(lam)//2] + freq[(len(lam)//2)-1])/2
            v = cc / 1e5 * (freq0 - freq) / freq0
            
            fig, ax = plt.subplots()
            
            ax.plot(v+vkms, fnu)
            # ax.set_xscale('log')
            # ax.set_yscale('log')
            ax.set_xlabel(r'v [km/s]')
            ax.set_ylabel(r'Flux Density [mJy]')
            # ax.set_ylim(1e-1, 1e3)
            ax.set_xlim(np.min(v+vkms), np.max(v+vkms))
            
            return fig, ax

        title = getattr(parms, 'title', '')
        vkms = getattr(parms, 'vkms', 5)
        fig, ax = plot_specta(lam, fnu, parms.d, vkms)
        ax.set_title(title)
        self.save_plot(fig=fig, parms=parms)
        
    def plot_sed(self, parms):
        
        if os.path.isdir(parms.sed_dir):
            if os.path.isfile(parms.sed_dir+parms.sed_fname):
                if parms.sed_fname.endswith(".out"):
                    sed = readSpectrum(parms.sed_dir+parms.sed_fname)
                    lam = sed[:, 0]
                    fnu = sed[:, 1]
                elif parms.sed_fname.endswith(".npz"):
                    raw_sed = np.load(parms.sed_dir+parms.sed_fname)
                    sed = npz_to_class(raw_sed)
                    lam = sed.lam
                    fnu = sed.fnu
                else:
                    print('No correct sed file is given')
                    pass
            else:
                print('No sed file is found')
                pass 
        else:
            print('No correct sed directory is given')
            pass
        
        def plot_spectral_energy_distribution(lam, fnu, d):
            fnu = fnu * 1e26 / d ** 2  # mJy
            nu  = 1e-9*(1e-2*cc)/(1e-6*lam) # GHz
            fig, ax = plt.subplots()
            
            ax.plot(nu, fnu)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\nu$ [GHz]')
            ax.set_ylabel(r'Flux Density [mJy]')
            ax.set_ylim(1e-1, 1e3)
            ax.set_xlim(np.min(nu), np.max(nu))
            if parms.CB68 is True:
                fnu_cb68 = [   56,    55,    59,    62,    60,    60,    61,    66]
                nu_cb68  = [233.8, 233.8, 233.8, 246.7, 246.7, 246.7, 246.7, 246.7]
                ax.scatter(nu_cb68, fnu_cb68, color='k')
            
            return fig, ax

        title = getattr(parms, 'title', '')
        
        fig, ax = plot_spectral_energy_distribution(lam, fnu, parms.d)
        ax.set_title(title)
        self.save_plot(fig=fig, parms=parms)

        return
    
    def save_plot(self, fig, parms, **kwargs):
        
        dir = getattr(parms, 'dir', './test/')
        
        if 'fname' in kwargs.keys():
            fname = kwargs['fname']
        else:
            fname = getattr(parms, 'fname', 'test')
        
        if 'f' in kwargs.keys():
            f = kwargs['f']
        else:
            f = ''

        os.makedirs(dir, exist_ok=True)
        fig.savefig(dir+fname+f+'.pdf', transparent=True)
        plt.close('all')

