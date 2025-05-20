import os
import numpy as np
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from radmc3dPy.data import *

class generate_simulation:
    
    def __init__(self,
                 save_npz=True,
                 save_out=True,
                 ):
        
        self.save_out = save_out
        self.save_npz = save_npz
        if (self.save_out is False) and (self.save_npz is False):
            self.save_out = True

    def generate_cube(self,
                      dir       = './test/',
                      fname     = 'test', 
                      npix      = 100,
                      sizeau    = 100,
                      incl      = 73,
                      line      = 240,
                      v_width   = 10,
                      vkms      = 0,
                      nlam      = 11,
                      posang    = 45,
                      nodust    = False,
                      scat      = True,
                      extract_gas=True,
                      **kwargs):

        """
        This function will generate a cube of spectral line.

        Parameters
        -----------------
        dir                : str
            Directory to save the output
        fname              : str
            File name to save the output
        npix               : int
            Number of map's pixels
        sizeau             : float
            Map's span
        incl               : float
            Inclination angle of the disk
        line               : int
            Transistion level (see 'molecule_ch3oh.inp')
        v_width            : float
            Range of velocity to simulate
        vkms               : float
            Velocity of the line center
        nlam               : int
            Number of velocities
        nodust             : bool
            If False, dust effect is included
        scat               : bool
            If True and nodust=False, scattering is included. (Time-consuming)
        extracted_gas      : bool
            If True, spectral line is extracted (I_{dust+gas}-I_{dust})
        
        """
        
        prompt = f'npix {npix} sizeau {sizeau} incl {incl} posang {-posang} iline {line} vkms {vkms} widthkms {v_width} linenlam {nlam}'
        
        if nlam > 15:
            type_note = 'pv'
        else:
            type_note = 'channel'
        
        
        if extract_gas is False:
            if nodust is True:
                prompt = prompt + ' noscat nodust'
                f      = '_nodust'
            elif nodust is False:
                if scat is True:
                    f      = '_scat'
                elif scat is False:
                    prompt = prompt +' noscat'
                    f      = '_noscat'
                else:
                    pass
            else:
                pass
                    
            os.system(f"radmc3d image "+prompt)
            

            self.cube = readImage('image.out')
            if self.save_npz is True:
                self.save_npzfile(self.cube, dir=dir, fname=fname, f=f, note=type_note)

            if self.save_out is True:
                self.save_outfile(dir=dir, fname=fname, f=f, note=type_note)
                    
        elif extract_gas is True:
            os.system(f"radmc3d image "+prompt)

            self.cube = readImage('image.out')
            if self.save_npz is True:
                self.save_npzfile(self.cube, dir=dir, fname=fname, f='_scat', note=type_note)
            
            if self.save_out is True:
                self.save_outfile(dir=dir, fname=fname, f='_scat', note=type_note)

            if self.cube.nwav%2 == 0:
                mid_wav = 0.5 * (self.cube.wav[self.cube.nwav//2] + self.cube.wav[(self.cube.nwav//2)+1])
            else:
                mid_wav = self.cube.wav[(self.cube.nwav//2)+1]
            
            os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} posang {-posang} lambda {mid_wav} noline")

            self.conti = readImage('image.out')
            if self.save_npz is True:
                self.save_npzfile(self.conti, dir=dir, fname=fname, f='_conti', note=type_note)
            
            if self.save_out is True:
                self.save_outfile(dir=dir, fname=fname, f='_conti', note=type_note)


            self.cube_list = [self.cube, self.conti]
            if self.save_npz is True:
                self.save_npzfile(self.cube_list, dir=dir, fname=fname, f='_extracted', note=type_note)
        else:
            pass
            
    def generate_continuum(self,
                           dir      = './test/', 
                           fname    = 'test',
                           incl     = 73,
                           wav      = 1300,
                           npix     = 200,
                           sizeau   = 100,
                           posang   = 45,
                           scat     = True, 
                           **kwargs):
        """
        This function will generate a continuum image.

        Parameters
        -----------------
        dir                : str
            Directory to save the output
        fname              : str
            File name to save the output
        incl               : float
            Inclination angle of the disk
        wav                : float
            Wavelength to simulate
        npix               : int
            Number of map's pixels
        sizeau             : float
            Map's span
        posang             : float
            Position angle of the disk
        scat               : bool
            If True, scattering is included. (Time-consuming)
        """

        type_note = 'conti'

        prompt = f'radmc3d image npix {npix} sizeau {sizeau} incl {incl} lambda {wav} posang {-posang} noline'
        
        f = '_scat'
        
        if scat is False:
            prompt = prompt + ' noscat'
            f = '_noscat'
        os.system(prompt)
        
        
        self.conti = readImage('image.out')
        if self.save_npz is True:
            self.save_npzfile(self.conti, dir=dir, fname=fname, f=f, note=type_note)
        
        if self.save_out is True:
            self.save_outfile(dir=dir, fname=fname, f=f, note=type_note)
 
    def generate_sed(self,
                     dir        = './test/',
                     fname      = 'test',
                     incl       = 73,
                     freq_min   = 5e1,
                     freq_max   = 1e3,
                     nlam       = 100,
                     scat       = True,
                     **kwargs):
        
        """
        This function will generate a spectral energy distribution (SED).

        Parameters
        -----------------
        dir                : str
            Directory to save the output
        fname              : str
            File name to save the output
        incl               : float
            Inclination angle of the disk
        freq_min           : float
            Minimum frequency to simulate
        freq_max           : float
            Maximum frequency to simulate
        nlam               : int
            Number of wavelengths
        scat               : bool
            If True, scattering is included. (Time-consuming)
        """

        type_note = 'sed'
        
        wav_max  = ((cc*1e-2)/(freq_min*1e9))*1e+6
        wav_min  = ((cc*1e-2)/(freq_max*1e9))*1e+6
        
        
        prompt = f"radmc3d spectrum incl {incl} lambdarange {wav_min} {wav_max} nlam {nlam} noline"
        f = '_scat'
        if scat is False:
            prompt = prompt + ' noscat'
            f = '_noscat'

        os.system(prompt)

        if self.save_npz is True:
            self.spectrum = readSpectrum('spectrum.out')
            self.save_npzfile(self.spectrum, dir=dir, fname=fname, f=f, note=type_note)
        
        if self.save_out is True:
            self.save_outfile(dir=dir, fname=fname, f=f, note=type_note)

    def generate_line_spectrum(self,
                               dir        = './test/',
                               fname      = 'test',
                               incl       = 73,
                               line       = 240,
                               v_width    = 10,
                               nlam       = 10,
                               vkms       = 0,
                               nodust     = False,
                               scat       = True,
                               extract_gas= True,
                               **kwargs):
        """
        This function will generate a line spectrum.

        Parameters
        -----------------
        dir                : str
            Directory to save the output
        fname              : str
            File name to save the output
        incl               : float
            Inclination angle of the disk
        line               : int
            Transistion level (see 'molecule_ch3oh.inp')
        v_width            : float
            Range of velocity to simulate
        nlam               : int
            Number of velocities
        vkms               : float
            Velocity of the line center
        nodust             : bool
            If False, dust effect is included
        scat               : bool
            If True and nodust=False, scattering is included. (Time-consuming)
        extracted_gas      : bool
            If True, spectral line is extracted (I_{dust+gas}-I_{dust})
        """
        
        type_note = 'spectrum'
        
        
        prompt = f"radmc3d spectrum incl {incl} iline {line} vkms {vkms} widthkms {v_width} linenlam {nlam}"
        
        
        if extract_gas is False:
            if nodust is True:
                prompt = prompt + ' noscat nodust'
                f      = '_nodust'
            elif nodust is False:
                if scat is True:
                    f      = '_scat'
                elif scat is False:
                    prompt = prompt +' noscat'
                    f      = '_noscat'
                else:
                    pass
            else:
                pass
                    
            os.system(prompt)
            

            if self.save_npz is True:
                self.spectrum = readSpectrum('spectrum.out')
                self.save_npzfile(self.spectrum, dir=dir, fname=fname, f=f, note=type_note)
            
            if self.save_out is True:
                self.save_outfile(dir=dir, fname=fname, f=f, note=type_note)
                    
        elif extract_gas is True:

            os.system(prompt)

            self.spectrum = readSpectrum('spectrum.out')
            if self.save_npz is True:
                self.save_npzfile(self.spectrum, dir=dir, fname=fname, f='_scat', note=type_note)
            
            if self.save_out is True:
                self.save_outfile(dir=dir, fname=fname, f='_scat', note=type_note)

            
            os.system(f"radmc3d spectrum incl {incl} lambdarange {self.spectrum[0,0]} {self.spectrum[-1,0]} nlam {nlam} noline")

            self.conti = readSpectrum('spectrum.out')
            if self.save_npz is True:
                self.save_npzfile(self.conti, dir=dir, fname=fname, f='_conti', note=type_note)
                self.spectrum_list = [self.spectrum, self.conti]
                self.save_npzfile(self.spectrum_list, dir=dir, fname=fname, f='_extracted', note=type_note)
            
            if self.save_out is True:
                self.save_outfile(dir=dir, fname=fname, f='_conti', note=type_note)
            
        else:
            pass

    def save_outfile(self, dir, fname, **kwargs):
        """
        This will save whole information of simulation.
        The file type will be '*.out', which the storage may be MB-scale
        """
        
        if 'f' in kwargs.keys():
            f = kwargs['f']
        else:
            f = ''
            
        if 'note' in kwargs.keys():
            head = kwargs['note'] + '_'
        else:
            head = ''
        
        os.makedirs(dir+'outfile/', exist_ok=True)
        if (kwargs['note'] == 'sed') or (kwargs['note'] == 'spectrum'):
            os.system('mv spectrum.out '+dir+'outfile/'+head+fname+f+'.out')
        else:
            os.system('mv image.out '+dir+'outfile/'+head+fname+f+'.out')

    def save_npzfile(self, data, dir, fname, **kwargs):
        """
        This will only save image data regardless of other information
        The file tyep will be '*.npz', which the storage may be kB-scale
        """
        
        
        if 'f' in kwargs.keys():
            f = kwargs['f']
        else:
            f = ''
        
        if 'note' in kwargs.keys():
            head = kwargs['note'] + '_'
        else:
            head = ''
        
        os.makedirs(dir+'npzfile/', exist_ok=True)
        
        if (kwargs['note'] == 'sed') or (kwargs['note'] == 'spectrum'):
            np.savez(dir+'npzfile/'+head+fname+f+'.npz',
                    lam = data[:, 0],
                    fnu = data[:, 1])
        else:
            if isinstance(data, list):
                line_cube = data[0]
                dust_conti_value = data[1].imageJyppix
                dust_conti_value = np.tile(dust_conti_value[:, :, np.newaxis], (1, 1, line_cube.nwav))

                np.savez(dir+'npzfile/'+head+fname+f+'.npz',
                        imageJyppix = line_cube.imageJyppix - dust_conti_value,
                        x           = line_cube.x,
                        nx          = line_cube.nx,
                        y           = line_cube.y,
                        ny          = line_cube.ny,
                        wav         = line_cube.wav,
                        freq        = line_cube.freq,
                        nwav        = len(line_cube.wav),
                        nfreq       = len(line_cube.freq),)
            else:
                np.savez(dir+'npzfile/'+head+fname+f+'.npz',
                        imageJyppix = data.imageJyppix,
                        x           = data.x,
                        nx          = data.nx,
                        y           = data.y,
                        ny          = data.ny,
                        wav         = data.wav,
                        freq        = data.freq,
                        nwav        = len(data.wav),
                        nfreq       = len(data.freq),)
        
