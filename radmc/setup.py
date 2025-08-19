import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os
import io
import contextlib
from time import gmtime, strftime
from multiprocessing import cpu_count
from radmc3dPy.analyze import *
import optool

#
# Some natural constants
#
au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]
ms  = 1.98892e33     # Solar mass              [g]
ts  = 5.78e3         # Solar temperature       [K]
ls  = 3.8525e33      # Solar luminosity        [erg/s]
rs  = 6.96e10        # Solar radius            [cm]
#
# Disk Model
#

from X22_model.disk_model import *
from radmc.spherical_x22 import DiskModel_spherical

class Grid:
    def __init__(self):
        self.grid_center = None
        self.grid_edge   = None
        self.n1 = None
        self.n2 = None
        self.n3 = None
        return
    
    def sph_grid(self,
                       r_bound = [ 1e-3*au,   50*au],
                   theta_bound = [np.pi/12,   np.pi],
                     phi_bound = [       0, 2*np.pi],
                            nr = 200,
                        ntheta = 200,
                          nphi = 100,
                     r_spacing = 'log',
                 theta_spacing = 'log',
                   phi_spacing = 'lin',
          theta_fined_midplane = True,):
        
        self.n1 = nr
        self.n2 = ntheta
        self.n3 = nphi

        if r_spacing == 'log':
            if r_bound[0] <= 0:
                print('Warning: r_bound[0] is set to 0 or negative, which may cause issues with log spacing.')
                print('Setting r_bound[0] to a small value (1e-6 AU).')
                r_bound[0] = 1e-6 * au
            r_edge = np.logspace(np.log10(r_bound[0]), np.log10(r_bound[1]), nr+1, endpoint=True)
        else:
            r_edge = np.linspace(r_bound[0], r_bound[1], nr+1, endpoint=True)
        
        if theta_spacing == 'log' and theta_fined_midplane is True:
            if theta_bound[0] <= 0:
                print('Warning: theta_bound[0] is set to 0 or negative, which may cause issues with log spacing.')
                print('Setting theta_bound[0] to a small value (1e-6 radians).')
                theta_bound[0] = 1e-6
            theta_edge_upper = np.logspace(np.log10(theta_bound[0]), np.log10(np.pi/2), ntheta//2+1, endpoint=True)
            theta_edge_upper = -theta_edge_upper + np.pi/2 + theta_bound[0]
            theta_edge_upper = theta_edge_upper[::-1]
            theta_edge_lower = -theta_edge_upper[::-1] + np.pi
            theta_edge = np.concatenate((theta_edge_upper[:-1], theta_edge_lower))
        else:
            theta_edge = np.linspace(theta_bound[0], theta_bound[1], ntheta+1, endpoint=True)

        if phi_spacing == 'log':
            print('Warning: phi_spacing is set to log, which is not supported. Using linear spacing instead.')
        phi_edge = np.linspace(phi_bound[0], phi_bound[1], nphi+1, endpoint=True)


        r_center     = 0.5 * (    r_edge[:-1] + r_edge[1:])
        theta_center = 0.5 * (theta_edge[:-1] + theta_edge[1:])
        phi_center   = 0.5 * (  phi_edge[:-1] + phi_edge[1:])

        self.grid_center = [r_center, theta_center, phi_center]
        self.grid_edge   = [  r_edge,   theta_edge,   phi_edge]

        self.grid_type = 'sph'

    '''
    Under construction, not used yet.
    '''
    def rect_grid(self,
                   x_bound = [-50*au, 50*au],
                   y_bound = [-50*au, 50*au],
                   z_bound = [-50*au, 50*au],
                        nx = 200,
                        ny = 200,
                        nz = 200,
                 x_spacing = 'lin',
                 y_spacing = 'lin',
                 z_spacing = 'lin'):
        self.n1 = nx
        self.n2 = ny
        self.n3 = nz

        self.grid_type = 'rect'

    def cyl_grid(self,
                     r_bound = [1e-3*au,   50*au],
                   phi_bound = [      0, 2*np.pi],
                     z_bound = [ -50*au,   50*au],
                          nr = 200,
                        nphi = 100,
                          nz = 200,
                   r_spacing = 'log',
                 phi_spacing = 'lin',
                   z_spacing = 'lin'):
        self.n1 = nr
        self.n2 = nphi
        self.n3 = nz

        self.grid_type = 'cyl'



class Model:
    '''
    A class to handle the disk model.
    This is a base class, and should be inherited by other disk models.
    '''

    def __init__(self):
        return

    def generate_opacity_optool(
            self, 
            a_min=1e-5, 
            a_max=0.1, 
            q=-3.5,
            composition='X22',
            fnames = ['temp_regime_1', 'temp_regime_2', 'temp_regime_3', 'temp_regime_4'],
            table_cache='./kappa/',
            inputstyle=20
        ):
        
        if not os.path.exists(table_cache):
            os.makedirs(table_cache)

        prompt = f'optool -a {a_min*1e3} {a_max*1e3} {-q} -l 0.1 100000 101 -mie -radmc '

        if inputstyle == 10 or inputstyle == 20:
            prompt += '-s '

        if composition.lower() == 'x22':

            material_frac = ['h2o-w 0.2', 'c-org 0.3966', 'fes 0.0743', 'astrosil 0.3291']
            if fnames is None:
                fnames = ['temp_regime_1', 'temp_regime_2', 'temp_regime_3', 'temp_regime_4']
            for idx in range(len(material_frac)):
                p = optool.particle(prompt + f'-c {' '.join(material_frac[(idx):])}',
                                    cache=f'./{table_cache}/{fnames[idx]}/',
                                    silent=True)
        elif composition.lower() == 'dsharp':
            p = optool.particle(prompt + f'-dsharp',
                                cache=f'./{table_cache}/dsharp/',
                                silent=True)

    def combine_opacity_tables(
            self,
            opacity_tables,
            T_crit=[150, 425, 680, 1200],
            fraction = [0.2, 0.3966, 0.0743, 0.3291],
            dust_to_gas=0.01, T_min=20, T_max=2000, N_T=100,
        ):

        if  len(opacity_tables) != len(T_crit) or \
            len(opacity_tables) != len(fraction) or \
            len(T_crit) != len(fraction):
            raise ValueError('opacity_tables, T_crit, and fraction must have the same length')

        self.ndust_spec = len(T_crit)
        self.T_crit = T_crit

        kappa   = []
        kappa_s = []
        kappa_p = []
        kappa_r = []
        g       = []
        
        for idx, p in enumerate(opacity_tables):
            kappa.append(  p.kabs[0,:]*dust_to_gas*np.sum(fraction[idx:]))
            kappa_s.append(p.ksca[0,:]*dust_to_gas*np.sum(fraction[idx:]) * (1-p.gsca[0,:]))
            with warnings.catch_warnings(): 
                warnings.simplefilter('ignore')
                p.computemean(tmin=T_min, tmax=T_max, ntemp=N_T)
            kappa_p.append(p.kplanck[0,:]*dust_to_gas*np.sum(fraction[idx:]))
            kappa_r.append(p.kross[0,:]*dust_to_gas*np.sum(fraction[idx:]))
            g.append(p.gsca[0,:]*dust_to_gas*np.sum(fraction[idx:]))

        T_grid = p.temp
        kappa   = np.array(kappa)
        kappa_s = np.array(kappa_s)
        kappa_p = np.array(kappa_p)
        kappa_r = np.array(kappa_r)
        g = np.array(g)
        # combine different temperature regimes
        def combine_temperature_regimes(y):
            if y.ndim>2:
                return np.array([combine_temperature_regimes(y1) for y1 in y])
            y_out = np.zeros(y.shape[1:])
            for i in range(len(T_crit))[::-1]:
                y_out[T_grid<T_crit[i]] = y[i,T_grid<T_crit[i]]
            return y_out

        kappa_p = combine_temperature_regimes(kappa_p)
        kappa_r = combine_temperature_regimes(kappa_r)

        opacity_table = {}
        opacity_table['T_crit'] = T_crit
        opacity_table['T'] = T_grid
        opacity_table['lam'] = p.lam * 1e-4
        opacity_table['kappa'] = kappa
        opacity_table['kappa_s'] = kappa_s
        opacity_table['kappa_p'] = kappa_p
        opacity_table['kappa_r'] = kappa_r
        opacity_table['g'] = g
        self.opacity_table = opacity_table
        return opacity_table

    def read_opacity_table(
            self,
            p,
            dust_to_gas=0.01,
            T_crit=[1200],
            T_min=20,
            T_max=2000,
            N_T=100,
        ):
        self.ndust_spec = len(T_crit)
        self.T_crit = T_crit

        kappa   = []
        kappa_s = []
        kappa_p = []
        kappa_r = []
        g       = []

        kappa.append(  p.kabs[0,:]*dust_to_gas)
        kappa_s.append(p.ksca[0,:]*dust_to_gas * (1-p.gsca[0,:]))
        with warnings.catch_warnings(): 
            warnings.simplefilter('ignore')
            p.computemean(tmin=T_min, tmax=T_max, ntemp=N_T)
        kappa_p.append(p.kplanck[0,:]*dust_to_gas)
        kappa_r.append(p.kross[0,:]*dust_to_gas)
        g.append(p.gsca[0,:]*dust_to_gas)
        
        T_grid = p.temp
        kappa   = np.array(kappa)
        kappa_s = np.array(kappa_s)
        kappa_p = np.array(kappa_p)
        kappa_r = np.array(kappa_r)
        g = np.array(g)
        # combine different temperature regimes
        def combine_temperature_regimes(y):
            if y.ndim>2:
                return np.array([combine_temperature_regimes(y1) for y1 in y])
            y_out = np.zeros(y.shape[1:])
            for i in range(len(T_crit))[::-1]:
                y_out[T_grid<T_crit[i]] = y[i,T_grid<T_crit[i]]
            return y_out

        kappa_p = combine_temperature_regimes(kappa_p)
        kappa_r = combine_temperature_regimes(kappa_r)

        opacity_table = {}
        opacity_table['T_crit'] = T_crit
        opacity_table['T'] = T_grid
        opacity_table['lam'] = p.lam * 1e-4
        opacity_table['kappa'] = kappa
        opacity_table['kappa_s'] = kappa_s
        opacity_table['kappa_p'] = kappa_p
        opacity_table['kappa_r'] = kappa_r
        opacity_table['g'] = g
        self.opacity_table = opacity_table
        return opacity_table
    
    def X22(self,
           opacity_table = None,
            d_to_g_ratio = 0.01,
                   a_max = 0.1,
                       q = -3.5,
            Mass_of_star = None,
          Accretion_rate = None,
          Radius_of_disk = None,
                       Q = None,
                      NR = 200,
                  NTheta = 200,
                    NPhi = 200,
      dustkap_inputstyle = None):
        '''
        GIdisk2obs published by Xu+22.

        Parameters
        ----------------------------
        d_to_g_ratio : float
            dust-to-gas mass ratio 
        a_max : float
            Maxmum grain size (unit: mm)
        q : float
            Slope for grain size distribution
        Mass_of_star : float
            Mass of protostar (unit: M_sun)
        Accretion_rate : float
            Accretion rate    (unit: M_sun/yr)
        Radius_of_disk : float
            Radius of disk    (unit: AU)
        Q : float
            Toomre index
        NR : int
            Resolution in R axis
        NTheta : int
            Resolution in theta axis
        NPhi : int
            Resolution in phi axis
        '''

        self.dust_to_gas_ratio = d_to_g_ratio
        # note: the original a_max is in cm
        # a_min is set to be 0.05 um = 5e-6 cm
        if opacity_table is None:
            if self.opacity_table is None:
                if dustkap_inputstyle is None: self.dustkap_inputstyle = 20
                self.opacity_table  = generate_opacity_table_opt(a_min=1e-6, a_max=a_max*0.1,
                                                            q=q, dust_to_gas=d_to_g_ratio,
                                                            inputstyle=self.dustkap_inputstyle)
        else:
            self.opacity_table = opacity_table
        self.disk_property_table = generate_disk_property_table(self.opacity_table)
        DM = DiskModel_spherical(self.opacity_table, self.disk_property_table)
        
        self.Mstar = Mass_of_star
        self.Rd = Radius_of_disk
        DM.input_disk_parameter(Mstar=Mass_of_star*Msun,
                                Mdot=Accretion_rate*Msun/yr,
                                Rd=self.Rd*au,
                                Q=Q,
                                N_R=NR
                                )

        DM.extend_to_spherical(NTheta=NTheta, NPhi=NPhi)

        rho_spec = np.empty((DM.rho_sph.shape[0], 
                             DM.rho_sph.shape[1], 
                             DM.rho_sph.shape[2], 
                             self.ndust_spec), dtype=np.float64)
        for i, t_crit in enumerate(self.T_crit):
            if i == 0:
                rho_spec[:, :, :, i] = np.where(
                    DM.T_sph <= t_crit,
                    DM.rho_sph,
                    1e-18
                    )
            else:
                density_dust = np.where(
                    DM.T_sph <= t_crit,
                    DM.rho_sph,
                    1e-18
                    )
                density_dust = np.where(
                    self.T_crit[i-1] < DM.T_sph,
                    density_dust,
                    1e-18
                    )
                rho_spec[:, :, :, i] = density_dust
            
        self.DM = DM
        self.rho_dust = self.dust_to_gas_ratio * rho_spec
        self.rho_gas = DM.rho_sph
        self.T = np.tile(DM.T_sph[:, :, :, np.newaxis], (1, 1, 1, self.ndust_spec))
        self.model_type = 'X22'
        self.Grid = None
        self.model_grid = [DM.r_sph*au, DM.theta_sph, DM.phi_sph]
        self.model_grid_type = 'sph'
        return DM

    def uniform_sphere(self,
                       rho=1e-12,
                       temperature=100,
                       dust_to_gas_ratio=0.01,
                       sph_grid=None,):
        
        self.dust_to_gas_ratio = dust_to_gas_ratio

        rho_sph = np.full((sph_grid.n1, sph_grid.n2, sph_grid.n3, self.ndust_spec), rho)
        T_sph = np.full((sph_grid.n1, sph_grid.n2, sph_grid.n3, self.ndust_spec), temperature)
        self.rho_dust = rho_sph * self.dust_to_gas_ratio
        self.rho_gas = rho_sph
        self.T = T_sph
        self.model_type = 'uniform_sphere'
        self.Grid = sph_grid
        self.model_grid = [sph_grid.grid_center[0], sph_grid.grid_center[1], sph_grid.grid_center[2]]
        self.model_grid_type = 'sph'

    def interp_model_to_grid(self,
                             grid=None):
        '''
        Interpolate the disk model to the given grid.
        
        Parameters
        ----------------------------
        grid : Grid
            The grid to interpolate the disk model to.
        '''
        if not isinstance(grid, Grid):
            raise TypeError('grid must be an instance of Grid class')

        if self.model_grid_type == grid.grid_type:
            if self.Grid == grid:
                return
            def interp_to_grid(data_map, old_grid, new_grid, fill_value=1e-20):
                '''
                Interpolate the model to the grid.
                '''
                interpolator = RegularGridInterpolator(
                    (old_grid[0], old_grid[1], old_grid[2]),
                    data_map,
                    bounds_error=False,
                    fill_value=fill_value,
                )

                X_grid_new, Y_grid_new, Z_grid_new = np.meshgrid( # X, Y, Z can also be R, Theta, Phi, etc.
                    new_grid[0],
                    new_grid[1],
                    new_grid[2],
                    indexing='ij'
                )

                points_new = np.stack((X_grid_new.ravel(), Y_grid_new.ravel(), Z_grid_new.ravel()), axis=-1)
                return interpolator(points_new).reshape(X_grid_new.shape)
            

            rho_dust_new = np.empty((grid.n1, grid.n2, grid.n3, self.ndust_spec), dtype=np.float64)
            T_new = np.empty((grid.n1, grid.n2, grid.n3, self.ndust_spec), dtype=np.float64)
            for i in range(self.ndust_spec):
                rho_dust_new[:, :, :, i] = interp_to_grid(
                    self.rho_dust[:, :, :, i],
                    self.model_grid,
                    grid.grid_center,
                    fill_value=1e-18 * self.dust_to_gas_ratio
                )
                T_new[:, :, :, i] = interp_to_grid(
                    self.T[:, :, :, i],
                    self.model_grid,
                    grid.grid_center,
                    fill_value=20
                )
            self.rho_dust = rho_dust_new
            self.T = T_new
            self.rho_gas = interp_to_grid(self.rho_gas, self.model_grid, grid.grid_center, fill_value=1e-18)
            self.model_grid = grid.grid_center
            self.model_grid_type = grid.grid_type
        
        else:
            print('Warning: The grid type is not the same as the model grid type. The interpolation is under construction.')
            exit()


class Setup:
    def __init__(self, model_class, grid_class=None, silent=False):
        if not isinstance(model_class, Model):
            raise TypeError('model must be an instance of Model class')
        if grid_class is not None and not isinstance(grid_class, Grid):
            raise TypeError('grid must be an instance of Grid class')
        self.model = model_class
        self.grid = grid_class
        self.now = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        # counting number of processors
        self.num_cpu = cpu_count()
        if silent == False:
            print("You have {0} Processors".format(self.num_cpu))
    
    def __del__(self):
        pass

    def get_mastercontrol(self, filename = None,
                              comment    = None,
                              incl_dust  = None,
                              incl_lines = None,
                              nphot      = 1000000,
                              nphot_scat = 1000000,
                              scattering_mode_max = None,
                              istar_sphere   = 1,
                              num_cpu  = None,
                              **kwargs
                       ):
        '''
        Preparing the master control file for radmc3d.

        Example:
            test.get_mastercontrol(comment = 'this is a test', a=1.0, b=2.0, c=3.0)

        Parameters
        -----------------------
        
        filename : string
            output filename. (default: radmc3d.inp)
            It will still creat a file with default name, but will duplicate an output file with the specified name.
        comment : string
            comment to add to the file header (default: None)
        incl_dust : 0/1/None  
            0: force not include dust/ 1: force include / None: let radmc3d determine
        incl_lines : 0/1/None
            0: force not include line/ 1: force include / None: let radmc3d determine
        nphot : int
            The number of photon packages used for the thermal Monte Carlo simulation (default: 1000000)
        nphot_scat : int
            The number of photon packages for the scattering Monte Carlo simulations, done before image-rendering (default: 1000000)
        scattering_mode_max : 0/1/2/None
            0: no scattering / 1: isotropic scattering / 2: full scattering / None: let radmc decide (default: None)
        istar_sphere : 0/1
            If 0/1, treat stars as point-source/sphere (default: 1)
        num_cpu : int
            number of cpu core to use (default: available threads-2)
        
        Note
        ------------------------
        Other possible options (including using **kwargs) see
            https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/inputoutputfiles.html
        '''
        default_filename = 'radmc3d.inp'
        with open(default_filename, 'w+') as f:
            # setting parameters
            if num_cpu == None:
                num_cpu = self.num_cpu -2

            # writing parameters to output files
            if incl_dust != None:
                f.write('incl_dust = {}\n'.format(incl_dust))
            if incl_lines != None:
                f.write('incl_lines = {}\n'.format(incl_lines))
            f.write('nphot = {}\n'.format(nphot))
            f.write('nphot_scat = {}\n'.format(nphot_scat))
            if scattering_mode_max != None:
                f.write('scattering_mode_max = {}\n'.format(scattering_mode_max))
            f.write('istar_sphere = {}\n'.format(istar_sphere))
            f.write('setthreads = {}\n'.format(num_cpu))
            # f.write('mc_scat_maxtauabs = 5.d0\n')
            # print additional keyword parameters
            for k, v in kwargs.items():
                f.write('{} = {} \n'.format(k, v))

        # duplicate output file
        if filename != None:
            self.duplicate_file(default_filename, filename, comment = comment, timemark = self.now)

        self.filename = filename
    
    def get_continuumlambda(self, filename = None,
                                  comment = None,
                                  lambda_micron = None,
                                  append = False,
                                  silent = False):
        '''
        Preparing the wavelength file (wavelengths are in units of micron).
        This is the file that sets the discrete wavelength points for the continuum radiative transfer calculations.
        If this method is called for multiple times, by default it concatenate the wavelengths in each input,
        unless the input wavelengths do not increase or decrease monotonically. 
        In that case, it gives a warning without editing the files.
        If no lambda_micron is given, it recreates the 'wavelength_micron.inp' file using the default wavelengths.

        Format:
        nlam
        lambda[1]
        ...
        ...
        lambda[nlam]

        https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/inputoutputfiles.html#sec-wavelengths

        Parameters
        ------------------------
        filename : string
            output filename. (default: wavelength_micron.inp).
            It will still creat a file with default name,
            but will duplicate an output file with the specified name.
        comment  : string
            comment to add to the file header (default: None)
        lambda_micron : numpy array
            wavelength to calculate continuum (in units of micron).
        append : bool
            if False, remove the existing wavelength_micron.inp and ignore any information in it (default: False)

                                
        Note
        ----------------------------
        Wavelengths must be monotonically increasing/decreasing.

        Note
        ----------------------------
        Wavelength coverage must include the wavelengths at which the stellar spectra have most of their energy, 
        and at which the dust cools predominantly. This in practice means that this should go all the way from 
        0.1 micron to 1000 micron

        '''
        num_lambda = 0

        default_filename = 'wavelength_micron.inp'
        if append == False:
            os.system('rm -rf ' + default_filename)

        # creating output file using default wavelengths.
        num_input_lambda = 0
        try:
            num_input_lambda =  len(lambda_micron)
        except:
            if silent is False:
                print( 'get_continuumlambda: No input wavelength.' )
                print( 'get_continuumlambda: Re-creating {} using default wavelengths.'.format(default_filename))
            else:
                pass
            lam1,lam2,lam3,lam4 = 1e-1, 5.0e2, 5.0e3, 1.0e5
            n12, n23, n34       = 100, 100, 100
            lam12    = np.logspace(np.log10(lam1),np.log10(lam2),n12,endpoint=False)
            lam23    = np.logspace(np.log10(lam2),np.log10(lam3),n23,endpoint=False)
            lam34    = np.logspace(np.log10(lam3),np.log10(lam4),n34,endpoint=True)
            lam = np.concatenate([lam12,lam23,lam34])
            f = open(default_filename, 'w+')
            f.write('{}\n'.format(len(lam)))
            for value in lam:
                f.write('{}\n'.format(value))
            f.close()
            if filename != None:
                self.duplicate_file(default_filename, filename, comment = comment, timemark = self.now)
            self.nlam = len(lam)
            self.lam = lam
            return None

        # Using user-input wavelengths to create output files
        try:
            lambda_micron_temp = np.loadtxt(default_filename, skiprows = 0)
            num_lambda = int( lambda_micron_temp[0] )
            lambda_micron_temp = lambda_micron_temp[1:]
        except:
            print( 'get_continuumlambda: {} not exist or ignored.'.format( default_filename ) )
            print( 'get_continuumlambda: Will create a new one.')

        radmc_healthy = True
        if radmc_healthy == True:
            # sanity check (if wavelength increase/decrease monotonically)
            if (num_lambda == 1):
                gap_increment     = lambda_micron[0] - lambda_micron_temp[-1]
                if ( num_input_lambda > 1 ):
                    increment         = lambda_micron[1] - lambda_micron[0]
                    if (increment * gap_increment < 0):
                        radmc_healthy = False

            if (num_lambda > 1):
                gap_increment     = lambda_micron[0] - lambda_micron_temp[-1]
                present_increment = lambda_micron_temp[1] - lambda_micron_temp[0]
                if (present_increment * gap_increment < 0):
                    radmc_healthy = False

                if ( num_input_lambda > 1 ):
                    increment         = lambda_micron[1] - lambda_micron[0]
                    if (present_increment * increment < 0):
                        radmc_healthy = False

        # outputing wavelengths
        if radmc_healthy == False:
            print( 'get_continuumlambda: Wavelength does not increase/decrease monotonically.')
            print( 'get_continuumlambda: {} is not updated.'.format(default_filename))
            return None

        else:
            if num_lambda == 0:
                num_lambda = len(lambda_micron)
                lambda_micron_out = lambda_micron
            else:
                num_lambda = num_lambda + len(lambda_micron)
                lambda_micron_out = np.concatenate([lambda_micron_temp,lambda_micron])

            f = open(default_filename+'_temp', 'w+')
            f.write('{}\n'.format(num_lambda))
            for value in lambda_micron_out:
                f.write('{}\n'.format(value))
            f.close()

            os.system('rm -rf ' + default_filename)
            os.system('mv ' + default_filename+'_temp ' + default_filename)

            self.nlam = num_lambda
            self.lam = lambda_micron_out
        # duplicate output file
        if filename != None:
            self.duplicate_file(default_filename, filename, comment = comment, timemark = self.now)

    def write_dust_opac(self,
                        dust_type=['temp_regime_1', 'temp_regime_2', 'temp_regime_3', 'temp_regime_4'], 
                        inputstyle=20, 
                        grain_align=True):    
        '''
        Preparing the control file for dust opacity.
        '''
        self.dust_type = dust_type
        self.dust_spec = len(self.dust_type)

        if isinstance(inputstyle, str):
            if inputstyle.lower() == 'dustkappa':
                inputstyle = 10
            elif inputstyle.lower() == 'dustkapscatmat':
                if grain_align is True:
                    inputstyle = 20
                else:
                    inputstyle = 10
        self.dustkap_inputstyle = inputstyle
        '''
        1 : dustkappa_*.inp file
        10: dustkapscatmat_*.inp file without grain alignment (read Z matrix)
        20: dustkapscatmat_*.inp file with grain alignment (required dustkapalignfact_*.inp)
        '''
        with open('dustopac.inp','w+') as f:
            f.write('2                          Format number of this file\n')
            f.write(f'{self.dust_spec}                          Nr of dust species\n')
            f.write('============================================================================\n')
            for i, dust in enumerate(self.dust_type):
                f.write(f'{inputstyle}                          Way in which this dust species is read\n')
                f.write('0                          0=Thermal grain\n')
                f.write(f'{dust}                     Extension of name of dustkappa_***.inp file\n')
                f.write('============================================================================\n')
        
        for dust in dust_type:
            if inputstyle == 1:
                os.system(f'cp -r ./kappa/{dust}/dustkappa.inp ./dustkappa_{dust}.inp')
            elif inputstyle == 20 or inputstyle == 10:
                os.system(f'cp -r ./kappa/{dust}/dustkapscatmat.inp ./dustkapscatmat_{dust}.inp')
    
    def write_amr_grid(self):

        if self.grid.grid_type == 'sph':
            coordsystem = 150  # 100 <= coordsystem < 200 is spherical
        elif self.grid.grid_type == 'rect':
            coordsystem = 1
        elif self.grid.grid_type == 'cyl':
            raise NotImplementedError('Cylindrical grid is not implemented yet.')
        
        grid_info   = 0  # advised to set =0
        incl_r      = 1
        incl_theta  = 1
        incl_phi    = 1

        with open('amr_grid.inp', "w+") as f:
            f.write('1\n')
            f.write('0\n')
            f.write(str(coordsystem)+'\n')
            f.write(str(grid_info)+'\n')
            f.write('%d %d %d\n'%(incl_r, incl_theta, incl_phi))
            f.write('%d %d %d\n'%(self.grid.n1, self.grid.n2, self.grid.n3))
            for value in self.grid.grid_edge[0]:
                f.write('%13.13e\n'%(value))
            for value in self.grid.grid_edge[1]:
                f.write('%13.13e\n'%(value))
            for value in self.grid.grid_edge[2]:
                f.write('%13.13e\n'%(value))

    def write_density_file(self):

        nspec = self.dust_spec

        with open('dust_density.inp', "w+") as f:
            f.write(str(1)+'\n')
            f.write('%d\n'%(self.grid.n1*self.grid.n2*self.grid.n3))
            f.write(str(nspec)+'\n')
            for i in range(nspec):
                data = self.model.rho_dust[:, :, :, i].ravel(order='F')
                data.tofile(f, sep='\n', format="%13.6e")
                f.write('\n')
            f.write('\n')

    

    def write_temperature_file(self):

        mstar    = ms  # This is useless in the current version.
        rstar    = rs * 1
        tstar    = ts*(0.1**(1/4))*(1**(-1/2))
        pstar    = np.array([0.,0.,0.])
        with open('stars.inp','w+') as f:
            f.write('2\n')
            f.write('1 %d\n\n'%(self.nlam))
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n'%(rstar,mstar,pstar[0],pstar[1],pstar[2]))
            for value in self.lam:
                f.write('%13.6e\n'%(value))
            f.write('\n%13.6e\n'%(-tstar))

        nspec = self.dust_spec
        with open('dust_temperature.dat', "w+") as f:
            f.write('1\n')
            f.write('%d\n'%(self.grid.n1*self.grid.n2*self.grid.n3))
            f.write(str(nspec)+'\n')
            for i in range(nspec):
                data = self.model.T[:, :, :, i].ravel(order='F')
                data.tofile(f, sep='\n', format="%13.6e")
                f.write('\n')
            f.write('\n')
    
    def get_dustalignmentcontrol(self,
                                 alpha=1e-33,
                                 hourglass=False,
                                 uniform_x=False,
                                 uniform_y=False,
                                 uniform_z=False,
                                 toroidal=False):

        R_mesh, Theta_mesh, Phi_mesh = np.meshgrid(self.grid.grid_center[0], self.grid.grid_center[1], self.grid.grid_center[2], indexing='ij')

        XX = R_mesh * np.sin(Theta_mesh) * np.cos(Phi_mesh)
        YY = R_mesh * np.sin(Theta_mesh) * np.sin(Phi_mesh)
        ZZ = R_mesh * np.cos(Theta_mesh)

        alvec = np.zeros((self.grid.n1, self.grid.n2, self.grid.n3, 3))

        Bx = np.zeros_like(XX)
        By = np.zeros_like(YY)
        Bz = np.zeros_like(ZZ)

        if uniform_x is True: Bx += 1
        if uniform_y is True: By += 1
        if uniform_z is True: Bz += 1

        if hourglass is True:
            Bx += alpha * XX * ZZ * np.exp(-alpha * ZZ**2)
            By += alpha * YY * ZZ * np.exp(-alpha * ZZ**2)

        if toroidal is True:
            Bx += YY
            By += -XX


        alvec[:, :, :, 0] = Bx / np.sqrt(Bx**2 + By**2 + Bz**2)
        alvec[:, :, :, 1] = By / np.sqrt(Bx**2 + By**2 + Bz**2)
        alvec[:, :, :, 2] = Bz / np.sqrt(Bx**2 + By**2 + Bz**2)


        with open('grainalign_dir.inp', 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % (self.grid.n1*self.grid.n2*self.grid.n3))
            for ix in range(self.grid.n1):           # Nr of cells
                for iy in range(self.grid.n2):
                    for iz in range(self.grid.n3):
                        f.write('%13.6e %13.6e %13.6e\n' % (
                            alvec[ix, iy, iz, 0], alvec[ix, iy, iz, 1], alvec[ix, iy, iz, 2]))


        nlam = 101
        nang = 90

        lam = np.logspace(np.log10(1e-1), np.log10(1e5), nlam, endpoint=True)  # Wavelengths in microns
        ang = np.linspace(0, 90, nang, endpoint=True)  # Angles in degrees

        k_orth = 1 + 0.2*(1-np.cos(np.radians(ang))**2)  # Orthogonal component of kappa
        k_para = 1 - 0.2*(1-np.cos(np.radians(ang))**2)  # Parallel component of kappa

        # k_orth = np.ones(nang)  # Orthogonal component of kappa
        # k_para = 1 - np.sin(np.deg2rad(ang))  # Parallel component of kappa

        # k_orth = np.ones(nang)  # Orthogonal component of kappa
        # k_para = np.ones(nang)  # Parallel component of kappa

        for i, dust in enumerate(self.dust_type):
            with open(f'dustkapalignfact_{dust}.inp','w+') as f:
                f.write('1\n')
                f.write('%d\n'%(nlam))
                f.write('%d\n\n'%(nang))
                for value in lam:
                    f.write('%13.6e\n'%(value))
                f.write('\n')
                for value in ang:
                    f.write('%13.6e\n'%(value))
                f.write('\n')
                for inu in range(nlam):
                    for imu in range(nang):
                        f.write('%13.6e %13.6e\n'%(k_orth[imu],k_para[imu]))
                    f.write('\n')