import numpy as np
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

class radmc3d_setup:
    """
    A class to initialize the radmc3d simulation.

    Examples
    --------
    Simplest usage (defaulting everything):

    .. code-block:: python

        test = radmc3d_setup()
        test.get_mastercontrol()
        test.get_continuumlambda()

    More complicated cases:

    .. code-block:: python

        test = radmc3d_setup(silent=False)

        test.get_mastercontrol(filename='radmctest.inp',
                               comment='this is a test',
                               incl_dust=1)

        test.get_linecontrol(filename='lines_test.inp',
                             comment='this is a test',
                             line1='ch3oh leiden 0 0',
                             line2='co leiden 0 0')

        lam1, lam2, lam3, lam4 = 0.1e0, 1.0e2, 5.0e3, 1.0e4
        n12, n23, n34 = 100, 100, 50
        lam12 = np.logspace(np.log10(lam1), np.log10(lam2), n12, endpoint=False)
        lam23 = np.logspace(np.log10(lam2), np.log10(lam3), n23, endpoint=False)
        lam34 = np.logspace(np.log10(lam3), np.log10(lam4), n34, endpoint=True)
        lams = [lam12, lam23, lam34]

        for i in range(len(lams)):
            append = (i != 0)
            test.get_continuumlambda(filename='wavelength_micron_test.inp',
                                     comment='this is a test',
                                     lambda_micron=lams[i], append=append)
    """

    def __init__(self, silent = True):
      '''
      Keywords:
        silent (True/False): if True, provide more informations (default: False).

      Properties:
        self.now (str): the time this class is created
        self.numcpu (int): number of threads on this computer


      '''

      # obtain the time this class is created
      self.now = strftime("%Y-%m-%d %H:%M:%S", gmtime())

      # counting number of processors
      self.num_cpu = cpu_count()
      if silent == False:
        print("You have {0} Processors".format(self.num_cpu))

    def __del__(self):
      pass



    ##### methods in this class #########################
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

    def get_linecontrol(self, filename = None,
                               comment = None,
                              **kwargs):
      '''
      Preparing the control file for lines.

      Input :
      
      filename (string) : output filename. (default: lines.inp).
                          It will still creat a file with default name,
                          but will duplicate an output file with the specified name.
      See example for how to include lines

      '''
      # setting variables 
      num_lines = len( kwargs.items() )

      # writing variables to output file
      default_filename = 'lines.inp'
      with open(default_filename, 'w+') as f:
        f.write('2\n')
        f.write('{}\n'.format(num_lines))
        for k, v in kwargs.items():
          f.write('{}\n'.format(v))

      # duplicate output file
      if filename != None:
        self.duplicate_file(default_filename, filename, comment = comment, timemark = self.now)

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
        lam1,lam2,lam3,lam4 = 0.1e0, 1.0e2, 5.0e3, 1.0e4
        n12, n23, n34       = 100, 100, 50
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
    
    
    
    def get_diskcontrol(self, 
                        d_to_g_ratio = 0.01,
                               a_max = None,
                                   q = -3.5,
                        Mass_of_star = None,
                      Accretion_rate = None,
                      Radius_of_disk = None,
                                   Q = 1.5,
                                  NR = None,
                              NTheta = None,
                                NPhi = None,
                 generate_table_only = False

                      ):
      '''
      Preparing the control file for disk model.

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
          Resolution in ohi axis

      '''
      if a_max is None:                   a_max = 0.1  # 100 um
      self.amax = a_max
      # CB68's properties
      if Mass_of_star is None:     Mass_of_star = 0.14
      if Accretion_rate is None: Accretion_rate = 5e-7
      if Radius_of_disk is None: Radius_of_disk = 50
        
      # grid's resolution
      if NR is None:                         NR = 200
      if NTheta is None:                 NTheta = 200
      if NPhi is None:                     NPhi = 10
       
      # Disk model
      
      self.dust_to_gas_ratio = d_to_g_ratio
      # note: the original a_max is in cm
      # a_min is set to be 0.05 um = 5e-6 cm
      self.opacity_table  = generate_opacity_table_opt(a_min=1e-6, a_max=a_max*0.1,
                                                   q=q, dust_to_gas=d_to_g_ratio)
      
      self.disk_property_table = generate_disk_property_table(self.opacity_table)
      if generate_table_only is True:
        return
      
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
      self.DM = DM
      
      self.NR    = DM.NR
      self.NTheta = DM.NTheta
      self.NPhi  = DM.NPhi
      

      
      # f = open(self.filename, 'r+')
      # content = f.read()
      # f.seek(0,0)
      # f.write(f'# a_max = {a_max} mm \n')
      # f.write(f'# Mstar = {Mass_of_star} Msun \n')
      # f.write(f'# Mdot = {Accretion_rate} Msun/yr \n')
      # f.write(f'# Rd = {Radius_of_disk} AU \n')
      # f.write(content)
      # f.close()
      
      self.write_amr_grid()
      self.write_dust_opac()
      self.write_dust_density()
    
    def write_amr_grid(self):
        
      coordsystem = 150  # 100 <= coordsystem < 200 is spherical
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
        f.write('%d %d %d\n'%(self.NR, self.NTheta, self.NPhi))
        for value in self.DM.r_sph_grid:
          f.write('%13.13e\n'%(value))
        for value in self.DM.theta_sph_grid:
          f.write('%13.13e\n'%(value))
        for value in self.DM.phi_sph_grid:
          f.write('%13.13e\n'%(value))
    
    
    def write_dust_opac(self):    
      '''
      Preparing the control file for dust opacity.
      '''
      self.dust_spec = 4
      with open('dustopac.inp','w+') as f:
        f.write('2                          Format number of this file\n')
        f.write('4                          Nr of dust species\n')
        f.write('============================================================================\n')
        f.write('1                          Way in which this dust species is read\n')
        f.write('0                          0=Thermal grain\n')
        f.write('temp_1                     Extension of name of dustkappa_***.inp file\n')
        f.write('============================================================================\n')
        f.write('1                          Way in which this dust species is read\n')
        f.write('0                          0=Thermal grain\n')
        f.write('temp_2                     Extension of name of dustkappa_***.inp file\n')
        f.write('============================================================================\n')
        f.write('1                          Way in which this dust species is read\n')
        f.write('0                          0=Thermal grain\n')
        f.write('temp_3                     Extension of name of dustkappa_***.inp file\n')
        f.write('============================================================================\n')
        f.write('1                          Way in which this dust species is read\n')
        f.write('0                          0=Thermal grain\n')
        f.write('temp_4                     Extension of name of dustkappa_***.inp file\n')
        f.write('============================================================================\n')
        
    def write_dust_density(self):
      '''
      Preparing the control file for dust density.
      '''
      nspec     = self.dust_spec
      T_crit = [150, 425, 680, 1200]
      rho_dust = []
      for i, t_crit in enumerate(T_crit):
        if i == 0:
          rho_dust.append(
            np.where(
              self.DM.T_sph <= t_crit,
              self.DM.rho_sph,
              1e-20
            )
          )
        else:
          density_dust = np.where(
              self.DM.T_sph <= t_crit,
              self.DM.rho_sph,
              1e-20
            )
          density_dust = np.where(
              T_crit[i-1]<self.DM.T_sph ,
              density_dust,
              1e-20
            )
          rho_dust.append(density_dust)
        
        
      self.rho_dust = self.dust_to_gas_ratio * np.array(rho_dust)
      self.rho_gas = self.DM.rho_sph
      self.mask = self.rho_gas < 1e-18

      with open('dust_density.inp', "w+") as f:
        f.write(str(1)+'\n')
        f.write('%d\n'%(self.NR*self.NTheta*self.NPhi))
        f.write(str(nspec)+'\n')
        for i in range(nspec):
          data = self.rho_dust[i, :, :, :].ravel(order='F')
          data.tofile(f, sep='\n', format="%13.6e")
          f.write('\n')
        f.write('\n')
    
    def get_vfieldcontrol(self, Kep = True,
                            vinfall = None,
                                Rcb = None,
                            outflow = None
                        ):
      '''
      Preparing the control file for velocity field.

      Parameters
      ----------------------------
      Kep : bool
          Keplerian azimuthal velocity field
      vinfall : float
          Infall velocity (unit: Keplerian velocity) e.g., 1 for 1 Keplerian velocity of infall direction
      Rcb : float
          Centrifugal barrier (unit: AU)
      outflow : float
          Outflow velocity (unit: Keplerian velocity)
      '''
      if vinfall is None: vinfall = 0
      
      self.rcb = Rcb
      if Rcb is None:
        vr        = -vinfall*np.sqrt(G*self.Mstar*Msun/(self.DM.r_sph*au)) # Infall velocity
        vtheta    = 0  # No vertical movement in this version
        if Kep is True:
          vphi    = np.sqrt(G*self.Mstar*Msun/(self.DM.r_sph*au))    # Keplerian velocity
        elif Kep is False:
          vphi    = np.zeros(vr.shape)
      elif Rcb is not None:  # velocity field in Oya et al. 2014
        rcb_idx   = np.searchsorted(self.DM.r_sph, Rcb)
        if outflow is None:
          vr_inside    = np.zeros(self.DM.r_sph[:rcb_idx].shape)  # no infall compoonent inside the centrifugal barrier
        elif outflow is not None:
          vr_inside    = +outflow*np.sqrt(G*self.Mstar*Msun/(self.DM.r_sph[:rcb_idx]*au))  # if expanding
        vr_infall = -vinfall*np.sqrt(G*self.Mstar*Msun/(self.DM.r_sph[rcb_idx:]*au))
        vr        = np.concatenate((vr_inside, vr_infall))
        vtheta    = 0
        vphi      = np.sqrt(G*self.Mstar*Msun/(self.DM.r_sph*au))
        
      with open('gas_velocity.inp','w+') as f:
        f.write('1\n')
        f.write('%d\n'%(self.NR*self.NTheta*self.NPhi))
        for idx_phi in range(self.NPhi):
          for idx_theta in range(self.NTheta):
            for idx_r in range(self.NR):
              f.write('%13.6e %13.6e %13.6e \n'%(vr[idx_r],vtheta,vphi[idx_r]))
            
    def get_heatcontrol(self, L_star = None,
                              R_star = None,
                           accretion = True,
                         irradiation = True,
                            **kwargs
                      ):
      '''
      Preparing the control file for heating mechanism.
      
      Parameters
      ----------------------
      accretion : bool
          Accretion heating mechanism due to release of gravitational energy
      irradiation : bool
          Irradiation heating from central protostar
      L_star : float
          Luminosity of central protostar (unit: L_sun)
      
      Note
      ----------------------
      If both heating mechanisms are True, combine accretion and irradiation temperature map.
      
      kwargs
      ----------------------
      heat        : 'accretion'/'radiation'/'combine'
          Directly assigning heating mechanism by words 
      '''

      if L_star is None: L_star = 0.89
      if R_star is None: R_star = 1
      # Star parameters and Write the star.inp file
      mstar    = ms  # This is useless in the current version.
      rstar    = rs * R_star
      tstar    = ts*(L_star**(1/4))*(R_star**(-1/2))
      pstar    = np.array([0.,0.,0.])
      with open('stars.inp','w+') as f:
        f.write('2\n')
        f.write('1 %d\n\n'%(self.nlam))
        f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n'%(rstar,mstar,pstar[0],pstar[1],pstar[2]))
        for value in self.lam:
          f.write('%13.6e\n'%(value))
        f.write('\n%13.6e\n'%(-tstar))
      
      
      if 'heat' in kwargs:
        heat_type = kwargs['heat'].lower()
        if heat_type == 'accretion':
          accretion   = True
          irradiation = False
        elif heat_type == 'irradiation' or heat_type == 'radiation':
          accretion   = False
          irradiation = True
        elif heat_type == 'combine':
          accretion   = True
          irradiation = True
      
      # Heating mechanism
      if irradiation is True:# Irradiation heating calculated by RADMC-3D
        if accretion is False:
          os.system('radmc3d mctherm > /dev/null 2>&1')
          # os.system('radmc3d mctherm')
          with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            d = readData(dtemp=True, ddens=True)
          T = np.where(d.dusttemp<20, 20, d.dusttemp)
            
        elif accretion is True:  # Combination of two heating mechanisms, irradiation and accretion

          T_acc = np.tile(self.DM.T_sph[:, :, :, np.newaxis], (1, 1, 1, self.dust_spec))
          
          os.system('radmc3d mctherm > /dev/null 2>&1')
          d = readData(dtemp=True)
          T_irr = d.dusttemp
          
          T = (T_irr**4+T_acc**4)**(1/4)
          T = np.where(T<20, 20, T)
        
        T[self.mask,:]=20
        with open('dust_temperature.dat', "w+") as f:
          f.write('1\n')
          f.write('%d\n'%(self.NR*self.NTheta*self.NPhi))
          f.write(str(self.dust_spec)+'\n')
          for i in range(self.dust_spec):
            data = T[:, :, :, i].ravel(order='F')
            data.tofile(f, sep='\n', format="%13.6e")
            f.write('\n')
          f.write('\n')
            
        self.T_avg = np.sum(T, axis=3)/self.dust_spec  # averaging temperatures of four dust species
        with open('gas_temperature.inp', "w+") as f:
          f.write('1\n')
          f.write('%d\n'%(self.NR*self.NTheta*self.NPhi))
          f.write(str(self.dust_spec)+'\n')
          data = self.T_avg.ravel(order='F')
          data.tofile(f, sep='\n', format="%13.6e")
          f.write('\n')

      elif irradiation is False:  # Accretion heating calculated by Wenrui's Disk Model 

        T = np.tile(self.DM.T_sph[np.newaxis:, :, :], (self.dust_spec, 1, 1, 1))

          
        with open('dust_temperature.dat', "w+") as f:
          f.write('1\n')
          f.write('%d\n'%(self.NR*self.NTheta*self.NPhi))
          f.write(str(self.dust_spec)+'\n')
          for i in range(self.dust_spec):
            data = T[i, :, :, :].ravel(order='F')
            data.tofile(f, sep='\n', format="%13.6e")
            f.write('\n')
          f.write('\n')
            
        self.T_avg = np.sum(T, axis=0)/self.dust_spec
        with open('gas_temperature.inp', "w+") as f:
          f.write('1\n')
          f.write('%d\n'%(self.NR*self.NTheta*self.NPhi))
          f.write(str(1)+'\n')
          data = self.DM.T_sph.ravel(order='F')
          data.tofile(f, sep='\n', format="%13.6e")
          f.write('\n')
      self.T_dust = T
    
    def get_gasdensitycontrol(self, 
                                 abundance   = 1e-10,
                                 snowline    = None,
                                 enhancement = 1e5,
                              gas_inside_rcb = True,
                              **kwargs
                            ):
      '''
      Preparing the control file for gas density.

      Parameters
      ----------------------------
      abundance : float
          Abundance of the simulated gas molecule compared with hydrogen
      snowline : float
          At which temperature causes desorption from dust grains (unit:K)
          (if None, there is no abundance enhancement)
      enhancement : float
          How much the abundance is enhanced inside snowline
      gas_inside_rcb : bool
          Whether gas is absent inside centrifugal barrier
      '''
      
      if self.rcb is None:  
        if snowline is not None:
          abunch3oh = np.where(self.T_avg < snowline,
                               abundance,
                               abundance * enhancement
                               )
          """
          This is over simplified to determine how the abundance is enhanced.
          Realistic chemical configuration is required.
          """
          rho_gas   = self.rho_gas
          
        elif snowline is None:
          abunch3oh = abundance
          rho_gas   = self.rho_gas
          
      elif self.rcb is not None:  # The main assumption of Oya's velocity field is that there is no gas inside the centrifugal barrier.
        rcb_idx = np.searchsorted(self.DM.r_sph, self.rcb)
        if snowline is not None:
          abunch3oh = np.where(self.T_avg < snowline,
                               abundance,
                               abundance * enhancement
                               )
          rho_gas   = self.rho_gas
          
          if gas_inside_rcb is False:
            rho_gas[:rcb_idx, :, :] = 1e-20
            
        elif snowline is None:
          abunch3oh = abundance
          rho_gas   = self.rho_gas
            
          if gas_inside_rcb is False:
            rho_gas[:rcb_idx, :, :] = 1e-20
            
      factch3oh = abunch3oh/(2.3*mp)
      nch3oh    = rho_gas*factch3oh
      
      with open('numberdens_ch3oh.inp','w+') as f:
        f.write('1\n')                       # Format number
        f.write('%d\n'%(self.NR*self.NTheta*self.NPhi))           # Nr of cells
        data = nch3oh.ravel(order='F')          # Create a 1-D view, fortran-style indexing
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')
        
      
    
    def duplicate_file(self, default_filename, filename, comment = None, timemark = None):
      '''
      A small function to duplicate the .inp files.

      Parameters
      -----------------------
      default_filename : str
          Default output file name
      filename         : str
          Filename of the duplication
      comment          : str
          If present, include it as header in the duplicated output file.
      timemark         : str
          If present, include it as header in the duplicated output file.

      '''
      os.system('rm -rf {}'.format(filename))
      os.system('cp -r {} {}'.format(default_filename, filename))

      f = open(filename, 'r+')
      content = f.read()
      f.seek(0,0)
      if timemark != None:
        f.write('# created: {} \n'.format(timemark) )
      if comment != None:
        f.write('# {} \n'.format(comment) )
      f.write(content)
      f.close()