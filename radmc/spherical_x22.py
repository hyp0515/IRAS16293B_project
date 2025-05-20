import numpy as np
from X22_model.disk_model import *
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar
# All unit are in cgs except height and radius are in au
#########################################################################################

class DiskModel_spherical:

    def __init__(self, opacity_table, disk_property_table):
        self.DM_horizontal = DiskModel(opacity_table, disk_property_table)
        return
    
    def input_disk_parameter(self, Mstar, Mdot, Rd, Q, N_R=500):
        """
        Run Wenrui's radial profile to get initial r-dependent profiles.

        Parameters
        -----------------
        Mstar     : float
            Mass of protostar
        Mdot      : float
            Rate of mass infall from the envelope onto the disk
        Rd        : float
            Radius of the disk
        Q         : float
            Toomre index
        N_R       : float
            Resolution of radius grid (Default = 500)
        """
        self.Mstar = Mstar
        self.Mdot  = Mdot
        self.Rd    = Rd        
        self.NR    = N_R

        self.DM_horizontal.generate_disk_profile(Mstar=self.Mstar, Mdot=self.Mdot, Rd=self.Rd, Q=Q, N_R=self.NR)
        self.R_grid    = self.DM_horizontal.R[1:]/au  # radial grid
        self.T_mid     = self.DM_horizontal.T_mid  # temperature at midplane(z=0)
        self.T_eff     = self.DM_horizontal.T_eff  # effective temperature extracted from Wenrui's disk_model
        self.M         = self.DM_horizontal.Sigma/2  # Sigma/2 is the definition of M in Hubeny+90
        self.tau_r_mid = self.DM_horizontal.tau_r_mid
        self.tau_p_mid = self.DM_horizontal.tau_p_mid
        
        Z_max = self.Rd
        self.NZ = self.NR
        Z_grid = np.append(np.logspace(np.log10(Z_max/au), np.log10(0.001), self.NZ-1), 0)
        self.Z_grid = Z_grid[::-1]
        self.NZ = len(self.Z_grid)
        
        self.mask_index = np.argmax(self.M > 0)
        self.make_position_map()
             
    def make_position_map(self):
        """
        Make grid of position in cylindrical coordinates
        """
        R, Z = np.meshgrid(self.R_grid, self.Z_grid, indexing = 'ij')
        pos_map = np.dstack((R, Z))
        self.pos_map = pos_map
        self.precompute_property()

    
    def precompute_property(self, miu=2.3, ionized_factor=1):
        """
        To calculate the gas pressure scale height
        miu   : mean molecular weight (Default = 2.3)
        ionized_factor: N/(N-n_e) (ionized factor; Default = 1)
        """
        def cg(miu, factor,  T):  # (5.4) sound speed assiciated with the gas pressure
            """
            miu   : mean molecular weight
            factor: N/(N-n_e)
            T     : temperature

            "In the initialization step, c_g is determined by eq(5.4) with T = T_eff, 
            and with some suitable estimate for N/(N-n_e)"    
            """
            return (kB*T*factor/(miu*mp))**0.5       

        def H_g(cg, effective_gravity):  # (4.4) gas pressure scale height
            H_g = (2*cg**2/effective_gravity)**0.5
            return H_g/au        
        
        sound_speed       = cg(miu, ionized_factor, self.T_eff)
        effective_gravity = G*self.Mstar*(self.R_grid*au)**(-3)  # (2.2) effective vertical gravity
        self.H_g          = H_g(sound_speed, effective_gravity)
        self.make_rho_and_m_map()

        
    def make_rho_and_m_map(self):
        """
        Calculate volume density and mass-depth scale
        """
        M       = self.M * 1
        h_grid  = np.empty((self.NR))
        rho_map = np.empty((self.NR, self.NZ))
        m_map   = np.empty((self.NR, self.NZ))
        
        def mass_error(h, z_grid, M_r, h_g):
            rho_0    = M_r / h
            rho_grid = rho_0 * np.exp(-(z_grid**2 / h_g**2))
            rho_grid = np.maximum(rho_grid, 1e-10)
            dz       = np.diff(z_grid, prepend=z_grid[0])
            m_grid   = np.cumsum(rho_grid * dz)
            return np.abs(1 - m_grid[-1] / M_r)


        for r in range(self.NR):
            if r >= self.mask_index:
                z_grid = self.Z_grid
                h_g = self.H_g[r]
                '''
                Using scipy.optimize.minimize_scalar to numerically find h to prevent loops
                '''
                result = minimize_scalar(mass_error, bounds=(0.01, 100), args=(z_grid, M[r], h_g), method='bounded')
                h = result.x

                rho_0    = M[r] / h
                rho_grid = rho_0 * np.exp(-(z_grid**2 / h_g**2))
                rho_grid = np.maximum(rho_grid, 1e-7)
                dz       = np.diff(z_grid, prepend=z_grid[0])
                m_grid   = np.cumsum(rho_grid * dz)

                h_grid[r]     = h
                rho_map[r, :] = rho_grid/au
                m_map[r, :]   = m_grid[::-1]
            else:
                h_grid[r]     = 0
                rho_map[r, :] = 1e-20 * np.ones((len(self.Z_grid)))
                m_map[r, :]   = 1e-20 * np.ones((len(self.Z_grid)))
        
        self.rho_map = np.maximum(rho_map, 1e-20)
        self.m_map   = m_map
        self.H       = h_grid
        self.make_tau_and_T_map()

    def make_tau_and_T_map(self):
        """
        Using disk_property_table to calculate kappa_r from T, and further calculating tau_r and T
        """
        # t_start = time.time()
        def get_kappa_from_T(T):  # 2(d)
            # using the functions below can avoid using saha equation to calculate n_e, the step described on 2(d) hubney 1990
            get_kappa_r = self.DM_horizontal.get_kappa_r  # interpolating function extracted from Wenrui's Disk_Model class
            kappa_r     = get_kappa_r(T)
            return kappa_r
        
        def get_tau_from_kappa(m_grid, kappa_r):  # 2(a)
            dm        = m_grid.copy()
            dm[:, 1:] = m_grid[:, 1:]-m_grid[:, :-1]
            dm        = dm[:, :len(kappa_r[0, :])]
            dtau_r    = dm*kappa_r  # (3.4)
            # The upper and lower bound are m and 0 in m-variable,
            # meaning that the upper and lower bound in z-variable are z and infinity.
            tau_r = np.cumsum(dtau_r, axis=1)[:, -1]
            return tau_r
        
        def get_T_from_tau(tau, T_eff):  # 2(b)
            tau_p = self.tau_p_mid[self.mask_index:]*2
            tau_r = self.tau_r_mid[self.mask_index:]*2
            T = ((T_eff**4)*(3/4)*(tau*(1-tau/tau_r)+(1/np.sqrt(3))+(1/(1.5*tau_p))))**(1/4)  # (3) from Wenrui+22
            return T
        
        
        t_eff_map   = np.tile(self.T_eff[self.mask_index:, np.newaxis], (1, self.NZ))
        m_grid      = self.m_map[self.mask_index:, ::-1]
        kappa_r_map = np.empty((self.NR-self.mask_index, self.NZ))
        T_map       = np.ones((self.NR-self.mask_index, self.NZ))  # np.ones is to prevent RuntimeWarning
        tau_r_map   = np.empty((self.NR-self.mask_index, self.NZ))
        

        for z in range(self.NZ):

            if z == 0:  # initialize kappa_r_grid
                T_old = 20 * np.ones((self.NR-self.mask_index))  # the lowest temperature in Wenrui's code
            else:
                T_old = T_map[:, -1]

            # iterate until T_new = T_old
            for _ in range(50):
                kappa_r_map[:, z] = get_kappa_from_T(T_old)
                # print(m_grid[:, :(z+1)].shape, kappa_r_map[:, :(z+1)].shape)
                tau_r = get_tau_from_kappa(m_grid[:, :(z+1)], kappa_r_map[:, :(z+1)])
                T_new = get_T_from_tau(tau_r, t_eff_map[:, z])
                if np.max(np.abs((T_old-T_new)/T_old)) < 1e-10:
                    break
                # print(np.min(T_new)) 
                T_old = T_new
            
            T_map[:, z] = T_new
            tau_r_map[:, z] = tau_r
            kappa_r_map[:, z] = get_kappa_from_T(T_new)
            
        T_map = T_map[:, ::-1]
        # tau_r_map = tau_r_map[:, ::-1]
        # kappa_r_map = kappa_r_map[:, 1:]
        # kappa_r_map = kappa_r_map[:, ::-1]
        
        T_map = np.concatenate((20*np.ones((self.mask_index, self.NZ)), T_map), axis=0)
        T_map = np.where(self.rho_map<=1e-20,
                         20,
                         T_map)
        self.T_map = np.maximum(T_map, 20)

    def extend_to_spherical(self, NTheta, NPhi, theta_min_deg=20):

        """
        Extend the disk model to spherical coordinates

        Parameters
        ------------------
        NTheta          : float
            The number of theta grid
        NPhi            : float
            The number of phi grid
        theta_min_deg   : float
            The minimum angle of theta grid (Default = 30 degree)

        """
        self.NTheta = NTheta # since theta is symmetric, we only need half of the grid
        self.NPhi = NPhi  

        pos_map = self.pos_map.copy()        
        r_map = np.sqrt(pos_map[:, :, 0]**2+ pos_map[:, :, 1]**2) # distance map of every points
        r_min = np.min(r_map)
        self.rmin = r_min
        r_grid = np.logspace(np.log10(r_min), np.log10(self.Rd/au), self.NR)
        self.r_sph = r_grid

        theta_map = np.arccos(pos_map[:, :, 1]/r_map)
        theta_min = np.max([np.deg2rad(theta_min_deg), 
                           np.pi/2-3*np.arctan(self.H[-1]/self.R_grid[-1])]) # the starting angle of theta
        theta_grid = np.logspace(np.log10(theta_min), np.log10(np.max(theta_map)), NTheta//2)  
        theta_grid = -1*theta_grid + 0.5*np.pi + theta_min
        theta_grid = theta_grid[::-1] - 1e-5 # 1e-4 is to meet the RADMC's requirement that the theta boundary must contain pi/2
        theta_grid_down = -theta_grid[::-1] + np.pi + 1e-5
        self.theta_sph = np.concatenate((theta_grid, theta_grid_down))     
    
        """
        using interpolation to project data from cylindrical coordinates to 
        spherical coordinates (This alg is about to be modified for better efficiency) 
        """
        z_sph_in_cyl = r_grid[:, np.newaxis] * np.cos(theta_grid) # decreasing when theta increases
        r_sph_in_cyl = r_grid[:, np.newaxis] * np.sin(theta_grid) # increasing when theta increases
        '''
        sph_in_cyl[i, j] : 2D array
        i                : index of r_sph
        j                : index of theta_sph
        '''
        # def interpolate(data_map):  
        #     interpolator = interp2d(self.Z_grid, self.R_grid, data_map, kind='linear')
        #     interpolated_data = np.empty((self.NR, self.NTheta//2))
        #     for r in range(self.NR):
        #         for theta in range(self.NTheta//2): 
        #             interpolated_data[r, theta] = interpolator(z_sph_in_cyl[r, theta], r_sph_in_cyl[r, theta])            
        #     return interpolated_data
        def interpolate(data_map):  
            interpolator = RegularGridInterpolator((self.Z_grid, self.R_grid), data_map.T, bounds_error=False, fill_value=None)
            interpolated_data = np.empty((self.NR, self.NTheta//2))
            for r in range(self.NR):
                for theta in range(self.NTheta//2):
                    point = [z_sph_in_cyl[r, theta], r_sph_in_cyl[r, theta]]
                    interpolated_data[r, theta] = interpolator(point)          
            return interpolated_data     
        def mirror_with_r_plane(map):
            map_mirror = np.fliplr(map)            
            return np.concatenate((map, map_mirror), axis= 1)        
        def rotate_around_theta_axis(map):
            map_3d = np.tile(map[:, :, np.newaxis], (1, 1, self.NPhi))            
            return map_3d
            
        rho_sph_2d = interpolate(self.rho_map)
        T_sph_2d = interpolate(self.T_map)        
        self.rho_sph = rotate_around_theta_axis(mirror_with_r_plane(rho_sph_2d)) 
        self.T_sph = rotate_around_theta_axis(mirror_with_r_plane(T_sph_2d))
        self.make_spherical_grid()

    def make_spherical_grid(self):
        """
        Making the grid which RADMC3D adopts
        """
        def make_boundary(grid):
            grid = grid.copy()
            difference = (grid[1:] - grid[:-1])/2
            difference = np.append(difference, difference[-1])
            grid = grid - difference
            return np.append(grid, grid[-1] + 2*difference[-1])
        self.r_sph_grid = make_boundary(self.r_sph)*au
        theta_sph_grid_top = make_boundary(self.theta_sph[:self.NTheta//2])
        # theta_sph_grid_top[self.NTheta//2] = np.pi/2
        theta_sph_grid_bottom = make_boundary(self.theta_sph[self.NTheta//2:])
        theta_sph_grid = np.concatenate((theta_sph_grid_top, theta_sph_grid_bottom))
        theta_sph_grid = np.delete(theta_sph_grid, [self.NTheta//2, self.NTheta//2])
        theta_sph_grid[self.NTheta//2] = np.pi/2
        self.theta_sph_grid = theta_sph_grid
        self.phi_sph_grid = np.linspace(0, 2*np.pi, self.NPhi+1)