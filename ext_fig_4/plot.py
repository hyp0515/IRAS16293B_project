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
    # os.system('radmc3d mctherm')
    setup.get_dustalignmentcontrol(alpha=1/(10*au*au), 
                                    hourglass=True, 
                                    uniform_z=True, 
                                    uniform_x=False, 
                                    uniform_y=False,
                                    toroidal=False)

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