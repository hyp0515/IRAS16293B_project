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
model.get_diskcontrol(  d_to_g_ratio    = 0.01,
                        a_max           = 1, # mm
                        a_max_outer     = 0.01,  # mm
                        Mass_of_star    = 0.5, # Msun
                        Accretion_rate  = 1e-5, # Msun/yr
                        Radius_of_disk  = 35,   # AU
                        Q               = 0.5, # Toomre Q
                        NR    =200,
                        NTheta=200,
                        NPhi  =100,
                        )
model.write_dust_opac(inputstyle=20, grain_align=True)
model.get_heatcontrol(L_star=.1, # Lsun
                    R_star=1,
                    heat='accretion') # radiation/accretion

