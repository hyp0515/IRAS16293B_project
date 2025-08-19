import numpy as np
from radmc3dPy.analyze import * 
import sys
sys.path.append('../..')
from radmc.setup import *
from radmc.simulate import Simulation

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
Initialize observation (continuum and polarization)
"""
distance = 140
crop_sizeau = 150

obs_wav = np.array([
    1300, 
    3000, 
    7000, 
    18000
])


"""
Fig. 4
"""
incl = 45
ain = 1.2
amid = 0.8
aout = 0.08
mdot= 4.5e-5
rd = 35
Q = 0.4

setup_model(ain, amid, aout, mdot, rd, Q, align=True)
simulation = Simulation(save_out=True, save_npz=False)
simulate_mutual_parms = {
    "incl"      : incl,
    "npix"      : 500,
    "sizeau"    : crop_sizeau,
    "posang"    : 90,
    "phi"       : 0,
    "dir"       : f'/home/hyp0515/simulation/fig_4/ain_{ain}_amid_{amid}_aout_{aout}_Mdot_{mdot}_Q_{Q}_incl_{incl}_rd_{rd}/',
}
simulation.generate_continuum(
    scat=True,
    stokes=True,
    read_lambda=obs_wav*1e-3,
    load_simulation=False,
    fname=f'conti',
    **simulate_mutual_parms
)


"""
Ext. Fig. 1
"""
incl = 45
ain = 1.2
amid = 0.8
aout = 0.08
mdot= 4.5e-5
rd = 35

for Q in [0.3, 0.5, 0.8]:
    setup_model(ain, amid, aout, mdot, rd, Q, align=True)
    simulation = Simulation(save_out=True, save_npz=False)
    simulate_mutual_parms = {
        "incl"      : incl,
        "npix"      : 500,
        "sizeau"    : crop_sizeau,
        "posang"    : 0,
        "phi"       : 0,
        "dir"       : f'/home/hyp0515/simulation/ext_fig_1/ain_{ain}_amid_{amid}_aout_{aout}_Mdot_{mdot}_Q_{Q}_incl_{incl}_rd_{rd}/',
    }
    simulation.generate_continuum(
        scat=True,
        stokes=True,
        read_lambda=obs_wav*1e-3,
        load_simulation=False,
        fname=f'conti',
        **simulate_mutual_parms
    )
    simulation.generate_sed(
        scat=True,
        read_lambda=obs_wav*1e-3,
        load_simulation=False,
        fname='sed',
        **simulate_mutual_parms
    )


"""
Ext. Fig. 2 
"""

# Left panel
incl = 45
ain = 1.2
amid = 0.8
aout = 0.08
mdot= 4.5e-5
rd = 35
lambda_to_simulate = np.array([8.7e2, 1.3e3, 2e3, 3e3, 6.8e3, 9e3, 1.3e4, 2e4, 3e4, 3.75e4, 5e4, 7.5e4])*1e-3
for Q in [0.3, 0.5, 0.8, 1.0, 1.5]:
    setup_model(ain, amid, aout, mdot, rd, Q, align=True)
    simulation = Simulation(save_out=True, save_npz=False)
    simulate_mutual_parms = {
        "incl"      : incl,
        "npix"      : 500,
        "sizeau"    : crop_sizeau,
        "posang"    : 0,
        "phi"       : 0,
        "dir"       : f'/home/hyp0515/simulation/ext_fig_2/ain_{ain}_amid_{amid}_aout_{aout}_Mdot_{mdot}_Q_{Q}_incl_{incl}_rd_{rd}/',
    }
    simulation.generate_sed(
        scat=True,
        read_lambda=lambda_to_simulate,
        load_simulation=False,
        fname='sed',
        **simulate_mutual_parms
    )


# Middle panel
incl = 45
ain = 1.2
amid = 0.8
aout = 0.08
Q = 0.4
rd = 35
for mdot in [5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
    setup_model(ain, amid, aout, mdot, rd, Q, align=True)
    simulation = Simulation(save_out=True, save_npz=False)
    simulate_mutual_parms = {
        "incl"      : incl,
        "npix"      : 500,
        "sizeau"    : crop_sizeau,
        "posang"    : 0,
        "phi"       : 0,
        "dir"       : f'/home/hyp0515/simulation/ext_fig_2/ain_{ain}_amid_{amid}_aout_{aout}_Mdot_{mdot}_Q_{Q}_incl_{incl}_rd_{rd}/',
    }
    simulation.generate_sed(
        scat=True,
        read_lambda=lambda_to_simulate,
        load_simulation=False,
        fname='sed',
        **simulate_mutual_parms
    )

# Right panel
incl = 45
mdot= 4.5e-5
Q = 0.4
rd = 35
for a in [10, 1, 0.7, 0.5, 0.3, 0.1, 0.01]:
    ain = a
    amid = a
    aout = a
    setup_model(ain, amid, aout, mdot, rd, Q, align=True)
    simulation = Simulation(save_out=True, save_npz=False)
    simulate_mutual_parms = {
        "incl"      : incl,
        "npix"      : 500,
        "sizeau"    : crop_sizeau,
        "posang"    : 0,
        "phi"       : 0,
        "dir"       : f'/home/hyp0515/simulation/ext_fig_2/ain_{ain}_amid_{amid}_aout_{aout}_Mdot_{mdot}_Q_{Q}_incl_{incl}_rd_{rd}/',
    }
    simulation.generate_sed(
        scat=True,
        read_lambda=lambda_to_simulate,
        load_simulation=False,
        fname='sed',
        **simulate_mutual_parms
    )

"""
Ext. Fig. 3
"""
incl = 45
mdot= 4.5e-5
rd = 35
Q = 0.4
for a in [10, 50, 100, 200, 400, 800, 1600, 3200, 6400]:
    ain = a * 1e-3
    amid = ain
    aout = ain
    setup_model(ain, amid, aout, mdot, rd, Q, align=True)
    simulation = Simulation(save_out=True, save_npz=False)
    simulate_mutual_parms = {
        "incl"      : incl,
        "npix"      : 500,
        "sizeau"    : crop_sizeau,
        "posang"    : 0,
        "phi"       : 0,
        "dir"       : f'/home/hyp0515/simulation/ext_fig_3/ain_{ain}_amid_{amid}_aout_{aout}_Mdot_{mdot}_Q_{Q}_incl_{incl}_rd_{rd}/',
    }
    simulation.generate_continuum(
        scat=True,
        stokes=True,
        read_lambda=obs_wav*1e-3,
        load_simulation=False,
        fname=f'conti',
        **simulate_mutual_parms
    )
    simulation.generate_sed(
        scat=True,
        read_lambda=obs_wav*1e-3,
        load_simulation=False,
        fname='sed',
        **simulate_mutual_parms
    )