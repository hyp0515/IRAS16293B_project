This directory includes pipelines to produce synthetic simulation by RADMC-3D and make plots

# vertical_profile_class.py
Extend disk_model.py to spherical coordinates which RADMC-3D adopts.

# setup.py
This prepares the prerequisite input files to run RADMC-3D according to disk_model.py and vertical_profile_class.py. 

Velocity distributions, molecular abundances or snowlines, and heating mechanisms can also be assigned in this pipeline.

# simulate.py
This is the simulation pipeline that can produce synthetic observations of "dust continuum", "SED", "molecular line image cube", and "molecular line spectrum."

The results can be saved as original '.out' files or '.npz' files, which data arrays are easier to access.

# plot.py
This generates plots from the simulations, including "dust continuum", "SED", "channel map", "position-velocity (PV) diagram", and "molecular line spectrum." The density and temperature profiles can also be plotted.