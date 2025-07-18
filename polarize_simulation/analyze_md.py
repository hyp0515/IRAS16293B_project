import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(
    "./log.txt", 
    delim_whitespace=True, 
    comment="=",
    skiprows=2,
    names=["amax", "Mstar", "Mdot", "Rd", "Q", "Md"]
)


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].scatter(data["amax"], data["Md"], marker='o', color='b')
ax[0].set_title("Maximum grain size vs. Md")
ax[0].set_xlabel("amax (mm)")
ax[0].set_xscale("log")
ax[0].set_ylabel("Disk Mass (Msun)") 

ax[1].scatter(data["Q"], data["Md"], marker='o', color='g')
ax[1].set_title("Toomre Q vs. Md")
ax[1].set_xlabel("Toomre Q")
ax[1].set_ylabel("Disk Mass (Msun)")    

ax[2].scatter(data["Mdot"], data["Md"], marker='o', color='r')
ax[2].set_title("Accretion rate vs. Md")
ax[2].set_xlabel("Mdot (Msun/yr)")
ax[2].set_xscale("log")
ax[2].set_ylabel("Disk Mass (Msun)")  
plt.tight_layout()
plt.savefig("Md_different_params.pdf", transparent=True)
plt.show()