import numpy as np
import matplotlib.pyplot as plt

HG_19_data = [
    {"Freq": 3.0, "Bmaj": 1.10, "Bmin": 0.46, "BPa": -20.7, "rms": 0.013, "S_B": 0.052, "sigma": 0.005},
    {"Freq": 10.0, "Bmaj": 0.35, "Bmin": 0.14, "BPa": -17.5, "rms": 0.014, "S_B": 1.08, "sigma": 0.11},
    {"Freq": 15.0, "Bmaj": 0.23, "Bmin": 0.10, "BPa": 19.6, "rms": 0.008, "S_B": 2.64, "sigma": 0.26},
    {"Freq": 15.0, "Bmaj": 0.24, "Bmin": 0.09, "BPa": -23.7, "rms": 0.005, "S_B": 2.68, "sigma": 0.27},
    {"Freq": 10.0, "Bmaj": 0.39, "Bmin": 0.14, "BPa": -26.9, "rms": 0.006, "S_B": 1.05, "sigma": 0.11},
    {"Freq": 7.0, "Bmaj": 0.44, "Bmin": 0.19, "BPa": -168.4, "rms": 0.006, "S_B": 0.55, "sigma": 0.06},
    {"Freq": 33.0, "Bmaj": 0.11, "Bmin": 0.04, "BPa": -15.6, "rms": 0.027, "S_B": 15.3, "sigma": 2.3},
    {"Freq": 41.0, "Bmaj": 0.09, "Bmin": 0.04, "BPa": -4.6, "rms": 0.072, "S_B": 26.2, "sigma": 3.9},
    {"Freq": 41.0, "Bmaj": 0.36, "Bmin": 0.16, "BPa": -10.0, "rms": 0.030, "S_B": 31.3, "sigma": 4.7},
    {"Freq": 41.0, "Bmaj": 0.30, "Bmin": 0.14, "BPa": 11.5, "rms": 0.060, "S_B": 28.9, "sigma": 4.3},
    {"Freq": 41.0, "Bmaj": 0.13, "Bmin": 0.10, "BPa": -13.9, "rms": 0.027, "S_B": 26.9, "sigma": 4.0},
    {"Freq": 41.0, "Bmaj": 0.08, "Bmin": 0.05, "BPa": -174.4, "rms": 0.037, "S_B": 29.1, "sigma": 4.4},
    {"Freq": 6.0, "Bmaj": 0.50, "Bmin": 0.20, "BPa": -12.5, "rms": 0.006, "S_B": 0.38, "sigma": 0.04},
    {"Freq": 227.0, "Bmaj": 0.53, "Bmin": 0.25, "BPa": 87.2, "rms": 1.81, "S_B": 1.79e3, "sigma": 0.27e3},
    {"Freq": 318.0, "Bmaj": 0.39, "Bmin": 0.34, "BPa": -62.2, "rms": 3.89, "S_B": 3.21e3, "sigma": 0.48e3},
    {"Freq": 318.0, "Bmaj": 0.35, "Bmin": 0.31, "BPa": -62.2, "rms": 2.27, "S_B": 3.23e3, "sigma": 0.48e3},
    {"Freq": 323.0, "Bmaj": 0.39, "Bmin": 0.32, "BPa": 87.3, "rms": 3.15, "S_B": 3.35e3, "sigma": 0.50e3},
    {"Freq": 338.2, "Bmaj": 0.17, "Bmin": 0.13, "BPa": -83.9, "rms": 3.49, "S_B": 3.60e3, "sigma": 0.54e3},
    {"Freq": 342.3, "Bmaj": 0.25, "Bmin": 0.13, "BPa": -76.9, "rms": 6.03, "S_B": 3.12e3, "sigma": 0.57e3},
    {"Freq": 404.0, "Bmaj": 0.28, "Bmin": 0.22, "BPa": -78.3, "rms": 7.95, "S_B": 4.67e3, "sigma": 0.93e3},
    {"Freq": 453.0, "Bmaj": 0.31, "Bmin": 0.21, "BPa": -81.9, "rms": 8.86, "S_B": 6.14e3, "sigma": 1.23e3},
    {"Freq": 695.0, "Bmaj": 0.29, "Bmin": 0.16, "BPa": -70.4, "rms": 19.90, "S_B": 13.5e3, "sigma": 2.7e3},
]

this_work_data = [
    {"Freq": 16.7,  "Bmaj": 0.23, "Bmin": 0.09, "BPa":  5.19, "rms": 0.047, "S_B":    2.204, "sigma": 0.166},
    {"Freq": 42.8,  "Bmaj": 0.39, "Bmin": 0.25, "BPa": 73.75, "rms": 0.047, "S_B":   32.406, "sigma": 0.327},
    {"Freq": 99.9,  "Bmaj": 0.05, "Bmin": 0.04, "BPa": 79.33, "rms": 0.017, "S_B":  213.512, "sigma": 2.633},
    {"Freq": 230.6, "Bmaj": 0.11, "Bmin": 0.07, "BPa": -88.2, "rms": 0.104, "S_B": 1165.770, "sigma": 8.886}
]

# Extracting data for plotting
freqs_HG19 = np.array([d["Freq"] for d in HG_19_data])
S_B_HG19   = np.array([d["S_B"] for d in HG_19_data])
sigma_HG19 = np.array([d["sigma"] for d in HG_19_data])
freqs_this_work = np.array([d["Freq"] for d in this_work_data])
S_B_this_work   = np.array([d["S_B"] for d in this_work_data])
sigma_this_work = np.array([d["sigma"] for d in this_work_data])

# Plotting the data
if __name__ == "__main__":
    plt.figure(figsize=(6, 10))
    plt.errorbar(freqs_HG19, S_B_HG19, yerr=sigma_HG19, fmt='o', 
                color='blue', ecolor='lightblue', elinewidth=2, capsize=4, label='Hernández-Gómez et al. 2019')
    plt.errorbar(freqs_this_work, S_B_this_work, yerr=sigma_this_work, fmt='o', 
                color='r', ecolor='r', elinewidth=2, capsize=4, label='This Work')
    plt.xscale('log'); plt.xlim((5e-1, 1e3))
    plt.yscale('log'); plt.ylim((1e-2, 2e5))
    plt.xlabel('Frequency (GHz)', fontsize=14)
    plt.ylabel('Flux Density (mJy)', fontsize=14)
    plt.title('Spectral Energy Distribution (SED)', fontsize=16)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sed_plot.pdf', transparent=True)
    plt.show()
