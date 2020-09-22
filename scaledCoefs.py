import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import optimize
from Converter import Constants

udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TBHPdir = os.path.join(udrive, "TBHP")
QOOHdir = os.path.join(udrive, "QOOH")

def pull_Eldata(mol_dir):
    FEdir = os.path.join(mol_dir, "Fourier Expansions")
    # Vel is actually the electronic energy + the ZPE of the reaction path work... but Vel for short.
    Vel_data = np.loadtxt(os.path.join(FEdir, "VelZPE_Data_TBHP.csv"), delimiter=",")
    rad = Vel_data[:, 0] * (np.pi/180)
    return np.column_stack((rad, Vel_data[:, 1]))  # (RADIANS, energy(hartree))

def pull_DVR_data(mol_dir, gmatrix_coeffs, DVR_fn=None, DVR_npz=None,  levs_to_calc=7):
    if DVR_npz is None:
        dvr_energies = np.loadtxt(os.path.join(mol_dir, DVR_fn))
        dvr_energies[:, 1:] = Constants.convert(dvr_energies[:, 1:], "wavenumbers", to_AU=True)
    else:
        all_dvr_dat = np.load(os.path.join(mol_dir, DVR_npz), allow_pickle=True)
        dvr_energies = all_dvr_dat["energies"]
        dvr_energies[:, 1:] = Constants.convert(dvr_energies[:, 1:], "wavenumbers", to_AU=True)
    # [degrees vOH=0 ... vOH=6]
    coeff_dict = dict()  # build new coeff dict based on extended scans
    gmatrix_coeffs /= 2
    coeff_dict["Bel"] = gmatrix_coeffs
    rad = dvr_energies[:, 0] * (np.pi/180)
    Eldat = pull_Eldata(mol_dir)
    coeff_dict["Vel"] = calc_coefs(Eldat)
    for i in np.arange(1, levs_to_calc + 1):  # loop through saved energies
        shifted_en = dvr_energies[:, i]
        energies = np.column_stack((rad, shifted_en))
        coeff_dict[f"V{i-1}"] = calc_coefs(energies)
    return coeff_dict
        
def scale_barrier(energy_dat, barrier_height):
    """finds the minima and cuts those points between, scales barrier and returns full data set back"""
    # (90, 270) for TBHP
    mins = np.argsort(energy_dat[:, 1])
    mins = np.sort(mins[:2])
    center = energy_dat[mins[0]:mins[-1]+1, :]
    max_idx = np.argmax(center[:, 1])
    true_bh = center[max_idx, 1] - center[0, 1]
    barrier_har = Constants.convert(barrier_height, "wavenumbers", to_AU=True)
    scaling_factor = barrier_har / true_bh
    new_center = np.column_stack((center[:, 0], scaling_factor*center[:, 1]))
    left = energy_dat[0:mins[0], :]
    right = energy_dat[mins[-1]+1:, :]
    scaled_energies = np.vstack((left, new_center, right))
    return scaling_factor, scaled_energies

def cos_func(x, c0, c1, c2, c3, c4, c5, c6):
    return c0 + c1*np.cos(x) + c2*np.cos(2*x) + c3*np.cos(3*x) + c4*np.cos(4*x) + c5*np.cos(5*x) + c6*np.cos(6*x)

def calc_coefs(energy_dat):
    """conducts a cos expansion to fit the energy curve given, returns sixth order coefs."""
    popt, pcov = optimize.curve_fit(cos_func, energy_dat[:, 0], energy_dat[:, 1])
    return popt

def sin_func(x, c0, c1, c2, c3, c4, c5, c6):
    return c0 + c1*np.sin(x) + c2*np.sin(2*x) + c3*np.sin(3*x) + c4*np.sin(4*x) + c5*np.sin(5*x) + c6*np.sin(6*x)

def calc_sin_coefs(energy_dat):
    """conducts a sin expansion to fit the energy curve given, returns sixth order coefs."""
    popt, pcov = optimize.curve_fit(sin_func, energy_dat[:, 0], energy_dat[:, 1])
    return popt

def calc_curves(x, coefs):
    energies = cos_func(x, *coefs)
    plt.plot(x, energies)
    plt.show()
    return energies

def runtest():
    dat = pull_Eldata(TBHPdir)
    x = np.linspace(0, 2*np.pi, 100)
    sf, scaled_dat = scale_barrier(dat, barrier_height=275)
    scaled_coefs = calc_coefs(scaled_dat)
    calc_curves(x, scaled_coefs)

if __name__ == '__main__':
    runtest()
    # dvvr = "energiesDVR_TBHP_extended.txt"
    # gcoeffs = np.loadtxt(os.path.join(TBHPdir, "Fourier Expansions", "Gel_Fit_Coeffs_TBHP.csv"), delimiter=",")
    # pull_DVR_data(TBHPdir, gcoeffs, DVR_fn=dvvr)
