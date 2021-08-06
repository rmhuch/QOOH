import numpy as np
from PyDVR import *
from DVRtools import *

# All of these functions are implemented through the MolecularInfo/MolecularResults class

def run_OH_DVR(cut_dict, extrapolate=0, NumPts=1000, desiredEnergies=3, plotPhasedWfns=False):
    """ Runs anharmonic DVR over the OH coordinate at every degree value."""
    from Converter import Constants, Potentials1D
    dvr_1D = DVR("ColbertMiller1D")
    degrees = np.array(list(cut_dict.keys()))
    potential_array = np.zeros((len(cut_dict), NumPts, 2))
    energies_array = np.zeros((len(cut_dict), desiredEnergies))
    wavefunctions_array = np.zeros((len(cut_dict), NumPts, desiredEnergies))
    mO = Constants.mass("O", to_AU=True)
    mH = Constants.mass("H", to_AU=True)
    muOH = 1/(1/mO + 1/mH)

    for j, n in enumerate(cut_dict):
        x = Constants.convert(cut_dict[n][:, 0], "angstroms", to_AU=True)
        en = cut_dict[n][:, 1] - np.min(cut_dict[n][:, 1])
        # subtract minimum at each cut to eliminate the electronic component since we fit our adiabats without it.
        res = dvr_1D.run(potential_function=Potentials1D().potlint(x, en), mass=muOH,
                         divs=NumPts, domain=(min(x)-extrapolate, max(x)+extrapolate), num_wfns=desiredEnergies)
        potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
        grid = Constants.convert(res.grid, "angstroms", to_AU=False)
        potential_array[j, :, 0] = grid
        potential_array[j, :, 1] = potential
        ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
        energies_array[j, :] = ens
        wavefunctions_array[j, :, :] = res.wavefunctions.wavefunctions
    epsilon_pots = np.column_stack((degrees, energies_array[:, :desiredEnergies+1]))
    # data saved in wavenumbers/degrees
    wavefuns_array = wfn_flipper(wavefunctions_array, plotPhasedWfns=plotPhasedWfns, pot_array=potential_array)
    return potential_array, epsilon_pots, wavefuns_array

def calcFreqs(epsilonPots):
    Ens = epsilonPots[:, 1:]
    freqs = np.zeros((len(epsilonPots), 5))
    for i in range(len(epsilonPots)):
        E10 = Ens[i, 1] - Ens[i, 0]
        E20 = Ens[i, 2] - Ens[i, 0]
        omega_x = (E20 - (2*E10))/-2
        omega = E10 + (2*omega_x)
        freqs[i, :] = [epsilonPots[i, 0], E10, E20, omega_x, omega]
    return freqs  # this is frequencies.txt

def run_2D_OHDVR(data_dict, NumPts=1000, desiredEnergies=3, plotPhasedWfns=False):
    """ Runs anharmonic 1D DVR over the OH coordinate at every degree value in a 2D scan.
    NOTE: domain is set to STATIONARY (0.7, 1.5) ang"""
    from Converter import Constants, Potentials1D
    dvr_1D = DVR("ColbertMiller1D")
    degree_vals = np.array(list(data_dict.keys()))
    potential_array = np.zeros((len(data_dict), NumPts, 2))
    energies_array = np.zeros((len(data_dict), desiredEnergies))
    wavefunctions_array = np.zeros((len(data_dict), NumPts, desiredEnergies))
    mO = Constants.mass("O", to_AU=True)
    mH = Constants.mass("H", to_AU=True)
    mD = Constants.mass("D", to_AU=True)
    muOH = 1 / (1 / mO + 1 / mH)
    muOD = 1 / (1 / mO + 1 / mD)
    # we will run dvr over the SAME grid (0.7, 1.5 A) for all OH cuts for uniformity
    domain_max = Constants.convert(1.5, "angstroms", to_AU=True)
    domain_min = Constants.convert(0.7, "angstroms", to_AU=True)
    for j, n in enumerate(data_dict):
        if n == (90, 0) or n == (90, 180) or (90, 360):
            x = Constants.convert(data_dict[n]["B6"][:28], "angstroms", to_AU=True)
            en = data_dict[n]["Energy"][:28] - np.min(data_dict[n]["Energy"][:28])
        else:
            x = Constants.convert(data_dict[n]["B6"], "angstroms", to_AU=True)
            en = data_dict[n]["Energy"] - np.min(data_dict[n]["Energy"])
        # subtract minimum at each cut to eliminate the electronic component since we add later
        res = dvr_1D.run(potential_function=Potentials1D().potlint(x, en), mass=muOD,
                         divs=NumPts, domain=(domain_min, domain_max), num_wfns=desiredEnergies)
        potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
        grid = Constants.convert(res.grid, "angstroms", to_AU=False)
        potential_array[j, :, 0] = grid
        potential_array[j, :, 1] = potential
        energies_array[j, :] = res.wavefunctions.energies
        wavefunctions_array[j, :, :] = res.wavefunctions.wavefunctions
    epsilon_pots = np.column_stack((np.radians(degree_vals), energies_array[:, :desiredEnergies + 1]))
    # data saved in radians/HARTREES
    wavefuns_array = wfn_flipper(wavefunctions_array, plotPhasedWfns=plotPhasedWfns, pot_array=potential_array)
    return potential_array, epsilon_pots, wavefuns_array

def plot_wfns(wavefuns, pot_array, data_pts, saveDir):
    import matplotlib.pyplot as plt
    import os
    colors = ["b", "r", "g"]
    for i, val in enumerate(data_pts):
        fig = plt.figure()
        # plt.plot(pot_array[i+26, :, 0], pot_array[i+26, :, 1], "-k")
        plt.plot(np.repeat(1.5, 10), np.linspace(-0.08, 0.08, 10), "-k")
        for j in np.arange(3):
            plt.plot(pot_array[i, :, 0], wavefuns[i, :, j], color=colors[j], label=f"wfn {j}")
        plt.legend()
        plt.title(f"{val} (HOOC, OCCX)")
        plt.savefig(os.path.join(saveDir, "ODwfns_QOOH_{}_{}.png".format(val[0], val[1])))
        plt.close()
