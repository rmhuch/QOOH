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

def run_2D_OHDVR(data_dict, extrapolate=0, NumPts=1000, desiredEnergies=3, plotPhasedWfns=False):
    """ Runs anharmonic 1D DVR over the OH coordinate at every degree value in a 2D scan."""
    from Converter import Constants, Potentials1D
    dvr_1D = DVR("ColbertMiller1D")
    degree_vals = np.array(list(data_dict.keys()))
    potential_array = np.zeros((len(data_dict), NumPts, 2))
    energies_array = np.zeros((len(data_dict), desiredEnergies))
    wavefunctions_array = np.zeros((len(data_dict), NumPts, desiredEnergies))
    mO = Constants.mass("O", to_AU=True)
    mH = Constants.mass("H", to_AU=True)
    muOH = 1 / (1 / mO + 1 / mH)

    for j, n in enumerate(data_dict):
        if n == (90, 0):
            x = Constants.convert(data_dict[n]["B6"][:28], "angstroms", to_AU=True)
            en = data_dict[n]["Energy"][:28] - np.min(data_dict[n]["Energy"][:28])
        else:
            x = Constants.convert(data_dict[n]["B6"], "angstroms", to_AU=True)
            en = data_dict[n]["Energy"] - np.min(data_dict[n]["Energy"])
        # subtract minimum at each cut to eliminate the electronic component since we add later
        res = dvr_1D.run(potential_function=Potentials1D().potlint(x, en), mass=muOH,
                         divs=NumPts, domain=(min(x) - extrapolate, max(x) + extrapolate), num_wfns=desiredEnergies)
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

