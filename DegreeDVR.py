import numpy as np
from PyDVR import *
from McUtils import *
from DVRtools import *

def formatData(Rmins_file, Renergies_file, step=0.02):
    Rmins = np.loadtxt(Rmins_file)
    Renergies = np.loadtxt(Renergies_file, skiprows=1)
    # load in min points
    minE = np.loadtxt("mins-ens.txt")
    datDict = dict()
    for i in range(len(Rmins)):
        # create min array, given start (Rmins[i, 1]), step size 0.02, and len
        maxi = Rmins[i, 1] + (step*len(Renergies[:, i]))
        x = np.arange(Rmins[i, 1], maxi, step)
        x = x[:len(Renergies[:, i])]
        cut = np.column_stack((x, Renergies[:, i]))
        cut = np.vstack((minE[i, 1:], cut))
        idx = np.argsort(cut[:, 0])
        cut_sort = cut[idx, :]
        cut_sort[:, 0] = np.around(cut_sort[:, 0], 5)
        u, u_idx = np.unique(cut_sort[:, 0], return_index=True)
        u_sort = cut_sort[u_idx, :]
        datDict[Rmins[i, 0]] = u_sort
    # data in degrees: angstroms/hartrees
    return datDict

def run_anharOH_DVR(cut_dict, NumPts=500, desiredEnergies=3, plotPhasedWfns=False):
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
        mini = min(x) - 0.3
        maxi = max(x) + 0.3
        en = cut_dict[n][:, 1] - np.min(cut_dict[n][:, 1])
        min_arg = np.argmin(cut_dict[n][:, 1])
        min_x = cut_dict[n][min_arg, :]
        print(n, *min_x)
        res = dvr_1D.run(potential_function=Potentials1D().potlint(x, en), mass=muOH,
                         divs=NumPts, domain=(mini, maxi), num_wfns=desiredEnergies)
        potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
        grid = Constants.convert(res.grid, "angstroms", to_AU=False)
        potential_array[j, :, 0] = grid
        potential_array[j, :, 1] = potential
        ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
        energies_array[j, :] = ens
        wavefunctions_array[j, :, :] = res.wavefunctions.wavefunctions
    epsilon_pots = np.column_stack((degrees, energies_array[:, :desiredEnergies+1]))  # this is energies.txt
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

if __name__ == '__main__':
    RminsFN = "Rmin_degreeScans.txt"
    RenergiesFN = "Energies_degreeScans.txt"
    dat = formatData(RminsFN, RenergiesFN)
    res = run_anharOH_DVR(dat)
    # np.savetxt("frequencies.txt", calcFreqs(res[1]))
    # ohWfn_plots(res)
    # np.savetxt("potentials.txt", res[0])
    # np.savetxt("energies.txt", res[1])
    # np.savetxt("wavefunctions.txt", res[2])
