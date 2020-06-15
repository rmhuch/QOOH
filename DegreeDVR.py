import numpy as np
import os
from PyDVR import *
from DVRtools import *

def formatData(Rmins_file, Renergies_file, addMins=False, step=0.02):
    Rmins = np.loadtxt(Rmins_file)
    Renergies = np.loadtxt(Renergies_file, skiprows=1)
    datDict = dict()
    for i in range(len(Rmins)):
        # create min array, given start (Rmins[i, 1]), step size 0.02, and len
        maxi = Rmins[i, 1] + (step*len(Renergies[:, i]))
        x = np.arange(Rmins[i, 1], maxi, step)
        x = x[:len(Renergies[:, i])]
        cut = np.column_stack((x, Renergies[:, i]))
        if addMins:
            minE = np.loadtxt("mins-ens.txt")  # load in min points
            cut = np.vstack((minE[i, 1:], cut))
            idx = np.argsort(cut[:, 0])
            cut_sort = cut[idx, :]
            cut_sort[:, 0] = np.around(cut_sort[:, 0], 5)
            u, u_idx = np.unique(cut_sort[:, 0], return_index=True)
            u_sort = cut_sort[u_idx, :]
            finalCut = u_sort
        else:
            finalCut = cut
        datDict[Rmins[i, 0]] = finalCut
    # data in degrees: angstroms/hartrees
    return datDict

def formatvaryingstepData(Renergies_file, mins, steps, minpoint, addMins=True):
    datDict = dict()
    RenergiesFull = np.loadtxt(f"{Renergies_file}_0.005.txt")
    for i in range(len(steps)):
        # create min array, given start mins, steps, and len
        # Renergies = np.loadtxt(f"{Renergies_file}_{steps[i]}.txt")
        if steps[i] == 0.02:
            Renergies = RenergiesFull[::4]
        elif steps[i] == 0.01:
            Renergies = RenergiesFull[::2]
        else:
            Renergies = RenergiesFull
        maxi = mins[i] + (steps[i]*len(Renergies))
        x = np.arange(mins[i], maxi, steps[i])
        x = x[:len(Renergies)]
        cut = np.column_stack((x, Renergies))
        if addMins:
            cut = np.vstack((minpoint, cut))
            idx = np.argsort(cut[:, 0])
            cut_sort = cut[idx, :]
            cut_sort[:, 0] = np.around(cut_sort[:, 0], 5)
            u, u_idx = np.unique(cut_sort[:, 0], return_index=True)
            u_sort = cut_sort[u_idx, :]
            finalCut = u_sort
        else:
            finalCut = cut
        datDict[steps[i]] = finalCut
    return datDict

def run_anharOH_DVR(cut_dict, NumPts=1000, desiredEnergies=3, plotPhasedWfns=False):
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
        # min_x = cut_dict[n][min_arg, :]
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
    udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    TBHPdir = os.path.join(udrive, "TBHP")
    RminsTBHP = "Rmins_TBHP.txt"
    RenergiesTBHP = "Energies_TBHP.txt"
    dat = formatData(os.path.join(TBHPdir, RminsTBHP), os.path.join(TBHPdir, RenergiesTBHP))
    res = run_anharOH_DVR(dat)
    np.savetxt(os.path.join(TBHPdir, "frequenciesDVR_TBHP.txt"), calcFreqs(res[1]))
    np.savetxt(os.path.join(TBHPdir, "energiesDVR_TBHP.txt"), res[1])

    # minsTest90 = [0.75078, 0.75078, 0.75078]
    # minsTest180 = [0.75905, 0.75905, 0.75905]
    # minsTest270 = [0.75104, 0.75104, 0.75104]
    # stepsTest = [0.02, 0.01, 0.005]
    # energiesFN90 = "Energies_QOOH_90"
    # energiesFN180 = "Energies_QOOH_180"
    # energiesFN270 = "Energies_QOOH_270"
    # min90 = [0.9657800000000002, -308.0062614]
    # min180 = [0.9640500000000002, -308.0055606]
    # min270 = [0.9660400000000002, -308.0078036]
    # data = formatvaryingstepData(energiesFN270, minsTest270, stepsTest, min270)
    # resu = run_anharOH_DVR(data)
    # np.savetxt("frequencies_QOOH_270.txt", calcFreqs(resu[1]))
    # # ohWfn_plots(resu, degree=90)
    # np.savetxt("energies_QOOH_270.txt", resu[1])
