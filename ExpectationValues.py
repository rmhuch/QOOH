import numpy as np
import os
from PyDVR import *

def pullEnergies(txtfile):
    ens = np.loadtxt(txtfile)
    maxi = 0.7247 + (0.02 * len(ens))
    x = np.arange(0.7247, maxi, 0.02)
    x = x[:len(ens)]
    energy = np.column_stack((x, ens))
    return energy

def run_DVR(Epot_array, extrapolate=0, NumPts=1000, desiredEnergies=3):
    """ Runs anharmonic DVR over the OH coordinate at every degree value."""
    from Converter import Constants, Potentials1D
    dvr_1D = DVR("ColbertMiller1D")
    mO = Constants.mass("O", to_AU=True)
    mH = Constants.mass("H", to_AU=True)
    muOH = 1/(1/mO + 1/mH)

    x = Constants.convert(Epot_array[:, 0], "angstroms", to_AU=True)
    en = Epot_array[:, 1] - np.min(Epot_array[:, 1])
    res = dvr_1D.run(potential_function=Potentials1D().potlint(x, en), mass=muOH,
                     divs=NumPts, domain=(min(x)-extrapolate, max(x)+extrapolate), num_wfns=desiredEnergies)
    ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
    energies_array = ens  # wavenumbers
    wavefunctions_array = np.column_stack((res.grid, res.wavefunctions.wavefunctions))  # bohr
    return energies_array, wavefunctions_array

def calc_expectation(dvr_wavefunctions):
    import matplotlib.pyplot as plt
    # dvr_wavefunctions holds grid, and wfns.
    for i in np.arange(1, dvr_wavefunctions.shape[1]):  # i starts at 1 to skip grid in indexing, but is gs (ie E0)
        wfn2 = dvr_wavefunctions[:, i]**2
        expect = wfn2.dot(dvr_wavefunctions[:, 0])
        print(f"expected rOH for E{i-1}: ", "%0.5f" % expect)
        plt.plot(dvr_wavefunctions[:, 0], dvr_wavefunctions[:, i])
    plt.show()

def calc_inverse_expectation(dvr_wavefunctions):
    expect = np.zeros((dvr_wavefunctions.shape[1]-1, 2))
    # dvr_wavefunctions holds grid, and wfns.
    for i in np.arange(dvr_wavefunctions.shape[1]-1):
        expect[i, 0] = i
        wfn2 = dvr_wavefunctions[:, i+1]**2
        expect[i, 1] = wfn2.dot(np.reciprocal(dvr_wavefunctions[:, 0]))
    return expect

def calc_inverse_expectation_squared(dvr_wavefunctions):
    expect2 = np.zeros((dvr_wavefunctions.shape[1]-1, 2))
    # dvr_wavefunctions holds grid, and wfns.
    for i in np.arange(dvr_wavefunctions.shape[1]-1):
        expect2[i, 0] = i
        wfn2 = dvr_wavefunctions[:, i+1]**2
        expect2[i, 1] = wfn2.dot(np.reciprocal(dvr_wavefunctions[:, 0]**2))
    return expect2


if __name__ == '__main__':
    udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file = os.path.join(udrive, "TBHP", "tbhp_eq_energies.txt")
    resE = pullEnergies(file)
    # print(resE)
    x,  wfns = run_DVR(resE, NumPts=2000, desiredEnergies=7)
    calc_expectation(wfns)
