import numpy as np
import matplotlib.pyplot as plt
import os
from Converter import Constants
from scaledCoefs import *

udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TBHPdir = os.path.join(udrive, "TBHP")


def pullCoefs(dir):
    FEdir = os.path.join(dir, "Fourier Expansions")
    # Vel is actually the electronic energy + the ZPE of the reaction path work... but Vel for short.
    Vel_coeffs = np.loadtxt(os.path.join(FEdir, "VelZPE_Fit_Coeffs_TBHP.csv"), delimiter=",")
    V0_coeffs = np.loadtxt(os.path.join(FEdir, "V0_Fit_Coeffs_TBHP.csv"), delimiter=",")
    V1_coeffs = np.loadtxt(os.path.join(FEdir, "V1_Fit_Coeffs_TBHP.csv"), delimiter=",")
    V2_coeffs = np.loadtxt(os.path.join(FEdir, "V2_Fit_Coeffs_TBHP.csv"), delimiter=",")
    Bel_coeffs = np.loadtxt(os.path.join(FEdir, "Gel_Fit_Coeffs_TBHP.csv"), delimiter=",")
    Bel_coeffs /= 2
    coeff_dict = {"Vel": Vel_coeffs, "V0": V0_coeffs, "V1": V1_coeffs, "V2": V2_coeffs, "Bel": Bel_coeffs}
    # these coeffs for V0 and up are raw, they need to be added to the Vel coeffs first to make a real potential...
    return coeff_dict

def evaluatePot(Vcoefs, x):
    """at this point, Vcoefs should be either the 'electronic' energy or
    the sum of the 'electronic' with an excited state"""
    VelFull = Vcoefs[0] + Vcoefs[1] * np.cos(x) + Vcoefs[2] * np.cos(2 * x) + Vcoefs[3] * np.cos(3 * x) \
          + Vcoefs[4] * np.cos(4 * x) + Vcoefs[5] * np.cos(5 * x) + Vcoefs[6] * np.cos(6 * x)
    return VelFull

def make_Potential_plots(pots):
    """creates a plot of the potentials fed in"""
    coeffDict = pullCoefs(TBHPdir)
    for pot in pots:
        if pot == "Vel":
            Vcoefs = coeffDict[pot]
            pot_label = "Electronic Energy + ZPE of R Path"
        else:
            Vcoefs = coeffDict[pot] + coeffDict["Vel"]
            pot_label = pot
        x = np.linspace(0, 2 * np.pi, 100)
        V = evaluatePot(Vcoefs, x)
        V_wave = Constants.convert(V, "wavenumbers", to_AU=False)
        plt.plot(x, V_wave, label=pot_label)
    plt.legend()
    plt.show()

def make_scaledPots(coeffDict, barrier_height, makeplot=False):
    # calculate any necessary scaling factors/ create plots
    Velcoefs = coeffDict["Vel"]
    V0coefs = coeffDict["V0"]
    V1coefs = coeffDict["V1"]
    V2coefs = coeffDict["V2"]
    # barrier_har = Constants.convert(barrier_height, "wavenumbers", to_AU=True)
    # barrier = evaluatePot(Velcoefs, np.pi)
    # scaling_factor = barrier_har/barrier
    newVel = coeffDict["scaledVel"]
    scaledV0 = newVel + V0coefs
    scaledV1 = newVel + V1coefs
    scaledV2 = newVel + V2coefs
    if makeplot:
        x = np.linspace(0, 2 * np.pi, 100)
        Vel = evaluatePot(Velcoefs, x)
        Vel_scaled = evaluatePot(newVel, x)
        Vel_wave = Constants.convert(Vel, "wavenumbers", to_AU=False)
        Vel_scaled_wave = Constants.convert(Vel_scaled, "wavenumbers", to_AU=False)
        plt.plot(x, Vel_wave, "-k", label=f"Electronic Energy + R Path")
        plt.plot(x, Vel_scaled_wave, label=f"Scaled Energy with Barrier {barrier_height} $cm^-1$")
        plt.show()
    return scaledV0, scaledV1, scaledV2

def calcHam(V_coeffs, B_coeffs, HamSize=None):
    """This should be fed the appropriately scaled Vcoefs, just calculates the Hamiltonian based of what it is given"""
    if HamSize is None:  # make more flexible to different order sizes in expansion - currently only supports 6
        HamSize = 7
    fullsize = 2*HamSize + 1
    Ham = np.zeros((fullsize, fullsize))
    for l in np.arange(fullsize):
        k = l - HamSize
        Ham[l, l] = B_coeffs[0] * (k**2) + V_coeffs[0]  # m=0
        for kprime in np.arange(k+1, k-7, -1):
            m = k - kprime
            if m > 0 and l-m >= 0:
                Ham[l, l-m] = (kprime ** 2 + k ** 2 - m ** 2) * (B_coeffs[m] / 4) + (V_coeffs[m] / 2)
                Ham[l-m, l] = Ham[l, l-m]
            else:
                pass
    return Ham

def calc_sinHam(M_coeffs, HamSize=None):
    if HamSize is None:  # make more flexible to different order sizes in expansion - currently only supports 6
        HamSize = 7
    fullsize = 2*HamSize + 1
    Ham = np.zeros((fullsize, fullsize))
    for l in np.arange(fullsize):
        k = l - HamSize
        for kprime in np.arange(k+1, k-7, -1):
            m = k - kprime
            if m > 0 and l-m >= 0:
                Ham[l, l-m] = M_coeffs[m] / 2
                Ham[l-m, l] = -Ham[l, l-m]
            else:
                pass
    return Ham

def calcEns(V_coeffs, B_coeffs, HamSize=None):
    ham = calcHam(V_coeffs, B_coeffs, HamSize=HamSize)
    energy, eigvecs = np.linalg.eigh(ham)
    spacings = np.array((energy[1]-energy[0], energy[2]-energy[0]))
    enwfn = {'V': V_coeffs,
             'energy': energy,
             'eigvecs': eigvecs,
             'spacings': spacings}
    return enwfn

def PORwfns(eigenvectors, theta):
    ks = np.arange(len(eigenvectors)//2*-1, len(eigenvectors)//2+1)
    vals = np.zeros((len(theta), len(eigenvectors)))
    for n in np.arange(len(eigenvectors)):
        c_ks = eigenvectors[:, n]
        re = np.zeros((len(theta), len(eigenvectors)))
        im = np.zeros((len(theta), len(eigenvectors)))
        for i, k in enumerate(ks):
            im[:, i] = c_ks[i] * (1 / np.sqrt(2 * np.pi)) * np.sin(k * theta)
            re[:, i] = c_ks[i] * (1 / np.sqrt(2 * np.pi)) * np.cos(k * theta)
        vals[:, n] = np.sum(re, axis=1) + np.sum(im, axis=1)
    return vals

def make_plots(resDicts, levels=False, wfns=True, title=None):
    for resDict in resDicts:
        x = np.linspace(0, 2*np.pi, 100)
        pot = evaluatePot(resDict["V"], x)
        pot_wave = Constants.convert(pot, "wavenumbers", to_AU=False)
        mins = np.argsort(pot_wave)
        mins = np.sort(mins[:2])
        center = pot_wave[mins[0]:mins[-1] + 1]
        max_idx = np.argmax(center)
        true_bh = center[max_idx] - center[0]
        print(true_bh)
        plt.plot(x, pot_wave, '-k')
        en0 = Constants.convert(resDict['energy'][0], "wavenumbers", to_AU=False)
        en1 = Constants.convert(resDict['energy'][1], "wavenumbers", to_AU=False)
        en2 = Constants.convert(resDict['energy'][2], "wavenumbers", to_AU=False)
        # en3 = Constants.convert(resDict['energy'][3], "wavenumbers", to_AU=False)
        print(en0, en1, en2)
        if levels:
            enX = np.linspace(np.pi/3, 2*np.pi - np.pi/3, 10)
            plt.plot(enX, np.repeat(en0, len(enX)), "-C0", label=f"{en0}")
            plt.plot(enX, np.repeat(en1, len(enX)), "-C1", label=f"{en1}")
            # plt.plot(enX, np.repeat(en2, len(enX)), "-C2", label=f"{en2}")
        if wfns:
            wfnns = PORwfns(resDict["eigvecs"], x)
            plt.plot(x, en0 + wfnns[:, 0]*100, "-C0", label=f"{en0:.2f}")
            plt.plot(x, en1 + wfnns[:, 1]*100, "-C1", label=f"{en1:.2f}")
            # plt.plot(x, en2 + wfnns[:, 2]*100, "-C2", label=f"{en2:.2f}")
            # plt.plot(x, en3 + wfnns[:, 3]*50, label=f"3")
    plt.title(title)
    # plt.legend()
    plt.show()

def run(barrier_height=None, plots=False):
    coefDict = pullCoefs(TBHPdir)
    # if barrier_height is None:
    res = calcEns(coefDict["Vel"], coefDict["Bel"], HamSize=15)  # converged (15-50)
    V0 = coefDict["Vel"] + coefDict["V0"]
    V1 = coefDict["Vel"] + coefDict["V1"]
    V2 = coefDict["Vel"] + coefDict["V2"]
    res0 = calcEns(V0, coefDict["Bel"], HamSize=15)
    res1 = calcEns(V1, coefDict["Bel"], HamSize=15)
    res2 = calcEns(V2, coefDict["Bel"], HamSize=15)
    if plots:
        make_plots([res0, res1], wfns=False, levels=True, title="Excitation of OH Unscaled")
    # else:
    #     sf, scaledV0, scaledV1, scaledV2 = make_scaledPots(barrier_height, makeplot=True)
    #     res0 = calcEns(scaledV0, coefDict["Bel"], HamSize=15)
    #     res1 = calcEns(scaledV1, coefDict["Bel"], HamSize=15)
    #     res2 = calcEns(scaledV2, coefDict["Bel"], HamSize=15)
    #     make_plots([res0, res1, res2], wfns=True, title=f"Excitation of OH Scaled = {sf:.4f}")
    return res, res0, res1, res2

def run_newScaling():
    dat = pull_data(TBHPdir)  # pull electronic energy data as per scaledCoefs
    coefDict = pullCoefs(TBHPdir)  # pulls coefs from Mark's fits
    sf, scaled_dat = scale_barrier(dat, barrier_height=200)
    scaled_coefs = calc_coefs(scaled_dat)
    coefDict["scaledVel"] = scaled_coefs
    # res = calcEns(coefDict["Vel"], coefDict["Bel"], HamSize=15)
    # resS = calcEns(scaled_coefs, coefDict["Bel"], HamSize=15)
    # make_plots([res, resS], wfns=True, title=f"Excitation of OH Scaled = {sf:.4f}")
    scaledV0, scaledV1, scaledV2 = make_scaledPots(coefDict, barrier_height=200, makeplot=False)
    res0 = calcEns(scaledV0, coefDict["Bel"], HamSize=15)
    res1 = calcEns(scaledV1, coefDict["Bel"], HamSize=15)
    res2 = calcEns(scaledV2, coefDict["Bel"], HamSize=15)
    make_plots([res0, res1, res2], wfns=True, title=f"Excitation of OH Scaled = {sf:.4f}")
    return res0, res1, res2  # calc resEL too ?


if __name__ == '__main__':
    run(plots=True)
    # run_newScaling()

