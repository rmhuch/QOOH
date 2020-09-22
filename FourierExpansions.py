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
    # barrier_har = Constants.convert(barrier_height, "wavenumbers", to_AU=True)
    # barrier = evaluatePot(Velcoefs, np.pi)
    # scaling_factor = barrier_har/barrier
    newVel = coeffDict["scaledVel"]
    scaled_coeffs = []
    for i in np.arange(1, len(coeffDict.keys()) - 2):  # loop through saved energies
        scaled_coeffs.append(coeffDict["scaledVel"] + coeffDict[f"V{i - 1}"])
    if makeplot:
        x = np.linspace(0, 2 * np.pi, 100)
        Vel = evaluatePot(Velcoefs, x)
        Vel_scaled = evaluatePot(newVel, x)
        Vel_wave = Constants.convert(Vel, "wavenumbers", to_AU=False)
        Vel_scaled_wave = Constants.convert(Vel_scaled, "wavenumbers", to_AU=False)
        plt.plot(x, Vel_scaled_wave, "-g", label=f"Scaled Energy with Barrier {barrier_height} $cm^-1$")
        plt.plot(x, Vel_wave, "-k", label=f"Electronic Energy + R Path")
        plt.show()
    return scaled_coeffs

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
        en3 = Constants.convert(resDict['energy'][3], "wavenumbers", to_AU=False)
        print(en0, en1, en2, en3)
        if levels:
            enX = np.linspace(np.pi/3, 2*np.pi - np.pi/3, 10)
            plt.plot(enX, np.repeat(en0, len(enX)), "-r", label=f"{en0}")
            plt.plot(enX, np.repeat(en1, len(enX)), "-b", label=f"{en1}")
            plt.plot(enX, np.repeat(en2, len(enX)), "-g", label=f"{en2}")
            plt.plot(enX, np.repeat(en3, len(enX)), color="indigo", label=f"{en3}")
        if wfns:
            wfnns = PORwfns(resDict["eigvecs"], x)
            plt.plot(x, en0 + wfnns[:, 0]*100, "-r", label=f"{en0:.2f}")
            plt.plot(x, en1 + wfnns[:, 1]*100, "-b", label=f"{en1:.2f}")
            plt.plot(x, en2 + wfnns[:, 2]*100, "-g", label=f"{en2:.2f}")
            plt.plot(x, en3 + wfnns[:, 3]*100, color="indigo", label=f"{en3:.2f}")

    plt.title(title)
    # plt.legend()
    plt.show()
    
def run(levs_to_calc=7):  # WATCH THIS! IT NEEDS DVR-NPZ BACK EVENTUALLY
    gcoeffs = np.loadtxt(os.path.join(TBHPdir, "Fourier Expansions", "Gel_Fit_Coeffs_TBHP.csv"), delimiter=",")
    dvvr = "energiesDVR_TBHP_extended1500.txt"
    coefDict = pull_DVR_data(TBHPdir, gcoeffs, DVR_fn=dvvr, levs_to_calc=levs_to_calc)
    # coefDict = pullCoefs(TBHPdir)
    res = calcEns(coefDict["Vel"], coefDict["Bel"], HamSize=15)  # converged (15-50)
    results = []
    for i in np.arange(1, len(coefDict.keys())-1):  # loop through saved energies
        Vi = coefDict["Vel"] + coefDict[f"V{i-1}"]
        results.append(calcEns(Vi, coefDict["Bel"], HamSize=15))
    return res, results  # returns electronic energies and list of results of higher values

def run_Scaling(barrier_height=275, levs_to_calc=7):
    dat = pull_Eldata(TBHPdir)  # pull electronic energy data as per scaledCoefs
    # dvvr = "energiesDVR_TBHP_extended1500.txt"
    dvvr = "energiesDVR_TBHP_extended.txt"
    gcoeffs = np.loadtxt(os.path.join(TBHPdir, "Fourier Expansions", "Gel_Fit_Coeffs_TBHP.csv"), delimiter=",")
    coefDict = pull_DVR_data(TBHPdir, gcoeffs, DVR_fn=dvvr, levs_to_calc=levs_to_calc)
    # coefDict = pullCoefs(TBHPdir)  # pulls coefs from Mark's fits
    sf, scaled_dat = scale_barrier(dat, barrier_height=barrier_height)
    scaled_ELcoefs = calc_coefs(scaled_dat)
    coefDict["scaledVel"] = scaled_ELcoefs
    all_scaledcoeffs = make_scaledPots(coefDict, barrier_height=barrier_height, makeplot=False)
    results = []
    for i in np.arange(len(all_scaledcoeffs)):  # loop through scaled coefficients
        Vi = all_scaledcoeffs[i]
        results.append(calcEns(Vi, coefDict["Bel"], HamSize=15))
    return results  # calc resEL too ?


if __name__ == '__main__':
    run_Scaling(levs_to_calc=3)
