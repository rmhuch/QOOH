import numpy as np
import os
from Converter import Constants

udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TBHPdir = os.path.join(udrive, "TBHP")

# pull in dipole moments
def pull_dipoles(dirname, npz_fn):
    data = np.load(os.path.join(dirname, npz_fn))
    # pulls in data as a dictionary that is keyed by angles, with values: (roh, x, y, z)
    return data

def pull_dipole_derivs(dirname, eqdip_fn, deriv_fn):
    import matplotlib.pyplot as plt
    ed_fn = os.path.join(dirname, eqdip_fn)
    de_fn = os.path.join(dirname, deriv_fn)
    EQdip = np.loadtxt(ed_fn)
    derivs = np.loadtxt(de_fn)  # derivs in au/bohr
    # degrees = np.linspace(0, 360, len(EQdip))
    # for c, val in enumerate(["X", "Y", "Z"]):
    #     plt.plot(degrees, EQdip[:, c], 'o', label=f"{val} - Component")
    # plt.title("EQ Dipole")
    # plt.legend()
    # plt.show()
    # for c, val in enumerate(["X", "Y", "Z"]):
    #     plt.plot(degrees, derivs[:, c], 'o', label=f"{val} - Component")
    # plt.title("OH Dipole Derivative")
    # plt.legend()
    # plt.show()
    return EQdip, derivs

# calculate OH wfns
def run_DVR(dirname, Rmins_fn, Renergies_fn):
    from DegreeDVR import formatData, run_anharOH_DVR
    dat = formatData(os.path.join(dirname, Rmins_fn), os.path.join(dirname, Renergies_fn))
    potential_array, epsilon_pots, wavefuns_array = run_anharOH_DVR(dat, desiredEnergies=7, NumPts=1500)
    return wavefuns_array, potential_array  # pot in ang/har

# reshape dips and wfns
def interpDW(dirname, Rmins_fn, Renergies_fn, npz_fn=None, eqdip_fn=None, deriv_fn=None):
    from scipy import interpolate
    wavefuns, pot = run_DVR(dirname, Rmins_fn, Renergies_fn)
    Rmins = np.loadtxt(os.path.join(dirname, Rmins_fn))
    if npz_fn is not None:
        flag = "dipoles"
        dipole_dict = pull_dipoles(dirname, npz_fn)
        tor_degrees = np.linspace(0, 360, len(dipole_dict.keys())).astype(int)
        EQdips, OHderivs = None, None
    elif eqdip_fn is not None and deriv_fn is not None:
        flag = "derivs"
        EQdips, OHderivs = pull_dipole_derivs(dirname, eqdip_fn, deriv_fn)
        dipole_dict, tor_degrees = None, None
    else:
        raise Exception("Data missing, can't pull dipoles.")
    pot_bohr = Constants.convert(pot[:, :, 0], "angstroms", to_AU=True)  # convert to bohr before using with derivs
    grid_min = np.min(pot_bohr)
    grid_max = np.max(pot_bohr)
    new_grid = np.linspace(grid_min, grid_max, 500)
    interp_wfns = np.zeros((wavefuns.shape[0], len(new_grid), wavefuns.shape[2]))
    interp_dips = np.zeros((wavefuns.shape[0], len(new_grid), 3))
    for i in np.arange(wavefuns.shape[0]):  # loop through torsion degrees
        for s in np.arange(wavefuns.shape[2]):  # loop through OH wfns states
            f = interpolate.interp1d(pot_bohr[i, :], wavefuns[i, :, s],
                                     kind="cubic", bounds_error=False, fill_value="extrapolate")
            interp_wfns[i, :, s] = f(new_grid)
        for c in np.arange(3):  # loop through dipole components
            if flag == "dipoles":
                dipole_dat = dipole_dict[f"tbhp_{tor_degrees[i]:0>3}.log"]
                new_dipole_dat = np.column_stack((dipole_dat[:, 0], dipole_dat[:, 3], dipole_dat[:, 1], dipole_dat[:, 2]))
                new_dipole_dat[:, 0] = Constants.convert(new_dipole_dat[:, 0], "angstroms", to_AU=True)
                f = interpolate.interp1d(new_dipole_dat[:-1, 0], new_dipole_dat[:-1, c+1],
                                         kind="cubic", bounds_error=False, fill_value="extrapolate")
                interp_dips[i, :, c] = f(new_grid)
            elif flag == "derivs":
                delta = new_grid - Rmins[i, 1]
                interp_dips[i, :, c] = OHderivs[i, c]*delta + EQdips[i, c]
    return new_grid, interp_wfns, interp_dips  # dips in all au

def plot_interpDW(dirname, Rmins_fn, Renergies_fn, npz_fn=None, eqdip_fn=None, deriv_fn=None):
    import matplotlib.pyplot as plt
    g, wfns, dips = interpDW(dirname, Rmins_fn, Renergies_fn, npz_fn=npz_fn, eqdip_fn=eqdip_fn, deriv_fn=deriv_fn)
    tor_degrees = np.linspace(0, 360, wfns.shape[0]).astype(int)
    for i in np.arange(wfns.shape[0]):
        # for s in np.arange(3):
        #     plt.plot(g, wfns[i, :, s])
        for j, c in enumerate(["A", "B", "C"]):
            plt.plot(g, dips[i, :, j], label=f"{c} - Component")
        plt.title(f"{tor_degrees[i]}  Degrees")
        plt.show()

# calculate TDM
def calc_TDM(dipoles, ohWfns, transition="0 -> 1"):
    """calculates the transition moment at each degree value. Returns the TDM at each degree"""
    mus = np.zeros((len(dipoles), 3))
    if transition == "0 -> 1":
        Exlevel = 1
    elif transition == "0 -> 2":
        Exlevel = 2
    elif transition == "0 -> 3":
        Exlevel = 3
    elif transition == "0 -> 4":
        Exlevel = 4
    elif transition == "0 -> 5":
        Exlevel = 5
    elif transition == "0 -> 6":
        Exlevel = 6

    else:
        raise Exception("Sorry, I don't know that transition")
    for k in np.arange(len(dipoles)):  # loop through degree values
        for j in np.arange(3):  # loop through x, y, z
            gs_wfn = ohWfns[k, :, 0].T
            es_wfn = ohWfns[k, :, Exlevel].T
            es_wfn_t = es_wfn.reshape(-1, 1)
            soup = np.diag(dipoles[k, :, j]).dot(es_wfn_t)
            mu = gs_wfn.dot(soup)
            mus[k, j] = mu
    return mus  # in all au, returns only the TDM of the given transition

def plot_TDM(dipoles, ohWfns, transition="0 -> 1"):  # fix this
    import matplotlib.pyplot as plt
    mus = calc_TDM(dipoles, ohWfns, transition=transition)
    degrees = np.linspace(0, 360, len(mus))
    for c, val in enumerate(["A", "B", "C"]):
        plt.plot(degrees, mus[:, c], 'o', label=f"{val} - Component")
    plt.legend()
    plt.show()

# expand the transition dipole moment into a matrix representation so that it can be used to solve for intensity
def calc_TDM_matrix(dirname, Rmins_fn, Renergies_fn, eqdip_fn, deriv_fn, npz_fn, size=15, transition="0 -> 1"):
    from scaledCoefs import calc_coefs, calc_sin_coefs
    from FourierExpansions import calcHam, calc_sinHam
    grid, OHwfns, OHdips = interpDW(dirname, Rmins_fn, Renergies_fn,
                                    npz_fn=npz_fn, eqdip_fn=eqdip_fn, deriv_fn=deriv_fn)
    mus = calc_TDM(OHdips, OHwfns, transition=transition)  # returns only for given transition
    x = np.linspace(0, 360, len(mus)) * (np.pi/180)  # radians
    matsize = (size*2) + 1
    TDM_mats = np.zeros((3, matsize, matsize))
    for c, val in enumerate(["A", "B", "C"]):
        if val == "A":
            # fit to 6th order sin functions
            data = np.column_stack((x, mus[:, c]))
            Mcoefs = calc_sin_coefs(data)
            # expand Mcoefs into matrix representation following "calcHam" infrastructure
            TDM_mats[c, :, :] = calc_sinHam(Mcoefs, size)
        else:
            # first fit B/C mus to 6th order cos functions (as we fit V in "scaledcoefs")
            data = np.column_stack((x, mus[:, c]))
            Mcoefs = calc_coefs(data)
            # expand Mcoefs as before using "calcHam" note here B-Coefs is not appicable so is set to zeros
            TDM_mats[c, :, :] = calcHam(Mcoefs, np.zeros(len(Mcoefs)), size)
    return TDM_mats  # in au, only for given transition

# pull torsion wavefunction coefficients
def run_FE(barrier_height=None):
    from FourierExpansions import run, run_Scaling
    if barrier_height is None:
        resEl, results = run()  # plots bool
    else:
        results = run_Scaling(barrier_height=barrier_height, plots=False)  # plots bool
    torWfn_coefs = dict()
    torEnergies = dict()
    dict_names = ["res0", "res1", "res2", "res3", "res4", "res5", "res6"]
    for i, dic in enumerate(results):
        torWfn_coefs[dict_names[i]] = dic["eigvecs"]
        torEnergies[dict_names[i]] = dic["energy"]
    return torWfn_coefs, torEnergies

# calculate overlap of torsion wfns with TDM
def calc_intensity(dirname, Rmins_fn, Renergies_fn, eqdip_fn=None, deriv_fn=None, npz_fn=None,
                   size=15, transition="0 -> 1", barrier_height=None):
    TDM_mats = calc_TDM_matrix(dirname, Rmins_fn, Renergies_fn, eqdip_fn, deriv_fn, npz_fn,
                               size=size, transition=transition)
    torWfn_coefs, torEnergies = run_FE(barrier_height=barrier_height)
    # print(Constants.convert(torEnergies["res4"][:10], "wavenumbers", to_AU=False))
    if transition == "0 -> 1":
        Exlevel = "res1"
    elif transition == "0 -> 2":
        Exlevel = "res2"
    elif transition == "0 -> 3":
        Exlevel = "res3"
    elif transition == "0 -> 4":
        Exlevel = "res4"
    elif transition == "0 -> 5":
        Exlevel = "res5"
    elif transition == "0 -> 6":
        Exlevel = "res6"
    else:
        raise Exception("Sorry, I don't know that transition")
    intensities = []
    for gState in np.arange(4):
        for exState in np.arange(4):
            freq = torEnergies[Exlevel][exState] - torEnergies["res0"][gState]
            freq_wave = Constants.convert(freq, "wavenumbers", to_AU=False)
            print("\n")
            print(f"OH - {transition}, tor - {gState} -> {exState} : {freq_wave:.2f} cm^-1")
            comp_intents = np.zeros(3)
            for c, val in enumerate(["A", "B", "C"]):  # loop through components
                supere = np.dot(TDM_mats[c, :, :], torWfn_coefs[Exlevel][:, exState].T)
                matEl = np.dot(torWfn_coefs["res0"][:, gState], supere)
                comp_intents[c] = (abs(matEl) ** 2) * freq_wave * 2.506 / (0.393456 ** 2)
                print(f"{val} : {comp_intents[c]:.8f} km/mol")
            print(f"total : {np.sum(comp_intents):.6f}")
            if comp_intents[1] > 1E-10:
                ratio = comp_intents[1] / comp_intents[2]
                print(f"B/C ratio : {ratio:.4f}")
            intensities.append([gState, exState, freq_wave, np.sum(comp_intents)])
    return intensities  # [tor gstate, tor exstate, frequency (cm^-1), intensity (km/mol)]

def calc_stat_intensity(dirname, Rmins_fn, Renergies_fn, eqdip_fn=None, deriv_fn=None, npz_fn=None,
                        transition="0 -> 1", values=[270]):
    from FourierExpansions import evaluatePot, run
    grid, OHwfns, OHdips = interpDW(dirname, Rmins_fn, Renergies_fn,
                                    npz_fn=npz_fn, eqdip_fn=eqdip_fn, deriv_fn=deriv_fn)
    mus = calc_TDM(OHdips, OHwfns, transition=transition)
    degrees = np.linspace(0, 360, len(mus))
    resEl, results = run()
    OH_0coefs = results[0]["V"]
    if transition == "0 -> 1":
        OH_excoefs = results[1]["V"]
    elif transition == "0 -> 2":
        OH_excoefs = results[2]["V"]
    else:
        raise Exception("Sorry, I don't know that transition")
    intensity = np.zeros((len(values), 3))
    for i, value in enumerate(values):
        idx = np.argwhere(degrees == value)[0][0]
        OH_0 = evaluatePot(OH_0coefs, value)
        OH_ex = evaluatePot(OH_excoefs, value)
        freq = OH_ex - OH_0
        freq_wave = Constants.convert(freq, "wavenumbers", to_AU=False)
        print(f"Stationary Frequency : {freq_wave}")
        for c, val in enumerate(["A", "B", "C"]):  # loop through components
            intensity[i, c] = (abs(mus[idx, c]))**2 * freq_wave * 2.506 / (0.393456 ** 2)
            print(f"Stationary Intensity @ {value} {val} : {intensity[i, c]}")
    return intensity

def plot_stat_intensity(dirname, Rmins_fn, Renergies_fn, eqdip_fn=None, deriv_fn=None, npz_fn=None,
                        transition="0 -> 1", values=[270]):
    import matplotlib.pyplot as plt
    intents = calc_stat_intensity(dirname, Rmins_fn, Renergies_fn, eqdip_fn=eqdip_fn, deriv_fn=deriv_fn, npz_fn=npz_fn,
                                  transition=transition, values=values)
    for c, val in enumerate(["A", "B", "C"]):
        plt.plot(values, intents[:, c], 'o', label=f"{val} - Component")
    plt.legend()
    plt.show()

def tor_freq_plot(dirname, Rmins_fn, Renergies_fn, barrier_heights, eqdip_fn=None, deriv_fn=None, npz_fn=None,
                  size=15, transition="0 -> 1"):
    import matplotlib.pyplot as plt
    data = dict()
    data[312] = calc_intensity(dirname, Rmins_fn, Renergies_fn, npz_fn=npz_fn, eqdip_fn=eqdip_fn,
                               deriv_fn=deriv_fn, size=size, transition=transition)
    trans02t = data[312][2]
    trans22t = data[312][10]
    # print(f"312 splitting : {trans02t[2] - trans22t[2]}")
    plt.plot(312, trans02t[2]-trans22t[2], 'om')
    for i, bh in enumerate(barrier_heights):
        print(bh)
        intensities = calc_intensity(dirname, Rmins_fn, Renergies_fn, npz_fn=npz_fn, eqdip_fn=eqdip_fn,
                                     deriv_fn=deriv_fn, size=size, transition=transition, barrier_height=bh)
        trans02 = intensities[2]
        trans22 = intensities[10]
        # print(f"{bh} splitting : {trans02[2]-trans22[2]}")
        plt.plot(bh, trans02[2]-trans22[2], 'og')
        data[bh] = intensities  # bh : [tor gstate, tor exstate, frequency (cm^-1), intensity (km/mol)]
    plt.xlabel("Vel + R path Barrier Height")
    plt.ylabel("Energy Splitting (cm^-1)")
    plt.show()
    return data

if __name__ == '__main__':
    RminsTBHP = "Rmins_TBHP.txt"
    RenergiesTBHP = "Energies_TBHP_extended.txt"
    eqdipsTBHP = "ek_dip.dat"
    dipDerivsTBHP = "ekOHDerivs.dat"
    # g, wfns, dips = interpDW(TBHPdir, RminsTBHP, RenergiesTBHP, npz_fn="rotated_dipoles.npz")
    # plot_interpDW(TBHPdir, RminsTBHP, RenergiesTBHP, npz_fn="rotated_dipoles.npz")
    # plot_TDM(dips, wfns, transition="0 -> 2")
    # calc_intensity(TBHPdir, RminsTBHP, RenergiesTBHP, npz_fn="rotated_dipoles.npz", transition="0 -> 4")
    # v = list(np.arange(0, 370, 10))
    # plot_stat_intensity(TBHPdir, RminsTBHP, RenergiesTBHP, eqdip_fn=eqdipsTBHP, deriv_fn=dipDerivsTBHP,
    #                     transition="0 -> 2", values=v)
    bhs = list(np.arange(250, 375, 25))
    tor_freq_plot(TBHPdir, RminsTBHP, RenergiesTBHP, bhs, npz_fn="rotated_dipoles.npz", transition="0 -> 4")