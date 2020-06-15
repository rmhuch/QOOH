import numpy as np
import os
from Converter import Constants

udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TBHPdir = os.path.join(udrive, "TBHP")

# pull in dipole moments
def pull_dipoles(dirname, eqdip_fn, deriv_fn):
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
    potential_array, epsilon_pots, wavefuns_array = run_anharOH_DVR(dat)
    return wavefuns_array, potential_array  # pot in ang/har

# reshape dips and wfns
def interpDW(dirname, eqdip_fn, deriv_fn, Rmins_fn, Renergies_fn):
    from scipy import interpolate
    wavefuns, pot = run_DVR(dirname, Rmins_fn, Renergies_fn)
    Rmins = np.loadtxt(os.path.join(dirname, Rmins_fn))
    EQdips, OHderivs = pull_dipoles(dirname, eqdip_fn, deriv_fn)
    pot_bohr = Constants.convert(pot[:, :, 0], "angstroms", to_AU=True)  # convert to bohr before using with derivs
    grid_min = np.min(pot_bohr)
    grid_max = np.max(pot_bohr)
    new_grid = np.linspace(grid_min, grid_max, 500)
    interp_wfns = np.zeros((wavefuns.shape[0], len(new_grid), 2))  # for now, we only calculate 0, 1 states
    interp_dips = np.zeros((wavefuns.shape[0], len(new_grid), 3))
    for i in np.arange(wavefuns.shape[0]):  # loop through torsion degrees
        for s in np.arange(2):  # loop through OH wfns states
            f = interpolate.interp1d(pot_bohr[i, :], wavefuns[i, :, s],
                                     kind="cubic", bounds_error=False, fill_value="extrapolate")
            interp_wfns[i, :, s] = f(new_grid)
        for c in np.arange(3):  # loop through dipole components
            delta = new_grid - Rmins[i, 1]
            interp_dips[i, :, c] = OHderivs[i, c]*delta + EQdips[i, c]
    return new_grid, interp_wfns, interp_dips  # dips in all au

def plot_interpDW(dirname, eqdip_fn, deriv_fn, Rmins_fn, Renergies_fn):
    import matplotlib.pyplot as plt
    g, wfns, dips = interpDW(dirname, eqdip_fn, deriv_fn, Rmins_fn, Renergies_fn)
    for i in np.arange(wfns.shape[0]):
        for s in np.arange(2):
            plt.plot(g, wfns[i, :, s])
        for c in np.arange(3):
            plt.plot(g, dips[i, :, c])
        plt.show()

# calculate TDM
def calc_TDM(dipoles, ohWfns):
    """calculates the transition moment at each degree value. Returns the TDM at each degree"""
    mus = np.zeros((len(dipoles), 3))
    for k in np.arange(len(dipoles)):  # loop through degree values
        for j in np.arange(3):  # loop through x, y, z
            gs_wfn = ohWfns[k, :, 0].T
            es_wfn = ohWfns[k, :, 1].T
            es_wfn_t = es_wfn.reshape(-1, 1)
            soup = np.diag(dipoles[k, :, j]).dot(es_wfn_t)
            mu = gs_wfn.dot(soup)
            mus[k, j] = mu
    return mus  # in all au

# expand the transition dipole moment into a matrix representation so that it can be used to solve for intensity
def calc_TDM_matrix(dirname, eqdip_fn, deriv_fn, Rmins_fn, Renergies_fn, size=15):
    from scaledCoefs import calc_coefs, calc_sin_coefs
    from FourierExpansions import calcHam, calc_sinHam
    grid, OHwfns, OHdips = interpDW(dirname, eqdip_fn, deriv_fn, Rmins_fn, Renergies_fn)
    mus = calc_TDM(OHdips, OHwfns)
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
    return TDM_mats  # in au


def plot_TDM(dipoles, ohWfns):
    import matplotlib.pyplot as plt
    mus = calc_TDM(dipoles, ohWfns)
    degrees = np.linspace(0, 360, len(mus))
    for c, val in enumerate(["A", "B", "C"]):
        plt.plot(degrees, mus[:, c], 'o', label=f"{val} - Component")
    plt.legend()
    plt.show()

# pull torsion wavefunction coefficients
def run_FE():
    from FourierExpansions import run
    resEl, res0, res1, res2 = run()
    torWfn_coefs = dict()
    torEnergies = dict()
    dict_names = ["resEl", "res0", "res1", "res2"]
    for i, dic in enumerate([resEl, res0, res1, res2]):
        torWfn_coefs[dict_names[i]] = dic["eigvecs"]
        torEnergies[dict_names[i]] = dic["energy"]
    return torWfn_coefs, torEnergies

# calculate overlap of torsion wfns with TDM
def calc_intensity(dirname, eqdip_fn, deriv_fn, Rmins_fn, Renergies_fn, size=15):
    TDM_mats = calc_TDM_matrix(dirname, eqdip_fn, deriv_fn, Rmins_fn, Renergies_fn, size=size)
    torWfn_coefs, torEnergies = run_FE()
    intensities = dict()
    for c, val in enumerate(["A", "B", "C"]):  # loop through components
        for exState in np.arange(2):
            for gState in np.arange(2):
                supere = np.dot(TDM_mats[c, :, :], torWfn_coefs["res1"][:, exState].T)
                matEl = np.dot(torWfn_coefs["res0"][:, gState], supere)
                freq = torEnergies["res1"][exState] - torEnergies["res0"][gState]
                freq_wave = Constants.convert(freq, "wavenumbers", to_AU=False)
                intensity = (abs(matEl) ** 2) * freq_wave * 2.506 / (0.393456 ** 2)
                print(f"0,{gState} -> 1, {exState} : {freq_wave}")
                print(f"{val} 0,{gState} -> 1, {exState} : {intensity}")
    return intensities

def calc_stat_intensity(dirname, eqdip_fn, deriv_fn, Rmins_fn, Renergies_fn, values=[270]):
    from FourierExpansions import run, evaluatePot
    grid, OHwfns, OHdips = interpDW(dirname, eqdip_fn, deriv_fn, Rmins_fn, Renergies_fn)
    mus = calc_TDM(OHdips, OHwfns)
    degrees = np.linspace(0, 360, len(mus))
    resEl, res0, res1, res2 = run()
    OH_0coefs = res0["V"]
    OH_1coefs = res1["V"]
    intensity = np.zeros((len(values), 3))
    for i, value in enumerate(values):
        idx = np.argwhere(degrees == value)[0][0]
        OH_0 = evaluatePot(OH_0coefs, value)
        OH_1 = evaluatePot(OH_1coefs, value)
        freq = OH_1 - OH_0
        freq_wave = Constants.convert(freq, "wavenumbers", to_AU=False)
        print(f"Stationary Frequency : {freq_wave}")
        for c, val in enumerate(["A", "B", "C"]):  # loop through components
            intensity[i, c] = (abs(mus[idx, c]))**2 * freq_wave * 2.506 / (0.393456 ** 2)
            print(f"Stationary Intensity @ {value} {val} : {intensity[i, c]}")
    return intensity

def plot_stat_intensity(dirname, eqdip_fn, deriv_fn, Rmins_fn, Renergies_fn, values=[270]):
    import matplotlib.pyplot as plt
    intents = calc_stat_intensity(dirname, eqdip_fn, deriv_fn, Rmins_fn, Renergies_fn, values=values)
    for c, val in enumerate(["A", "B", "C"]):
        plt.plot(values, intents[:, c], 'o', label=f"{val} - Component")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    RminsTBHP = "Rmins_TBHP.txt"
    RenergiesTBHP = "Energies_TBHP.txt"
    eqdipsTBHP = "ek_dip.dat"
    dipDerivsTBHP = "ekOHDerivs.dat"
    # g, wfns, dips = interpDW(TBHPdir, eqdipsTBHP, dipDerivsTBHP, RminsTBHP, RenergiesTBHP)
    # plot_TDM(dips, wfns)
    v = list(np.arange(0, 370, 10))
    plot_stat_intensity(TBHPdir, eqdipsTBHP, dipDerivsTBHP, RminsTBHP, RenergiesTBHP, values=v)
