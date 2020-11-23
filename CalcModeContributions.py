import numpy as np
import os
from Converter import Constants
from McUtils import Numputils


def pull_coordsandmodes(coord_vals, modedir):
    coords = []
    modes = []
    demass = np.sqrt(np.array([12., 12., 1.00782504, 1.00782504, 15.9949146, 15.9949146,
                               1.00782504, 12., 1.00782504, 1.00782504, 1.00782504, 12.,
                               1.00782504, 1.00782504, 1.00782504, 1.00782504]))
    for i in coord_vals:
        dat = np.loadtxt(os.path.join(modedir, f"tbhp_eq_v{i}_modes.dat"))  # f"tbhp_eq_{i}oh_modes.dat"
        coordies = Constants.convert(np.reshape(dat[:1, :], (-1, 3)), "angstroms", to_AU=False)
        coords.append(coordies)
        big_mode_data = Constants.convert(dat[1:, :], "angstroms", to_AU=False)
        # modies = np.reshape(big_mode_data, (len(big_mode_data), -1, 3)) / demass[np.newaxis, :, np.newaxis]
        modes.append(big_mode_data)  # np.reshape(modies, big_mode_data.shape)
    return np.array(coords), np.array(modes)

def OOHContribs(coord_vals, modedir):
    """Returns a dictionary keyed by rOH values with a 2column array as the key.
    [:, 0]: mode numbers and [:, 1]: absolute value of contributions"""
    coords, modes = pull_coordsandmodes(coord_vals, modedir)
    contribDict = dict()
    for val, coord, mode in zip(coord_vals, coords, modes):
        di, dj, dk = Numputils.angle_deriv(coord, [5], [4], [6])
        deriv_OOH = np.zeros((len(coord), 3))
        deriv_OOH[5] = di
        deriv_OOH[4] = dj
        deriv_OOH[6] = dk
        norm_derivOOH = deriv_OOH / np.linalg.norm(deriv_OOH)
        contribs = np.dot(mode, norm_derivOOH.flatten())
        norm_contribs = (contribs / np.linalg.norm(contribs))**2
        idx = np.where(abs(norm_contribs) > 0.3)[0]
        contribdat = np.column_stack((idx, abs(norm_contribs[idx])))
        sortdat = contribdat[contribdat[:, 1].argsort()[::-1]]
        contribDict[val] = sortdat
    return contribDict

def OOContribs(coord_vals, modedir):
    """Returns a dictionary keyed by rOH values with a 2column array as the key.
    [:, 0]: mode numbers and [:, 1]: absolute value of contributions"""
    coords, modes = pull_coordsandmodes(coord_vals, modedir)
    contribDict = dict()
    for val, coord, mode in zip(coord_vals, coords, modes):
        di, dj = Numputils.dist_deriv(coord, [4], [5])
        deriv_OO = np.zeros((len(coord), 3))
        deriv_OO[4] = di
        deriv_OO[5] = dj
        norm_derivOO = deriv_OO / np.linalg.norm(deriv_OO)
        contribs = np.dot(mode, norm_derivOO.flatten())
        norm_contribs = contribs / np.linalg.norm(contribs)
        idx = np.where(abs(norm_contribs) > 0.3)[0]
        contribdat = np.column_stack((idx, abs(norm_contribs[idx])))
        sortdat = contribdat[contribdat[:, 1].argsort()[::-1]]
        contribDict[val] = sortdat
    return contribDict

def make_rOH_freqdiffplot(freq_files):  # COME BACK TO THIS... SOMETHING IS NOT RIGHT
    import matplotlib.pyplot as plt
    from FourierExpansions import calc_cos_coefs, calc_curves
    params = {'text.usetex': False,
              'mathtext.fontset': 'dejavusans',
              'font.size': 14}
    eq_freqs = np.loadtxt("no_tor_tor_freqs.csv", delimiter=",")
    sel = [x for x in range(49) if x not in list(range(8)) + [29, 48]]
    roh = np.array((0.98149, 1.01601, 1.05222, 1.09009, 1.12931, 1.17051))
    ZPE_diff = np.zeros((48, len(roh)))
    for i, file in enumerate(freq_files):
        fn = os.path.join(udrive, "TBHP", "VibStateFchks", "vib_state_mode_freqs", file)
        roh_freqs = np.loadtxt(fn)
        ZPE_diff[:, i] = (roh_freqs - eq_freqs)/2
    print('hi')


if __name__ == '__main__':
    udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    OHdir = os.path.join(udrive, "TBHP", "eq_scans")
    OHfreqdir = os.path.join(OHdir, "oh_mode_freqs")
    OHmodedir = os.path.join(OHdir, "oh_modes")

    # OHvals = ["09", "10", "11", "12", "13", "14", "15", "16"]
    # vals = OOContribs(OHvals, OHmodedir)
    fie_names = [f"tbhp_eq_v{i}_freqs.dat" for i in np.arange(6)]
    make_rOH_freqdiffplot(fie_names)
