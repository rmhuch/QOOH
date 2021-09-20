"""this script calculates the energies and intensities of 2D Oh cuts at QOOH1 & QOOH2"""
import numpy as np
import os
from Converter import Constants
from ExpectationValues import run_DVR
from McUtils.GaussianInterface import GaussianLogReader
from runQOOH2D import qooh

log1 = os.path.join(qooh.MoleculeDir, "qooh12_eq2.log")
dip1 = os.path.join(qooh.MoleculeDir, "rotated_dipoles_qooh_eq2_AA.npz")
log2 = os.path.join(qooh.MoleculeDir, "qooh12_eq1.log")
dip2 = os.path.join(qooh.MoleculeDir, "rotated_dipoles_qooh_eq1_AA.npz")

def get_logData(logfile):
    with GaussianLogReader(logfile) as parser:
        parse = parser.parse("OptimizedScanEnergies")
        internal_coords, ens = parse["OptimizedScanEnergies"]
    return internal_coords, ens

def get_Dipoles(dipfile):
    dat = np.load(dipfile)
    for file in dat.files:
        hooc_str, occx_str = file.split("_")
        HOOC = int(hooc_str)
        OCCX = int(occx_str)
        dipoles = dat[file][:, :4]
    return dipoles

def calc_EnsWfns(logfile):
    internal_coords, ens = get_logData(logfile)
    OH = internal_coords["B6"][np.logical_not(np.isnan(internal_coords["B6"]))]
    en = ens[np.logical_not(np.isnan(ens))]
    Epot = np.column_stack((OH, en))
    ens_array, wfns = run_DVR(Epot, NumPts=2000)  # converts x to Bohr
    print("Energies: ", ens_array)
    return ens_array, wfns

def calc_TDM(dips, wfns, transition_str):
    """calculates the transition moment at each degree value. Returns the TDM at each degree.
        Normalize the mu value with the normalization of the wavefunction!!!"""
    from scipy import interpolate
    interpG = wfns[:, 0]
    interpW = wfns[:, 1:]
    interpD = np.zeros((len(interpG), 3))
    dip_range = Constants.convert(dips[:, 0], "angstroms", to_AU=True)
    for i in np.arange(3):
        f = interpolate.interp1d(dip_range, dips[:, i+1], kind="cubic", bounds_error=False, fill_value="extrapolate")
        interpD[:, i] = f(interpG)
    mus = np.zeros(3)
    Glevel = int(transition_str[0])
    Exlevel = int(transition_str[-1])
    for j in np.arange(3):  # loop through x, y, z
        gs_wfn = interpW[:, Glevel].T
        es_wfn = interpW[:, Exlevel].T
        es_wfn_t = es_wfn.reshape(-1, 1)
        soup = np.diag(interpD[:, j]).dot(es_wfn_t)
        mu = gs_wfn.dot(soup)
        normMu = mu / (gs_wfn.dot(gs_wfn) * es_wfn.dot(es_wfn))  # a triple check that everything is normalized
        mus[j] = normMu
    return mus

def calc_Intensity(dipfile, logfile):
    dips = get_Dipoles(dipfile)
    ens_array, wfns = calc_EnsWfns(logfile)
    comp_intents = np.zeros(3)
    for i, trans in enumerate(["0->0", "0->1", "0->2"]):
        print(trans)
        mus = calc_TDM(dips, wfns, trans)
        print(mus)
        freq_wave = ens_array[i]-ens_array[0]
        for c, val in enumerate(["A", "B", "C"]):  # loop through components
            comp_intents[c] = (abs(mus[c]) ** 2) * freq_wave * 2.506 / (0.393456 ** 2)
        intensity = np.sum(comp_intents)
        print(intensity)

if __name__ == '__main__':
    calc_Intensity(dip2, log2)



