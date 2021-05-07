from MolecularResults import *
import os
import matplotlib.pyplot as plt
import numpy as np
from FourierExpansions import calc_curves, fourier_coeffs
from Converter import Constants
from TransitionMoments import TransitionMoments
params = {'text.usetex': False,
          'mathtext.fontset': 'dejavusans',
          'font.size': 14}
plt.rcParams.update(params)

qooh = MoleculeInfo(MoleculeName="QOOH",
                    atom_array=["C", "C", "H", "H", "O", "O", "H", "C", "H", "H", "H", "C", "H", "H", "H"],
                    eqTORangle=[113.65694, 261.13745],
                    oh_scan_npz="QOOH_OH_energies_internals.npz",
                    TorFchkDirectory="TorFchks",
                    TorFiles=[f"q12_{i}.fchk" for i in ["000", "010", "020", "030", "040", "050", "060", "070",
                                                        "080", "090", "100", "110", "120", "130", "140", "150",
                                                        "160", "170", "180", "190", "200", "210", "220", "230",
                                                        "240", "250", "260", "270", "280", "290", "300", "310",
                                                        "320", "330", "340", "350", "360"]],
                    dipole_npz="rotated_dipoles_coords_qooh.npz")  # need to fix parser code for this data

qooh_res_obj = MoleculeResults(MoleculeInfo_obj=qooh,
                               DVRparams={"desired_energies": 8,
                                          "num_pts": 2000,
                                          "plot_phased_wfns": False,
                                          "extrapolate": 0.3,
                                          "PrintResults": True},
                               PORparams={"None": True, 
                                          "twoD": False},
                               # this flag ignores POR functions, expanding in full fourier series, and doing `PiDVR`
                               Intenseparams={"numGstates": 4,
                                              "numEstates": 6,
                                              "FranckCondon": False},
                               transition=[f"0->{i}" for i in [1, 2, 3, 4, 5]])

def run_pot_plots():
    res = qooh_res_obj.torDVRresults
    from PlotDVROHAdiabats import make_Potplots, make_PotWfnplots, make_one_Potplot
    make_one_Potplot(res[1], ZPE=False, filename=os.path.join(qooh.MoleculeDir, "figures", f"vOH1_woZPE"))
    # for i in [0, 1, 2, 3, 4]:
    #     make_Potplots(res[0], res[i], potfunc="fourier", numStates=5, ZPE=True,
    #                   filename=os.path.join(qooh.MoleculeDir, "figures", f"vOH0{i}_wZPE"))
    #     make_Potplots(res[0], res[i], potfunc="fourier", numStates=5, ZPE=False,
    #                   filename=os.path.join(qooh.MoleculeDir, "figures", f"vOH0{i}_woZPE"))
    #     make_PotWfnplots(res[i], wfn_idx=[0, 1, 2, 3, 4],
    #                      filename=os.path.join(qooh.MoleculeDir, "figures", f"vOH{i}_wfns"))

def make_Vel_plots(mol_res_obj):
    Vel = mol_res_obj.RxnPathResults["electronicE"][:, 1]
    Vel -= min(Vel)
    degrees = mol_res_obj.RxnPathResults["electronicE"][:, 0]
    plt.plot(degrees, Constants.convert(Vel, "wavenumbers", to_AU=False), "o", label="Vel")
    Vel_coeffs = fourier_coeffs(np.column_stack((np.radians(degrees), Vel)), sin_order=6, cos_order=6)
    Vel_fit = calc_curves(np.radians(np.arange(0, 360, 1)), Vel_coeffs, function="fourier")
    Vel_res = np.column_stack((np.arange(0, 360, 1), Constants.convert(Vel_fit, "wavenumbers", to_AU=False)))
    max_arg1 = np.argmax(Vel_res[100:270, 1])
    print("Vel", Vel_res[max_arg1+100, :])
    plt.plot(Vel_res[:, 0], Vel_res[:, 1], "-b")
    Vel_ZPE = mol_res_obj.VelCoeffs
    velZPE = calc_curves(np.radians(np.arange(0, 360, 1)), Vel_ZPE, function="fourier")
    res = np.column_stack((np.arange(0, 360, 1), Constants.convert(velZPE, "wavenumbers", to_AU=False)))
    max_arg = np.argmax(res[100:270, 1])
    print("Vel+ZPE", res[max_arg+100, :])
    plt.plot(res[:, 0], res[:, 1], label="Vel+ZPE")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # run_pot_plots()
    a = qooh_res_obj.TransitionIntensities
    # make_Vel_plots(qooh_res_obj)
    # coefs = qooh_res_obj.VelCoeffs
    # vals = calc_curves(np.radians(np.arange(183, 184, 0.001)), coefs, function="fourier")
    # all_vals = np.column_stack((np.arange(183, 184, 0.001), Constants.convert(vals, "wavenumbers", to_AU=False)))
    # max_arg = np.argmax(vals)
    # print(all_vals[max_arg, :])
