from MolecularResults2D import *
import os
import matplotlib.pyplot as plt
from Converter import Constants
import numpy as np

qooh = MoleculeInfo2D(MoleculeName="QOOH",
                      atom_array=["C", "C", "H", "H", "O", "O", "H", "C", "H", "H", "H", "C", "H", "H", "H"],
                      eqTORangle=[113.65694, 261.132145],
                      oh_scan_npz="QOOH_OH_2d_energies_internals.npz",
                      tor_scan_npz="QOOH_2d_energies_internals.npz",
                      cartesians_npz="QOOH_2d_cartesians.npz",
                      dipole_npz="rotated_dipoles_qooh_3d_AA.npz",
                      ccsd_potfile="mdhr_pot_qtbuooh_15.dat")
qooh_2Dres = MolecularResults2D(MolObj=qooh,
                                DVRparams={"desired_energies": 4,
                                           "num_pts": 1000,
                                           "plot_phased_wfns": False,
                                            "extrapolate": 0},
                                TransitionStr=["0->1", "0->2"],
                                IntensityType="TDM")

test = qooh_2Dres.TwoD_transitions()
# meh = qooh_2Dres.ConstGRes
# print(meh["CCSD_avg"]["energy_array"])
# print(meh["DFT"]["energy_array"])
# pot = meh["CCSD"]["potential"].reshape((51, 51)).T
# pot2 = meh["DFT"]["potential"].reshape((51, 51)).T
# grid = meh["CCSD"]["grid"][0]
# grid_for_real = np.moveaxis(grid, -1, 0)
# hooc = grid_for_real[0].flatten()
# x = np.unique(hooc)
# a = np.argmin(pot[:, 37])
# b = np.argmax(pot[a:, 37])
# aa = np.argmin(pot[a+b:, 37])
# print("min arg:", a, (a+b+aa), "val:", pot[a, 37], pot[a+b+aa, 37])
# plt.plot(x, pot[:, 37], label="CCSD")
# plt.plot(x, pot2[:, 36], label="DFT")
# plt.legend()
# plt.xlabel("OCCH (degrees)")
# plt.ylabel("Potential Energy (cm^-1)")
# plt.show()

