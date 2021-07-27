from MolecularResults2D import *
import os
import matplotlib.pyplot as plt
from Converter import Constants
import numpy as np

qooh = MoleculeInfo2D(MoleculeName="QOOH",
                      atom_array=["C", "C", "H", "H", "O", "O", "H", "C", "H", "H", "H", "C", "H", "H", "H"],
                      eqTORangle=[113.65694, 261.13745],
                      oh_scan_npz="QOOH_OH_2d_energies_internals.npz",
                      tor_scan_npz="QOOH_2d_energies_internals.npz",
                      cartesians_npz="QOOH_2d_cartesians.npz",
                      dipole_npz="rotated_dipoles_coords_qooh_3d.npz")
qooh_2Dres = MolecularResults2D(MolObj=qooh,
                                DVRparams={"desired_energies": 4,
                                           "num_pts": 1000,
                                           "plot_phased_wfns": False,
                                            "extrapolate": 0},
                                TransitionStr=["0->1", "0->2"])

test = qooh_2Dres.TwoD_transitions()
# HOOC = qooh.TORScanData["D4"]
# CH2 = qooh.TORScanData["D2"]
# OH = qooh.TORScanData["B6"]
# energies = qooh.TORScanData["Energy"]
# en_cut = energies - min(energies)
# qooh_2Dres.plot_Surfaces(HOOC, CH2, zcoord=OH)
# plt.show()
