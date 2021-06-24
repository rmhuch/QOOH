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
                      cartesians_npz="QOOH_2d_cartesians.npz")

test = qooh.Gmatrix
# HOOC = qooh.TORScanData["D4"]
# CH2 = qooh.TORScanData["D2"]
# energies = qooh.TORScanData["Energy"]
# en_cut = energies - min(energies)
# qooh.plot_Surfaces(HOOC, CH2, zcoord=Constants.convert(en_cut, "wavenumbers", to_AU=False))
