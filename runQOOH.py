from MolecularResults import *
import os
import matplotlib.pyplot as plt
import numpy as np
from FourierExpansions import calc_curves
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
                    TorFchkDirectory="TorFchks",  # need to find/pull this data
                    TorFiles=[f"qooh_{i}.fchk" for i in ["000", "010", "020", "030", "040", "050", "060", "070",
                                                         "080", "090", "100", "110", "120", "130", "140", "150",
                                                         "160", "170", "180", "190", "200", "210", "220", "230",
                                                         "240", "250", "260", "270", "280", "290", "300", "310",
                                                         "320", "330", "340", "350", "360"]],
                    dipole_npz="rotated_dipoles_qooh.npz")  # need to find/pull this data

qooh_res_obj = MoleculeResults(MoleculeInfo_obj=qooh,
                               DVRparams={"desired_energies": 8,
                                          "num_pts": 2000,
                                          "plot_phased_wfns": False,
                                          "extrapolate": 0.3},
                               PORparams={"HamSize": 15,
                                          "PrintResults": True,
                                          "Vexpansion": "sixth"},  # add "barrier_height" : ... to scale(one at a time)
                               Intenseparams={"numGstates": 4,
                                              "numEstates": 6,
                                              "FranckCondon": True},
                               transition=[f"0->{i}" for i in [1, 2, 3, 4, 5]])

qooh_res_obj.plot_degreeDVR()
