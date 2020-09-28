from MolecularResults import *
import matplotlib.pyplot as plt
import numpy as np
from FourierExpansions import calc_curves
from Converter import Constants

tbhp = MoleculeInfo(MoleculeName="TBHP",
                    atom_array=["C", "C", "H", "H", "O", "O", "H", "C", "H", "H", "H", "C", "H", "H", "H", "H"],
                    eqTORangle=247.79178000000002,
                    oh_scan_npz="TBHP_OH_energies_internals.npz",
                    TorFchkDirectory="TorFchks",
                    TorFiles = [f"tbhp_{i}.fchk" for i in ["000", "010", "020", "030", "040", "050", "060", "070",
                                                           "080", "090", "100","110", "120", "130", "140", "150",
                                                           "160", "170", "180", "190", "200", "210", "220", "230",
                                                           "240", "250", "260", "270", "280", "290", "300", "310",
                                                           "320", "330", "340", "350", "360"]])

tbhp_res_obj = MoleculeResults(MoleculeInfo_obj=tbhp,
                               DVRparams={"desired_energies": 8,
                                          "num_pts": 2000,
                                          "plot_phased_wfns": False,
                                          "extrapolate": 0.3},
                               PORparams={"HamSize": 15})

res, wfns = tbhp_res_obj.run_tor_adiabats()
from PlotOHAdiabats import make_Potplots, make_Wfnplots
make_Potplots(res[0], res[2], ZPE=True, filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_wZPE"))
make_Potplots(res[0], res[2], ZPE=False, filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_woZPE"))
make_Wfnplots(res, wfns, doublet="lower", filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_wfns"))
