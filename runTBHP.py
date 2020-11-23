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

tbhp = MoleculeInfo(MoleculeName="TBHP",
                    atom_array=["C", "C", "H", "H", "O", "O", "H", "C", "H", "H", "H", "C", "H", "H", "H", "H"],
                    eqTORangle=247.79178000000002,
                    oh_scan_npz="TBHP_OH_energies_internals.npz",
                    TorFchkDirectory="TorFchks",
                    TorFiles=[f"tbhp_{i}.fchk" for i in ["000", "010", "020", "030", "040", "050", "060", "070",
                                                         "080", "090", "100", "110", "120", "130", "140", "150",
                                                         "160", "170", "180", "190", "200", "210", "220", "230",
                                                         "240", "250", "260", "270", "280", "290", "300", "310",
                                                         "320", "330", "340", "350", "360"]],
                    dipole_npz="rotated_dipoles_tbhp.npz")

tbhp_res_obj = MoleculeResults(MoleculeInfo_obj=tbhp,
                               DVRparams={"desired_energies": 8,
                                          "num_pts": 2000,
                                          "plot_phased_wfns": False,
                                          "extrapolate": 0.3},
                               PORparams={"HamSize": 15,
                                          "PrintResults": True,
                                          "Vexpansion": "sixth"},  # add "barrier_height" : ... to scale(one at a time)
                               Intenseparams={"numGstates": 4,
                                              "numEstates": 6},
                               transition=[f"0->{i}" for i in [1, 2, 3, 4, 5]])

def run_pot_plots():
    res, wfns = tbhp_res_obj.PORresults
    from PlotOHAdiabats import make_Potplots, make_Wfnplots, make_allWfnplots
    # rad_x = np.linspace(0, 2*np.pi, 100)
    # x = np.linspace(0, 360, 100)
    # ePot = tbhp_res_obj.RxnPathResults["electronicE"]
    # ePot[:, 1] = Constants.convert(ePot[:, 1]-min(ePot[:, 1]), "wavenumbers", to_AU=False)
    # plt.plot(ePot[:, 0], ePot[:, 1], "-r", linewidth=2.5)
    # # elPot = Constants.convert(calc_curves(rad_x, tbhp_res_obj.VelCoeffs), "wavenumbers", to_AU=False)
    # # plt.plot(x, elPot, '-b', linewidth=2.5)
    # gsPot = Constants.convert(calc_curves(rad_x, res[0]["V"]), "wavenumbers", to_AU=False)
    # plt.plot(x, gsPot, '-k', linewidth=2.5)
    # gsPot1 = Constants.convert(calc_curves(rad_x, res[1]["V"]), "wavenumbers", to_AU=False)
    # plt.plot(x, gsPot1, '-k', linewidth=2.5)
    # esPot = Constants.convert(calc_curves(rad_x, res[2]["V"]), "wavenumbers", to_AU=False)
    # plt.plot(x, esPot, '-k', linewidth=2.5)
    # plt.xticks(np.arange(0, 390, 60))
    # plt.ylabel(r"Energy (cm$^{-1}$)")
    # plt.xlabel(r"$\tau$")
    # plt.show()
    make_Potplots(res[0], res[2], ZPE=True, filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_wZPE"))
    # make_Potplots(res[0], res[2], ZPE=False, filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_woZPE"))
    # gsResDict, esResDict, wfnns, gs_idx=0, es_idx=1, filename=None
    # make_Wfnplots(res[0], res[2], wfns, filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_1overtonewfns"))
    # make_Wfnplots(res[0], res[2], wfns, upper_idx=(2, 3),
    #               filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_combowfns"))
    # make_Wfnplots(res[0], res[2], wfns, lower_idx=(2, 3), upper_idx=(4, 5),
    #               filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_hotcombowfns"))
    # make_allWfnplots(res[0], res[2], wfns, filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_allbuthotwfns"))

def tor_freq_plot_bh(barrier_heights, expvals, PORresults, true_bh, oh_levels=None):
    if oh_levels is None:  # OLD need to patch if going to use
        oh_levels = [2, 3, 4, 5]
    else:
        pass
    colors = ["r", "b", "g", "indigo"]
    for i, level in enumerate(oh_levels):
        plt_data = np.zeros((len(barrier_heights), 4))
        for j, bh in enumerate(barrier_heights):
            if bh is None:
                bh = true_bh
                torEnergies = PORresults["truebh"][level]["spacings"]
            else:
                torEnergies = PORresults[bh][level]["spacings"]  # 0+->1+, 0+->1-, 0-->1+, 0-->1-
            torEnergiesWave = Constants.convert(torEnergies, "wavenumbers", to_AU=False)
            plt_data[j] = np.column_stack((bh, torEnergiesWave[0], torEnergiesWave[1],
                                           np.average((torEnergiesWave[0], torEnergiesWave[1]))))
        plt.plot(plt_data[:, 0], plt_data[:, 1], 'o', color=colors[i], label=f"vOH = {level}")
        plt.plot(plt_data[:, 0], plt_data[:, 2], '^', color=colors[i])
        plt.plot(plt_data[:, 0], plt_data[:, 3], 'x', color=colors[i])
        plt.plot(barrier_heights[1:], np.repeat(expvals[i], len(barrier_heights[1:])), color=colors[i])
    plt.xlabel("Vel + R path Barrier Height")
    plt.ylabel("Torsion Frequency (cm^-1)")
    plt.legend(loc="lower right")
    plt.show()

def tor_freq_plot_sf(scaling_factors, expvals, IntensityDict, oh_levels=None):
    if oh_levels is None:
        oh_levels = [2, 3, 4, 5]
    else:
        pass
    colors = ["r", "b", "g", "indigo"]
    for i, level in enumerate(oh_levels):
        plt_data = np.zeros((len(scaling_factors), 4))
        for j, sf in enumerate(scaling_factors):
            if sf is None:
                sf = 1
                tor_intents_vals = np.array(IntensityDict["truebh"][f"0->{level}"])
            else:
                tor_intents_vals = np.array(IntensityDict[sf][f"0->{level}"])
            zero_zero = tor_intents_vals[0, 2] * tor_intents_vals[0, 5]
            zero_one = tor_intents_vals[1, 2] * tor_intents_vals[1, 5]
            one_zero = tor_intents_vals[6, 2] * tor_intents_vals[6, 5]
            one_one = tor_intents_vals[7, 2] * tor_intents_vals[7, 5]
            two_nuOH = np.sum((zero_zero, zero_one, one_one, one_zero)) / \
                       np.sum((tor_intents_vals[0, 5], tor_intents_vals[1, 5], tor_intents_vals[6, 5],
                              tor_intents_vals[7, 5]))
            zero_two = tor_intents_vals[2, 2] * tor_intents_vals[2, 5]
            zero_three = tor_intents_vals[3, 2] * tor_intents_vals[3, 5]
            one_two = tor_intents_vals[8, 2] * tor_intents_vals[8, 5]
            one_three = tor_intents_vals[9, 2] * tor_intents_vals[9, 5]
            two_four = tor_intents_vals[16, 2] * tor_intents_vals[16, 5]
            two_five = tor_intents_vals[17, 2] * tor_intents_vals[17, 5]
            three_four = tor_intents_vals[22, 2] * tor_intents_vals[22, 5]
            three_five = tor_intents_vals[23, 2] * tor_intents_vals[23, 5]
            two_nuOH_tor = np.sum((zero_two, zero_three, one_two, one_three,
                                       two_four, two_five, three_four, three_five)) / \
                           np.sum((tor_intents_vals[2, 5], tor_intents_vals[3, 5], tor_intents_vals[8, 5],
                                  tor_intents_vals[9, 5], tor_intents_vals[16, 5], tor_intents_vals[17, 5],
                                  tor_intents_vals[22, 5], tor_intents_vals[23, 5]))
            torEnergy = two_nuOH_tor - two_nuOH
            plt_data[j] = np.column_stack((sf, torEnergy, two_nuOH_tor, two_nuOH))
        plt.plot(plt_data[:, 0], plt_data[:, 1], 'o', color=colors[i], label=f"vOH = {level}")
        print(plt_data)
        expX = np.arange(0.45, 1.1, 0.05)
        plt.plot(expX, np.repeat(expvals[i], len(expX)), color=colors[i])
    plt.xlabel("Scaling Factor of Vel + R path Barrier")
    plt.ylabel("Torsion Frequency (cm^-1)")
    plt.legend(loc="lower right")
    plt.show()

def make_tor_freq_plot(exp_vals, bhs=None, sfs=None):
    IntenseDict = dict()
    IntenseDict["truebh"] = tbhp_res_obj.TransitionIntensities
    if sfs is None:
        for i in bhs[1:]:
            obj = MoleculeResults(MoleculeInfo_obj=tbhp,
                                   DVRparams={"desired_energies": 8,
                                              "num_pts": 2000,
                                              "plot_phased_wfns": False,
                                              "extrapolate": 0.3},
                                   PORparams={"HamSize": 15,
                                              "Vexpansion": "fourth",
                                              "barrier_height": i},
                                   Intenseparams={"numGstates": 4,
                                                  "numEstates": 6},
                                   transition=[f"0->{i}" for i in [2, 3, 4, 5]])
            IntenseDict[i] = obj.TransitionIntensities
        tor_freq_plot_bh(bhs, exp_vals, IntenseDict, true_bh=318.4198)  # FOR TBHP
    if bhs is None:
        for i in sfs[1:]:
            obj = MoleculeResults(MoleculeInfo_obj=tbhp,
                                  DVRparams={"desired_energies": 8,
                                             "num_pts": 2000,
                                             "plot_phased_wfns": False,
                                             "extrapolate": 0.3},
                                  PORparams={"HamSize": 15,
                                             "Vexpansion": "sixth",
                                             "PrintResults": False,
                                             "scaling_factor": i},
                                  Intenseparams={"numGstates": 4,
                                                 "numEstates": 6},
                                  transition=[f"0->{i}" for i in [2, 3, 4, 5]])
            IntenseDict[i] = obj.TransitionIntensities
        tor_freq_plot_sf(sfs, exp_vals, IntenseDict)

def calc_stationary_intensities():
    for i in ["1", "2", "3", "4", "5"]:
        obj = MoleculeResults(MoleculeInfo_obj=tbhp,
                               DVRparams={"desired_energies": 8,
                                          "num_pts": 2000,
                                          "plot_phased_wfns": False,
                                          "extrapolate": 0.3},
                               PORparams={"HamSize": 15},
                               Intenseparams={"numGstates": 4,
                                              "numEstates": 4},
                               transition=f"0->{i}")
        a = obj.StatTransitionIntensities

def run_emil_data(plot=False):
    emil_res = MoleculeResults(MoleculeInfo_obj=tbhp,
                               DVRparams={"desired_energies": 8,
                                          "num_pts": 2000,
                                          "plot_phased_wfns": False,
                                          "extrapolate": 0.3},
                               PORparams={"HamSize": 15,
                                          "EmilData": True,
                                          "Vexpansion": "fourth",
                                          "PrintResults": True},
                               Intenseparams={"numGstates": 4,
                                              "numEstates": 6},
                               transition=[f"0->{i}" for i in [2, 3, 4, 5]])
    if plot:
        res, wfns = emil_res.PORresults
        from PlotOHAdiabats import make_Potplots, make_Wfnplots
        make_Potplots(res[0], res[2], ZPE=True, filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_wZPE_CCSD"))
        make_Potplots(res[0], res[2], ZPE=False, filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_woZPE_CCSD"))
        make_Wfnplots(res[0], res[2], wfns,
                      filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_1overtonewfns_CCSD"))
        make_Wfnplots(res[0], res[2], wfns, upper_idx=(2, 3),
                      filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_combowfns_CCSD"))
        make_Wfnplots(res[0], res[2], wfns, lower_idx=(2, 3), upper_idx=(4, 5),
                      filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_hotcombowfns_CCSD"))
    return emil_res

def run_mixed_data(plot=False, type=1):
    if type == 1:
        por_params = {"HamSize": 15, "MixedData1": True, "PrintResults": True, "Vexpansion": "fourth"}
    elif type == 2:
        por_params = {"HamSize": 15, "MixedData2": True, "PrintResults": True, "Vexpansion": "fourth"}
    else:
        raise Exception(f"Can not mix with type {type}")
    mixed_res = MoleculeResults(MoleculeInfo_obj=tbhp,
                                DVRparams={"desired_energies": 8,
                                           "num_pts": 2000,
                                           "plot_phased_wfns": False,
                                           "extrapolate": 0.3},
                                PORparams=por_params,
                                Intenseparams={"numGstates": 4,
                                               "numEstates": 6},
                                transition="0->2")
    if plot:
        res, wfns = mixed_res.PORresults
        from PlotOHAdiabats import make_Potplots, make_Wfnplots
        # make_Potplots(res[0], res[2], ZPE=True, filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_wZPE_Mixed2"))
        # make_Potplots(res[0], res[2], ZPE=False, filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_woZPE_Mixed2"))
        # make_Wfnplots(res[0], res[2], wfns,
        #               filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_1overtonewfns_Mixed2"))
        # make_Wfnplots(res[0], res[2], wfns, upper_idx=(2, 3),
        #               filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_combowfns_Mixed2"))
        make_Wfnplots(res[0], res[2], wfns, lower_idx=(2, 3), upper_idx=(4, 5),
                      filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH02_hotcombowfns_Mixed2"))
    return mixed_res

def make_CCSD_tor_freq_plot_bh(exp_vals, bhs):  # OLD need to patch if going to use
    PORtest = dict()
    res_obj = run_emil_data()
    PORtest["truebh"] = res_obj.PORresults[0]
    for i in bhs[1:]:
        obj = MoleculeResults(MoleculeInfo_obj=tbhp,
                               DVRparams={"desired_energies": 8,
                                          "num_pts": 2000,
                                          "plot_phased_wfns": False,
                                          "extrapolate": 0.3},
                               PORparams={"HamSize": 15,
                                          "EmilData": True,
                                          "PrintResults": True,
                                          "barrier_height": i},
                               Intenseparams={"numGstates": 4,
                                              "numEstates": 4},
                               transition="0->2")
        PORtest[i] = obj.PORresults[0]
    tor_freq_plot_bh(bhs, exp_vals, PORtest, true_bh=275.2565)

def make_CCSD_tor_freq_plot_sf(exp_vals, sfs):
    IntenseDict = dict()
    res_obj = run_emil_data()
    IntenseDict["truebh"] = res_obj.TransitionIntensities
    for i in sfs[1:]:
        obj = MoleculeResults(MoleculeInfo_obj=tbhp,
                               DVRparams={"desired_energies": 8,
                                          "num_pts": 2000,
                                          "plot_phased_wfns": False,
                                          "extrapolate": 0.3},
                               PORparams={"HamSize": 15,
                                          "EmilData": True,
                                          "Vexpansion": "fourth",
                                          "scaling_factor": i},
                               Intenseparams={"numGstates": 4,
                                              "numEstates": 6},
                               transition=[f"0->{i}" for i in [2, 3, 4, 5]])
        IntenseDict[i] = obj.TransitionIntensities
    tor_freq_plot_sf(sfs, exp_vals, IntenseDict)

def make_Vel_plots(mol_res_obj):
    Vel = mol_res_obj.RxnPathResults["electronicE"][:, 1]
    Vel -= min(Vel)
    degrees = mol_res_obj.RxnPathResults["electronicE"][:, 0]
    plt.plot(degrees, Constants.convert(Vel, "wavenumbers", to_AU=False))
    print(np.column_stack((degrees, Constants.convert(Vel, "wavenumbers", to_AU=False))))
    Vel_ZPE = mol_res_obj.VelCoeffs
    velZPE = calc_curves(np.radians(np.arange(0, 360, 1)), Vel_ZPE)
    res = np.column_stack((np.arange(0, 360, 1), Constants.convert(velZPE, "wavenumbers", to_AU=False)))
    print(res[180])
    plt.plot(np.arange(0, 360, 1), Constants.convert(velZPE, "wavenumbers", to_AU=False))
    plt.show()

def make_Intensity_plots(mol_res_obj):
    from PlotOHAdiabats import make_PotWfnplots, make_one_Potplot, make_Potplots
    TMobj = TransitionMoments(mol_res_obj.DegreeDVRresults, mol_res_obj.PORresults, mol_res_obj.MoleculeInfo,
                                          transition=[f"0->{i}" for i in np.arange(1, 6)], MatSize=mol_res_obj.PORparams["HamSize"])
    # TMobj.plot_TDM(filename=os.path.join(tbhp.MoleculeDir, "figures", "TDM"))
    res, wfns = mol_res_obj.PORresults
    # make_one_Potplot(res[0], ZPE=False, filename=os.path.join(tbhp.MoleculeDir, "figures", "vOH0_levels"))
    # make_Potplots(res[0], res[2], ZPE=True)
    # make_PotWfnplots(res[0], wfns[0], ZPE=False, wfn_idx=[(0, 1), (2, 3), (4, 5)],
    #                  filename=os.path.join(tbhp.MoleculeDir, "figures", f"vOH0_allwfns"))
    # for i in np.arange(1, 6):
    #     make_PotWfnplots(res[i], wfns[i], wfn_idx=[(0, 1), (2, 3), (4, 5)],
    #                      filename=os.path.join(tbhp.MoleculeDir, "figures", f"vOH{i}_allwfns"))
    # make_one_Wfn_plot(res[2], wfns[2], idx=(2, 3))
    # make_PA_plots(wfns)
    
if __name__ == '__main__':
    from RotConstants import calc_all_RotConstants
    udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    tbhp_fdir = os.path.join(udrive, "TBHP", "TorFchks")
    bhs = [None, 225, 250, 275, 300, 325]
    sf = [None, 0.9, 0.8, 0.7, 0.6, 0.5]
    TBHPexp = [186, 198, 217, 262]
    # make_CCSD_tor_freq_plot_sf(TBHPexp, sfs=sf)
    # e = run_mixed_data(type=2, plot=True)
    # a = e.TransitionIntensities
    # run_pot_plots()
    # make_Vel_plots(tbhp_res_obj)
    # run_pot_plots()
    make_Intensity_plots(tbhp_res_obj)
    # a = tbhp_res_obj.TransitionIntensities
    # for i in np.arange(1, 6):
    #     calc_all_RotConstants(tbhp_fdir, torWfn_coefs=tbhp_res_obj.PORresults[0][i]["eigvecs"],
    #                           numstates=6,
    #                           vOH=f"{i}")
    a = tbhp_res_obj.Gmatrix
