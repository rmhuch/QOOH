import numpy as np
from TransitionMoments import *
from runTBHP import tbhp, tbhp_res_obj

def pull_TDM(mol_res_obj, transition):
    """calculates the TDM in the ABC axis system"""
    trans_mom_obj = TransitionMoments(mol_res_obj.DegreeDVRresults, mol_res_obj.PORresults, mol_res_obj.MoleculeInfo,
                                      transition=transition, MatSize=mol_res_obj.PORparams["HamSize"])
    TDM = TransitionMoments.calc_TDM(trans_mom_obj, transition)
    return TDM

def calc_new_axis(r_co, r_oo):
    """calculates the unit vectors for the new axis, OO, perpendicular to COO plane, and cross product of first two"""
    n_coo = np.cross(r_oo, r_co)
    p_coo = np.cross(r_oo, n_coo)
    new_axis = np.zeros((3, 3))
    for i, vec in enumerate([r_oo, n_coo, p_coo]):
        norm = np.linalg.norm(vec)
        new_axis[:, i] = vec/norm  # so that it is column-oriented
    # "rotation matrix" describes new axes in terms of the original ABC
    return new_axis

def calc_rotated_TDM(mol_obj, mol_res_obj, transition):
    """takes current ABC TDM and rotates to axis system defined above"""
    TDM = pull_TDM(mol_res_obj, transition)
    tor_degrees = np.linspace(0, 360, len(mol_obj.DipoleMomentSurface.keys())).astype(int)
    rot_TDM = np.zeros(TDM.shape)
    for i, tau in enumerate(tor_degrees):
        dipole_dat = mol_obj.DipoleMomentSurface[
            f"{mol_obj.MoleculeName.lower()}_{tau:0>3}.log"]
        coords = np.reshape(dipole_dat[:, 4:][12], (16, 3))
        r_co = coords[0, :] - coords[4, :]
        r_oo = coords[4, :] - coords[5, :]
        new_axis = calc_new_axis(r_co, r_oo)
        if tau == 110 or tau == 180:
            test = np.linalg.norm(r_oo)
            ans = r_oo / test
            print(ans)
            # print(new_axis)
        rot_TDM[i, :] = np.dot(TDM[i, :], new_axis)  # 1x3 dot 3x3
    return rot_TDM

def plot_TDM(mol_obj, mol_res_obj, filename=None):
    import matplotlib.pyplot as plt
    from FourierExpansions import calc_curves
    params = {'text.usetex': False,
              'mathtext.fontset': 'dejavusans',
              'font.size': 18}
    plt.rc('axes', labelsize=20)
    plt.rc('axes', labelsize=20)
    plt.rcParams.update(params)
    for Tstring in mol_res_obj.transition:
        fig = plt.figure(figsize=(7, 8), dpi=600)
        TDM = calc_rotated_TDM(mol_obj, mol_res_obj, Tstring)  # START HERE
        M_coeffs = TransitionMoments.calc_Mcoeffs(mol_obj, TDM)
        degrees = np.linspace(0, 360, len(TDM))
        plt.plot(degrees, np.repeat(0, len(degrees)), "-k", linewidth=3)
        colors = ["C0", "C1", "C2"]
        for c, val in enumerate(["A", "B", "C"]):
            # plt.plot(degrees, TDM[:, c] / 0.393456, 'o', color=colors[c], label=r"$\alpha$ = % s" % val)
            # plt.plot(degrees, TDM[:, c] / 0.393456, '-', color=colors[c], linewidth=3)
            if val == "C":
                line = calc_curves(np.radians(np.linspace(0, 360)), M_coeffs[:, c], function="sin")
                # print(Tstring, line[0]/ 0.393456, line[-1]/ 0.393456)
            else:
                line = calc_curves(np.radians(np.linspace(0, 360)), M_coeffs[:, c], function="cos")
            plt.plot(np.linspace(0, 360), line / 0.393456, "-", color=colors[c], linewidth=3,
                     label=r"$\alpha$ = % s" % val)
        plt.plot(np.repeat(113, 10), np.linspace(-0.1, 0.1, 10), '--', color="gray", label=r"$\tau_{eq}$")
        plt.plot(np.repeat(247, 10), np.linspace(-0.1, 0.1, 10), '--', color="gray")
        if Tstring[-1] is "1":
            plt.ylim(-0.07, 0.07)
            plt.legend(loc="lower left")
        elif Tstring[-1] is "2":
            plt.ylim(-0.020, 0.020)
            plt.legend(loc="upper left")
        elif Tstring[-1] is "3":
            plt.ylim(-0.004, 0.004)
            plt.legend(loc="upper left")
        elif Tstring[-1] is "4":
            plt.ylim(-0.001, 0.001)
            plt.legend(loc="lower left")
        elif Tstring[-1] is "5":
            plt.ylim(-0.0003, 0.0003)
            plt.legend(loc="lower left")
        plt.ylabel(r"$M^{\alpha}_{0 \rightarrow % s}$ (Debye)" % Tstring[-1])
        plt.xlim(-10, 370)
        plt.xlabel(r"$\tau$ (Degrees)")
        plt.xticks(np.arange(0, 390, 60))
        if filename is None:
            plt.show()
        else:
            fig.savefig(f"{filename}_{Tstring[0]}{Tstring[-1]}.jpg", dpi=fig.dpi, bbox_inches="tight")
            print(f"Figure saved to {filename}_{Tstring[0]}{Tstring[-1]}")

if __name__ == '__main__':
    plot_TDM(tbhp, tbhp_res_obj, filename="tbhp_rotated")
