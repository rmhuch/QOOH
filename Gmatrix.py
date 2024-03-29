import numpy as np
from Converter import Constants


def calc_rOH_expectation(rOH_wfns):
    from ExpectationValues import calc_inverse_expectation, calc_inverse_expectation_squared
    inverse = calc_inverse_expectation(rOH_wfns)
    inverse2 = calc_inverse_expectation_squared(rOH_wfns)
    expectation_array = np.column_stack((inverse[:, 0], inverse[:, 1], inverse2[:, 1]))
    # [energy_level, <1/rOH>, <1/rOH^2>]
    return expectation_array  # values in bohr

def get_bonds_angles(tor_angles, internal_coords):
    # pulls in the bond lengths and angles needed for the g matrix
    value_dict = dict()  # ultimately will be a nested dict value_dict[tor_angle][value]
    for i in tor_angles:
        internal_vals = dict()
        internal_coord = internal_coords[i]
        internal_vals["r12"] = Constants.convert(internal_coord["B6"][12], "angstroms", to_AU=True)  # pulls eq_roh
        internal_vals["r23"] = Constants.convert(internal_coord["B5"], "angstroms", to_AU=True)
        internal_vals["r34"] = Constants.convert(internal_coord["B4"], "angstroms", to_AU=True)
        internal_vals["phi123"] = np.radians(internal_coord["A5"])
        internal_vals["phi234"] = np.radians(internal_coord["A4"])
        internal_vals["tau1234"] = np.radians(internal_coord["D4"])
        value_dict[i] = internal_vals
    return value_dict  # values in bohr/radians

def get_tor_gmatrix(rOH_wfns, tor_angles, internal_coords, masses):
    """based off of g_tau,tau from Frederick and Woywod J.Chem.Phys., doi: 140.254.87.101
        TBHP/QOOH: 1-H, 2-O, 3-O, 4-C"""
    # this calculates the gmatrix element for the torsion as a function of the roh expectation value
    # masses is in atomic units!
    expectation_array = calc_rOH_expectation(rOH_wfns)
    internal_dict = get_bonds_angles(tor_angles, internal_coords)
    g_matrix = np.zeros((len(expectation_array[:, 0]), len(tor_angles), 2))  # [level,[num tor points, gmatrix element]]
    for level in expectation_array[:, 0]:
        for i, angle in enumerate(tor_angles):
            g_matrix[int(level), i, 0] = angle
            ic = internal_dict[angle]
            lambda_123 = (1/np.sin(ic["phi123"]))*(expectation_array[int(level), 1] - (np.cos(ic["phi123"])/ic["r23"]))
            lambda_432 = (1/np.sin(ic["phi234"]))*(1/ic["r34"] - (np.cos(ic["phi234"])/ic["r23"]))
            g_matrix[int(level), i, 1] = expectation_array[int(level), 2] * \
                                         (1 / (masses[0] * np.sin(ic["phi123"]) ** 2)) + \
                                          1 / (masses[3] * ic["r34"] ** 2 * np.sin(ic["phi234"]) ** 2) + \
                                 1 / masses[1] * (lambda_123 ** 2 + (1/np.tan(ic["phi234"]) ** 2) / ic["r23"] ** 2) + \
                                 1 / masses[2] * (lambda_432 ** 2 + (1/np.tan(ic["phi123"]) ** 2) / ic["r23"]**2) - \
                                          2 * (np.cos(ic["tau1234"])/ic["r23"]) * \
                                          ((lambda_123/masses[1])*1/np.tan(ic["phi234"]) +
                                            (lambda_432/masses[2])*1/np.tan(ic["phi123"]))
    return g_matrix

def make_gmat_plot(g_matrix):
    import matplotlib.pyplot as plt
    params = {'text.usetex': False,
              'mathtext.fontset': 'dejavusans',
              'font.size': 14}
    plt.rcParams.update(params)
    fig = plt.figure(figsize=(4, 4), dpi=600)
    # colors = ["m", "darkturquoise", "mediumseagreen", "indigo", "goldenrod", "r", "b"]
    colors = ["b", "r", "goldenrod", "indigo", "mediumseagreen", "darkturquoise", "m"]
    for level in np.arange(6):
        y = g_matrix[level, :, 1]/(9.1093837015e-28*6.02214076e23)*1.88973**2
        y_full = [y[-1], *y]
        x_full = [0, *g_matrix[level, :, 0]]
        plt.plot(x_full, y_full,
                 color=colors[level], label="v$_{OH}$ = % s" % level, linewidth=2.5)
    plt.xlabel(r"$\tau$ (Degrees)")
    plt.xticks(np.arange(0, 450, 90))
    plt.xlim(0, 360)
    plt.ylim(0.9, 1.4)
    plt.ylabel(r"G$_{\tau\tau}$ (1/amu$\AA^2$)")
    # plt.legend()
    plt.savefig("DFT_Gmatrix.jpg", dpi=fig.dpi, bbox_inches="tight")

def get_eq_g(tor_angles, internal_coords, masses):
    """ KEEP FOR NOW. Need to create g-matrix  plot. NOT USED IN CALCULATIONS.
        based off of g_tau,tau from Frederick and Woywod J.Chem.Phys., doi: 140.254.87.101
        TBHP/QOOH: 1-H, 2-O, 3-O, 4-C"""
    # this calculates the gmatrix element for the torsion as a function of the roh expectation value
    # masses is in atomic units!
    internal_dict = get_bonds_angles(internal_coords.keys(), internal_coords)
    g_matrix = np.zeros((len(tor_angles), 2))  # [[num tor points, gmatrix element]]
    for i, angle in enumerate(tor_angles):
        g_matrix[i, 0] = angle
        ic = internal_dict[angle]
        lambda_123 = (1/np.sin(ic["phi123"]))*(1/ic["r12"] - (np.cos(ic["phi123"])/ic["r23"]))
        lambda_432 = (1/np.sin(ic["phi234"]))*(1/ic["r34"] - (np.cos(ic["phi234"])/ic["r23"]))
        g_matrix[i, 1] = 1/ic["r12"]**2 * \
                                     (1 / (masses[0] * np.sin(ic["phi123"]) ** 2)) + \
                                      1 / (masses[3] * ic["r34"] ** 2 * np.sin(ic["phi234"]) ** 2) + \
                             1 / masses[1] * (lambda_123 ** 2 + (1/np.tan(ic["phi234"]) ** 2) / ic["r23"] ** 2) + \
                             1 / masses[2] * (lambda_432 ** 2 + (1/np.tan(ic["phi123"]) ** 2) / ic["r23"]**2) - \
                                      2 * (np.cos(ic["tau1234"])/ic["r23"]) * \
                                      ((lambda_123/masses[1])*1/np.tan(ic["phi234"]) +
                                        (lambda_432/masses[2])*1/np.tan(ic["phi123"]))
    return g_matrix
