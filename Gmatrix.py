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
    print(expectation_array)
    for level in expectation_array[:, 0]:
        for i, angle in enumerate(tor_angles):
            g_matrix[int(level), i, 0] = angle
            ic = internal_dict[angle]
            lambda_123 = (1/np.sin(ic["phi123"]))*(expectation_array[int(level), 1] - (np.cos(ic["phi123"])/ic["r23"]))
            lambda_432 = (1/np.sin(ic["phi234"]))*(expectation_array[int(level), 1] - (np.cos(ic["phi234"])/ic["r34"]))
            g_matrix[int(level), i, 1] = expectation_array[int(level), 2] * \
                                         (1 / (masses[0] * np.sin(ic["phi123"]) ** 2)) + \
                                          1 / (masses[3] * ic["r34"] ** 2 * np.sin(ic["phi234"]) ** 2) + \
                                 1 / masses[1] * (lambda_123 ** 2 + (np.arctan(ic["phi234"]) ** 2) / ic["r23"] ** 2) + \
                                 1 / masses[2] * (lambda_432 ** 2 + (np.arctan(ic["phi123"]) ** 2) / ic["r23"]**2) - \
                                          2 * (np.cos(ic["tau1234"])/ic["r23"]) * \
                                          ((lambda_123/masses[1])*np.arctan(ic["phi234"]) +
                                            (lambda_432/masses[2])*np.arctan(ic["phi123"]))
    return g_matrix

