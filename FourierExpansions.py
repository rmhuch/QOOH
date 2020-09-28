import numpy as np
from scipy import optimize

# All of these functions are implemented through the MolecularInfo/MolecularResults class

def cos_func(x, c0, c1, c2, c3, c4, c5, c6):
    """returns a 6th order cos function based off passed x (angle values IN RADIANS)
    and coefficients from calc_cos_coefs"""
    return c0 + c1*np.cos(x) + c2*np.cos(2*x) + c3*np.cos(3*x) + c4*np.cos(4*x) + c5*np.cos(5*x) + c6*np.cos(6*x)

def calc_cos_coefs(energy_dat):
    """conducts a cos expansion to fit the energy curve given, returns sixth order coefs.
        :param energy_dat: [coordinate (RADIANS), energy]"""
    popt, pcov = optimize.curve_fit(cos_func, energy_dat[:, 0], energy_dat[:, 1])
    return popt

def sin_func(x, c0, c1, c2, c3, c4, c5, c6):
    """returns a 6th order sin function based off passed x (angle values IN RADIANS)
    and coefficients from calc_sin_coefs"""
    return c0 + c1*np.sin(x) + c2*np.sin(2*x) + c3*np.sin(3*x) + c4*np.sin(4*x) + c5*np.sin(5*x) + c6*np.sin(6*x)

def calc_sin_coefs(energy_dat):
    """conducts a sin expansion to fit the energy curve given, returns sixth order coefs.
        :param energy_dat: [coordinate (RADIANS), energy]"""
    popt, pcov = optimize.curve_fit(sin_func, energy_dat[:, 0], energy_dat[:, 1])
    return popt

def calc_curves(x, coefs, function="cos"):
    """calculates and plots the expansion based off passed x (angle values IN RADIANS) and coefficients"""
    if function == "cos":
        energies = cos_func(x, *coefs)
    elif function == "sin":
        energies = sin_func(x, *coefs)
    else:
        raise Exception(f"Can not use {function} as an expansion function.")
    return energies

