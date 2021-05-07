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

def cos4_func(x, c0, c1, c2, c3, c4):
    """returns a 6th order cos function based off passed x (angle values IN RADIANS)
    and coefficients from calc_cos_coefs"""
    return c0 + c1*np.cos(x) + c2*np.cos(2*x) + c3*np.cos(3*x) + c4*np.cos(4*x)

def calc_4cos_coefs(energy_dat):
    """conducts a cos expansion to fit the energy curve given, returns sixth order coefs.
        :param energy_dat: [coordinate (RADIANS), energy]"""
    popt, pcov = optimize.curve_fit(cos4_func, energy_dat[:, 0], energy_dat[:, 1])
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

def fourier_func(x, cos_coeffs, sin_coeffs):
    """
    Evals a Fourier series
    """
    cos_evals = sum(c * np.cos(i * x) for i, c in enumerate(cos_coeffs))
    # we assume sin_coeffs has explicitly dropped the 0 term
    sin_evals = sum(c * np.sin((i + 1) * x) for i, c in enumerate(sin_coeffs))
    return cos_evals + sin_evals

def fourier_fitter(x, *coeffs, cos_order=6):
    """
    Evaluates a Fourier series so scipy can fit it.
    Assume coeffs starts with the cos part and then follows with the
    sin part
    """
    cos_coeffs = coeffs[:cos_order+1]
    sin_coeffs = coeffs[cos_order+1:]
    return fourier_func(x, cos_coeffs, sin_coeffs)

def fourier_coeffs(energy_dat, cos_order=6, sin_order=6):
    """
    ....
    """
    initial_guess = np.random.rand(cos_order + sin_order)
    # array of length n coeffs (sin + cos) for scipy to work from
    popt, pcov = optimize.curve_fit(fourier_fitter, energy_dat[:, 0], energy_dat[:, 1], p0=initial_guess)
    cos_coeffs = popt[:cos_order+1]
    sin_coeffs = popt[cos_order+1:]
    return cos_coeffs, sin_coeffs

def fourier_barrier(coefs):
    """
    uses scipy minimize/maximize to calculate the barrier of a given potential and its proper minima.
    """
    res = np.zeros((3, 2))
    min1Res = optimize.minimize(fourier_func, x0=np.radians(110), args=(coefs[0], coefs[1]))
    res[0, 0] = min1Res["x"][0]
    res[0, 1] = fourier_func(res[0, 0], coefs[0], coefs[1])
    min2Res = optimize.minimize(fourier_func, x0=np.radians(270), args=(coefs[0], coefs[1]))
    res[1, 0] = min2Res["x"][0]
    res[1, 1] = fourier_func(res[1, 0], coefs[0], coefs[1])
    centerx = np.linspace(res[0, 0], res[1, 0], 10000)
    center = calc_curves(centerx, coefs, function="fourier")
    res[2, 0] = centerx[np.argmax(center)]
    res[2, 1] = center[np.argmax(center)]
    return res

def calc_curves(x, coefs, function="cos"):
    """calculates the expansion function based off passed x (angle values IN RADIANS) and coefficients"""
    if function == "cos":
        energies = cos_func(x, *coefs)
    elif function == "4cos":
        energies = cos4_func(x, *coefs)
    elif function == "sin":
        energies = sin_func(x, *coefs)
    elif function == "fourier":
        energies = fourier_func(x, coefs[0], coefs[1])
    else:
        raise Exception(f"Can not use {function} as an expansion function.")
    return energies

def calc_derivs(x, coefs, function="cos"):
    """calculates the derivative of a sin or cos expansion given x (IN RADIANS) and expansion coefficients"""
    if function == "cos":
        derivs = np.dot(-coefs, [n*np.sin(n*x) for n in range(len(coefs))])
    elif function == "sin":
        derivs = np.dot(coefs, [n*np.cos(n*x) for n in range(len(coefs))])
    else:
        raise Exception(f"Can not calculate derivatives of a {function} function.")
    return derivs
