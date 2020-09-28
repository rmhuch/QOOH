import numpy as np
from Converter import Constants

# These functions are implemented in the TorsionPOR class

def scale_barrier(energy_dat, barrier_height):
    """finds the minima and cuts those points between, scales barrier and returns full data set back"""
    print(f"Beginning Scaling to {barrier_height} cm^-1")
    energy_dat[:, 0] = np.radians(energy_dat[:, 0])
    mins = np.argsort(energy_dat[:, 1])
    mins = np.sort(mins[:2])
    center = energy_dat[mins[0]:mins[-1]+1, :]
    max_idx = np.argmax(center[:, 1])
    true_bh = center[max_idx, 1] - center[0, 1]
    print(f"True Barrier Height: {Constants.convert(true_bh, 'wavenumbers', to_AU=False)} cm ^-1")
    barrier_har = Constants.convert(barrier_height, "wavenumbers", to_AU=True)
    scaling_factor = barrier_har / true_bh
    print(f"Scaling Factor: {scaling_factor}")
    new_center = np.column_stack((center[:, 0], scaling_factor*center[:, 1]))
    left = energy_dat[0:mins[0], :]
    right = energy_dat[mins[-1]+1:, :]
    scaled_energies = np.vstack((left, new_center, right))
    return scaling_factor, scaled_energies  # float64, [degree, energy]

def calc_scaled_Vcoeffs(energy_dat, Vcoeffs, barrier_height, ens_to_calc):
    from FourierExpansions import calc_cos_coefs
    scaling_factor, scaled_energies = scale_barrier(energy_dat, barrier_height)
    newVel = calc_cos_coefs(scaled_energies)
    scaled_coeffs = dict()
    scaled_coeffs["Vel"] = newVel
    for i in np.arange(ens_to_calc):  # loop through saved energies
        scaled_coeffs[f"V{i + 1}"] = (newVel + Vcoeffs[f"V{i + 1}"])
    return scaled_coeffs

