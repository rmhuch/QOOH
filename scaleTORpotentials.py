import numpy as np
from Converter import Constants

# These functions are implemented in the TorsionPOR class

def scale_barrier(energy_dat, barrier_height, scaling_factor):
    """finds the minima and cuts those points between, scales barrier and returns full data set back"""
    print(f"Beginning Electronic Barrier Scaling")
    energy_dat[:, 0] = np.radians(energy_dat[:, 0])
    mins = np.argsort(energy_dat[:, 1])
    mins = np.sort(mins[:2])
    center = energy_dat[mins[0]:mins[-1]+1, :]
    max_idx = np.argmax(center[:, 1])
    true_bh = center[max_idx, 1] - center[0, 1]
    print(f"True Barrier Height: {Constants.convert(true_bh, 'wavenumbers', to_AU=False)} cm ^-1")
    if barrier_height is None:  # scale based on scaling factor
        new_center = np.column_stack((center[:, 0], scaling_factor * center[:, 1]))
        max_idx = np.argmax(new_center[:, 1])
        scaled_bh = new_center[max_idx, 1] - new_center[0, 1]
        print(f"Scaled Barrier Height: {Constants.convert(scaled_bh, 'wavenumbers', to_AU=False)} cm ^-1")
        print(f"using a scaling factor of {scaling_factor}")
        left = energy_dat[0:mins[0], :]
        right = energy_dat[mins[-1] + 1:, :]
        scaled_energies = np.vstack((left, new_center, right))
    elif scaling_factor is None:  # scale to a defined barrier height
        barrier_har = Constants.convert(barrier_height, "wavenumbers", to_AU=True)
        scaling_factor = barrier_har / true_bh
        print(f"Scaling Factor: {scaling_factor}")
        print(f"Scaled Barrier Height: {barrier_height} cm^-1")
        new_center = np.column_stack((center[:, 0], scaling_factor*center[:, 1]))
        left = energy_dat[0:mins[0], :]
        right = energy_dat[mins[-1]+1:, :]
        scaled_energies = np.vstack((left, new_center, right))
    else:
        raise Exception("Can not scale with barrier_height or scaling_factor undefined")
    return scaling_factor, scaled_energies  # float64, [degree, energy]

def scale_full_barrier(energy_dat, barrier_height):
    """scales barrier of full data set"""
    print(f"Beginning Full Scaling to {barrier_height} cm^-1")
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
    scaled_energies = np.column_stack((energy_dat[:, 0], scaling_factor*energy_dat[:, 1]))
    return scaling_factor, scaled_energies  # float64, [degree, energy]

def calc_scaled_Vcoeffs(energy_dat, Vcoeffs, ens_to_calc, barrier_height, scaling_factor, order):
    from FourierExpansions import calc_cos_coefs, calc_4cos_coefs
    scaling_factor, scaled_energies = scale_barrier(energy_dat, barrier_height, scaling_factor)
    if order == 6:
        newVel = calc_cos_coefs(scaled_energies)
    elif order == 4:
        newVel = calc_4cos_coefs(scaled_energies)
    else:
        raise Exception(f"{order}th order expansions not currently supported")
    scaled_coeffs = dict()
    scaled_coeffs["Vel"] = newVel
    for i in np.arange(len(Vcoeffs.keys())-1):  # loop through saved energies
        scaled_coeffs[f"V{i}"] = newVel + Vcoeffs[f"V{i}"]
    return scaled_coeffs

