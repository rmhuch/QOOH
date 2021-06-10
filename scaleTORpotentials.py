import numpy as np
from Converter import Constants

# These functions are implemented in the TorsionPOR class

def scale_barrier(energy_dat, barrier_height, scaling_factor):
    """finds equivalent minima and cuts those points between, scales barrier and returns full data set back"""
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

def scale_uneven_barrier(energy_dat, barrier_height):
    """ scales maximum to given value, by scaling the left to the left minimum and right to right minimum
        energy_dat: 2D array - torsion angles (degrees), energies (hartree)
        barrier_height: final barrier height (wavenumbers)"""
    print(f"Scaling Electronic Barrier to {barrier_height} cm^-1")
    bh_au = Constants.convert(barrier_height, "wavenumbers", to_AU=True)
    center_args = np.nonzero(energy_dat[:, 0] == 100)[0], np.nonzero(energy_dat[:, 0] == 260)[0]
    max_arg = np.argmax(energy_dat[int(center_args[0]):int(center_args[1]), 1]) + int(center_args[0])
    leftM_arg = np.argmin(energy_dat[:max_arg, 1])
    rightM_arg = np.argmin(energy_dat[max_arg:, 1]) + max_arg
    # scale LHS
    DeltaEl = energy_dat[max_arg, 1] - energy_dat[leftM_arg, 1]
    beta = (bh_au - energy_dat[leftM_arg, 1]) / DeltaEl
    scaled_left = ((energy_dat[leftM_arg:max_arg, 1] - energy_dat[leftM_arg, 1]) * beta) + energy_dat[leftM_arg, 1]
    left_dat = np.column_stack((energy_dat[leftM_arg:max_arg, 0], scaled_left))
    # scale RHS
    DeltaEr = energy_dat[max_arg, 1] - energy_dat[rightM_arg, 1]
    alpha = (bh_au - energy_dat[rightM_arg, 1]) / DeltaEr
    scaled_right = ((energy_dat[max_arg:rightM_arg, 1] - energy_dat[rightM_arg, 1]) * alpha) + energy_dat[rightM_arg, 1]
    right_dat = np.column_stack((energy_dat[max_arg:rightM_arg, 0], scaled_right))
    # stack together
    scaling_factors = (beta, alpha)  # return L/R scaling factors in tuple
    scaled_energies = np.vstack((energy_dat[:leftM_arg, :], left_dat, right_dat, energy_dat[rightM_arg:, :]))
    return scaling_factors, scaled_energies

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



