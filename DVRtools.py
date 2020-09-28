def wfn_flipper(wavefunctions_array, plotPhasedWfns=False, pot_array=None):
    """ Rephases output wavefunctions such that they are phased to the same orientation as the one before"""
    import numpy as np
    import matplotlib.pyplot as plt
    wfns = np.zeros((wavefunctions_array.shape[0], wavefunctions_array.shape[1], wavefunctions_array.shape[2]))
    for j in np.arange(wavefunctions_array.shape[2]):  # loop through wavefunctions and assign 0 adiabat
        wfns[0, :, j] = wavefunctions_array[0, :, j]
    for k in np.arange(wavefunctions_array.shape[0]):  # loop through adiabats
        if k >= 1:
            for j in np.arange(wavefunctions_array.shape[2]):  # loop through wavefunctions
                # calculate overlaps to check for phase consistency
                ovn = np.dot(wavefunctions_array[k - 1, :, j], wavefunctions_array[k, :, j])
                if ovn <= 0:
                    wfns[k, :, j] = wavefunctions_array[k, :, j] * -1
                else:
                    wfns[k, :, j] = wavefunctions_array[k, :, j]
    if plotPhasedWfns:
        for k in np.arange(len(wavefunctions_array)):
            x = pot_array[k, :, 0]
            for j in np.arange(wavefunctions_array.shape[2]):  # loop through wavefunctions
                plt.plot(x, wfns[k, :, j]+j)
        plt.show()
    return wfns


def ohWfn_plots(dvr_results, wfns2plt=3, degree=None, **kwargs):
    import matplotlib.pyplot as plt
    potz = dvr_results["potential"]
    eps = dvr_results["energies"]
    wfns = dvr_results["wavefunctions"]
    degrees = eps[:, 0]
    colors = ["violet", "plum", "orchid", "hotpink", "royalblue", "crimson"]
    for i, j in enumerate(degrees):
        plt.plot(potz[i, :, 0], potz[i, :, 1], '-k', linewidth=3.0)
        for k in range(wfns2plt):
            plt.plot(potz[i, :, 0], (wfns[i, :, k] * 8000) + eps[i, (k + 1)], colors[k], linewidth=2.0)
        if j > 1:
            plt.title(f"{j} Degrees")
        else:
            if degree is not None:
                plt.title(f"{degree} Degrees {j} Step size")
            else:
                plt.title(f"{j} Step Size")
        plt.ylim(0, 30000)
        plt.tight_layout()
        plt.show()
        plt.close()
