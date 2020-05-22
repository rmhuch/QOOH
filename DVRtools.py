def wfn_flipper(wavefunctions_array, plotPhasedWfns=False, pot_array=None):
    """ Rephases output wavefunctions such that they are phased to the same orientation as the one before"""
    import numpy as np
    import matplotlib.pyplot as plt
    wfns = np.zeros((len(wavefunctions_array), len(wavefunctions_array[0]), 4))
    for k in np.arange(len(wavefunctions_array)):
        gs_wfn = wavefunctions_array[k, :, 0]
        es_wfn = wavefunctions_array[k, :, 1]
        nes_wfn = wavefunctions_array[k, :, 2]
        # nnes_wfn = wavefunctions_array[k, :, 3]
        # calculate overlaps to check for phase consistency
        if k >= 1:
            gs_ovn = np.dot(wavefunctions_array[k - 1, :, 0], gs_wfn)
            if gs_ovn <= 0:
                gs_wfn *= -1
            es_ovn = np.dot(wavefunctions_array[k - 1, :, 1], es_wfn)
            if es_ovn <= 0:
                es_wfn *= -1
            nes_ovn = np.dot(wavefunctions_array[k - 1, :, 2], nes_wfn)
            if nes_ovn <= 0:
                nes_wfn *= -1
            # nnes_ovn = np.dot(wavefunctions_array[k - 1, :, 3], nnes_wfn)
            # if nnes_ovn <= 0:
            #     nnes_wfn *= -1
        wfns[k, :, 0] = gs_wfn
        wfns[k, :, 1] = es_wfn
        wfns[k, :, 2] = nes_wfn
        # wfns[k, :, 3] = nnes_wfn
    if plotPhasedWfns:
        for k in np.arange(len(wavefunctions_array)):
            x = pot_array[k, :, 0]
            plt.plot(x, wfns[k, :, 0])
            plt.plot(x, wfns[k, :, 1]+1)
            plt.plot(x, wfns[k, :, 2]+2)
            # plt.plot(x, wfns[k, :, 3]+3)
    return wfns


def ohWfn_plots(dvr_results, wfns2plt=3, degree=None, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    potz = dvr_results[0]
    eps = dvr_results[1]
    wfns = dvr_results[2]
    degrees = eps[:, 0]
    colors = ["royalblue", "crimson", "violet", "orchid", "plum", "hotpink"]
    for i, j in enumerate(degrees):
        # plt.rcParams.update({'font.size': 20})
        # plt.figure(figsize=(6, 6), dpi=300)
        plt.plot(potz[i, :, 0], potz[i, :, 1], '-k', linewidth=3.0)
        # minIdx = np.argmin(potz[i, :, 1])
        # print(f"{j} min : {potz[i, minIdx, 0]}")
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
