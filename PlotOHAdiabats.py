from FourierExpansions import *  # don't need the other import statements bc they are at the top of FourierExpansions.py

params = {'text.usetex': False,
          'mathtext.fontset': 'dejavusans',
          'font.size': 14}
plt.rcParams.update(params)

def make_Wfnplots(resDicts, doublet="lower", filename=None):
    """To do this we will make a broken axis plot.. good to keep in back pocket.."""
    # 'break' the y-axis into two portions - use the top (ax) for the upper level (resDicts[n])
    # and the bottom (ax2) for the lower level (resDicts[0])
    f, (ax, ax2) = plt.subplots(2, 1, sharex="all", figsize=(7, 8), dpi=350)
    for i, resDict in enumerate(resDicts):
        # complete analysis and gather data
        x = np.linspace(0, 360, 100)
        rad_x = np.linspace(0, 2*np.pi, 100)
        wfnns = PORwfns(resDict["eigvecs"], rad_x)
        if doublet == "lower":
            en0 = Constants.convert(resDict['energy'][0], "wavenumbers", to_AU=False)
            if i == 0:
                wfn0 = en0 + wfnns[:, 0]*-10  # *10 for aestheics only (on all wfn)
            else:
                wfn0 = en0 + wfnns[:, 0] * 10  # *10 for aestheics only (on all wfn)
            en1 = Constants.convert(resDict['energy'][1], "wavenumbers", to_AU=False)
            wfn1 = en1 + wfnns[:, 1]*10
        elif doublet == "upper":
            en0 = Constants.convert(resDict['energy'][2], "wavenumbers", to_AU=False)
            wfn0 = en0 + wfnns[:, 2]*10
            en1 = Constants.convert(resDict['energy'][3], "wavenumbers", to_AU=False)
            wfn1 = en1 + wfnns[:, 3]*10
        else:
            raise Exception(f"Can't evaluate a {doublet} doublet.")

        # plot the same data on both axes
        ax.plot(x, np.repeat(en0, len(x)), "-k", linewidth=3)
        ax.plot(x, np.repeat(en1, len(x)), "-k", linewidth=3)
        ax2.plot(x, np.repeat(en0, len(x)), "-k", linewidth=3)
        ax2.plot(x, np.repeat(en1, len(x)), "-k", linewidth=3)

        ax.plot(x, wfn0, "-r", linewidth=3)
        ax.plot(x, wfn1, "-b", linewidth=3)
        ax2.plot(x, wfn0, "-r", linewidth=3)
        ax2.plot(x, wfn1, "-b", linewidth=3)

    # zoom-in / limit the view to different portions of the data
    ax.set_ylim(9040, 9065)  # excited state (n = 2)
    ax2.set_ylim(1980, 2005)  # ground state

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.axes.set_xticks(np.arange(0, 390, 30))
    ax.xaxis.tick_top()  # create tick marks on top of plot
    ax.tick_params(labeltop=False, labelleft=False, left=False)  # only keep ticks along bottom
    ax2.tick_params(labelleft=False, left=False)  # only keep ticks along bottom
    ax2.xaxis.tick_bottom()
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # add big axis for labels
    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.ylabel(r"Energy (cm$^{-1}$)")
    plt.xlabel(r"$\tau$")
    if filename is None:
        plt.show()
    else:
        f.savefig(f"{filename}.png", dpi=f.dpi, bbox_inches="tight")


def make_Potplots(gsRes, esRes, ZPE=True, filename=None):
    """ Plot the potential curves and energy levels of the given transitions. If ZPE plots include the ZPE,
    else they are plotted with the ZPE subtracted off so that min(gsPot) = 0 """
    f, (ax, ax2) = plt.subplots(2, 1, sharex="all", figsize=(7, 8), dpi=350)
    x = np.linspace(0, 360, 100)
    rad_x = np.linspace(0, 2*np.pi, 100)
    # create gs plot
    gsPot = Constants.convert(evaluatePot(gsRes["V"], rad_x), "wavenumbers", to_AU=False)
    gsPotshift = gsPot - min(gsPot)
    en0 = Constants.convert(gsRes['energy'][0], "wavenumbers", to_AU=False)
    en1 = Constants.convert(gsRes['energy'][1], "wavenumbers", to_AU=False)
    en2 = Constants.convert(gsRes['energy'][2], "wavenumbers", to_AU=False)
    en3 = Constants.convert(gsRes['energy'][3], "wavenumbers", to_AU=False)
    enX = np.linspace(180 / 3, 2 * 180 - 180 / 3, 10)
    if ZPE:
        ax2.plot(x, gsPot, '-k', linewidth=2.5)
        ax2.plot(enX, np.repeat(en0, len(enX)), "-r", linewidth=2.5)
        ax2.plot(enX, np.repeat(en1, len(enX)), "--b", linewidth=2.5)
        ax2.plot(enX, np.repeat(en2, len(enX)), "-g", linewidth=2.5)
        ax2.plot(enX, np.repeat(en3, len(enX)), "--", color="indigo", linewidth=2.5)
    else:
        ax2.plot(x, gsPotshift, '-k', linewidth=2.5)
        ax2.plot(enX, np.repeat(en0 - min(gsPot), len(enX)), "-r", linewidth=2.5)
        ax2.plot(enX, np.repeat(en1 - min(gsPot), len(enX)), "--b", linewidth=2.5)
        ax2.plot(enX, np.repeat(en2 - min(gsPot), len(enX)), "-g", linewidth=2.5)
        ax2.plot(enX, np.repeat(en3 - min(gsPot), len(enX)), "--", color="indigo", linewidth=2.5)

    # create es plot
    esPot = Constants.convert(evaluatePot(esRes["V"], rad_x), "wavenumbers", to_AU=False)
    esPotshift = esPot - min(gsPot)  # subtract off ZPE
    en0 = Constants.convert(esRes['energy'][0], "wavenumbers", to_AU=False)
    en1 = Constants.convert(esRes['energy'][1], "wavenumbers", to_AU=False)
    en2 = Constants.convert(esRes['energy'][2], "wavenumbers", to_AU=False)
    en3 = Constants.convert(esRes['energy'][3], "wavenumbers", to_AU=False)
    enX = np.linspace(180 / 3, 2 * 180 - 180 / 3, 10)
    if ZPE:
        ax.plot(x, esPot, '-k', linewidth=2.5)
        ax.plot(enX, np.repeat(en0, len(enX)), "-r", linewidth=2.5)
        ax.plot(enX, np.repeat(en1, len(enX)), "--b", linewidth=2.5)
        ax.plot(enX, np.repeat(en2, len(enX)), "-g", linewidth=2.5)
        ax.plot(enX, np.repeat(en3, len(enX)), "--", color="indigo", linewidth=2.5)
    else:
        ax.plot(x, esPotshift, '-k', linewidth=2.5)
        ax.plot(enX, np.repeat(en0 - min(gsPot), len(enX)), "-r", linewidth=2.5)
        ax.plot(enX, np.repeat(en1 - min(gsPot), len(enX)), "--b", linewidth=2.5)
        ax.plot(enX, np.repeat(en2 - min(gsPot), len(enX)), "-g", linewidth=2.5)
        ax.plot(enX, np.repeat(en3 - min(gsPot), len(enX)), "--", color="indigo", linewidth=2.5)

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.axes.set_xticks(np.arange(0, 390, 30))
    ax.xaxis.tick_top()  # create tick marks on top of plot
    ax.tick_params(labeltop=False)  # get rid of ticks on top
    ax2.xaxis.tick_bottom()
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # add big axis for labels
    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.ylabel(r"Energy (cm$^{-1}$)", labelpad=15.0)
    plt.xlabel(r"$\tau$")
    if filename is None:
        plt.show()
    else:
        f.savefig(f"{filename}.png", dpi=f.dpi, bbox_inches="tight")

if __name__ == '__main__':
    Eres, res = run(levs_to_calc=3)
    # make_Wfnplots(res, doublet="lower", filename="VOH02_wfns")
    make_Potplots(res[0], res[2], filename="vOH02_wZPE")
    make_Potplots(res[0], res[2], ZPE=False, filename="vOH02_woZPE")
