import matplotlib.pyplot as plt
import numpy as np
from Converter import Constants

params = {'text.usetex': False,
          'mathtext.fontset': 'dejavusans',
          'font.size': 14}
plt.rcParams.update(params)

def make_Wfnplots(gsResDict, esResDict, wfnns, lower_idx=(0, 1), upper_idx=(0, 1), filename=None):
    """To do this we will make a broken axis plot.. good to keep in back pocket.."""
    # 'break' the y-axis into two portions - use the top (ax2) for the upper level (gsResDict)
    # and the bottom (ax) for the lower level (gsResDict)
    f, (ax2, ax) = plt.subplots(2, 1, sharex="all", figsize=(7, 8), dpi=350)
    # complete analysis and gather data
    x = np.linspace(0, 360, 100)
    en0g = Constants.convert(gsResDict['energy'][lower_idx[0]], "wavenumbers", to_AU=False)
    wfn0g = en0g + wfnns[0][:, lower_idx[0]]*10  # *10 for aestheics only (on all wfn)
    en1g = Constants.convert(gsResDict['energy'][lower_idx[1]], "wavenumbers", to_AU=False)
    wfn1g = en1g + wfnns[0][:, lower_idx[1]]*10

    en0e = Constants.convert(esResDict['energy'][upper_idx[0]], "wavenumbers", to_AU=False)
    wfn0e = en0e + wfnns[2][:, upper_idx[0]]*-10
    en1e = Constants.convert(esResDict['energy'][upper_idx[1]], "wavenumbers", to_AU=False)
    wfn1e = en1e + wfnns[2][:, upper_idx[1]]*-10
    # plot
    colors = ["b", "r", "g", "indigo", "teal", "mediumvioletred"]
    ax.plot(x, np.repeat(en0g, len(x)), "-k", linewidth=3)
    ax.plot(x, np.repeat(en1g, len(x)), "-k", linewidth=3)
    ax.plot(x, wfn0g, color=colors[lower_idx[0]], linewidth=3)
    ax.plot(x, wfn1g, color=colors[lower_idx[1]], linewidth=3)
    ax2.plot(x, np.repeat(en0e, len(x)), "-k", linewidth=3)
    ax2.plot(x, np.repeat(en1e, len(x)), "-k", linewidth=3)
    ax2.plot(x, wfn0e, color=colors[upper_idx[0]], linewidth=3)
    ax2.plot(x, wfn1e, color=colors[upper_idx[1]], linewidth=3)

    # hide the spines between ax and ax2
    ax2.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax2.axes.set_xticks(np.arange(0, 390, 60))
    ax2.xaxis.tick_top()  # create tick marks on top of plot
    ax2.tick_params(labeltop=False, labelleft=False, left=False)  # only keep ticks along bottom
    ax.tick_params(labelleft=False, left=False)  # only keep ticks along bottom
    ax.xaxis.tick_bottom()
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    ax2.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
    ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    # add big axis for labels
    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.ylabel("Energy")
    plt.xlabel(r"$\tau$")
    if filename is None:
        plt.show()
    else:
        f.savefig(f"{filename}.png", dpi=f.dpi, bbox_inches="tight")
        print(f"Figure saved to {filename}")

def make_allWfnplots(gsResDict, esResDict, wfnns, filename=None):
    """To do this we will make a broken axis plot.. good to keep in back pocket.."""
    # 'break' the y-axis into two portions - use the top (ax2) for the upper level (gsResDict)
    # and the bottom (ax) for the lower level (gsResDict)
    f, (ax2, ax) = plt.subplots(2, 1, sharex="all", figsize=(7, 8), dpi=350)
    # complete analysis and gather data
    x = np.linspace(0, 360, 100)
    en0g = Constants.convert(gsResDict['energy'][0], "wavenumbers", to_AU=False)
    wfn0g = en0g + wfnns[0][:, 0]*-50  # *10 for aestheics only (on all wfn)
    en1g = Constants.convert(gsResDict['energy'][1], "wavenumbers", to_AU=False)
    wfn1g = en1g + wfnns[0][:, 1]*50
    en2g = Constants.convert(gsResDict['energy'][2], "wavenumbers", to_AU=False)
    wfn2g = en2g + wfnns[0][:, 2]*-50  # *10 for aestheics only (on all wfn)
    en3g = Constants.convert(gsResDict['energy'][3], "wavenumbers", to_AU=False)
    wfn3g = en3g + wfnns[0][:, 3]*50

    en0e = Constants.convert(esResDict['energy'][0], "wavenumbers", to_AU=False)
    wfn0e = en0e + wfnns[2][:, 0]*-50
    en1e = Constants.convert(esResDict['energy'][1], "wavenumbers", to_AU=False)
    wfn1e = en1e + wfnns[2][:, 1]*50
    en2e = Constants.convert(esResDict['energy'][2], "wavenumbers", to_AU=False)
    wfn2e = en2e + wfnns[2][:, 2]*-50  # *10 for aestheics only (on all wfn)
    en3e = Constants.convert(esResDict['energy'][3], "wavenumbers", to_AU=False)
    wfn3e = en3e + wfnns[2][:, 3]*50
    # plot
    colors = ["b", "r", "g", "indigo"]
    ax.plot(x, np.repeat(en0g, len(x)), "-k", linewidth=3)
    ax.plot(x, np.repeat(en1g, len(x)), "-k", linewidth=3)
    ax.plot(x, wfn0g, color=colors[0], linewidth=3)
    ax.plot(x, wfn1g, color=colors[1], linewidth=3)
    # ax.plot(x, np.repeat(en2g, len(x)), "-k", linewidth=3)
    # ax.plot(x, np.repeat(en3g, len(x)), "-k", linewidth=3)
    # ax.plot(x, wfn2g, color=colors[2], linewidth=3)
    # ax.plot(x, wfn3g, color=colors[3], linewidth=3)
    ax2.plot(x, np.repeat(en0e, len(x)), "-k", linewidth=3)
    ax2.plot(x, np.repeat(en1e, len(x)), "-k", linewidth=3)
    ax2.plot(x, wfn0e, color=colors[0], linewidth=3)
    ax2.plot(x, wfn1e, color=colors[1], linewidth=3)
    ax2.plot(x, np.repeat(en2e, len(x)), "-k", linewidth=3)
    ax2.plot(x, np.repeat(en3e, len(x)), "-k", linewidth=3)
    ax2.plot(x, wfn2e, color=colors[2], linewidth=3)
    ax2.plot(x, wfn3e, color=colors[3], linewidth=3)

    ax.set_ylim(1900, 2400)
    ax2.set_ylim(8900, 9400)

    # hide the spines between ax and ax2
    ax2.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax2.axes.set_xticks(np.arange(0, 390, 60))
    ax2.xaxis.tick_top()  # create tick marks on top of plot
    ax2.tick_params(labeltop=False, labelleft=False, left=False)  # only keep ticks along bottom
    ax.tick_params(labelleft=False, left=False)  # only keep ticks along bottom
    ax.xaxis.tick_bottom()
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    ax2.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
    ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    # add big axis for labels
    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.ylabel("Energy")
    plt.xlabel(r"$\tau$")
    if filename is None:
        plt.show()
    else:
        f.savefig(f"{filename}.png", dpi=f.dpi, bbox_inches="tight")
        print(f"Figure saved to {filename}")


def make_Potplots(gsRes, esRes, ZPE=True, filename=None):
    """ Plot the potential curves and energy levels of the given transitions. If ZPE plots include the ZPE,
    else they are plotted with the ZPE subtracted off so that min(gsPot) = 0 """
    from FourierExpansions import calc_curves
    f, (ax, ax2) = plt.subplots(2, 1, sharex="all", figsize=(7, 8), dpi=350)
    x = np.linspace(0, 360, 100)
    rad_x = np.linspace(0, 2*np.pi, 100)
    # create gs plot
    if len(gsRes["V"]) < 7:
        gsRes["V"] = np.hstack((gsRes["V"], np.zeros(7 - len(gsRes["V"]))))
    gsPot = Constants.convert(calc_curves(rad_x, gsRes["V"]), "wavenumbers", to_AU=False)
    en0 = Constants.convert(gsRes['energy'][0], "wavenumbers", to_AU=False)
    en1 = Constants.convert(gsRes['energy'][1], "wavenumbers", to_AU=False)
    en2 = Constants.convert(gsRes['energy'][2], "wavenumbers", to_AU=False)
    en3 = Constants.convert(gsRes['energy'][3], "wavenumbers", to_AU=False)
    enX = np.linspace(180 / 3, 2 * 180 - 180 / 3, 10)
    if ZPE:
        ax2.plot(x, gsPot, '-k', linewidth=2.5)
        ax2.plot(enX, np.repeat(en0, len(enX)), "-b", linewidth=2.5)
        ax2.plot(enX, np.repeat(en1, len(enX)), "--r", linewidth=2.5)
        ax2.plot(enX, np.repeat(en2, len(enX)), "-g", linewidth=2.5)
        ax2.plot(enX, np.repeat(en3, len(enX)), "--", color="indigo", linewidth=2.5)
    else:
        gsPotshift = gsPot - min(gsPot)
        ax2.plot(x, gsPotshift, '-k', linewidth=2.5)
        ax2.plot(enX, np.repeat(en0 - min(gsPot), len(enX)), "-b", linewidth=2.5)
        ax2.plot(enX, np.repeat(en1 - min(gsPot), len(enX)), "--r", linewidth=2.5)
        ax2.plot(enX, np.repeat(en2 - min(gsPot), len(enX)), "-g", linewidth=2.5)
        ax2.plot(enX, np.repeat(en3 - min(gsPot), len(enX)), "--", color="indigo", linewidth=2.5)

    # create es plot
    if len(esRes["V"]) < 7:
        esRes["V"] = np.hstack((esRes["V"], np.zeros(7 - len(esRes["V"]))))
    esPot = Constants.convert(calc_curves(rad_x, esRes["V"]), "wavenumbers", to_AU=False)
    en0 = Constants.convert(esRes['energy'][0], "wavenumbers", to_AU=False)
    en1 = Constants.convert(esRes['energy'][1], "wavenumbers", to_AU=False)
    en2 = Constants.convert(esRes['energy'][2], "wavenumbers", to_AU=False)
    en3 = Constants.convert(esRes['energy'][3], "wavenumbers", to_AU=False)
    enX = np.linspace(180 / 3, 2 * 180 - 180 / 3, 10)
    if ZPE:
        ax.plot(x, esPot, '-k', linewidth=2.5)
        ax.plot(enX, np.repeat(en0, len(enX)), "-b", linewidth=2.5)
        ax.plot(enX, np.repeat(en1, len(enX)), "--r", linewidth=2.5)
        ax.plot(enX, np.repeat(en2, len(enX)), "-g", linewidth=2.5)
        ax.plot(enX, np.repeat(en3, len(enX)), "--", color="indigo", linewidth=2.5)
    else:
        esPotshift = esPot - min(gsPot)  # subtract off ZPE
        ax.plot(x, esPotshift, '-k', linewidth=2.5)
        ax.plot(enX, np.repeat(en0 - min(gsPot), len(enX)), "-b", linewidth=2.5)
        ax.plot(enX, np.repeat(en1 - min(gsPot), len(enX)), "--r", linewidth=2.5)
        ax.plot(enX, np.repeat(en2 - min(gsPot), len(enX)), "-g", linewidth=2.5)
        ax.plot(enX, np.repeat(en3 - min(gsPot), len(enX)), "--", color="indigo", linewidth=2.5)

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.axes.set_xticks(np.arange(0, 390, 60))
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
        print(f"Figure saved to {filename}")

def make_scaledPots(mol_res_obj):
    from FourierExpansions import calc_curves
    x = np.linspace(0, 2 * np.pi, 100)
    Vel = calc_curves(x, mol_res_obj.Velcoefs)
    newVel = mol_res_obj.Vcoeffs["Vel"]  # MolecularResults hold coefficient dict for 1 specific pot/scaling
    Vel_scaled = calc_curves(x, newVel)
    Vel_wave = Constants.convert(Vel, "wavenumbers", to_AU=False)
    Vel_scaled_wave = Constants.convert(Vel_scaled, "wavenumbers", to_AU=False)
    plt.plot(x, Vel_scaled_wave, "-g", label=f"Scaled Energy with Barrier {mol_res_obj.barrier_height} $cm^-1$")
    plt.plot(x, Vel_wave, "-k", label=f"Electronic Energy + R Path")
    plt.show()

if __name__ == '__main__':
    # make_Wfnplots(res, doublet="lower", filename="VOH02_wfns")
    make_Potplots(PORres[0], PORres[2], ZPE=False, filename="vOH02_woZPE")
