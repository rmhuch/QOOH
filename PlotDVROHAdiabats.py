import matplotlib.pyplot as plt
import numpy as np
from Converter import Constants

params = {'text.usetex': False,
          'mathtext.fontset': 'dejavusans',
          'font.size': 10}
plt.rcParams.update(params)

def make_Wfnplots(gsResDict, esResDict, lower_idx=(0, 1), upper_idx=(0, 1), filename=None):
    """To do this we will make a broken axis plot.. good to keep in back pocket.."""
    # 'break' the y-axis into two portions - use the top (ax2) for the upper level (gsResDict)
    # and the bottom (ax) for the lower level (gsResDict)
    f, (ax2, ax) = plt.subplots(2, 1, sharex="all", figsize=(7, 8), dpi=600)
    # complete analysis and gather data
    x = np.linspace(0, 360, 100)
    en0g = Constants.convert(gsResDict['energy'][lower_idx[0]], "wavenumbers", to_AU=False)
    wfn0g = gsResDict['eigvecs'][:, lower_idx[0]]
    if wfn0g[21] < wfn0g[0]:
        wfn0g *= -1
    wfn0g = en0g + wfn0g * 10  # *10 for aestheics only (on all wfn)
    en1g = Constants.convert(gsResDict['energy'][lower_idx[1]], "wavenumbers", to_AU=False)
    wfn1g = gsResDict['eigvecs'][:, lower_idx[1]]
    if wfn1g[21] < wfn1g[0]:
        wfn1g *= -1
    wfn1g = en1g + wfn1g * 10  # *10 for aestheics only (on all wfn)

    en0e = Constants.convert(esResDict['energy'][upper_idx[0]], "wavenumbers", to_AU=False)
    wfn0e = esResDict['eigvecs'][:, upper_idx[0]]
    if wfn0e[21] < wfn0e[0]:
        wfn0e *= -1
    wfn0e = en0e + wfn0e * 10  # *10 for aestheics only (on all wfn)
    en1e = Constants.convert(esResDict['energy'][upper_idx[1]], "wavenumbers", to_AU=False)
    wfn1e = esResDict['eigvecs'][:, upper_idx[1]]
    if wfn1e[21] < wfn1e[0]:
        wfn1e *= -1
    wfn1e = en1e + wfn1e * 10  # *10 for aestheics only (on all wfn)
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
        f.savefig(f"{filename}.jpg", dpi=f.dpi, bbox_inches="tight")
        print(f"Figure saved to {filename}")
        plt.close()

def make_one_Wfn_plot(ResDict, idx=(0, 1)):
    # complete analysis and gather data
    x = np.linspace(0, 360, 100)
    en0g = Constants.convert(ResDict['energy'][idx[0]], "wavenumbers", to_AU=False)
    wfn0g = ResDict['eigvecs'][:, idx[0]]  # *10 for aestheics only (on all wfn)
    en1g = Constants.convert(ResDict['energy'][idx[1]], "wavenumbers", to_AU=False)
    wfn1g = ResDict['eigvecs'][:, idx[1]]
    # plot
    colors = ["b", "r", "g", "indigo", "teal", "mediumvioletred"]
    plt.plot(x, np.repeat(0, len(x)), "-k", linewidth=3)
    # plt.plot(x, np.repeat(en1g, len(x)), "-k", linewidth=3)
    plt.plot(x, wfn0g, color=colors[idx[0]], linewidth=3)
    plt.plot(x, wfn1g, color=colors[idx[1]], linewidth=3)
    plt.ylim(-1, 1)
    plt.xlim(-10, 370)
    plt.xticks(np.arange(0, 390, 30))
    plt.show()

def make_PA_plots(wfnns):
    x = np.linspace(0, 360, 100)
    colors = ["b", "r", "g", "indigo", "teal", "mediumvioletred"]
    # for wfn in np.arange(5):
    for wfn in [5]:
        shift = 0
        for i in [0, 1, 2, 3, 4, 5]:
            plt.plot(x, wfnns[wfn][:, i]**2+shift, color=colors[i], label=f"{i}")
            plt.fill_between(x, wfnns[wfn][:, i]**2+shift, shift, facecolor=colors[i], color=colors[i], alpha=0.2)
            shift += 0.05
        plt.title(f"vOH = {wfn}")
        plt.legend()
        plt.xlim(-10, 370)
        plt.xticks(np.arange(0, 390, 30))
        plt.show()

def make_Potplots(gsRes, esRes, numStates=4, potfunc="cos", ZPE=True, filename=None):
    """ Plot the potential curves and energy levels of the given transitions. If ZPE plots include the ZPE,
    else they are plotted with the ZPE subtracted off so that min(gsPot) = 0 """
    from FourierExpansions import calc_curves
    f, (ax, ax2) = plt.subplots(2, 1, sharex="all", figsize=(7, 8), dpi=600)
    x = np.linspace(0, 360, 100)
    rad_x = np.linspace(0, 2*np.pi, 100)
    # create gs plot
    if len(gsRes["V"]) < 7:
        gsRes["V"] = np.hstack((gsRes["V"], np.zeros(7 - len(gsRes["V"]))))
        esRes["V"] = np.hstack((esRes["V"], np.zeros(7 - len(esRes["V"]))))
    gsPot = Constants.convert(calc_curves(rad_x, gsRes["V"], function=potfunc), "wavenumbers", to_AU=False)
    esPot = Constants.convert(calc_curves(rad_x, esRes["V"], function=potfunc), "wavenumbers", to_AU=False)
    enX = np.linspace(180 / 3, 2 * 180 - 180 / 3, 10)
    colors = ["b", "r", "g", "indigo", "teal", "mediumvioletred"]
    for i in np.arange(numStates):
        en = Constants.convert(gsRes['energy'][i], "wavenumbers", to_AU=False)
        ene = Constants.convert(esRes['energy'][i], "wavenumbers", to_AU=False)
        if ZPE is False:
            en -= min(gsPot)
            ene -= min(gsPot)
        if i % 2 == 0:  # if even (lower)
            ax2.plot(enX, np.repeat(en, len(enX)), "-", color=colors[i], linewidth=2.5)
            ax.plot(enX, np.repeat(ene, len(enX)), "-", color=colors[i], linewidth=2.5)
        else:  # if odd (upper)
            ax2.plot(enX, np.repeat(en, len(enX)), "--", color=colors[i], linewidth=2.5)
            ax.plot(enX, np.repeat(ene, len(enX)), "--", color=colors[i], linewidth=2.5)
    if ZPE is False:
        esPot -= min(gsPot)
        gsPot -= min(gsPot)
    ax2.plot(x, gsPot, '-k', linewidth=2.5, zorder=1)
    ax.plot(x, esPot, '-k', linewidth=2.5, zorder=1)

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
    plt.xlabel(r"$\tau (Degrees)$")
    if filename is None:
        plt.show()
    else:
        f.savefig(f"{filename}.jpg", dpi=f.dpi, bbox_inches="tight")
        print(f"Figure saved to {filename}")
        plt.close()

def make_one_Potplot(ResDict, ZPE=False, filename=None):
    """send in the res dict of the one level you want"""
    from FourierExpansions import calc_curves
    fig = plt.figure(figsize=(7, 8), dpi=600)
    x = np.linspace(0, 360, 100)
    rad_x = np.linspace(0, 2*np.pi, 100)
    Pot = Constants.convert(calc_curves(rad_x, ResDict["V"], function="fourier"), "wavenumbers", to_AU=False)
    enX = np.linspace(180 / 3, 2 * 180 - 180 / 3, 10)
    colors = ["b", "r", "g", "indigo", "teal", "mediumvioletred"]
    for i in np.arange(6):
        en = Constants.convert(ResDict['energy'][i], "wavenumbers", to_AU=False)
        if ZPE is False:
            en -= min(Pot)
        if i % 2 == 0:  # if even (lower)
            plt.plot(enX, np.repeat(en, len(enX)), "-", color=colors[i], linewidth=2.5)
        else:  # if odd (upper)
            plt.plot(enX, np.repeat(en, len(enX)), "--", color=colors[i], linewidth=2.5)
    if ZPE is False:
        Pot -= min(Pot)
    plt.plot(x, Pot, '-k', linewidth=2.5, zorder=1)
    plt.ylabel(r"V($\tau$) [cm$^{-1}$]", labelpad=15.0)
    plt.ylim(0, 610)
    plt.xlabel(r"$\tau$ [Degrees]")
    plt.xticks(np.arange(60, 360, 60))
    plt.xlim(60, 300)
    if filename is None:
        plt.show()
    else:
        fig.savefig(f"{filename}.jpg", dpi=fig.dpi, bbox_inches="tight")
        print(f"Figure saved to {filename}")
        plt.close()

def make_pot_comp_plot(fullporRes, filename=None):
    from FourierExpansions import calc_curves
    fig = plt.figure(figsize=(4, 4), dpi=600)
    x = np.linspace(0, 360, 100)
    rad_x = np.linspace(0, 2 * np.pi, 100)
    colors = ["b", "r", "goldenrod", "indigo", "mediumseagreen", "darkturquoise"]
    for i, porRes in enumerate(fullporRes):
        if len(porRes["V"]) < 7:
            porRes["V"] = np.hstack((porRes["V"], np.zeros(7 - len(porRes["V"]))))
        Pot = Constants.convert(calc_curves(rad_x, porRes["V"]), "wavenumbers", to_AU=False)
        Pot -= min(Pot)
        plt.plot(x, Pot, color=colors[i], label=r"v$_\mathrm{OH}$ = % s" % i)
    plt.ylabel(r"$V_{\mathrm{v_{OH}}}^\mathrm{eff.}$($\tau$) - "
               r"$V_{\mathrm{v_{OH}}}^\mathrm{eff.}$($\tau_\mathrm{min}$)(cm$^{-1}$)")
    plt.xlabel(r"$\tau$ (Degrees)")
    plt.xticks(np.arange(0, 450, 90))
    plt.xlim(0, 360)
    plt.legend()
    if filename is None:
        plt.show()
    else:
        fig.savefig(f"{filename}.jpg", dpi=fig.dpi, bbox_inches="tight")
        print(f"Figure saved to {filename}")
        plt.close()


def make_PotWfnplots(ResDict, wfn_idx=[0, 1], ZPE=True, filename=None):
    """ Plot the potential curves and energy levels of the given transitions. If ZPE plots include the ZPE,
    else they are plotted with the ZPE subtracted off so that min(gsPot) = 0 """
    from FourierExpansions import calc_curves
    fig = plt.figure(figsize=(7, 8), dpi=600)
    x = np.linspace(0, 360, len(ResDict["eigvecs"][:, 0]))
    rad_x = np.radians(x)
    # create gs plot
    if len(ResDict["V"]) < 7:
        ResDict["V"] = np.hstack((ResDict["V"], np.zeros(7 - len(ResDict["V"]))))
    gsPot = Constants.convert(calc_curves(rad_x, ResDict["V"], function="fourier"), "wavenumbers", to_AU=False)
    colors = ["b", "r", "g", "indigo", "teal", "mediumvioletred"]
    for i in wfn_idx:
        en0g = Constants.convert(ResDict['energy'][i], "wavenumbers", to_AU=False)
        if ZPE is False:
            en0g -= min(gsPot)
        wfn0g = ResDict["eigvecs"][:, i]
        if wfn0g[150] < wfn0g[0]:
            wfn0g *= -1
        plt_wfn0g = en0g + wfn0g * 500  # for aestheics only (on all wfn)
        plt.plot(x, np.repeat(en0g, len(x)), "-k", linewidth=2)
        plt.plot(x, plt_wfn0g, color=colors[i], linewidth=3)

        # en1g = Constants.convert(ResDict['energy'][tunnel_pair[1]], "wavenumbers", to_AU=False)
        # if ZPE is False:
        #     en1g -= min(gsPot)
        # wfn1g = ResDict["eigvecs"][:, tunnel_pair[1]]
        # if wfn1g[21] < wfn0g[0]:
        #     wfn1g *= -1
        # plt_wfn1g = en1g + ResDict["eigvecs"][:, tunnel_pair[1]] * 100
        # plt.plot(x, np.repeat(en1g, len(x)), "-k", linewidth=2)
        # plt.plot(x, plt_wfn1g, color=colors[tunnel_pair[1]], linewidth=3)

    if ZPE is False:
        gsPot -= min(gsPot)
    plt.plot(x, gsPot, "-k", linewidth=3)

    plt.xticks(np.arange(0, 390, 60))
    plt.xlabel(r"$\tau$ (Degrees)")
    plt.ylim(min(gsPot)-20, min(gsPot)+1020)
    plt.ylabel(r"Energy (cm$^{-1}$)", labelpad=15.0)
    if filename is None:
        plt.show()
    else:
        fig.savefig(f"{filename}.jpg", dpi=fig.dpi, bbox_inches="tight")
        print(f"Figure saved to {filename}")
        plt.close()

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

