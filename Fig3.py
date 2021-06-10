import matplotlib.pyplot as plt
import numpy as np
from Converter import Constants
from runTBHP import tbhp, tbhp_res_obj
from runQOOH import qooh, qooh_res_obj

params = {'text.usetex': False,
          'mathtext.fontset': 'dejavusans',
          'font.size': 11}
plt.rcParams.update(params)
plt.rc('axes', labelsize=15)
plt.rc('axes', labelsize=15)

def make_ohPA_plots(wfnns):
    x = wfnns[:, 0]
    x_ang = Constants.convert(x, "angstroms", to_AU=False)
    colors = ["gold", "goldenrod", "darkorange"]

    for i, idx in enumerate([5, 2, 0]):
        plt.plot(x_ang, (wfnns[:, idx+1]**2)*18750, color=colors[i], label=r"$|\phi_{%s_{OH}}|^2$" % idx)
        plt.fill_between(x_ang, (wfnns[:, idx+1]**2)*18750, 0, facecolor=colors[i], color=colors[i], alpha=0.2)


def make_ohPanel(wfnns, datfile):
    fig = plt.figure(figsize=(4, 4), dpi=600)
    dat = np.loadtxt(datfile, delimiter=",", skiprows=1)
    make_ohPA_plots(wfnns)
    leg1 = plt.legend(loc="lower left")
    ax = plt.gca().add_artist(leg1)
    colors = ["teal", "b", "darkgreen", "indigo", "mediumvioletred"]
    markers = ["o", "o", "^", "s", "s"]
    labels = ["OOH", "OO", "CH", "CO"]
    for i in np.arange(1, dat.shape[1]):
        plt.plot(dat[:, 0], dat[:, i], color=colors[i])
    h = [plt.plot(dat[:, 0], dat[:, i], markers[i], color=colors[i],
                  label=labels[i-1])[0] for i in np.arange(1, dat.shape[1])]
    leg2 = plt.legend(handles=h, loc='upper right')
    plt.xlim(0.85, 1.7)
    plt.ylim(-100, 150)
    plt.ylabel(r"$\Delta$ ZPVE (cm$^{-1}$)")
    plt.xlabel(r"OH bond length ($\rm{\AA}$)")
    plt.savefig("Figure_OHvsE2.jpg", dpi=fig.dpi, bbox_inches="tight")

def make_torPA_plots(wfnns):
    x = np.linspace(0, 360, 100)
    colors = ["gold", "goldenrod", "darkorange"]
    wfn = 0
    for i, idx in enumerate([5, 2, 0]):
        plt.plot(x, (wfnns[wfn][:, idx] ** 2)*37, color=colors[i], label=r"${|\chi^{0_{OH}}_%s}|^2$" % idx)
        plt.fill_between(x, (wfnns[wfn][:, idx] ** 2)*37, facecolor=colors[i], color=colors[i], alpha=0.2)

def make_torPanel(wfnns, datfile):
    fig = plt.figure(figsize=(4, 4), dpi=600)
    dat = np.loadtxt(datfile, delimiter=",", skiprows=1)
    # make_torPA_plots(wfnns)
    # leg1 = plt.legend(loc="lower left")
    # ax = plt.gca().add_artist(leg1)
    colors = ["teal", "b", "darkgreen", "indigo", "mediumvioletred", "teal"]
    markers = ["o", "o", "^", "s", "s", "o"]
    labels = ["OOH", "OO", "CH", "CO", "OH"]
    for i in np.arange(1, dat.shape[1]):
        plt.plot(dat[:, 0], dat[:, i], color=colors[i])
    h = [plt.plot(dat[:, 0], dat[:, i], markers[i], color=colors[i],
                  label=labels[i-1])[0] for i in np.arange(1, dat.shape[1])]
    leg2 = plt.legend(handles=h, loc='upper right')
    plt.xticks(np.arange(0, 390, 30))
    plt.xlim(60, 300)
    plt.ylim(-50, 70)
    plt.rc('axes', labelsize=15)
    plt.rc('axes', labelsize=15)
    plt.ylabel(r"$\Delta$ ZPVE (cm$^{-1}$)")
    plt.xlabel(r"$\tau$ (Degrees)")
    plt.savefig("Figure_TorvsE_nowfns2.jpg", dpi=fig.dpi, bbox_inches="tight")

def make_ZPEplot(twoDcoeffs, fullcoeffs):
    from FourierExpansions import calc_curves
    ZPEcoeffs = fullcoeffs - twoDcoeffs
    ZPE = calc_curves(np.radians(np.arange(0, 360, 1)), ZPEcoeffs, function="cos")
    ZPE_cm = Constants.convert(ZPE, "wavenumbers", to_AU=False)
    fig = plt.figure(figsize=(4, 4), dpi=600)
    plt.plot(np.arange(0, 360, 1), ZPE_cm, color="k", linewidth=3.0)
    plt.plot(np.linspace(0, 360, 10), np.repeat(0, 10), color="k")
    # plt.plot(np.repeat(113, 10), np.linspace(-10, 10, 10), '--', color="gray", label=r"$\tau_{eq}$")
    # plt.plot(np.repeat(247, 10), np.linspace(-10, 10, 10), '--', color="gray")
    # plt.legend()
    plt.ylim(-10, 10)
    plt.xticks(np.arange(0, 390, 60))
    plt.xlim(0, 360)
    plt.ylabel(r"$\Delta$ ZPVE (cm$^{-1}$)")
    plt.xlabel(r"$\tau$ (Degrees)")
    plt.savefig("Figure_ZPE.jpg", dpi=fig.dpi, bbox_inches="tight")


if __name__ == '__main__':
    wfns = np.loadtxt("EqTOR_OHwfns_forCoire.dat")
    # make_ohPanel(wfns, "TBHP Data/Fig3_data_oh.csv")
    # ZPE_coeffs = np.load("/home/netid.washington.edu/rmhuch/udrive/TBHP/TBHP_Velcoeffs_6order_2D.npy")
    # Vel_coeffs = np.load("/home/netid.washington.edu/rmhuch/udrive/TBHP/TBHP_Velcoeffs_6order.npy")
    # make_ZPEplot(ZPE_coeffs, Vel_coeffs)
    # make_torPanel(torWfns, "TBHP Data/Fig3_data_tor.csv")
    # plt.show()
