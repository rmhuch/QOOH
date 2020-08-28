import os
import numpy as np
import matplotlib.pyplot as plt
from Converter import Constants

params = {'text.usetex': False,
          'mathtext.fontset': 'dejavusans',
          'font.size': 14}
plt.rcParams.update(params)

def loadModeData(mode, freqdir, modedir):
    x = mode
    f = []
    m = []
    for i in x:
        f.append(np.loadtxt(os.path.join(freqdir, f"tbhp_eq_{i}oh_freqs.dat")))
        m.append(np.loadtxt(os.path.join(modedir, f"tbhp_eq_{i}oh_modes.dat"))[1:, :])
    freqarray = np.array(f)  # coord x num modes array
    modearray = m
    return modearray, freqarray

def plot_OOfreqsAC(data):
    """Plots frequencies to look at avoided crossings"""
    rOH = np.arange(0.9, 1.7, 0.1)
    for num, mode in enumerate(data[1].T):  # for each mode in the array
        if 18 <= num <= 23:
            plt.plot(rOH, mode, label=f"Mode {num+1}")
        else:
            pass
    # plt.ylim(500, 2000)
    plt.ylabel(r"OO Stretch Frequency (cm$^{-1}$)")
    plt.xlabel(r"rOH $\AA$")
    plt.legend()
    plt.show()

def plot_OOvsrOH(data, vals, modedir, filename=None):
    from CalcModeContributions import OOContribs
    rOH = np.arange(0.96568, 1.76568, 0.1)
    contribDict = OOContribs(vals, modedir)
    freqs = data[1]
    one = []
    two = []
    three = []
    for i, x, val in zip(np.arange(len(rOH)), rOH, vals):
        modes = contribDict[val][:, 0]
        if len(modes) == 3:
            one.append([x, int(modes[0]), (contribDict[val][0, 1]*100), freqs[i, int(modes[0])]])
            two.append([x, int(modes[1]), (contribDict[val][1, 1]*100), freqs[i, int(modes[1])]])
            three.append([x, int(modes[2]), (contribDict[val][2, 1]*100), freqs[i, int(modes[2])]])
        elif len(modes) == 2:
            one.append([x, int(modes[0]), (contribDict[val][0, 1]*100), freqs[i, int(modes[0])]])
            two.append([x, int(modes[1]), (contribDict[val][1, 1]*100), freqs[i, int(modes[1])]])
        else:
            one.append([x, int(modes[0]), (contribDict[val][0, 1]*100), freqs[i, int(modes[0])]])
    one = np.array(one)
    two = np.array(two)
    three = np.array(three)
    f, (ax, ax2) = plt.subplots(2, 1, sharex="all", figsize=(7, 8), dpi=350)
    ax2.plot(one[:, 0], one[:, 3], "ro")
    ax2.plot(two[:, 0], two[:, 3], "bo")
    ax2.plot(three[:, 0], three[:, 3], "go")
    ax2.axes.set_ylim(850, 975)
    ax2.axes.set_ylabel(r"OO Stretch Frequency (cm$^{-1}$)")
    ax2.axes.set_xlabel(r"rOH ($\AA$)")

    ax.plot(one[:, 0], one[:, 2], "ro")
    ax.plot(two[:, 0], two[:, 2], "bo")
    ax.plot(three[:, 0], three[:, 2], "go")
    ax.tick_params(labelbottom=True)
    ax.axes.set_ylim(30, 100)
    ax.axes.set_ylabel("OO Stretch Contribution (%)")
    ax.axes.set_xlabel(r"rOH ($\AA$)")
    if filename is None:
        f.show()
    else:
        f.savefig(f"{filename}.png", dpi=f.dpi, bbox_inches="tight")

def plot_OOHvsrOH(data, vals, modedir, filename=None):
    from CalcModeContributions import OOHContribs
    rOH = np.arange(0.96568, 1.76568, 0.1)
    contribDict = OOHContribs(vals, modedir)
    freqs = data[1]
    one = []
    two = []
    for i, x, val in zip(np.arange(len(rOH)), rOH, vals):
        modes = contribDict[val][:, 0]
        if len(modes) > 1:
            one.append([x, int(modes[0]), (contribDict[val][0, 1]*100), freqs[i, int(modes[0])]])
            two.append([x, int(modes[1]), (contribDict[val][1, 1]*100), freqs[i, int(modes[1])]])
        else:
            one.append([x, int(modes[0]), (contribDict[val][0, 1]*100), freqs[i, int(modes[0])]])
    one = np.array(one)
    two = np.array(two)

    f, (ax, ax2) = plt.subplots(2, 1, sharex="all", figsize=(7, 8), dpi=350)
    ax2.plot(one[:, 0], one[:, 3], "ro")
    ax2.plot(two[:, 0], two[:, 3], "bo")
    ax2.axes.set_ylim(1050, 1575)
    ax2.axes.set_ylabel(r"OOH Bend Frequency (cm$^{-1}$)")
    ax2.axes.set_xlabel(r"rOH ($\AA$)")

    ax.plot(one[:, 0], one[:, 2], "ro")
    ax.plot(two[:, 0], two[:, 2], "bo")
    ax.tick_params(labelbottom=True)
    ax.axes.set_ylim(30, 100)
    ax.axes.set_ylabel("OOH Bend Contribution (%)")
    ax.axes.set_xlabel(r"rOH ($\AA$)")
    if filename is None:
        f.show()
    else:
        f.savefig(f"{filename}.png", dpi=f.dpi, bbox_inches="tight")

def plot_HarmonicZPEvsrOH(data, potfile, filename=None):
    pot_dat = np.loadtxt(potfile)
    rOH = pot_dat[:, 0]
    freqs = data[1]
    plot_dat = []
    for i, x in enumerate(rOH):
        y = np.sum(freqs[i, :])/2 - np.sum(freqs[0, :])/2
        plot_dat.append([x, y])
    plot_dat = np.array(plot_dat)

    f, (ax, ax2) = plt.subplots(2, 1, sharex="all", figsize=(7, 8), dpi=350)
    ax2.plot(plot_dat[:, 0], plot_dat[:, 1], "o", color="indigo")
    ax2.axes.set_ylabel(r"Change in Harmonic ZPE (cm$^{-1}$)")
    ax2.axes.set_xlabel(r"rOH ($\AA$)")

    ax.plot(rOH, pot_dat[:, 1], "o", color="indigo")
    ax.tick_params(labelbottom=True)
    ax.axes.set_ylabel(r"Potential Energy (cm$^{-1}$)")
    ax.axes.set_xlabel(r"rOH ($\AA$)")
    if filename is None:
        f.show()
    else:
        f.savefig(f"{filename}.png", dpi=f.dpi, bbox_inches="tight")


if __name__ == '__main__':
    udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    OHdir = os.path.join(udrive, "TBHP", "eq_scans")
    OHfreqdir = os.path.join(OHdir, "oh_mode_freqs")
    OHmodedir = os.path.join(OHdir, "oh_modes")

    OHfiles = ["07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
    OHvals = ["09", "10", "11", "12", "13", "14", "15", "16"]

    dat = loadModeData(OHvals, OHfreqdir, OHmodedir)
    potfile = os.path.join(OHdir, "eq_PotentialEnergy.txt")
    plot_OOvsrOH(dat, OHvals, OHmodedir, filename="OOFreqsvsrOH")
    plot_HarmonicZPEvsrOH(dat, potfile, filename="HarmonicZPEvsrOH")
    plot_OOHvsrOH(dat, OHvals, OHmodedir, filename="OOHFreqsvsrOH")
