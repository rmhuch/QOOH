import os
import numpy as np
import matplotlib.pyplot as plt
from Converter import Constants

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
    plt.ylabel("OO Stretch Frequency (wavenumbers)")
    plt.xlabel("rOH (angstroms)")
    plt.legend()
    plt.show()

def plot_OOvsrOH(data, vals, modedir):
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
    plt.subplot(212)
    plt.plot(one[:, 0], one[:, 3], "ro")
    plt.plot(two[:, 0], two[:, 3], "bo")
    plt.plot(three[:, 0], three[:, 3], "go")
    plt.ylabel("OO Stretch Frequency (wavenumbers)")
    plt.xlabel("rOH (angstroms)")
    plt.subplot(211)
    plt.plot(one[:, 0], one[:, 2], "ro")
    plt.plot(two[:, 0], two[:, 2], "bo")
    plt.plot(three[:, 0], three[:, 2], "go")
    plt.ylabel("OO Stretch Contribution")
    plt.xlabel("rOH (angstroms)")
    plt.show()

def plot_OOHvsrOH(data, vals, modedir):
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
    plt.subplot(212)
    plt.plot(one[:, 0], one[:, 3], "ro")
    plt.plot(two[:, 0], two[:, 3], "bo")
    plt.ylabel("OOH Bend Frequency (wavenumbers)")
    plt.xlabel("rOH (angstroms)")
    plt.subplot(211)
    plt.plot(one[:, 0], one[:, 2], "ro")
    plt.plot(two[:, 0], two[:, 2], "bo")
    plt.ylabel("OOH Bend Contribution (%)")
    plt.xlabel("rOH (angstroms)")
    plt.show()

if __name__ == '__main__':
    udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    OHdir = os.path.join(udrive, "TBHP", "eq_scans")
    OHfreqdir = os.path.join(OHdir, "oh_mode_freqs")
    OHmodedir = os.path.join(OHdir, "oh_modes")

    OHfiles = ["07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
    OHvals = ["09", "10", "11", "12", "13", "14", "15", "16"]

    dat = loadModeData(OHvals, OHfreqdir, OHmodedir)
    plot_OOvsrOH(dat, OHvals, OHmodedir)
