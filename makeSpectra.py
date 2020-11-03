import numpy as np
import matplotlib.pyplot as plt
params = {'text.usetex': False,
          'mathtext.fontset': 'dejavusans',
          'font.size': 14}
plt.rcParams.update(params)

def plot_exp():
    exp = np.loadtxt("emil_expt.dat")
    plt.plot(exp[:, 0], exp[:, 1], "-r", linewidth=4.0, zorder=1, label="Experiment")

def plot_sticks(FourD=False, TwoD=False, Shift=None):
    all_theory = np.loadtxt("TBHP_theory_sticks.txt", skiprows=1)
    if FourD:
        mkline, stline, baseline = plt.stem(all_theory[:, 2], all_theory[:, 3]/max(all_theory[:, 3]), linefmt="-b",
                                            markerfmt=' ', use_line_collection=True, label="2D Reaction Path Model")
    elif TwoD:
        mkline, stline, baseline = plt.stem(all_theory[:, 0], all_theory[:, 1]/max(all_theory[:, 1]), linefmt="-g",
                                            markerfmt=' ', use_line_collection=True, label="2D Reaction Path Model")
    elif Shift is not None:
        mkline, stline, baseline = plt.stem(all_theory[:, 0]-Shift, all_theory[:, 1]/max(all_theory[:, 1]),
                                            linefmt="-g", markerfmt=' ', use_line_collection=True,
                                            label=f"2D Reaction Path Model Shifted by {Shift}")
    else:
        raise Exception("no model selected")
    plt.setp(stline, "linewidth", 4.0)
    plt.setp(baseline, visible=False)
    plt.yticks(visible=False)

def plot_gauss(FourD=False, TwoD=False, TwoD_ccsd=False, Shift=None, delta=5, linetype="-g", label=None):
    all_theory = np.loadtxt("TBHP_theory_sticks.txt", skiprows=1)

    if FourD:
        freqs = all_theory[:, 2]
        intents = all_theory[:, 3]
    elif TwoD:
        freqs = all_theory[:, 0]
        intents = all_theory[:, 1]
    elif TwoD_ccsd:
        TwoDCCSD = np.loadtxt("2D_CCSD_results.txt")
        freqs = TwoDCCSD[:, 0]
        intents = TwoDCCSD[:, 1]
    elif Shift is not None:
        freqs = all_theory[:, 0]-Shift
        intents = all_theory[:, 1]
    else:
        raise Exception("no model selected")
    x = np.arange(6950, 7400, 1)
    y = np.zeros(len(x))
    for i in np.arange(len(intents)):
        for j, val in enumerate(x):
            y[j] += intents[i]*np.exp(-(val-freqs[i])**2/delta**2)
    plt.plot(x, y/max(y), linetype, linewidth=4.0, label=label)


if __name__ == '__main__':
    # plt.ylabel("Normalized Intensity")
    # plt.xlabel("Wavenumbers (cm$^{-1}$)")
    plot_exp()
    # plot_gauss(FourD=True, delta=15, linetype='-b', label="4D Local Mode Model")
    # plot_gauss(TwoD=True, delta=15, linetype="-g", label="2D Reaction Path Model")
    plot_gauss(TwoD_ccsd=True, delta=15, linetype="-C5", label="2D Reaction Path Model with CCSD")
    # plot_gauss(Shift=33, delta=15, linetype="-C4", label="2D Reaction Path Model Shifted 33 cm$^(-1)$")
    # plt.legend()
    plt.xlim(6950, 7350)
    plt.tight_layout()
    plt.show()

