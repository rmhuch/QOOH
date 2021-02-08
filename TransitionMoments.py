import numpy as np
from Converter import Constants

class TransitionMoments:
    """To increase flexibility, assume self.transition is list of strings relating to different transitions,
     therefore, TDM, M_coeffs, and the Dipole Moment Matrix are calculated for each transition as needed. """
    def __init__(self, DVR_res, PORresults, MolecularInfo_obj, transition=None, MatSize=None):
        self.DVRresults = DVR_res
        self.PORresults = PORresults
        self.MolecularInfo_obj = MolecularInfo_obj
        self.transition = transition
        self.MatSize = MatSize
        self._InterpedGDW = None

    @property
    def InterpedGDW(self):
        if self._InterpedGDW is None:
            self._InterpedGDW = self.interpDW()
            # returns a "tuple" of grid, wfns, dips
        return self._InterpedGDW

    def interpDW(self):
        """Interpolate dipole moment and wavefunction to be the same length and range"""
        from scipy import interpolate
        wavefuns = self.DVRresults["wavefunctions"]
        pot = self.DVRresults["potential"]
        tor_degrees = np.linspace(0, 360, len(self.MolecularInfo_obj.DipoleMomentSurface.keys())).astype(int)

        pot_bohr = Constants.convert(pot[:, :, 0], "angstroms", to_AU=True)  # convert to bohr
        grid_min = np.min(pot_bohr)
        grid_max = np.max(pot_bohr)
        new_grid = np.linspace(grid_min, grid_max, wavefuns.shape[1])
        # interpolate wavefunction to same size, but still renormalize (when using values) to be safe
        interp_wfns = np.zeros((wavefuns.shape[0], len(new_grid), wavefuns.shape[2]))
        interp_dips = np.zeros((wavefuns.shape[0], len(new_grid), 3))
        for i in np.arange(wavefuns.shape[0]):  # loop through torsion degrees
            for s in np.arange(wavefuns.shape[2]):  # loop through OH wfns states
                f = interpolate.interp1d(pot_bohr[i, :], wavefuns[i, :, s],
                                         kind="cubic", bounds_error=False, fill_value="extrapolate")
                interp_wfns[i, :, s] = f(new_grid)
            for c in np.arange(3):  # loop through dipole components
                dipole_dat = self.MolecularInfo_obj.DipoleMomentSurface[f"tbhp_{tor_degrees[i]:0>3}.log"]
                new_dipole_dat = dipole_dat[:, :4]
                # DipoleMomentSurface also contains rotated coordinates, so we chop those off.
                new_dipole_dat[:, 0] = Constants.convert(new_dipole_dat[:, 0], "angstroms", to_AU=True)
                f = interpolate.interp1d(new_dipole_dat[:-1, 0], new_dipole_dat[:-1, c+1],
                                         kind="cubic", bounds_error=False, fill_value="extrapolate")
                interp_dips[i, :, c] = f(new_grid)
        return new_grid, interp_dips, interp_wfns  # dips in all au

    def plot_interpDW(self):
        import matplotlib.pyplot as plt
        g, wfns, dips = self.InterpedGDW
        tor_degrees = np.linspace(0, 360, wfns.shape[0]).astype(int)
        for i in np.arange(wfns.shape[0]):
            for j, c in enumerate(["A", "B", "C"]):
                plt.plot(g, dips[i, :, j], label=f"{c} - Component")
            plt.title(f"{tor_degrees[i]}  Degrees")
            plt.show()

    def calc_TDM(self, transition_str):
        """calculates the transition moment at each degree value. Returns the TDM at each degree.
            Normalize the mu value with the normalization of the wavefunction!!!"""
        mus = np.zeros((len(self.InterpedGDW[1]), 3))
        Glevel = int(transition_str[0])
        Exlevel = int(transition_str[-1])
        for k in np.arange(len(self.InterpedGDW[1])):  # loop through degree values
            for j in np.arange(3):  # loop through x, y, z
                gs_wfn = self.InterpedGDW[2][k, :, Glevel].T
                es_wfn = self.InterpedGDW[2][k, :, Exlevel].T
                es_wfn_t = es_wfn.reshape(-1, 1)
                soup = np.diag(self.InterpedGDW[1][k, :, j]).dot(es_wfn_t)
                mu = gs_wfn.dot(soup)
                normMu = mu / (gs_wfn.dot(gs_wfn) * es_wfn.dot(es_wfn))  # a triple check that everything is normalized
                mus[k, j] = normMu
        return mus  # in all au, returns only the TDM of the given transition (transition_str)

    def plot_TDM(self, filename=None):
        import matplotlib.pyplot as plt
        params = {'text.usetex': False,
                  'mathtext.fontset': 'dejavusans',
                  'font.size': 15}
        plt.rc('axes', labelsize=20)
        plt.rc('axes', labelsize=20)
        plt.rcParams.update(params)
        for Tstring in self.transition:
            fig = plt.figure(figsize=(7, 8), dpi=600)
            TDM = self.calc_TDM(Tstring)
            degrees = np.linspace(0, 360, len(TDM))
            plt.plot(degrees, np.repeat(0, len(degrees)), "-k", linewidth=3)
            colors = ["C0", "C1", "C2"]
            for c, val in enumerate(["A", "B", "C"]):
                plt.plot(degrees, TDM[:, c] / 0.393456, 'o', color=colors[c], label=f"{val} - Component")
                plt.plot(degrees, TDM[:, c] / 0.393456, '-', color=colors[c], linewidth=3)
            plt.plot(np.repeat(113, 10), np.linspace(-0.1, 0.1, 10), '--', color="gray", label=r"$\tau_{eq}$")
            plt.plot(np.repeat(247, 10), np.linspace(-0.1, 0.1, 10), '--', color="gray")
            if Tstring[-1] is "1":
                plt.ylim(-0.07, 0.07)
                plt.legend(loc="lower left")
            elif Tstring[-1] is "2":
                plt.ylim(-0.020, 0.020)
                plt.legend(loc="upper left")
            elif Tstring[-1] is "3":
                plt.ylim(-0.004, 0.004)
                plt.legend(loc="upper left")
            elif Tstring[-1] is "4":
                plt.ylim(-0.001, 0.001)
                plt.legend(loc="lower left")
            elif Tstring[-1] is "5":
                plt.ylim(-0.0003, 0.0003)
                plt.legend(loc="lower left")
            plt.ylabel(r"$M_{0 \rightarrow % s}$ (Debye)" % Tstring[-1])
            plt.xlim(-10, 370)
            plt.xlabel(r"$\tau$ (Degrees)")
            plt.xticks(np.arange(0, 390, 60))
            if filename is None:
                plt.show()
            else:
                fig.savefig(f"{filename}_{Tstring[0]}{Tstring[-1]}_eq.jpg", dpi=fig.dpi, bbox_inches="tight")
                print(f"Figure saved to {filename}_{Tstring[0]}{Tstring[-1]}")

    def calc_Mcoeffs(self, TDM):
        from FourierExpansions import calc_sin_coefs, calc_cos_coefs
        Mcoefs = np.zeros((7, 3))  # [num_coeffs, (ABC)]
        x = np.radians(np.linspace(0, 360, len(TDM)))
        for c, val in enumerate(["A", "B", "C"]):
            if val == "C":
                # fit to 6th order sin functions
                data = np.column_stack((x, TDM[:, c]))
                Mcoefs[:, c] = calc_sin_coefs(data)
            else:
                # first fit A/B mus to 6th order cos functions
                data = np.column_stack((x, TDM[:, c]))
                Mcoefs[:, c] = calc_cos_coefs(data)
        return Mcoefs

    def calc_DipoleMomentMatrix(self, M_coeffs):
        """expand the transition dipole moment into a matrix representation so that it can
        be used to solve for intensity"""
        if self.MatSize is None:
            MatSize = 7  # make more flexible to different order sizes in expansion - currently only supports 6
        else:
            MatSize = self.MatSize
        fullsize = 2 * MatSize + 1
        fullMat = np.zeros((3, fullsize, fullsize))
        for c, val in enumerate(["A", "B", "C"]):
            Mat = np.zeros((fullsize, fullsize))
            if val == "C":
                for l in np.arange(fullsize):
                    k = l - MatSize
                    for kprime in np.arange(k + 1, k - 7, -1):
                        m = k - kprime
                        if m > 0 and l - m >= 0:
                            Mat[l, l - m] = M_coeffs[:, c][m] / 2
                            Mat[l - m, l] = -Mat[l, l - m]
                        else:
                            pass
                fullMat[c, :, :] = Mat
            else:
                for l in np.arange(fullsize):
                    k = l - MatSize
                    Mat[l, l] = M_coeffs[:, c][0]  # m=0
                    for kprime in np.arange(k + 1, k - 7, -1):
                        m = k - kprime
                        if m > 0 and l - m >= 0:
                            Mat[l, l - m] = (M_coeffs[:, c][m] / 2)
                            Mat[l - m, l] = Mat[l, l - m]
                        else:
                            pass
                fullMat[c, :, :] = Mat
        return fullMat  # in all au, returns only the Matrix of the given transition (based on M_coeffs passed)

    # calculate overlap of torsion wfns with TDM
    def calc_intensity(self, numGstates=4, numEstates=4, FC=False, twoD=False, ccsd=False):
        from collections import OrderedDict
        import csv
        from FourierExpansions import calc_derivs
        all_intensities = OrderedDict()
        for Tstring in self.transition:  # should be a list of strings
            print(Tstring)
            Glevel = int(Tstring[0])
            Exlevel = int(Tstring[-1])
            if FC and not twoD and not ccsd:
                filename = f"TransitionIntensities_vOH{Glevel}tovOH{Exlevel}_FC_units.csv"
            elif FC and twoD and not ccsd:
                filename = f"TransitionIntensities_vOH{Glevel}tovOH{Exlevel}_2DFC_units.csv"
            elif FC and ccsd and not twoD:
                filename = f"TransitionIntensities_vOH{Glevel}tovOH{Exlevel}_FC_CCSD.csv"
            elif twoD and not FC and not ccsd:
                filename = f"TransitionIntensities_vOH{Glevel}tovOH{Exlevel}_2D.csv"
            else:
                filename = f"TransitionIntensities_vOH{Glevel}tovOH{Exlevel}.csv"
            with open(filename, mode="w") as results:
                results_writer = csv.writer(results, delimiter=',')
                # if FC:
                #     results_writer.writerow(["initial", "final", "Ei (cm^-1)", "frequency(cm^-1)", "FC factor",
                #                              "boltzman weight", "BW intensity"])
                # else:
                results_writer.writerow(["initial", "final", "Ei (cm^-1)", "frequency(cm^-1)", "mu_a(Debye)",
                                             "mu_b", "mu_c", "tot_intensity(km/mol)", "boltzman weight",
                                             "BW intensity(km/mol)"])
                gstorWfn_coefs = self.PORresults[0][Glevel]["eigvecs"]
                gstorEnergies = self.PORresults[0][Glevel]["energy"]
                extorWfn_coefs = self.PORresults[0][Exlevel]["eigvecs"]
                extorEnergies = self.PORresults[0][Exlevel]["energy"]
                intensities = []
                TDM = self.calc_TDM(Tstring)
                M_coeffs = self.calc_Mcoeffs(TDM)
                derivs = np.zeros(3)
                derivs[0] = calc_derivs(np.radians(110), M_coeffs[:, 0], function="cos")
                derivs[1] = calc_derivs(np.radians(110), M_coeffs[:, 1], function="cos")
                derivs[2] = calc_derivs(np.radians(110), M_coeffs[:, 2], function="sin")
                print("slope:", derivs)
                DipMomMat = self.calc_DipoleMomentMatrix(M_coeffs)
                if FC:  # to be used for units
                    M_coeffsFC = np.zeros(3)
                    M_coeffsFC[0] = TDM[11, 0]
                    M_coeffsFC[1] = TDM[11, 1]
                    M_coeffsFC[2] = TDM[11, 2]
                    # DipMomMatFC = self.calc_DipoleMomentMatrix(M_coeffsFC)

                for gState in np.arange(numGstates):
                    for exState in np.arange(numEstates):
                        freq = extorEnergies[exState] - gstorEnergies[gState]
                        freq_wave = Constants.convert(freq, "wavenumbers", to_AU=False)
                        if FC:
                            # matEl = np.dot(gstorWfn_coefs[:, gState].T, extorWfn_coefs[:, exState])
                            # intensity = abs(matEl)**2  # no units in this value
                            matEl = np.zeros(3)  # to be used if units
                            matEl_D = np.zeros(3)
                            comp_intents = np.zeros(3)
                            for c, val in enumerate(["A", "B", "C"]):  # loop through components
                                supere = np.dot(M_coeffsFC[c], extorWfn_coefs[:, exState].T)
                                matEl[c] = np.dot(gstorWfn_coefs[:, gState], supere)
                                matEl_D[c] = matEl[c] / 0.393456
                                comp_intents[c] = (abs(matEl[c]) ** 2) * freq_wave * 2.506 / (0.393456 ** 2)
                            intensity = np.sum(comp_intents)
                        else:
                            matEl = np.zeros(3)
                            matEl_D = np.zeros(3)
                            comp_intents = np.zeros(3)
                            for c, val in enumerate(["A", "B", "C"]):  # loop through components
                                supere = np.dot(DipMomMat[c, :, :], extorWfn_coefs[:, exState].T)
                                matEl[c] = np.dot(gstorWfn_coefs[:, gState], supere)
                                matEl_D[c] = matEl[c] / 0.393456
                                comp_intents[c] = (abs(matEl[c]) ** 2) * freq_wave * 2.506 / (0.393456 ** 2)
                            intensity = np.sum(comp_intents)
                        pop = 0.6950356  # Boltzman in cm^-1 / K from nist.gov
                        Ei = Constants.convert(gstorEnergies[gState]-gstorEnergies[0], "wavenumbers", to_AU=False)
                        if gState == 0:
                            Ef = Constants.convert(extorEnergies[exState]-extorEnergies[0], "wavenumbers", to_AU=False)
                            print(Ef)
                        BW = np.exp(-1*Ei/(pop*300))
                        BW_intensity = BW * intensity
                        # if comp_intents[1] > 1E-10:
                        #     ratio = comp_intents[1] / comp_intents[2]
                        intensities.append([gState, exState, freq_wave, intensity, BW, BW_intensity])
                        # if FC:
                        #     results_writer.writerow([gState, exState, Ei, freq_wave, intensity, BW, BW_intensity])
                        # else:
                        results_writer.writerow([gState, exState, Ei, freq_wave, *matEl_D, intensity, BW, BW_intensity])
                all_intensities[Tstring] = intensities
        return all_intensities  # ordered dict keyed by transition, holding a list of all the tor transitions

    def plot_sticks(self, numGstates=4, numEstates=4, FC=False, twoD=False, ccsd=False):
        import matplotlib.pyplot as plt
        params = {'text.usetex': False,
                  'mathtext.fontset': 'dejavusans',
                  'font.size': 10}
        plt.rcParams.update(params)
        dat = self.calc_intensity(numGstates=numGstates, numEstates=numEstates, FC=FC, twoD=twoD, ccsd=ccsd)
        lims = [(3600, 3900), (7050, 7300), (10300, 10600), (13300, 13900), (16300, 16600)]
        for i, Tstring in enumerate(self.transition):
            spect_dat = np.array(dat[Tstring])
            fig = plt.figure(figsize=(8, 5), dpi=600)
            mkline, stline, baseline = plt.stem(spect_dat[:, 2], spect_dat[:, 5]/spect_dat[0, 5], linefmt="-r",
                                                markerfmt=' ', basefmt="-k", use_line_collection=True)
            plt.setp(stline, "linewidth", 1.5)
            plt.setp(baseline, "linewidth", 0.5)
            plt.yticks(visible=False)
            plt.xlim(*lims[i])
            plt.savefig(f"stickspect_RP_{Tstring[0]}{Tstring[-1]}.jpg", dpi=fig.dpi, bbox_inches="tight")

    def calc_stat_intensity(self, values=None):
        from FourierExpansions import calc_curves
        from collections import OrderedDict
        if values is None:
            values = np.arange(230, 280, 10)
        else:
            pass
        OH_0coefs = self.PORresults[0][0]["V"]
        all_intensities = OrderedDict()
        for Tstring in self.transition:  # should be a list of strings
            TDM = self.calc_TDM(Tstring)
            degrees = np.linspace(0, 360, len(TDM))
            OH_excoefs = self.PORresults[0][int(Tstring[-1])]["V"]
            intensity = np.zeros((len(values), 3))
            for i, value in enumerate(values):
                idx = np.argwhere(degrees == value)[0][0]
                OH_0 = calc_curves(value, OH_0coefs)
                OH_ex = calc_curves(value, OH_excoefs)
                freq = OH_ex - OH_0
                freq_wave = Constants.convert(freq, "wavenumbers", to_AU=False)
                print("\n")
                print(f"Stationary Frequency {Tstring}: {freq_wave}")
                for c, val in enumerate(["A", "B", "C"]):  # loop through components
                    intensity[i, c] = (abs(TDM[idx, c]))**2 * freq_wave * 2.506 / (0.393456 ** 2)  # convert to km/mol
                    # print(f"Stationary Intensity @ {value} {val} : {intensity[i, c]}")
                # print(f"Total Intensity {value} : ", np.sum(intensity[i, :]))
                all_intensities[Tstring] = np.sum(intensity[i, :])
                oStrength = np.sum(intensity[i, :]) / 5.33E6
                print(f"Oscillator Strength @ {value} : {oStrength}")
        return all_intensities

    def plot_stat_intensity(self, values=None):
        import matplotlib.pyplot as plt
        if values is None:
            values = np.arange(230, 280, 10)
        else:
            pass
        intents = self.calc_stat_intensity()
        for c, val in enumerate(["A", "B", "C"]):
            plt.plot(values, intents[:, c], 'o', label=f"{val} - Component")
        plt.legend()
        plt.show()

