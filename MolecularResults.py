import numpy as np
import os


class MoleculeInfo:
    def __init__(self, MoleculeName, atom_array, eqTORangle, oh_scan_npz=None, TorFchkDirectory=None,
                 TorFiles=None, ModeFchkDirectory=None, ModeFiles=None, dipole_npz=None):
        self.MoleculeName = MoleculeName
        self.atom_array = atom_array
        self.eqTORangle = eqTORangle
        self.oh_scan_npz = oh_scan_npz
        self.TorFchkDirectory = TorFchkDirectory  # to calculate Vel + rxn path
        self.TorFiles = TorFiles
        self.ModeFchkDirectory = ModeFchkDirectory  # to calculate modes from rxn path
        self.ModeFiles = ModeFiles  # must be proper file names as str not just changing piece
        self.dipole_npz = dipole_npz
        self._MoleculeDir = None
        self._mass_array = None
        self._PES_DegreeDict = None
        self._eqPES = None
        self._InternalCoordDict = None
        self._eqICD = None
        self._DipoleMomentSurface = None

    @property
    def MoleculeDir(self):
        if self._MoleculeDir is None:
            udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self._MoleculeDir = os.path.join(udrive, self.MoleculeName)
        return self._MoleculeDir

    @property
    def mass_array(self):
        from Converter import Constants
        if self._mass_array is None:
            m = np.array([Constants.mass(x, to_AU=True) for x in self.atom_array])
            self._mass_array = m  # returns masses in AMU
        return self._mass_array

    @property
    def PES_DegreeDict(self):
        if self._PES_DegreeDict is None:
            self._PES_DegreeDict, self._eqPES = self.get_DegreeDict()
        return self._PES_DegreeDict

    @property
    def eqPES(self):
        if self._eqPES is None:
            self._PES_DegreeDict, self._eqPES = self.get_DegreeDict()
            return self._eqPES

    @property
    def InternalCoordDict(self):
        if self._InternalCoordDict is None:
            self._InternalCoordDict, self._eqICD = self.get_InternalCoordDict()
        return self._InternalCoordDict

    @property
    def eqICD(self):
        if self._eqICD is None:
            self._InternalCoordDict, self._eqICD = self.get_InternalCoordDict()
        return self._eqICD

    @property
    def DipoleMomentSurface(self):
        if self._DipoleMomentSurface is None:
            self._DipoleMomentSurface = np.load(os.path.join(self.MoleculeDir, self.dipole_npz))
            # pulls in data as a dictionary that is keyed by angles, with values: (roh, x, y, z)
        return self._DipoleMomentSurface

    def get_DegreeDict(self):
        degreedict = dict()
        vals = np.load(os.path.join(self.MoleculeDir, self.oh_scan_npz), allow_pickle=True)
        sort_degrees = np.arange(0, 370, 10)
        for i in sort_degrees:
            if i == 0:
                E = vals[f"360.0"][0]
                rOH = vals[f"360.0"][1]["B6"]
            else:
                E = vals[f"{i}.0"][0]
                rOH = vals[f"{i}.0"][1]["B6"]
            idx = np.argsort(rOH)
            degreedict[i] = np.column_stack((rOH[idx], E[idx]))
        if type(self.eqTORangle) == list:
            eqvals = []
            for i in self.eqTORangle:
                eqE = vals[str(i)][0]
                eqE -= min(eqE)  # subtract of min E of JUST EQ scan here
                eqrOH = vals[str(i)][1]["B6"]
                eqvvals = np.column_stack((eqrOH, eqE))
                eqvals.append(eqvvals)
        else:
            eqE = vals[str(self.eqTORangle)][0]
            eqE -= min(eqE)  # subtract of min E of JUST EQ scan here
            eqrOH = vals[str(self.eqTORangle)][1]["B6"]
            eqvals = np.column_stack((eqrOH, eqE))
        # print(degreedict[180])
        # data in degrees: angstroms/hartrees
        return degreedict, eqvals

    def get_InternalCoordDict(self):
        internaldict = dict()
        vals = np.load(os.path.join(self.MoleculeDir, self.oh_scan_npz), allow_pickle=True)
        sort_degrees = np.arange(0, 370, 10)
        for i in sort_degrees:
            if i == 0:
                full_dict = vals["360.0"][1]
                full_dict["D4"] = np.repeat(0.0, 52)
            else:
                full_dict = vals[f"{i}.0"][1]
            for j in full_dict.keys():  # go through and if the value is just repeated only return one value.
                if full_dict[j][0] == full_dict[j][1] and full_dict[j][0] == full_dict[j][2]:
                    full_dict[j] = full_dict[j][0]
            internaldict[i] = full_dict
        if type(self.eqTORangle) == list:
            eq_dict = vals[str(self.eqTORangle[0])][1]
        else:
            eq_dict = vals[str(self.eqTORangle)][1]
        for j in eq_dict.keys():  # go through and if the value is just repeated only return one value.
            if eq_dict[j][0] == eq_dict[j][1] and eq_dict[j][0] == eq_dict[j][2]:
                eq_dict[j] = eq_dict[j][0]
        eqICD = eq_dict
        # data in degrees: angstroms/degrees
        return internaldict, eqICD


class MoleculeResults:
    def __init__(self, MoleculeInfo_obj, DVRparams, PORparams, Intenseparams, transition="0->2"):
        self.MoleculeInfo = MoleculeInfo_obj
        self.DVRparams = DVRparams
        if "Vexpansion" in PORparams:  # note expansion to 6tau if not implicitly in call
            pass
        else:
            PORparams["Vexpansion"] = "sixth"
        self.PORparams = PORparams
        self.Intenseparams = Intenseparams
        self.transition = transition
        self._DegreeDVRresults = None
        self._OHmodes = None
        self._TORmodes = None
        self._PotentialCoeffs = None
        self._Gmatrix = None
        self._fittedGmatrix = None
        self._RxnPathResults = None
        self._VelCoeffs = None
        self._torDVRresults = None
        self._PORresults = None
        self._TransitionIntensities = None
        self._StatTransitionIntensities = None

    @property
    def DegreeDVRresults(self):
        if self._DegreeDVRresults is None:
            resultspath = os.path.join(self.MoleculeInfo.MoleculeDir,
                                       f"{self.MoleculeInfo.MoleculeName}_vOH_DVRresults_{self.DVRparams['num_pts']}.npz")
            if os.path.exists(resultspath):
                self._DegreeDVRresults = np.load(resultspath)
            else:
                print(f"{resultspath} not found. beginning Degree DVR")
                results = self.run_degree_DVR()
                self._DegreeDVRresults = np.load(results)
        return self._DegreeDVRresults

    @property
    def Gmatrix(self):
        if self._Gmatrix is None:
            resultspath = os.path.join(self.MoleculeInfo.MoleculeDir,
                                       f"{self.MoleculeInfo.MoleculeName}_Gmatrix_elements.npz")
            if os.path.exists(resultspath):
                self._Gmatrix = np.load(resultspath)
            else:
                print(f"{resultspath} not found. Beginning Gmatrix calculation")
                results = self.run_gmatrix()
                self._Gmatrix = np.load(results)
        return self._Gmatrix

    @property
    def fittedGmatrix(self):
        if self._fittedGmatrix is None:
            self._fittedGmatrix = self.fit_gmat()
        return self._fittedGmatrix

    @property
    def RxnPathResults(self):
        if self._RxnPathResults is None:
            self._RxnPathResults = self.run_reaction_path()
        return self._RxnPathResults

    @property
    def VelCoeffs(self):
        if self._VelCoeffs is None:
            if self.PORparams["None"]:
                if self.PORparams["twoD"] and "scaled_barrier" in self.PORparams:
                    resultspath = os.path.join(self.MoleculeInfo.MoleculeDir,
                                               f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6cos6sin_2D_scaledVel.npy")
                elif self.PORparams["twoD"]:
                    resultspath = os.path.join(self.MoleculeInfo.MoleculeDir,
                                               f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6cos6sin_2D.npy")
                elif "scaled_barrier" in self.PORparams:
                    resultspath = os.path.join(self.MoleculeInfo.MoleculeDir,
                                               f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6cos6sin_scaledVel.npy")
                else:
                    resultspath = os.path.join(self.MoleculeInfo.MoleculeDir,
                                               f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6cos6sin.npy")
            else:
                if self.PORparams["twoD"]:
                    resultspath = os.path.join(self.MoleculeInfo.MoleculeDir,
                                               f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6order_2D.npy")
                else:
                    resultspath = os.path.join(self.MoleculeInfo.MoleculeDir,
                                               f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6order.npy")
            if os.path.exists(resultspath):
                print(f"Using {resultspath} to calculate Vel + ZPE coeffs")
                self._VelCoeffs = np.load(resultspath, allow_pickle=True)
            else:
                print(f"{resultspath} not found. beginning Vel + ZPE calculation")
                results = self.calculate_harmZPE()
                self._VelCoeffs = np.load(results)
        return self._VelCoeffs

    @property
    def torDVRresults(self):
        if self._torDVRresults is None:
            self._torDVRresults = self.run_tor_DVR()
        return self._torDVRresults

    @property
    def PORresults(self):
        if self._PORresults is None:
            self._PORresults = self.run_tor_adiabats()
        return self._PORresults

    @property
    def TransitionIntensities(self):
        if self._TransitionIntensities is None:
            self._TransitionIntensities = self.run_transitions()
        return self._TransitionIntensities

    @property
    def StatTransitionIntensities(self):
        if self._StatTransitionIntensities is None:
            self._StatTransitionIntensities = self.run_StatTransitions()  # prints intensity and oscillator strength
        return self._StatTransitionIntensities

    def run_degree_DVR(self):
        from DegreeDVR import run_OH_DVR, calcFreqs
        params = self.DVRparams
        potential_array, epsilon_pots, wavefuns_array = run_OH_DVR(self.MoleculeInfo.PES_DegreeDict,
                                                                   desiredEnergies=params["desired_energies"],
                                                                   NumPts=params["num_pts"],
                                                                   plotPhasedWfns=params["plot_phased_wfns"],
                                                                   extrapolate=params["extrapolate"])
        freqs = calcFreqs(epsilon_pots)
        npz_name = f"{self.MoleculeInfo.MoleculeName}_vOH_DVRresults_{params['num_pts']}.npz"
        np.savez(os.path.join(self.MoleculeInfo.MoleculeDir, npz_name), potential=potential_array,
                 frequencies=freqs, energies=epsilon_pots, wavefunctions=wavefuns_array)
        print(f"DVR Results saved to {npz_name} in {self.MoleculeInfo.MoleculeDir}")
        return os.path.join(self.MoleculeInfo.MoleculeDir, npz_name)

    def plot_degreeDVR(self):
        from DVRtools import ohWfn_plots
        ohWfn_plots(self.DegreeDVRresults, wfns2plt=6, degree=None)

    def run_gmatrix(self):
        from ExpectationValues import run_DVR
        from Gmatrix import get_tor_gmatrix, get_eq_g, make_gmat_plot
        Epot_array = self.MoleculeInfo.eqPES
        if type(Epot_array) == list:
            Epot_array = Epot_array[1]  # use QOOH1 point
        else:
            pass
        params = self.DVRparams
        ens, rOH_wfns = run_DVR(Epot_array, NumPts=params["num_pts"], desiredEnergies=params["desired_energies"],
                                extrapolate=params["extrapolate"])
        tor_angles = self.MoleculeInfo.PES_DegreeDict.keys()
        mass_array = self.MoleculeInfo.mass_array
        tor_masses = np.array((mass_array[6], mass_array[4], mass_array[5], np.inf))
        res = get_tor_gmatrix(rOH_wfns, tor_angles, self.MoleculeInfo.InternalCoordDict, tor_masses)
        # eq_g = get_eq_g(tor_angles, self.MoleculeInfo.eqICD, tor_masses)
        # SOMETHING in eq_g is messed up... but we don't use it.. so.. maybe fix later? or remove completely
        for i in np.arange(len(res)):
            file_name = f"B2PLYP_Gmatrix_elements_voh{i}.txt"
            np.savetxt(os.path.join(self.MoleculeInfo.MoleculeDir, file_name), res[i])
        make_gmat_plot(res)
        npz_name = f"{self.MoleculeInfo.MoleculeName}_Gmatrix_elements.npz"
        np.savez(os.path.join(self.MoleculeInfo.MoleculeDir, npz_name), gmatrix=res)  # , eq_gmatrix=eq_g)
        return os.path.join(self.MoleculeInfo.MoleculeDir, npz_name)

    def fit_gmat(self):
        from FourierExpansions import fourier_coeffs, calc_cos_coefs
        fittedG = dict()
        for i in np.arange(len(self.Gmatrix["gmatrix"])):
            if "None" in self.PORparams:
                coeffs = fourier_coeffs(np.column_stack((np.radians(self.Gmatrix["gmatrix"][i][:, 0]),
                                                         self.Gmatrix["gmatrix"][i][:, 1])), cos_order=6, sin_order=6)
            else:
                coeffs = calc_cos_coefs(np.column_stack((np.radians(self.Gmatrix["gmatrix"][i][:, 0]),
                                                         self.Gmatrix["gmatrix"][i][:, 1])))
            fittedG[i] = coeffs
        return fittedG

    def run_reaction_path(self):
        from ReactionPath import run_energies, run_Emil_energies
        fchkDir = os.path.join(self.MoleculeInfo.MoleculeDir, self.MoleculeInfo.TorFchkDirectory)
        rxn_path_res = run_energies(fchkDir, self.MoleculeInfo.TorFiles)
        if "EmilData" in self.PORparams or "MixedData1" in self.PORparams:
            datfile = os.path.join(self.MoleculeInfo.MoleculeDir, "Emil_Energies_tor_OH.txt")
            EmilelectronicE = run_Emil_energies(datfile)
            rxn_path_res["EmilelectronicE"] = EmilelectronicE
        return rxn_path_res

    def plot_fourier(self):
        from FourierExpansions import fourier_coeffs, calc_curves, calc_cos_coefs
        import matplotlib.pyplot as plt
        interp_degree = np.linspace(0, 360, 100)
        interp_rad = np.radians(np.linspace(0, 360, 100))
        EE = np.copy(self.RxnPathResults["electronicE"])
        EE[:, 0] = np.radians(EE[:, 0])
        cos_coeffs, sin_coeffs = fourier_coeffs(EE, cos_order=6, sin_order=6)
        coeff2 = calc_cos_coefs(EE)
        print("cos", cos_coeffs)
        print("sin", sin_coeffs)
        interpE = calc_curves(interp_rad, [cos_coeffs, sin_coeffs], function="fourier")
        interp2 = calc_curves(interp_rad, coeff2, function="cos")
        plt.plot(self.RxnPathResults["electronicE"][:, 0], self.RxnPathResults["electronicE"][:, 1], "o")
        plt.plot(interp_degree, interpE)
        plt.plot(interp_degree, interp2)
        plt.show()

    def calculate_Vel(self):
        if "scaled_barrier" in self.PORparams:
            import matplotlib.pyplot as plt
            from scaleTORpotentials import scale_uneven_barrier
            ens = self.RxnPathResults["electronicE"][:, 1] - min(self.RxnPathResults["electronicE"][:, 1])
            # we subtract the min off so energies are positive for scaling
            dat = np.column_stack((self.RxnPathResults["electronicE"][:, 0], ens))
            sf, scaled_energies = scale_uneven_barrier(dat, self.PORparams["scaled_barrier"])
            Vel = scaled_energies[:, 1] + min(self.RxnPathResults["electronicE"][:, 1])
            # then we add the scaling energy back on..
            print("Scaling Factors: ", sf)
        else:
            Vel = self.RxnPathResults["electronicE"][:, 1]
        return Vel  # returns 1D array of electronic energies in Hartree and < 0

    def calculate_harmZPE(self):
        from Converter import Constants
        from FourierExpansions import calc_cos_coefs, fourier_coeffs, calc_curves
        degree_vals = np.linspace(0, 360, len(self.MoleculeInfo.TorFiles))
        idx = np.where(self.RxnPathResults["norm_grad"][:, 1] > 4E-4)
        new_degree = degree_vals[idx]
        Vel = self.calculate_Vel()
        Vel_ZPE_dict = dict()
        ZPE_dict = dict()
        for i, j in enumerate(degree_vals):
            freqs = self.RxnPathResults[j]["freqs"]  # in hartree
            nonzero_freqs = freqs[7:-1]  # throw out translations/rotations and OH frequency
            nonzero_freqs_har = Constants.convert(nonzero_freqs, "wavenumbers", to_AU=True)
            ZPE = np.sum(nonzero_freqs_har) / 2
            if j in new_degree:
                if self.PORparams["twoD"]:  # "twoD" in self.PORparams and
                    Vel_ZPE_dict[j] = Vel[i]
                else:
                    Vel_ZPE_dict[j] = Vel[i] + ZPE
                ZPE_dict[j] = ZPE
            else:
                pass
        Vel_ZPE = np.array([(d, v) for d, v in Vel_ZPE_dict.items()])
        sort_idx = np.argsort(Vel_ZPE[:, 0])
        Vel_ZPE = Vel_ZPE[sort_idx]
        Vel_ZPE[:, 0] = np.radians(Vel_ZPE[:, 0])
        new_x = np.radians(np.arange(0, 360, 1))
        if self.PORparams["None"]:
            VelwZPE_coeffs1 = fourier_coeffs(Vel_ZPE)
            y = calc_curves(new_x, VelwZPE_coeffs1, function="fourier")
            y -= min(y)  # shift curve so minima are at 0 instead of negative
            VelwZPE_coeffs = fourier_coeffs(np.column_stack((new_x, y)))
            if self.PORparams["twoD"] and "scaled_barrier" in self.PORparams:
                npy_name = f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6cos6sin_2D_scaledVel.npy"
                # csv_name = f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6cos6sin_2D_scaledVel.csv"
            elif self.PORparams["twoD"]:
                npy_name = f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6cos6sin_2D.npy"
                # csv_name = f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6cos6sin_2D.csv"
            elif "scaled_barrier" in self.PORparams:
                npy_name = f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6cos6sin_scaledVel.npy"
                # csv_name = f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6cos6sin_scaledVel.csv"
            else:
                npy_name = f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6cos6sin.npy"
                # csv_name = f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6cos6sin.csv"
        else:
            VelwZPE_coeffs1 = calc_cos_coefs(Vel_ZPE)
            y = calc_curves(new_x, VelwZPE_coeffs1)
            y -= min(y)  # shift curve so minima are at 0 instead of negative
            VelwZPE_coeffs = calc_cos_coefs(np.column_stack((new_x, y)))
            if self.PORparams["twoD"]:
                npy_name = f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6order_2D.npy"
                csv_name = f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6order_2D.csv"
            else:
                npy_name = f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6order.npy"
                csv_name = f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6order.csv"
        # save results
        np.save(os.path.join(self.MoleculeInfo.MoleculeDir, npy_name), VelwZPE_coeffs)
        # np.savetxt(os.path.join(self.MoleculeInfo.MoleculeDir, csv_name), VelwZPE_coeffs)
        return os.path.join(self.MoleculeInfo.MoleculeDir, npy_name)

    def make_diff_freq_plots(self):
        from FourierExpansions import calc_cos_coefs, calc_curves
        import matplotlib.pyplot as plt
        params = {'text.usetex': False,
                  'mathtext.fontset': 'dejavusans',
                  'font.size': 14}
        plt.rcParams.update(params)
        degree1 = np.arange(10, 110, 10)
        degree4 = np.arange(120, 180, 10)
        degree2 = np.arange(190, 250, 10)
        degree3 = np.arange(260, 360, 10)
        degrees = np.concatenate((degree1, degree4, degree2, degree3))
        freq_data = np.zeros((len(degrees), 49))
        sel = [x for x in range(49) if x not in list(range(8)) + [29, 48]]
        for i, deg in enumerate(degrees):  # pull the frequencies
            freqs = self.RxnPathResults[deg]["freqs"]
            freq_data[i] = np.column_stack((deg, *freqs))
        # add in data from Mark
        dat = np.loadtxt("stat_no_tor_tor_freqs.csv", delimiter=",")
        freq_data = np.concatenate((freq_data, dat), axis=0)
        sym_dict = dict()
        for idx, deg in enumerate(freq_data[:, 0]):  # symmeterize the frequencies
            if deg > 180:
                friend = 180 - (deg - 180)
            else:
                friend = 180 + (180 - deg)
            pos = np.where(freq_data[:, 0] == friend)[0]
            if len(pos) == 0:
                sym_dict[deg] = freq_data[idx, 1:]
                sym_dict[friend] = freq_data[idx, 1:]
            else:
                val1 = freq_data[idx, 1:]
                val2 = freq_data[pos[0], 1:]
                sym_dict[deg] = (val1 + val2) / 2
                sym_dict[friend] = (val1 + val2) / 2
        sym_data = np.column_stack([list(sym_dict.keys()), list(sym_dict.values())])
        key_idxs = np.argsort(list(sym_dict.keys()))
        sym_data = sym_data[key_idxs]
        # subtract eq frequencies and divide by 2 for ZPE
        eq_freqs = np.loadtxt("no_tor_tor_freqs.csv", delimiter=",")
        plt_sym_data = np.zeros((len(sym_data), 49))
        plt_sym_data[:, 0] = sym_data[:, 0]
        for i in np.arange(1, sym_data.shape[1]):
            plt_sym_data[:, i] = (sym_data[:, i] - eq_freqs[i - 1]) / 2
        # fit OOH and OH to cos function expansions
        x_deg = np.arange(10, 351, 1)
        x_rad = np.radians(x_deg)
        eq_idx = np.argwhere(x_deg == 112)[0]
        OOH = plt_sym_data[:, 29]
        OOH_coefs = calc_cos_coefs(np.column_stack([np.radians(plt_sym_data[:, 0]), OOH]))
        yOOH = calc_curves(x_rad, OOH_coefs)
        yOOH -= yOOH[eq_idx[0]]
        plt.plot(x_deg, yOOH, "-r", linewidth=2.5, zorder=50, label="OOH Bend")
        # plt.plot(plt_sym_data[:, 0], OOH, "--r", linewidth=2.5, zorder=50, label="OOH Bend")
        OH = plt_sym_data[:, 48]
        OH_coefs = calc_cos_coefs(np.column_stack([np.radians(plt_sym_data[:, 0]), OH]))
        yOH = calc_curves(x_rad, OH_coefs)
        yOH -= yOH[eq_idx[0]]
        plt.plot(x_deg, yOH, "-g", linewidth=2.5, zorder=50, label="OH Stretch")
        # plt.plot(plt_sym_data[:, 0], OH, "--g", linewidth=2.5, zorder=50, label="OH Stretch")
        # for i in sel:
        #     plt.plot(plt_sym_data[:, 0], plt_sym_data[:, i], color="gray", linewidth=1.5)
        # plot sum of modes - OOH and OH
        eq_freqss = np.concatenate([[1000], eq_freqs])
        eq_sum = np.sum(eq_freqss[sel])
        mode_y = np.zeros(len(plt_sym_data))
        for j, d in enumerate(plt_sym_data[:, 0]):
            mode_sum = np.sum(sym_data[j, sel])
            mode_y[j] = (mode_sum - eq_sum) / 2
        y_coeffs = calc_cos_coefs(np.column_stack([np.radians(plt_sym_data[:, 0]), mode_y]))
        y_vals = calc_curves(x_rad, y_coeffs)
        y_vals -= y_vals[eq_idx[0]]
        plt.plot(x_deg, y_vals, color="k", linewidth=2.5, label=r"$\Delta$ZPE - OOH Bend - OH Stretch")
        # # pot sum of modes - OH
        # mode_y2 = np.zeros(len(plt_sym_data))
        # sel2 = [x for x in range(49) if x not in list(range(8)) + [48]]
        # eq_sum2 = np.sum(eq_freqss[sel2])
        # for j, d in enumerate(plt_sym_data[:, 0]):
        #     mode_sum = np.sum(sym_data[j, sel2])
        #     mode_y2[j] = (mode_sum-eq_sum2)/2
        # y_coeffs2 = calc_cos_coefs(np.column_stack([np.radians(plt_sym_data[:, 0]), mode_y2]))
        # y_vals2 = calc_curves(np.radians(np.arange(10, 351, 1)), y_coeffs2)
        # plt.plot(np.arange(10, 351, 1), y_vals2, color="k", linewidth=2.5, label=r"$\Delta$ZPE - OH Stretch")

        plt.legend()
        plt.xticks(np.arange(30, 330, 30))
        plt.xlim(30, 330)
        plt.xlabel(r"$\tau$ [Degrees]")
        plt.ylim(-30, 30)
        plt.ylabel(r"$\Delta$ ZPE [wavenumbers]")
        plt.show()

    def run_tor_adiabats(self):
        from TorsionPOR import POR
        from Converter import Constants
        if "EmilData" in self.PORparams or "MixedData2" in self.PORparams:
            file_name = os.path.join(self.MoleculeInfo.MoleculeDir, "Emil_vOHenergies.txt")
            self.PORparams["EmilEnergies"] = np.loadtxt(file_name, skiprows=1)
        PORobj = POR(DVR_res=self.DegreeDVRresults, fit_gmatrix=self.fittedGmatrix, Velcoeffs=self.VelCoeffs,
                     params=self.PORparams)
        results = PORobj.solveHam()  # returns a list of dictionaries for each torsional potential
        if "PrintResults" in self.PORparams:
            for i in np.arange(len(results)):
                bh = Constants.convert(results[i]["barrier"], "wavenumbers", to_AU=False)
                print(f"Barrier Height : {bh} cm^-1")
                energy = results[i]["energy"]
                ens = Constants.convert(energy, "wavenumbers", to_AU=False)
                print(f"vOH = {i} : E0+ : {ens[0]}  E0- : {ens[1]} / {ens[1]-ens[0]:.3f} \n"
                      f"            E1+ : {ens[2]} / {ens[2]-ens[0]:.3f} E1- : {ens[3]} / {ens[3]-ens[0]:.3f} \n"
                      f"            E2+ : {ens[4]} / {ens[4]-ens[0]:.3f} E2- : {ens[5]} / {ens[5]-ens[0]:.3f} \n"
                      f"            E3+ : {ens[6]} / {ens[6]-ens[0]:.3f} E3- : {ens[7]} / {ens[7]-ens[0]:.3f} \n")
        wfns = PORobj.PORwfns()  # returns a list of arrays with each wfn for each torsional potential
        return results, wfns

    def calc_Vcoeffs(self, func):
        from Converter import Constants
        from FourierExpansions import fourier_coeffs, calc_cos_coefs
        print("Expanding potential coefficients in Full Fourier...")
        dvr_energies = self.DegreeDVRresults["energies"]  # [degrees vOH=0 ... vOH=6]
        dvr_energies[:, 1:] = Constants.convert(dvr_energies[:, 1:], "wavenumbers", to_AU=True)
        coeff_dict = dict()  # build coeff dict
        rad = np.radians(dvr_energies[:, 0])
        coeff_dict["Vel"] = self.VelCoeffs
        if func == "cos":
            for i in np.arange(1, dvr_energies.shape[1]):  # loop through saved energies
                energies = np.column_stack((rad, dvr_energies[:, i]))
                coeff_dict[f"V{i - 1}"] = calc_cos_coefs(energies)
        else:
            for i in np.arange(1, dvr_energies.shape[1]):  # loop through saved energies
                energies = np.column_stack((rad, dvr_energies[:, i]))
                coeff_dict[f"V{i - 1}"] = fourier_coeffs(energies)
        return coeff_dict

    def run_tor_DVR(self):
        from PiDVR import PiDVR
        from Converter import Constants
        all_res = []
        pot_coeffs = self.calc_Vcoeffs(func="fourier")
        for i in np.arange(len(pot_coeffs.keys()) - 1):
            pot_coeffs1 = pot_coeffs["Vel"] + pot_coeffs[f"V{i}"]
            DVRobj = PiDVR(PotentialCoeffs=pot_coeffs1,
                           GmatCoeffs=self.fittedGmatrix[i], Nval=100,
                           params={"desired_energies": self.DVRparams["desired_energies"]})
            all_res.append(DVRobj.Results)
        if "PrintResults" in self.DVRparams:
            for i in np.arange(len(all_res)):
                mins = all_res[i]["minima"]
                mins[:, 1] = Constants.convert(mins[:, 1], "wavenumbers", to_AU=False)
                print(f"QOOH1 : {np.degrees(mins[0, 0])} / {mins[0, 1]} cm^-1")
                print(f"QOOH2 : {np.degrees(mins[1, 0])} / {mins[1, 1]} cm^-1")
                bh = Constants.convert(all_res[i]["barrier"], "wavenumbers", to_AU=False)
                print(f"Barrier Height : {np.degrees(mins[2, 0])} / {bh} cm^-1")
                energy = all_res[i]["energy"]
                ens = Constants.convert(energy, "wavenumbers", to_AU=False)
                Lps = all_res[i]["probs"][0]
                Rps = all_res[i]["probs"][1]
                print(f"vOH = {i} : E0 : {ens[0]} L:{Lps[0]} R: {Rps[0]} \n "
                      f"            E1 : {ens[1]} / {ens[1]-ens[0]:.3f} L:{Lps[1]} R: {Rps[1]}\n"
                      f"            E2 : {ens[2]} / {ens[2]-ens[0]:.3f} L:{Lps[2]} R: {Rps[2]}\n"
                      f"            E3 : {ens[3]} / {ens[3]-ens[0]:.3f} L:{Lps[3]} R: {Rps[3]}\n"
                      f"            E4 : {ens[4]} / {ens[4]-ens[0]:.3f} L:{Lps[4]} R: {Rps[4]}\n"
                      f"            E5 : {ens[5]} / {ens[5]-ens[0]:.3f} L:{Lps[5]} R: {Rps[5]}\n"
                      f"            E6 : {ens[6]} / {ens[6]-ens[0]:.3f} L:{Lps[6]} R: {Rps[6]}\n"
                      f"            E7 : {ens[7]} / {ens[7]-ens[0]:.3f} L:{Lps[7]} R: {Rps[7]}\n")
        return all_res

    def run_transitions(self):
        from TransitionMoments import TransitionMoments
        if "None" in self.PORparams:
            trans_mom_obj = TransitionMoments(self.DegreeDVRresults, self.torDVRresults, self.MoleculeInfo,
                                              self.PORparams, transition=self.transition)
        else:
            trans_mom_obj = TransitionMoments(self.DegreeDVRresults, self.PORresults, self.MoleculeInfo,
                                              self.PORparams, transition=self.transition)

        intensities = trans_mom_obj.calc_intensity(numGstates=self.Intenseparams["numGstates"],
                                                   numEstates=self.Intenseparams["numEstates"],
                                                   FC=self.Intenseparams["FranckCondon"],
                                                   twoD=self.PORparams["twoD"])
        # trans_mom_obj.plot_sticks(numGstates=self.Intenseparams["numGstates"],
        #                                            numEstates=self.Intenseparams["numEstates"],
        #                                            FC=self.Intenseparams["FranckCondon"],
        #                                            twoD=self.PORparams["twoD"])
        # ordered dict keyed by transition, holding a list (see below) for all the tor transitions
        # [tor gstate, tor exstate, frequency (cm^-1), intensity (km/mol), BW, Boltzmann-Weighted Intensity (km/mol)]
        return intensities

    def run_StatTransitions(self):
        from TransitionMoments import TransitionMoments
        trans_mom_obj = TransitionMoments(self.DegreeDVRresults, self.PORresults, self.MoleculeInfo,
                                          self.PORparams, transition=self.transition)
        intensities = trans_mom_obj.calc_stat_intensity()
        # ordered dict keyed by transition, holding a list (see below) for all the tor transitions
        # [tor gstate, tor exstate, frequency (cm^-1), intensity (km/mol)]
        return intensities
