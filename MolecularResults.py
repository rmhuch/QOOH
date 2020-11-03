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
            E = vals[f"{i}.0"][0]
            rOH = vals[f"{i}.0"][1]["B6"]
            degreedict[i] = np.column_stack((rOH, E))
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
            full_dict = vals[f"{i}.0"][1]
            for j in full_dict.keys():  # go through and if the value is just repeated only return one value.
                if full_dict[j][0] == full_dict[j][1] and full_dict[j][0] == full_dict[j][2]:
                    full_dict[j] = full_dict[j][0]
            internaldict[i] = full_dict
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
        self._RxnPathResults = None
        self._VelCoeffs = None
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
        #     if os.path.exists(resultspath):
        #         self._Gmatrix = np.load(resultspath)
        #     else:
            print(f"{resultspath} not found. Beginning Gmatrix calculation")
            results = self.run_gmatrix()
            self._Gmatrix = np.load(results)
        return self._Gmatrix

    @property
    def RxnPathResults(self):
        if self._RxnPathResults is None:
            self._RxnPathResults = self.run_reaction_path()
        return self._RxnPathResults

    @property
    def VelCoeffs(self):
        if self._VelCoeffs is None:
            if "EmilData" in self.PORparams or "MixedData1" in self.PORparams:
                resultspath = os.path.join(self.MoleculeInfo.MoleculeDir,
                                           f"{self.MoleculeInfo.MoleculeName}_Emil_Velcoeffs_4order.npy")
                if os.path.exists(resultspath):
                    print(f"Using {resultspath} to calculate Vel + ZPE coeffs")
                    self._VelCoeffs = np.load(resultspath)
                else:
                    print(f"{resultspath} not found. beginning Vel + ZPE calculation")
                    results = self.calculate_VelwZPE()
                    self._VelCoeffs = np.load(results)
            else:
                if self.PORparams["Vexpansion"] is "sixth":
                    resultspath = os.path.join(self.MoleculeInfo.MoleculeDir,
                                               f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6order.npy")
                elif self.PORparams["Vexpansion"] is "fourth":
                    resultspath = os.path.join(self.MoleculeInfo.MoleculeDir,
                                               f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_4order.npy")
                else:
                    raise Exception(f"Can not expand Vel in {self.PORparams['Vexpansion']}")
                if os.path.exists(resultspath):
                    print(f"Using {resultspath} to calculate Vel + ZPE coeffs")
                    self._VelCoeffs = np.load(resultspath)
                else:
                    print(f"{resultspath} not found. beginning Vel + ZPE calculation")
                    results = self.calculate_VelwZPE()
                    self._VelCoeffs = np.load(results)
        return self._VelCoeffs

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
        # print(epsilon_pots)
        print(f"DVR Results saved to {npz_name} in {self.MoleculeInfo.MoleculeDir}")
        return os.path.join(self.MoleculeInfo.MoleculeDir, npz_name)

    def run_gmatrix(self):
        from ExpectationValues import run_DVR
        from Gmatrix import get_tor_gmatrix, get_eq_g
        Epot_array = self.MoleculeInfo.eqPES
        params = self.DVRparams
        ens, rOH_wfns = run_DVR(Epot_array, NumPts=params["num_pts"], desiredEnergies=params["desired_energies"],
                                extrapolate=params["extrapolate"])
        tor_angles = self.MoleculeInfo.PES_DegreeDict.keys()
        mass_array = self.MoleculeInfo.mass_array
        tor_masses = np.array((mass_array[6], mass_array[4], mass_array[5], np.inf)) 
        res = get_tor_gmatrix(rOH_wfns, tor_angles, self.MoleculeInfo.InternalCoordDict, tor_masses)
        eq_g = get_eq_g(tor_angles, self.MoleculeInfo.InternalCoordDict, tor_masses)
        # for i in np.arange(len(res)):
        #     np.savetxt(f"B2PLYP_Gmatrix_elements_voh{i}.txt", res[i])
        npz_name = f"{self.MoleculeInfo.MoleculeName}_Gmatrix_elements.npz"
        np.savez(os.path.join(self.MoleculeInfo.MoleculeDir, npz_name), gmatrix=res, eq_gmatrix=eq_g)
        return os.path.join(self.MoleculeInfo.MoleculeDir, npz_name)

    def run_reaction_path(self):
        from ReactionPath import run_energies, run_Emil_energies
        fchkDir = os.path.join(self.MoleculeInfo.MoleculeDir, self.MoleculeInfo.TorFchkDirectory)
        rxn_path_res = run_energies(fchkDir, self.MoleculeInfo.TorFiles)
        if "EmilData" in self.PORparams or "MixedData1" in self.PORparams:
            datfile = os.path.join(self.MoleculeInfo.MoleculeDir, "Emil_Energies_tor_OH.txt")
            EmilelectronicE = run_Emil_energies(datfile)
            rxn_path_res["EmilelectronicE"] = EmilelectronicE
        return rxn_path_res

    def calculate_VelwZPE(self):
        from Converter import Constants
        from FourierExpansions import calc_cos_coefs, calc_4cos_coefs, calc_curves
        import matplotlib.pyplot as plt
        degree_vals = np.linspace(0, 360, len(self.MoleculeInfo.TorFiles))
        norm_grad = self.RxnPathResults["norm_grad"]
        idx = np.where(norm_grad[:, 1] > 4E-4)
        new_degree = degree_vals[idx]
        Vel = self.RxnPathResults["electronicE"][:, 1]
        Vel_ZPE_dict = dict()
        ZPE_dict = dict()
        for i, j in enumerate(degree_vals):
            freqs = self.RxnPathResults[j]["freqs"]  # in hartree
            nonzero_freqs = freqs[7:-1]  # throw out translations/rotations and OH frequency
            nonzero_freqs_har = Constants.convert(nonzero_freqs, "wavenumbers", to_AU=True)
            ZPE = np.sum(nonzero_freqs_har)/2
            if j in new_degree:
                Vel_ZPE_dict[j] = Vel[i] + ZPE
                ZPE_dict[j] = ZPE
            else:
                pass
        if "EmilData" in self.PORparams or "MixedData1" in self.PORparams:
            # put together ZPE
            print("CCSD Vel")
            ZPE = np.array([(d, v) for d, v in ZPE_dict.items()])
            sort_idxZ = np.argsort(ZPE[:, 0])
            ZPE = ZPE[sort_idxZ]
            ZPE[:, 0] = np.radians(ZPE[:, 0])
            fit_ZPE = calc_4cos_coefs(ZPE)
            zpe_y = calc_curves(np.radians(np.arange(0, 360, 1)), fit_ZPE, function="4cos")
            emil_angles = self.RxnPathResults["EmilelectronicE"][:, 0]
            emil_ZPE = calc_curves(np.radians(emil_angles), fit_ZPE, function="4cos")
            Vel_ZPE = np.column_stack((np.radians(emil_angles), emil_ZPE+self.RxnPathResults["EmilelectronicE"][:, 1]))
            VelwZPE_coeffs1 = calc_4cos_coefs(Vel_ZPE)
            new_x = np.radians(np.arange(0, 360, 1))
            y = calc_curves(new_x, VelwZPE_coeffs1, function="4cos")
            y -= min(y)  # shift curve so minima are at 0 instead of negative
            VelwZPE_coeffs = calc_4cos_coefs(np.column_stack((new_x, y)))
            npy_name = f"{self.MoleculeInfo.MoleculeName}_Emil_Velcoeffs_4order.npy"
        else:
            print("DFT Vel")
            Vel_ZPE = np.array([(d, v) for d, v in Vel_ZPE_dict.items()])
            sort_idx = np.argsort(Vel_ZPE[:, 0])
            Vel_ZPE = Vel_ZPE[sort_idx]
            Vel_ZPE[:, 0] = np.radians(Vel_ZPE[:, 0])
            VelwZPE_coeffs1 = calc_cos_coefs(Vel_ZPE)
            new_x = np.radians(np.arange(0, 360, 1))
            y = calc_curves(new_x, VelwZPE_coeffs1)
            y -= min(y)  # shift curve so minima are at 0 instead of negative
            if "MixedData2" in self.PORparams or self.PORparams["Vexpansion"] is "fourth":
                VelwZPE_coeffs = calc_4cos_coefs(np.column_stack((new_x, y)))
                npy_name = f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_4order.npy"
            else:
                VelwZPE_coeffs = calc_cos_coefs(np.column_stack((new_x, y)))
                npy_name = f"{self.MoleculeInfo.MoleculeName}_Velcoeffs_6order.npy"
        # save results
        np.save(os.path.join(self.MoleculeInfo.MoleculeDir, npy_name), VelwZPE_coeffs)
        return os.path.join(self.MoleculeInfo.MoleculeDir, npy_name)

    # def run_reaction_path_for_modes(self, save_file_header):
    #     from ReactionPath import run_modes
    #     for i in self.MoleculeInfo.ModeFiles:
    #         fname = os.path.join(self.MoleculeInfo.MoleculeDir, self.MoleculeInfo.ModeFchkDirectory, i)
    #         run_modes(fname, save_file_header)
    #     # this saves frequency and mode dat files
    #
    # def make_OOHfreq_plot(self, rOH, freq_dir_name, mode_dir_name, vibfile_handles, fig_filename):
    #     # COME BACK TO THIS. VERY VERY SLOPPY IMPLEMENTATION
    #     from PlotModes import loadModeData, plot_OOHvsrOH
    #     VIBfreqdir = os.path.join(self.MoleculeInfo.MoleculeDir, self.MoleculeInfo.ModeFchkDirectory, freq_dir_name)
    #     VIBmodedir = os.path.join(self.MoleculeInfo.MoleculeDir, self.MoleculeInfo.ModeFchkDirectory, mode_dir_name)
    #     dat = loadModeData(vibfile_handles, VIBfreqdir, VIBmodedir)
    #     plot_OOHvsrOH(dat, vibfile_handles, VIBmodedir, fig_filename)

    def run_tor_adiabats(self):
        from TorsionPOR import POR
        from Converter import Constants
        if "EmilData" in self.PORparams or "MixedData2" in self.PORparams:
            file_name = os.path.join(self.MoleculeInfo.MoleculeDir, "Emil_vOHenergies.txt")
            self.PORparams["EmilEnergies"] = np.loadtxt(file_name, skiprows=1)
        PORobj = POR(DVR_res=self.DegreeDVRresults, g_matrix=self.Gmatrix["gmatrix"], Velcoeffs=self.VelCoeffs,
                     params=self.PORparams)
        results = PORobj.solveHam()  # returns a list of dictionaries for each torsional potential
        if "PrintResults" in self.PORparams:
            for i in np.arange(len(results)):
                bh = Constants.convert(results[i]["barrier"], "wavenumbers", to_AU=False)
                print(f"Barrier Height : {bh} cm^-1")
                energy = results[i]["energy"]
                print(f"vOH = {i} : E0+ : {Constants.convert(energy[0], 'wavenumbers', to_AU=False)}"
                      f"            E0- : {Constants.convert(energy[1], 'wavenumbers', to_AU=False)} \n"
                      f"            E1+ : {Constants.convert(energy[2], 'wavenumbers', to_AU=False)}"
                      f"            E1- : {Constants.convert(energy[3], 'wavenumbers', to_AU=False)} \n"
                      f"            E2+ : {Constants.convert(energy[4], 'wavenumbers', to_AU=False)}"
                      f"            E2- : {Constants.convert(energy[5], 'wavenumbers', to_AU=False)}")
        wfns = PORobj.PORwfns()  # returns a list of arrays with each wfn for each torsional potential
        return results, wfns

    def run_transitions(self):
        from TransitionMoments import TransitionMoments
        trans_mom_obj = TransitionMoments(self.DegreeDVRresults, self.PORresults, self.MoleculeInfo,
                                          transition=self.transition, MatSize=self.PORparams["HamSize"])
        intensities = trans_mom_obj.calc_intensity(numGstates=self.Intenseparams["numGstates"],
                                                   numEstates=self.Intenseparams["numEstates"])
        # ordered dict keyed by transition, holding a list (see below) for all the tor transitions
        # [tor gstate, tor exstate, frequency (cm^-1), intensity (km/mol), BW, Boltzmann-Weighted Intensity (km/mol)]
        return intensities

    def run_StatTransitions(self):
        from TransitionMoments import TransitionMoments
        trans_mom_obj = TransitionMoments(self.DegreeDVRresults, self.PORresults, self.MoleculeInfo,
                                          transition=self.transition, MatSize=self.PORparams["HamSize"])
        intensities = trans_mom_obj.calc_stat_intensity()
        # ordered dict keyed by transition, holding a list (see below) for all the tor transitions
        # [tor gstate, tor exstate, frequency (cm^-1), intensity (km/mol)]
        return intensities
