import numpy as np
import os

class MoleculeInfo:
    def __init__(self, MoleculeName, atom_array, eqTORangle, oh_scan_npz=None,
                 TorFchkDirectory=None, TorFiles=None, ModeFchkDirectory=None, ModeFiles=None):
        self.MoleculeName = MoleculeName
        self.atom_array = atom_array
        self.eqTORangle = eqTORangle
        self.oh_scan_npz = oh_scan_npz
        self.TorFchkDirectory = TorFchkDirectory  # to calculate Vel + rxn path
        self.TorFiles = TorFiles
        self.ModeFchkDirectory = ModeFchkDirectory  # to calculate modes from rxn path
        self.ModeFiles = ModeFiles  # must be proper file names as str not just changing piece
        self._MoleculeDir = None
        self._mass_array = None
        self._PES_DegreeDict = None
        self._eqPES = None
        self._InternalCoordDict = None
        self._eqICD = None

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
    def __init__(self, MoleculeInfo_obj, DVRparams, PORparams, transition="0->2"):
        self.MoleculeInfo = MoleculeInfo_obj
        self.DVRparams = DVRparams
        self.PORparams = PORparams
        self.transition = transition
        self._DegreeDVRresults = None
        self._OHmodes = None
        self._TORmodes = None
        self._PotentialCoeffs = None
        self._Gmatrix = None
        self._RxnPathResults = None
        self._PORresults = None
        self._TransitionIntensities = None
        self._OscillatorStrength = None

    @property
    def DegreeDVRresults(self):
        if self._DegreeDVRresults is None:
            resultspath = os.path.join(self.MoleculeInfo.MoleculeDir,
                                       f"{self.MoleculeInfo.MoleculeName}_vOH_DVRresults_{self.DVRparams['num_pts']}.npz")
            if os.path.exists(resultspath):
                self._DegreeDVRresults = np.load(resultspath)
                print(f"Degree DVR Results loaded from {resultspath}")
            else:
                results = self.run_degree_DVR()
                self._DegreeDVRresults = np.load(results)
        return self._DegreeDVRresults

    @property
    def Gmatrix(self):
        if self._Gmatrix is None:
            self._Gmatrix = self.run_gmatrix()
        return self._Gmatrix

    @property
    def RxnPathResults(self):
        if self._RxnPathResults is None:
            self._RxnPathResults = self.run_reaction_path()
        return self._RxnPathResults

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
        return res, eq_g

    def run_reaction_path(self):
        from ReactionPath import run_energies
        fchkDir = os.path.join(self.MoleculeInfo.MoleculeDir, self.MoleculeInfo.TorFchkDirectory)
        rxn_path_res = run_energies(fchkDir, self.MoleculeInfo.TorFiles)
        return rxn_path_res

    def calculate_VelwZPE(self):
        from Converter import Constants
        from FourierExpansions import calc_cos_coefs
        degree_vals = np.linspace(0, 360, len(self.MoleculeInfo.TorFiles))
        norm_grad_2 = self.RxnPathResults["norm_grad_2"]
        idx = np.where(norm_grad_2[:, 1] > 5E-8)
        new_degree = degree_vals[idx]
        Vel = self.RxnPathResults["electronicE"][:, 1]
        new_Vel = Vel[idx]
        Vel_ZPE = np.zeros((len(new_degree), 2))
        for i, j in enumerate(new_degree):
            freqs = self.RxnPathResults[j]["freqs"]
            nonzero_freqs = freqs[7:-1]  # throw out translations/rotations and OH frequency
            nonzero_freqs_har = Constants.convert(nonzero_freqs, "wavenumbers", to_AU=True)
            ZPE = np.sum(nonzero_freqs_har)/2
            Vel_ZPE[i] = np.column_stack((np.radians(j), new_Vel[i] + ZPE))
        Vel_ZPE[:, 1] -= min(Vel_ZPE[:, 1])
        VelwZPE_coeffs = calc_cos_coefs(Vel_ZPE)
        return VelwZPE_coeffs

    def run_reaction_path_for_modes(self, save_file_header):
        from ReactionPath import run_modes
        for i in self.MoleculeInfo.ModeFiles:
            fname = os.path.join(self.MoleculeInfo.MoleculeDir, self.MoleculeInfo.ModeFchkDirectory, i)
            run_modes(fname, save_file_header)
        # this saves frequency and mode dat files

    def make_OOHfreq_plot(self, rOH, freq_dir_name, mode_dir_name, vibfile_handles, fig_filename):
        # COME BACK TO THIS. VERY VERY SLOPPY IMPLEMENTATION
        from PlotModes import loadModeData, plot_OOHvsrOH
        VIBfreqdir = os.path.join(self.MoleculeInfo.MoleculeDir, self.MoleculeInfo.ModeFchkDirectory, freq_dir_name)
        VIBmodedir = os.path.join(self.MoleculeInfo.MoleculeDir, self.MoleculeInfo.ModeFchkDirectory, mode_dir_name)
        dat = loadModeData(vibfile_handles, VIBfreqdir, VIBmodedir)
        plot_OOHvsrOH(dat, vibfile_handles, VIBmodedir, fig_filename)

    def run_tor_adiabats(self):
        from TorsionPOR import POR
        if "barrier_height" in self.PORparams:
            PORobj = POR(DVR_res=self.DegreeDVRresults, g_matrix=self.Gmatrix[0], Velcoeffs=self.calculate_VelwZPE(),
                         HamSize=self.PORparams["HamSize"], barrier_height=self.PORparams["barrier_height"])
        else:
            PORobj = POR(DVR_res=self.DegreeDVRresults, g_matrix=self.Gmatrix[0], Velcoeffs=self.calculate_VelwZPE(),
                         HamSize=self.PORparams["HamSize"])
        results = PORobj.solveHam()  # returns a list of dictionaries for each torsional potential
        wfns = PORobj.PORwfns()  # returns a list of arrays with each wfn for each torsional potential
        return results, wfns
