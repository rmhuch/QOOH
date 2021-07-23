import numpy as np
import os
from PyDVR.DVR import *


class MoleculeInfo2D:
    def __init__(self, MoleculeName, atom_array, eqTORangle, oh_scan_npz=None, tor_scan_npz=None, cartesians_npz=None,
                 TorFchkDirectory=None, TorFiles=None, ModeFchkDirectory=None, ModeFiles=None, dipole_npz=None):
        self.MoleculeName = MoleculeName
        self.atom_array = atom_array
        self.eqTORangle = eqTORangle
        self.oh_scan_npz = oh_scan_npz
        self.tor_scan_npz = tor_scan_npz
        self.cart_npz = cartesians_npz
        self.TorFchkDirectory = TorFchkDirectory  # to calculate Vel + rxn path
        self.TorFiles = TorFiles
        self.ModeFchkDirectory = ModeFchkDirectory  # to calculate modes from rxn path
        self.ModeFiles = ModeFiles  # must be proper file names as str not just changing piece
        self.dipole_npz = dipole_npz
        self._MoleculeDir = None
        self._mass_array = None
        self._OHScanDict = None
        self._TORScanData = None
        self._TorScanDict = None
        self._cartesians = None
        self._GmatCoords = None
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
    def OHScanDict(self):
        if self._OHScanDict is None:
            OH_scanData = self.get_Data(npz_filename=self.oh_scan_npz)
            self._OHScanDict = self.get_GroupedDict(dat=OH_scanData)
        return self._OHScanDict

    @property
    def TORScanData(self):
        if self._TORScanData is None:
            self._TORScanData = self.get_Data(npz_filename=self.tor_scan_npz)
        return self._TORScanData

    @property
    def TorScanDict(self):
        if self._TorScanDict is None:
            self._TorScanDict = self.get_GroupedDict(dat=self.TORScanData)
        return self._TorScanDict

    @property
    def cartesians(self):
        if self._cartesians is None:
            self._cartesians = self.get_finalCarts()
        return self._cartesians

    @property
    def GmatCoords(self):
        if self._GmatCoords is None:
            self._GmatCoords = self.get_GmatCoords()
        return self._GmatCoords

    @property
    def DipoleMomentSurface(self):
        if self._DipoleMomentSurface is None:
            self._DipoleMomentSurface = self.get_DipoleDict()
            # pulls in data as a dictionary that is keyed by angles, with values: (roh, x, y, z)
        return self._DipoleMomentSurface

    def get_Data(self, npz_filename=None):
        if npz_filename is None:
            raise Exception("no data to read")
        else:
            vals = np.load(os.path.join(self.MoleculeDir, npz_filename), allow_pickle=True)
        # this loads in a dictionary of internals and energies for every point in the COOH, CH2, OH scan
        return vals

    def get_GroupedDict(self, dat):
        """ Takes Internal Dict and converts it into a grouped nested, dict {(HOOC, OCCX} : {'coord_name' : val}
         with val in degree/angstrom"""
        from Grouper import Grouper
        all_names = dat.files
        int_coord_array = np.column_stack((list(dat.values())))
        group = Grouper(int_coord_array, (30, 28))  # groups into (HOOC, OCCX)
        Gdict = dict()
        for k, v in group.group_dict.items():
            val_dict = {kk: val for kk, val in zip(all_names, v.T)}
            Gdict[k] = val_dict
        return Gdict

    def get_finalCarts(self):
        """ Pulls Cartesians from scan file, then saves only the optimized geometry's cartesian coordinates as a grouped
         dict {(HOOC, OCCX) : [cartesians] """
        from Grouper import Grouper
        from McUtils.Numputils import pts_dihedrals
        carts = np.load(os.path.join(self.MoleculeDir, self.cart_npz), allow_pickle=True)
        HOOC = []
        OCCH = []
        OCCHp = []
        all_coords = []
        for file in carts.keys():
            coord_array = carts[file]
            HOOC.extend(np.degrees(
                pts_dihedrals(coord_array[:, 6], coord_array[:, 5], coord_array[:, 4], coord_array[:, 0])))
            OCCH.extend(np.degrees(
                pts_dihedrals(coord_array[:, 4], coord_array[:, 0], coord_array[:, 1], coord_array[:, 3])))
            OCCHp.extend(np.degrees(
                pts_dihedrals(coord_array[:, 4], coord_array[:, 0], coord_array[:, 1], coord_array[:, 2])))
            all_coords.extend(coord_array)
        HOOC = np.round(HOOC)
        HOOC = [-x if x < 0 else 360 - x for x in HOOC]
        OCCH = np.round(OCCH)
        OCCH = np.array([-x if x < 0 else 360 - x for x in OCCH])
        OCCHp = np.round(OCCHp)
        OCCHp = np.array([-x if x < 0 else 360 - x for x in OCCHp])
        OCCX = ((OCCH + OCCHp)/2)
        round_OCCX = np.round(OCCX/10)*10

        groupie = Grouper(all_coords, np.array((HOOC, round_OCCX)).T)
        all_carts = groupie.group_dict
        finalCarts = {k: v[-1] for k, v in all_carts.items()}
        for t1, t2 in list(finalCarts.keys()):
            if t2 == 0:
                finalCarts[(t1, 180)] = finalCarts[(t1, 0)]
            elif t2 == 180:
                finalCarts[(t1, 0)] = finalCarts[(t1, 180)]
            else:
                pass
        for t1, t2 in list(finalCarts.keys()):
            if t1 == 0:
                finalCarts[(360, t2)] = finalCarts[(0, t2)]
            else:
                pass
        return finalCarts

    def get_GmatCoords(self):
        """ Calculates extra internals from ^finalCarts and combines with other internal dictionary to return one
        groupedDict keyed by tuple (COOH, OCCH) """
        from McUtils.Numputils import vec_angles, pts_dihedrals
        internals = self.TorScanDict
        carts = self.cartesians
        for k, coord_array in carts.items():
            internals[k]["ACCpH"] = np.degrees(
                vec_angles(coord_array[0]-coord_array[1], coord_array[3]-coord_array[1])[0])
            internals[k]["ACCpHp"] = np.degrees(
                vec_angles(coord_array[0]-coord_array[1], coord_array[2]-coord_array[1])[0])
            internals[k]["AOCCp"] = np.degrees(
                vec_angles(coord_array[4]-coord_array[0], coord_array[1]-coord_array[0])[0])
            internals[k]["DOCCpH"] = np.degrees(
                pts_dihedrals(coord_array[4], coord_array[0], coord_array[1], coord_array[3]))
            internals[k]["DOCCpHp"] = np.degrees(
                pts_dihedrals(coord_array[4], coord_array[0], coord_array[1], coord_array[2]))
        return internals

    def get_DipoleDict(self):
        dat = np.load(os.path.join(self.MoleculeDir, self.dipole_npz))
        dipoleDict = dict()
        for file in dat.files:
            hooc_str, occx_str = file.split("_")
            HOOC = int(hooc_str)
            OCCX = int(occx_str)
            dipoleDict[(HOOC, OCCX)] = dat[file][:, :4]
            dipoleDict[(HOOC, (OCCX+180))] = dat[file][:, :4]
        return dipoleDict

    def plot_DipoleSurface(self, component="a"):
        import matplotlib.pyplot as plt
        unHOOC = np.unique([k[0] for k in self.DipoleMomentSurface.keys()])
        unOCCX = np.unique([k[1] for k in self.DipoleMomentSurface.keys()])
        XX, YY = np.meshgrid(unHOOC, unOCCX)
        zz = np.zeros(XX.shape)
        if component == "a":
            component = 1
        elif component == "b":
            component = 2
        elif component == "c":
            component = 3
        else:
            raise Exception(f"No Dipole for {component} component.")
        for i, j in enumerate(unHOOC):
            for ip, jp in enumerate(unOCCX):
                diff = self.DipoleMomentSurface[(j, jp)][:, 0] - 0.96
                eq_ind = np.argmin(abs(diff))
                if eq_ind != 6:
                    raise Exception(f"({j}, {jp}) eq_ind = {eq_ind}")
                # we assign in "switched" indices b.c. mesh grid is not "ij" indexed :angry_face:
                zz[ip, i] = self.DipoleMomentSurface[(j, jp)][eq_ind, component]
        plt.contourf(XX, YY, zz)
        plt.colorbar()
        plt.show()

class MolecularResults2D:
    def __init__(self, MolObj, DVRparams, TransitionStr):
        from TorTorGmatrix import TorTorGmatrix
        self.MoleculeObj = MolObj
        self.MoleculeDir = MolObj.MoleculeDir
        self.GmatObj = TorTorGmatrix(self.MoleculeObj)
        self.GHdpHdp = self.GmatObj.GHdpHdp
        self.GXX = self.GmatObj.GXX
        self.GXHdp = self.GmatObj.GXHdp
        self.DVRparams = DVRparams
        self.Transition = TransitionStr
        self._Adiabats = None
        self._Gderiv_XX = None
        self._Gderiv_HdpHdp = None
        self._squareCoords = None
        self._TwoDdvrRes = None

    @property
    def Gderiv_XX(self):
        if self._Gderiv_XX is None:
            def getGXX_deriv(coords):
                return self.GXX[1](coords, dx=0, dy=2)
            self._Gderiv_XX = getGXX_deriv
        return self._Gderiv_XX

    @property
    def Gderiv_HdpHdp(self):
        if self._Gderiv_HdpHdp is None:
            def getGHdpHdp_deriv(coords):
                return self.GHdpHdp[1](coords, dx=2, dy=0)
            self._Gderiv_HdpHdp = getGHdpHdp_deriv
        return self._Gderiv_HdpHdp

    @property
    def squareCoords(self):
        """ this duplicates the OCCX data so that it is on (0, 2pi) instead of (0, pi)
         ULTIMATELY THE DIPOLE MOMENT WILL NEED TO BE INCORPORATED HERE"""
        if self._squareCoords is None:
            twoD_dat = np.column_stack((
                self.MoleculeObj.TORScanData["D4"], self.MoleculeObj.TORScanData["D2"],
                (self.MoleculeObj.TORScanData["Energy"] - min(self.MoleculeObj.TORScanData["Energy"]))))
            rm_180 = np.delete(twoD_dat, np.argwhere(twoD_dat[:, 1] == 0), axis=0)
            # remove OCCX == 0 before duplicating so we do not get 2 180's
            twoD_dat_again = np.concatenate([twoD_dat, rm_180 + np.array([[0, 180, 0]])], axis=0)
            # this duplicates the OCCX data so that it is on (0, 2pi) instead of (0, pi)
            sort_ind = np.lexsort((twoD_dat_again[:, 1], twoD_dat_again[:, 0]))
            twoD_grid = twoD_dat_again[sort_ind]
            self._squareCoords = twoD_grid
        return self._squareCoords  # degrees, hartrees

    @property
    def Adiabats(self):
        if self._Adiabats is None:
            resultspath = os.path.join(self.MoleculeObj.MoleculeDir,
                                       f"{self.MoleculeObj.MoleculeName}_Adiabat_EpsilonPots_02pi_30.npy")
            if os.path.exists(resultspath):
                self._Adiabats = np.load(resultspath)
            else:
                print(f"{resultspath} not found. beginning OH DVR")
                results = self.Build_adiabats()
                self._Adiabats = np.load(results)
        return self._Adiabats

    @property
    def TwoDdvrRes(self):
        if self._TwoDdvrRes is None:
            resDict = dict()
            for i, val in enumerate(['VOH_0', 'VOH_1', 'VOH_2', 'VOH_3']):
                resultspath = os.path.join(self.MoleculeDir, f"FullG_2D_DVR_{val}.npz")
                # if os.path.exists(resultspath):
                #     print(f"Loading {resultspath}... ")
                #     resDict[val] = np.load(resultspath, allow_pickle=True)
                # else:
                print(f"Calculating {resultspath}... ")
                res = self.TwoDTorTor_FullG(val)
                resDict[val] = np.load(res, allow_pickle=True)
        return resDict

    def plot_Surfaces(self, xcoord, ycoord, zcoord=None, title=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d
        fig = plt.figure()
        if zcoord is None:
            plt.plot(xcoord, ycoord, "o")
            plt.show()
        else:
            ax = plt.axes(projection='3d')
            ax.scatter(xcoord, ycoord, zcoord, c="k")
            # x = np.unique(xcoord)
            # y = np.unique(ycoord)
            # X, Y = np.meshgrid(x, y)
            # z = np.reshape(zcoord, (len(y), len(x)))
            # ax.plot_surface(X, Y, z)
            ax.set_xlabel("COOH Torsion")
            ax.axes.set_xticks(np.arange(0, 390, 60))
            ax.set_ylabel("OCCH Torsion")
            ax.axes.set_yticks(np.arange(0, 240, 60))
            ax.set_zlabel("Electronic Energy")
            plt.title(title)

    def Build_adiabats(self):
        """Runs 1D OH DVR at every scan point, returns array([HOOC, OCCX, En0, En1, En2, En3])  """
        from DegreeDVR import run_2D_OHDVR
        params = self.DVRparams
        potential_array, epsilon_pots, wavefuns_array = run_2D_OHDVR(self.MoleculeObj.OHScanDict,
                                                                     desiredEnergies=params["desired_energies"],
                                                                     NumPts=params["num_pts"],
                                                                     plotPhasedWfns=params["plot_phased_wfns"],
                                                                     extrapolate=params["extrapolate"])
        rm_180 = np.delete(epsilon_pots, np.argwhere(epsilon_pots[:, 1] == 0), axis=0)
        # remove OCCX == 0 before duplicating so we do not get 2 180's
        full_epsilon_pots = np.concatenate([epsilon_pots, rm_180 + np.array([[0, 180, 0, 0, 0, 0]])])
        sort_ind = np.lexsort((full_epsilon_pots[:, 1], full_epsilon_pots[:, 0]))
        finalGrid = full_epsilon_pots[sort_ind]
        file_name = os.path.join(self.MoleculeDir, f"{self.MoleculeObj.MoleculeName}_Adiabat_EpsilonPots_02pi_30.npy")
        np.save(file_name, finalGrid)
        return file_name

    def TwoDTorTor_constG(self):
        """Runs 2D DVR using a Constant (HOOC=210, OCCX=150) value for the G-matrix"""
        from Converter import Constants
        from McUtils.Plots import ContourPlot
        dvr_2D = DVR("ColbertMillerND")
        print("Conducting 2D DVR with Constant G")
        gHdpHdp = self.GHdpHdp[0][26, 15]  # calls the calculate g-mat element at the equilibrium geom
        gXX = self.GXX[0][26, 15]
        squareCoords = self.squareCoords.copy()
        squareCoords[:2] = np.radians(self.squareCoords[:2])
        res = dvr_2D.run(potential_grid=squareCoords, flavor="[0,2Pi]",
                         divs=(51, 51), mass=[1/gHdpHdp, 1/gXX], num_wfns=25,
                         domain=((min(squareCoords[:, 0]),  max(squareCoords[:, 0])),
                                 (min(squareCoords[:, 1]), max(squareCoords[:, 1]))),
                         results_class=ResultsInterpreter)
        res.plot_potential(plot_class=ContourPlot, plot_units="wavenumbers", energy_threshold=2000, colorbar=True,
                           plot_style=dict(levels=15)).show()
        all_ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
        print(all_ens)
        # ResultsInterpreter.wfn_contours(res)
        return res

    def TwoDTorTor_diagG(self):
        """Runs 2D DVR using a Diagonal G-matrix (GH''H'' & GXX)"""
        from Converter import Constants
        dvr_2D = DVR("ColbertMillerND")
        print("Conducting 2D DVR with Diagonal G")
        squareCoords = self.squareCoords.copy()
        squareCoords[:2] = np.radians(self.squareCoords[:2])
        res = dvr_2D.run(potential_grid=squareCoords, flavor="[0,2Pi]",
                         divs=(51, 51), g=[[self.GHdpHdp[1], 0], [0, self.GXX[1]]],
                         g_deriv=[self.Gderiv_HdpHdp, self.Gderiv_XX], num_wfns=25,
                         domain=((min(squareCoords[:, 0]),  max(squareCoords[:, 0])),
                                 (min(squareCoords[:, 1]), max(squareCoords[:, 1]))),
                         results_class=ResultsInterpreter)
        all_ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
        print(all_ens)
        # ResultsInterpreter.wfn_contours(res)
        return res

    def TwoDTorTor_FullG(self, val):
        """Runs 2D DVR using the Full 2x2 G-matrix array (GH''H'' & GXX & GXH''"""
        from Converter import Constants
        from McUtils.Plots import ContourPlot
        dvr_2D = DVR("ColbertMillerND")
        print("Conducting 2D DVR with Full G")
        Epot = self.squareCoords
        Epot1 = np.delete(Epot, np.argwhere(Epot[:, 0] % 30 != 0), axis=0)
        Epot2 = np.delete(Epot1, np.argwhere(Epot1[:, 1] % 30 != 0), axis=0)
        adiabat_pots = self.Adiabats
        # run for given adiabat, loop outside of this function
        npz_filename = os.path.join(self.MoleculeDir, f"FullG_2D_DVR_{val}.npz")
        i = int(val[-1])
        potential = Epot2[:, 2] + adiabat_pots[:, 2+i]
        pot_vals = np.column_stack((np.radians(self.Adiabats[:, 0]), np.radians(self.Adiabats[:, 1]),
                                    potential))
        res = dvr_2D.run(potential_grid=pot_vals, flavor="[0,2Pi]",
                         divs=(51, 51), g=[[self.GHdpHdp[1], self.GXHdp[1]], [self.GXHdp[1], self.GXX[1]]],
                         g_deriv=[self.Gderiv_HdpHdp, self.Gderiv_XX], num_wfns=10,
                         domain=((min(pot_vals[:, 0]),  max(pot_vals[:, 0])),
                                 (min(pot_vals[:, 1]), max(pot_vals[:, 1]))),
                         results_class=ResultsInterpreter)
        dvr_grid = np.degrees(res.grid)
        dvr_pot = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
        # res.plot_potential(plot_class=ContourPlot, plot_units="wavenumbers", colorbar=True,
        #                    plot_style=dict(levels=15)).show()
        all_ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
        print(val, all_ens)
        print(min(dvr_pot))
        ResultsInterpreter.wfn_contours(res)
        np.savez(npz_filename, grid=[dvr_grid], potential=[dvr_pot],
                 energy_array=all_ens, wfns_array=res.wavefunctions.wavefunctions)
        return npz_filename

    def TwoD_transitions(self):
        from TransitionMoments2D import TransitionMoments2D
        tm_obj = TransitionMoments2D("OH_res", self.TwoDdvrRes, self.MoleculeObj, transition=self.Transition)
        AllIntensities = tm_obj.calc_intensity(numGstates=2, numEstates=10)
        return AllIntensities
