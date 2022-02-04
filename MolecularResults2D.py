import numpy as np
import os
import matplotlib.pyplot as plt
from PyDVR.DVR import *
from Converter import Constants

class MoleculeInfo2D:
    def __init__(self, MoleculeName, atom_array, eqTORangle, oh_scan_npz=None, tor_scan_npz=None, cartesians_npz=None,
                 TorFchkDirectory=None, TorFiles=None, ModeFchkDirectory=None, ModeFiles=None, dipole_npz=None,
                 ccsd_potfile=None):
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
        self.ccsd_potfile = ccsd_potfile
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
            # duplicate OCCX data so it is on 0, 2pi
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
            # duplicate OCCX data so it is on 0, 2pi
        return self._TorScanDict

    @property
    def cartesians(self):
        if self._cartesians is None:
            self._cartesians = self.get_finalCarts()
            # checks that HOOC and OCCX are on the 0, 2pi range
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
            # pulls in data as a dictionary that is keyed by angles (HOOC, OCCX), with values: (roh, x, y, z)
            # duplicate OCCX data so it is on 0, 2pi
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
            if k[1] < 180:
                if k[1] == 0:
                    Gdict[(k[0], k[1]+360)] = val_dict
                else:
                    pass
                Gdict[k] = val_dict
                Gdict[(k[0], k[1]+180)] = val_dict  # duplicates the OCCX data
            else:
                pass
        sort_dict = {k: Gdict[k] for k in sorted(Gdict.keys())}
        return sort_dict

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
        HOOC = [x + 360 if x < 0 else x for x in HOOC]
        OCCH = np.round(OCCH)
        OCCHp = np.round(OCCHp)
        OCCX = ((OCCH + OCCHp)/2) + 90
        round_OCCX = np.round(OCCX/10)*10

        groupie = Grouper(all_coords, np.array((HOOC, round_OCCX)).T)
        all_carts = groupie.group_dict
        finalCarts = {k: v[-1] for k, v in all_carts.items()}
        for t1, t2 in list(finalCarts.keys()):  # add in OCCX = 0, then dupe OCCX
            if t2 % 180 == 0:
                finalCarts[(t1, 0)] = finalCarts[(t1, t2)]
                finalCarts[(t1, 180)] = finalCarts[(t1, t2)]
                finalCarts[(t1, 360)] = finalCarts[(t1, t2)]
            else:
                finalCarts[(t1, t2+180)] = finalCarts[(t1, t2)]
        for t1, t2 in list(finalCarts.keys()):  # add in HOOC = 360
            if t1 == 0:
                finalCarts[(360, t2)] = finalCarts[(0, t2)]
            else:
                pass
        sort_dict = {k: finalCarts[k] for k in sorted(finalCarts.keys())}
        return sort_dict

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
            dipoleDict[(HOOC, (OCCX+180))] = dat[file][:, :4]  # duplicates the OCCX data
        sort_dict = {k: dipoleDict[k] for k in sorted(dipoleDict.keys())}
        return sort_dict

    def plot_DipoleSurface(self, component="a"):
        import matplotlib.pyplot as plt
        unHOOC = np.unique([k[0] for k in self.DipoleMomentSurface.keys()])
        unOCCX = np.unique([k[1] for k in self.DipoleMomentSurface.keys()])
        XX, YY = np.meshgrid(unHOOC, unOCCX)
        zz = np.zeros(XX.shape)
        if component == "a":
            component_ind = 1
        elif component == "b":
            component_ind = 2
        elif component == "c":
            component_ind = 3
        else:
            raise Exception(f"No Dipole for {component} component.")
        for i, j in enumerate(unHOOC):
            for ip, jp in enumerate(unOCCX):
                diff = self.DipoleMomentSurface[(j, jp)][:, 0] - 0.96
                eq_ind = np.argmin(abs(diff))
                # we assign in "switched" indices b.c. mesh grid is not "ij" indexed :angry_face:
                zz[ip, i] = self.DipoleMomentSurface[(j, jp)][eq_ind, component_ind]
        plt.contourf(XX, YY, zz)
        plt.ylabel("OCCX")
        plt.yticks(np.arange(0, 390, 30))
        plt.xlabel("HOOC")
        plt.xticks(np.arange(0, 390, 30))
        plt.title(f"{component}-component of the Dipole")
        plt.colorbar()
        plt.show()

class MolecularResults2D:
    def __init__(self, MolObj, DVRparams, TransitionStr, IntensityType):
        from TorTorGmatrix import TorTorGmatrix
        self.MoleculeObj = MolObj
        self.MoleculeDir = MolObj.MoleculeDir
        self.GmatObj = TorTorGmatrix(self.MoleculeObj)
        self.GHdpHdp = self.GmatObj.GHdpHdp
        self.GXX = self.GmatObj.GXX
        self.GXHdp = self.GmatObj.GXHdp
        self.DVRparams = DVRparams
        self.Transition = TransitionStr
        self.IntensityType = IntensityType
        self._OHres = None
        self._Gderiv_XX = None
        self._Gderiv_HdpHdp = None
        self._squareCoords = None
        self._deg_squareCoords = None
        self._CCSDpot = None
        self._ConstGRes = None
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
        """ takes data from TorScanDict and makes an array [HOOC, OCCX, Energy (min shifted)] in rad/hartree"""
        if self._squareCoords is None:
            angs = list(self.MoleculeObj.TorScanDict.keys())
            ens = [self.MoleculeObj.TorScanDict[k]["Energy"] for k in angs]
            ens -= min(ens)
            twoD_dat = np.column_stack((np.radians(np.array(angs)), np.array(ens).flatten()))
            sort_ind = np.lexsort((twoD_dat[:, 1], twoD_dat[:, 0]))
            twoD_grid = twoD_dat[sort_ind]
            self._squareCoords = twoD_grid
        return self._squareCoords  # degrees, hartrees

    @property
    def deg_squareCoords(self):
        """ takes data from TorScanDict and makes an array [HOOC, OCCX, Energy (min shifted)] in DEGREES/hartree"""
        if self._deg_squareCoords is None:
            angs = list(self.MoleculeObj.TorScanDict.keys())
            ens = [self.MoleculeObj.TorScanDict[k]["Energy"] for k in angs]
            ens -= min(ens)
            twoD_dat = np.column_stack((np.array(angs), np.array(ens).flatten()))
            sort_ind = np.lexsort((twoD_dat[:, 1], twoD_dat[:, 0]))
            self._deg_squareCoords = twoD_dat[sort_ind]
        return self._deg_squareCoords  # degrees, hartrees

    @property
    def CCSDpot(self):
        """loads in the Klippenstien CCSD Potential Energy surface [HOOC, OCCX, Energy (min shifted)] in rad/hartree"""
        if self._CCSDpot is None:
            dat = np.loadtxt(os.path.join(self.MoleculeDir, self.MoleculeObj.ccsd_potfile), skiprows=1)
            dat_inds = np.argwhere(dat[:, 1] == 1)  # pull out the 3d coordinate in the scan (we don't need)
            twoD_dat = dat[dat_inds].squeeze()
            ens_har = Constants.convert(twoD_dat[:, 3], "kcal/mol", to_AU=True)
            # ens_wave = Constants.convert(ens_har, "wavenumbers", to_AU=False)
            hooc = [(x-1)*15 for x in twoD_dat[:, 2]]
            hooc_shift = [(x+98) % 360 for x in hooc]
            hooc_2shift = [360 - x for x in hooc_shift]
            occx = [(x-1)*15 for x in twoD_dat[:, 0]]
            occx_shift = [(x+25) % 360 for x in occx]
            occx_2shift = [360 - x for x in occx_shift]
            # pot = np.column_stack((np.radians(hooc_2shift), np.radians(occx_2shift), ens_har))
            stack1 = np.column_stack((hooc_2shift, occx_2shift, ens_har))
            smol_args = np.argwhere(stack1[:, 1] < 180).flatten()
            onehalf = stack1[smol_args]
            big_args = np.argwhere(stack1[:, 1] > 180).flatten()
            twohalf = stack1[big_args]
            avg_energies = (onehalf[:, 2] + twohalf[:, 2]) / 2
            pothalf = np.column_stack((np.radians(onehalf[:, 0]), np.radians(onehalf[:, 1]), avg_energies))
            pot = np.vstack((pothalf, np.column_stack((pothalf[:, 0], pothalf[:, 1]+np.pi, pothalf[:, 2]))))
            sort_ind = np.lexsort((pot[:, 1], pot[:, 0]))
            # test = pot[sort_ind]
            # plt.tricontourf(test[:, 0], test[:, 1], test[:, 2], levels=15)
            # plt.colorbar()
            # plt.show()
            self._CCSDpot = pot[sort_ind]
        return self._CCSDpot

    @property
    def OHres(self):
        if self._OHres is None:
            resultspath = os.path.join(self.MoleculeObj.MoleculeDir,
                                       f"{self.MoleculeObj.MoleculeName}_OHDVR_results_02pi_30.npz")
            if os.path.exists(resultspath):
                print(f"Loading {resultspath}... ")
                self._OHres = np.load(resultspath)
            else:
                print(f"{resultspath} not found. beginning OH DVR")
                results = self.run_OHdvr()
                self._OHres = np.load(results)
        return self._OHres

    @property
    def ConstGRes(self):
        if self._ConstGRes is None:
            resDict = dict()
            for i, val in enumerate(['CCSD_avg', 'DFT']):
                resultspath = os.path.join(self.MoleculeDir, f"ConstG_2D_DVR_{val}.npz")
                if os.path.exists(resultspath):
                    print(f"Loading {resultspath}... ")
                    resDict[val] = np.load(resultspath, allow_pickle=True)
                else:
                    print(f"{resultspath} not found. Beginning CONSTANT G {val} 2D DVR... ")
                    res = self.TwoDTorTor_constG(resultspath, val)
                    resDict[val] = np.load(res, allow_pickle=True)
            self._ConstGRes = resDict
        return self._ConstGRes

    @property
    def TwoDdvrRes(self):
        if self._TwoDdvrRes is None:
            resDict = dict()
            for i, val in enumerate(['electronic', 'VOH_0', 'VOH_1', 'VOH_2']):
                resultspath = os.path.join(self.MoleculeDir, f"FullG_2D_DVR_{val}_symm.npz")
                # if os.path.exists(resultspath):
                #     print(f"Loading {resultspath}... ")
                #     resDict[val] = np.load(resultspath, allow_pickle=True)
                # else:
                print(f"{resultspath} not found. Beginning {val} 2D DVR... ")
                res = self.TwoDTorTor_FullG(resultspath, val)
                resDict[val] = np.load(res, allow_pickle=True)
            self._TwoDdvrRes = resDict
        return self._TwoDdvrRes

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
            ax.set_xlabel("COOH Torsion")
            ax.axes.set_xticks(np.arange(0, 390, 60))
            ax.set_ylabel("OCCH Torsion")
            ax.axes.set_yticks(np.arange(0, 240, 60))
            ax.set_zlabel("Electronic Energy")
            plt.title(title)

    def plot_PotCuts(self):
        import matplotlib.pyplot as plt
        pot = self.TwoDdvrRes["electronic"]["potential"]
        grid = self.TwoDdvrRes["electronic"]["grid"]
        grid_reshape = np.moveaxis(grid[0], -1, 0)
        vals = np.column_stack((grid_reshape[0].flatten(), grid_reshape[1].flatten(), pot.T))
        iter = np.unique(grid_reshape[0].flatten())
        for i in iter[::3]:
            inds = np.argwhere(vals[:, 0] == i)
            fig = plt.figure()
            plt.plot(vals[inds, 1], vals[inds, 2])
            plt.xlabel("CH2 Rotation (degrees)")
            plt.ylabel("Energy (cm^-1)")
            plt.title(f"HOOC = {i}")
            fig.savefig(os.path.join(self.MoleculeDir, "figures", "pot_cuts", f"pot_HOOC_{int(i)}.png"))
            plt.close()
        for i in iter[::3]:
            inds = np.argwhere(vals[:, 1] == i)
            fig = plt.figure()
            plt.plot(vals[inds, 0], vals[inds, 2])
            plt.xlabel("HOOC Torsion (degrees)")
            plt.ylabel("Energy (cm^-1)")
            plt.title(f"CH2 Rotation = {i}")
            fig.savefig(os.path.join(self.MoleculeDir, "figures", "pot_cuts", f"pot_OCCX_{int(i)}.png"))
            plt.close()

    def run_OHdvr(self):
        """Runs 1D OH DVR at every scan point,
           returns npz(epsilonPots=[HOOC, OCCX, En0, En1, En2, En3], wavefuns_array=[grid, # wfns])  """
        from DegreeDVR import run_2D_OHDVR, plot_wfns
        params = self.DVRparams
        potential_array, epsilon_pots, wavefuns_array = run_2D_OHDVR(self.MoleculeObj.OHScanDict,
                                                                     desiredEnergies=params["desired_energies"],
                                                                     NumPts=params["num_pts"],
                                                                     plotPhasedWfns=params["plot_phased_wfns"])
        npz_name = f"{self.MoleculeObj.MoleculeName}_OHDVR_results_02pi_30.npz"
        np.savez(os.path.join(self.MoleculeDir, npz_name), potential=potential_array,
                 epsilon=epsilon_pots, wavefunctions=wavefuns_array, data_pts=list(self.MoleculeObj.OHScanDict.keys()))
        # plot_wfns(wavefuns_array, potential_array, list(self.MoleculeObj.OHScanDict.keys()),
        #           os.path.join(self.MoleculeObj.MoleculeDir, "figures", "2DOHwfns"))
        return os.path.join(self.MoleculeObj.MoleculeDir, npz_name)

    def TwoDTorTor_constG(self, resultspath, val):
        """Runs 2D DVR using a Constant (HOOC=210, OCCX=150) value for the G-matrix,
        currently only on electronic potential"""
        from Converter import Constants
        from McUtils.Plots import ContourPlot
        dvr_2D = DVR("ColbertMillerND")
        print("Conducting 2D DVR with Constant G")
        gHdpHdp = self.GHdpHdp[0][26, 15]  # calls the calculate g-mat element at the equilibrium geom
        gXX = self.GXX[0][26, 15]
        if val == "CCSD_avg":
            pot = self.CCSDpot
        elif val == "DFT":
            pot = self.squareCoords
        else:
            raise Exception(f"Can not load {val} Potential")
        res = dvr_2D.run(potential_grid=pot, flavor="[0,2Pi]",
                         divs=(51, 51), mass=[1/gHdpHdp, 1/gXX], num_wfns=25,
                         domain=((min(pot[:, 0]),  max(pot[:, 0])),
                                 (min(pot[:, 1]), max(pot[:, 1]))),
                         results_class=ResultsInterpreter)
        # res.plot_potential(plot_class=ContourPlot, plot_units="wavenumbers", energy_threshold=2000, colorbar=True,
        #                    plot_style=dict(levels=15)).show()
        dvr_grid = np.degrees(res.grid)
        dvr_pot = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
        all_ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
        print(all_ens)
        if val == "CCSD_avg":
            save_tmp = os.path.join(self.MoleculeDir, "figures", "2D_CCSD_torwfns", f"{val}_torWfn_")
            ResultsInterpreter.wfn_contours(res, filename=save_tmp)
        else:
            save_tmp = os.path.join(self.MoleculeDir, "figures", "2D_ConstG_torwfns", f"{val}_torWfn_")
            ResultsInterpreter.wfn_contours(res, filename=save_tmp)
        np.savez(resultspath, grid=[dvr_grid], potential=[dvr_pot],
                 energy_array=all_ens, wfns_array=res.wavefunctions.wavefunctions)
        return resultspath

    def TwoDTorTor_diagG(self):
        """Runs 2D DVR using a Diagonal G-matrix (GH''H'' & GXX), currently only on electronic potential"""
        from Converter import Constants
        dvr_2D = DVR("ColbertMillerND")
        print("Conducting 2D DVR with Diagonal G")
        res = dvr_2D.run(potential_grid=self.squareCoords, flavor="[0,2Pi]",
                         divs=(51, 51), g=[[self.GHdpHdp[1], 0], [0, self.GXX[1]]],
                         g_deriv=[self.Gderiv_HdpHdp, self.Gderiv_XX], num_wfns=25,
                         domain=((min(self.squareCoords[:, 0]),  max(self.squareCoords[:, 0])),
                                 (min(self.squareCoords[:, 1]), max(self.squareCoords[:, 1]))),
                         results_class=ResultsInterpreter)
        all_ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
        print(all_ens)
        # ResultsInterpreter.wfn_contours(res)
        return res

    def TwoDTorTor_FullG(self, resultspath, val):
        """Runs 2D DVR using the Full 2x2 G-matrix array (GH''H'' & GXX & GXH''), val = str(VOH_X)
        property TwoDdvrRes loops through the various adiabats and saves all info into dictionary"""
        from Converter import Constants
        from McUtils.Plots import ContourPlot
        from scipy import interpolate
        params = {'text.usetex': False,
                  'mathtext.fontset': 'dejavusans',
                  'font.size': 10}
        plt.rcParams.update(params)
        dvr_2D = DVR("ColbertMillerND")
        print("Conducting 2D DVR with Full G")
        Epot = self.squareCoords
        if val == "electronic":
            potential = Epot[:, 2]
        else:
            adiabat_pots = self.OHres["epsilon"]  # [HOOC, OCCX, en0, etc] radians/hartrees
            x = np.unique(adiabat_pots[:, 0])
            y = np.unique(adiabat_pots[:, 1])
            i = int(val[-1])
            fit_pot = interpolate.interp2d(x, y, np.reshape(adiabat_pots[:, 2+i], (len(x), len(y))).T, kind="cubic")
            newx = np.unique(self.squareCoords[:, 0])
            newy = np.unique(self.squareCoords[:, 1])
            dense_adiabat = fit_pot(newx, newy).T.flatten()
            potential = Epot[:, 2] + dense_adiabat
        pot_vals = np.column_stack((self.squareCoords[:, 0], self.squareCoords[:, 1], potential))
        res = dvr_2D.run(potential_grid=pot_vals, flavor="[0,2Pi]",
                         divs=(51, 51), g=[[self.GHdpHdp[1], self.GXHdp[1]], [self.GXHdp[1], self.GXX[1]]],
                         g_deriv=[self.Gderiv_HdpHdp, self.Gderiv_XX], num_wfns=35,
                         domain=((min(pot_vals[:, 0]),  max(pot_vals[:, 0])),
                                 (min(pot_vals[:, 1]), max(pot_vals[:, 1]))),
                         results_class=ResultsInterpreter)
        dvr_grid = np.degrees(res.grid)
        grid_reshape = np.moveaxis(dvr_grid, -1, 0)
        dvr_pot = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
        # res.plot_potential(plot_class=ContourPlot, plot_units="wavenumbers", colorbar=True,
        #                    plot_style=dict(levels=15)).show()
        all_ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
        avg_en = np.average((all_ens[0], all_ens[1]))
        all_ens[0] = avg_en
        all_ens[1] = avg_en
        wfns = res.wavefunctions.wavefunctions
        sum_wfn = (wfns[:, 0] + wfns[:, 1]) / np.sqrt(2)
        diff_wfn = (wfns[:, 0] - wfns[:, 1]) / np.sqrt(2)
        wfns[:, 0] = sum_wfn
        fig = plt.figure(figsize=(6, 6), dpi=600)
        plt.tricontourf(grid_reshape[0].flatten(), grid_reshape[1].flatten(), sum_wfn)
        plt.xlabel("HOOC")
        plt.xticks(np.arange(0, 390, 60))
        plt.ylabel("CH2 Rotation")
        plt.yticks(np.arange(0, 390, 60))
        plt.title("0+ Wavefunction")
        filename = os.path.join(self.MoleculeDir, "figures", "2Dtorwfns", f"{val}_torWfn_sum")
        fig.savefig(f"{filename}.jpg", dpi=fig.dpi, bbox_inches="tight")
        wfns[:, 1] = diff_wfn
        plt.tricontourf(grid_reshape[0].flatten(), grid_reshape[1].flatten(), diff_wfn)
        plt.xlabel("HOOC")
        plt.xticks(np.arange(0, 390, 60))
        plt.ylabel("CH2 Rotation")
        plt.xticks(np.arange(0, 390, 60))
        plt.title("DIFF wfn")
        filename2 = os.path.join(self.MoleculeDir, "figures", "2Dtorwfns", f"{val}_torWfn_diff")
        fig.savefig(f"{filename2}.jpg", dpi=fig.dpi, bbox_inches="tight")
        print(val, all_ens)
        # print(min(dvr_pot))
        save_tmp = os.path.join(self.MoleculeDir, "figures", "2Dtorwfns", f"{val}_torWfn_")
        ResultsInterpreter.wfn_contours(res, filename=save_tmp)
        np.savez(resultspath, grid=[dvr_grid], potential=[dvr_pot],
                 energy_array=all_ens, wfns_array=wfns)
        return resultspath

    def calc_probs(self):
        for val in ['VOH_0', 'VOH_1', 'VOH_2', 'VOH_3']:
            eigvecs = self.TwoDdvrRes[val]["wfns_array"]
            pas = np.zeros((len(eigvecs[0]), 2))
            grid = self.TwoDdvrRes[val]["grid"][0]
            grid_for_real = np.moveaxis(grid, -1, 0)
            grrid = grid_for_real[0].flatten()
            for i in np.arange(len(eigvecs[0])):
                left_wfn = eigvecs[np.argwhere(grrid < 180), i]
                pas[i, 0] = (np.dot(left_wfn.T, left_wfn))*100
                right_wfn = eigvecs[np.argwhere(grrid > 180), i]
                pas[i, 1] = (np.dot(right_wfn.T, right_wfn))*100
            print(val, pas)

    def TwoD_transitions(self):
        from TransitionMoments2D import TransitionMoments2D
        tm_obj = TransitionMoments2D(self.OHres, self.TwoDdvrRes, self.MoleculeObj, transition=self.Transition)
        # test = tm_obj.plot_TDM(filename=os.path.join(self.MoleculeDir, "2DTDM"))
        if self.IntensityType == "FC":
            AllIntensities = tm_obj.calc_intensity(numGstates=2, numEstates=10, FC=True)
        elif self.IntensityType == "TDM":
            AllIntensities = tm_obj.calc_intensity(numGstates=6, numEstates=35, FC=False)
        else:
            raise Exception(f"Can not compute {self.IntensityType} intensities.")
        return AllIntensities
