import numpy as np
import os


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
        self._OHScanData = None
        self._TORScanData = None
        self._TorScanDict = None
        self._cartesians = None
        self._GmatCoords = None
        self._Gmatrix = None
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
    def OHScanData(self):
        if self._OHScanData is None:
            self._OHScanData = self.get_Data(npz_filename=self.oh_scan_npz)
        return self._OHScanData

    @property
    def TORScanData(self):
        if self._TORScanData is None:
            self._TORScanData = self.get_Data(npz_filename=self.tor_scan_npz)
        return self._TORScanData

    @property
    def TorScanDict(self):
        if self._TorScanDict is None:
            self._TorScanDict = self.get_GroupedDict()
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
    def Gmatrix(self):
        if self._Gmatrix is None:
            self._Gmatrix = self.get_Gmatrix()
        return self._Gmatrix

    @property
    def DipoleMomentSurface(self):
        if self._DipoleMomentSurface is None:
            self._DipoleMomentSurface = np.load(os.path.join(self.MoleculeDir, self.dipole_npz))
            # pulls in data as a dictionary that is keyed by angles, with values: (roh, x, y, z)
        return self._DipoleMomentSurface

    def get_Data(self, npz_filename=None):
        if npz_filename is None:
            raise Exception("no data to read")
        else:
            vals = np.load(os.path.join(self.MoleculeDir, npz_filename), allow_pickle=True)
        # this loads in a dictionary of internals and energies for every point in the COOH, CH2, OH scan
        return vals

    def get_GroupedDict(self):
        """ Takes Internal Dict and converts it into a grouped nested, dict {(HOOC, OCCX} : {'coord_name' : val}
         with val in degree/angstrom"""
        from Grouper import Grouper
        ints = self.TORScanData
        all_names = ints.files
        int_coord_array = np.column_stack((list(ints.values())))
        group = Grouper(int_coord_array, (30, 28))  # groups into (HOOC, OCCX)
        Gdict = {k: dict(zip(all_names, v[0])) for k, v in group.group_dict.items()}
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
        OCCH = np.round(OCCH)
        OCCHp = np.round(OCCHp)
        # OCCHP = OCCHp + 180
        OCCX = ((OCCH + OCCHp)/2) + 90
        round_OCCX = np.round(OCCX/10)*10
        HOOC = [x + 360 if x < 0 else x for x in HOOC]
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
            internals[k]["ACCpH"] = np.degrees(vec_angles(coord_array[0]-coord_array[1], coord_array[3]-coord_array[1])[0])
            internals[k]["ACCpHp"] = np.degrees(vec_angles(coord_array[0]-coord_array[1], coord_array[4]-coord_array[1])[0])
            internals[k]["DOCCpH"] = np.degrees(pts_dihedrals(coord_array[4], coord_array[0], coord_array[1], coord_array[3]))
            internals[k]["DOCCpHp"] = np.degrees(pts_dihedrals(coord_array[4], coord_array[0], coord_array[1], coord_array[2]))
        return internals

    def get_Gmatrix(self):
        from TorTorGmatrix import TorTorGmatrix
        GmatObj = TorTorGmatrix(self)
        Gmat = GmatObj.diagGmatrix
        return Gmat

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
            plt.show()
