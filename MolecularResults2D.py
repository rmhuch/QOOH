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
    def cartesians(self):
        if self._cartesians is None:
            self._cartesians = self.get_Data(npz_filename=self.cart_npz)
        return self._cartesians

    @property
    def GmatCoords(self):
        if self._GmatCoords is None:
            self._GmatCoords = self.get_GmatCoords()
        return self._GmatCoords

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

    def get_GmatCoords(self):
        """ Calculates extra internals and combines with other internal dictionary to return one groupedDict keyed by
         tuple (COOH, OCCH) """
        from Grouper import Grouper
        from McUtils.Numputils import vec_angles
        internals = self.TORScanData
        coord_array = np.column_stack((list(internals.values())))
        coord_names = list(internals.keys())
        # calculate extra internals
        carts = self.cartesians
        ang_CCpH = []
        ang_CCpHp = []
        for file in carts.keys():
            coord_array = carts[file]
            ang_CCpH.extend(vec_angles(coord_array[:, 0]-coord_array[:, 1], coord_array[:, 3]-coord_array[:, 1]))
            ang_CCpHp.extend(vec_angles(coord_array[:, 0]-coord_array[:, 1], coord_array[:, 4]-coord_array[:, 1]))
        coord_array = np.column_stack((coord_array, ang_CCpH, ang_CCpHp))
        coord_names.append("ACCpH", "ACCpHp")
        groupedDict = Grouper  # PICK UP HERE
        groupedDict["coord_names"] = coord_names
        return groupedDict

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
