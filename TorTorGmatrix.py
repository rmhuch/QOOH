import numpy as np
from scipy import interpolate
from Converter import Constants
import matplotlib.pyplot as plt


class TorTorGmatrix:
    def __init__(self, MoleculeObj):
        self.GmatCoords = MoleculeObj.GmatCoords
        self.massarray = MoleculeObj.mass_array
        self.tor1key = np.unique([k[0] for k in self.GmatCoords.keys()])
        self.tor1rad = np.radians(self.tor1key)
        self.tor2key = np.unique([k[1] for k in self.GmatCoords.keys()])
        self.tor2rad = np.radians(self.tor2key)
        self._GHdpHdp = None
        self._GXX = None
        self._GXHdp = None

    @property
    def GHdpHdp(self):
        """ returns the matrix GH''H'' [0] AND the interp2d function for GH''H'' [1] as a tuple"""
        if self._GHdpHdp is None:
            self._GHdpHdp = self.calc_GHdpHdp()
        return self._GHdpHdp

    @property
    def GXX(self):
        """ returns the matrix GXX [0] AND the interp2d function for GXX [1] as a tuple"""
        if self._GXX is None:
            self._GXX = self.calc_GXX()
        return self._GXX

    @property
    def GXHdp(self):
        """ returns the matrix GXH'' [0] AND the interp2d function for GXH'' [1] as a tuple"""
        if self._GXHdp is None:
            self._GXHdp = self.calc_GXHdp()
        return self._GXHdp

    def calc_Gaa(self, masses, ic):
        """uses g_tau,tau (1234, 1234) from Frederick and Woywod J.Chem.Phys., doi: 140.254.87.101 to
        calculate the g-matrix of an input mass array and internal coordinate dictionary
        mass_array = mass [ 1, 2, 3, 4]"""
        lambda_123 = (1 / np.sin(ic["phi123"])) * (1 / ic["r12"] - (np.cos(ic["phi123"]) / ic["r23"]))
        lambda_432 = (1 / np.sin(ic["phi234"])) * (1 / ic["r34"] - (np.cos(ic["phi234"]) / ic["r23"]))
        Gaa = (1 / (masses[0] * ic["r12"] ** 2 * np.sin(ic["phi123"]) ** 2)) + \
              1 / (masses[3] * ic["r34"] ** 2 * np.sin(ic["phi234"]) ** 2) + \
              1 / masses[1] * (lambda_123 ** 2 + (1 / np.tan(ic["phi234"]) ** 2) / ic["r23"] ** 2) + \
              1 / masses[2] * (lambda_432 ** 2 + (1 / np.tan(ic["phi123"]) ** 2) / ic["r23"] ** 2) - \
              2 * (np.cos(ic["tau1234"]) / ic["r23"]) * ((lambda_123 / masses[1]) * 1 / np.tan(ic["phi234"]) +
                                                         (lambda_432 / masses[2]) * 1 / np.tan(ic["phi123"]))
        return Gaa

    def calc_Gab(self, masses, ic):
        """uses g_tau,tau (1234, 2156) from Frederick and Woywod J.Chem.Phys., doi: 140.254.87.101
        mass_array = mass [1, 2]"""
        lambda_123 = (1 / np.sin(ic["phi123"])) * (1 / ic["r12"] - (np.cos(ic["phi123"]) / ic["r23"]))
        lambda_215 = (1 / np.sin(ic["phi215"])) * (1 / ic["r12"] - (np.cos(ic["phi215"]) / ic["r15"]))
        term1 = (1 / (masses[0] * ic["r12"] * np.sin(ic["phi123"]))) * (
                    lambda_215 * np.cos(ic["tau3215"]) - ((1 / ic["r15"]) * (1 / np.tan(ic["phi156"]))) * (
                                np.cos(ic["tau3215"]) * np.cos(ic["tau2156"]) +
                                np.cos(ic["phi215"]) * np.sin(ic["tau3215"]) * np.sin(ic["tau2156"])))
        term2 = (1 / (masses[1] * ic["r12"] * np.sin(ic["phi215"]))) * (
                      lambda_123 * np.cos(ic["tau3215"]) - ((1 / ic["r23"]) * (1 / np.tan(ic["phi234"]))) * (
                                np.cos(ic["tau3215"]) * np.cos(ic["tau1234"]) +
                                np.cos(ic["phi123"]) * np.sin(ic["tau3215"]) * np.sin(ic["tau1234"])))
        Gab = term1 + term2
        return Gab

    def calc_Gbc(self, masses, ic):
        """uses g_tau,tau (1234, 1235) from Frederick and Woywod J.Chem.Phys., doi: 140.254.87.101
           mass_array = mass [1, 2, 3, 4, 5]"""
        lambda_123 = (1 / np.sin(ic["phi123"])) * (1 / ic["r12"] - (np.cos(ic["phi123"]) / ic["r23"]))
        lambda_432 = (1 / np.sin(ic["phi234"])) * (1 / ic["r34"] - (np.cos(ic["phi234"]) / ic["r23"]))
        lambda_532 = (1 / np.sin(ic["phi235"])) * (1 / ic["r35"] - (np.cos(ic["phi235"]) / ic["r23"]))
        Gbc = (1 / (masses[0] * ic["r12"] ** 2 * np.sin(ic["phi123"]) ** 2)) + \
              ((lambda_123 ** 2 / masses[1]) + ((1 / np.tan(ic["phi123"])) ** 2 / (masses[2] * ic["r23"] ** 2))) + \
              (1 / (np.tan(ic["phi234"]) * np.tan(ic["phi235"]) * masses[1] * ic["r23"] ** 2) +
               (lambda_432 * lambda_532 / masses[2])) * np.cos(ic["tau1235"] - ic["tau1234"]) - (1 / ic["r23"]) * \
              (((lambda_123 / masses[1]) * (1 / np.tan(ic["phi234"])) + (lambda_432 / masses[2]) * (
                      1 / np.tan(ic["phi123"]))) *
               np.cos(ic["tau1234"]) + (
                       (lambda_123 / masses[1]) * (1 / np.tan(ic["phi235"])) + (lambda_532 / masses[2]) *
                       (1 / np.tan(ic["phi123"]))) * np.cos(ic["tau1235"]))
        return Gbc

    def calc_GHdpHdp(self):
        """ calculates the HOOC torsion with HOOC torsion g-matrix element with H''-1, O-2, O-3, C-4 and returns
         the 2d interpolation of the g-matrix"""
        Ghh = np.zeros((len(self.tor1key), len(self.tor2key)))
        masses = [self.massarray[6], self.massarray[5], self.massarray[4], self.massarray[0]]
        for k, v in self.GmatCoords.items():
            HOOC = k[0]
            OCCX = k[1]
            tor1_ind = np.where(self.tor1key == HOOC)[0][0]
            tor2_ind = np.where(self.tor2key == OCCX)[0][0]
            ic = {'r12': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B6"], "angstroms", to_AU=True),
                  'r23': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B5"], "angstroms", to_AU=True),
                  'r34': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B4"], "angstroms", to_AU=True),
                  'phi123': np.radians(self.GmatCoords[(HOOC, OCCX)]["A5"]),
                  'phi234': np.radians(self.GmatCoords[(HOOC, OCCX)]["A4"]),
                  'tau1234': np.radians(self.GmatCoords[(HOOC, OCCX)]["D4"])}
            Ghh[tor1_ind, tor2_ind] = self.calc_Gaa(masses, ic)
        Ghh_func = interpolate.interp2d(self.tor1rad, self.tor2rad, Ghh.T, kind="cubic")
        # Transpose G because interp2d expects (y,x) matrix
        # plt.tricontourf(np.array(list(self.GmatCoords.keys()))[:, 0], np.array(list(self.GmatCoords.keys()))[:, 1],
        #                 Ghh.flatten()/(9.1093837015e-28*6.02214076e23)*1.88973**2)
        # plt.title("GH''H''")
        # cbar = plt.colorbar()
        # cbar.set_label(r"1/amu$\AA^2$")
        # plt.show()

        def caller(coords, **deriv_kwargs):
            unx = np.unique(coords[:, 0])
            uny = np.unique(coords[:, 1])
            return Ghh_func(unx, uny, **deriv_kwargs).T.flatten()
        return Ghh, caller

    def calc_GHpHp(self):
        """ calculates the OCC'H' torsion with OCC'H' torsion g-matrix element with H'-1, C'-2, C-3, O-4 and returns
         the 2d interpolation of the g-matrix"""
        GHH = np.zeros((len(self.tor1key), len(self.tor2key)))
        masses = [self.massarray[2], self.massarray[1], self.massarray[0], self.massarray[4]]
        for k, v in self.GmatCoords.items():
            HOOC = k[0]
            OCCX = k[1]
            tor1_ind = np.where(self.tor1key == HOOC)[0][0]
            tor2_ind = np.where(self.tor2key == OCCX)[0][0]
            ic = {'r12': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B2"], "angstroms", to_AU=True),
                  'r23': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B1"], "angstroms", to_AU=True),
                  'r34': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B4"], "angstroms", to_AU=True),
                  'phi123': np.radians(self.GmatCoords[(HOOC, OCCX)]["ACCpHp"]),
                  'phi234': np.radians(self.GmatCoords[(HOOC, OCCX)]["A3"]),
                  'tau1234': np.radians(self.GmatCoords[(HOOC, OCCX)]["DOCCpHp"])}
            GHH[tor1_ind, tor2_ind] = self.calc_Gaa(masses, ic)
        return GHH

    def calc_GHH(self):
        """ calculates the OCC'H torsion with OCC'H torsion g-matrix element with H-1, C'-2, C-3, O-4 and returns
         the 2d interpolation of the g-matrix"""
        GHH = np.zeros((len(self.tor1key), len(self.tor2key)))
        masses = [self.massarray[3], self.massarray[1], self.massarray[0], self.massarray[4]]
        for k, v in self.GmatCoords.items():
            HOOC = k[0]
            OCCX = k[1]
            tor1_ind = np.where(self.tor1key == HOOC)[0][0]
            tor2_ind = np.where(self.tor2key == OCCX)[0][0]
            if "ACCpH" not in self.GmatCoords[(HOOC, OCCX)]:
                raise Exception(f"No ACCpH in {(HOOC, OCCX)}")
            ic = {'r12': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B3"], "angstroms", to_AU=True),
                  'r23': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B1"], "angstroms", to_AU=True),
                  'r34': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B4"], "angstroms", to_AU=True),
                  'phi123': np.radians(self.GmatCoords[(HOOC, OCCX)]["ACCpH"]),
                  'phi234': np.radians(self.GmatCoords[(HOOC, OCCX)]["A3"]),
                  'tau1234': np.radians(self.GmatCoords[(HOOC, OCCX)]["DOCCpH"])}
            GHH[tor1_ind, tor2_ind] = self.calc_Gaa(masses, ic)
        return GHH

    def calc_GHpH(self):
        """ calculates the OCC'H' torsion with OCC'H torsion g-matrix element with O-1, C-2, C'-3, H'-4, H-5 and returns
         the 2d interpolation of the g-matrix"""
        GHH = np.zeros((len(self.tor1key), len(self.tor2key)))
        masses = [self.massarray[4], self.massarray[0], self.massarray[1], self.massarray[2], self.massarray[3]]
        for k, v in self.GmatCoords.items():
            HOOC = k[0]
            OCCX = k[1]
            tor1_ind = np.where(self.tor1key == HOOC)[0][0]
            tor2_ind = np.where(self.tor2key == OCCX)[0][0]
            ic = {'r12': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B4"], "angstroms", to_AU=True),
                  'r23': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B1"], "angstroms", to_AU=True),
                  'r34': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B2"], "angstroms", to_AU=True),
                  'r35': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B3"], "angstroms", to_AU=True),
                  'phi123': np.radians(self.GmatCoords[(HOOC, OCCX)]["A3"]),
                  'phi234': np.radians(self.GmatCoords[(HOOC, OCCX)]["ACCpHp"]),
                  'phi235': np.radians(self.GmatCoords[(HOOC, OCCX)]["ACCpH"]),
                  'tau1234': np.radians(self.GmatCoords[(HOOC, OCCX)]["DOCCpHp"]),
                  'tau1235': np.radians(self.GmatCoords[(HOOC, OCCX)]["DOCCpH"])}
            GHH[tor1_ind, tor2_ind] = self.calc_Gbc(masses, ic)
        return GHH

    def calc_GXX(self):
        """calculate the G-matrix of the OCCX torsion"""
        GXX = (1/4) * (self.calc_GHH() + self.calc_GHpHp() + self.calc_GHpH())
        GXX_func = interpolate.interp2d(self.tor1rad, self.tor2rad, GXX.T, kind="cubic")
        # Transpose G because interp2d expects (y,x) matrix
        # plt.tricontourf(np.array(list(self.GmatCoords.keys()))[:, 0], np.array(list(self.GmatCoords.keys()))[:, 1],
        #                 GXX.flatten()/(9.1093837015e-28*6.02214076e23)*1.88973**2)
        # plt.title("GXX")
        # cbar = plt.colorbar()
        # cbar.set_label(r"1/amu$\AA^2$")
        # plt.show()

        def caller(coords, **deriv_kwargs):
            unx = np.unique(coords[:, 0])
            uny = np.unique(coords[:, 1])
            return GXX_func(unx, uny, **deriv_kwargs).T.flatten()
        return GXX, caller

    def calc_GHHdp(self):
        GHH = np.zeros((len(self.tor1key), len(self.tor2key)))
        masses = [self.massarray[0], self.massarray[4]]  # only pass masses of atoms 1 & 2
        for k, v in self.GmatCoords.items():
            HOOC = k[0]
            OCCX = k[1]
            tor1_ind = np.where(self.tor1key == HOOC)[0][0]
            tor2_ind = np.where(self.tor2key == OCCX)[0][0]
            ic = {'r12': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B4"], "angstroms", to_AU=True),
                  'r23': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B5"], "angstroms", to_AU=True),
                  'r15': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B1"], "angstroms", to_AU=True),
                  'phi123': np.radians(self.GmatCoords[(HOOC, OCCX)]["A4"]),
                  'phi215': np.radians(self.GmatCoords[(HOOC, OCCX)]["A3"]),
                  'phi156': np.radians(self.GmatCoords[(HOOC, OCCX)]["ACCpH"]),
                  'phi234': np.radians(self.GmatCoords[(HOOC, OCCX)]["A5"]),
                  'tau1234': np.radians(self.GmatCoords[(HOOC, OCCX)]["D4"]),
                  'tau2156': np.radians(self.GmatCoords[(HOOC, OCCX)]["DOCCpH"]),
                  'tau3215': np.radians(self.GmatCoords[(HOOC, OCCX)]["D3"])}
            GHH[tor1_ind, tor2_ind] = self.calc_Gab(masses, ic)
        return GHH

    def calc_GHpHdp(self):
        GHH = np.zeros((len(self.tor1key), len(self.tor2key)))
        masses = [self.massarray[0], self.massarray[4]]  # only pass masses of atoms 1 & 2
        for k, v in self.GmatCoords.items():
            HOOC = k[0]
            OCCX = k[1]
            tor1_ind = np.where(self.tor1key == HOOC)[0][0]
            tor2_ind = np.where(self.tor2key == OCCX)[0][0]
            ic = {'r12': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B4"], "angstroms", to_AU=True),
                  'r23': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B5"], "angstroms", to_AU=True),
                  'r15': Constants.convert(self.GmatCoords[(HOOC, OCCX)]["B1"], "angstroms", to_AU=True),
                  'phi123': np.radians(self.GmatCoords[(HOOC, OCCX)]["A4"]),
                  'phi215': np.radians(self.GmatCoords[(HOOC, OCCX)]["A3"]),
                  'phi156': np.radians(self.GmatCoords[(HOOC, OCCX)]["ACCpHp"]),
                  'phi234': np.radians(self.GmatCoords[(HOOC, OCCX)]["A5"]),
                  'tau1234': np.radians(self.GmatCoords[(HOOC, OCCX)]["D4"]),
                  'tau2156': np.radians(self.GmatCoords[(HOOC, OCCX)]["DOCCpHp"]),
                  'tau3215': np.radians(self.GmatCoords[(HOOC, OCCX)]["D3"])}
            GHH[tor1_ind, tor2_ind] = self.calc_Gab(masses, ic)
        return GHH

    def calc_GXHdp(self):
        """calculate the cross term for the G-matrix (OCCX with HOOC) (off diagonal)"""
        GXHdp = (1/2) * (self.calc_GHHdp() + self.calc_GHpHdp())
        GXHdp_func = interpolate.interp2d(self.tor1rad, self.tor2rad, GXHdp.T, kind="cubic")
        # Transpose G because interp2d expects (y,x) matrix
        # plt.tricontourf(np.array(list(self.GmatCoords.keys()))[:, 0], np.array(list(self.GmatCoords.keys()))[:, 1],
        #                 GXHdp.flatten()/(9.1093837015e-28*6.02214076e23)*1.88973**2)
        # plt.title("GXH''")
        # cbar = plt.colorbar()
        # cbar.set_label(r"1/amu$\AA^2$")
        # plt.show()

        def caller(coords, **deriv_kwargs):
            unx = np.unique(coords[:, 0])
            uny = np.unique(coords[:, 1])
            return GXHdp_func(unx, uny, **deriv_kwargs).T.flatten()
        return GXHdp, caller
