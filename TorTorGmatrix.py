import numpy as np
from Converter import Constants


class TorTorGmatrix:
    def __init__(self, MoleculeObj, tor1key, tor2key):
        self.TORScanData = MoleculeObj.TORScanData
        self.massarray = MoleculeObj.mass_array
        self.tor1key = tor1key
        self.tor2key = tor2key
        self._matsize = None
        self._Gmatrix = None

    @property
    def matsize(self):
        if self._matsize is None:
            tor_1 = np.unique(self.TORScanData[self.tor1key])
            tor_2 = np.unique(self.TORScanData[self.tor2key])
            self._matsize = (tor_1, tor_2)
        return self._matsize

    def calc_Gaa(self, masses, ic):
        """uses g_tau,tau (1234, 1234) from Frederick and Woywod J.Chem.Phys., doi: 140.254.87.101 to
        calculate the g-matrix of an input mass array and internal coordinate dictionary
        mass_array = mass [ 1, 2, 3, 4]"""
        lambda_123 = (1 / np.sin(ic["phi123"])) * (1 / ic["r12"] - (np.cos(ic["phi123"]) / ic["r23"]))
        lambda_432 = (1 / np.sin(ic["phi234"])) * (1 / ic["r34"] - (np.cos(ic["phi234"]) / ic["r23"]))
        Gaa = (1 / (masses[0] * ic["r12]"] ** 2 * np.sin(ic["phi123"]) ** 2)) + \
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
        lambda_215 = (1 / np.sin(ic["phi234"])) * (1 / ic["r34"] - (np.cos(ic["phi234"]) / ic["r23"]))
        Gab = (1 / (masses[0] * ic["r12"] * np.sin(ic["phi123"]))) * (
                    lambda_215 * np.cos(ic["tau3215"]) - ((1 / ic["r15"]) * (1 / np.tan(ic["phi156"]))) * (
                                np.cos(ic["tau3215"]) * np.cos(ic["tau2156"]) +
                                np.cos(ic["phi215"]) * np.sin(ic["tau3215"]) * np.sin(ic["tau2156"]))) + \
              (1 / masses[1] * ic["r12"] * np.sin(ic["phi215"])) * (
                      lambda_123 * np.cos(ic["tau3215"]) - ((1 / ic["r23"]) * (1 / np.tan(ic["phi234"]))) * (
                                np.cos(ic["tau3215"]) * np.cos(ic["tau1234"]) +
                                np.cos(ic["phi123"]) * np.sin(ic["tau3215"]) * np.sin(ic["tau1234"])))
        return Gab

    def calc_Gbc(self, masses, ic):
        """uses g_tau,tau (1234, 1235) from Frederick and Woywod J.Chem.Phys., doi: 140.254.87.101"""
        lambda_123 = (1 / np.sin(ic["phi123"])) * (1 / ic["r12"] - (np.cos(ic["phi123"]) / ic["r23"]))
        lambda_432 = (1 / np.sin(ic["phi234"])) * (1 / ic["r34"] - (np.cos(ic["phi234"]) / ic["r23"]))
        lambda_532 = (1 / np.sin(ic["phi235"])) * (1 / ic["r35"] - (np.cos(ic["phi235"]) / ic["r23"]))
        Gbc = (1 / (masses[0] * ic["r12"] ** 2 * np.sin(ic["phi123"]) ** 2)) + \
              ((lambda_123 ** 2 / masses[0]) + ((1 / np.cos(ic["phi123"])) ** 2 / (masses[2] * ic["r23"] ** 3))) + \
              (1 / (np.tan(ic["phi234"]) * np.tan(ic["phi235"]) * masses[1] * ic["r23"] ** 2) +
               (lambda_432 * lambda_532 / masses[2])) * np.cos(ic["tau1235"] - ic["tau1234"]) - (1 / ic["r23"]) * \
              (((lambda_123 / masses[1]) * (1 / np.tan(ic["phi234"])) + (lambda_432 / masses[2]) * (
                      1 / np.tan(ic["phi123"]))) *
               np.cos(ic["tau1234"]) + (
                       (lambda_123 / masses[1]) * (1 / np.tan(ic["phi235"])) + (lambda_532 / masses[2]) *
                       (1 / np.tan(ic["phi123"]))) * np.cos(ic["tau1235"]))
        return Gbc

    def calc_GHdprimeHdprime(self):
        """ calculates the HOOC torsion with HOOC torsion g-matrix element with H-1, O-2, O-3, C-4"""
        Ghh = np.zeros(self.matsize)
        masses = [self.massarray[6], self.massarray[4], self.massarray[5], self.massarray[0]]
        for i in np.arange(len(self.matsize[0])):
            for j in np.arange(len(self.matsize[1])):
                lin_idx = i*self.matsize[1] + j
                ic = {'r12': self.TORScanData["B6"][lin_idx],
                      'r34': self.TORScanData["B5"][lin_idx],
                      'phi123': self.TORScanData["A5"][lin_idx],
                      'phi234': self.TORScanData["A4"][lin_idx],
                      'tau1234': self.TORScanData["D4"][lin_idx]}
                Ghh[i, j] = self.calc_Gaa(masses, ic)
        return Ghh

    def calc_GHprimeHprime(self):
        GHH = np.zeros(self.matsize)
        return GHH

