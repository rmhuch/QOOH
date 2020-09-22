import numpy as np

class POR:
    def __init__(self, DVR_npz, g_matrix, FEtype="cos", HamSize=None):
        self.DVRresults = np.load(DVR_npz)
        self.Gmatrix = g_matrix
        self.FEtype = FEtype
        self.HamSize = HamSize
        self._Vcoeffs = None
        self._Gcoeffs = None
        self._Mcoeffs = None
        self._Hamiltonian = None

    @property
    def Vcoeffs(self):
        if self._Vcoeffs is None:
            # call into FourierExpansions.py return array of potential coefficients for each energy level
            self._Vcoeffs = ...
        return self._Vcoeffs
    @property
    def Gcoeffs(self):
        if self._Gcoeffs is None:
            # call into FourierExpansions.py return array of potential coefficients for each energy level
            self._Gcoeffs = ...
        return self._Gcoeffs

    @property
    def Mcoeffs(self):
        if self._Mcoeffs is None:
            # figure out difference between M and G matrix coeffs....
            self._Mcoeffs = ...
        return self._Mcoeffs

    @property
    def Hamiltonian(self):
        if self._Hamiltonian is None:
            if self.FEtype is "cos":
                self._Hamiltonian = self.calc_cosHam()
            elif self.FEtype is "sin":
                self._Hamiltonian = self.calc_sinHam()
        return self._Hamiltonian

    def solveHam(self):
        energy, eigvecs = np.linalg.eigh(self.Hamiltonian)
        spacings = np.array((energy[1] - energy[0], energy[2] - energy[0]))
        enwfn = {'V': self.Vcoeffs,
                 'energy': energy,
                 'eigvecs': eigvecs,
                 'spacings': spacings}
        return enwfn

    def calc_cosHam(self):
        """This should be fed the appropriately scaled Vcoefs, just calculates the Hamiltonian
        based of what it is given"""
        if self.HamSize is None:
            HamSize = 7  # make more flexible to different order sizes in expansion - currently only supports 6
        else:
            HamSize = self.HamSize
        fullsize = 2 * HamSize + 1
        Ham = np.zeros((fullsize, fullsize))
        for l in np.arange(fullsize):
            k = l - HamSize
            Ham[l, l] = self.Gcoeffs[0] * (k ** 2) + self.Vcoeffs[0]  # m=0
            for kprime in np.arange(k + 1, k - 7, -1):
                m = k - kprime
                if m > 0 and l - m >= 0:
                    Ham[l, l - m] = (kprime ** 2 + k ** 2 - m ** 2) * (self.Gcoeffs[m] / 4) + (self.Vcoeffs[m] / 2)
                    Ham[l - m, l] = Ham[l, l - m]
                else:
                    pass
        return Ham

    def calc_sinHam(self, M_coeffs):
        if self.HamSize is None:
            HamSize = 7  # make more flexible to different order sizes in expansion - currently only supports 6
        else:
            HamSize = self.HamSize
        fullsize = 2 * HamSize + 1
        Ham = np.zeros((fullsize, fullsize))
        for l in np.arange(fullsize):
            k = l - HamSize
            for kprime in np.arange(k + 1, k - 7, -1):
                m = k - kprime
                if m > 0 and l - m >= 0:
                    Ham[l, l - m] = M_coeffs[m] / 2
                    Ham[l - m, l] = -Ham[l, l - m]
                else:
                    pass
        return Ham

    def PORwfns(self):
        HamRes = self.solveHam()
        theta = np.linspace(0, 2*np.pi, 100)
        eigenvectors = HamRes["eigvecs"]
        ks = np.arange(len(eigenvectors) // 2 * -1, len(eigenvectors) // 2 + 1)
        vals = np.zeros((len(theta), len(eigenvectors)))
        for n in np.arange(len(eigenvectors)):
            c_ks = eigenvectors[:, n]
            re = np.zeros((len(theta), len(eigenvectors)))
            im = np.zeros((len(theta), len(eigenvectors)))
            for i, k in enumerate(ks):
                im[:, i] = c_ks[i] * (1 / np.sqrt(2 * np.pi)) * np.sin(k * theta)
                re[:, i] = c_ks[i] * (1 / np.sqrt(2 * np.pi)) * np.cos(k * theta)
            vals[:, n] = np.sum(re, axis=1) + np.sum(im, axis=1)
        return vals