import numpy as np

class POR:
    def __init__(self, DVR_res, g_matrix, Velcoeffs, HamSize=None, barrier_height=None):
        self.DVRresults = DVR_res
        self.Gmatrix = g_matrix
        self.Velcoeffs = Velcoeffs
        self.HamSize = HamSize
        self.barrier_height = barrier_height  # will only take one barrier height at a time.
        self._Bcoeffs = None
        self._Vcoeffs = None
        self._scaledVcoeffs = None
        self._Mcoeffs = None
        self._DipoleMomentMatrix = None

    @property
    def Bcoeffs(self):
        if self._Bcoeffs is None:
            # Kinetic Energy Coefficients
            self._Bcoeffs = self.calc_Bcoeffs()
        return self._Bcoeffs

    @property
    def Vcoeffs(self):
        if self._scaledVcoeffs is None:
            # Potential Energy Coefficients
            self._Vcoeffs = self.calc_Vcoeffs()
        return self._Vcoeffs

    @property
    def scaledVcoeffs(self):
        if self._Vcoeffs is None:
            # SCALED Potential Energy Coefficients
            if self.barrier_height is None:
                raise Exception("Can not scale Potential without self.barrier_height assigned")
            else:
                self._scaledVcoeffs = self.calc_scaled_Vcoeffs()
        return self._scaledVcoeffs

    @property
    def Mcoeffs(self):
        if self._Mcoeffs is None:
            # Dipole Moment Coefficients
            self._Mcoeffs = ...
        return self._Mcoeffs

    @property
    def DipoleMomentMatrix(self):
        if self._DipoleMomentMatrix is None:
            self._DipoleMomentMatrix = self.calc_DipoleMomentMatrix()
        return self._DipoleMomentMatrix

    def calc_Bcoeffs(self):
        from FourierExpansions import calc_cos_coefs
        coeff_dict = dict()  # build coeff dict
        for i in np.arange(self.Gmatrix.shape[0]):
            gmatrix_coeffs = calc_cos_coefs(self.Gmatrix[i]) / 2
            coeff_dict[f"B{i}"] = gmatrix_coeffs
        return coeff_dict

    def calc_Vcoeffs(self):
        from Converter import Constants
        from FourierExpansions import calc_cos_coefs
        dvr_energies = self.DVRresults["energies"]  # [degrees vOH=0 ... vOH=6]
        dvr_energies[:, 1:] = Constants.convert(dvr_energies[:, 1:], "wavenumbers", to_AU=True)
        coeff_dict = dict()  # build coeff dict
        rad = np.radians(dvr_energies[:, 0])
        coeff_dict["Vel"] = self.Velcoeffs
        for i in np.arange(1, dvr_energies.shape[1]):  # loop through saved energies
            energies = np.column_stack((rad, dvr_energies[:, i]))
            coeff_dict[f"V{i - 1}"] = calc_cos_coefs(energies)
        return coeff_dict

    def calc_scaled_Vcoeffs(self):
        from FourierExpansions import calc_curves
        from scaleTORpotentials import calc_scaled_Vcoeffs
        energy = calc_curves(np.radians(np.arange(0, 360, 10)), self.Velcoeffs)
        energy_dat = np.column_stack((np.arange(0, 360, 10), energy))
        ens_to_calc = self.DVRresults["energies"].shape[1] - 2  # subtract 2 for degree and E0 columns
        scaled_coeffs = calc_scaled_Vcoeffs(energy_dat, self.Vcoeffs, self.barrier_height, ens_to_calc)
        return scaled_coeffs

    def calc_Hamiltonian(self, pot_coeffs, kin_coeffs):
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
            Ham[l, l] = kin_coeffs[0] * (k ** 2) + pot_coeffs[0]  # m=0
            for kprime in np.arange(k + 1, k - 7, -1):
                m = k - kprime
                if m > 0 and l - m >= 0:
                    Ham[l, l - m] = (kprime ** 2 + k ** 2 - m ** 2) * (kin_coeffs[m] / 4) + (pot_coeffs[m] / 2)
                    Ham[l - m, l] = Ham[l, l - m]
                else:
                    pass
        return Ham

    def calc_DipoleMomentMatrix(self, M_coeffs):
        if self.HamSize is None:
            MatSize = 7  # make more flexible to different order sizes in expansion - currently only supports 6
        else:
            MatSize = self.HamSize
        fullsize = 2 * MatSize + 1
        Mat = np.zeros((fullsize, fullsize))
        for l in np.arange(fullsize):
            k = l - MatSize
            for kprime in np.arange(k + 1, k - 7, -1):
                m = k - kprime
                if m > 0 and l - m >= 0:
                    Mat[l, l - m] = M_coeffs[m] / 2
                    Mat[l - m, l] = -Mat[l, l - m]
                else:
                    pass
        return Mat

    def solveHam(self):
        from Converter import Constants
        results = []
        for i in np.arange(len(self.Vcoeffs.keys()) - 1):  # loop through saved energies
            if self.barrier_height is None:
                pot_coeffs = self.Vcoeffs["Vel"] + self.Vcoeffs[f"V{i}"]
            else:
                pot_coeffs = self.scaledVcoeffs["Vel"] + self.scaledVcoeffs[f"V{i}"]
            kin_coeffs = self.Bcoeffs[f"B{i}"]
            Hamiltonian = self.calc_Hamiltonian(pot_coeffs, kin_coeffs)
            energy, eigvecs = np.linalg.eigh(Hamiltonian)
            print(f"vOH = {i} : E0+ - {Constants.convert(energy[0], 'wavenumbers', to_AU=False)}"
                  f"            E0- - {Constants.convert(energy[1], 'wavenumbers', to_AU=False)}"
                  f"            E1- - {Constants.convert(energy[2], 'wavenumbers', to_AU=False)}"
                  f"            E1+ - {Constants.convert(energy[3], 'wavenumbers', to_AU=False)}")
            spacings = np.array((energy[1] - energy[0], energy[2] - energy[0]))
            results.append({'V': pot_coeffs,
                            'energy': energy,
                            'eigvecs': eigvecs,
                            'spacings': spacings})
        return results

    def PORwfns(self):
        HamRes = self.solveHam()
        theta = np.linspace(0, 2*np.pi, 100)
        all_vals = []
        for i in np.arange(len(HamRes)):
            eigenvectors = HamRes[i]["eigvecs"]
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
            all_vals.append(vals)
        return all_vals
