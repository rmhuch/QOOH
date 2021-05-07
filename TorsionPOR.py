import numpy as np

class POR:
    def __init__(self, DVR_res, fit_gmatrix, Velcoeffs, params):
        self.DVRresults = DVR_res
        self.FitGmatrix = fit_gmatrix
        self.Velcoeffs = Velcoeffs
        self.PORparams = params
        if "HamSize" in params:
            self.HamSize = params["HamSize"]
        if "barrier_height" in params:
            self.barrier_height = params["barrier_height"]  # will only take one barrier height at a time.
        else:
            self.barrier_height = None
        if "scaling_factor" in params:
            self.scaling_factor = params["scaling_factor"]
        else:
            self.scaling_factor = None
        self._Bcoeffs = None
        self._Vcoeffs = None
        self._scaledVcoeffs = None

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
            if "MixedData1" in self.PORparams:
                self._Vcoeffs = self.calc_Mixed1_Vcoeffs()
            if "MixedData2" in self.PORparams:
                self._Vcoeffs = self.calc_Mixed2_Vcoeffs()
            elif "EmilData" in self.PORparams:
                self._Vcoeffs = self.calc_Emil_Vcoeffs()
            else:
                self._Vcoeffs = self.calc_Vcoeffs()
        return self._Vcoeffs

    @property
    def scaledVcoeffs(self):
        if self._scaledVcoeffs is None:
            # SCALED Potential Energy Coefficients
            if self.scaling_factor is None and self.barrier_height is None:
                raise Exception("Can not scale Potential without self.barrier_height or self.scaling_factor assigned")
            else:
                self._scaledVcoeffs = self.calc_scaled_Vcoeffs()
        return self._scaledVcoeffs

    def calc_Bcoeffs(self):
        from FourierExpansions import calc_cos_coefs
        coeff_dict = dict()  # build coeff dict
        for i in self.FitGmatrix.keys():
            gmatrix_coeffs = self.FitGmatrix[i] / 2
            coeff_dict[f"B{i}"] = gmatrix_coeffs
        return coeff_dict

    def calc_Vcoeffs(self):
        from Converter import Constants
        from FourierExpansions import calc_cos_coefs, calc_4cos_coefs
        print("Using DFT for all potentials...")
        dvr_energies = self.DVRresults["energies"]  # [degrees vOH=0 ... vOH=6]
        dvr_energies[:, 1:] = Constants.convert(dvr_energies[:, 1:], "wavenumbers", to_AU=True)
        coeff_dict = dict()  # build coeff dict
        rad = np.radians(dvr_energies[:, 0])
        coeff_dict["Vel"] = self.Velcoeffs
        if "Vexpansion" in self.PORparams:
            if self.PORparams["Vexpansion"] == "fourth":
                print("Expaning potential coefficients to four tau")
                for i in np.arange(1, dvr_energies.shape[1]):  # loop through saved energies
                    energies = np.column_stack((rad, dvr_energies[:, i]))
                    coeff_dict[f"V{i - 1}"] = calc_4cos_coefs(energies)
            elif self.PORparams["Vexpansion"] == "sixth":
                print("Expaning potential coefficients to six tau")
                for i in np.arange(1, dvr_energies.shape[1]):  # loop through saved energies
                    energies = np.column_stack((rad, dvr_energies[:, i]))
                    coeff_dict[f"V{i - 1}"] = calc_cos_coefs(energies)
        else:
            raise Exception(f"Can not expand to {self.PORparams['Vexpansion']}")
        return coeff_dict

    def calc_Mixed1_Vcoeffs(self):
        from Converter import Constants
        from FourierExpansions import calc_4cos_coefs
        print("Using CCSD Vel and DFT VOH for potentials...")
        dvr_energies = self.DVRresults["energies"]  # [degrees vOH=0 ... vOH=6]
        dvr_energies[:, 1:] = Constants.convert(dvr_energies[:, 1:], "wavenumbers", to_AU=True)
        rad = np.radians(dvr_energies[:, 0])
        coeff_dict = dict()
        coeff_dict["Vel"] = self.Velcoeffs
        for i in np.arange(1, dvr_energies.shape[1]):  # loop through saved energies
            energies = np.column_stack((rad, dvr_energies[:, i]))
            coeff_dict[f"V{i-1}"] = calc_4cos_coefs(energies)
        return coeff_dict

    def calc_Mixed2_Vcoeffs(self):
        from Converter import Constants
        from FourierExpansions import calc_4cos_coefs
        print("Using DFT Vel and CCSD VOH for potentials...")
        all_data = self.PORparams["EmilEnergies"]
        tor_angles = all_data[0, :]
        rad = np.radians(tor_angles)
        ZPE = all_data[1, :]
        ens = all_data[2:, :] + ZPE
        ens = Constants.convert(ens, "wavenumbers", to_AU=True)
        coeff_dict = dict()
        coeff_dict["Vel"] = self.Velcoeffs
        for i in np.arange(len(ens)):
            dat = np.column_stack((rad, ens[i, :]))
            coeff_dict[f"V{i}"] = calc_4cos_coefs(dat)
        return coeff_dict

    def calc_Emil_Vcoeffs(self):
        from Converter import Constants
        from FourierExpansions import calc_4cos_coefs
        print("Using CCSD for all potentials...")
        all_data = self.PORparams["EmilEnergies"]
        tor_angles = all_data[0, :]
        rad = np.radians(tor_angles)
        ZPE = all_data[1, :]
        ens = all_data[2:, :] + ZPE
        ens = Constants.convert(ens, "wavenumbers", to_AU=True)
        coeff_dict = dict()
        coeff_dict["Vel"] = self.Velcoeffs
        for i in np.arange(len(ens)):
            dat = np.column_stack((rad, ens[i, :]))
            coeff_dict[f"V{i}"] = calc_4cos_coefs(dat)
        return coeff_dict

    def calc_scaled_Vcoeffs(self):
        from FourierExpansions import calc_curves
        from scaleTORpotentials import calc_scaled_Vcoeffs
        if len(self.Velcoeffs) == 7:
            energy = calc_curves(np.radians(np.arange(0, 360, 1)), self.Velcoeffs)
            order = 6
            ens_to_calc = self.DVRresults["energies"].shape[1] - 1  # subtract 1 for degree column
        elif len(self.Velcoeffs) == 5:
            energy = calc_curves(np.radians(np.arange(0, 360, 1)), self.Velcoeffs, function="4cos")
            ens_to_calc = 6
            order = 4
        else:
            raise Exception(f"Expansion order of {len(self.Velcoeffs)-1} currently not supported")
        energy_dat = np.column_stack((np.arange(0, 360, 1), energy))
        scaled_coeffs = calc_scaled_Vcoeffs(energy_dat, self.Vcoeffs, ens_to_calc, barrier_height=self.barrier_height,
                                            scaling_factor=self.scaling_factor, order=order)
        return scaled_coeffs

    def calc_Hamiltonian(self, pot_coeffs, kin_coeffs):
        """This should be fed the appropriately scaled Vcoefs, just calculates the Hamiltonian
        based of what it is given"""
        if self.HamSize is None:
            raise Exception("HamSize not defined in self.PORparams")
        else:
            HamSize = self.HamSize
        fullsize = 2 * HamSize + 1
        # check that the potential and kinetic energy are fit to equal orders, if not, pad with zeros
        if len(pot_coeffs) != len(kin_coeffs):
            if len(pot_coeffs) < len(kin_coeffs):
                pot_coeffs = np.hstack((pot_coeffs, np.zeros(len(kin_coeffs) - len(pot_coeffs))))
            elif len(pot_coeffs) > len(kin_coeffs):
                kin_coeffs = np.hstack((kin_coeffs, np.zeros(len(pot_coeffs) - len(kin_coeffs))))
        Ham = np.zeros((fullsize, fullsize))
        for l in np.arange(fullsize):
            k = l - HamSize
            Ham[l, l] = kin_coeffs[0] * (k ** 2) + pot_coeffs[0]  # m=0
            for kprime in np.arange(k + 1, k - len(pot_coeffs), -1):
                m = k - kprime
                if m > 0 and l - m >= 0:
                    Ham[l, l - m] = (kprime ** 2 + k ** 2 - m ** 2) * (kin_coeffs[m] / 4) + (pot_coeffs[m] / 2)
                    Ham[l - m, l] = Ham[l, l - m]
                else:
                    pass
        return Ham

    def solveHam(self):
        results = []
        for i in np.arange(len(self.Vcoeffs.keys()) - 1):  # loop through saved energies
            if self.barrier_height is None and self.scaling_factor is None:
                pot_coeffs = self.Vcoeffs["Vel"] + self.Vcoeffs[f"V{i}"]
            else:
                pot_coeffs = self.scaledVcoeffs[f"V{i}"]  # DO NOT ADD Vel because it is added in scaling function
            kin_coeffs = self.Bcoeffs[f"B{i}"]
            Hamiltonian = self.calc_Hamiltonian(pot_coeffs, kin_coeffs)
            energy, eigvecs = np.linalg.eigh(Hamiltonian)
            spacings = np.array((energy[2] - energy[0], energy[3] - energy[0],
                                 energy[2] - energy[1], energy[3]-energy[1]))
            barrier = self.calc_barrier(pot_coeffs)
            results.append({'V': pot_coeffs,
                            'barrier': barrier,
                            'energy': energy,
                            'eigvecs': eigvecs,
                            'spacings': spacings})  # 0+->1+, 0+->1-, 0-->1+, 0-->1-
        return results

    def calc_barrier(self, pot_coeffs):
        from FourierExpansions import calc_curves
        rad = np.radians(np.arange(0, 360, 1))
        if self.PORparams["Vexpansion"] is "fourth":
            pot_curve = calc_curves(rad, pot_coeffs, function="4cos")
        else:
            pot_curve = calc_curves(rad, pot_coeffs, function="cos")
        energy_dat = np.column_stack((np.degrees(rad), pot_curve))
        mins = np.argsort(energy_dat[:, 1])
        mins = np.sort(mins[:2])
        center = energy_dat[mins[0]:mins[-1] + 1, :]
        max_idx = np.argmax(center[:, 1])
        true_bh = center[max_idx, 1] - center[0, 1]
        return true_bh

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
