import numpy as np

class PiDVR:
    """ This class solve the Colbert-Miller DVR on the 0, 2pi range,
    using Fourier (sin&cos) expansions as the potential energy"""
    def __init__(self, PotentialCoeffs, GmatCoeffs, Nval, params=None):
        self.PotentialCoeffs = PotentialCoeffs  # list of coeffs where [0] is cos and [1] is sin
        # print(PotentialCoeffs)
        # print(GmatCoeffs)
        self.GmatCoeffs = GmatCoeffs  # should be the coeffs (^same format) for the SPECIFIC potential
        self.Nval = Nval  # the grid spacing in this scheme is (2N + 1) so, always input HALF of the pts you want
        self.nPts = 2*Nval+1
        self.params = params  # ideally this is a dictionary of other specifics that can be pulled from when neccessary
        self._grid = None  # must be in radians, same as tau_j,j' in notes
        self._Potential = None
        self._KineticEnergy = None
        self._Results = None

    @property
    def grid(self):
        if self._grid is None:
            grid = np.linspace(0, 2*np.pi, self.nPts+1)
            self._grid = grid[:-1]  # DVR domain is not inclusive of 2pi so we have to throw it out
        return self._grid

    @property
    def Potential(self):
        if self._Potential is None:
            self._Potential = self.get_pot()
        return self._Potential

    @property
    def KineticEnergy(self):
        if self._KineticEnergy is None:
            self._KineticEnergy = self.get_kinE()
        return self._KineticEnergy

    @property
    def Results(self):
        if self._Results is None:
            self._Results = self.solveHamiltonian()
        return self._Results

    def get_pot(self):
        from FourierExpansions import calc_curves
        PC = []
        for t in np.arange(len(self.PotentialCoeffs)):
            pot = self.PotentialCoeffs[t]
            g = self.GmatCoeffs[t]
            coeffs = np.zeros(len(pot))
            if t == 1:
                for i, val in enumerate(pot):
                    coeffs[i] = val - ((i+1)**2*g[i]*0.25)  # watch this. a fix to fucked up Gmat derivs
            else:
                for i, val in enumerate(pot):
                    coeffs[i] = val - (i**2*g[i]*0.25)  # watch this. a fix to fucked up Gmat derivs
            PC.append(coeffs)
        pot_vals = calc_curves(self.grid, PC, function="fourier")
        pot = np.diag(pot_vals)
        return pot

    def get_T(self):
        Tjj = np.repeat((1/2)*(self.Nval*((self.Nval+1)/3)), self.nPts)
        Tmat = np.diag(Tjj)
        for j in range(1, len(self.grid)):
            for j_prime in range(j):
                Tj_jprime = (1/2*((-1)**(j-j_prime))) * (np.cos((np.pi*(j-j_prime))/(2*self.Nval+1)) /
                                                         (2*np.sin((np.pi*(j-j_prime))/(2*self.Nval+1))**2))
                Tmat[j, j_prime] = Tj_jprime
                Tmat[j_prime, j] = Tj_jprime
        return Tmat

    # def calc_Gderivs(self):
    #     dG2tau = np.zeros(len(self.grid))
    #     for i, tau in enumerate(self.grid):
    #         sin_term = np.sum([(m+1)**2*self.GmatCoeffs[1][m]*np.sin((m+1)*tau)
    #                            for m in np.arange(len(self.GmatCoeffs[1]))])
    #         cos_term = np.sum([n**2*(self.GmatCoeffs[n]/4)*np.cos(n*tau)
    #                            for n in np.arange(1, len(self.GmatCoeffs))])
    #         dG2tau[i] = -sin_term - cos_term
    #     return dG2tau

    def get_kinE(self):
        # final KE consists of three parts: T_j,j', G(tau), and d^2G/dtau^2
        from FourierExpansions import calc_curves
        import matplotlib.pyplot as plt
        Tmat = self.get_T()  # nxn
        G_tau = calc_curves(self.grid, self.GmatCoeffs, function="fourier")
        # dG2_tau = self.calc_Gderivs()
        # G2_mat = np.diag(dG2_tau)  # project gmat out to diagonal like potential
        # calculate KE matrix
        kinE = np.zeros((self.nPts, self.nPts))
        for j in range(len(self.grid)):
            for j_prime in range(j+1):
                kinE[j, j_prime] = (1/2)*(Tmat[j, j_prime]*(G_tau[j]+G_tau[j_prime]))  #-G2_mat[j, j_prime])
                kinE[j_prime, j] = (1/2)*(Tmat[j, j_prime]*(G_tau[j]+G_tau[j_prime]))  #-G2_mat[j, j_prime])
        return kinE

    def solveHamiltonian(self):
        Ham = self.Potential + self.KineticEnergy
        energy, eigvecs = np.linalg.eigh(Ham)
        vals, barrier = self.calc_barrier(self.PotentialCoeffs)
        if "desired_energies" in self.params:
            end = self.params["desired_energies"]
            energy = energy[:end]
            eigvecs = eigvecs[:, :end]
        probabilites = self.calc_probs(eigvecs)  # tuple, 0 is left side, 1 is right side
        res = {'minima': vals,
               'barrier': barrier,
               'V': self.PotentialCoeffs,
               'grid': self.grid,
               'energy': energy,
               'eigvecs': eigvecs,
               'probs': probabilites}
        return res

    @staticmethod
    def calc_barrier(pot_coeffs):
        from FourierExpansions import fourier_barrier
        res = fourier_barrier(pot_coeffs)
        true_bh = res[2, 1] - res[1, 1]
        return res, true_bh

    def calc_probs(self, eigvecs):
        left = np.zeros(len(eigvecs[0]))
        right = np.zeros(len(eigvecs[0]))
        for i in np.arange(len(eigvecs[0])):
            left_wfn = eigvecs[np.argwhere(self.grid < np.pi), i]
            left[i] = np.dot(left_wfn.T, left_wfn)
            right_wfn = eigvecs[np.argwhere(self.grid > np.pi), i]
            right[i] = np.dot(right_wfn.T, right_wfn)
        return left, right
