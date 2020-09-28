from McUtils.GaussianInterface import GaussianFChkReader
import numpy as np


class FchkInterpreter:
    def __init__(self, *fchks, **kwargs):
        self.params = kwargs
        if len(fchks) == 0:
            raise Exception('Nothing to interpret.')
        self.fchks = fchks
        self._hessian = None
        self._cartesians = None  # dictionary of cartesian coordinates keyed by (x, y) distances
        self._gradient = None
        self._MP2Energy = None
        self._atomicmasses = None

    @property
    def cartesians(self):
        if self._cartesians is None:
            self._cartesians = self.get_coords()
        return self._cartesians

    @property
    def hessian(self):
        if self._hessian is None:
            self._hessian = self.get_hess()
        return self._hessian

    @property
    def gradient(self):
        if self._gradient is None:
            self._gradient = self.get_grad()
        return self._gradient

    @property
    def MP2Energy(self):
        if self._MP2Energy is None:
            self._MP2Energy = self.get_MP2energy()
        return self._MP2Energy

    @property
    def atomicmasses(self):
        if self._atomicmasses is None:
            self._atomicmasses = self.get_mass()
        return self._atomicmasses

    def get_coords(self):
        """Uses McUtils parser to pull cartesian coordinates
            :returns coords: nx3 coordinate matrix"""
        crds = []
        for fchk in self.fchks:
            with GaussianFChkReader(fchk) as reader:
                parse = reader.parse("Coordinates")
            coords = parse["Coordinates"]
            crds.append(coords)
        c = np.array(crds)
        if c.shape[0] == 1:
            c = np.squeeze(c)
        return c

    def get_hess(self):
        """Pulls the Hessian (Force Constants) from a Gaussian Frequency output file
            :arg chk_file: a Gaussian Frequency formatted checkpoint file
            :returns hess: full Hessian of system as an np.array"""

        for fchk in self.fchks:
            with GaussianFChkReader(fchk) as reader:
                parse = reader.parse("ForceConstants")
            forcies = parse["ForceConstants"]
        # ArrayPlot(forcies.array, colorbar=True).show()
        return forcies.array

    def get_grad(self):
        for fchk in self.fchks:
            with GaussianFChkReader(fchk) as reader:
                parse = reader.parse("Gradient")
            grad = parse["Gradient"]
        return grad

    def get_MP2energy(self):
        for fchk in self.fchks:
            with GaussianFChkReader(fchk) as reader:
                parse = reader.parse("MP2 Energy")
            ens = parse["MP2 Energy"]
        return ens

    def get_mass(self):
        for fchk in self.fchks:
            with GaussianFChkReader(fchk) as reader:
                parse = reader.parse("AtomicMasses")
            mass_array = parse["AtomicMasses"]
        return mass_array