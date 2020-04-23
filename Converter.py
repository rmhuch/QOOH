class Constants:
    # conversions from [unit] to a.u. equivalent
    atomic_units = {
        "wavenumbers": 4.55634e-6,
        "angstroms": 1 / 0.529177,
        "amu": 1.000000000000000000 / 6.02213670000e23 / 9.10938970000e-28
    }

    masses = {
        "H": (1.00782503223, "amu"),
        "O": (15.99491561957, "amu"),
        "D": (2.0141017778, "amu")
    }

    @classmethod
    def convert(cls, val, unit, to_AU=True):
        vv = cls.atomic_units[unit]
        return (val * vv) if to_AU else (val / vv)

    @classmethod
    def mass(cls, atom, to_AU=True):
        m = cls.masses[atom]
        if to_AU:
            m = cls.convert(*m)
        return m

class Potentials1D:
    def __init__(self):
        #  this should probably initialize an attribute so that you can call the returned function?
        pass

    def ho(self, grid, k=1):
        return k / 2 * np.power(grid, 2)

    def harmonic(self, x=None, y=None):
        from scipy import interpolate
        tck = interpolate.splrep(x, y, k=2, s=0)

        def pf(grid, extrap=tck):
            y_fit = interpolate.splev(grid, extrap, der=0)
            return y_fit
        return pf

    def potlint(self, x=None, y=None):
        from scipy import interpolate
        tck = interpolate.splrep(x, y, s=0)

        def pf(grid, extrap=tck):
            y_fit = interpolate.splev(grid, extrap, der=0)
            return y_fit
        return pf
