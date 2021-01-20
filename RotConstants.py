from ReactionPath import DataClass, COMcoords
from Converter import Constants
import numpy as np
import os

def RotConstants(MolecularInfo_obj, filename):
    data = MolecularInfo_obj.DipoleMomentSurface[filename]
    coords = data[:, 4:]
    # DipoleMomentSurface contains rotated coordinates, so we start after the surface.
    # These values are eckart rotated and center of mass shifted and mass weighted and in bohr
    # coordies = Constants.convert(coords, "angstroms", to_AU=False)
    coordies = coords.reshape((coords.shape[0], 16, 3))[12]
    I0_mat = sum(np.eye(3)*np.linalg.norm(ri)**2 - np.outer(ri, ri) for ri in coordies)
    I_inverse = np.linalg.inv(I0_mat)
    rot_consts2 = np.diag(I_inverse)
    rot_consts = rot_consts2/2
    return rot_consts

def RotConstants_prev(fchkDir, fchkName):
    data = DataClass(os.path.join(fchkDir, fchkName))
    coordies = COMcoords(data)  # mass weighted and COM shifted in bohr
    I0_mat = sum(np.eye(3) * np.linalg.norm(ri) ** 2 - np.outer(ri, ri) for ri in coordies)
    I_inverse = np.linalg.inv(I0_mat)
    rot_consts2 = np.diag(I_inverse)
    rot_consts = rot_consts2/2
    return rot_consts

def RotCoeffs(RotationConstants):
    from FourierExpansions import calc_cos_coefs, calc_curves
    import matplotlib.pyplot as plt
    Mcoefs = np.zeros((7, 3))  # [num_coeffs, (ABC)]
    x = np.radians(np.linspace(0, 360, len(RotationConstants)))
    for c, val in enumerate(["A", "B", "C"]):
        # fit to 6th order cos functions
        data = np.column_stack((x, RotationConstants[:, c]*6579689.7))
        Mcoefs[:, c] = calc_cos_coefs(data)
        y = calc_curves(np.radians(np.arange(0, 361, 1)), Mcoefs[:, c]/6579689.7, function="cos")
        plt.plot(np.arange(0, 361, 1), y)
        plt.plot(np.degrees(x), RotationConstants[:, c], "o")
    plt.show()
    return Mcoefs

def RotationMatrix(RotationCoeffs, MatSize=15):
    """expand the transition dipole moment into a matrix representation so that it can
    be used to solve for intensity"""
    RotationCoeffs /= 6579689.7
    fullsize = 2 * MatSize + 1
    fullMat = np.zeros((3, fullsize, fullsize))
    for c, val in enumerate(["A", "B", "C"]):
        Mat = np.zeros((fullsize, fullsize))
        for l in np.arange(fullsize):
            k = l - MatSize
            Mat[l, l] = RotationCoeffs[:, c][0]  # m=0
            for kprime in np.arange(k + 1, k - 7, -1):
                m = k - kprime
                if m > 0 and l - m >= 0:
                    Mat[l, l - m] = (RotationCoeffs[:, c][m] / 2)
                    Mat[l - m, l] = Mat[l, l - m]
                else:
                    pass
        fullMat[c, :, :] = Mat
    return fullMat


def calc_all_RotConstants(molInfo_obj, torWfn_coefs, numstates, vOH, filetags=""):
    import csv
    from Converter import Constants
    degrees = np.arange(0, 370, 10)
    rots = np.zeros((len(degrees), 3))
    with open(f"RotationalConstants_vOH{vOH}{filetags}.csv", mode="w") as results:
        results_writer = csv.writer(results, delimiter=',')
        results_writer.writerow(["initial", "final", "A", "B", "C"])
        for d, val in enumerate(["000", "010", "020", "030", "040", "050", "060", "070", "080", "090", "100", "110",
                                 "120", "130", "140", "150", "160", "170", "180", "190", "200", "210", "220", "230",
                                 "240", "250", "260", "270", "280", "290", "300", "310", "320", "330", "340", "350",
                                 "360"]):
            rots[d] = RotConstants(molInfo_obj, f"tbhp_{val}.log")
        R_coeffs = RotCoeffs(rots)
        R_mat = RotationMatrix(R_coeffs)
        for State in np.arange(numstates):
            matEl = np.zeros(3)
            for c, val in enumerate(["A", "B", "C"]):  # loop through components
                supere = np.dot(R_mat[c, :, :], torWfn_coefs[:, State].T)
                matEl_au = np.dot(torWfn_coefs[:, State], supere)
                matEl[c] = Constants.convert(matEl_au, "wavenumbers", to_AU=False)
            results_writer.writerow([State, State, *matEl])


if __name__ == '__main__':
    from MolecularResults import *
    from runTBHP import tbhp
    for name in tbhp.TorFiles:
        RotConstants(tbhp, name)

