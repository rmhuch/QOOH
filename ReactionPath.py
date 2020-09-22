import os
import numpy as np
from FChkInterpreter import FchkInterpreter
from Converter import Constants

def DataClass(fchk_names):
    data = FchkInterpreter(fchk_names)
    return data

def COMcoords(data, massweight=True):
    """
    This function takes coords for Gaussian output and translates them to them to the center of mass
    and then mass-weights them
    :param data: FchkInterpreter class of data from a Gaussian fchk file
    :type data: class
    :return: array of coordinates shifted to the center of mass and mass weighted
    :rtype: np.array
    """
    mass = Constants.convert(data.atomicmasses, "amu", to_AU=True)
    tot_mass = np.sum(mass)
    coords = Constants.convert(data.cartesians, "angstroms", to_AU=True)
    COM = np.zeros(3)
    for i in range(3):
        mr = 0
        for j in np.arange(len(mass)):
            mr += mass[j]*coords[j, i]
        COM[i] = (1/tot_mass) * mr
    x_coords = coords[:, 0] - COM[0]
    y_coords = coords[:, 1] - COM[1]
    z_coords = coords[:, 2] - COM[2]
    COM_coords = np.concatenate((x_coords[:, np.newaxis], y_coords[:, np.newaxis], z_coords[:, np.newaxis]), axis=1)
    if massweight:
        mwCOM_coords = np.zeros(COM_coords.shape)
        for j in np.arange(len(mass)):
            mwCOM_coords[j, :] = np.sqrt(mass[j]) * COM_coords[j, :]
        fcoords = mwCOM_coords
    else:
        fcoords = COM_coords
    return fcoords

def NormModeFreqs(fchk_names):
    """Calculate the normal mode frequencies"""
    from NormalModes import run
    data = DataClass(fchk_names)
    hess = data.hessian
    mass = Constants.convert(data.atomicmasses, "amu", to_AU=True)
    numCoords = 3 * len(mass)
    resAU = run(hess, numCoords, mass)
    freqsAU = np.sqrt(resAU["freq2"])
    freqs = Constants.convert(freqsAU, "wavenumbers", to_AU=False)
    print(freqs)
    return freqs

def mwFMatrix(data):
    """Mass weight the Hessian/Fmatrix"""
    hess = data.hessian
    mass = Constants.convert(data.atomicmasses, "amu", to_AU=True)
    num_coord = 3 * len(mass)
    tot_mass = np.zeros(num_coord)
    for i, m_val in enumerate(mass):
        for j in range(3):
            tot_mass[i*3+j] = m_val
    m = 1 / np.sqrt(tot_mass)
    g = np.outer(m, m)
    mwF_mat = g*hess
    return mwF_mat

def Translations(data):
    """Calculate the translation L matrix"""
    mass = Constants.convert(data.atomicmasses, "amu", to_AU=True)
    tot_mass = np.sum(mass)
    num_coord = 3 * len(mass)
    trans_mat = np.zeros((num_coord, 3))
    x_inds = np.arange(0, num_coord, 3)
    y_inds = np.arange(1, num_coord, 3)
    z_inds = np.arange(2, num_coord, 3)
    for i, x, y, z in zip(np.arange(num_coord), x_inds, y_inds, z_inds):
        trans_mat[x, 0] = np.sqrt(mass[i] / tot_mass)
        trans_mat[y, 1] = np.sqrt(mass[i] / tot_mass)
        trans_mat[z, 2] = np.sqrt(mass[i] / tot_mass)
    return trans_mat

def MomOfInertiaMatrix(data):
    """Calculate the moment of inertia matrix to be used in the rotation L matrix"""
    coordies = COMcoords(data)  # mass weighted and COM shifted in bohr
    I0_mat = sum(np.eye(3)*np.linalg.norm(ri)**2 - np.outer(ri, ri) for ri in coordies)
    I_eigv, I_eigfun = np.linalg.eigh(I0_mat)
    I_mat = I_eigfun @ np.diag(I_eigv**(-0.5)) @ I_eigfun.T
    # check = (I_mat @ I_mat) @ I0_mat  # should return idenity if things are right
    return I_mat

def Rotations(data):
    """Calculate the rotation L matrix"""
    coordies = COMcoords(data)  # mass weighted and COM shifted in bohr
    I_mat = MomOfInertiaMatrix(data)
    epsilon = np.array([
            [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
            [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]])
    num_coord = 3 * len(coordies)
    x_inds = np.arange(0, num_coord, 3)
    y_inds = np.arange(1, num_coord, 3)
    z_inds = np.arange(2, num_coord, 3)
    rot_mat = np.zeros((num_coord, 3))
    for n in np.arange(num_coord):
        i = n // 3
        if n in x_inds:
            gamma = 0
        elif n in y_inds:
            gamma = 1
        elif n in z_inds:
            gamma = 2
        else:
            raise Exception("I don't know that index")
        for k in np.arange(3):  # x, y, z
            for alpha in np.arange(3):  # x, y, z
                for beta in np.arange(3):  # x, y, z
                    rot_mat[n, k] += I_mat[k, alpha] * epsilon[alpha, beta, gamma] * coordies[i, beta]
    return rot_mat

def ProjectionMatrix(trans_mat, rot_mat):
    """
    The projection matrix of the translations and rotations (3N-6)
    :param trans_mat: resulting matrix of 'Translations'
    :type trans_mat: np.array
    :param rot_mat: resulting matrix of 'Rotations'
    :type rot_mat: np.array
    :return: a projection matrix (3n-6)
    :rtype: np.array
    """
    L = np.concatenate((trans_mat, rot_mat), axis=1)  # matrix of normal modes
    proj_mat = L@L.T
    return proj_mat

def RxnPathVector(data, proj_mat):
    """Calculate the reaction path vector and projection matrix including the reaction path."""
    grad = data.gradient
    mass = Constants.convert(data.atomicmasses, "amu", to_AU=True)
    # mass weight gradient - derivative of coordinate so mass term on the bottom, have to divide not multiply
    mw_grad = np.zeros(len(grad))
    for i in np.arange(len(grad)):
        mi = i // 3
        mw_grad[i] = grad[i] / np.sqrt(mass[mi])
    rxn_path = mw_grad / np.linalg.norm(mw_grad)
    r_proj = (np.eye(len(proj_mat)) - proj_mat) @ rxn_path
    r_projN = r_proj / np.linalg.norm(r_proj)
    return r_projN

def FullProjectionMatrix(trans_mat, rot_mat, r_proj):
    """
    The 'full' projection matrix: translations, rotations, rxn path (3N-7)
    :param trans_mat: resulting matrix of `Translations`
    :type trans_mat: np.array
    :param rot_mat: resulting matrix of `Rotations`
    :type rot_mat: np.array
    :param r_proj: normalized projection matrix of rxnpath
    :type r_proj: np.array
    :return: the full 3N-7 projection matrix
    :rtype: np.array
    """
    r_projM = r_proj[:, np.newaxis]
    L = np.concatenate((trans_mat, rot_mat, r_projM), axis=1)
    p_prime = L @ L.T
    return p_prime

def transOnlyFMatrix(data):
    """Calculate a new mass-weight Force constant matrix only accounting for translations (3N-3)"""
    mwF_mat = mwFMatrix(data)
    trans_mat = Translations(data)
    troj_mat = trans_mat@trans_mat.T
    f_trans = (np.identity(len(troj_mat)) - troj_mat) @ mwF_mat @ (np.identity(len(troj_mat)) - troj_mat)
    return f_trans

def transrotOnlyFMatrix(data):
    """Calculate a new mass-weight Force constant matrix accounting for translations and rotations (3N-6)"""
    mwF_mat = mwFMatrix(data)
    trans_mat = Translations(data)
    rot_mat = Rotations(data)
    proj_mat = ProjectionMatrix(trans_mat, rot_mat)
    f_trot = (np.identity(len(proj_mat))-proj_mat) @ mwF_mat @ (np.identity(len(proj_mat)) - proj_mat)
    return f_trot

def newFMatrix(data):
    """Calculate a new mass-weight Force constant matrix accounting for translations, rotations
     and a 1D reaction path (3N-7)"""
    mwF_mat = mwFMatrix(data)
    trans_mat = Translations(data)
    rot_mat = Rotations(data)
    proj_mat = ProjectionMatrix(trans_mat, rot_mat)
    r_proj = RxnPathVector(data, proj_mat)
    p_prime = FullProjectionMatrix(trans_mat, rot_mat, r_proj)
    f_prime = (np.identity(len(p_prime))-p_prime) @ mwF_mat @ (np.identity(len(p_prime))-p_prime)
    return f_prime

def run(fchk_names, fn):
    data = DataClass(fchk_names)
    forcies = newFMatrix(data)
    freqs2, qn = np.linalg.eigh(forcies)
    coords = COMcoords(data, massweight=False)
    np.savetxt(f"{fn}_modes.dat", np.row_stack((coords.flatten(), qn.T)))
    freqs = np.sign(freqs2) * Constants.convert(np.sqrt(np.abs(freqs2)), "wavenumbers", to_AU=False)
    np.savetxt(f"{fn}_freqs.dat", freqs)


if __name__ == '__main__':
    udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # OHdir = os.path.join(udrive, "TBHP", "eq_scans")
    # OHfreqdir = os.path.join(OHdir, "oh_mode_freqs")
    # OHfiles = ["07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
    # OHvals = ["09", "10", "11", "12", "13", "14", "15", "16"]
    #
    # for i in OHfiles:
    #     fn = os.path.join(OHdir, f"tbhp_eq_{i}.fchk")
    #     run(fn, f"tbhp_eq_{i}oh")
    #
    # TORdir = os.path.join(udrive, "TBHP", "TorFchks")
    # torfiles = ["000", "010", "020", "030", "040", "050", "060", "070", "080", "090", "100", "110", "120",
    #             "130", "140", "150", "160", "170", "180", "190", "200", "210", "220", "230", "240", "250",
    #             "260", "270", "280", "290", "300", "310", "320", "330", "340", "350", "360"]
    # for j in torfiles:
    #     fnn = os.path.join(TORdir, f"tbhp_{j}.fchk")
    #     run(fnn, f"tbhp_{j}tor")

    VIBdir = os.path.join(udrive, "TBHP", "VibStateFchks")
    vibfiles = np.arange(7)
    for k in vibfiles:
        fname = os.path.join(VIBdir, f"tbhp_eq_v{k}.fchk")
        run(fname, f"tbhp_eq_v{k}")

