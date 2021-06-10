import os
import numpy as np

def mass_weight(hessian, mass=None, num_coord=None, **opts):
    """Mass weights the Hessian based on given masses
        :arg hessian: full Hessian of system (get_hess)
        :arg mass: array of atomic masses (atomic units-me)
        :arg num_coord: number of atoms times 3 (3n)
        :returns gf: mass weighted hessian(ham)"""
    tot_mass = np.zeros(num_coord)
    for i, m in enumerate(mass):
        for j in range(3):
            tot_mass[i*3+j] = m
    m = 1 / np.sqrt(tot_mass)
    g = np.outer(m, m)
    gf = g*hessian
    return gf

def norms(ham):
    """solves (GF)*qn = lambda*qn
        :arg ham: Hamiltonian of system (massweighted hessian)
        :returns dictionary of frequencies squared(lambda) (atomic units) and normal mode coeficients (qns)"""
    freq2, qn = np.linalg.eigh(ham)
    normal_modes = {'freq2': freq2,
                    'qn': qn}
    return normal_modes

def run(hessian, numcoord=None, mass=None, **opts):
    """runs above functions and stores normal mode and frequency
        :arg hessian: the hessian of your system (probably from a formatted checkpoint file)
        :arg numcoord: number of atoms times 3 (3n)
        :arg mass: array of atomic masses (atomic units-me) **MUST be in order of gaussian coords
        :returns dictionary of frequencies squared(lambda) (atomic units) and normal mode coeficients (qns)"""
    ham = mass_weight(hessian, mass, numcoord)
    modes = norms(ham)
    # i, protfreq_ = getprotonmodes(modes, coordies)
    # res = {"Shared Proton Normal Mode": i-5, "Shared Proton Frequency": protfreq_}
    return modes


