'''
Physics routines for N-body gravitational simulations

This module calculates the gravitational acceleration and total energy of the
N-body system. 
'''

import numpy as np

def gravity(s, t, eps2, m):
    '''
    Compute the time derivative of the N-body system state.

    This function evaluates the equations of motion for an N-body system
    using softened Newtonian gravity (G = 1), returning the time derivative
    of the flattened state vector for use in numerical integrators.

    Parameters
    ----------
    s : ndarray, shape (6 * N,)
        Interleaved state vector [x0, vx0, y0, vy0, z0, vz0, x1, vx1, ...].
    t : float
        Time (not used but required by integrator).
    eps2 : float
        Softening parameter squared (may be zero).
    m : ndarray, shape (N,)
        Masses of particles.

    Returns
    -------
    d : ndarray, shape (6 * N,)
        Interleaved derivative vector [vx0, ax0, vy0, ay0, vz0, az0, ...].
        vx, vy, vz: velocity components; ax, ay, az: acceleration components.
    '''
    
    N = len(m)
    pos = s[0::2].reshape(N, 3)  # extract positions as vectors
    acc = np.zeros_like(pos)

    for i in range(N):
        rs = pos[i] - pos
        r2 = (rs**2).sum(axis=1) + eps2
        if eps2 > 0:
            ir3 = - m / (np.sqrt(r2) * r2)
        else:
            ir3 = - np.divide(m, np.sqrt(r2) * r2,
                              out=np.zeros_like(r2), where=r2 != 0)
        acc[i] = (ir3[:, np.newaxis] * rs).sum(axis=0)

    d = np.empty(6 * N)
    d[0::2] = s[1::2]
    d[1::2] = acc.flatten()
    return d


def energy(s, t, eps2, m):
    '''
    Compute the total energy of the system.
    
    This function evaluates the kinetic and gravitational potential energy
    of the system using softened Newtonian gravity (G = 1).

    Parameters
    ----------
    s : ndarray, shape (6 * N,)
        Interleaved state vector with format 
        [x0, vx0, y0, vy0, z0, vz0, x1, vx1, ...].
    t : float
        Time (not used but required by integrator).
    eps2 : float
        Softening parameter squared.
    m : ndarray, shape (N,)
        Masses of particles.

    Returns
    -------
    total_energy : float
        Total energy of the system.
    '''
    
    N = len(m)
    pos = s[0::2].reshape(N, 3)
    vel = s[1::2].reshape(N, 3)
    ke = 0.5 * (m[:, np.newaxis] * vel**2).sum()
    pe = 0.0
    for i in range(N):
        rs = pos[i] - pos
        r2 = (rs**2).sum(axis=1) + eps2
        pei = - np.divide(m, np.sqrt(r2),
                          out=np.zeros_like(r2), where=r2 != eps2)
        pe += m[i] * pei.sum()
    pe /= 2  # adjust for double counting
    
    total_energy = ke + pe
    
    return total_energy