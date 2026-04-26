'''
N-body initial conditions generator

This module generates initial conditions for an N-body gravitational
simulation and computes derived parameters required for stable numerical
integration. It also generates run identifiers for organizing simulation 
outputs.
'''

import numpy as np
from pathlib import Path

def generate_run_id(runs_dir):
    '''
    Generate a new sequential run ID based on existing run folders.
    
    This function scans the specified directory for subfolders named
    in the format "runXXXX" (where XXXX is a zero-padded integer),
    determines the highest existing run number, and returns the next
    available run ID.
    
    Parameters
    ----------
    runs_dir : str
        Path to the directory containing run output folders.

    Returns
    -------
    str
        A new run ID in the format "runXXXX", where XXXX is a
        zero-padded integer (e.g., "run0001", "run0002").
    '''
    
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    
    max_id = 0
    
    for folder in runs_path.iterdir():
        if folder.is_dir():
            name = folder.name
            
            if name.startswith("run"):
                suffix = name[3:]
                
                if suffix.isdigit():
                    run_number = int(suffix)
                    max_id = max(max_id, run_number)
             
    next_id = max_id + 1
    
    return f"run{next_id:04d}"

def center_of_mass(m, x):
    '''
    Compute the center of mass of a system of particles.
    
    Parameters
    ----------
    m : array_like, shape (N,)
        Masses of the particles.
    x : array_like, shape (N, D)
        Positions of the particles where D is the number of spatial dimensions.

    Returns
    -------
    ndarray, shape (D,)
        The center of mass position vector.
    '''
    
    m = np.asarray(m).reshape(-1, 1)
    return np.sum(m * x, axis=0) / np.sum(m)

def initialize_nbody_system(N, L, seed=None, sigma=None):
    '''
    Generate an array of random masses and coordinates for N particles in
    a box of length L.
    
    This function retrieves the specified number of particles N and produces
    random initial coordinates within a box of specified length L. A random 
    mass is assigned to each particle, and velocity is determined using a 
    normal distribution described by sigma, which is computed using system 
    properties if not included in the input. The seed will be random unless
    specified.
    
    Parameters
    ----------
    N : int
        Number of particles to generate.
    L : float
        Length of the box containing the initial coordinates.
    seed : int or None, optional
        Random seed for reproducibility.
        If None, results are stochastic.
    sigma: float or None, optional
        Velocity dispersion for initial velocities.
        If None, a default value based on system mass and size is used.
                
    Returns
    -------
    s_0 : ndarray, shape (N, 6)
         Interleaved initial conditions, where each row corresponds to one 
         particle and has the format [x, vx, y, vy, z, vz].
    system_properties : dict
        Dictionary containing derived physical quantities and parameters used
        to initialize state.
        - m : ndarray, shape (N,)
            Mass of each particle.
        - N : int
            Number of particles.
        - M : float
            Total mass of the system.
        - sigma : float
            Velocity dispersion used for initial conditions.
        - seed : int
            Random seed for reproducibility.
        - vol_eff : float
            Effective volume from particle bounding box.
        - R_eff : float
            Effective spherical radius derived from bounding-box volume 
            vol_eff.
    '''
    
    rng = np.random.default_rng(seed)
    
    # Random masses in range [1,100)
    m = rng.uniform(1, 1000, N)
    M = np.sum(m)

    # Uniform spatial distributions in a cubic box [0, L]
    x_0 = L * rng.random((N, 1))
    y_0 = L * rng.random((N, 1))
    z_0 = L * rng.random((N, 1))
    
    r = np.hstack([x_0, y_0, z_0])
    
    r_cm = center_of_mass(m, r)
    r -= r_cm
    
    # Actual volume occupied by particles
    Lx = np.max(r[:, 0]) - np.min(r[:, 0])
    Ly = np.max(r[:, 1]) - np.min(r[:, 1])
    Lz = np.max(r[:, 2]) - np.min(r[:, 2])
    
    # Effective volume from particle bounding box (not fixed box volume)
    vol_eff =  Lx * Ly * Lz
    
    # Effective spherical radius derived from bounding-box volume
    R_eff = (3 * vol_eff / (4 * np.pi))**(1/3)
    
    # Virial-inspired velocity dispersion (G = 1 units)
    if sigma is None:
        sigma = np.sqrt(M / R_eff)
        
    sigma *= 0.7
    
    # Initial velocities (Gaussian distribution)
    vx_0 = rng.normal(0, sigma, (N, 1))
    vy_0 = rng.normal(0, sigma, (N, 1))
    vz_0 = rng.normal(0, sigma, (N, 1))
    
    v = np.hstack([vx_0, vy_0, vz_0])
    
    v_cm = center_of_mass(m, v)
    v -= v_cm
    
    # Stack data into one array.
    s_0 = np.column_stack([x_0, vx_0, y_0, vy_0, z_0, vz_0])
    
    system_properties = {"m": m,"N": N, "M": M, "sigma": sigma, "seed": seed, 
            "vol_eff": vol_eff, "R_eff": R_eff}
    
    return s_0, system_properties

def compute_sim_parameters(system_properties):
    '''
    Compute the parameters of the simulation.
    
    This functions uses physical scales to estimate the softening parameter
    and timestep to ensure numerical stability.

    Parameters
    ----------
    system_properties : dict
        Dictionary containing derived physical quantities and parameters used
        to initialize state.
        See initialize_nbody_system().

    Returns
    -------
    sim_params : dict
        Dictionary containing estimated parameters used for integration.
        - eps : float
            Softening parameter to account for close interactions.
        - dt : float
            Timestep evaluated by considering local interactions and system
            evolution.
        - numsteps : int
            Total number of timesteps based upon the dynamical time of the 
            system.
        - outstep : int
            Number of steps in between successive outputs.
    '''
    
    N = system_properties["N"]
    M = system_properties["M"]
    vol_eff = system_properties["vol_eff"]
    
    # System's total density and number density from the effective volume 
    density = M / vol_eff
    num_density = N / vol_eff
    
    # Average interparticle distance
    d_avg = num_density ** (-1/3)
    
    # Softening parameter considering average interparticle distance
    eps = 0.01 * d_avg
    
    # Dynamical time of the system assuming G = 1
    dynamical_time = 3 / np.sqrt(density)
    
    # Global timestep based on entire system's evolution 
    dt_glob = 0.02 * dynamical_time
    
    # Local timestep based on close interactions 
    dt_local = 0.25 * np.sqrt(eps**3 / M)
    
    # Actual timestep will be the minimum of dt_glob and dt_local to resolve 
    # the fastest physical process and ensure numerical stability
    dt = min(dt_glob, dt_local)
    numsteps = int(np.ceil(dynamical_time / dt)) # Rounded up
    
    # Define how often the data is recorded based on dynamical time
    # Record date every 1% of the simulation
    output_dt = 0.01 * dynamical_time
    outstep = int(output_dt / dt)
    
    sim_params = {"eps": eps, "dt": dt, "numsteps": numsteps, 
                  "outstep": outstep}
    
    return sim_params
    