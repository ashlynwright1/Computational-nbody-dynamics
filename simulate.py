'''
3D N-body gravitational simulator

This script generates random initial conditions, evolves an N-body system
under Newtonian gravity using RK4 and leapfrog integrators, and saves 
trajectories, energies, and parameters for each run in a structured output 
directory.
'''

import numpy as np
import argparse
import json
from tqdm import tqdm
from pathlib import Path
from nbody.initialconditions import (generate_run_id, 
initialize_nbody_system, compute_sim_parameters)
from nbody.physics import gravity, energy
from nbody.integrator import rk4, leapfrog
from nbody.visualize import animate_traj_energy

def main():
    parser = argparse.ArgumentParser(
        description='3D N-Body Simulator.',
        epilog='Simulator uses RK4 and leapfrog integrators to evolve the'
        'system.\n'
        'Trajectories and energy plots can be found in data/runs.')

    parser.add_argument('-N', '--numparticles', type=int, default=10,
                        help= 'number of particles (default: %(default)d)')
    parser.add_argument('-L', '--length', type=float,default=1.0,
                        help='length of box (default: %(default)d)')
    parser.add_argument('--sigma', type=float, default=None,
                        help='velocity dispersion (default: None)')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for random number generator (default: None)')
    parser.add_argument('--output', type=str, default='data/runs',
                        help='output directory for simulation runs'
                        '(default: %(default)s)')

    args = parser.parse_args()
    
    N = args.numparticles
    L = args.length
    sigma = args.sigma
    seed = args.seed
    runs_dir = args.output
    
    if args.numparticles < 0:
        raise ValueError('Number of particles cannot be negative.')
    if args.length <= 0:
        raise ValueError('Length of box must be positive.')
    if args.sigma is not None and args.sigma < 0:
        raise ValueError('Velocity dispersion cannot be negative.')
    if args.seed is not None and args.seed < 0:
        raise ValueError('Seed cannot be negative.')
        
    # Generate a new run directory for current simulation 
    run_id = generate_run_id(runs_dir)
    output_path = Path(runs_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate initial properties of the system
    s_0, system_properties = initialize_nbody_system(N, L, seed, sigma)
    s = s_0.reshape(-1) # Shape (6 * N,)
    m = system_properties["m"]

    # Compute simulation parameters
    parameters = compute_sim_parameters(system_properties)
    eps2 = parameters["eps"]**2
    dt = parameters["dt"]
    numsteps = parameters["numsteps"]
    outstep = parameters["outstep"]
    
    sim_params = {
    "metadata": {
        "N": N,
        "dt": dt,
        "seed": seed,
        "numsteps": numsteps,
        "masses": m.tolist()
        }
    }

    # Define integration boundaries
    output_steps = np.arange(0, numsteps + 1, outstep)
    num_outputs = len(output_steps)
    
    # Empty arrays for storing trajectory at each outstep
    rk4_trajectory = np.zeros((num_outputs, 6 * N))
    lf_trajectory = np.zeros((num_outputs, 6 * N))
    rk4_s = s.copy()
    lf_s = s.copy()
    
    # Empty arrays for storing total energy at each outstep
    rk4_energy = np.zeros(num_outputs)
    lf_energy = np.zeros(num_outputs)
    times = np.zeros(num_outputs)
        
    # Integrate the system at each outstep 
    i_out = 0
    for step in tqdm(range(numsteps + 1)):

        t = step * dt

        # Store trajectory and energy at each outstep
        if i_out < num_outputs and step == output_steps[i_out]:

            rk4_trajectory[i_out] = rk4_s
            lf_trajectory[i_out] = lf_s

            times[i_out] = t

            rk4_energy[i_out] = energy(rk4_s, t, eps2, m)
            lf_energy[i_out] = energy(lf_s, t, eps2, m)

            i_out += 1

        if step < numsteps:
            rk4_s = rk4(rk4_s, t, dt, gravity, args=(eps2, m))
            lf_s = leapfrog(lf_s, t, dt, gravity, args=(eps2, m))
    
    assert i_out == num_outputs

    # Save data to output path
    np.save(output_path / "RK4_trajectory.npy", rk4_trajectory)
    np.save(output_path / "leapfrog_trajectory.npy", lf_trajectory)
    np.save(output_path / "RK4_energy.npy", rk4_energy)
    np.save(output_path / "leapfrog_energy.npy", lf_energy)
    np.save(output_path / "times.npy", times)
    np.save(output_path / "initial_coords.npy", s_0)

    with open(output_path / "parameters.json", "w") as f:
        json.dump(sim_params, f, indent=4)
    
    # Produce animations
    animate_traj_energy('RK4_trajectory.npy', 'times.npy', 'RK4_energy.npy',
                        output_path)
    
    animate_traj_energy('leapfrog_trajectory.npy', 'times.npy', 
                        'leapfrog_energy.npy', output_path)
    
    print(f"Simulation for {run_id} complete!\n"
          f"Output can be found in {output_path}")

if __name__ == '__main__':
    main()