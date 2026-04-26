"""
Visualization tools for N-body simulations

This module provides functions to analyze simulation outputs, including total 
energy diagnostics and 3D orbital animation of particle trajectories.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from matplotlib.animation import FuncAnimation
   
def animate_traj_energy(trajectory_file, time_file, energy_file, output_path):
    '''
    Animate 3D trajectories of particles next to the system's total energy
    evolution for an N-body system.

    This function loads trajectory, time, and energy data to produce a 3D
    animation of trajectories within an N-body system and a plot of total
    energy evolution to assess numerical stability and conservation for a 
    given integrator.

    Parameters
    ----------
    trajectory_file : str
        File containing particle state vectors at each output step.
    time_file : str
        File containing simulation time values at each output step.
    energy_file : str
        File containing total system energy at each output step.
    output_path : str
        Directory containing the simulation run data.

    Returns
    -------
    None.
    '''

    # Load in the data
    traj = np.load(output_path / trajectory_file)
    time = np.load(output_path / time_file)
    energy = np.load(output_path / energy_file)
    with open(output_path / "parameters.json", "r") as f:
        params = json.load(f)
        
    masses = params["metadata"]["masses"]
    masses = np.array(masses)
    
    integrator = Path(trajectory_file).stem.split("_")[0]
    
    # Set initial conditions
    x = traj[0, 0::6]
    y = traj[0, 2::6]
    z = traj[0, 4::6]
    
    # Set bounds
    x_all = traj[:, 0::6]
    y_all = traj[:, 2::6]
    z_all = traj[:, 4::6]

    # Set up figure for animation
    frames, N = traj.shape

    fig = plt.figure(figsize=(10,5))
    
    # Trajectory animation
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_3d.set_title(f"N-Body Trajectories Using the {integrator} Integrator")
    scatter = ax_3d.scatter(x, y, z, c=masses, cmap='viridis')
    cbar = plt.colorbar(scatter, location = 'bottom')
    cbar.set_label("Mass")
    ax_3d.set_xlim(np.min(x_all), np.max(x_all))
    ax_3d.set_ylim(np.min(y_all), np.max(y_all))
    ax_3d.set_zlim(np.min(z_all), np.max(z_all))
    
    # Energy animation
    ax_energy = fig.add_subplot(122)
    ax_energy.set_xlabel('Time')
    ax_energy.set_ylabel('Total Energy')
    ax_energy.set_title(f'Total Energy Using the {integrator} Integrator')
    ax_energy.set_xlim(np.min(time), np.max(time))
    ax_energy.set_ylim(np.min(energy), np.max(energy))
    
    # Initial energy
    line_energy, = ax_energy.plot([], [], lw=2)

    def update(frame):
        x = traj[frame, 0::6]
        y = traj[frame, 2::6]
        z = traj[frame, 4::6]
        scatter._offsets3d = (x, y, z)
        
        t = time[:frame]
        E = energy[:frame]
        line_energy.set_data(t, E)
        
        return scatter, line_energy

    ani = FuncAnimation(fig, update, frames=frames, interval=15, blit=False)

    plt.show()

    ani.save(output_path / f"traj_and_energy_{integrator}.gif", fps=15, 
             writer="pillow")