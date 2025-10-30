import numpy as np
import matplotlib.pyplot as plt
import os

def read_bd_dump_by_timestep(filename):
    """
    Generator yielding (timestep, box_bounds, atom_data) for each snapshot in a LAMMPS dump.
    `atom_data` is a list of dicts, each with keys like {col_id, col_x, col_y, col_z, ...}.
    """
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if "ITEM: TIMESTEP" not in line:
                continue

            timestep = int(f.readline().strip())

            _ = f.readline()
            n_atoms = int(f.readline().strip())

            _ = f.readline()
            box_bounds = []
            for _ in range(3):
                bounds_line = f.readline().split()
                box_bounds.append([float(bounds_line[0]), float(bounds_line[1])])

            line_atoms = f.readline()
            columns = line_atoms.strip().split()[2:]

            atom_data = []
            for _ in range(n_atoms):
                data_line = f.readline().split()
                entry = {}
                for cidx, col in enumerate(columns):
                    entry[col] = float(data_line[cidx])
                atom_data.append(entry)

            yield timestep, box_bounds, atom_data

def compute_bd_flux(dump_file, z_plane, area, dt_frame, window_size=20):
    """
    Computes BD flux with time averaging over the last window_size steps
    
    Parameters:
    -----------
    dump_file : str
        Path to the LAMMPS trajectory file
    z_plane : float
        Z-coordinate of the plane to measure flux across
    area : float
        Cross-sectional area of the system
    dt_frame : float
        Time step between frames
    window_size : int
        Number of steps to use for time averaging (default: 10)
    """
    prev_z = {}
    time_list = []
    flux_list = []
    raw_flux_values = []  # Store individual flux values before averaging
    
    frame_index = 0

    for timestep, box_bounds, atom_data in read_bd_dump_by_timestep(dump_file):
        current_time = frame_index * dt_frame

        # Count crossings since last frame
        crossing_count = 0
        crossing_count2 = 0
        if frame_index > 0:
            for a in atom_data:
                atom_id = int(a["id"])
                z_new = a["z"]
                z_old = prev_z.get(atom_id, None)
                if z_old is not None:
                    # Check crossing from below to above
                    if (z_old < z_plane) and (z_new >= z_plane):
                        crossing_count += 1
                    # Check crossing from above to below
                    if (z_old > z_plane) and (z_new <= z_plane):
                        crossing_count2 += 1

            # Calculate raw flux for this frame
            raw_flux = (crossing_count - crossing_count2) / (area * dt_frame)
            raw_flux_values.append(raw_flux)
            
            # Apply time averaging over the last window_size steps
            if len(raw_flux_values) >= window_size:
                # Average the last window_size values
                avg_flux = np.mean(raw_flux_values[-window_size:])
                flux_list.append(avg_flux)
            else:
                # If we don't have enough data points yet, use what we have
                avg_flux = np.mean(raw_flux_values)
                flux_list.append(avg_flux)
                
            time_list.append(current_time)

        # Update prev_z with current positions
        for a in atom_data:
            atom_id = int(a["id"])
            prev_z[atom_id] = a["z"]

        frame_index += 1

    return time_list, flux_list

def calculate_steady_state_flux(flux_list, time_list, steady_state_time=145.0):
    """Calculate the steady state flux by averaging over the last portion of the simulation."""
    # Convert lists to numpy arrays to ensure proper comparison
    time_array = np.array(time_list)
    flux_array = np.array(flux_list)
    
    # Find indices where time is greater than steady_state_time
    steady_indices = np.where(time_array >= steady_state_time)[0]
    
    if len(steady_indices) == 0:
        # If no steady state reached, use the last 20% of data
        steady_indices = np.arange(int(0.8 * len(flux_array)), len(flux_array))
    
    # Calculate average flux in steady state region
    steady_state_flux = np.mean(flux_array[steady_indices])
    return steady_state_flux

def calculate_flux_ratio(time_list, flux_list1, flux_list2):
    steady_flux_3 = calculate_steady_state_flux(flux_list1, time_list)
    
    # Calculate ratio for each time point
    flux_ratio = np.array(flux_list2) / steady_flux_3
    
    return flux_ratio, steady_flux_3

def plot_flux_ratio_comparison(time_lists, flux_ratios, steady_fluxes, run_names, output_name):
    """Plot the flux ratio comparison between different MOFs."""
    plt.figure(figsize=(10, 6))
    
    colors = ['b', 'r', 'g']
    
    for i, (time_list, flux_ratio, name, steady_flux) in enumerate(zip(time_lists, flux_ratios, run_names, steady_fluxes)):
        plt.plot(time_list[::2], flux_ratio[::2], 
                label=f'{name} (Steady flux: {steady_flux:.4f})', 
                color=colors[i], alpha=0.7)
    
    plt.xlabel('Time')
    plt.ylabel('Flux Ratio')
    plt.title('BD Flux Ratio Comparison Between MOFs')
    plt.grid(True)
    plt.legend()
    
    # Add horizontal line at y=1.0 for reference
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ratio = 1.0')
    
    plt.tight_layout()
    plt.savefig(f'{output_name}.png')
    plt.close()

def main():
    # Parameters
    z_plane1 = 2.5
    z_plane2 = 7.5
    area = 100.0
    dt_frame = 0.1
    n_bd_trials = 500

    bd_runs = [
        "./data/irmof1-open-dz32/",
        "./data/irmof10-open-dz32/"
    ]

    run_names = ["IRMOF-1", "IRMOF-10"]

    # Lists to store results for each run
    all_time_lists = []
    all_flux_lists1 = []
    all_flux_lists2 = []
    all_flux_ratios = []
    all_steady_fluxes = []

    for run_id in bd_runs:
        print(f"Processing {run_id}")
        
        # Add verification that files exist
        sample_file = f"{run_id}/trials/0/collection.lammpstrj"
        if not os.path.exists(sample_file):
            print(f"Warning: Could not find file {sample_file}")
            continue
            
        # Lists to store flux data from all trials for this run
        run_flux_lists1, run_flux_lists2 = [], []
        
        # Compute flux for each trial
        for trial in range(n_bd_trials):
            dump_file = f"{run_id}/trials/{trial}/collection.lammpstrj"
            
            # Add more detailed debugging output
            print(f"Trial #{trial} - Processing file: {dump_file}")
            
            # Compute flux for both planes
            time_list1, flux_list1 = compute_bd_flux(dump_file, z_plane1, area, dt_frame)
            _, flux_list2 = compute_bd_flux(dump_file, z_plane2, area, dt_frame)
            
            # Add some basic statistics output
            print(f"Flux statistics at z={z_plane1}: mean={np.mean(flux_list1):.3f}, std={np.std(flux_list1):.3f}")
            print(f"Flux statistics at z={z_plane2}: mean={np.mean(flux_list2):.3f}, std={np.std(flux_list2):.3f}")
            
            run_flux_lists1.append(flux_list1)
            run_flux_lists2.append(flux_list2)
        
        # Compute average flux across trials for this run
        avg_flux1 = np.mean(np.array(run_flux_lists1), axis=0)
        avg_flux2 = np.mean(np.array(run_flux_lists2), axis=0)
        
        # Calculate flux ratio
        flux_ratio, steady_flux = calculate_flux_ratio(time_list1, avg_flux1, avg_flux2)
        
        # Store results
        all_time_lists.append(time_list1)
        all_flux_lists1.append(avg_flux1)
        all_flux_lists2.append(avg_flux2)
        all_flux_ratios.append(flux_ratio)
        all_steady_fluxes.append(steady_flux)

    # Plot comparison of flux ratios
    plot_flux_ratio_comparison(
        all_time_lists,
        all_flux_ratios,
        all_steady_fluxes,
        run_names,
        'bd_flux_ratio_comparison'
    )
    
    # Also save the data for further analysis
    for i, name in enumerate(run_names):
        np.savez(
            f'bd_flux_{name}.npz',
            time=all_time_lists[i],
            flux_3=all_flux_lists1[i],
            flux_7=all_flux_lists2[i],
            flux_ratio=all_flux_ratios[i],
            steady_flux_3=all_steady_fluxes[i]
        )

if __name__ == "__main__":
    main() 
