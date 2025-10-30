# LLM generated code

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import h5py
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib import rcParams
import matplotlib as mpl
from datetime import datetime
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import shutil
import argparse

# Set up scientific plot style
def set_science_style():
    # Use LaTeX for text rendering
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    
    # Figure properties
    rcParams['figure.figsize'] = (16, 8)  # Wider figure for 1x2 layout
    rcParams['figure.dpi'] = 150
    rcParams['figure.constrained_layout.use'] = True
    
    # Font sizes - increased for paper publication
    rcParams['font.size'] = 16
    rcParams['axes.labelsize'] = 18
    rcParams['axes.titlesize'] = 20
    rcParams['xtick.labelsize'] = 16
    rcParams['ytick.labelsize'] = 16
    rcParams['legend.fontsize'] = 16
    
    # Line properties
    rcParams['lines.linewidth'] = 2
    rcParams['lines.markersize'] = 8
    
    # Axes properties
    rcParams['axes.linewidth'] = 1.5
    rcParams['axes.grid'] = True
    rcParams['grid.linestyle'] = '--'
    rcParams['grid.alpha'] = 0.7
    
    # Legend properties
    rcParams['legend.frameon'] = True
    rcParams['legend.framealpha'] = 0.9
    rcParams['legend.edgecolor'] = 'k'

def find_flux_ratio_files(prefix):
    """Find all flux ratio data files with the given prefix"""
    files = glob.glob(f"{prefix}*.npz")
    return files

def find_density_files(ddft_prefix, bd_prefix):
    """Find all density data files with the given prefixes"""
    ddft_files = glob.glob(f"{ddft_prefix}*.npy")
    bd_files = glob.glob(f"{bd_prefix}*.h5")
    return ddft_files, bd_files

def load_and_process_data(file_path):
    """Load and process data from NPZ files"""
    try:
        data = np.load(file_path)
        return {key: data[key] for key in data.files}
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_density_profiles(ax, ddft_files, bd_files):
    """Plot the density profiles and potential from DDFT and BD simulations"""
    # Create a single twin axis for potentials
    ax_twin = ax.twinx()
    
    # Store system potentials
    potentials = {}
    
    # Try to load the initial density (first timestep from first DDFT file)
    initial_density = None
    if ddft_files and len(ddft_files) > 0:
        try:
            # Look for the earliest rho file in the directory
            dir_path = os.path.dirname(ddft_files[0])
            all_files = sorted(glob.glob(f"{dir_path}/rho*.npy"))
            if all_files:
                # Get the first file in the sequence
                initial_file = all_files[0]
                print(f"Loading initial density from {initial_file}")
                initial_density = np.load(initial_file)
                initial_r = np.linspace(0, 10, len(initial_density))
                # Plot initial density with more visibility
                ax.plot(initial_r, initial_density, linestyle='--', color='black', alpha=0.7,
                      label='Initial density', linewidth=2.0, dashes=(2, 2))
        except Exception as e:
            print(f"Could not load initial density: {e}")
    
    # Ensure we have exactly two files of each type
    if len(ddft_files) >= 2 and len(bd_files) >= 2:
        # Use a fixed assignment for clarity - enforcing MOF-1 and MOF-2
        mof_assignments = [
            {"name": "MOF-1", "color": "blue", "ddft_file": ddft_files[0], "bd_file": bd_files[0]},
            {"name": "MOF-2", "color": "red", "ddft_file": ddft_files[1], "bd_file": bd_files[1]}
        ]
        
        print(f"Using strict file assignment:")
        for mof in mof_assignments:
            print(f"{mof['name']} ({mof['color']}): DDFT={mof['ddft_file']}, BD={mof['bd_file']}")
        
        # Process each MOF system
        for mof in mof_assignments:
            system_name = mof["name"]
            color = mof["color"]
            
            # Process BD file for this MOF
            bd_file = mof["bd_file"]
            if bd_file.endswith('.h5'):
                with h5py.File(bd_file, 'r') as f:
                    if 'r' in f and 'n' in f and 'V' in f:
                        r = np.array(f['r']).flatten()
                        density = np.array(f['n']).flatten()
                        V = np.array(f['V']).flatten()
                        
                        # Plot BD density with dashed line
                        ax.plot(r, density, linestyle='--', color=color, 
                               label=f'BD: {system_name}')
                        
                        # Store potential data for this system
                        potentials[system_name] = {'r': r, 'V': V, 'color': color}
            
            # Process DDFT file for this MOF
            ddft_file = mof["ddft_file"]
            if ddft_file.endswith('.npy'):
                # Load DDFT density data - use the latest timestep
                try:
                    dir_path = os.path.dirname(ddft_file)
                    all_files = sorted(glob.glob(f"{dir_path}/rho*.npy"))
                    # Get the second to last file (often the converged result)
                    final_density_file = all_files[-2] if len(all_files) > 1 else all_files[-1]
                    print(f"Loading final density from {final_density_file}")
                    rho = np.load(final_density_file)
                except Exception as e:
                    print(f"Error loading final density file: {e}, using provided file")
                    rho = np.load(ddft_file)
                
                r = np.linspace(0, 10, len(rho))  # Scale to 0-10 range
                
                # Plot DDFT density with solid line
                ax.plot(r, rho, linestyle='-', color=color, 
                       label=f'DDFT: {system_name}')
    else:
        print('Not enough files found!!')
    
    # Plot all potentials on the twin axis
    print(f"Plotting potentials: {list(potentials.keys())}")
    for system_name, data in potentials.items():
        r = data['r']
        V = data['V']
        color = data['color']
        
        # Plot potential with the system's color but with high transparency
        ax_twin.fill_between(r, V, color=color, alpha=0.1)
        ax_twin.plot(r, V, color=color, alpha=0.3, label=f"Vext: {system_name}")
    
    # Set up the twin axis if we have any potentials
    if potentials:
        ax_twin.set_ylabel(r'$\beta V_{\text{ext}}(z)$', color="gray", fontsize=18)
        ax_twin.tick_params(axis='y', labelcolor="gray", labelsize=16)
    
    # Add source and sink region labels
    # For source region (0-2)
    ax.axvspan(0, 2, alpha=0.1, color='lightgreen')
    ax.text(1, 1.1, 'Source', fontsize=16, ha='center', va='center', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))
    
    # For sink region (8-10)
    ax.axvspan(8, 10, alpha=0.1, color='lightblue')
    ax.text(9, 1.1, 'Sink', fontsize=16, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))
    
    # Set axes properties
    ax.set_xlabel('$z/\sigma$', fontsize=18)
    ax.set_ylabel('$\\rho(z)\\sigma^3$', fontsize=18)
    ax.set_xlim(0, 10)  # Set range from 0 to 10
    ax.set_ylim(0, 1.2)
    ax.tick_params(axis='both', labelsize=16)
    
    # Create a combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=16)
    
    # Add grid
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    return ax

def plot_flux_ratios(ax, ddft_files, bd_files):
    """Plot the flux ratios from DDFT and BD simulations"""
    
    # Process and organize data
    all_data = []
    
    # To ensure consistent color matching, sort the files first
    ddft_files = sorted(ddft_files)
    bd_files = sorted(bd_files)
    
    print("Processing DDFT flux files in order:", ddft_files)
    for file_path in ddft_files:
        data = load_and_process_data(file_path)
        if data is None:
            continue
        
        system_name = os.path.basename(file_path).replace('flux_ratio_data_', '').replace('.npz', '')
        all_data.append(('DDFT', system_name, data))
    
    print("Processing BD flux files in order:", bd_files)
    for file_path in bd_files:
        data = load_and_process_data(file_path)
        if data is None:
            continue
        
        system_name = os.path.basename(file_path).replace('bd_flux_ratio_data_', '').replace('.npz', '')
        all_data.append(('BD', system_name, data))
    
    # Plot data
    ddft_system_names = []
    bd_system_names = []
    
    # Force specific system association:
    # DDFT files: First = MOF-1, Second = MOF-2
    # BD files: First = MOF-1, Second = MOF-2
    mof_assignments = {
        f"DDFT_{ddft_files[0]}": {"name": "MOF-1", "color": "blue"},
        f"DDFT_{ddft_files[1]}": {"name": "MOF-2", "color": "red"},
        f"BD_{bd_files[0]}": {"name": "MOF-1", "color": "blue"},
        f"BD_{bd_files[1]}": {"name": "MOF-2", "color": "red"}
    }
    
    # Print the assignments for debugging
    print("File to MOF assignments:")
    for key, value in mof_assignments.items():
        print(f"  {key} -> {value['name']} ({value['color']})")
    
    # Ensure we assign different colors to different systems
    for method, system_name, data in all_data:
        
        for k, v in mof_assignments.items():
            if system_name in k:
                color = v["color"]
                system_label = v["name"]
                break
        
        print(f"Plotting {method} data for {system_name} as {system_label} in {color}")
        
        # Set linestyle based on method
        linestyle = '-' if method == 'DDFT' else '--'
        marker = 'o' if method == 'DDFT' else 's'
        
        # Get time and flux ratio data (with reduced points for clarity)
        stride = 10
        
        if method == 'DDFT':
            time = data['time'][::stride]
            flux_ratio = data['flux_ratio'][::stride]
            ddft_system_names.append(system_label)
        else:  # BD
            time = data['time'][::stride]
            flux_ratio = data['flux_ratio'][::stride]
            bd_system_names.append(system_label)

        # Plot data with appropriate color
        ax.plot(time, flux_ratio, linestyle=linestyle, color=color,
                marker=marker, markevery=5, markersize=8,
                alpha=0.8, label=f'{method}: {system_label}')
    
    # Add horizontal line at y=1.0 for reference
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.7, linewidth=1.5)
    
    # Set axes properties
    ax.set_xlabel('$t / \\tau$', fontsize=18)
    ax.set_ylabel('$J(z_2, t)/J(z_1, t=145\\tau)$', fontsize=18)    
    # Improve tick marks
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which='major', length=8, width=1.5, labelsize=16)
    ax.tick_params(which='minor', length=4, width=1)
    
    # Add grid for better readability
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    return ddft_system_names, bd_system_names

def create_combined_plot(ddft_flux_files, bd_flux_files, ddft_density_files, bd_density_files, output_path='combined_plot.pdf'):
    """Create a combined 1x2 plot with density profiles and flux ratios"""
    set_science_style()
    
    # Create a 1x2 figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
    
    # Plot density profiles in the first plot
    plot_density_profiles(axes[0], ddft_density_files, bd_density_files)
    axes[0].set_title('(a) Steady-state densities vs External potentials', fontsize=20)
    
    # Plot flux ratios in the second plot
    ddft_systems, bd_systems = plot_flux_ratios(axes[1], ddft_flux_files, bd_flux_files)
    axes[1].set_title('(b) Breakthrough curves', fontsize=20)
    
    # Get handles and labels from both plots for combined legend
    density_handles, density_labels = axes[0].get_legend_handles_labels()
    pot_handles = []
    pot_labels = []
    if hasattr(axes[0], 'twin_axes'):
        for twin_ax in axes[0].twin_axes:
            h, l = twin_ax.get_legend_handles_labels()
            pot_handles.extend(h)
            pot_labels.extend(l)
    
    flux_handles, flux_labels = axes[1].get_legend_handles_labels()
    
    # Create a combined, deduplicated legend
    all_labels = []
    all_handles = []
    
    # Helper to add only unique items
    added_labels = set()
    def add_unique(handle, label):
        if label not in added_labels:
            all_handles.append(handle)
            all_labels.append(label)
            added_labels.add(label)
    
    # Add in a specific order to ensure clear organization
    # First add initial density
    for i, label in enumerate(density_labels):
        if 'Initial' in label:
            add_unique(density_handles[i], label)
    
    # Then add DDFT and BD for MOF-1 and MOF-2
    for i, label in enumerate(density_labels):
        if 'DDFT: MOF-1' in label or 'BD: MOF-1' in label:
            add_unique(density_handles[i], label)
    
    for i, label in enumerate(density_labels):
        if 'DDFT: MOF-2' in label or 'BD: MOF-2' in label:
            add_unique(density_handles[i], label)
    
    # Then add potential labels
    for i, label in enumerate(pot_labels):
        add_unique(pot_handles[i], label)
    
    # Remove individual legends
    axes[0].get_legend().remove() if axes[0].get_legend() else None
    axes[1].get_legend().remove() if axes[1].get_legend() else None
    
    # Count the number of items for the legend and calculate the ncol value
    num_items = len(all_labels)
    # Use a single row legend
    ncol = min(num_items, 8)  # Max 8 columns, adjust as needed
    
    # Add combined legend at the top of the figure
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
              ncol=ncol, fontsize=16, frameon=True, fancybox=True)
    
    # Adjust spacing to make room for the legend (less space)
    fig.subplots_adjust(top=0.92)  # Increased from 0.85 to reduce whitespace
    
    # Save the figure with high resolution
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjusted to reduce whitespace
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")
    
    # Return the file names used
    return ddft_systems, bd_systems

def load_density_timesteps(dir_path):
    files = sorted(glob.glob(f"{dir_path}/rho*.npy"))
    return files

def load_bd_density_timesteps(dir_path):
    files = sorted(glob.glob(f"{dir_path}/collection*.h5"))
    return files

def create_combined_animation(
    ddft_density_dirs,
    bd_density_dirs,
    ddft_flux_files,
    bd_flux_files,
    output_path='combined_animation.mp4',
    fps=10,
    dpi=150
):
    set_science_style()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))

    # Gather timesteps
    ddft_timesteps = [load_density_timesteps(p) for p in ddft_density_dirs]
    bd_timesteps = [load_bd_density_timesteps(p) for p in bd_density_dirs]

    # Load flux data
    ddft_flux_files = sorted(ddft_flux_files)
    bd_flux_files = sorted(bd_flux_files)
    ddft_flux_data = [load_and_process_data(f) for f in ddft_flux_files]
    bd_flux_data = [load_and_process_data(f) for f in bd_flux_files]

    # Determine frames
    max_ddft_frames = max([len(f) for f in ddft_timesteps]) if ddft_timesteps else 0
    max_bd_frames = max([len(f) for f in bd_timesteps]) if bd_timesteps else 0
    max_frames = max(max_ddft_frames, max_bd_frames)
    if max_frames == 0:
        print("ERROR: No frames available for animation")
        return None

    # Assign MOFs (first -> MOF-1 blue, second -> MOF-2 red)
    mof_assignments = []
    if len(ddft_flux_files) >= 2 and len(bd_flux_files) >= 2:
        mof_assignments = [
            {"name": "MOF-1", "color": "blue", "ddft_idx": 0, "bd_idx": 0},
            {"name": "MOF-2", "color": "red", "ddft_idx": 1, "bd_idx": 1},
        ]

    # Density subplot setup
    ax_density = axes[0]
    ax_twin = ax_density.twinx()
    ax_density.set_xlabel('$z/\\sigma$', fontsize=18)
    ax_density.set_ylabel('$\\rho(z)\\sigma^3$', fontsize=18)
    ax_density.set_xlim(0, 10)
    ax_density.set_ylim(0, 1.2)
    ax_density.set_title('(a) Steady-state densities vs External potentials', fontsize=20)
    ax_density.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_density.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax_density.tick_params(which='major', length=8, width=1.5, labelsize=16)
    ax_density.tick_params(which='minor', length=4, width=1)
    ax_density.axvspan(0, 2, alpha=0.1, color='lightgreen')
    ax_density.text(1, 1.1, 'Source', fontsize=16, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))
    ax_density.axvspan(8, 10, alpha=0.1, color='lightblue')
    ax_density.text(9, 1.1, 'Sink', fontsize=16, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))

    # Flux subplot setup
    ax_flux = axes[1]
    ax_flux.set_xlabel('$t / \\tau$', fontsize=18)
    ax_flux.set_ylabel('$J(z_2, t)/J(z_1, t=145\\tau)$', fontsize=18)
    ax_flux.set_title('(b) Breakthrough curves', fontsize=20)
    ax_flux.axhline(y=1.0, color='k', linestyle=':', alpha=0.7, linewidth=1.5)
    ax_flux.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_flux.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax_flux.tick_params(which='major', length=8, width=1.5, labelsize=16)
    ax_flux.tick_params(which='minor', length=4, width=1)

    density_lines = {}
    flux_lines = {}

    # Initial density
    if ddft_timesteps and ddft_timesteps[0]:
        try:
            initial_density = np.load(ddft_timesteps[0][0])
            initial_r = np.linspace(0, 10, len(initial_density))
            line_init, = ax_density.plot(initial_r, initial_density, linestyle='--', color='black',
                                         alpha=0.7, label='Initial density', linewidth=2.0, dashes=(2, 2))
            density_lines['initial'] = line_init
        except Exception as e:
            print(f"Could not load initial density: {e}")

    # Initialize lines per MOF
    for mof in mof_assignments:
        name = mof['name']
        color = mof['color']
        ddft_idx = mof['ddft_idx']
        bd_idx = mof['bd_idx']

        ddft_density_line, = ax_density.plot([], [], linestyle='-', color=color, label=f'DDFT: {name}')
        bd_density_line, = ax_density.plot([], [], linestyle='--', color=color, label=f'BD: {name}')
        density_lines[f'ddft_{name}'] = ddft_density_line
        density_lines[f'bd_{name}'] = bd_density_line

        # Flux lines
        if ddft_idx < len(ddft_flux_data) and ddft_flux_data[ddft_idx] is not None:
            d_time = ddft_flux_data[ddft_idx]['time']
            d_flux = ddft_flux_data[ddft_idx]['flux_ratio']
            line, = ax_flux.plot([], [], linestyle='-', color=color, marker='o', markevery=5,
                                 alpha=0.8, label=f'DDFT: {name}')
            flux_lines[f'ddft_{name}'] = {'line': line, 'time': d_time, 'data': d_flux}

        if bd_idx < len(bd_flux_data) and bd_flux_data[bd_idx] is not None:
            b_time = bd_flux_data[bd_idx]['time']
            b_flux = bd_flux_data[bd_idx]['flux_ratio']
            line, = ax_flux.plot([], [], linestyle='--', color=color, marker='s', markevery=5,
                                 alpha=0.8, label=f'BD: {name}')
            flux_lines[f'bd_{name}'] = {'line': line, 'time': b_time, 'data': b_flux}

    # Potentials from first BD files
    potentials = {}
    for mof in mof_assignments:
        name = mof['name']
        color = mof['color']
        bd_idx = mof['bd_idx']
        if bd_idx < len(bd_timesteps) and bd_timesteps[bd_idx]:
            try:
                with h5py.File(bd_timesteps[bd_idx][0], 'r') as f:
                    if 'r' in f and 'V' in f:
                        r = np.array(f['r']).flatten()
                        V = np.array(f['V']).flatten()
                        ax_twin.fill_between(r, V, color=color, alpha=0.1)
                        ax_twin.plot(r, V, color=color, alpha=0.3)
                        potentials[name] = {'r': r, 'V': V}
            except Exception as e:
                print(f"Potential load error for {name}: {e}")

    if potentials:
        ax_twin.set_ylabel(r'$\\beta V_{\\text{ext}}(z)$', color="gray", fontsize=18)
        ax_twin.tick_params(axis='y', labelcolor="gray", labelsize=16)

    # Legend (combined)
    density_handles, density_labels = ax_density.get_legend_handles_labels()
    pot_handles, pot_labels = ax_twin.get_legend_handles_labels() if potentials else ([], [])
    flux_handles, flux_labels = ax_flux.get_legend_handles_labels()
    all_handles, all_labels = [], []
    added = set()
    def add_unique(h, l):
        if l not in added:
            all_handles.append(h)
            all_labels.append(l)
            added.add(l)
    for i, label in enumerate(density_labels):
        if 'Initial' in label:
            add_unique(density_handles[i], label)
    for system_name in ["MOF-1", "MOF-2"]:
        for prefix in ["DDFT", "BD"]:
            for i, label in enumerate(density_labels):
                if f'{prefix}: {system_name}' in label:
                    add_unique(density_handles[i], label)
            for i, label in enumerate(flux_labels):
                if f'{prefix}: {system_name}' in label:
                    add_unique(flux_handles[i], label)
    for i, label in enumerate(pot_labels):
        add_unique(pot_handles[i], label)
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=min(len(all_labels), 8), fontsize=16, frameon=True, fancybox=True)
    fig.subplots_adjust(top=0.92)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Grids
    ax_density.grid(True, which='major', linestyle='--', alpha=0.7)
    ax_density.grid(True, which='minor', linestyle=':', alpha=0.4)
    ax_flux.grid(True, which='major', linestyle='--', alpha=0.7)
    ax_flux.grid(True, which='minor', linestyle=':', alpha=0.4)

    timestamp = ax_density.text(0.02, 0.98, '', transform=ax_density.transAxes,
                                fontsize=16, ha='left', va='top',
                                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    def update(frame):
        # timestamp (assume dt=0.1)
        current_time = frame * 0.1
        timestamp.set_text(f'Time: {current_time:.1f} $\\tau$')

        # densities
        for mof in mof_assignments:
            name = mof['name']
            di = mof['ddft_idx']
            bi = mof['bd_idx']
            if di < len(ddft_timesteps) and frame < len(ddft_timesteps[di]):
                try:
                    rho = np.load(ddft_timesteps[di][frame])
                    r = np.linspace(0, 10, len(rho))
                    density_lines[f'ddft_{name}'].set_data(r, rho)
                except Exception as e:
                    pass
            if bi < len(bd_timesteps) and frame < len(bd_timesteps[bi]):
                try:
                    with h5py.File(bd_timesteps[bi][frame], 'r') as f:
                        if 'r' in f and 'n' in f:
                            r = np.array(f['r']).flatten()
                            n = np.array(f['n']).flatten()
                            density_lines[f'bd_{name}'].set_data(r, n)
                except Exception as e:
                    pass

        # fluxes
        for key, d in flux_lines.items():
            line = d['line']
            times = d['time']
            vals = d['data']
            if len(times) == 0:
                line.set_data([], [])
                continue
            idx = min(np.searchsorted(times, current_time), len(times) - 1)
            stride = 10
            vis_idx = np.arange(0, idx + 1, stride)
            if len(vis_idx) > 0:
                line.set_data(times[vis_idx], vals[vis_idx])
            else:
                line.set_data([], [])

        # keep axes ranges steady
        if frame >= 0:
            ax_flux.set_xlim(0, 150)
            ax_flux.set_ylim(0, 1.2)

        return list(density_lines.values()) + [timestamp] + [v['line'] for v in flux_lines.values()]

    frame_step = 10
    ani = FuncAnimation(fig, update, frames=range(0, max_frames, frame_step), blit=True)

    try:
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            print("WARNING: ffmpeg not found in PATH")
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='DDFT/BD Animation'), bitrate=1800)
        ani.save(output_path, writer=writer, dpi=dpi)
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"Failed to save animation: {e}")

    plt.close()
    return ani


def main():
    parser = argparse.ArgumentParser(description='Plot density profiles and optionally animate')
    parser.add_argument('--animate', action='store_true', help='Create animation of density profiles and RMSE over time')
    args = parser.parse_args()

    current_time = datetime.now().strftime("%Y-%m-%d")
    # Look for DDFT and BD flux ratio data files
    ddft_flux_files = find_flux_ratio_files("ddft_flux_")
    bd_flux_files = find_flux_ratio_files("bd_flux_")
    
    # Look for density profile files in the specified directories
    ddft_density_files = []
    bd_density_files = []
    
    # Specific paths for density files as provided by the user
    ddft_density_paths = [
        "./results/c2/irmof1-open-dz32",
        "./results/c2/irmof10-open-dz32"
    ]
    
    # Specific paths for BD density files (absolute paths)
    bd_density_paths = [
        "./data/irmof1-open-dz32/avg_500",
        "./data/irmof10-open-dz32/avg_500"
    ]
    
    # Search for DDFT density files in the specified directories
    for path in ddft_density_paths:
        # Find DDFT density files (rho*.npy)
        ddft_files = glob.glob(f"{path}/rho*.npy")
        if ddft_files:
            # Use the latest file (typically last timestep)
            ddft_density_files.append(sorted(ddft_files)[-2])  # Using -2 instead of -1 as used in plot_combined
    
    # Search for BD density files in the specified directories
    for path in bd_density_paths:
        # Find BD density files (collection*.h5)
        bd_files = glob.glob(f"{path}/collection*.h5")
        if bd_files:
            # Use the latest file
            bd_density_files.append(sorted(bd_files)[-1])
    
    # If no specific files found, try the fallback method
    if not ddft_density_files or not bd_density_files:
        print("No density files found...")

    
    print(f"Found {len(ddft_flux_files)} DDFT flux files and {len(bd_flux_files)} BD flux files")
    print(f"Found {len(ddft_density_files)} DDFT density files and {len(bd_density_files)} BD density files")
    print(f"DDFT density files: {ddft_density_files}")
    print(f"BD density files: {bd_density_files}")
    
    save_path = f"results/{current_time}-open-mof-dz32"
    os.makedirs(save_path, exist_ok=True)
    # Create the combined plot
    ddft_systems, bd_systems = create_combined_plot(
        ddft_flux_files, bd_flux_files, 
        ddft_density_files, bd_density_files,
        f"{save_path}/combined_plot.pdf"
    )
    
    print(f"DDFT systems plotted: {', '.join(ddft_systems)}")
    print(f"BD systems plotted: {', '.join(bd_systems)}")

    # Create the combined animation
    if args.animate:
        print("Creating animation...")
        create_combined_animation(
            ddft_density_dirs=ddft_density_paths,
            bd_density_dirs=bd_density_paths,
            ddft_flux_files=ddft_flux_files,
            bd_flux_files=bd_flux_files,
            output_path=f"{save_path}/combined_animation.mp4",
            fps=10,
            dpi=150
        )

if __name__ == "__main__":
    main()