# LLM assisted code

import glob
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import h5py
import os
from datetime import datetime

run_id = 'g1-closed-dz100'

# Load file lists
md_file_list = sorted(glob.glob(f"./data/{run_id}/avg_5000/collection.*.h5"))
ddft_c2_file_list = sorted(glob.glob(f"./results/c2/{run_id}/rho*.npy"))
ddft_c1_file_list = sorted(glob.glob(f"./results/c1/{run_id}/rho*.npy"))
fmt_file_list = sorted(glob.glob(f"./results/fmt/{run_id}/rho*.npy"))

if (len(md_file_list) == 0 or len(ddft_c2_file_list) == 0 or len(ddft_c1_file_list) == 0 or len(fmt_file_list) == 0):
    raise FileNotFoundError(f"No files found for {run_id}, check paths")

# Create save directory
current_time = datetime.now().strftime("%Y-%m-%d")
save_path = f"results/{current_time}-snapshots"
os.makedirs(save_path, exist_ok=True)

# Plotting setup
plt.style.use('science')
fig = plt.figure(figsize=(20, 5))
gs = fig.add_gridspec(1, 4)
axes = [fig.add_subplot(gs[0, j]) for j in range(4)]

subplot_ids = ['(a)', '(b)', '(c)', '(d)']

# Set font sizes
LABEL_SIZE = 18
LEGEND_SIZE = 18
TICK_SIZE = 16

# Determine total number of timesteps and calculate dt
total_steps = min(len(md_file_list), len(ddft_c2_file_list), len(ddft_c1_file_list), len(fmt_file_list))

# Find indices for snapshots at specific times
target_times = [0, 0.75, 1.5, 3]  # Target times in τ
snapshot_indices = [int(t / 0.01) for t in target_times]  # Convert times to frame indices

# Ensure indices are within bounds
snapshot_indices = [min(idx, total_steps - 1) for idx in snapshot_indices]

# Get the times corresponding to each snapshot
snapshot_times = [idx * 0.01 for idx in snapshot_indices]

# Load external potential and r values (assume same for all timesteps)
with h5py.File(md_file_list[0], 'r') as f:
    V = np.array(f['V']).flatten()
    r = np.array(f['r']).flatten()
    initial_md_density = np.array(f['n']).flatten()

# Plot snapshots
for idx, snapshot_idx in enumerate(snapshot_indices[:4]):
    ax = axes[idx]
    
    # Load densities for this timestep
    with h5py.File(md_file_list[snapshot_idx], 'r') as f:
        md_density = np.array(f['n']).flatten()
    
    ddft_c2_density = np.load(ddft_c2_file_list[snapshot_idx])
    ddft_c1_density = np.load(ddft_c1_file_list[snapshot_idx])
    fmt_density = np.load(fmt_file_list[snapshot_idx])
    
    # Plot densities
    ax.set_xlim(10, 20)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    
    ax.plot(r, md_density, ls='--', lw=2, label='Brownian Dynamics', color='green')
    ax.plot(r, fmt_density, lw=2, label='Hard Sphere + Mean Field Approx.', color='orange')
    ax.plot(r, ddft_c1_density, lw=2, label='Single-Body Direct Correlation Matching', color='blue')
    ax.plot(r, ddft_c2_density, lw=2, label='Pair-Correlation Matching', color='red')
    
    ax.set_xlabel(r'$z / \sigma$', fontsize=LABEL_SIZE)
    if idx == 0:  # Only show y-label for first plot
        ax.set_ylabel(r'$\rho(z) \sigma^3$', fontsize=LABEL_SIZE)
    
    # Add twin axis for external potential
    ax_twin = ax.twinx()
    ax_twin.fill_between(r, V, color="lightgray", alpha=0.1, label="Vext")
    ax_twin.plot(r, V, color="gray", label="Vext")
    ax_twin.tick_params(axis='y', labelcolor="gray", labelsize=TICK_SIZE)
    ax_twin.set_xlim(0, 10)
    if idx == 3:  # Show Vext label only on last plot
        ax_twin.set_ylabel(r'$\beta\text{V}_\text{ext}(z)$', color="gray", fontsize=LABEL_SIZE)
    
    # Add subplot label and time
    label_text = f'{subplot_ids[idx]}, $t = {snapshot_times[idx]:.2f}\\tau$'
    ax.text(0.5, 0.02, label_text, transform=ax.transAxes, 
            horizontalalignment='center', fontsize=LABEL_SIZE)

# Add legend to the top of the figure
handles, labels = axes[0].get_lines(), [l.get_label() for l in axes[0].get_lines()]
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=4, fontsize=LEGEND_SIZE, frameon=True, edgecolor='black')

plt.tight_layout()
fig.savefig(f'{save_path}/g1.pdf', bbox_inches='tight', pad_inches=0)
plt.close(fig)
print(f"Saved figure to {save_path}/g1.pdf")