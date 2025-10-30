import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import os
import scienceplots
from datetime import datetime
import argparse
import sys

# CLI
parser = argparse.ArgumentParser(description='Plot density profiles and optionally animate')
parser.add_argument('--animate', action='store_true', help='Create animation of density profiles and RMSE over time')
args = parser.parse_args()

run_id1 = "g28-closed-dz100"
run_id2 = "g3-closed-dz100"
run_id3 = "g4-closed-dz100"
run_id4 = "g14-closed-dz100"

current_time = datetime.now().strftime("%Y-%m-%d")
save_path = f"results/{current_time}-closed-gplots-dz100"
os.makedirs(save_path, exist_ok=True)

# Plotting setup
plt.style.use('science')
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 4)
axes_top = [fig.add_subplot(gs[0, i]) for i in range(4)]
axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(4)]

subplot_ids = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

# Set font sizes
LABEL_SIZE = 18
LEGEND_SIZE = 18
TICK_SIZE = 16

def compute_error(rho1, rho2):
	return np.sqrt(np.mean((rho1[5:-5] - rho2[5:-5])**2))

# Plot both density profiles and error plots
for i in range(4): 
	run_id = eval(f"run_id{i+1}")

	md_file_list = sorted(glob.glob(f"./data/{run_id}/avg_5000/collection.*.h5"))    
	ddft_file_list = sorted(glob.glob(f"./results/c2/{run_id}/rho*.npy"))
	ddft_c1_file_list = sorted(glob.glob(f"./results/c1/{run_id}/rho*.npy"))
	fmt_file_list = sorted(glob.glob(f"./results/fmt/{run_id}/rho*.npy"))

	if (len(md_file_list) == 0 or len(ddft_file_list) == 0 or len(ddft_c1_file_list) == 0 or len(fmt_file_list) == 0):
		raise FileNotFoundError(f"No files found for {run_id}, check paths")

	# Top row: Density profiles
	ax_top = axes_top[i]
	with h5py.File(md_file_list[-1], 'r') as f:
		V = np.array(f['V']).flatten()
		r = np.array(f['r']).flatten()
		md_density = np.array(f['n']).flatten()

	init_rho = np.load(ddft_file_list[0])
	final_ddft = np.load(ddft_file_list[-2])
	final_ddft_c1 = np.load(ddft_c1_file_list[-2])
	final_fmt = np.load(fmt_file_list[-2])

	ax_top.set_xlim(10, 20)
	ax_top.set_ylim(0, 1.2)
	ax_top.set_yticks(np.arange(0, 1.1, 0.1))
	ax_top.tick_params(axis='both', labelsize=TICK_SIZE)
	
	ax_top.plot(r, init_rho, ls='--', alpha=.3, lw=2, label='Initial density')
	ax_top.plot(r, md_density, ls='--', lw=2, label='Brownian Dynamics', color='green')
	ax_top.plot(r, final_fmt, lw=2, label='Hard Sphere + Mean Field Approx.', color='orange')
	ax_top.plot(r, final_ddft_c1, lw=2, label='Single-Body Direct Correlation Matching', color='blue')
	ax_top.plot(r, final_ddft, lw=2, label='Pair-Correlation Matching', color='red')
	
	ax_top.set_xlabel(r'$z / \sigma$', fontsize=LABEL_SIZE)
	if i == 0:
		ax_top.set_ylabel(r'$\rho(z) \sigma^3$', fontsize=LABEL_SIZE)
	
	ax_twin = ax_top.twinx()
	# Add shading under the V_ext curve
	ax_twin.fill_between(r, V, color="lightgray", alpha=0.1, label="Vext")
	ax_twin.plot(r, V, color="gray", label="Vext")
	ax_twin.tick_params(axis='y', labelcolor="gray", labelsize=TICK_SIZE)
	ax_twin.set_xlim(0, 10)
	if i == 3:
		ax_twin.set_ylabel(r'$\beta\text{V}_\text{ext}(z)$', color="gray", fontsize=LABEL_SIZE)

	ax_top.text(0.05, 0.95, f'{subplot_ids[i]}', transform=ax_top.transAxes, 
	            verticalalignment='top', fontsize=LABEL_SIZE)

	# Bottom row: Error plots
	ax_bottom = axes_bottom[i]
	
	# Calculate error at each timestep
	ddft_errors = []
	ddft_c1_errors = []
	fmt_errors = []
	times = []
	
	for j in range(min(len(ddft_file_list), len(ddft_c1_file_list), len(fmt_file_list), len(md_file_list))):
		ddft_rho = np.load(ddft_file_list[j])
		ddft_c1_rho = np.load(ddft_c1_file_list[j])
		fmt_rho = np.load(fmt_file_list[j])
		with h5py.File(md_file_list[j], 'r') as f:
			md_rho = np.array(f['n']).flatten()
		
		ddft_errors.append(compute_error(ddft_rho, md_rho))
		ddft_c1_errors.append(compute_error(ddft_c1_rho, md_rho))
		fmt_errors.append(compute_error(fmt_rho, md_rho))
		times.append(j * 0.01)  # Assuming dt=0.01 between frames
	
	# Store error data for later use when setting common y-axis limits
	if i == 0:
		all_errors = []
	all_errors.extend(ddft_errors)
	all_errors.extend(ddft_c1_errors)
	all_errors.extend(fmt_errors)

	ax_bottom.plot(times, fmt_errors, lw=2, label='Hard Sphere + Mean Field Approx.', color='orange')
	ax_bottom.plot(times, ddft_errors, lw=2, label='Pair-Correlation Matching', color='red')
	ax_bottom.plot(times, ddft_c1_errors, lw=2, label='Single-Body Direct Correlation Matching', color='blue')
	
	ax_bottom.set_xlabel(r'$t / \tau$', fontsize=LABEL_SIZE)
	if i == 0:
		ax_bottom.set_ylabel(r'RMSE $\sigma^3$', fontsize=LABEL_SIZE)
	ax_bottom.tick_params(axis='both', labelsize=TICK_SIZE)
	ax_bottom.text(0.05, 0.95, f'{subplot_ids[i+4]}', transform=ax_bottom.transAxes,
	              verticalalignment='top', fontsize=LABEL_SIZE)

# Set the same y-axis limits for all error plots based on global min and max
global_min_error = min(all_errors)
global_max_error = max(all_errors)
padding = (global_max_error - global_min_error) * 0.1  # Add 10% padding
for ax in axes_bottom:
	ax.set_ylim(global_min_error - padding, global_max_error + padding)

# Add legend to the top of the figure
handles, labels = axes_top[0].get_lines(), [l.get_label() for l in axes_top[0].get_lines()]
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=3, fontsize=LEGEND_SIZE, frameon=True, edgecolor='black')

plt.tight_layout()
fig.savefig(f'{save_path}/combined_plot_with_error.pdf', bbox_inches='tight', pad_inches=0)
plt.close(fig)
print(f"Saved figure to {save_path}/combined_plot_with_error.pdf")

# Optional animation branch
# Warning: LLM generated code below
if args.animate:
	# Animation setup
	plt.style.use('science')
	fig = plt.figure(figsize=(20, 11))
	gs = fig.add_gridspec(2, 4)
	axes_top = [fig.add_subplot(gs[0, i]) for i in range(4)]
	axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(4)]

	# Font sizes and labels
	LABEL_SIZE = 18
	LEGEND_SIZE = 18
	TICK_SIZE = 16
	subplot_ids = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

	def compute_error(rho1, rho2):
		return np.sqrt(np.mean((rho1[5:-5] - rho2[5:-5])**2))

	# Load all data and prepare for animation
	all_data = []
	all_errors = []  # For global error scale

	for i in range(4):
		run_id = eval(f"run_id{i+1}")

		md_file_list = sorted(glob.glob(f"./data/{run_id}/avg_5000/collection.*.h5"))
		ddft_file_list = sorted(glob.glob(f"./results/c2/{run_id}/rho*.npy"))
		ddft_c1_file_list = sorted(glob.glob(f"./results/c1/{run_id}/rho*.npy"))
		fmt_file_list = sorted(glob.glob(f"./results/fmt/{run_id}/rho*.npy"))

		if (len(md_file_list) == 0 or len(ddft_file_list) == 0 or len(ddft_c1_file_list) == 0 or len(fmt_file_list) == 0):
			raise FileNotFoundError(f"No files found for {run_id}, check paths")

		# Get r and V values for this column
		with h5py.File(md_file_list[0], 'r') as f:
			r = np.array(f['r']).flatten()
			V = np.array(f['V']).flatten()

		# Calculate number of frames
		num_frames_total = min(len(ddft_file_list), len(ddft_c1_file_list), len(fmt_file_list), len(md_file_list))

		# Calculate full error data 
		full_ddft_errors = []
		full_ddft_c1_errors = []
		full_fmt_errors = []
		full_times = []

		for j in range(num_frames_total):
			ddft_rho = np.load(ddft_file_list[j])
			ddft_c1_rho = np.load(ddft_c1_file_list[j])
			fmt_rho = np.load(fmt_file_list[j])
			with h5py.File(md_file_list[j], 'r') as f:
				md_rho = np.array(f['n']).flatten()

			full_ddft_errors.append(compute_error(ddft_rho, md_rho))
			full_ddft_c1_errors.append(compute_error(ddft_c1_rho, md_rho))
			full_fmt_errors.append(compute_error(fmt_rho, md_rho))
			full_times.append(j * 0.01)

			# For global error list
			all_errors.append(full_ddft_errors[-1])
			all_errors.append(full_ddft_c1_errors[-1])
			all_errors.append(full_fmt_errors[-1])

		ddft_errors = []
		ddft_c1_errors = []
		fmt_errors = []
		times = []
		md_densities = []
		ddft_densities = []
		ddft_c1_densities = []
		fmt_densities = []

		for j in range(num_frames_total):

			ddft_rho = np.load(ddft_file_list[j])
			ddft_c1_rho = np.load(ddft_c1_file_list[j])
			fmt_rho = np.load(fmt_file_list[j])
			with h5py.File(md_file_list[j], 'r') as f:
				md_rho = np.array(f['n']).flatten()

			ddft_errors.append(compute_error(ddft_rho, md_rho))
			ddft_c1_errors.append(compute_error(ddft_c1_rho, md_rho))
			fmt_errors.append(compute_error(fmt_rho, md_rho))
			times.append(j * 0.01)

			md_densities.append(md_rho)
			ddft_densities.append(ddft_rho)
			ddft_c1_densities.append(ddft_c1_rho)
			fmt_densities.append(fmt_rho)

		column_data = {
			'r': r,
			'V': V,
			'md_densities': md_densities,
			'ddft_densities': ddft_densities,
			'ddft_c1_densities': ddft_c1_densities,
			'fmt_densities': fmt_densities,
			'ddft_errors': ddft_errors,
			'ddft_c1_errors': ddft_c1_errors,
			'fmt_errors': fmt_errors,
			'times': times,
			'full_ddft_errors': full_ddft_errors,
			'full_ddft_c1_errors': full_ddft_c1_errors,
			'full_fmt_errors': full_fmt_errors,
			'full_times': full_times,
			'num_frames': len(times)
		}

		all_data.append(column_data)

	# Calculate global min/max for consistent y-axis scaling across all plots
	global_min_error = min(all_errors)
	global_max_error = max(all_errors)
	padding = (global_max_error - global_min_error) * 0.1
	error_y_min = global_min_error - padding
	error_y_max = global_max_error + padding

	# Set up the figure and axes
	for i in range(4):
		# Configure top row (density plots)
		ax_top = axes_top[i]
		ax_top.set_xlim(10, 20)
		ax_top.set_ylim(0, 1.2)
		ax_top.set_yticks(np.arange(0, 1.1, 0.1))
		ax_top.tick_params(axis='both', labelsize=TICK_SIZE)
		ax_top.set_xlabel(r'$z / \sigma$', fontsize=LABEL_SIZE)
		if i == 0:
			ax_top.set_ylabel(r'$\rho(z) \sigma^3$', fontsize=LABEL_SIZE)
		ax_top.text(0.05, 0.95, f'{subplot_ids[i]}', transform=ax_top.transAxes, verticalalignment='top', fontsize=LABEL_SIZE)

		# Configure twin axis for V_ext
		ax_twin = ax_top.twinx()
		ax_twin.fill_between(all_data[i]['r'], all_data[i]['V'], color="lightgray", alpha=0.1)
		ax_twin.plot(all_data[i]['r'], all_data[i]['V'], color="gray", label="Vext")
		ax_twin.tick_params(axis='y', labelcolor="gray", labelsize=TICK_SIZE)
		ax_twin.set_xlim(0, 10)
		if i == 3:
			ax_twin.set_ylabel(r'$\beta\text{V}_\text{ext}(z)$', color="gray", fontsize=LABEL_SIZE)

		# Configure bottom row (error plots)
		ax_bottom = axes_bottom[i]
		ax_bottom.set_ylim(error_y_min, error_y_max)
		ax_bottom.set_xlim(0, all_data[i]['times'][-1] if all_data[i]['times'] else 0.1)
		ax_bottom.set_xlabel(r'$t / \tau$', fontsize=LABEL_SIZE)
		if i == 0:
			ax_bottom.set_ylabel(r'RMSE $\sigma^3$', fontsize=LABEL_SIZE)
		ax_bottom.tick_params(axis='both', labelsize=TICK_SIZE)
		ax_bottom.text(0.05, 0.95, f'{subplot_ids[i+4]}', transform=ax_bottom.transAxes, verticalalignment='top', fontsize=LABEL_SIZE)

	# Create line objects for animation
	density_lines = []
	error_lines = []
	time_text = fig.text(0.5, 0.02, '', ha='center', va='center', fontsize=LABEL_SIZE, 
	                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

	for i in range(4):
		# Create lines for density plots
		init_rho_line, = axes_top[i].plot([], [], ls='--', alpha=.3, lw=2, label='Initial density')
		md_line, = axes_top[i].plot([], [], ls='--', lw=2, label='Brownian Dynamics', color='green')
		fmt_line, = axes_top[i].plot([], [], lw=2, label='Hard Sphere + Mean Field Approx.', color='orange')
		ddft_c1_line, = axes_top[i].plot([], [], lw=2, label='Single-Body Direct Correlation Matching', color='blue')
		ddft_line, = axes_top[i].plot([], [], lw=2, label='Pair-Correlation Matching', color='red')

		density_lines.append({
			'init': init_rho_line,
			'md': md_line,
			'fmt': fmt_line,
			'ddft_c1': ddft_c1_line,
			'ddft': ddft_line
		})

		# Create lines for error plots
		fmt_error_line, = axes_bottom[i].plot([], [], lw=2, label='Hard Sphere + Mean Field Approx.', color='orange')
		ddft_error_line, = axes_bottom[i].plot([], [], lw=2, label='Pair-Correlation Matching', color='red')
		ddft_c1_error_line, = axes_bottom[i].plot([], [], lw=2, label='Single-Body Direct Correlation Matching', color='blue')

		error_lines.append({
			'fmt': fmt_error_line,
			'ddft': ddft_error_line,
			'ddft_c1': ddft_c1_error_line
		})

	# Add legend to the top of the figure
	handles = [density_lines[0]['init'], density_lines[0]['md'], density_lines[0]['fmt'], density_lines[0]['ddft_c1'], density_lines[0]['ddft']]
	labels = ['Initial density', 'Brownian Dynamics', 'Hard Sphere + Mean Field Approx.', 'Single-Body Direct Correlation Matching', 'Pair-Correlation Matching']

	plt.tight_layout()
	legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=LEGEND_SIZE, frameon=True, edgecolor='black')
	legend.set_zorder(100)
	fig.subplots_adjust(top=0.85)

	# Animation function
	def animate(frame):
		current_time_val = frame * 0.01
		time_text.set_text(f'Time: {current_time_val:.2f} $\\tau$')

		for i in range(4):
			r = all_data[i]['r']
			data = all_data[i]

			# Initial density (use first DDFT frame)
			if data['ddft_densities']:
				density_lines[i]['init'].set_data(r, data['ddft_densities'][0])

			# Handle cases where frame might exceed available data
			actual_frame = min(frame, data['num_frames'] - 1) if data['num_frames'] > 0 else 0

			if data['md_densities']:
				density_lines[i]['md'].set_data(r, data['md_densities'][actual_frame])
			if data['fmt_densities']:
				density_lines[i]['fmt'].set_data(r, data['fmt_densities'][actual_frame])
			if data['ddft_c1_densities']:
				density_lines[i]['ddft_c1'].set_data(r, data['ddft_c1_densities'][actual_frame])
			if data['ddft_densities']:
				density_lines[i]['ddft'].set_data(r, data['ddft_densities'][actual_frame])

			# Update error lines up to current frame
			if actual_frame > 0:
				times = data['times'][:actual_frame+1]
				error_lines[i]['fmt'].set_data(times, data['fmt_errors'][:actual_frame+1])
				error_lines[i]['ddft'].set_data(times, data['ddft_errors'][:actual_frame+1])
				error_lines[i]['ddft_c1'].set_data(times, data['ddft_c1_errors'][:actual_frame+1])

		return sum([list(d.values()) for d in density_lines], []) + \
		       sum([list(d.values()) for d in error_lines], []) + \
		       [time_text]

	# Frames and writer settings
	max_frames = max([data['num_frames'] for data in all_data]) if all_data else 0
	fps = 10
	dpi = 200

	ani = animation.FuncAnimation(
		fig, animate, frames=max_frames, interval=200, blit=True, cache_frame_data=False
	)

	output_file = f"{save_path}/density_error_animation.mp4"
	print(f"Saving animation to {output_file}")
	ani.save(output_file, writer='ffmpeg', fps=fps, dpi=dpi, extra_args=['-vcodec', 'libx264'])
	plt.close(fig)
	print("Animation saved successfully!")