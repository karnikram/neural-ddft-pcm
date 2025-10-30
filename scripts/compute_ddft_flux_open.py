import torch
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
import matplotlib.animation as animation
from scipy.signal import savgol_filter
from datetime import datetime


def compute_ddft_flux(rho, V_ext, dFdrho, z, gamma, beta):
    """Compute DDFT flux using pre-calculated V_ext and dFdrho values"""
    velocity = (V_ext + 1/beta * dFdrho + 1/beta * np.log(rho))
    flux = - gamma * rho * np.gradient(velocity, z) * 1/32
    return flux

def compute_ddft_velocity(rho, V_ext, dFdrho, beta):
    """Compute DDFT velocity using pre-calculated V_ext and dFdrho values"""
    velocity = (V_ext + 1/beta * dFdrho + 1/beta * np.log(rho))
    return velocity

def compute_ddft_flux_smooth(rho, V_ext, dFdrho, z, gamma, beta, method='spline', smoothing=0.01):
    """Compute DDFT flux using smoother methods for calculating derivatives"""
    velocity = (V_ext + 1/beta * dFdrho + 1/beta * np.log(rho))
    
    if method == 'gaussian':
        # Apply Gaussian smoothing before differentiation
        velocity_smooth = gaussian_filter1d(velocity, sigma=1.5, mode='nearest')
        flux = - gamma * rho * np.gradient(velocity_smooth, z) * 1/32
    
    elif method == 'savgol':
        # Savitzky-Golay filter for smoothed derivatives
        velocity_gradient = savgol_filter(velocity, window_length=37, polyorder=1, deriv=0, delta=z[1]-z[0], mode='nearest')
        flux = - gamma * rho * velocity_gradient * 1/32
    
    else:
        # Default to original method
        flux = - gamma * rho * np.gradient(velocity, z) * 1/32
    
    return flux

def compute_flux_for_run(run_path, z, gamma, beta, dt_frame, plane_loc1, plane_loc2):
    """Compute flux using saved rho, V_ext, and dFdrho files if available, otherwise calculate them"""
    ddft_file_list = sorted(glob.glob(f"{run_path}/rho*.npy"))
    flux_list1 = []  # for first plane location
    flux_list2 = []  # for second plane location
    
    for _, file in enumerate(ddft_file_list):
        # Extract the timestamp from the filename
        timestamp = file.split('rho')[1].split('.npy')[0]
        # Load rho file
        rho = np.load(file)
        
        # Try to load V_ext and dFdrho files
        v_ext_path = f"{run_path}/V_ext{timestamp}.npy"
        dfdrho_path = f"{run_path}/dFdrho{timestamp}.npy"
        
        # Try to use saved files
        V_ext = np.load(v_ext_path)
        dFdrho = np.load(dfdrho_path)
        print(f"Using saved V_ext and dFdrho files for timestamp {timestamp}")
        
        # Calculate raw flux first
        flux_raw = compute_ddft_flux(rho, V_ext, dFdrho, z, gamma, beta)
        
        # Smooth the raw flux directly
        flux = savgol_filter(flux_raw, window_length=37, polyorder=1, deriv=0, mode='nearest') # Made smoothing gentler
        
        flux_list1.append(np.interp(plane_loc1, z, flux))
        flux_list2.append(np.interp(plane_loc2, z, flux))
    
    time_list = np.arange(len(flux_list1)) * dt_frame
    return time_list, flux_list1, flux_list2

def calculate_steady_state_flux(flux_list, time_list, steady_state_time=145.0):
    """Calculate the steady state flux by averaging over the last portion of the simulation."""
    # Find indices where time is greater than steady_state_time
    steady_indices = np.where(time_list >= steady_state_time)[0]
    
    if len(steady_indices) == 0:
        # If no steady state reached, use the last 20% of data
        steady_indices = np.arange(int(0.8 * len(flux_list)), len(flux_list))
    
    # Calculate average flux in steady state region
    steady_state_flux = np.mean(np.array(flux_list)[steady_indices])
    return steady_state_flux

def calculate_flux_ratio(time_list, flux_list1, flux_list2, plane_loc1, plane_loc2):
    """Calculate the ratio of flux at second plane location to steady state flux at first plane location."""
    # Get steady state flux at first plane location
    steady_flux_1 = calculate_steady_state_flux(flux_list1, time_list)

    # Calculate ratio for each time point
    flux_ratio = flux_list2 / steady_flux_1
    
    return flux_ratio, steady_flux_1

def compute_net_flux_for_region(run_path, z, gamma, beta, dt_frame, z_min=2.0, z_max=8.0):
    """Compute net flux within a specific region (z_min to z_max) for each time step"""
    ddft_file_list = sorted(glob.glob(f"{run_path}/rho*.npy"))
    net_flux_list = []
    
    # Find indices corresponding to the region boundaries
    z_min_idx = np.argmin(np.abs(z - z_min))
    z_max_idx = np.argmin(np.abs(z - z_max))
    
    for _, file in enumerate(ddft_file_list):
        # Extract the timestamp from the filename
        timestamp = file.split('rho')[1].split('.npy')[0]
        # Load rho file
        rho = np.load(file)
        
        # Load V_ext and dFdrho files
        v_ext_path = f"{run_path}/V_ext{timestamp}.npy"
        dfdrho_path = f"{run_path}/dFdrho{timestamp}.npy"
        
        V_ext = np.load(v_ext_path)
        dFdrho = np.load(dfdrho_path)
        
        # Compute flux at all points
        flux = compute_ddft_flux(rho, V_ext, dFdrho, z, gamma, beta)
        
        # Calculate net flux within the specified region
        region_flux = flux[z_min_idx:z_max_idx]
        region_z = z[z_min_idx:z_max_idx]
        net_flux = np.sum(region_flux) * (region_z[1] - region_z[0])  # Multiply by dz for proper integration
        net_flux_list.append(net_flux)
    
    time_list = np.arange(len(net_flux_list)) * dt_frame
    return time_list, net_flux_list

def compute_boundary_flux_difference(run_path, z, gamma, beta, dt_frame, z_min=2.0, z_max=8.0):
    """Compute the difference in flux between the boundaries of a region"""
    ddft_file_list = sorted(glob.glob(f"{run_path}/rho*.npy"))
    boundary_flux_diff_list = []
    
    # Find indices corresponding to the region boundaries
    z_min_idx = np.argmin(np.abs(z - z_min))
    z_max_idx = np.argmin(np.abs(z - z_max))
    
    for _, file in enumerate(ddft_file_list):
        # Extract the timestamp from the filename
        timestamp = file.split('rho')[1].split('.npy')[0]
        # Load rho file
        rho = np.load(file)
        
        # Load V_ext and dFdrho files
        v_ext_path = f"{run_path}/V_ext{timestamp}.npy"
        dfdrho_path = f"{run_path}/dFdrho{timestamp}.npy"
        
        V_ext = np.load(v_ext_path)
        dFdrho = np.load(dfdrho_path)
        
        # Compute flux at all points
        flux = compute_ddft_flux(rho, V_ext, dFdrho, z, gamma, beta)
        
        # Get flux at the boundaries
        flux_at_zmin = flux[z_min_idx]
        flux_at_zmax = flux[z_max_idx]
        
        # Calculate the difference (should be close to zero in steady state)
        boundary_flux_diff = flux_at_zmax - flux_at_zmin
        boundary_flux_diff_list.append(boundary_flux_diff)
    
    time_list = np.arange(len(boundary_flux_diff_list)) * dt_frame
    return time_list, boundary_flux_diff_list
    
def main():

    current_time = datetime.now().strftime("%Y-%m-%d")
    # Parameters
    k = 1
    T = 2
    beta = 1 / (k * T)
    gamma = 10
    dt_frame = 0.1
    
    # Define region for net flux calculation
    z_min = 2.5
    z_max = 7.5

    z = np.linspace(0, 10, 320)

    # Define the DDFT runs to compare
    run_paths = [
        "./results/c2/irmof1-open-dz32/",
        "./results/c2/irmof10-open-dz32/"
    ]
    
    run_names = ["IRMOF-1", "IRMOF-10"]

    # Compute fluxes for each run
    time_lists = []
    flux_lists1 = []
    flux_lists2 = []
    flux_ratios = []
    steady_fluxes = []
    net_flux_lists = []
    boundary_flux_diff_lists = []

    for run_path in run_paths:
        time_list, flux_list1, flux_list2 = compute_flux_for_run(
            run_path, z, gamma, beta, dt_frame, z_min, z_max
        )
        
        # Calculate flux ratio
        flux_ratio, steady_flux = calculate_flux_ratio(time_list, flux_list1, flux_list2, z_min, z_max)
        
        _, net_flux_list = compute_net_flux_for_region(run_path, z, gamma, beta, dt_frame, z_min, z_max)
        
        time_lists.append(time_list)
        flux_lists1.append(flux_list1)
        flux_lists2.append(flux_list2)
        flux_ratios.append(flux_ratio)
        steady_fluxes.append(steady_flux)
        net_flux_lists.append(net_flux_list)

    for i, name in enumerate(run_names):
        np.savez(
            f'ddft_flux_{name}.npz',
            time=time_lists[i],
            flux_plane1=flux_lists1[i],
            flux_plane2=flux_lists2[i],
            flux_ratio=flux_ratios[i],
            steady_flux_plane1=steady_fluxes[i],
            net_flux=net_flux_lists[i],
            plane_loc1=z_min,
            plane_loc2=z_max,
            z_min=z_min,
            z_max=z_max
        )

if __name__ == "__main__":
    main()