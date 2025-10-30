import matplotlib.pyplot as plt
import h5py
import numpy as np
import glob
from matplotlib.animation import FuncAnimation
import os
import multiprocessing as mp
from functools import partial

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def plot(path: str) -> None:
    file_pattern = f"{path}" + '/collection.*.h5'
    file_list = sorted(glob.glob(file_pattern))

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Density')
    ax2.set_ylabel('External potential', color="#D1D3D4")
    ax2.tick_params(axis='y', labelcolor="#D1D3D4")

    with h5py.File(file_list[-1], "r") as fp:
        r = np.array(fp["r"])
        n = np.array(fp["n"])
        V = np.array(fp["V"])

    with h5py.File(file_list[0], "r") as fp:
        initial_n = np.array(fp["n"])

    ax.plot(r, initial_n, color='red', ls='--', alpha=.3, label='Initial')
    ax.plot(r, n, lw=2, color="red", label="MD")
    ax2.plot(r, V, label="Vext", color="#D1D3D4")
    ax.legend()
    plt.savefig(f"{path}/0plot.png")


def plot_movie(path: str) -> None:
    file_pattern = f"{path}" + '/collection.*.h5'
    file_list = sorted(glob.glob(file_pattern))

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Density')
    ax2.set_ylabel('External potential', color="#D1D3D4")
    ax2.tick_params(axis='y', labelcolor="#D1D3D4")

    with h5py.File(file_list[0], "r") as fp:
        r = np.array(fp["r"])
        n = np.array(fp["n"])
        V = np.array(fp["V"])

    ax.plot(r, n, color='red', ls='--', alpha=.3, label='Initial')
    line, = ax.plot(r, n, lw=2, color="red", label="MD")
    line2, = ax2.plot(r, V, label="Vext", color="#D1D3D4")
    ax.legend()

    def update(frame):
        with h5py.File(frame, "r") as fp:
            n = np.array(fp["n"])
            V = np.array(fp["V"])

        line.set_ydata(n)
        line2.set_ydata(V)

        return line, line2

    ani = FuncAnimation(fig, update, frames=file_list, blit=False, repeat=False)
    ani.save(f"{path}/0plot_movie.mp4", fps=10, extra_args=['-vcodec', 'libx264'])
    plt.savefig(f"{path}/0plot.png")


def avg_results(path: str, N: int, M: int) -> None:
    '''
    Average the density results of N simulations of M steps each
    '''
    
    # Save average M-step density profile
    save_path = f"{path}/avg_{N}"
    os.makedirs(save_path, exist_ok=True)
    avg_n = {}
    for i in range(0, M):
        avg_n[i] = np.zeros((1000, 1))

    # Accumulate N density profiles
    for i in range(0, N):
        file_pattern = f"{path}/trials/{i}/collection.*.h5"
        file_list = sorted(glob.glob(file_pattern))

        for j, file in enumerate(file_list):
            with h5py.File(file, "r") as fp:
                avg_n[j] += np.array(fp["n"])

    with h5py.File(file_list[0], "r") as fp:
        r = np.array(fp["r"])
        V = np.array(fp["V"])

    for i in range(0, M):
        n = avg_n[i] / N
        filename = f"{save_path}/collection.{i:03d}.h5"
        with h5py.File(filename, "w") as fp:
            fp["n"] = n
            fp["r"] = r
            fp["V"] = V



def avg_results_open(path: str, N: int, M: int, name: str) -> None:
    '''
    Average the density results of N simulations of M steps each
    '''
    
    # Save average M-step density profile
    save_path = f"{path}/avg_{N}"
    os.makedirs(save_path, exist_ok=True)
    avg_n = {}
    for i in range(0, M):
        avg_n[i] = np.zeros((320, 1))

    # Accumulate N density profiles
    for i in range(0, N):
        file_pattern = f"{path}/trials/{i}/{name}.*.h5"
        file_list = sorted(glob.glob(file_pattern))

        for j, file in enumerate(file_list):
            with h5py.File(file, "r") as fp:
                avg_n[j] += np.array(fp["n"])

    with h5py.File(file_list[0], "r") as fp:
        r = np.array(fp["r"])
        V = np.array(fp["V"])

    for i in range(0, M):
        n = avg_n[i] / N
        filename = f"{save_path}/{name}.{i:05d}.h5"
        with h5py.File(filename, "w") as fp:
            fp["n"] = n
            fp["r"] = r
            fp["V"] = V


def process_simulation(sim_index: int, base_path: str, M: int) -> np.ndarray:
    """Process a single simulation and return its density arrays"""
    result = np.zeros((M, 320, 1))
    file_pattern = f"{base_path}/trials/{sim_index}/collection.*.h5"
    file_list = sorted(glob.glob(file_pattern))
    
    for j, file in enumerate(file_list):
        try:
            with h5py.File(file, "r") as fp:
                result[j] = np.array(fp["n"])

        except OSError as e:
            print(f"Failed to open {file}: {e}")
            result[j] = result[j-1]

    return result


def avg_results_parallel(path: str, N: int, M: int) -> None:
    '''
    Average the density results of N simulations of M steps each using parallel processing
    '''
    # Save average M-step density profile
    save_path = f"{path}/avg_{N}"
    os.makedirs(save_path, exist_ok=True)

    # Read first file to get r and V
    first_file = f"{path}/trials/4999/collection.000.h5"
    with h5py.File(first_file, "r") as fp:
        r = np.array(fp["r"])
        V = np.array(fp["V"])

    # Initialize the result array
    avg_n = np.zeros((M, 320, 1))

    # Set up parallel processing
    num_cores = mp.cpu_count() - 1  # Leave one core free
    process_sim = partial(process_simulation, base_path=path, M=M)

    # Process simulations in parallel
    with mp.Pool(num_cores) as pool:
        results = pool.map(process_sim, range(N))
    
    # Sum all results
    for result in results:
        avg_n += result

    # Calculate average
    avg_n /= N

    # Save results
    for i in range(M):
        filename = f"{save_path}/collection.{i:03d}.h5"
        with h5py.File(filename, "w") as fp:
            fp["n"] = avg_n[i]
            fp["r"] = r
            fp["V"] = V


if __name__ == "__main__":
    avg_results("results/2025-05-10-test", 10, 500)
    plot_movie("results/2025-05-10-test/avg_10")