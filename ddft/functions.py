import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import h5py
import glob
import numpy as np


def plot_movie(results_path, md_path):
    file_pattern = f"{md_path}/collection.*.h5"
    md_file_list = sorted(glob.glob(file_pattern))

    with h5py.File(md_file_list[0], 'r') as f:
        V = np.array(f['V']).flatten()
        r = np.array(f['r']).flatten()

    ddft_file_list = sorted(glob.glob(f"{results_path}/rho*.npy"))
    init_rho = np.load(ddft_file_list[0])
    ddft_file_list.pop(-1)

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.size'] = 18
    fig, ax = plt.subplots()
    ax.set_xlim(10, 20)
    ax.set_ylim(0, 2)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.plot(r, init_rho, color='red', ls='--', alpha=.3, label='Initial')
    line, = ax.plot(r, np.load(ddft_file_list[0]), lw=2, color='C3', label='DDFT')
    ax.set_ylabel('Density')

    ax2 = ax.twinx()
    ax2.set_ylabel('External potential', color="#D1D3D4")
    ax2.plot(r, V, label="Vext", color="#D1D3D4")
    ax2.tick_params(axis='y', labelcolor="#D1D3D4")
    ax2.set_xlim(0, 10)

    line2, = ax.plot(r, init_rho, lw=2, color='C1', label='MD')
    ax.legend()

    N = min(len(ddft_file_list), len(md_file_list))

    def update(i):
        if i < len(ddft_file_list):
            line.set_ydata(np.load(ddft_file_list[i]))
        if i < len(md_file_list):
            with h5py.File(md_file_list[i], 'r') as f:
                n = np.array(f['n']).flatten()
                line2.set_ydata(n)
        return line, line2,

    anim = FuncAnimation(fig, update, frames=range(N), blit=True, repeat=False)
    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save(f'{results_path}/0plot_movie.mp4', writer = FFwriter)
    fig.savefig(f'{results_path}/0plot.png', bbox_inches='tight', pad_inches=0, transparent=True)


def plot_movie_open(results_path, md_path, init_rho):
    file_pattern = f"{md_path}/collection.*.h5"
    md_file_list = sorted(glob.glob(file_pattern))

    with h5py.File(md_file_list[0], 'r') as f:
        init_rho = np.array(f['n']).flatten()
        V = np.array(f['V']).flatten()
        r = np.array(f['r']).flatten()

    ddft_file_list = sorted(glob.glob(f"{results_path}/rho*.npy"))

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.plot(r, init_rho, color='red', ls='--', alpha=.3, label='Initial')
    line, = ax.plot(r, np.load(ddft_file_list[0]), lw=2, color='C3', label='DDFT')

    ax2 = ax.twinx()
    ax2.set_ylabel('External potential', color="#D1D3D4")
    ax2.plot(r, V, label="Vext", color="#D1D3D4")
    ax2.tick_params(axis='y', labelcolor="#D1D3D4")
    ax2.set_xlim(0, 10)

    line2, = ax.plot(r, init_rho, lw=2, color='C1', label='MD')
    ax.legend()

    N = min(len(ddft_file_list), len(md_file_list))

    def update(i):
        if i < len(ddft_file_list):
            line.set_ydata(np.load(ddft_file_list[i]))
        if i < len(md_file_list):
            with h5py.File(md_file_list[i], 'r') as f:
                n = np.array(f['n']).flatten()
                line2.set_ydata(n)
        return line, line2,

    anim = FuncAnimation(fig, update, frames=range(N), blit=True, repeat=False)
    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save(f'{results_path}/0plot_movie.mp4', writer = FFwriter)
    fig.savefig(f'{results_path}/0plot.png')