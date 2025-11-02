### Learned Free-Energy Functionals from Pair-Correlation Matching for Dynamical Density Functional Theory

[PR-E article](https://doi.org/10.1103/22fd-ykkb) | [Preprint](https://arxiv.org/abs/2505.09543) | [Data](10.5281/zenodo.17116150)

DDFT and BD simulation code to reproduce the results in the PR-E article. Pre-computed simulation data is available on [Zenodo](10.5281/zenodo.17116150).

### Setup environment
```bash
mamba env create -f environment.yml # or conda
mamba activate neural-ddft
```

### Brownian dynamics simulations
```bash
cd bd
python lj_closed.py --collection_steps=400 --equilibration_steps=200 --trials=100 --gamma=8 --dt=0.001
python lj_open.py --collection_steps=1500 --equilibration_steps=200 --trials=100 --gamma=10 --dt=0.001
```

5000 trials were used for the closed system and 500 trials were used for the open system. The external potential is set inside the script. Note that steps here refers to the number of cycles, and each cycle consists of 10 and 100 steps for the closed and open system, respectively.

### DDFT simulations
```bash
cd ddft
python ddft_neural_closed.py --md_path=./data/g1-closed-dz100/avg_5000
python ddft_neural_open.py --md_path=./data/irmof1-open-dz32/avg_500
```

Use `--type=c1` to use the c1 model and the `ddft_fmt.py` script to use the FMT model. Note that the closed script uses dz=1/100 and dt=0.0001, and the open script uses dz=1/32 and dt=0.001. Code to compute the particle flux in the open system experiment, and to generate all plots in the article is available in the `scripts` folder. Paths to the simulated data are set inside the scripts.

### Acknowledgements

[mdext](https://github.com/shankar1729/mdext)<br>
[py-pde](https://github.com/zwicker-group/py-pde)<br>
[neural-free-energy-1d](https://github.com/jacobusdijkman/neural-free-energy-1d/tree/main)<br>
[PyDFTlj](https://github.com/elvissoares/PyDFTlj)
