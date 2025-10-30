### Learned Free-Energy Functionals from Pair-Correlation Matching for Dynamical Density Functional Theory

[PR-E article](https://doi.org/10.1103/22fd-ykkb) | [Preprint](https://arxiv.org/abs/2505.09543) | [Project page](https://karnikram.info/neural-ddft-pcm/)

### Setup environment
```bash
mamba env create -f environment.yml # or conda
mamba activate neural-ddft
```

### Brownian dynamics simulations
```bash
cd bd
python lj_closed.py --collection_steps=400 --equilibration_steps=200 --trials=50 --gamma=8 --dt=0.001
python lj_open.py --collection_steps=500 --equilibration_steps=100 --trials=50 --gamma=10 --dt=0.001
```

### DDFT simulations
```bash
cd ddft
python ddft_neural.py --md_path=./data/irmof1-closed-dz100/avg_5000
```

### Acknowledgements

[mdext](https://github.com/shankar1729/mdext)<br>
[py-pde](https://github.com/zwicker-group/py-pde)<br>
[neural-free-energy-1d](https://github.com/jacobusdijkman/neural-free-energy-1d/tree/main)<br>
[PyDFTlj](https://github.com/elvissoares/PyDFTlj)
