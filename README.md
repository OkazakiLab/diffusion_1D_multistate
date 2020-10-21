# diffusion_1D_multistate
Estimates 1D diffusion model with chemical-state dependent free energy profile from simulation/single-molecule trajectories.
## Installation
First, clone or download the repository,
```
git clone https://github.com/OkazakiLab/diffusion_1D_multistate.git
cd diffusion_1D_multistate/
```
Then, go to src/ directory and compile with a C++ compiler,
```
cd src/
g++ -o ../diffusion_1D_multistate -O2 -std=c++11 diffusion_1D_multistate.cpp
```
## To run diffusion_1D_multistate
```
./diffusion_1D_multistate traj-file1 traj-file2 ...
```
Some parameters can be set in the 'para' file.
## Example
'run.bash' runs a test analysis of 30 simulation trajectories in traj/ directory,
```
bash run.bash
```
## Parameters
A set of parameters that can be set in 'para',
```
ctrl  3
ctrl_sa  0
replicate  1
nmc  1000
nmc_update_stop  500
nmc_sa  0
k_sa  0.02
dx  0.1
xlim_low  -0.35
dt0  0.000001
lagstep  400
dt  0.0004
prior_dv  6.0
prior_sigma2  0.01
mcrate  1
xswitch  0.8 1.2
ratef_switch  100.0
rateb_switch  1.0
xshift  1.0
```

## Reference
K. Okazaki, A. Nakamura and R. Iino “Chemical-State-Dependent Free Energy Profile from Single-Molecule Trajectories of Biomolecular Motors: Application to Processive Chitinase”, J. Phys. Chem. B 124, 30, 6475−6487 (2020)
