# diffusion_1D_multistate
Estimates 1D diffusion model from simulation/single-molecule trajectories.
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
