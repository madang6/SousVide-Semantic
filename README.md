# SOUS VIDE

## Installation
1) Clone repository and load the submodules.
```
git clone https://github.com/StanfordMSL/SousVide.git
git submodule update --recursive --init
```
2) Build ACADOS locally.
```
# Navigate to acados folder
cd <repository-path>/SousVide/FiGS/acados/

# Compile
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4

# Add acados paths to bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
export ACADOS_SOURCE_DIR="<acados_root>"
```
3) Set up conda environment (in the main directory)
```
# Navigate to environment config location
cd <repository-path>/SousVide/

# Create and activate
conda env create -f environment_x86.yml
conda activate sv-env
```
4) Download Example GSplats
```
# Navigate to gsplats parent folder
cd <repository-path>/SousVide/

# Use gdown to download
gdown --folder https://drive.google.com/drive/folders/1Q3Jxt08MUev_jWzHjpdltze7X4VArsvA?usp=drive_link --remaining-ok

# Alternatively, you can download the zip-ed file below and unpack the contents (capture and workspace) into the gsplats folder
https://drive.google.com/file/d/1kW5dzsfD3rbRA3RIQDyJPG6_UJaO9ALP/view
```

## Run SOUS VIDE Examples
Check out the notebook examples in the notebooks folder:
  1. <b>figs_examples</b>: Example code for generating GSplats and executing trajectories within them (using FiGS).
  2. <b>sv_shakedown</b>: Use this notebook to verify all components before running other SV notebooks. It does not produce a usable policy
  3. <b>sv_robustness</b>: Produces the policies used in Section VI.A and VI.B.
  4. <b>sv_extended</b>: Produces the policy for the extended trajectory in Section VI.C.
  5. <b>sv_cluttered</b>: Produces the policy for the cluttered environment trajectory in Section VI.C.