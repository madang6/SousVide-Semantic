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
cd <repository-path>/SousVide-Semantic/

conda create --name sousvide-semantic-flight -y python=3.10

conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate sousvide-semantic-flight

python -m pip install --upgrade pip

pip install numpy==1.26.4

conda deactivate
```
4) Install Zed SDK and pyzed (Zed Python bindings)
Follow instructions on Zed website for this part, including the bindings.
They can also be installed at /usr/local/zed/get_python_api.py

6) Install pyzed outside of conda and rest of dependencies
```
conda activate sousvide-semantic-flight

python3 -m pip install --no-deps \
--ignore-installed /usr/local/zed/pyzed-5.0-cp310-cp310-linux_aarch64.whl

conda deactivate

conda env update --name sousvide-semantic-flight --file environment.yml

```
Obtain the correct versions of pytorch, torchvision, and torchaudio.
```
pip install torch-2.6.0-cp310-cp310-linux_aarch64.whl
pip install torchvision-0.21.0-cp310-cp310-linux_aarch64.whl
pip install torchaudio-2.6.0-cp310-cp310-linux_aarch64.whl

```
Update ~/.bashrc with the following:
```
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source ~/StanfordMSL/TrajBridge/TrajBridge/install/setup.bash

```
## Run SOUS VIDE Examples
Check out the notebook examples in the notebooks folder:
  1. <b>figs_examples</b>: Example code for generating GSplats and executing trajectories within them (using FiGS).
  2. <b>sv_shakedown</b>: Use this notebook to verify all components before running other SV notebooks. It does not produce a usable policy
  3. <b>sv_robustness</b>: Produces the policies used in Section VI.A and VI.B.
  4. <b>sv_extended</b>: Produces the policy for the extended trajectory in Section VI.C.
  5. <b>sv_cluttered</b>: Produces the policy for the cluttered environment trajectory in Section VI.C.

## Deploy Semantic SOUS VIDE in the Real World
Deploy SOUS VIDE policies on an [MSL Drone](https://github.com/StanfordMSL/TrajBridge/wiki/3.-Drone-Hardware).

ssh into the drone using a computer on the same network.
```
sudo jetson_clocks
sudo micro-xrce-dds-agent serial --dev /dev/ttyTHS1 --baudrate 921600
```
NOTE: If the ttyTHS1 doesn't work, try ttyTHS0

ssh into the drone in a second terminal on the same network.
```
cd ~/StanfordMSL/SousVide-Semantic/notebooks
conda activate sousvide-semantic-flight
./run
```

## APPENDIX
# Installing vscode on Jetson Orin systems without snap:
```
# On the Jetson host
sudo apt update
sudo apt install -y wget gpg apt-transport-https ca-certificates

# import Microsoft’s GPG key
wget -qO- https://packages.microsoft.com/keys/microsoft.asc \
  | gpg --dearmor \
  | sudo tee /usr/share/keyrings/packages.microsoft.gpg > /dev/null

# add the Code repo, specifying arm64
echo "deb [arch=arm64 signed-by=/usr/share/keyrings/packages.microsoft.gpg] \
https://packages.microsoft.com/repos/code stable main" \
  | sudo tee /etc/apt/sources.list.d/vscode.list

# install VS Code
sudo apt update
sudo apt install -y code

# verify
code --version
```
