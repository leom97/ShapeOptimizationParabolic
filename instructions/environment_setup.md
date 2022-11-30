# Instructions for environment setup

The following instructions are tested on Ubuntu 22.04 and Python 3.9. We used anaconda throughout, where a pre-packaged version of FEniCS is available.

Perform the following steps to set up the environment:
- change directory into this position (e.g. ` cd ./instructions`)
- create a virtual environment: `conda create --name thesis_env python==3.9`, take note of the installation location
- activate it: `conda activate thesis_env`
- run `source install_packages.sh` to install the required packages. This will require some interaction with the command line, and some time
- run `source apply_patches.sh` to suitably patch the already installed packages (*): this requires the knowledge of the location of the conda environment

(*) This will apply the commit https://bitbucket.org/fenics-project/ffc/pull-requests/98 to the file tools.py. It will also add two optimization algorithms to moola, together with the possibility of interrupting the optimization by pressing CTRL + C.

You can move on to `../code/getting_started.md` for an overview of the code and some indications on how to run it.