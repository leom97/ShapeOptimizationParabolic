# Code for my master's thesis: finite element-based shape optimization with parabolic PDEs

Here I have collected the code I utilized to run the simulations that are reported in my master's thesis.

## Functionalities

This repository contains a Python implementation of a shape optimization pipeline based on Fenics and dolfin-afjoint.

The concrete problem being solved is addressed in `material/manuscript.pdf`. In a nutshell, it involves:
- PDE-constrained shape optimization, with heat equations as constraints
- radial mesh movement
- a BFGS optimization algorithm

With this code you can:
- solve shape optimization test cases
- analyze the approximation error of shape gradients

## Getting started

To obtain the code, perform the following instructions (on Linux):
- clone the GitHub repository: `git clone https://github.com/leom97/ShapeOptimizationParabolic.git`
- change the current directory: `cd ShapeOptimizationParabolic`
- head now to `intructions/environment_setup.md` for instructions on how to set up your enviroment

In the setup instructions, additional details on how to utilize the code are given.
