# Code basic usage

This file provides a brief overview of the code usage.

## `shape_optimization_main.py`

Let us start from `shape_optimization_main.py`. This is the file that launches a shape optimization run.

To do so, a configuration file is needed, where details about the geometry, PDE data etc. are provided. 

A sample version of such a configuration file is `./runs/exemplary_run/problem_data.py`, it corresponds to a basic shape optimization example (see the other runs for more examples). It is required that the name of this file is `problem_data.py`.

The sample version is commented, to illustrate the various changes that are possible from this baseline.

To run shape optimization the user should therefore:

- create a run folder inside e.g. `./runs`, with the name of the run
- create the `problem_data.py` inside this run folder
- make sure the variables `run_name` and `runs_path` in `shape_optimization_main.py` are correctly set. They come already configured for the basic example
- run `shape_optimization_main.py`. We strongly recommend doing this through an IDE (we used PyCharm), for a better visualization of the plots.  The default version will run the basic example

- For caching and reloading data, follow the instructions in `shape_optimization_main.py`

Here is what will happen (see the thesis for additional details):
- the exact domain (the one to be reconstructed) is generated and meshed (or a pre-built mesh is loaded)
- synthetic data is produced (or reloaded)
- the cost functional from dolfin-adjoint is set up
- shape optimization starts: in a window, the current shape of the optimization domain is visualized
- to interrupt the process before convergence/the maximum number of iterations is reached, one can press CTRL + C
- in any case, at the end of the optimization process, the results are visualized, and several plots are saved to file

## `applications/shape_gradients_ooc/shape_gradients_ooc_verification.py`

From this file we can test the convergence behaviour of the shape gradient.

A (documented) configuration file is present here as well. The user should therefore:

- suitably tweak `applications/shape_gradients_ooc/configuration.py`
- run `applications/shape_gradients_ooc/shape_gradients_ooc_verification.py`

The computed orders of convergence are printed to the console. See the thesis for additional details.

