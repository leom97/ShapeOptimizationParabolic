# Code basic usage

This file goes on a very brief overview of the code usage.

## `shape_optimization_main.py`

Let us start from `shape_optimization_main.py`. This is the file that launches a shape optimization run.

To do so, a configuration file is needed, where details about the geometry, PDE data etc. are provided. 

A sample version of such a configuration file is `./runs/exemplary_run/problem_data.py` (see the other runs for more examples). It is required that the name of this file is `problem_data.py`. It should be placed in the folder of a specific run.

The sample version is commented, to illustrate the various changes that are possible from this baseline.

To run shape optimization the user should therefore:

- create a problem folder inside e.g. `./runs`, with the name of the run
- create the `problem_data.py` inside this folder
- make sure the variables `run_name` and `runs_path` in `shape_optimization_main.py` are correctly set. They come already configured for the basic example
- run `shape_optimization_main.py`. We recommend doing this through an IDE (we used PyCharm), for a better visualization of the plots


For caching and reloading data, follow the instructions in `shape_optimization_main.py`.

## `applications/shape_gradients_ooc/shape_gradients_ooc_verification.py`

From this file we can test the convergence behaviour of the shape gradient.

A (documented) configuration file is present here as well. The user should therefore:

- suitably tweak `applications/shape_gradients_ooc/configuration.py`
- run `applications/shape_gradients_ooc/shape_gradients_ooc_verification.py`

