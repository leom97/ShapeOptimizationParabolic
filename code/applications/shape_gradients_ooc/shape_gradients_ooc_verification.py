"""
This script is devoted to verifying the estimates for the shape gradients of section 3.2.
It serves to generate the experiments of section 4.2.2.

We test the shape gradients with coarse and fine vector fields, and refine the spatio/temporal discretization
The OOC is computed as seen in section 4.2.2. It is expected to asymptotically approach 2 (because of the chosen
coupling between spatial and temporal discretization parameters).

Note: the run for the implicit euler method is very computationally expensive, consider downscaling the experiment

Head to configuration.py to set up the experiment (or leave as is to reproduce the results from the thesis).
This script depends also on code/tools/ooc_verification.py

Then, run the script normally through e.g. the command line.

"""

# %% Imports

from dolfin import *
import logging
import numpy as np
from tqdm import tqdm

from utilities.overloads import radial_displacement, backend_radial_displacement
from utilities.shape_optimization import ShapeOptimizationProblem
from utilities.ooc_verification import get_spiky_radial_function, get_assembled_shape_gradient, W1i_norm, \
    string_to_vector_field
from applications.shape_gradients_ooc.configuration import _f, _g

# %% Setting log and global parameters

parameters['allow_extrapolation'] = True  # I want a function to be taken from a mesh to a slightly different one
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M')
set_log_level(LogLevel.ERROR)
runs_path = "./"  # path to the directory containing this .py file, try to change to absolute path in case of problems
mesh_path = "./mesh_data"

# %% Set-ups

# Please head to configuration.py to tweak settings and data

try:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("configuration", runs_path + "configuration.py")
    pd = importlib.util.module_from_spec(spec)
    sys.modules["configuration"] = pd
    spec.loader.exec_module(pd)

    geometry_dict = pd.geometry_dict
    pde_dict = pd.pde_dict
    cost_functional_dict = pd.cost_functional_dict
    experiment_dict = pd.experiment_dict
    smooth_displacements_dict = pd.smooth_displacements_dict
    manual = pd.manual
    h_tentative = pd.h_tentative
except:
    raise Exception("Couldn't load configuration file from path")

# the data for problem 2.1.1
f = _f(t=0)
g = _g(t=0)

if not manual:
    from dolfin_adjoint import *

if pde_dict["ode_scheme"] == "crank_nicolson":
    dt_power = 1
else:
    dt_power = 2
    experiment_dict["N_it"] = 5  # downscaling implicit euler's experiment

dt_multiplier = experiment_dict["dt_multiplier"]

h_actual = np.array([])
dt_actual = np.array([])

# Displacement parameters
amp = [0.1, 0.2]
spikes = range(10)

results = []

# %% Run

for h, k in zip(h_tentative, range(len(h_tentative))):

    logging.info("#######################################")
    logging.info(f"{(k + 1)}/{len(h_tentative)}")
    logging.info("#######################################")

    geometry_dict["domain"]["resolution"] = h
    problem = ShapeOptimizationProblem()  # let's create a problem class for easily holding various useful variables
    problem.problem_folder = mesh_path
    M2 = problem.create_optimal_geometry(geometry_dict)

    h_actual = np.append(h_actual, problem.exact_domain.mesh.hmax())

    V_vol = FunctionSpace(problem.exact_domain.mesh, "CG", 1)  # scalar, linear FEM on the volume
    V_sph = FunctionSpace(problem.exact_sphere.mesh, "CG", 1)  # scalar, linear FEM on the unit sphere
    V_def = VectorFunctionSpace(problem.exact_domain.mesh, "CG", 1)  # vector (2D), linear FEM on the volume
    Q = VectorFunctionSpace(problem.exact_domain.mesh, 'DG', 0)  # space for gradients of V_def functions

    pde_dict["N_steps"] = int(np.ceil(dt_multiplier * pde_dict["T"] / (h_actual[-1] ** (dt_power))))
    dt_actual = np.append(dt_actual, pde_dict["T"] / pde_dict["N_steps"])

    dj = get_assembled_shape_gradient(V_sph, V_def, V_vol, M2, problem.exact_domain, pde_dict,
                                      cost_functional_dict, f, g, manual=manual)
    evaluations = {"dj": [], "norms": []}

    logging.info("Evaluating the gradient")
    with tqdm(total=len(amp) * len(spikes)) as pbar:
        for A in amp:
            for s in spikes:
                dq = get_spiky_radial_function(problem, V_sph, A, s)

                if not manual:
                    W = radial_displacement(dq, M2, V_def)
                    dj_dq = dq._ad_dot(dj)
                else:
                    W = backend_radial_displacement(dq, M2, V_def)
                    dj_dq = np.dot(dj[:], W.vector()[:])

                evaluations["dj"].append(dj_dq)
                evaluations["norms"].append(W1i_norm(W, Q, problem, V_vol))

                pbar.update(1)
    if manual:
        for i in tqdm(range(len(smooth_displacements_dict["x"]))):
            W = string_to_vector_field(smooth_displacements_dict["x"][i], smooth_displacements_dict["y"][i], problem)

            dj_dq = np.dot(dj[:], W.vector()[:])

            evaluations["dj"].append(dj_dq)
            evaluations["norms"].append(W1i_norm(W, Q, problem, V_vol))

            pbar.update(1)

    evaluations["dj"] = np.array(evaluations["dj"])
    evaluations["norms"] = np.array(evaluations["norms"])

    results.append(evaluations)

    if not manual:
        tape = get_working_tape()
        tape.clear_tape()

# %% Post-processing

# Let's create a matrix of gradient evaluations, and of gradient norms
dj = []
norms = []

for e in results:
    dj.append(e["dj"])
    norms.append(e["norms"])

dj = np.array(dj)
norms = np.array(norms)
dual_errors = np.max(np.abs(dj - dj[-1, :]) / norms, axis=1)
ooc = np.log(dual_errors[1:] / dual_errors[:-1]) / np.log(h_actual[1:] / h_actual[:-1])

logging.info(f"OOCs are {ooc}")

# %% Gather results (to save, e.g. by means of pickles)

results_dict = {
    "pde_dict": pde_dict,
    "geometry_dict": geometry_dict,
    "experiment_dict": experiment_dict,
    "dj": dj,
    "norms": norms,
    "dual_errors": dual_errors,
    "ooc": ooc
}
