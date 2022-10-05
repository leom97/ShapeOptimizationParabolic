"""
Here we generate some controlled perturbation fields, that are smooth on not so smooth
We will have two versions:
- a coarse displacement sphere, from which we will transfer the basis functions -> only W^1,\infty volume displacements
- a fine displacement sphere and Fourier like radial functions -> smooth displacements
"""

from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt
import logging
import numpy as np
from tqdm import tqdm

from utilities.shape_optimization import ShapeOptimizationProblem
from utilities.meshing import sea_urchin
from utilities.overloads import radial_displacement
from utilities.shape_optimization import ShapeOptimizationProblem
from utilities.pdes import HeatEquation

# %% Setting log and global parameters

# The refining algortithm is no longer necessary, but let's keep it
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
parameters['allow_extrapolation'] = True  # I want a function to be taken from a mesh to a slightly different one

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%H:%M')

set_log_level(LogLevel.ERROR)

runs_path = "/home/leonardo_mutti/PycharmProjects/masters_thesis/code/examples/shape_gradients_ooc/"


# %% Useful functions

# for fr in range(10):
#     for amp in [0.05, 0.1, 0.2]:

def get_displacement(problem, V_sph, V_def, amp, fr):
    with stop_annotating():
        dq = Function(V_sph)
        circle_coords = problem.exact_sphere.mesh.coordinates()[dof_to_vertex_map(V_sph), :]
        dq.vector()[:] = sea_urchin(circle_coords, shift=0, amplitude=amp, omega=fr)

        # W = radial_displacement(dq, problem.M2_exact, V_def)

    return dq


def create_cost_functional(V_sph, V_def, V_vol, M2, exact_domain, exact_pde_dict, cost_functional_dict, u_D, u_N):
    # MUST MOVE MESH TO GET THE CONTROL!
    q = Function(V_sph)  # null displacement (but doesn't need to be so)
    W = radial_displacement(q, M2, V_def)
    ALE.move(exact_domain.mesh, W)

    # Generic set-up
    v_equation = HeatEquation(efficient=False)
    v_equation.interpolate_data = True
    w_equation = HeatEquation(efficient=False)
    w_equation.interpolate_data = True

    v_equation.set_ODE_scheme(exact_pde_dict["ode_scheme"])
    v_equation.verbose = True
    v_equation.set_time_discretization(exact_pde_dict["T"],
                                       N_steps=int(exact_pde_dict["N_steps"]),
                                       relevant_mesh_size=exact_domain.mesh.hmax())

    w_equation.set_ODE_scheme(exact_pde_dict["ode_scheme"])
    w_equation.verbose = True
    w_equation.set_time_discretization(exact_pde_dict["T"],
                                       N_steps=int(exact_pde_dict["N_steps"]),
                                       relevant_mesh_size=exact_domain.mesh.hmax())

    # Pre-assembling the boundary conditions
    if exact_pde_dict["ode_scheme"] == "crank_nicolson":
        times = (v_equation.times[1:] + v_equation.times[:-1]) / 2
    elif exact_pde_dict["ode_scheme"] in ["implicit_euler", "implicit_explicit_euler"]:
        times = v_equation.times[1:]
    else:
        raise Exception("No matching ode scheme")

    # PDEs definitions
    u0 = Function(V_vol)  # zero initial condition
    u_D_inner = Constant(0.0)  # zero inner BC

    # Dirichlet state
    v_equation.set_mesh(exact_domain.mesh, exact_domain.facet_function, V_vol, order=1)
    v_equation.set_PDE_data(u0, marker_dirichlet=[2, 3], dirichlet_BC=[u_D, u_D_inner])

    # Dirichlet-Neumann state
    w_equation.set_mesh(exact_domain.mesh, exact_domain.facet_function, V_vol, order=1)
    w_equation.set_PDE_data(u0, marker_dirichlet=[3], marker_neumann=[2], dirichlet_BC=[u_D_inner], neumann_BC=[u_N])

    # The cost functional
    v_equation.solve()
    w_equation.solve()

    J = 0
    final_smoothing = eval(cost_functional_dict["final_smoothing_lambda"])

    # A check on the cost functional discretization type
    if w_equation.ode_scheme == "implicit_euler":
        cost_functional_dict["discretization"] = "rectangle_end"
    elif w_equation.ode_scheme == "crank_nicolson":
        cost_functional_dict["discretization"] = "trapezoidal"

    it = zip(v_equation.solution_list[1:],
             w_equation.solution_list[1:],
             v_equation.dts,
             v_equation.times[1:],
             range(len(v_equation.dts)))

    # Building cost functional
    dx = Measure("dx", domain=exact_domain.mesh)
    fs = 1.0  # final_smoothing
    for (v, w, dt, t, i) in it:
        fs = final_smoothing(exact_pde_dict["T"] - t)
        if fs > DOLFIN_EPS:
            if cost_functional_dict["discretization"] == "trapezoidal" and i == len(
                    v_equation.dts) - 1:
                J += assemble(.25 * dt * fs * (v - w) ** 2 * dx)
            else:
                J += assemble(.5 * dt * fs * (v - w) ** 2 * dx)

    return ReducedFunctional(J, Control(q))


def W1i_norm(W, Q, problem):
    q = TestFunction(Q)

    V1 = Function(V_vol)
    V2 = Function(V_vol)

    V1.vector()[:] = W.vector()[::2]
    V2.vector()[:] = W.vector()[1::2]

    MinvL = (1 / CellVolume(problem.exact_domain.mesh)) * inner(grad(V1), q) * dx(problem.exact_domain.mesh)
    x = assemble(MinvL)
    DV1 = Function(Q, x)

    MinvL = (1 / CellVolume(problem.exact_domain.mesh)) * inner(grad(V2), q) * dx(problem.exact_domain.mesh)
    x = assemble(MinvL)
    DV2 = Function(Q, x)

    return np.concatenate((W.vector()[:], DV1.vector()[:], DV2.vector()[:])).max()


# %% Data

exact_geometry_dict = {  # note, the geometry here is finer than in hourglass_IE
    "domain": {"type": "annulus", "resolution": None, "ext_refinement": 1.0, "int_refinement": 1.0, "inner_radius": 1,
               "outer_radius": 2,
               "center": np.array([0, 0]), "reload_xdmf": False},
    "sphere": {"dimension": 2, "resolution": 0.5},
    "q_ex_lambda": 'lambda x: 0'
    # circle_coords are in dof order
}

exact_pde_dict = {
    "ode_scheme": "implicit_euler",
    "u_N": "Expression('sin(3*t)*(pow(x[0],2)-cos(4*x[1]))', t=0, a=.05, p = np.pi, degree=5)",
    "marker_dirichlet": [3],
    "marker_neumann": [2],
    "T": 1,
    "N_steps": None  # a lot of time steps for the exact equation
}

cost_functional_dict = {
    "final_smoothing_lambda": "lambda t: 1",  # exp(-0.05/pow(t,2)) if t > DOLFIN_EPS else 0.0
    "discretization": None,
    "H1_smoothing": 0
}

experiment_dict = {
    "N_it": 6,
    "dt_multiplier": 1,
    "start_index": 1,
}


# Pde data
class _u_D(UserExpression):
    def __init__(self, t, **kwargs):
        super().__init__(self, **kwargs)
        self.t = t

    def eval(self, values, x):
        values[0] = self.t ** 3 * np.cos(5 * self.t) * (x[0] ** 2 + x[1])

    def value_shape(self):
        return ()


class _u_N(UserExpression):
    def __init__(self, t, **kwargs):
        super().__init__(self, **kwargs)
        self.t = t

    def eval(self, values, x):
        values[0] = self.t ** 2 * np.sin(5 * self.t) * np.sin(x[0])

    def value_shape(self):
        return ()


u_D = _u_D(t=0)
u_N = _u_N(t=0)

# %% Setting up the tests

h_tentative = 1 / (2 ** np.arange(0, experiment_dict["N_it"]))
h_tentative = h_tentative[experiment_dict["start_index"]:]
dt_multiplier = experiment_dict["dt_multiplier"]

if exact_pde_dict["ode_scheme"] == "crank_nicolson":
    dt_power = 1  # ie dt = 1/dt_multiplier * h^dt_power
else:
    dt_power = 2

h_actual = np.array([])
dt_actual = np.array([])

# Displacement parameters
amp = [0.1, 0.2]
fr = range(10)

path = "/home/leonardo_mutti/PycharmProjects/masters_thesis/code/examples/shape_gradients_ooc/mesh_data"
results = []

# %% Run

for h, k in zip(h_tentative, range(len(h_tentative))):

    logging.info("#######################################")
    logging.info(f"{(k + 1)}/{len(h_tentative)}")
    logging.info("#######################################")

    exact_geometry_dict["domain"]["resolution"] = h

    # Cost functional

    problem = ShapeOptimizationProblem()  # used for the methods it has, and to hold some interesting global variables
    problem.problem_folder = path
    M2 = problem.create_optimal_geometry(exact_geometry_dict)

    h_actual = np.append(h_actual, problem.exact_domain.mesh.hmax())

    V_vol = FunctionSpace(problem.exact_domain.mesh, "CG", 1)
    V_sph = FunctionSpace(problem.exact_sphere.mesh, "CG", 1)
    V_def = VectorFunctionSpace(problem.exact_domain.mesh, "CG", 1)

    exact_pde_dict["N_steps"] = int(np.ceil(dt_multiplier * exact_pde_dict["T"] / (h_actual[-1] ** (dt_power))))
    dt_actual = np.append(dt_actual, exact_pde_dict["T"] / exact_pde_dict["N_steps"])

    j = create_cost_functional(V_sph, V_def, V_vol, M2, problem.exact_domain, exact_pde_dict, cost_functional_dict, u_D,
                               u_N)

    # h = Function(V_sph)
    # q = Function(V_sph)
    # h.vector()[:] = .1
    # td = taylor_to_dict(j, q, h)
    #
    # logging.info("Results of Taylor tests: rates and residuals")
    # for i in range(3):
    #     logging.info(td["R" + str(i)]["Rate"])
    #     logging.info(td["R" + str(i)]["Residual"])
    #
    # 16:46 INFO     Results of Taylor tests: rates and residuals
    # 16:46 INFO     [1.0014579210501169, 1.0007289238939692, 1.0003644527649274]
    # 16:46 INFO     [3.5869399033466864e-05, 1.7916584691357595e-05, 8.953767297897874e-06, 4.475752844096403e-06]
    # 16:46 INFO     [2.0007775258947604, 2.000388770320626, 2.0001944153256472]
    # 16:46 INFO     [7.242028184078494e-08, 1.8095315544555717e-08, 4.522609991354528e-09, 1.1305001431434575e-09]
    # 16:46 INFO     [3.0002334059834626, 2.9999109802764035, 2.9997542656597145]
    # 16:46 INFO     [7.802670261860501e-11, 9.751760014133906e-12, 1.2190452191324563e-12, 1.5240660960858877e-13]

    # Testing with a vector field

    Q = VectorFunctionSpace(problem.exact_domain.mesh, 'DG', 0)
    evaluations = {"dj": [], "norms": []}

    logging.info("Evaluating the gradient")
    with tqdm(total=len(amp) * len(fr)) as pbar:
        for a in amp:
            for f in fr:
                dq = get_displacement(problem, V_sph, V_def, a, f)
                dj = j.derivative()
                dj_dq = dq._ad_dot(dj)

                W = radial_displacement(dq, M2, V_def)

                evaluations["dj"].append(dj_dq)
                evaluations["norms"].append(W1i_norm(W, Q, problem))

                pbar.update(1)

    evaluations["dj"] = np.array(evaluations["dj"])
    evaluations["norms"] = np.array(evaluations["norms"])

    results.append(evaluations)

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

print(ooc)

# %% Save results

results_dict = {
    "exact_pde_dict": exact_pde_dict,
    "exact_geometry_dict": exact_geometry_dict,
    "experiment_dict": experiment_dict,
    "dj": dj,
    "norms": norms,
    "dual_errors": dual_errors,
    "ooc": ooc
}

import pickle
with open("/home/leonardo_mutti/PycharmProjects/masters_thesis/code/examples/shape_gradients_ooc/results/" + 'IE_no_eta_rough.pickle', 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)