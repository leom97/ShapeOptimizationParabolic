"""
Here we check whether the computed shape derivatives are correct or not
"""

from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt
import logging

from utilities.pdes import TimeExpressionFromList, HeatEquation
from utilities.shape_optimization import ShapeOptimizationProblem

# %% Setting log and global parameters

# The refining algortithm is no longer necessary, but let's keep it
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
parameters['allow_extrapolation'] = True  # I want a function to be taken from a mesh to a slightly different one

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%H:%M')

set_log_level(LogLevel.ERROR)

runs_path = "/home/leonardo_mutti/PycharmProjects/masters_thesis/code/runs/old_runs/"

# %% Functional set-up

problem_name = "sq_oakley_oscillations_test"
problem_path = runs_path + problem_name + "/"

problem = ShapeOptimizationProblem()
problem.initialize_from_data(problem_path, regenerate_exact_data=True)
problem.save_exact_data(problem_path)
_, M2 = problem.create_cost_functional(start_at_optimum=True)
# problem.create_cost_functional(disable_radial_parametrization=True, start_at_optimum=True)    # todo: uncomment me to enable the test for vector fields

# %% The correct shape derivative

dj_exact = problem.j.derivative().vector()

# %% My shape derivative

# 1) flip time in the pde solution and create a time expression from the pde solutions
source_p_list = []
source_q_list = []

for (v, w) in zip(problem.v_equation.solution_list, problem.w_equation.solution_list):
    d = Function(v.function_space())
    md = Function(v.function_space())

    d.vector()[:] = v.vector()[:] - w.vector()[:]
    md.vector()[:] = -v.vector()[:] + w.vector()[:]

    source_p_list.append(md)
    source_q_list.append(d)

source_p = TimeExpressionFromList(0.0, problem.v_equation.times, source_p_list, reverse=True)
source_q = TimeExpressionFromList(0.0, problem.v_equation.times, source_q_list, reverse=True)

# 2) solve the adjoints with IEE
# Dirichlet state
p_equation = HeatEquation()
p_equation.set_mesh(problem.optimization_domain.mesh, problem.optimization_domain.facet_function, problem.V_vol)
p_equation.set_PDE_data(Constant(0.0), marker_dirichlet=[2, 3], source=source_p,
                        dirichlet_BC=[Constant(0.0),
                                      Constant(0.0)])
p_equation.set_ODE_scheme("implicit_explicit_euler")
p_equation.verbose = True
p_equation.set_time_discretization(problem.T, N_steps=problem.optimization_pde_dict["N_steps"],
                                   relevant_mesh_size=problem.exact_domain.mesh.hmax())

# Dirichlet-Neumann state
q_equation = HeatEquation()
q_equation.set_mesh(problem.optimization_domain.mesh, problem.optimization_domain.facet_function, problem.V_vol)
q_equation.set_PDE_data(Constant(0.0), marker_dirichlet=[3], marker_neumann=[2],
                        source=source_q,
                        dirichlet_BC=[Constant(0.0)],
                        neumann_BC=[Constant(0.0)])
q_equation.set_ODE_scheme("implicit_explicit_euler")
q_equation.verbose = True
q_equation.set_time_discretization(problem.T, N_steps=problem.optimization_pde_dict["N_steps"],
                                   relevant_mesh_size=problem.exact_domain.mesh.hmax())

p_equation.solve()
q_equation.solve()

# 3) flip their times
p_equation.solution_list.reverse()
q_equation.solution_list.reverse()

# %% 4) assemble shape derivative
dt = Constant(q_equation.dts[0])
h = TestFunction(problem.V_def)
I = Identity(problem.optimization_domain.mesh.ufl_cell().geometric_dimension())
A = div(h) * I - grad(h) - grad(h).T

# part due to cost functional
cost_part = div(h) * Constant(0.0) * dx(
    problem.optimization_domain.mesh)  # the div term is not to have arity mismatches
for (v, w) in zip(problem.v_equation.solution_list[1:], problem.w_equation.solution_list[1:]):
    cost_part += Constant(1 / 2) * dt * div(h) * (v - w) ** 2 * dx(problem.optimization_domain.mesh)

# part due to p, v
pv_part = div(h) * Constant(0.0) * dx(problem.optimization_domain.mesh)
for (vj, vjm, pjm) in zip(problem.v_equation.solution_list[1:],
                          problem.v_equation.solution_list[:-1],
                          p_equation.solution_list[:-1]):
    pv_part += ((vj - vjm) / dt * pjm * div(h) + inner(A * grad(vj), grad(pjm))) * dt * dx(
        problem.optimization_domain.mesh)

# part due to q, w
qw_part = div(h) * Constant(0.0) * dx(problem.optimization_domain.mesh)
for (wj, wjm, qjm) in zip(problem.w_equation.solution_list[1:],
                          problem.w_equation.solution_list[:-1],
                          q_equation.solution_list[:-1]):
    qw_part += ((wj - wjm) / dt * qjm * div(h) + inner(A * grad(wj), grad(qjm))) * dt * dx(
        problem.optimization_domain.mesh)

dj_mine = assemble(cost_part + pv_part + qw_part)

import numpy as np

# d = dj_mine - dj_exact    # todo: uncomment me to enable the test for vector fields
# DirichletBC(problem.V_def, Constant([0, 0]), problem.optimization_domain.facet_function, 2).apply(d)  # todo: uncomment me to enable the test for vector fields

d = dj_mine @ M2 - dj_exact

logging.info(f"Error of shape gradients: {np.max(np.abs(d))}")
