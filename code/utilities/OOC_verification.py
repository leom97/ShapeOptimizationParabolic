"""
Helper file for code/applications/shape_gradients_ooc/shape_gradients_ooc_verification.py
"""

# %% Imports

from dolfin import *
import logging
import numpy as np

from utilities.meshing import sea_urchin
from utilities.overloads import radial_displacement, backend_radial_displacement
from utilities.pdes import HeatEquation, TimeExpressionFromList

# %% Setting log and global parameters

parameters['allow_extrapolation'] = True  # I want a function to be taken from a mesh to a slightly different one
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M')
set_log_level(LogLevel.ERROR)


# %% Functions

def get_spiky_radial_function(problem, V_sph, A, s):
    """
    From an amplitute A and a number of spikes, it returns the spherical function sigma, as seen in section 4.2.2.
    In particular, sigma(t) = A cos(st)

    :param V_sph: linear FEM space on the unit sphere
    :param A: amplitude
    :param s: number of spikes
    """

    sigma = Function(V_sph)
    circle_coords = problem.exact_sphere.mesh.coordinates()[dof_to_vertex_map(V_sph), :]
    sigma.vector()[:] = sea_urchin(circle_coords, shift=0, amplitude=A, omega=s)

    return sigma


def get_assembled_shape_gradient(V_sph, V_def, V_vol, M2, exact_domain, exact_pde_dict, cost_functional_dict, f, g,
                                 manual=False):
    """
    This function sets up the cost functional of section 3.1, computes its shape gradient. It is parametrized on
    spherical functions living on V_sph if manual is False, on vector fields in V_def otherwise.

    If manual is False, the shape gradient is computed automatically by dolfin-adjoint, otherwise the implementation is
    manual.

    :return: vector representing the testing of the shape gradient with every basis test function (of either V_sph or
    V_def)
    """

    # Register the control
    q = Function(V_sph)
    if not manual:
        W = radial_displacement(q, M2, V_def)
    else:
        W = backend_radial_displacement(q, M2, V_def)
    ALE.move(exact_domain.mesh, W)

    # Generic set-up for state equations
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

    # PDEs definitions
    u0 = Function(V_vol)  # zero initial condition
    u_D_inner = Constant(0.0)  # zero inner BC

    # Dirichlet state
    v_equation.set_mesh(exact_domain.mesh, exact_domain.facet_function, V_vol, order=1)
    v_equation.set_PDE_data(u0, marker_dirichlet=[2, 3], dirichlet_BC=[f, u_D_inner])

    # Dirichlet-Neumann state
    w_equation.set_mesh(exact_domain.mesh, exact_domain.facet_function, V_vol, order=1)
    w_equation.set_PDE_data(u0, marker_dirichlet=[3], marker_neumann=[2], dirichlet_BC=[u_D_inner], neumann_BC=[g])

    # The cost functional
    v_equation.solve()
    w_equation.solve()

    J = 0
    final_smoothing = eval(cost_functional_dict["final_smoothing_lambda"])

    dx = Measure("dx", domain=exact_domain.mesh)
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

    if manual == False:  # in this case, let us dolfin-adjoint compute the shape gradient for us
        # Building cost functional
        fs = 1.0  # final_smoothing
        for (v, w, dt, t, i) in it:
            fs = final_smoothing(exact_pde_dict["T"] - t)
            if fs > DOLFIN_EPS:
                if cost_functional_dict["discretization"] == "trapezoidal" and i == len(
                        v_equation.dts) - 1:
                    J += assemble(.25 * dt * fs * (v - w) ** 2 * dx)
                else:
                    J += assemble(.5 * dt * fs * (v - w) ** 2 * dx)

        j = ReducedFunctional(J, Control(q))
        return j.derivative()
    else:  # let's do it ourselves

        # the right hand sides for the adjoint equations
        source_p_list = []
        source_q_list = []

        for (v, w, t) in zip(v_equation.solution_list, w_equation.solution_list, v_equation.times):
            fs = final_smoothing(exact_pde_dict["T"] - t)

            d = Function(v.function_space())
            md = Function(v.function_space())

            d.vector()[:] = fs * (v.vector()[:] - w.vector()[:])
            md.vector()[:] = fs * (-v.vector()[:] + w.vector()[:])

            source_p_list.append(md)  # note, this same logic should be good for the current CN implementation too:
            source_q_list.append(d)

        source_p = TimeExpressionFromList(0.0, v_equation.times, source_p_list, reverse=True)
        source_q = TimeExpressionFromList(0.0, v_equation.times, source_q_list, reverse=True)

        adjoint_scheme = "implicit_explicit_euler"
        if exact_pde_dict["ode_scheme"] == "crank_nicolson":
            adjoint_scheme = "crank_nicolson"

        # Adjoint to Dirichlet state
        p_equation = HeatEquation(efficient=False)
        p_equation.interpolate_data = True
        p_equation.set_mesh(exact_domain.mesh, exact_domain.facet_function, V_vol, order=1)
        p_equation.set_PDE_data(Constant(0.0), marker_dirichlet=[2, 3], source=source_p,
                                dirichlet_BC=[Constant(0.0),
                                              Constant(0.0)])
        p_equation.set_ODE_scheme(adjoint_scheme)
        p_equation.verbose = True
        p_equation.set_time_discretization(exact_pde_dict["T"],
                                           N_steps=int(exact_pde_dict["N_steps"]),
                                           relevant_mesh_size=exact_domain.mesh.hmax())

        # Adjoint to Dirichlet-Neumann state
        q_equation = HeatEquation(efficient=False)
        q_equation.interpolate_data = True
        q_equation.set_mesh(exact_domain.mesh, exact_domain.facet_function, V_vol, order=1)
        q_equation.set_PDE_data(Constant(0.0), marker_dirichlet=[3], marker_neumann=[2],
                                source=source_q,
                                dirichlet_BC=[Constant(0.0)],
                                neumann_BC=[Constant(0.0)])
        q_equation.set_ODE_scheme(adjoint_scheme)
        q_equation.verbose = True
        q_equation.set_time_discretization(exact_pde_dict["T"],
                                           N_steps=int(exact_pde_dict["N_steps"]),
                                           relevant_mesh_size=exact_domain.mesh.hmax())

        p_equation.solve()
        q_equation.solve()

        # flip their times, as the adjoint equation will be simulated in reverse time
        p_equation.solution_list.reverse()
        q_equation.solution_list.reverse()

        # create shape gradient
        dt = Constant(q_equation.dts[0])
        h = TestFunction(V_def)
        I = Identity(exact_domain.mesh.ufl_cell().geometric_dimension())
        A = div(h) * I - grad(h) - grad(h).T

        # result initialization
        dj_mine = 0  # dj manual

        # part due to cost functional
        cost_part = div(h) * Constant(0.0) * dx(exact_domain.mesh)  # the div term is not to have arity mismatches
        fs = 1.0  # final_smoothing
        for (v, w, dt, t, i) in it:
            fs = final_smoothing(exact_pde_dict["T"] - t)
            if fs > DOLFIN_EPS:
                if cost_functional_dict["discretization"] == "trapezoidal" and i == len(
                        v_equation.dts) - 1:
                    cost_part += Constant(.25) * dt * div(h) * fs * (v - w) ** 2 * dx
                else:
                    cost_part += Constant(1 / 2) * dt * div(h) * fs * (v - w) ** 2 * dx

                if i % 600 == 599:  # to make sure I don't hit recursion limits
                    dj_mine += assemble(cost_part)  # add contribution up to now
                    cost_part = div(h) * Constant(0.0) * dx(exact_domain.mesh)  # reset temporary contribution

        # part due to p, v
        pv_part = div(h) * Constant(0.0) * dx(exact_domain.mesh)
        for (vj, vjm, pj, pjm, i) in zip(v_equation.solution_list[1:], v_equation.solution_list[:-1],
                                         p_equation.solution_list[1:], p_equation.solution_list[:-1],
                                         range(len(p_equation.solution_list[:-1]))):
            if adjoint_scheme == "implicit_explicit_euler":
                P = pjm
                V = vj
            else:
                P = (pjm + pj) / 2
                V = (vjm + vj) / 2
            pv_part += ((vj - vjm) / dt * P * div(h) + inner(A * grad(V), grad(P))) * dt * dx(exact_domain.mesh)
            if i % 600 == 599:
                dj_mine += assemble(pv_part)
                pv_part = div(h) * Constant(0.0) * dx(exact_domain.mesh)

        # part due to q, w
        qw_part = div(h) * Constant(0.0) * dx(exact_domain.mesh)
        for (wj, wjm, qj, qjm, i) in zip(w_equation.solution_list[1:], w_equation.solution_list[:-1],
                                         q_equation.solution_list[1:], q_equation.solution_list[:-1],
                                         range(len(p_equation.solution_list[:-1]))):

            if adjoint_scheme == "implicit_explicit_euler":
                Qa = qjm
                Wa = wj
            else:
                Qa = (qjm + qj) / 2
                Wa = (wjm + wj) / 2
            qw_part += ((wj - wjm) / dt * Qa * div(h) + inner(A * grad(Wa), grad(Qa))) * dt * dx(exact_domain.mesh)
            if i % 600 == 599:
                dj_mine += assemble(qw_part)
                qw_part = div(h) * Constant(0.0) * dx(exact_domain.mesh)

        dj_mine += assemble(cost_part + pv_part + qw_part)
        return dj_mine


def W1i_norm(W, Q, problem, V_vol):
    """
    Computes the W^{1,\infty} norm of W
    :param Q: space of DW (piecewise constants)
    :return: the W^{1,\infty} norm of W
    """

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


def string_to_vector_field(Vx, Vy, problem):
    """
    Takes two strings, Vx, Vy, signifying the components of a vector field V

    :param Vx: string of the x component
    :param Vy:
    :return the vector field whose components are given by Vx, Vy
    """

    x, y = SpatialCoordinate(problem.exact_domain.mesh)

    V = [Vx, Vy]
    W = []
    for s in V:
        s = s.replace("^", "**")
        W.append(eval(s))

    return project(as_vector(W))
