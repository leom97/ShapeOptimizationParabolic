from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt
import logging
import numpy as np
import pickle

from utilities.meshing import AnnulusMesh, CircleMesh
from utilities.pdes import HeatEquation, TimeExpressionFromList
from utilities.overloads import compute_spherical_transfer_matrix, \
    compute_radial_displacement_matrix, radial_displacement, radial_function_to_square

# %% Setting log and global parameters

parameters[
    "refinement_algorithm"] = "plaza_with_parent_facets"  # must do this to be able to refine stuff associated with a mesh that is being refined
parameters[
    'allow_extrapolation'] = True  # needed if I want a function to be taken from a mesh to a slightly different one

# ffc_logger = logging.getLogger('FFC')
# ffc_logger.setLevel(logging.ERROR)
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%H:%M')

set_log_level(LogLevel.ERROR)

# %% Let's create a mesh and deform it... todo: move to separate file
with stop_annotating():
    ode_scheme = "crank_nicolson"

    p = np.array([0, 0])  # star shape point
    f_D = lambda x: 2 * np.ones(x.shape[0])  # external point

    annulus = AnnulusMesh(resolution=.05)  # we can be very fine here
    circle = CircleMesh(resolution=.15)

    L1_vol = FiniteElement("Lagrange", annulus.mesh.ufl_cell(), 1)
    V_vol = FunctionSpace(annulus.mesh, L1_vol)
    L1_sph = FiniteElement("Lagrange", circle.mesh.ufl_cell(), 1)
    V_sph = FunctionSpace(circle.mesh, L1_sph)
    VD = VectorFunctionSpace(annulus.mesh, "Lagrange", 1)

    q_ex = Function(V_sph)
    circle_coords = circle.mesh.coordinates()[dof_to_vertex_map(V_sph), :]
    # q_ex.vector()[:] = -.5 * circle_coords[:, 0] ** 2 + .7 * circle_coords[:, 1] ** 2
    r_sq = radial_function_to_square(circle_coords, L=1)
    q_ex.vector()[:] = r_sq - 1

    M = compute_spherical_transfer_matrix(V_vol, V_sph, p=p)
    M2 = compute_radial_displacement_matrix(M, VD, p=p, f_D=f_D)
    W = radial_displacement(q_ex, M2, VD)

    ALE.move(annulus.mesh, W)

    plot(annulus.mesh)
    plt.show()

    # %% Let's simulate the heat equation in here: luckily the facet function moved with the mesh!
    # We simulate the a Neumann-Dirichlet PDE, to then obtain the full Dirichlet conditions

    u0 = Function(V_vol)  # zero initial condition
    u_D_inner = Function(V_vol)  # zero inner BC
    # u_N = Expression('exp(-a/pow(t,2))*sin(3*t)*pow(x[0],2)', t=0, a=.05, degree=4)
    u_N = Expression('exp(-a/pow(t,2))*sin(3*t)*(pow(x[0],2)-cos(4*x[1]))', t=0, a=.05, degree=5)
    marker_dirichlet = [3]  # 0 BC on inner ring
    marker_neumann = [2]
    T = 1.0

    heq = HeatEquation()
    heq.set_PDE_data(u0, marker_neumann=marker_neumann, marker_dirichlet=marker_dirichlet, neumann_BC=[u_N],
                     dirichlet_BC=[u_D_inner])  # note, no copy is done, the attributes of heq are EXACTLY these guys
    heq.set_ODE_scheme(ode_scheme)
    heq.verbose = True

    heq.set_mesh(annulus.mesh, annulus.facet_function)
    N_steps = int(np.ceil(T / (heq.mesh.hmax() ** 2)))
    heq.set_time_discretization(T, N_steps=N_steps)  # dt = h^2 and N = T/dt
    solution_list, _ = heq.solve()
    u_ex = TimeExpressionFromList(0.0, heq.times, solution_list)

# %% Mesh and function spaces definition

annulus = AnnulusMesh(resolution=.05)  # we can be very fine here
circle = CircleMesh(resolution=.1)

L1_vol = FiniteElement("Lagrange", annulus.mesh.ufl_cell(), 1)
V_vol = FunctionSpace(annulus.mesh, L1_vol)
L1_sph = FiniteElement("Lagrange", circle.mesh.ufl_cell(), 1)
V_sph = FunctionSpace(circle.mesh, L1_sph)
VD = VectorFunctionSpace(annulus.mesh, "Lagrange", 1)

M = compute_spherical_transfer_matrix(V_vol, V_sph, p=p)
M2 = compute_radial_displacement_matrix(M, VD, p=p, f_D=f_D)

# %% Mesh movement

q = Function(V_sph)  # null displacement
W = radial_displacement(q, M2, VD)
ALE.move(annulus.mesh, W)

plot(annulus.mesh)
plt.show()

# %% PDEs definition

u0 = Constant(0.0)  # zero initial condition
u_D_inner = Constant(0.0)  # zero inner BC
u_D_outer = u_ex  # a time dependent expression

# Dirichlet state
heqv = HeatEquation()
heqv.set_mesh(annulus.mesh, annulus.facet_function)
heqv.set_PDE_data(u0, marker_dirichlet=[2, 3],
                  dirichlet_BC=[u_D_outer,
                                u_D_inner])  # note, no copy is done, the attributes of heq are EXACTLY these guys
heqv.set_ODE_scheme(ode_scheme)
heqv.verbose = True
heqv.set_time_discretization(T, N_steps=N_steps)  # dt = h^2 and N = T/dt

# Dirichlet-Neumann state
heqw = HeatEquation()
heqw.set_mesh(annulus.mesh, annulus.facet_function)
heqw.set_PDE_data(u0, marker_dirichlet=[3], marker_neumann=[2],
                  dirichlet_BC=[u_D_inner],
                  neumann_BC=[u_N])  # note, no copy is done, the attributes of heq are EXACTLY these guys
heqw.set_ODE_scheme(ode_scheme)
heqw.verbose = True
heqw.set_time_discretization(T, N_steps=N_steps)  # dt = h^2 and N = T/dt

# %% The cost functional

solution_list_v, _ = heqv.solve()
solution_list_w, _ = heqw.solve()

J = 0
logging.warning("Integral discretization only valid for implicit euler")
for v, w, dt, t in zip(solution_list_v[1:], solution_list_w[1:], heqv.dts, heqv.times[1:]):
    J += assemble(dt * (v - w) ** 2 * dx(annulus.mesh))  # *exp(-0.05/(T-t)**2)

j = ReducedFunctional(J, Control(q))

h = Function(V_sph)
h.vector()[:] = .1
taylor_test(j, q, h)

# %% The optimization

q_opt, res = minimize(j, tol=1e-6, options={"ftol": 1e-14, "gtol": 1e-14, "maxiter": 1, "disp": True})
plot(annulus.mesh)
plt.show()

taylor_test(j, q_opt, h)

# %% Notes

# With final smoothing (around 40 minutes with another simulation running in parallel)
#    N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
#    63    800    923      1     0     0   4.681D-08   5.325D-10
#   F =   5.3247090595756105E-010
# STOP: TOTAL NO. of ITERATIONS REACHED LIMIT

# Without final smoothing (also 40 minutes): the cost is higher because there is no final smoothing here)
#    N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
#    63    800    971      2     0     0   5.054D-08   2.949D-09
#   F =   2.9487536940310408E-009
# STOP: TOTAL NO. of ITERATIONS REACHED LIMIT

# With more complicated BCs, no final smoothing (in space)
#    N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
#    63    800    923      1     0     0   3.278D-08   1.128D-09
#   F =   1.1281012793695380E-009
# STOP: TOTAL NO. of ITERATIONS REACHED LIMIT

# With standard BCs, no final smoothing, turning a circle into a square (Note, it stopped at 172!)
#    N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
#    63    172    184      1     0     0   3.419D-09   2.943D-11
#   F =   2.9434237016963961E-011
