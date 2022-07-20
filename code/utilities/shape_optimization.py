from dolfin import *
from dolfin_adjoint import *
import numpy as np
import pickle
import logging

from utilities.meshing import AnnulusMesh, CircleMesh
from utilities.pdes import HeatEquation, TimeExpressionFromList
from utilities.overloads import compute_spherical_transfer_matrix, \
    compute_radial_displacement_matrix, radial_displacement, radial_function_to_square

# %% Setting log and global parameters

parameters[
    "refinement_algorithm"] = "plaza_with_parent_facets"  # must do this to be able to refine stuff associated with a mesh that is being refined
parameters[
    'allow_extrapolation'] = True  # needed if I want a function to be taken from a mesh to a slightly different one

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%H:%M')

set_log_level(LogLevel.ERROR)

# %% Class definitions

class ShapeOptimizationProblem:

    def __init__(self):
        # Geometry data
        self.exact_domain = None  # it will be deformed by q_ex
        self.optimization_domain = None
        self.exact_sphere = None
        self.optimization_sphere = None
        self.q_ex = None  # radial function describing the analytical solution to the shape optimization problem
        self.exact_geometry_dict = None
        self.optimization_geometry_dict = None

        # PDE data: dictionaries that contain info to simulate pdes
        self.exact_pde_dict = None
        self.optimization_pde_dict = None
        self.marker_neumann = None
        self.marker_dirichlet = None
        self.u_N = None
        self.T = None
        self.u_ex = None
        self.exact_pde = None  # it will be of HeatEquation class

        # Function spaces
        self.V_vol = None
        self.V_def = None
        self.V_sph = None

        # Optimization
        self.optimization_dict = None
        self.j = None
        self.q_opt = None
        self.opt_results = None

    def get_domain(self, domain_dict):
        if domain_dict["type"] == "annulus":
            domain = AnnulusMesh(resolution=domain_dict["resolution"],
                                 inner_radius=domain_dict["inner_radius"],
                                 outer_radius=domain_dict["outer_radius"], center=domain_dict["center"])
        else:
            raise ValueError("Domain unsupported")
        return domain

    def get_sphere(self, sphere_dict):
        if sphere_dict["dimension"] == 2:
            sphere = CircleMesh(resolution=sphere_dict["resolution"])
        else:
            raise ValueError("Only 2D is supported for now")
        return sphere

    def create_optimal_geometry(self, exact_geometry_dict):
        with stop_annotating():
            self.exact_domain = self.get_domain(exact_geometry_dict["domain"])
            self.exact_sphere = self.get_sphere(exact_geometry_dict["sphere"])
            self.exact_geometry_dict = exact_geometry_dict

            p = self.exact_domain.center  # star shape point
            f_D = self.exact_domain.boundary_radial_function  # radial function to external boundary

            L1_vol = FiniteElement("Lagrange", self.exact_domain.mesh.ufl_cell(), 1)
            V_vol = FunctionSpace(self.exact_domain.mesh, L1_vol)
            L1_sph = FiniteElement("Lagrange", self.exact_sphere.mesh.ufl_cell(), 1)
            V_sph = FunctionSpace(self.exact_sphere.mesh, L1_sph)
            VD = VectorFunctionSpace(self.exact_domain.mesh, "Lagrange", 1)

            self.q_ex = Function(V_sph)
            circle_coords = self.exact_sphere.mesh.coordinates()[dof_to_vertex_map(V_sph), :]
            q_ex_lambda = eval(exact_geometry_dict["q_ex_lambda"])
            self.q_ex.vector()[:] = q_ex_lambda(circle_coords)

            M = compute_spherical_transfer_matrix(V_vol, V_sph, p=p)
            M2 = compute_radial_displacement_matrix(M, VD, p=p, f_D=f_D)
            W = radial_displacement(self.q_ex, M2, VD)

            ALE.move(self.exact_domain.mesh, W)  # note, id(domain) = id(self.domain), so, both are moved

    def simulate_exact_pde(self, exact_pde_dict):
        if self.exact_domain is None:
            raise Exception("Call first create_optimal_geometry")

        with stop_annotating():

            logging.info("Simulating heat equation on exact deformed domain")

            self.exact_pde_dict = exact_pde_dict
            self.marker_neumann = self.exact_pde_dict["marker_neumann"]
            self.marker_dirichlet = self.exact_pde_dict["marker_dirichlet"]
            exec("self.u_N = " + self.exact_pde_dict["u_N"])
            self.T = self.exact_pde_dict["T"]

            heq = HeatEquation()
            N_steps = self.exact_pde_dict["N_steps"]
            heq.set_mesh(self.exact_domain.mesh, self.exact_domain.facet_function)
            heq.set_ODE_scheme(self.exact_pde_dict["ode_scheme"])
            heq.set_time_discretization(self.T, N_steps=N_steps)  # dt = h^2 and N = T/dt

            u0 = Function(heq.S1h)  # zero initial condition
            u_D_inner = Constant(0.0)  # zero inner BC

            heq.set_PDE_data(u0, marker_neumann=self.marker_neumann, marker_dirichlet=self.marker_dirichlet,
                             neumann_BC=[self.u_N],
                             dirichlet_BC=[
                                 u_D_inner])  # note, no copy is done, the attributes of heq are EXACTLY these guys
            heq.verbose = True

            heq.solve()
            self.u_ex = TimeExpressionFromList(0.0, heq.times, heq.solution_list)

            self.exact_pde = heq

    def initialize_optimization_domain(self, simulated_geometry_dict):
        if self.exact_geometry_dict is None:
            raise Exception("Exact geometry dictionary not initialized")

        optimization_geometry_dict = self.exact_geometry_dict.copy()
        if simulated_geometry_dict["domain_resolution"] is not None:
            optimization_geometry_dict["domain"]["resolution"] = simulated_geometry_dict["domain_resolution"]
        if simulated_geometry_dict["sphere_resolution"] is not None:
            optimization_geometry_dict["sphere"]["resolution"] = simulated_geometry_dict["sphere_resolution"]

        self.optimization_domain = self.get_domain(optimization_geometry_dict["domain"])
        self.optimization_sphere = self.get_sphere(optimization_geometry_dict["sphere"])

        self.optimization_geometry_dict = optimization_geometry_dict

    def initialize_pde_simulation(self, simulated_pde_dict):

        if self.exact_pde_dict is None:
            raise Exception("Exact PDE dictionary not initialized")

        self.optimization_pde_dict = self.exact_pde_dict.copy()
        if simulated_pde_dict["ode_scheme"] is not None:
            self.optimization_pde_dict["ode_scheme"] = simulated_pde_dict["ode_scheme"]
        if simulated_pde_dict["N_steps"] is not None:
            self.optimization_pde_dict["N_steps"] = simulated_pde_dict["N_steps"]

    def set_optimization_dict(self, optimization_dict):
        self.optimization_dict = optimization_dict

    def save_problem_data(self, path):

        if self.optimization_dict is None:
            raise Exception("No optimization options were found")

        pickles_path = path
        import os
        os.makedirs(path, exist_ok=False)
        with open(pickles_path + "exact_geometry_dict" + '.pickle', 'wb') as handle:
            pickle.dump(self.exact_geometry_dict, handle)
        with open(pickles_path + "exact_pde_dict" + '.pickle', 'wb') as handle:
            pickle.dump(self.exact_pde_dict, handle)

        simulated_geometry_dict = {
            "domain_resolution": self.optimization_geometry_dict["domain"]["resolution"],
            "sphere_resolution": self.optimization_geometry_dict["sphere"]["resolution"],
        }

        simulated_pde_dict = {
            "ode_scheme": self.optimization_pde_dict["ode_scheme"],
            "N_steps": self.optimization_pde_dict["N_steps"]
        }

        with open(pickles_path + "simulated_geometry_dict" + '.pickle', 'wb') as handle:
            pickle.dump(simulated_geometry_dict, handle)
        with open(pickles_path + "simulated_pde_dict" + '.pickle', 'wb') as handle:
            pickle.dump(simulated_pde_dict, handle)

        with open(pickles_path + "optimization_dict" + '.pickle', 'wb') as handle:
            pickle.dump(self.optimization_dict, handle)

    def initialize_from_data(self, path, method="pickle"):

        if method == "pickle":
            try:
                with open(path + "exact_geometry_dict" + '.pickle', 'rb') as handle:
                    exact_geometry_dict = pickle.load(handle)
                with open(path + "exact_pde_dict" + '.pickle', 'rb') as handle:
                    exact_pde_dict = pickle.load(handle)
                with open(path + "simulated_geometry_dict" + '.pickle', 'rb') as handle:
                    simulated_geometry_dict = pickle.load(handle)
                with open(path + "simulated_pde_dict" + '.pickle', 'rb') as handle:
                    simulated_pde_dict = pickle.load(handle)
                with open(path + "optimization_dict" + '.pickle', 'rb') as handle:
                    optimization_dict = pickle.load(handle)
            except:
                raise FileNotFoundError("Some pickles to load from don't exist")

        elif method == "python":
            try:
                import importlib.util
                import sys
                spec = importlib.util.spec_from_file_location("problem_data", path + "problem_data.py")
                pd = importlib.util.module_from_spec(spec)
                sys.modules["problem_data"] = pd
                spec.loader.exec_module(pd)

                exact_geometry_dict = pd.exact_geometry_dict
                exact_pde_dict = pd.exact_pde_dict
                simulated_geometry_dict = pd.simulated_geometry_dict
                simulated_pde_dict = pd.simulated_pde_dict
                optimization_dict = pd.optimization_dict
            except:
                raise Exception("Couldn't load variables from path")

        self.create_optimal_geometry(exact_geometry_dict)
        self.simulate_exact_pde(exact_pde_dict)
        self.initialize_optimization_domain(simulated_geometry_dict)
        self.initialize_pde_simulation(simulated_pde_dict)
        self.set_optimization_dict(optimization_dict)

        logging.info("Shape optimization data succesfully loaded")

    def create_cost_functional(self):

        logging.info("Setting uo the cost functional")

        L1_vol = FiniteElement("Lagrange", self.optimization_domain.mesh.ufl_cell(), 1)
        L1_sph = FiniteElement("Lagrange", self.optimization_sphere.mesh.ufl_cell(), 1)

        self.V_vol = FunctionSpace(self.optimization_domain.mesh, L1_vol)
        self.V_sph = FunctionSpace(self.optimization_sphere.mesh, L1_sph)
        self.V_def = VectorFunctionSpace(self.optimization_domain.mesh, "Lagrange", 1)

        with stop_annotating():
            M = compute_spherical_transfer_matrix(self.V_vol, self.V_sph, p=self.optimization_domain.center)
            M2 = compute_radial_displacement_matrix(M, self.V_def, p=self.optimization_domain.center,
                                                    f_D=self.optimization_domain.boundary_radial_function)

        # Mesh movement
        self.q_opt = Function(self.V_sph)  # null displacement
        W = radial_displacement(self.q_opt, M2, self.V_def)
        ALE.move(self.optimization_domain.mesh, W)

        # PDEs definition

        u0 = Function(self.V_vol)  # zero initial condition
        u_D_inner = Constant(0.0)  # zero inner BC

        # Dirichlet state
        v_equation = HeatEquation()
        v_equation.set_mesh(self.optimization_domain.mesh, self.optimization_domain.facet_function, self.V_vol)
        v_equation.set_PDE_data(u0, marker_dirichlet=[2, 3],
                                dirichlet_BC=[self.u_ex,
                                              u_D_inner])
        v_equation.set_ODE_scheme(self.optimization_pde_dict["ode_scheme"])
        v_equation.verbose = True
        v_equation.set_time_discretization(self.T, N_steps=self.optimization_pde_dict["N_steps"])

        # Dirichlet-Neumann state
        w_equation = HeatEquation()
        w_equation.set_mesh(self.optimization_domain.mesh, self.optimization_domain.facet_function, self.V_vol)
        w_equation.set_PDE_data(u0, marker_dirichlet=[3], marker_neumann=[2],
                                dirichlet_BC=[u_D_inner],
                                neumann_BC=[self.u_N])
        w_equation.set_ODE_scheme(self.optimization_pde_dict["ode_scheme"])
        w_equation.verbose = True
        w_equation.set_time_discretization(self.T, N_steps=self.optimization_pde_dict["N_steps"])

        # The cost functional

        v_equation.solve()
        w_equation.solve()

        J = 0
        logging.warning("Integral discretization only valid for implicit euler")
        for v, w, dt, t in zip(v_equation.solution_list[1:], w_equation.solution_list[1:], v_equation.dts,
                               v_equation.times[1:]):
            J += assemble(dt * (v - w) ** 2 * dx(self.optimization_domain.mesh))  # *exp(-0.05/(T-t)**2)

        self.j = ReducedFunctional(J, Control(self.q_opt))

    def do_taylor_test(self):
        if self.j is None:
            raise Exception("No reduced cost functional is available")
        h = Function(self.V_sph)
        h.vector()[:] = .1
        taylor_test(self.j, self.q_opt, h)

    def solve(self):
        self.q_opt, self.opt_results = minimize(self.j, tol=1e-6, options=self.optimization_dict)


# %%  Exemplary usage

if __name__ == "__main__":
    results_path = "/home/leonardo_mutti/PycharmProjects/masters_thesis/code/results/test0/"

    # u_N = Expression('exp(-a/pow(t,2))*sin(3*t)*pow(x[0],2)', t=0, a=.05, degree=4)
    # u_N = Expression('exp(-a/pow(t,2))*sin(3*t)*(pow(x[0],2)-cos(4*x[1]))', t=0, a=.05, degree=5)
    # q_ex_lambda = lambda circle_coords: -.5 * circle_coords[:, 0] ** 2 + .7 * circle_coords[:, 1] ** 2

    shpb = ShapeOptimizationProblem()
    shpb.initialize_from_data("/home/leonardo_mutti/PycharmProjects/masters_thesis/code/results/test0/",
                              method="python")
    shpb.create_cost_functional()
    shpb.do_taylor_test()
