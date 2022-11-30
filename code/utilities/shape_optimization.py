"""
Implements a class representing the shape optimization problem, with several methods to solve it, and visualize the
results. It is used in shape_optimization_main.py.
"""

# %% Imports

from dolfin import *
from dolfin_adjoint import *
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
import moola
from tqdm import tqdm

from utilities.meshing import AnnulusMesh, CircleMesh, SquareAnnulusMesh, \
    SmoothedSquareAnnulusMesh, sea_urchin
from utilities.pdes import HeatEquation, TimeExpressionFromList, PreAssembledBC
from utilities.overloads import compute_spherical_transfer_matrix, \
    compute_radial_displacement_matrix, radial_displacement, radial_function_to_square
import fenics_adjoint.shapead_transformations as adjf

# %% Setting log and global parameters

parameters[
    "refinement_algorithm"] = "plaza_with_parent_facets"  # must do this to be able to refine stuff associated with a mesh that is being refined
parameters[
    'allow_extrapolation'] = True  # needed if I want a function to be taken from a mesh to a slightly different one

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%H:%M')

set_log_level(LogLevel.ERROR)
logging.getLogger('FFC').setLevel(logging.ERROR)
logging.getLogger('UFL').setLevel(logging.ERROR)


# %% Class definitions

class ShapeOptimizationProblem:

    def __init__(self):
        # Geometry data
        self.exact_domain = None  # it will be deformed by q_ex
        self.optimization_domain = None
        self.exact_sphere = None
        self.optimization_sphere = None
        self.q_ex = None  # radial function describing the analytical solution to the shape optimization problem
        self.W_ex = None  # associated radial displacement
        self.exact_geometry_dict = None
        self.optimization_geometry_dict = None
        self.simulated_geometry_dict = None
        self.problem_folder = None

        # PDE data: dictionaries and variables that contain info to simulate pdes
        self.exact_pde_dict = None
        self.optimization_pde_dict = None
        self.marker_neumann = None
        self.marker_dirichlet = None
        self.u_N = None  # expression for exterior neumann data
        self.u_D = None  # expression for exterior dirichlet data
        self.T = None
        self.u_ex = None
        self.exact_pde = None  # it will be of HeatEquation class
        self.exact_exterior_BC = None  # can be "N" or "D" (it corresponds to which one is the exact data we prescribe)
        self.pre_assembled_BCs = None  # it contains time-lists of functions, representing the time-dependent boundary conditions at the right instants

        # Function spaces
        self.V_vol = None
        self.V_def = None
        self.V_sph = None

        # Optimization
        self.optimization_dict = None
        self.cost_functional_dict = None
        self.j = None
        self.q_opt = None
        self.opt_results = None
        self.duration = None
        self.v_equation = None
        self.w_equation = None

    def get_domain(self, domain_dict, ground_truth):
        if ground_truth:
            subfolder = "meshes/domain/ground_truth/"
        else:
            subfolder = "meshes/domain/simulation/"
        xdmf_path = None
        if "reload_xdmf" in domain_dict.keys() and ground_truth == True:
            if domain_dict["reload_xdmf"]:
                xdmf_path = self.problem_folder + subfolder
        if domain_dict["type"] == "annulus":
            domain = AnnulusMesh(resolution=domain_dict["resolution"],
                                 path=self.problem_folder + subfolder,
                                 int_refinement=domain_dict["int_refinement"],
                                 ext_refinement=domain_dict["ext_refinement"],
                                 inner_radius=domain_dict["inner_radius"],
                                 outer_radius=domain_dict["outer_radius"],
                                 center=domain_dict["center"],
                                 xdmf_path=xdmf_path)
        elif domain_dict["type"] == "square_annulus":
            domain = SquareAnnulusMesh(resolution=domain_dict["resolution"],
                                       path=self.problem_folder + subfolder,
                                       inner_radius=domain_dict["inner_radius"],
                                       side_length=domain_dict["side_length"],
                                       int_refinement=domain_dict["int_refinement"],
                                       ext_refinement=domain_dict["ext_refinement"],
                                       xdmf_path=xdmf_path
                                       )
        elif domain_dict["type"] == "smoothed_square_annulus":
            domain = SmoothedSquareAnnulusMesh(resolution=domain_dict["resolution"],
                                               path=self.problem_folder + subfolder,
                                               inner_radius=domain_dict["inner_radius"],
                                               side_length=domain_dict["side_length"],
                                               int_refinement=domain_dict["int_refinement"],
                                               ext_refinement=domain_dict["ext_refinement"],
                                               xdmf_path=xdmf_path,
                                               smoothing_radius=domain_dict["smoothing_radius"]
                                               )
        else:
            raise ValueError("Domain unsupported")
        return domain

    def get_sphere(self, sphere_dict, ground_truth):
        if ground_truth:
            subfolder = "ground_truth/"
        else:
            subfolder = "simulation/"
        if sphere_dict["dimension"] == 2:
            sphere = CircleMesh(resolution=sphere_dict["resolution"],
                                path=self.problem_folder + "meshes/sphere/" + subfolder)
        else:
            raise ValueError("Only 2D is supported for now")
        return sphere

    def create_optimal_geometry(self, exact_geometry_dict, reusables_dict=None):

        """
        Create the domain which we will later want to reconstruct.
        """

        logging.info("Creating optimal geometry")

        with stop_annotating():
            self.exact_domain = self.get_domain(exact_geometry_dict["domain"], ground_truth=True)
            self.exact_sphere = self.get_sphere(exact_geometry_dict["sphere"], ground_truth=True)
            self.exact_geometry_dict = exact_geometry_dict

            p = self.exact_domain.center  # star shape point
            f_D = self.exact_domain.boundary_radial_function  # radial function to external boundary

            L1_vol = FiniteElement("Lagrange", self.exact_domain.mesh.ufl_cell(), 1)
            V_vol = FunctionSpace(self.exact_domain.mesh, L1_vol)
            L1_sph = FiniteElement("Lagrange", self.exact_sphere.mesh.ufl_cell(), 1)
            V_sph = FunctionSpace(self.exact_sphere.mesh, L1_sph)
            self.V_sph_ex = V_sph
            VD = VectorFunctionSpace(self.exact_domain.mesh, "Lagrange", 1)

            self.q_ex = Function(V_sph)

            if reusables_dict is None:

                circle_coords = self.exact_sphere.mesh.coordinates()[dof_to_vertex_map(V_sph), :]
                q_ex_lambda = eval(exact_geometry_dict["q_ex_lambda"])
                self.q_ex.vector()[:] = q_ex_lambda(circle_coords)

                M = compute_spherical_transfer_matrix(V_vol, V_sph, p=p)
                M2 = compute_radial_displacement_matrix(M, VD, p=p, f_D=f_D)
                self.W_ex = radial_displacement(self.q_ex, M2, VD)
            else:
                logging.info("Loading exact geometry from pickle")
                self.q_ex.vector()[:] = reusables_dict["q_ex"]
                self.W_ex = Function(VD)
                self.W_ex.vector()[:] = reusables_dict["W_ex"]

            ALE.move(self.exact_domain.mesh, self.W_ex)  # note, id(domain) = id(self.domain), so, both are moved

            if reusables_dict is None:
                return M2

    def simulate_exact_pde(self, exact_pde_dict, reusables_dict=None):

        """
        This step is performed to generate the Dirichlet boundary data, to then be used to carry out the shape
        optimization, as describe in section 4.1
        """

        if self.exact_domain is None:
            raise Exception("Call first create_optimal_geometry")

        logging.info("Simulating heat equation on exact deformed domain")

        with stop_annotating():
            self.exact_pde_dict = exact_pde_dict
            self.marker_neumann = self.exact_pde_dict["marker_neumann"]
            self.marker_dirichlet = self.exact_pde_dict["marker_dirichlet"]
            if "u_N" in self.exact_pde_dict.keys():
                exec("self.u_N = " + self.exact_pde_dict["u_N"])
                self.exact_exterior_BC = "N"
            elif "u_D" in self.exact_pde_dict.keys():
                exec("self.u_D = " + self.exact_pde_dict["u_D"])
                self.exact_exterior_BC = "D"
                raise NotImplementedError("We only support specification of the Neumann derivative, at the moment")

            else:
                raise Exception("No boundary data was provided")
            self.T = self.exact_pde_dict["T"]

            heq = HeatEquation()
            heq.interpolate_data = True  # super important for the modified Crank-Nicolson, see pdes.py, include_neumann
            N_steps = self.exact_pde_dict["N_steps"]
            heq.set_mesh(self.exact_domain.mesh, self.exact_domain.facet_function)
            heq.set_ODE_scheme(self.exact_pde_dict["ode_scheme"])
            heq.set_time_discretization(self.T, N_steps=N_steps)  # dt = h^2 and N = T/dt

            u0 = Function(heq.S1h)  # zero initial condition
            u_D_inner = Constant(0.0)  # zero inner BC

            if self.exact_exterior_BC == "N":
                heq.set_PDE_data(u0, marker_neumann=self.marker_neumann, marker_dirichlet=self.marker_dirichlet,
                                 neumann_BC=[self.u_N],
                                 dirichlet_BC=[
                                     u_D_inner])  # note, no copy is done, the attributes of heq are EXACTLY these guys
            elif self.exact_exterior_BC == "D":
                heq.set_PDE_data(u0, marker_neumann=self.marker_neumann, marker_dirichlet=self.marker_dirichlet,
                                 dirichlet_BC=[self.u_D, u_D_inner])

            heq.verbose = True

            if reusables_dict is None:
                heq.solve()
            else:
                logging.info("Reloading exact PDE solution from pickle")
                for v in reusables_dict["u_ex"]:
                    u = Function(heq.S1h)
                    u.vector()[:] = v
                    heq.solution_list.append(u)

            self.u_ex = TimeExpressionFromList(0.0, heq.times, heq.solution_list)
            if self.exact_exterior_BC == "N":
                self.u_D = self.u_ex
            elif self.exact_exterior_BC == "D":
                self.u_N = self.u_ex

            self.exact_pde = heq

    def initialize_optimization_domain(self, simulated_geometry_dict):
        if self.exact_geometry_dict is None:
            raise Exception("Exact geometry dictionary not initialized")

        optimization_geometry_dict = self.exact_geometry_dict.copy()
        optimization_geometry_dict["reload_xdmf"] = False
        if simulated_geometry_dict["additional_domain_data"] is not None:
            for key in simulated_geometry_dict["additional_domain_data"].keys():
                optimization_geometry_dict["domain"][key] = simulated_geometry_dict["additional_domain_data"][key]
        if simulated_geometry_dict["sphere_resolution"] is not None:
            optimization_geometry_dict["sphere"]["resolution"] = simulated_geometry_dict["sphere_resolution"]

        self.optimization_domain = self.get_domain(optimization_geometry_dict["domain"], ground_truth=False)
        self.optimization_sphere = self.get_sphere(optimization_geometry_dict["sphere"], ground_truth=False)

        self.optimization_geometry_dict = optimization_geometry_dict

    def initialize_pde_simulation(self, simulated_pde_dict):

        if self.exact_pde_dict is None:
            raise Exception("Exact PDE dictionary not initialized")

        self.optimization_pde_dict = self.exact_pde_dict.copy()
        if simulated_pde_dict["ode_scheme"] is not None:
            self.optimization_pde_dict["ode_scheme"] = simulated_pde_dict["ode_scheme"]
        if simulated_pde_dict["N_steps"] is not None:
            self.optimization_pde_dict["N_steps"] = simulated_pde_dict["N_steps"]

        # Generic set-up
        L1_vol = FiniteElement("Lagrange", self.optimization_domain.mesh.ufl_cell(), 1)
        self.V_vol = FunctionSpace(self.optimization_domain.mesh, L1_vol)  # linear finite elements on the moving domain

        self.v_equation = HeatEquation(
            efficient=True)  # to save some time for the dirichlet condition, expensive to evaluate
        self.w_equation = HeatEquation(
            efficient=False)  # in order to have the correct CN: not the midpoint one (plus, no speed-up is observed)

        self.v_equation.set_ODE_scheme(self.optimization_pde_dict["ode_scheme"])
        self.v_equation.interpolate_data = True  # important for the correct functioning of CN
        self.v_equation.verbose = True
        self.v_equation.set_time_discretization(self.T,
                                                N_steps=int(self.optimization_pde_dict["N_steps"]),
                                                relevant_mesh_size=self.exact_domain.mesh.hmax())

        self.w_equation.set_ODE_scheme(self.optimization_pde_dict["ode_scheme"])
        self.w_equation.verbose = True
        self.w_equation.interpolate_data = True
        self.w_equation.set_time_discretization(self.T,
                                                N_steps=int(self.optimization_pde_dict["N_steps"]),
                                                relevant_mesh_size=self.exact_domain.mesh.hmax())

        # Pre-assembling the boundary conditions
        if self.optimization_pde_dict["ode_scheme"] == "crank_nicolson":
            times = (self.v_equation.times[1:] + self.v_equation.times[:-1]) / 2
        elif self.optimization_pde_dict["ode_scheme"] in ["implicit_euler", "implicit_explicit_euler"]:
            times = self.v_equation.times[1:]
        else:
            raise Exception("No matching ode scheme")

        logging.info("Pre-assembling the boundary conditions")
        external_DBC = PreAssembledBC(self.u_ex, self.v_equation.times[1:], self.V_vol)
        external_NBC = PreAssembledBC(self.u_N, times, self.V_vol)

        # Add noise
        external_NBC.perturb(simulated_pde_dict["noise_level_on_exact_BC"])
        external_DBC.perturb(simulated_pde_dict["noise_level_on_exact_BC"])

        self.pre_assembled_BCs = {"ext_neumann": external_NBC, "ext_dirichlet": external_DBC}

    def initialize_from_data(self, path, regenerate_exact_data=True):

        """
        Initializes the shape optimization problem. If regenerate_exact_data is False, it will attempt to reload cached
        files, otherwise fresh ones will be generated.
        """

        self.problem_folder = path
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
            cost_functional_dict = pd.cost_functional_dict
        except:
            raise Exception("Couldn't load variables from path")

        if regenerate_exact_data:
            self.create_optimal_geometry(exact_geometry_dict)
            self.simulate_exact_pde(exact_pde_dict)
        else:
            try:
                logging.info("Opening reusables pickle")
                with open(path + "reusables" + '.pickle', 'rb') as handle:
                    reusables_dict = pickle.load(handle)

                # create_optimal_geometry
                self.create_optimal_geometry(exact_geometry_dict, reusables_dict)
                # simulate_exact_pde
                self.simulate_exact_pde(exact_pde_dict, reusables_dict)
            except:
                raise Exception("Could not load exact data from files")

        self.initialize_optimization_domain(simulated_geometry_dict)
        self.initialize_pde_simulation(simulated_pde_dict)
        self.optimization_dict = optimization_dict
        self.cost_functional_dict = cost_functional_dict
        self.simulated_geometry_dict = simulated_geometry_dict

        if regenerate_exact_data:
            self.exact_geometry_dict["domain"]["reload_xdmf"] = False

        logging.info("Shape optimization data successfully loaded")

    def create_cost_functional(self, disable_radial_parametrization=False, start_at_optimum=False):

        """
        Creates the cost functional to be later optimized, a dolfin adjoint object

        :param disable_radial_parametrization: debugging option, the displacement field will be the optimization variable itself
        :param start_at_optimum: if True, the initial deformation is the interpolated optimum
        :return:
        """

        logging.info("Setting up the cost functional")

        L1_sph = FiniteElement("Lagrange", self.optimization_sphere.mesh.ufl_cell(), 1)
        self.V_sph = FunctionSpace(self.optimization_sphere.mesh,
                                   L1_sph)  # spherical controls live here: piecewise linear,
        # scalar FEM on the unit sphere
        self.V_def = VectorFunctionSpace(self.optimization_domain.mesh, "Lagrange",
                                         1)  # deformation fields on the reference domain induced by the scalar controls

        with stop_annotating():  # matrices to go from the sphere to the volume
            M = compute_spherical_transfer_matrix(self.V_vol, self.V_sph, p=self.optimization_domain.center)
            M2 = compute_radial_displacement_matrix(M, self.V_def, p=self.optimization_domain.center,
                                                    f_D=self.optimization_domain.boundary_radial_function)

        # Mesh movement: register the domain parametrization in dolfin-adjoint
        if disable_radial_parametrization:
            if start_at_optimum:
                self.q_opt = radial_displacement(interpolate(self.q_ex, self.V_sph), M2, self.V_def)
            else:
                self.q_opt = Function(self.V_def)
            ALE.move(self.optimization_domain.mesh, self.q_opt)
        else:  # parametrizing shapes by radial functions
            if start_at_optimum:
                self.q_opt = interpolate(self.q_ex, self.V_sph)
            else:
                self.q_opt = Function(self.V_sph)  # null displacement (but doesn't need to be so)
            W = radial_displacement(self.q_opt, M2, self.V_def)
            ALE.move(self.optimization_domain.mesh, W)

        # PDEs definitions
        u0 = Function(self.V_vol)  # zero initial condition
        u_D_inner = Constant(0.0)  # zero inner BC

        # Dirichlet state (2 is external boundary, 3 internal)
        self.v_equation.set_mesh(self.optimization_domain.mesh, self.optimization_domain.facet_function, self.V_vol,
                                 order=1)
        self.v_equation.set_PDE_data(u0, marker_dirichlet=[2, 3],
                                     dirichlet_BC=[self.u_D, u_D_inner],
                                     pre_assembled_BCs={2: {"type": "dirichlet", "marker": 2,
                                                            "data": self.pre_assembled_BCs["ext_dirichlet"]}})

        # Dirichlet-Neumann state
        self.w_equation.set_mesh(self.optimization_domain.mesh, self.optimization_domain.facet_function, self.V_vol,
                                 order=1)
        self.w_equation.set_PDE_data(u0, marker_dirichlet=[3], marker_neumann=[2],
                                     dirichlet_BC=[u_D_inner],
                                     neumann_BC=[self.u_N],
                                     pre_assembled_BCs={
                                         2: {"type": "neumann", "data": self.pre_assembled_BCs["ext_neumann"]}})

        # The cost functional
        self.v_equation.solve()
        self.w_equation.solve()

        J = 0
        final_smoothing = eval(self.cost_functional_dict["final_smoothing_lambda"])

        # A check on the cost functional discretization type
        if self.cost_functional_dict["discretization"] in ["rectangle_end", "trapezoidal", None]:
            if self.cost_functional_dict["discretization"] is None:
                logging.info("The cost functional is discretized with the default scheme")
                if self.w_equation.ode_scheme == "implicit_euler":
                    self.cost_functional_dict["discretization"] = "rectangle_end"
                elif self.w_equation.ode_scheme == "crank_nicolson":
                    self.cost_functional_dict["discretization"] = "trapezoidal"
                else:
                    raise Exception("ODE scheme not supported")
            else:
                logging.warning("A custom cost functional discretization has been chosen")
        else:
            raise Exception("Quadrature rule not supported")

        it = zip(self.v_equation.solution_list[1:],
                 self.w_equation.solution_list[1:],
                 self.v_equation.dts,
                 self.v_equation.times[1:],
                 range(len(self.v_equation.dts)))

        # Building cost functional
        # ds = Measure("ds", subdomain_data=self.optimization_domain.facet_function, subdomain_id=2)
        dx = Measure("dx", domain=self.optimization_domain.mesh)
        fs = 1.0  # final_smoothing
        for (v, w, dt, t, i) in it:
            fs = final_smoothing(self.T - t)
            if fs > DOLFIN_EPS:
                if self.cost_functional_dict["discretization"] == "trapezoidal" and i == len(
                        self.v_equation.dts) - 1:
                    J += assemble(.25 * dt * fs * (v - w) ** 2 * dx)
                else:
                    J += assemble(.5 * dt * fs * (v - w) ** 2 * dx)

        if "H1_smoothing" in self.cost_functional_dict:
            if disable_radial_parametrization:
                logging.warning("H1 smoothing is not implemented for vector fields, assuming 0 smoothing")
            else:
                J += self.cost_functional_dict["H1_smoothing"] * assemble(
                    inner(grad(self.q_opt), grad(self.q_opt)) * dx(self.optimization_sphere.mesh))

        self.j = ReducedFunctional(J, Control(self.q_opt))

        return M, M2

    def do_taylor_test(self):
        if self.j is None:
            raise Exception("No reduced cost functional is available")
        h = Function(self.V_sph)
        h.vector()[:] = .1
        td = taylor_to_dict(self.j, self.q_opt, h)

        logging.info("Results of Taylor tests: rates and residuals")
        for i in range(3):
            logging.info(td["R" + str(i)]["Rate"])
            logging.info(td["R" + str(i)]["Residual"])

    def solve(self):

        def callback(x):
            plt.clf()
            plot(self.optimization_domain.mesh)
            plt.draw()
            plt.pause(0.001)

        logging.info("Shape optimization starts now")
        import time
        duration = -time.time()

        if self.optimization_dict["solver"] == "moola_BFGS":
            # the moola way
            # Set up moola problem and solve optimisation
            problem_moola = MoolaOptimizationProblem(self.j)

            m_moola = moola.DolfinPrimalVector(self.q_opt, inner_product=self.optimization_dict["inner_product"])
            solver = moola.CustomBFGS(problem_moola, m_moola, options=self.optimization_dict["options"])

            plt.ion()
            plt.show()
            self.q_opt, self.opt_results = solver.solve(callback=callback)

        elif self.optimization_dict["solver"] == "moola_newton":
            problem_moola = MoolaOptimizationProblem(self.j)

            m_moola = moola.DolfinPrimalVector(self.q_opt, inner_product=self.optimization_dict["inner_product"])
            solver = moola.RegularizedNewton(problem_moola, m_moola, options=self.optimization_dict["options"])

            plt.ion()
            plt.show()
            self.q_opt, self.opt_results = solver.solve(callback=callback)
        else:
            raise Exception("Unsupported solver")

        duration += time.time()

        self.duration = duration / 60  # in minutes

        logging.info(f"{self.duration} minutes elapsed")

    def visualize_result(self):
        logging.info("Visualizing the geometries")

        plt.ioff()

        plt.figure()
        plot(self.exact_domain.mesh, title="Exact solution")
        plt.show()

        plt.figure()
        plot(self.optimization_domain.mesh, title="Computed solution")
        plt.show()

        plt.figure()
        plt.plot(np.log(np.array(self.opt_results.gradient_infty_hist)))
        plt.title("Logarithm of infinity norm of gradient")
        plt.show()

        plt.figure()
        plt.plot(np.log(np.array(self.opt_results.energy_hist)))
        plt.title("Logarithm of cost function value")
        plt.show()

    def save_results_to_file(self, path):
        lw = 1
        b = (0, 219 / 255, 1)
        r = (.89, 0, 0)
        self.j(self.q_opt)
        plot(self.exact_domain.mesh, linewidth=lw, color=b)
        plt.savefig(path + "exact_domain.pdf", bbox_inches="tight", pad_inches=0)
        plt.clf()
        plot(self.optimization_domain.mesh, linewidth=lw, color=r)
        plt.savefig(path + "estimated_domain.pdf", bbox_inches="tight", pad_inches=0)
        plt.clf()
        plt.plot(np.log(np.array(self.opt_results.gradient_infty_hist)), color=b)
        plt.savefig(path + "gradient_infty_norm.pdf", bbox_inches="tight", pad_inches=0)
        plt.clf()
        plt.plot(np.log(np.array(self.opt_results.energy_hist)), color=b)
        plt.savefig(path + "cost_function.pdf", bbox_inches="tight", pad_inches=0)
        plt.clf()
        self.j(Function(self.V_sph))
        plot(self.optimization_domain.mesh, linewidth=lw, color=r)
        plt.savefig(path + "initial_domain.pdf", bbox_inches="tight", pad_inches=0)
        plt.clf()

        self.plot_boundary(path)

        zero_function = Function(self.V_vol)  # for mesh visualization
        mesh_file = File(path + "final_mesh.pvd")
        mesh_file << zero_function

        coefficients_list_v = []
        coefficients_list_w = []

        for v, w in zip(self.v_equation.solution_list, self.w_equation.solution_list):
            coefficients_list_v.append(v.vector()[:])
            coefficients_list_w.append(w.vector()[:])

        results_dict = {
            "final_cost_function_value": self.opt_results.fun,
            "final_gradient_infty_norm": np.max(np.abs(self.opt_results.jac)),
            "duration_minutes": self.duration,
            "iterations": self.opt_results.nit,
            "message": self.opt_results.message,
            "success": self.opt_results.success,
            "function_evaluations": self.opt_results.nfev,
            "jacobian_evaluations": self.opt_results.njev,
            "cost_function_value_history": np.array(self.opt_results.energy_hist),
            "gradient_infty_norm_history": np.array(self.opt_results.gradient_infty_hist),
            "v": coefficients_list_v,
            "w": coefficients_list_w
        }

        with open(path + "results" + '.pickle', 'wb') as handle:
            pickle.dump(results_dict, handle)

        logging.info("Results and plots were saved")

    def save_exact_data(self, path):

        coefficients_list_ex = []

        for u in self.exact_pde.solution_list:
            coefficients_list_ex.append(u.vector()[:])

        reusables_dict = {
            "u_ex": coefficients_list_ex,
            "q_ex": self.q_ex.vector()[:],
            "W_ex": self.W_ex.vector()[:]
        }

        with open(path + "reusables" + '.pickle', 'wb') as handle:
            pickle.dump(reusables_dict, handle)

        logging.info("Exact data successfully saved")

    def plot_boundary(self, path):
        lw = 1
        b = (0, 219 / 255, 1)
        r = (.89, 0, 0)

        self.j(self.q_opt)
        u = Function(self.V_vol)

        dbc_int = DirichletBC(self.V_vol, 1, self.optimization_domain.facet_function, 3)
        dbc_int.apply(u.vector())

        int_dofs = u.vector()[:] > .5

        dbc_ext = DirichletBC(self.V_vol, -1, self.optimization_domain.facet_function, 2)
        dbc_ext.apply(u.vector())

        ext_dofs = u.vector()[:] < -.5

        mesh_coords_dofs = self.optimization_domain.mesh.coordinates()[dof_to_vertex_map(self.V_vol), ...]

        Bxytmp = mesh_coords_dofs[int_dofs]
        Btheta = np.arctan2(Bxytmp[:, 1], Bxytmp[:, 0])
        Binds = np.argsort(Btheta)
        Bxy = np.vstack((Bxytmp[Binds, :], Bxytmp[Binds[0], :]))
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect("equal", "box")
        ax.plot(Bxy[:, 0], Bxy[:, 1], '-', color=r, linewidth=lw)

        Bxytmp = mesh_coords_dofs[ext_dofs]
        Btheta = np.arctan2(Bxytmp[:, 1], Bxytmp[:, 0])
        Binds = np.argsort(Btheta)
        Bxy = np.vstack((Bxytmp[Binds, :], Bxytmp[Binds[0], :]))
        ax.plot(Bxy[:, 0], Bxy[:, 1], '-', color=b, linewidth=lw)

        self.j(interpolate(self.q_ex, self.V_sph))

        mesh_coords_dofs = self.optimization_domain.mesh.coordinates()[dof_to_vertex_map(self.V_vol), ...]
        u = Function(self.V_vol)

        dbc_int = DirichletBC(self.V_vol, 1, self.optimization_domain.facet_function, 3)
        dbc_int.apply(u.vector())

        int_dofs = u.vector()[:] > .5

        Bxytmp = mesh_coords_dofs[int_dofs]
        Btheta = np.arctan2(Bxytmp[:, 1], Bxytmp[:, 0])
        Binds = np.argsort(Btheta)
        Bxy = np.vstack((Bxytmp[Binds, :], Bxytmp[Binds[0], :]))
        ax.plot(Bxy[:, 0], Bxy[:, 1], '-', color=b, linewidth=lw)

        plt.savefig(path + "comparison.pdf", bbox_inches="tight", pad_inches=0, transparent=True)

        self.j(self.q_opt)