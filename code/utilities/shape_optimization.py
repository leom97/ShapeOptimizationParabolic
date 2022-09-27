from dolfin import *
from dolfin_adjoint import *
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
import moola
from tqdm import tqdm

from utilities.meshing import AnnulusMesh, CircleMesh, SquareAnnulusMesh, EfficientAnnulusMesh, SmoothedSquareAnnulusMesh, sea_urchin
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
        elif domain_dict["type"] == "efficient_annulus":
            domain = EfficientAnnulusMesh(resolution=domain_dict["resolution"],
                                          path=self.problem_folder + subfolder,
                                          int_refinement=domain_dict["int_refinement"],
                                          ext_refinement=domain_dict["ext_refinement"],
                                          inner_radius=domain_dict["inner_radius"],
                                          outer_radius=domain_dict["outer_radius"],
                                          center=domain_dict["center"],
                                          xdmf_path=xdmf_path,
                                          power=domain_dict["power"]
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

    def save_firedrake_files(self, path):
        """
        We obtain some interesting Firedrake stuff and then save it to file
        """

        vtd = vertex_to_dof_map(self.u_ex.solution_list[-1].function_space())
        dtv = dof_to_vertex_map(self.u_ex.solution_list[-1].function_space())

        firedrake_dict = {"vertex_to_dof_map": vtd,
                          "dof_to_vertex_map": dtv}

        with open(path + "firedrake_data" + '.pickle', 'wb') as handle:
            pickle.dump(firedrake_dict, handle)



    def simulate_exact_pde(self, exact_pde_dict, reusables_dict=None):
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
                logging.warning("Be sure that the order of the markers in the configuration file is correct")
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

            logging.warning("A perturbation on the boundary data will yield a pertubation on the exact solution")
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
        # if simulated_geometry_dict["domain_resolution"] is not None:
        #     optimization_geometry_dict["domain"]["resolution"] = simulated_geometry_dict["domain_resolution"]
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

        logging.fatal("Remove perturbation from here, put it elsewhere")
        if self.exact_exterior_BC == "N":
            self.u_D.perturb(simulated_pde_dict["noise_level_on_exact_BC"])
        elif self.exact_exterior_BC == "D":
            self.u_N.perturb(simulated_pde_dict["noise_level_on_exact_BC"])

        # Generic set-up
        L1_vol = FiniteElement("Lagrange", self.optimization_domain.mesh.ufl_cell(), 1)
        self.V_vol = FunctionSpace(self.optimization_domain.mesh, L1_vol)

        self.v_equation = HeatEquation(efficient=True)
        self.w_equation = HeatEquation(efficient=False)

        self.v_equation.set_ODE_scheme(self.optimization_pde_dict["ode_scheme"])
        self.v_equation.verbose = True
        self.v_equation.set_time_discretization(self.T,
                                                N_steps=int(self.optimization_pde_dict["N_steps"]),
                                                relevant_mesh_size=self.exact_domain.mesh.hmax())

        self.w_equation.set_ODE_scheme(self.optimization_pde_dict["ode_scheme"])
        self.w_equation.verbose = True
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
        self.pre_assembled_BCs = {"ext_neumann": external_NBC, "ext_dirichlet": external_DBC}

    # def save_problem_data(self, path):
    #
    #     if self.optimization_dict is None:
    #         raise Exception("No optimization options were found")
    #
    #     pickles_path = path
    #     import os
    #     os.makedirs(path, exist_ok=False)
    #     with open(pickles_path + "exact_geometry_dict" + '.pickle', 'wb') as handle:
    #         pickle.dump(self.exact_geometry_dict, handle)
    #     with open(pickles_path + "exact_pde_dict" + '.pickle', 'wb') as handle:
    #         pickle.dump(self.exact_pde_dict, handle)
    #
    #     simulated_geometry_dict = {
    #         "domain_resolution": self.optimization_geometry_dict["domain"]["resolution"],
    #         "sphere_resolution": self.optimization_geometry_dict["sphere"]["resolution"],
    #     }
    #
    #     simulated_pde_dict = {
    #         "ode_scheme": self.optimization_pde_dict["ode_scheme"],
    #         "N_steps": self.optimization_pde_dict["N_steps"]
    #     }
    #
    #     with open(pickles_path + "simulated_geometry_dict" + '.pickle', 'wb') as handle:
    #         pickle.dump(simulated_geometry_dict, handle)
    #     with open(pickles_path + "simulated_pde_dict" + '.pickle', 'wb') as handle:
    #         pickle.dump(simulated_pde_dict, handle)
    #
    #     with open(pickles_path + "optimization_dict" + '.pickle', 'wb') as handle:
    #         pickle.dump(self.optimization_dict, handle)

    def initialize_from_data(self, path, regenerate_exact_data=True):
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
                # simulated_geometry_dict["additional_domain_data"]["reload_xdmf"] = False
                # exact_geometry_dict["domain"]["reload_xdmf"] = False
                # self.create_optimal_geometry(exact_geometry_dict)
                # self.simulate_exact_pde(exact_pde_dict)

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

        :param time_steps_factor: we can increase or decrease the timesteps, this is a debugging option
        :param disable_radial_parametrization: debugging option, the displacement field will be the optimization variable itself
        :param start_at_optimum: if True, the initial deformation is the interpolated optimum
        :return:
        """

        logging.info("Setting up the cost functional")

        L1_sph = FiniteElement("Lagrange", self.optimization_sphere.mesh.ufl_cell(), 1)

        self.V_sph = FunctionSpace(self.optimization_sphere.mesh, L1_sph)
        self.V_def = VectorFunctionSpace(self.optimization_domain.mesh, "Lagrange", 1)

        with stop_annotating():
            M = compute_spherical_transfer_matrix(self.V_vol, self.V_sph, p=self.optimization_domain.center)
            M2 = compute_radial_displacement_matrix(M, self.V_def, p=self.optimization_domain.center,
                                                    f_D=self.optimization_domain.boundary_radial_function)

        # Mesh movement
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

        # V = FunctionSpace(self.optimization_domain.mesh, "CG", 1)
        # u = Function(V)
        #
        # w = TrialFunction(V)
        # v = TestFunction(V)
        # bcs = DirichletBC(V, 0, "on_boundary")
        #
        # DX = Measure("dx", domain=self.optimization_domain.mesh)
        #
        # solve(inner(grad(w), grad(v)) * DX == v * DX, u, bcs)
        #
        # J = assemble(sin(u) * DX)
        #
        # Jhat = ReducedFunctional(J, [Control(self.q_opt)])
        # d = Function(self.V_sph)
        # d.vector()[:] = (np.random.rand(len(d.vector())) - .5) * 1e-5
        # td = taylor_to_dict(Jhat, self.q_opt, d)
        #
        # print(td['R2']['Rate'])
        # print(td['R2'])

        # PDEs definitions
        u0 = Function(self.V_vol)  # zero initial condition
        u_D_inner = Constant(0.0)  # zero inner BC

        logging.warning(
            "Working under the assumptions that the marker 2 is for the outer boundary, and 3 for the inner one")

        # Dirichlet state
        self.v_equation.set_mesh(self.optimization_domain.mesh, self.optimization_domain.facet_function, self.V_vol, order=1)
        self.v_equation.set_PDE_data(u0, marker_dirichlet=[2, 3],
                                     dirichlet_BC=[self.u_D, u_D_inner],
                                     pre_assembled_BCs={2: {"type": "dirichlet", "marker": 2,
                                                            "data": self.pre_assembled_BCs["ext_dirichlet"]}})

        # Dirichlet-Neumann state
        self.w_equation.set_mesh(self.optimization_domain.mesh, self.optimization_domain.facet_function, self.V_vol, order=1)
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

        # J+=1e-3*assemble(self.q_opt**2*dx(self.optimization_sphere.mesh)+1e-2*inner(grad(self.q_opt), grad(self.q_opt)) * dx(self.optimization_sphere.mesh))
        self.j = ReducedFunctional(J, Control(self.q_opt))

        return M, M2

    def do_taylor_test(self):
        if self.j is None:
            raise Exception("No reduced cost functional is available")
        h = Function(self.V_sph)
        h.vector()[:] = .1
        logging.info(taylor_to_dict(self.j, self.q_opt, h))

    def solve(self):

        def callback(x):
            # self.w_equation.solve()
            # self.v_equation.solve()
            # pl = plot(self.w_equation.solution_list[-1] - self.v_equation.solution_list[-1])
            # plt.colorbar(pl)
            plot(self.optimization_domain.mesh)
            plt.show()

        logging.info("Shape optimization starts now")
        import time
        duration = -time.time()

        if self.optimization_dict["solver"] == "scipy_BFGS":
            logging.warning("Optimization in the L2 scalar product")
            # the scipy way
            bounds = None
            self.q_opt, self.opt_results = minimize(self.j, tol=1e-6, options=self.optimization_dict["options"],
                                                    bounds=bounds,
                                                    callback=callback)
        elif self.optimization_dict["solver"] == "moola_BFGS":
            # the moola way
            # Set up moola problem and solve optimisation
            problem_moola = MoolaOptimizationProblem(self.j)

            m_moola = moola.DolfinPrimalVector(self.q_opt, inner_product=self.optimization_dict["inner_product"])
            solver = moola.BFGS(problem_moola, m_moola, options=self.optimization_dict["options"])

            self.q_opt, self.opt_results = solver.solve(callback=callback)

        elif self.optimization_dict["solver"] == "moola_newton":
            problem_moola = MoolaOptimizationProblem(self.j)

            m_moola = moola.DolfinPrimalVector(self.q_opt, inner_product=self.optimization_dict["inner_product"])
            solver = moola.NewtonCG(problem_moola, m_moola, options=self.optimization_dict["options"])

            self.q_opt, self.opt_results = solver.solve(callback=callback)
        else:
            raise Exception("Unsupported solver")

        duration += time.time()

        self.duration = duration / 60  # in minutes

        logging.info(f"{self.duration} minutes elapsed")

    def visualize_result(self):
        logging.info("Visualizing the geometries")

        plot(self.exact_domain.mesh, title="Exact solution")
        plt.show()

        plot(self.optimization_domain.mesh, title="Computed solution")
        plt.show()

        plt.plot(np.log(np.array(self.opt_results.gradient_infty_hist)))
        plt.title("Logarithm of infinity norm of gradient")
        plt.show()

        plt.plot(np.log(np.array(self.opt_results.energy_hist)))
        plt.title("Logarithm of cost function value")
        plt.show()

    def save_results_to_file(self, path):

        plot(self.exact_domain.mesh)
        plt.savefig(path + "exact_domain.png", bbox_inches="tight", pad_inches=0)
        plt.clf()
        plot(self.optimization_domain.mesh)
        plt.savefig(path + "estimated_domain.svg", bbox_inches="tight", pad_inches=0)
        plt.clf()
        plt.plot(np.log(np.array(self.opt_results.gradient_infty_hist)))
        plt.savefig(path + "gradient_infty_norm.svg", bbox_inches="tight", pad_inches=0)
        plt.clf()
        plt.plot(np.log(np.array(self.opt_results.energy_hist)))
        plt.savefig(path + "cost_function.svg", bbox_inches="tight", pad_inches=0)
        plt.clf()

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

        self.save_firedrake_files(path)

        logging.info("Exact data successfully saved")

    def reset(self, disable_radial_parametrization=False,
              start_at_optimum=False):  # resets the optimization to the initial conditions
        logging.info("Resetting the problem")
        self.initialize_optimization_domain(self.simulated_geometry_dict)
        self.create_cost_functional(disable_radial_parametrization=disable_radial_parametrization,
                                    start_at_optimum=start_at_optimum)

    def debug_generic(self, path):

        logging.info("Debugging")

        consistency_error = self.j(interpolate(self.q_ex, self.V_sph))

        # Solve equations on the perturbed domains, apparently the solution vectors are not automatically updated, after calling
        self.w_equation.solve()
        self.v_equation.solve()

        # logging.info(f"Energy at interpolated optimum is {consistency_error}")
        # derivative_at_optimum = self.j.derivative()
        # logging.info(f"Gradient at interpolated optimum:\n{derivative_at_optimum.vector()[:][:, None]}")
        #
        # File(path + "gradient_at_interpolated_optimum.pvd") << derivative_at_optimum
        #
        # boundary_mesh_optimization = BoundaryMesh(self.optimization_domain.mesh,
        #                                           "exterior")  # at the interpolated optimum
        # # Let's plot the boundary values of w, that should be close to the boundary values of u_ex
        # fw = File(path + "/w_dirichlet_plot/w_dirichlet_at_interpolated_optimum.pvd")
        # fu = File(path + "/u_dirichlet_plot/u_dirichlet.pvd")
        # Nw = len(self.w_equation.solution_list)
        # wb = adjf.mesh_to_boundary(self.w_equation.solution_list[0], boundary_mesh_optimization)
        # ub = adjf.mesh_to_boundary(self.w_equation.solution_list[0], boundary_mesh_optimization)
        #
        # ext_boundary_errors = []
        # mf = self.w_equation.facet_indicator
        # logging.info("Computing L2 norms of w-u on the boundary")
        # for i in tqdm(range(Nw)):
        #     w = self.w_equation.solution_list[i]
        #
        #     # Put everything to zero, but on the external boundary
        #     ww = Function(self.V_vol)
        #     dbc = DirichletBC(self.V_vol, w, mf, 2)
        #     dbc.apply(ww.vector())
        #
        #     wb.vector()[:] = (adjf.mesh_to_boundary(ww, boundary_mesh_optimization)).vector()[:]
        #     fw << wb
        #
        #     self.u_ex.t = self.w_equation.times[i]
        #     u = interpolate(self.u_ex, self.V_vol)
        #
        #     # Let's also put everything to zero here
        #     uu = Function(self.V_vol)
        #     dbc = DirichletBC(self.V_vol, u, mf, 2)
        #     dbc.apply(uu.vector())
        #
        #     ub.vector()[:] = (adjf.mesh_to_boundary(uu, boundary_mesh_optimization)).vector()[:]
        #
        #     fu << ub
        #
        #     ext_boundary_errors.append(errornorm(self.u_ex, w))
        #
        # logging.info(f"L2 norms of w-u on the boundary:\n{np.array(ext_boundary_errors)[:, None]}")
        #
        # logging.info("Studying PDEs discretization error")
        # e_l2t = []
        # a = 9
        # b = 13
        # sol_ie_past = []
        # sol_cn_past = []
        # change_ie = []
        # change_cn = []
        # for i in range(a, b):
        #     self.w_equation.set_ODE_scheme("implicit_euler")
        #     self.w_equation.set_time_discretization(self.T, int(2 ** i))
        #     self.w_equation.solve()
        #     sol_ie = []
        #     for w in self.w_equation.solution_list:
        #         u = Function(self.V_vol)
        #         u.assign(w)
        #         sol_ie.append(u)
        #     self.w_equation.set_ODE_scheme("crank_nicolson")
        #     self.w_equation.set_time_discretization(self.T, int(2 ** i))
        #     self.w_equation.solve()
        #     sol_cn = []
        #     for w in self.w_equation.solution_list:
        #         u = Function(self.V_vol)
        #         u.assign(w)
        #         sol_cn.append(u)
        #     e = 0
        #     for k in range(len(sol_ie)):
        #         e += self.w_equation.dts[0] * assemble((sol_ie[k] - sol_cn[k]) ** 2 * dx(self.optimization_domain.mesh))
        #     e_l2t.append(e)
        #     if i > a:
        #         rel_change_ie = 0
        #         rel_change_cn = 0
        #         for j in range(2 ** i // 2):
        #             rel_change_ie += self.w_equation.dts[0] * assemble(
        #                 (sol_ie[2 * j] - sol_ie_past[j]) ** 2 * dx(self.optimization_domain.mesh))
        #             rel_change_cn += self.w_equation.dts[0] * assemble(
        #                 (sol_cn[2 * j] - sol_cn_past[j]) ** 2 * dx(self.optimization_domain.mesh))
        #         change_ie.append(rel_change_ie)
        #         change_cn.append(rel_change_cn)
        #
        #     sol_ie_past = sol_ie
        #     sol_cn_past = sol_cn

        logging.info("Studying error behaviour")
        a = 9
        b = 10
        coarse_sol = []
        for i in range(a, b):
            self.v_equation.set_ODE_scheme("implicit_euler")
            self.v_equation.set_time_discretization(self.T, int(self.T * 2 ** i))
            ts = self.v_equation.times
            self.v_equation.set_time_discretization(self.T, custom_times_array=ts)
            self.v_equation.solve()
            for w in self.v_equation.solution_list:
                u = Function(self.V_vol)
                u.assign(w)
                coarse_sol.append(u)
        un = TimeExpressionFromList(0, self.v_equation.times, coarse_sol)

        a = 10
        b = 11
        esol = []
        for i in range(a, b):
            self.v_equation.set_ODE_scheme("crank_nicolson")
            self.v_equation.set_time_discretization(self.T, int(self.T * 2 ** i))
            self.v_equation.solve()
            for w in self.v_equation.solution_list:
                u = Function(self.V_vol)
                u.assign(w)
                esol.append(u)
        ue = TimeExpressionFromList(0, self.v_equation.times, esol)

        data = []
        point = [-1.5, 0]
        for t in un.discrete_times:
            ue.t = t
            un.t = t
            data.append(-ue(*point) + un(*point))

        plt.plot(ts, data)
        plt.show()

        self.reset()
