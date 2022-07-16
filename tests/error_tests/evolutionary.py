from dolfin import *
from dolfin_adjoint import *
import numpy as np
import logging
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# %% Setting log parameters

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

# %% Preliminary variables

mesh_path = "/home/leonardo_mutti/PycharmProjects/masters_thesis/meshes/annulus/"
pickles_path = "/home/leonardo_mutti/PycharmProjects/masters_thesis/pde_data/pickles/test_problems/"

resolutions = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625]  # mesh resolutions


# %% The PDE class

class PDE:

    def __init__(self):
        self.implemented_time_schemes = ["implicit_euler", "crank_nicolson", "crank_nicolson_midpoint",
                                         "implicit_explicit_euler"]
        self.verbose = False

    def read_mesh(self, resolution, mesh_path):
        # The volumetric mesh
        mesh = Mesh()
        with XDMFFile(mesh_path + "mesh_" + str(resolution) + ".xdmf") as infile:
            infile.read(mesh)

        # The boundary meshes
        mvc = MeshValueCollection("size_t", mesh, 1)  # 1 means: we consider lines, 1D things
        with XDMFFile(mesh_path + "facet_mesh_" + str(resolution) + ".xdmf") as infile:
            infile.read(mvc, "name_to_read")
        mf = MeshFunction("size_t", mesh, mvc)  # remember, tag 3 is inner ring, tag 2 outer ring

        self.mesh = mesh
        self.facet_indicator = mf

        self.set_function_space()
        self.compute_domain_volume()

    def set_ODE_scheme(self, ode_scheme):
        if ode_scheme not in self.implemented_time_schemes:
            raise Exception("Unrecognized time stepping")
        self.ode_scheme = ode_scheme

    def set_time_discretization(self, final_time, N_steps=None, custom_times_array=None):
        '''
        Note, one of N_steps and custom_time_array must be provided
        :param final_time: final time of simulation
        :param N_steps: optional, if present, we will use uniform timestepping with dt = final_time/N_steps
        :param custom_times_array: we can also set a custom time discretization of the form np.array([0, ..., final_time])
        :return:
        '''

        if final_time <= 0:
            raise Exception("Final time must be positive")
        self.T = final_time

        if N_steps is None and custom_times_array is None:
            logging.warning("No time discretization provided, the default will be created")
        elif N_steps is not None and custom_times_array is not None:
            raise Exception("Over specification of the time discretization mode")
        elif N_steps is not None:
            if not isinstance(N_steps, int) or N_steps <= 0:
                raise Exception("Invalid number of time steps (non positive or non integer)")
            self.times = np.linspace(0, self.T, N_steps + 1)
            dt = self.T / N_steps
            if dt < DOLFIN_EPS:
                raise Exception("Negative or too small time step")
            self.dts = dt * np.ones(N_steps)  # we have N_steps + 1 points, so, N_steps intervals
        else:
            if not isinstance(custom_times_array, np.ndarray):
                raise Exception("The array of custom times must be a numpy array")
            if custom_times_array[0] > DOLFIN_EPS or custom_times_array[0] < -DOLFIN_EPS:
                raise Exception("The initial time must be 0")
            if custom_times_array[-1] > self.T or custom_times_array[-1] < self.T - DOLFIN_EPS:
                raise Exception("The final time must be T")

            dts = np.diff(custom_times_array)
            if np.min(dts) < DOLFIN_EPS:
                raise Exception("Too small time step")

            self.dts = dts
            self.times = custom_times_array

    def set_PDE_data(self, initial_value, source=None, marker_neumann=None, marker_dirichlet=None, neumann_BC=None,
                     dirichlet_BC=None,
                     exact_solution=None):
        '''
        :param source: right hand side f in u_t - \Delta u = f. It can be either a function on self.S1h or an expression
        :param marker_neumann: with reference to self.mf, it is the integer describing the part of the boundary where we
        put Neumann BCs
        :param marker_dirichlet: same thing
        :param neumann_BC: the Neumann data. It can be either a function on self.S1h or an expression
        :param dirichlet_BC: the Dirichlet data. Same thing.
        :param exact_solution: the exact solution to the PDE. Same thing.
        :return:
        '''

        if marker_neumann is None and marker_dirichlet is None:
            raise Exception("Some boundary conditions must be specified")
        if (marker_neumann is not None and neumann_BC is None) or (
                marker_dirichlet is not None and dirichlet_BC is None):
            raise Exception("No boundary data supplied")
        if marker_dirichlet is not None:
            if not isinstance(marker_dirichlet, list):
                raise Exception("The Dirichlet marker should be a list")
            for v in marker_dirichlet:
                if not isinstance(v, int):
                    raise Exception("The Dirichlet marker should be an integer")
        if marker_neumann is not None:
            if not isinstance(marker_neumann, list):
                raise Exception("The Neumann marker should be a list")
            for v in marker_neumann:
                if not isinstance(v, int):
                    raise Exception("The Neumann marker should be an integer")

        self.initial_value = initial_value

        # Some values may be None
        self.f = source
        self.f_D = dirichlet_BC
        self.f_N = neumann_BC
        self.dirichlet_marker = marker_dirichlet
        self.neumann_marker = marker_neumann
        self.exact_solution = exact_solution

    def set_function_space(self):
        if not hasattr(self, "mesh"):
            raise Exception("A mesh needs to be defined at first")

        L1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)  # only linear FEM for now
        self.S1h = FunctionSpace(self.mesh, L1)
        self.FEM = L1

    def compute_domain_volume(self):

        if not hasattr(self, "S1h"):
            raise Exception("The function space is not defined")
        if not hasattr(self, "mesh"):
            raise Exception("No mesh was defined")

        self.domain_measure = assemble(Constant(1.0) * dx(self.mesh))

    def get_instant_variational_formulation(self, time_index, last_solution):
        '''
        :param time_index: we call this method at a certain time, and the instant variational formulation between times
        might vary. time_index runs from 1 to len(self.time_steps), and we return the variational formulation to get the
         solution at self.time_steps[time_index]
        :param last_solution: the solution of the last time step
        :return: the variational formulation of the time dependent PDE, discretized with the ODE scheme of choice, at a
        specific time instant
        '''

        if not hasattr(self, "ode_scheme"):
            raise Exception("Set an ODE scheme first")

        if not isinstance(time_index, int):
            raise Exception("Time index must be an integer")
        if time_index <= 0:
            raise Exception("There is no need to have a time step for/before the initial condition")
        if time_index >= len(self.times) or time_index < 0:
            raise Exception("Time index out of range")

        # We are now sure to be at time > 0, and that the ODE scheme was set
        if self.ode_scheme not in self.implemented_time_schemes:
            raise Exception("Unrecognized time stepping")
        else:
            ode_scheme_index = self.implemented_time_schemes.index(self.ode_scheme)

        # Trial and test functions
        if not hasattr(self, "S1h"):
            raise Exception("The finite element space must be defined")

        u = TrialFunction(self.S1h)
        v = TestFunction(self.S1h)
        dx = Measure("dx", self.mesh)

        # Choose the correct time scheme
        match ode_scheme_index:

            case 0:  # implicit euler
                dt = Constant(self.dts[time_index - 1])
                t = self.times[time_index]

                a = u * v * dx + dt * inner(grad(u), grad(v)) * dx
                L = last_solution * v * dx

                if self.neumann_marker is not None:
                    self.f_N.t = t
                    L = L + dt * self.f_N * v * ds

                if self.f is not None:
                    self.f.t = t
                    L = L + dt * self.f * v * dx

                if self.dirichlet_marker is not None:
                    self.f_D.t = t

                return a, L

            case 1:  # CN (standard)
                dt = Constant(self.dts[time_index - 1])
                t = self.times[time_index]
                t_prev = self.times[time_index - 1]

                a = u * v * dx + dt / 2 * inner(grad(u), grad(v)) * dx
                L = last_solution * v * dx - dt / 2 * inner(grad(last_solution), grad(v)) * dx

                if self.neumann_marker is not None:
                    self.f_N.t = (t + t_prev) / 2
                    L = L + dt * self.f_N * v * ds

                if self.f is not None:
                    self.f.t = (t + t_prev) / 2
                    L = L + dt * self.f * v * dx

                if self.dirichlet_marker is not None:
                    self.f_D.t = t

                return a, L

            case 3:  # implicit-explicit euler
                dt = Constant(self.dts[time_index - 1])
                t = self.times[time_index]
                t_last = self.times[time_index - 1]

                a = u * v * dx + dt * inner(grad(u), grad(v)) * dx
                L = last_solution * v * dx

                if self.neumann_marker is not None:
                    self.f_N.t = t_last
                    L = L + dt * self.f_N * v * ds

                if self.f is not None:
                    self.f.t = t_last
                    L = L + dt * self.f * v * dx

                if self.dirichlet_marker is not None:
                    self.f_D.t = t_last # this could be changed, depending the kind of consistency want

                return a, L

    def solve_single_step(self, time_index, last_solution=None, err_mode="none"):

        if not isinstance(time_index, int):
            raise Exception("Time index must be an integer")
        if time_index >= len(self.times) or time_index < 0:
            raise Exception("Time index out of range")

        if time_index == 0:
            current_solution = interpolate(self.initial_value, self.S1h)
            if err_mode != "none":
                current_error = self.compute_instant_error(err_mode, current_solution, time_index)
                return current_solution, current_error
            else:
                return current_solution, None
        elif last_solution is None:
            raise Exception("The previous time solution was not provided")

        a, L = self.get_instant_variational_formulation(time_index, last_solution)
        current_solution = Function(self.S1h)

        # Dirichlet BCs
        dirichlet_BC = []
        if self.dirichlet_marker is not None:
            # self.f_D.t = self.times[time_index] # this is already taken care of in get_instant_variational_formulation
            for v in self.dirichlet_marker:
                dirichlet_BC.append(DirichletBC(self.S1h, self.f_D, self.facet_indicator, v))

        solve(a == L, current_solution,
              dirichlet_BC)  # dirichlet BC might be empty. Also, they're imposed at possibily a different time than the Neumann conditions, in the case of CN

        # Error computation
        if err_mode != "none":
            current_error = self.compute_instant_error(err_mode, current_solution, time_index)
            return current_solution, current_error
        else:
            return current_solution, None

    def compute_instant_error(self, err_mode, current_solution, time_index):
        if not hasattr(self, "exact_solution"):
            raise Exception("No exact solution is available for error computation")
        if self.exact_solution is None:
            raise Exception("No exact solution is available for error computation")
        else:
            self.exact_solution.t = self.times[time_index]
        match err_mode:
            case "l2":
                error = errornorm(self.exact_solution, current_solution) / self.domain_measure
                return error
            case "linf":
                u_ex_interp = interpolate(self.exact_solution, self.S1h)
                error = np.max(np.abs(u_ex_interp.vector()[:] - current_solution.vector()[:]))
                return error
            case _:
                raise Exception("Error norm not implemented")

    def solve(self, err_mode="none"):

        solution_list = []
        error_list = []
        last_solution = Function(self.S1h)   # container for the last solution

        first_solution, first_error = self.solve_single_step(0, err_mode=err_mode)
        solution_list.append(first_solution)
        error_list.append(first_error)
        last_solution.assign(first_solution)    # no need to create a copy of first solution

        for time_step in tqdm(range(1, len(self.times))):
            current_solution, current_error = self.solve_single_step(time_step, last_solution, err_mode=err_mode)
            solution_list.append(current_solution)
            error_list.append(current_error)
            last_solution.assign(current_solution)  # no need for current_solution.copy()

        if err_mode == "none":
            return solution_list, None
        else:
            return solution_list, error_list


# %% Some particular test sets

class TestProblems:

    def __init__(self):
        self.marker_dirichlet = [3]
        self.marker_neumann = [2]
        self.T = 2
        self.override_pickle = True

    def get_data(self, problem_name):
        match problem_name:

            # We are given everything correct, the only problem is that this HEQ has 0th but not 1st compatibility
            case "no_first_compatibility":
                class solution(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = np.sqrt(self.t) * (x[0] ** 2 + x[1] ** 2)

                    def value_shape(self):
                        return ()

                class source(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = 1 / (2 * np.sqrt(self.t)) * (x[0] ** 2 + x[1] ** 2) - 4 * np.sqrt(self.t)

                    def value_shape(self):
                        return ()

                class initial_solution(UserExpression):
                    def __init__(self, **kwargs):
                        super().__init__(self, **kwargs)

                    def eval(self, values, x):
                        values[0] = 0

                    def value_shape(self):
                        return ()

                class neumann_trace(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        if (x[0] ** 2 + x[1] ** 2) < 1.5 ** 2:
                            values[0] = -2 * np.sqrt(self.t)
                        else:
                            values[0] = 4 * np.sqrt(self.t)

                    def value_shape(self):
                        return ()

                u_ex = solution(t=0)
                f = source(t=0)
                u0 = initial_solution()
                u_D = solution(t=0)
                u_N = neumann_trace(t=0)

            # We are given everything correct, the only problem is that this HEQ has 0th, 1st but not 2nd compatibility
            case "no_second_compatibility":
                class solution(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = np.sqrt(self.t) ** 3 * (x[0] ** 2 + x[1] ** 2)

                    def value_shape(self):
                        return ()

                class source(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = 3 / 2 * np.sqrt(self.t) * (x[0] ** 2 + x[1] ** 2) - 4 * np.sqrt(self.t) ** 3

                    def value_shape(self):
                        return ()

                class initial_solution(UserExpression):
                    def __init__(self, **kwargs):
                        super().__init__(self, **kwargs)

                    def eval(self, values, x):
                        values[0] = 0

                    def value_shape(self):
                        return ()

                class neumann_trace(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        if (x[0] ** 2 + x[1] ** 2) < 1.5 ** 2:
                            values[0] = -2 * np.sqrt(self.t) ** 3
                        else:
                            values[0] = 4 * np.sqrt(self.t) ** 3

                    def value_shape(self):
                        return ()

                u_ex = solution(t=0)
                f = source(t=0)
                u0 = initial_solution()
                u_D = solution(t=0)
                u_N = neumann_trace(t=0)

            # Everything is good
            case "continuous":
                class solution(UserExpression):
                    def __init__(self, t, w, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t
                        self.w = w

                    def eval(self, values, x):
                        R = x[0] ** 2 + x[1] ** 2
                        values[0] = R * cos(self.w * self.t)

                    def value_shape(self):
                        return ()

                class source(UserExpression):
                    def __init__(self, t, w, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t
                        self.w = w

                    def eval(self, values, x):
                        R = x[0] ** 2 + x[1] ** 2
                        values[0] = -4 * cos(self.w * self.t) - self.w * R * sin(self.w * self.t)

                    def value_shape(self):
                        return ()

                class initial_solution(UserExpression):
                    def __init__(self, **kwargs):
                        super().__init__(self, **kwargs)

                    def eval(self, values, x):
                        values[0] = x[0] ** 2 + x[1] ** 2

                    def value_shape(self):
                        return ()

                class neumann_trace(UserExpression):
                    def __init__(self, t, w, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t
                        self.w = w

                    def eval(self, values, x):
                        R = x[0] ** 2 + x[1] ** 2
                        if R <= 1.5 ** 2:
                            values[0] = - 2 * cos(self.w * self.t)
                        else:
                            values[0] = 4 * cos(self.w * self.t)

                    def value_shape(self):
                        return ()

                w = 0
                u_ex = solution(t=0, w=w)
                f = source(t=0, w=w)
                u0 = initial_solution()
                u_D = solution(t=0, w=w)
                u_N = neumann_trace(t=0, w=w)

            case "constant":
                class solution(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = 1

                    def value_shape(self):
                        return ()

                class source(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = 0

                    def value_shape(self):
                        return ()

                class initial_solution(UserExpression):
                    def __init__(self, **kwargs):
                        super().__init__(self, **kwargs)

                    def eval(self, values, x):
                        values[0] = 1

                    def value_shape(self):
                        return ()

                class neumann_trace(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = 0

                    def value_shape(self):
                        return ()

                u_ex = solution(t=0)
                f = source(t=0)
                u0 = initial_solution()
                u_D = solution(t=0)
                u_N = neumann_trace(t=0)

            case "linear_in_time":
                class solution(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = self.t

                    def value_shape(self):
                        return ()

                class source(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = 1

                    def value_shape(self):
                        return ()

                class initial_solution(UserExpression):
                    def __init__(self, **kwargs):
                        super().__init__(self, **kwargs)

                    def eval(self, values, x):
                        values[0] = 0

                    def value_shape(self):
                        return ()

                class neumann_trace(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = 0

                    def value_shape(self):
                        return ()

                u_ex = solution(t=0)
                f = source(t=0)
                u0 = initial_solution()
                u_D = solution(t=0)
                u_N = neumann_trace(t=0)

            case "quadratic_in_time":
                class solution(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = self.t ** 2 / 2

                    def value_shape(self):
                        return ()

                class source(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = self.t

                    def value_shape(self):
                        return ()

                class initial_solution(UserExpression):
                    def __init__(self, **kwargs):
                        super().__init__(self, **kwargs)

                    def eval(self, values, x):
                        values[0] = 0

                    def value_shape(self):
                        return ()

                class neumann_trace(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = 0

                    def value_shape(self):
                        return ()

                u_ex = solution(t=0)
                f = source(t=0)
                u0 = initial_solution()
                u_D = solution(t=0)
                u_N = neumann_trace(t=0)

            case "no_source_dirichlet_neumann_smooth":  # actually this satisfies RC 0, 1, 2 but not 3

                class dirichlet_data(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = 0

                    def value_shape(self):
                        return ()

                class initial_solution(UserExpression):
                    def __init__(self, **kwargs):
                        super().__init__(self, **kwargs)

                    def eval(self, values, x):
                        values[0] = 0

                    def value_shape(self):
                        return ()

                class neumann_trace(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = self.t ** 2

                    def value_shape(self):
                        return ()

                u0 = initial_solution()
                u_D = dirichlet_data(t=0)
                u_N = neumann_trace(t=0)
                f = None
                u_ex = None
                self.marker_dirichlet = [3]
                self.marker_neumann = [2]

                # Now, let's simulate the exact solution on a fine mesh and on a fine time scale
                heq = PDE()
                heq.read_mesh(resolutions[5], mesh_path)

                N_steps = int(np.ceil(self.T / (heq.mesh.hmax() ** 2)))  # many many time steps!
                heq.set_time_discretization(self.T, N_steps=N_steps)  # dt = h^2 and N = T/dt

                try:
                    if self.override_pickle:  # I lazily generate an error to go into the except
                        1 / 0
                    solution_list = self.from_pickle(heq.S1h, "no_source_dirichlet_neumann_smooth")
                except:
                    heq.set_PDE_data(u0, source=f, marker_neumann=self.marker_neumann,
                                     marker_dirichlet=self.marker_dirichlet,
                                     neumann_BC=u_N,
                                     dirichlet_BC=u_D,
                                     exact_solution=u_ex)  # note, no copy is done, the attributes of heq are EXACTLY these guys
                    heq.set_ODE_scheme("crank_nicolson")
                    heq.verbose = True

                    logging.info(f"Simulating exact solution for test problem")
                    solution_list, _ = heq.solve()

                    # To pickle
                    self.to_pickle(solution_list, "no_source_dirichlet_neumann_smooth")

                # Now, create an expression from solution list, hopefully it works

                u_ex = self.create_solution(0, heq.times, solution_list)

            case "no_source_dirichlet_neumann_no_compatibility":  # no compatibility at all

                class dirichlet_data(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = 0

                    def value_shape(self):
                        return ()

                class initial_solution(UserExpression):
                    def __init__(self, **kwargs):
                        super().__init__(self, **kwargs)

                    def eval(self, values, x):
                        values[0] = 1

                    def value_shape(self):
                        return ()

                class neumann_trace(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = self.t ** 2

                    def value_shape(self):
                        return ()

                u0 = initial_solution()
                u_D = dirichlet_data(t=0)
                u_N = neumann_trace(t=0)
                f = None
                u_ex = None
                self.marker_dirichlet = [3]
                self.marker_neumann = [2]

                # Now, let's simulate the exact solution on a fine mesh and on a fine time scale
                heq = PDE()
                heq.read_mesh(resolutions[5], mesh_path)

                N_steps = int(np.ceil(self.T / (heq.mesh.hmax() ** 2)))  # many many time steps!
                heq.set_time_discretization(self.T, N_steps=N_steps)  # dt = h^2 and N = T/dt

                pickle_name = "no_source_dirichlet_neumann_no_compatibility"
                try:
                    if self.override_pickle:  # I lazily generate an error to go into the except
                        1 / 0
                    solution_list = self.from_pickle(heq.S1h, pickle_name)
                except:
                    heq.set_PDE_data(u0, source=f, marker_neumann=self.marker_neumann,
                                     marker_dirichlet=self.marker_dirichlet,
                                     neumann_BC=u_N,
                                     dirichlet_BC=u_D,
                                     exact_solution=u_ex)  # note, no copy is done, the attributes of heq are EXACTLY these guys
                    heq.set_ODE_scheme("crank_nicolson")
                    heq.verbose = True

                    logging.info(f"Simulating exact solution for test problem")
                    solution_list, _ = heq.solve()

                    # To pickle
                    self.to_pickle(solution_list, pickle_name)

                u_ex = self.solution(0, heq.times, solution_list)

            case "no_source_dirichlet_neumann_zero_compatibility":  # 0 but not 1 compatibility

                class dirichlet_data(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = 0

                    def value_shape(self):
                        return ()

                class initial_solution(UserExpression):
                    def __init__(self, **kwargs):
                        super().__init__(self, **kwargs)

                    def eval(self, values, x):
                        values[0] = 1 - x[0] ** 2 - x[1] ** 2

                    def value_shape(self):
                        return ()

                class neumann_trace(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = self.t ** 2

                    def value_shape(self):
                        return ()

                u0 = initial_solution()
                u_D = dirichlet_data(t=0)
                u_N = neumann_trace(t=0)
                f = None
                u_ex = None
                self.marker_dirichlet = [3]
                self.marker_neumann = [2]

                # Now, let's simulate the exact solution on a fine mesh and on a fine time scale
                heq = PDE()
                heq.read_mesh(resolutions[5], mesh_path)

                N_steps = int(np.ceil(self.T / (heq.mesh.hmax() ** 2)))  # many many time steps!
                heq.set_time_discretization(self.T, N_steps=N_steps)  # dt = h^2 and N = T/dt

                pickle_name = "no_source_dirichlet_neumann_zero_compatibility"
                try:
                    if self.override_pickle:  # I lazily generate an error to go into the except
                        1 / 0
                    solution_list = self.from_pickle(heq.S1h, pickle_name)
                except:
                    heq.set_PDE_data(u0, source=f, marker_neumann=self.marker_neumann,
                                     marker_dirichlet=self.marker_dirichlet,
                                     neumann_BC=u_N,
                                     dirichlet_BC=u_D,
                                     exact_solution=u_ex)  # note, no copy is done, the attributes of heq are EXACTLY these guys
                    heq.set_ODE_scheme("crank_nicolson")
                    heq.verbose = True

                    logging.info(f"Simulating exact solution for test problem")
                    solution_list, _ = heq.solve()

                    # To pickle
                    self.to_pickle(solution_list, pickle_name)

                u_ex = self.create_solution(0, heq.times, solution_list)

            case "no_source_dirichlet_neumann_no_compatibility_difficult":  # same as non difficult, but with spatially varying BCs

                class dirichlet_data(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = np.sin(x[1]) * np.cos(4 * self.t)

                    def value_shape(self):
                        return ()

                class initial_solution(UserExpression):
                    def __init__(self, **kwargs):
                        super().__init__(self, **kwargs)

                    def eval(self, values, x):
                        values[0] = 1 - x[0] ** 2 - x[1] ** 2

                    def value_shape(self):
                        return ()

                class neumann_trace(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = (self.t ** 2) * x[1]

                    def value_shape(self):
                        return ()

                u0 = initial_solution()
                u_D = dirichlet_data(t=0)
                u_N = neumann_trace(t=0)
                f = None
                u_ex = None
                self.marker_dirichlet = [3]
                self.marker_neumann = [2]

                # Now, let's simulate the exact solution on a fine mesh and on a fine time scale
                heq = PDE()
                heq.read_mesh(resolutions[6], mesh_path)

                N_steps = int(np.ceil(self.T / (heq.mesh.hmax() ** 1)))  # many many time steps!
                heq.set_time_discretization(self.T, N_steps=N_steps)  # dt = h^2 and N = T/dt

                pickle_name = "no_source_dirichlet_neumann_no_compatibility_difficult"
                try:
                    if self.override_pickle:  # I lazily generate an error to go into the except
                        1 / 0
                    solution_list = self.from_pickle(heq.S1h, pickle_name)
                except:
                    heq.set_PDE_data(u0, source=f, marker_neumann=self.marker_neumann,
                                     marker_dirichlet=self.marker_dirichlet,
                                     neumann_BC=u_N,
                                     dirichlet_BC=u_D,
                                     exact_solution=u_ex)  # note, no copy is done, the attributes of heq are EXACTLY these guys
                    heq.set_ODE_scheme("crank_nicolson")
                    heq.verbose = True

                    logging.info(f"Simulating exact solution for test problem")
                    solution_list, _ = heq.solve()

                    # To pickle
                    self.to_pickle(solution_list, pickle_name)

                u_ex = self.create_solution(0, heq.times, solution_list)

            case "no_source_dirichlet_neumann_first_compatibility_difficult":   # RC 0, 1 and not 2

                class dirichlet_data(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = np.sin(x[1]) * (self.t**2)

                    def value_shape(self):
                        return ()

                class initial_solution(UserExpression):
                    def __init__(self, **kwargs):
                        super().__init__(self, **kwargs)

                    def eval(self, values, x):
                        values[0] = 0

                    def value_shape(self):
                        return ()

                class neumann_trace(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = self.t * x[1]

                    def value_shape(self):
                        return ()

                u0 = initial_solution()
                u_D = dirichlet_data(t=0)
                u_N = neumann_trace(t=0)
                f = None
                u_ex = None
                self.marker_dirichlet = [3]
                self.marker_neumann = [2]

                # Now, let's simulate the exact solution on a fine mesh and on a fine time scale
                heq = PDE()
                heq.read_mesh(resolutions[5], mesh_path)

                N_steps = int(np.ceil(2 / (heq.mesh.hmax() ** 2)))  # many many time steps!
                heq.set_time_discretization(self.T, N_steps=N_steps)  # dt = h^2 and N = T/dt

                pickle_name = "no_source_dirichlet_neumann_no_compatibility_difficult"
                try:
                    if self.override_pickle:  # I lazily generate an error to go into the except
                        1 / 0
                    solution_list = self.from_pickle(heq.S1h, pickle_name)
                except:
                    heq.set_PDE_data(u0, source=f, marker_neumann=self.marker_neumann,
                                     marker_dirichlet=self.marker_dirichlet,
                                     neumann_BC=u_N,
                                     dirichlet_BC=u_D,
                                     exact_solution=u_ex)  # note, no copy is done, the attributes of heq are EXACTLY these guys
                    heq.set_ODE_scheme("crank_nicolson")
                    heq.verbose = True

                    logging.info(f"Simulating exact solution for test problem")
                    solution_list, _ = heq.solve()

                    # To pickle
                    self.to_pickle(solution_list, pickle_name)

                u_ex = self.create_solution(0, heq.times, solution_list)

            case "no_source_dirichlet_neumann_zero_compatibility_neumann":   # RC 0, not 1, because of Neumann BC

                class dirichlet_data(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = np.sin(x[1]) * (self.t**3)

                    def value_shape(self):
                        return ()

                class initial_solution(UserExpression):
                    def __init__(self, **kwargs):
                        super().__init__(self, **kwargs)

                    def eval(self, values, x):
                        values[0] = 0

                    def value_shape(self):
                        return ()

                class neumann_trace(UserExpression):
                    def __init__(self, t, **kwargs):
                        super().__init__(self, **kwargs)
                        self.t = t

                    def eval(self, values, x):
                        values[0] = x[1] * (1 - self.t) # note, incompatible at order 1

                    def value_shape(self):
                        return ()

                u0 = initial_solution()
                u_D = dirichlet_data(t=0)
                u_N = neumann_trace(t=0)
                f = None
                u_ex = None
                self.marker_dirichlet = [3]
                self.marker_neumann = [2]

                # Now, let's simulate the exact solution on a fine mesh and on a fine time scale
                heq = PDE()
                heq.read_mesh(resolutions[5], mesh_path)

                N_steps = int(np.ceil(2 / (heq.mesh.hmax() ** 2)))  # many many time steps!
                heq.set_time_discretization(self.T, N_steps=N_steps)  # dt = h^2 and N = T/dt

                pickle_name = "no_source_dirichlet_neumann_no_compatibility_difficult"
                try:
                    if self.override_pickle:  # I lazily generate an error to go into the except
                        1 / 0
                    solution_list = self.from_pickle(heq.S1h, pickle_name)
                except:
                    heq.set_PDE_data(u0, source=f, marker_neumann=self.marker_neumann,
                                     marker_dirichlet=self.marker_dirichlet,
                                     neumann_BC=u_N,
                                     dirichlet_BC=u_D,
                                     exact_solution=u_ex)  # note, no copy is done, the attributes of heq are EXACTLY these guys
                    heq.set_ODE_scheme("crank_nicolson")
                    heq.verbose = True

                    logging.info(f"Simulating exact solution for test problem")
                    solution_list, _ = heq.solve()

                    # To pickle
                    self.to_pickle(solution_list, pickle_name)

                u_ex = self.create_solution(0, heq.times, solution_list)

            case _:
                raise Exception("No problem found with this name")

        # If u_ex is simulated, the data timesteps will be modified
        if u_ex is not None:
            u_ex.t = 0
        if f is not None:
            f.t = 0
        if u_D is not None:
            u_D.t = 0
        if u_N is not None:
            u_N.t = 0
        return (u_ex, f, u_D, u_N, u0, self.marker_neumann, self.marker_dirichlet, self.T)

    def to_pickle(self, solution_list, pickle_name):

        coefficients_list = []

        for u in solution_list:
            coefficients_list.append(u.vector()[:])

        # Save to file
        with open(pickles_path + pickle_name + '.pickle', 'wb') as handle:
            pickle.dump(coefficients_list, handle)

        logging.info("Exact solution successfully pickled")

    def from_pickle(self, V, pickle_name):

        with open(pickles_path + pickle_name + '.pickle', 'rb') as handle:
            pk = pickle.load(handle)

        solution_list = []

        for nodal_values in pk:
            u = Function(V)
            u.vector()[:] = nodal_values
            solution_list.append(u)

        logging.info("Successfully loaded exact solution from pickle")

        return solution_list

    def create_solution(self, t, times, solution_list):

        # Now, create an expression from solution list, hopefully it works
        class solution(UserExpression):
            def __init__(self, t, discrete_times, solution_list, **kwargs):
                super().__init__(self, **kwargs)
                self.t = t

                if not isinstance(discrete_times, np.ndarray):
                    raise Exception("Times must be a numpy array")
                self.discrete_times = np.sort(discrete_times)
                self.t_min = np.min(self.discrete_times)
                self.t_max = np.max(self.discrete_times)

                self.solution_list = solution_list

            def eval(self, values, x):

                if self.t <= self.t_min:
                    values[0] = solution_list[0](*x)
                elif self.t >= self.t_max:
                    values[0] = solution_list[-1](*x)
                else:

                    i_right = np.searchsorted(self.discrete_times, self.t, side='right')
                    i_left = i_right - 1

                    t_left = self.discrete_times[i_left]
                    t_right = self.discrete_times[i_right]

                    v_left = solution_list[i_left](*x)
                    v_right = solution_list[i_right](*x)

                    dt = t_right - t_left

                    # Linear interpolation
                    w = (self.t - t_left) / dt
                    values[0] = v_left + w * (v_right - v_left)

            def value_shape(self):
                return ()

        return solution(t, times, solution_list)


# %% Error check

problem_name = "no_source_dirichlet_neumann_zero_compatibility"  # what problem from TestProblems are we going to solve?

# Test problem
problems = TestProblems()
problems.override_pickle = True
u_ex, f, u_D, u_N, u0, marker_neumann, marker_dirichlet, T = problems.get_data(problem_name)

# Setting up the PDE class
heq = PDE()
heq.set_PDE_data(u0, source=f, marker_neumann=marker_neumann, marker_dirichlet=marker_dirichlet, neumann_BC=u_N,
                 dirichlet_BC=u_D,
                 exact_solution=u_ex)  # note, no copy is done, the attributes of heq are EXACTLY these guys
heq.set_ODE_scheme("implicit_euler")
heq.verbose = True

# We check that the last error decreases as expected
errors_in_one_time, hs, dts = [], [], []  # errors are only at a certain index
error_index = 2  # this is the index: it means at T/2

# Solve the PDE many times...
for i in range(4):

    heq.read_mesh(resolutions[i], mesh_path)
    hs.append(heq.mesh.hmax())

    match heq.implemented_time_schemes.index(heq.ode_scheme):
        case 0 | 3:  # implicit and implicit-explicit euler
            N_steps = int(np.ceil(2 / (heq.mesh.hmax() ** 2)))
        case 1:  # crank-nicolson
            N_steps = int(np.ceil(2 / (heq.mesh.hmax() ** 1)))

    logging.info(f"Error level {i} with {N_steps} timesteps")

    heq.set_time_discretization(T, N_steps=N_steps)  # dt = h^2 and N = T/dt
    dts.append(heq.dts[0])

    solution_list, error_list = heq.solve(err_mode="l2")
    current_error = error_list[len(error_list) // 2]
    errors_in_one_time.append(current_error)
    logging.info(f"Error: {current_error}")

errors_in_one_time = np.array(errors_in_one_time)
hs = np.array(hs)
dts = np.array(dts)

# The order of convergence
ooc_space = np.log(errors_in_one_time[1:] / errors_in_one_time[:-1]) / np.log(hs[1:] / hs[:-1])
ooc_time = np.log(errors_in_one_time[1:] / errors_in_one_time[:-1]) / np.log(dts[1:] / dts[:-1])
logging.info(f"OOC wrt space: {ooc_space}")
logging.info(f"OOC wrt time: {ooc_time}")

# %% Notes and findings

# All the errors check the half time

# Problem: discontinuous/continuous
# 1) A Neumann condition requires cannot recover from a wrong initial condition, understandably
# 2) Pure Dirichlet conditions manage to decrease final errors even with wrong initial condition

# Efficiency
# Here are the timings of assembling A, b, and solving A\b (pure Dirichlet)
# 0.5430436134338379
# 0.4644930362701416
# 1.0107779502868652
# For pure Neumann -> maybe one could pre-assemble the mass and stiffness matrices prior to every time stepping, and the rhs for neumann equations.. or just the latter
# 0.5088815689086914
# 3.3294856548309326
# 1.201446533203125

# We wouldn't get a significant speed-up if assembling A only once (the speed-up would be around 25 %, consider it)
# At least in the forward pass!! Maybe this is an issue in the backward pass!
# In that case, to be able to pursue this, you must use assemble_system, because it is supported by dolfin-adjoint, and
# the variational formulation of the Dirichlet problem must implement the solution split, to be able not to impose
# time varying BCs
# This will probably NOT work with Crank Nicholson, if the rhs is approximated with the midpoint rule
# And it will also increase the cost in the linear form assembly: if anything, pre-assemble the neumann BCs (but then
# you need to also assemble A and so you have to apply the BC, which is NOT possible in dolfin!

# Compatibility (implicit Euler)
# 1) LUCK: with pure Dirichlet conditions and only 0 comp. we observe the correct OOC with implicit Euler (1 in time)
# 2) WEIRD: add some Neumann condition and the convergence is halved, in implicit Euler (.5 in time) (can be checked by putting N_steps = int(np.ceil(2 / (heq.mesh.hmax() ** 4))))
# 3) but with one more compatibility relation we recover order 2 even with some Neumann BC
# Note, the errors are large in absolute value with pure Neumann and missing compatibilities. They become better if compatibility of all orders is satisfied (check problem "continuous")
# Note, it is very slow!

# Compatibility (Crank-Nicolson)
# 1) LUCK: with 0 but not 1st compatibility, and pure Neumann BC, I also get OOC of 1 in time with CN
# 2) WEIRD: with 0th, 1st compatibility, and pure Neumann BC, CN attains OOC of 1.5 in time
# 3) but in both cases (i.e. 0, or 0+1 compatibility, it achieves better errors than IE) ?

# Compatibility (implicit-explicit Euler)
# 1) I could only check the second order compatibility test case, because the first, has a singularity which I can't evaluate
#    The results are the same, Neumann BC only case: OOC 1 in time and very large error
# 2) for the Dirichlet BC only case, as in the other tests, the OOC grows from .5 to 1 in time, and is much smaller

# l\infty error
# 1) it is very! large if only neumann BC are employed, less large is one puts a Dirichlet BC to start with, and rather
#    small in the case of two Dirichlet BC. As a reference, without first compatibility:
# Dirichlet only (linf)
# Step  0
# Error:  0.12442452479304311
# Step  1
# Error:  0.023295529396900605
# Step  2
# Error:  0.006574207012854494
# Step  3
# Error:  0.0016244579753790234
# OOC wrt space:  [2.47995436 2.08544838 2.2561029 ]
# OOC wrt time:  [1.20857293 1.07334844 1.12584761]
# Neumann-Dirichlet (linf)
# Step  0
# Error:  0.6800491004505749
# Step  1
# Error:  0.2839366900476814
# Step  2
# Error:  0.1364176776022359
# Step  3
# Error:  0.06631416776945498
# OOC wrt space:  [1.292812   1.20835275 1.1640849 ]
# OOC wrt time:  [0.63003481 0.62192071 0.58090533]
# And here the results with the "continuous" problem
# Step  0
# Error:  0.06013085396579987
# Step  1
# Error:  0.02355834089524933
# Step  2
# Error:  0.009826203883144924
# Step  3
# Error:  0.002241456609800796
# OOC wrt space:  [1.38699521 1.4414371  2.38512384]
# OOC wrt time:  [0.67593375 0.74188567 1.19023204]

########################################################################################################################
########################################################################################################################
########################################################################################################################

# Non manufactured solutions
# Note, we don't seem to notice the fact that we have a non-convex domain

# Test case: dirichlet_neumann_no_source_smooth (RC 0,1,2 not 3)
# Crank-Nicolson yieds:
# 15:53 INFO     OOC wrt space: [2.81700117 2.42270643 1.78903402]
# 15:53 INFO     OOC wrt time: [2.74565646 2.12032932 1.9809368 ]
# Implicit Euler yields:
# 16:02 INFO     OOC wrt space: [1.48227177 1.8484389  2.02326267]
# 16:02 INFO     OOC wrt time: [0.72236553 0.95136328 1.00965494]
# Implicit-explicit Euler yields:
# 16:10 INFO     OOC wrt space: [1.8375965  1.86598621 1.96569092]
# 16:10 INFO     OOC wrt time: [0.89552833 0.96039462 0.9809253 ]

# Test case: dirichlet_neumann_no_source_no_compatibility (first tests were done on resolution 5, t = ^2)
# Crank-Nicolson yields
# 16:54 INFO     OOC wrt space: [1.75220456 1.23866264 0.94963896]
# 16:54 INFO     OOC wrt time: [1.70782739 1.08406561 1.05150307]
# which calls for another simulation with h = sqrt(t)
# 17:03 INFO     OOC wrt space: [3.0005001  2.05201655 2.08074759] <- the spatial convergence is okay now
# 17:03 INFO     OOC wrt time: [1.462254   1.05614159 1.03834124]
# Implicit-explicit Euler yields
# 16:35 INFO     OOC wrt space: [1.96068424 1.89119407 1.97066711]
# 16:35 INFO     OOC wrt time: [0.95551351 0.97336871 0.98340853]
# At // 8 time we get
# 16:51 INFO     OOC wrt space: [-50.57158325   1.91592578   2.39523286]
# 16:51 INFO     OOC wrt time: [-24.64539161   0.98609775   1.19527668]
# Back to T/2
# Now resolution = 6 and t = h
# If we make the BC and initial conditions a bit crazy, for implicit Euler and the explicit version:
# 22:23 INFO     OOC wrt space: [1.94230849 1.57970113 1.97950822] # this oscillation was already present in the stationary version
# 22:23 INFO     OOC wrt time: [0.94655833 0.81304805 0.98782045]
# 22:02 INFO     OOC wrt space: [1.92104608 1.54172669 1.86860215]
# 22:02 INFO     OOC wrt time: [0.93619638 0.79350319 0.93247575]
# Changing the condition to 1-x^-y^2 instead of x^2+y^2 it gets to (implicit Euler):
# 11:46 INFO     OOC wrt space: [1.6000088  1.82109667 1.85190656]
# 11:46 INFO     OOC wrt time: [0.77974311 0.93729065 0.92414427]


# Test case: ...zero_compatibility: note, here a more complicated initial condition is present ((first tests were done on resolution 5, t = ^2)
# Implicit euler
# 17:31 INFO     OOC wrt space: [1.8336602  1.77978011 1.83181312]
# 17:31 INFO     OOC wrt time: [0.89361003 0.91602565 0.91411717]
# Note that we don't get OOC 1 in time, but this may be due to the testing method
# It is not true that OOC in space is 1.8 as testified by the run
# 17:44 INFO     OOC wrt space: [1.75626961 1.45982934 1.47160392]
# 17:44 INFO     OOC wrt time: [0.94712415 0.80609388 0.842287  ]
# Maybe the "exact" solution is not fine enough
# Resolution 6, t = 1/4 * h
# Implicit Euler
# 00:18 INFO     OOC wrt space: [1.83511875 1.78707689 1.8564375 ]
# 00:18 INFO     OOC wrt time: [0.89432083 0.91978119 0.92640531]   <- WTF??
# Now resolution = 6 and t = h
# Implicit-explicit Euler
# 22:41 INFO     OOC wrt space: [1.79061738 1.85178169 1.93181537]
# 22:41 INFO     OOC wrt time: [0.87263367 0.95308376 0.96402062]
# Back to WTF: is the problem the spatial discretization? Is OOC(x) = 1.82 or 2*OOC(t) = 1.82?
# We ruled out the first
# We have to test the second... how?
# After correcting self.t I get
# Implicit Euler
# 11:15 INFO     OOC wrt space: [1.7650171  1.61782814 1.68422548]
# 11:15 INFO     OOC wrt time: [0.86015772 0.83267144 0.84046752]
# Crank-Nicolson (conclusion: Crank-Nicolson doesn't have order 2 with missing CC)
# 10:51 INFO     OOC wrt space: [0.7775563  1.8630501  1.32423689]
# 10:51 INFO     OOC wrt time: [0.75786354 1.63052349 1.46628268]

# Test case first_compatibility (but not second order):
# Crank-Nicolson
# 12:29 INFO     OOC wrt space: [2.09223375 2.05557765 2.79857228]
# 12:29 INFO     OOC wrt time: [2.03924484 1.79902175 3.09876433]
# Implicit Euler
# 12:37 INFO     OOC wrt space: [1.80697055 1.87881522 2.03183475]
# 12:37 INFO     OOC wrt time: [0.88060318 0.96699751 1.0139326 ]
# Implicit-explicit
# 12:46 INFO     OOC wrt space: [1.65395919 1.83481033 1.95941007]
# 12:46 INFO     OOC wrt time: [0.80603511 0.94434886 0.97779101]

# Test case zero_compatibility_neumann
# Implicit Euler
# 11:17 INFO     OOC wrt space: [1.398448   1.79279154 1.96634886]
# 11:17 INFO     OOC wrt time: [0.68151512 0.92272244 0.98125363]
# Implicit-explicit Euler
# 11:25 INFO     OOC wrt space: [1.4580682  1.71368181 1.92754846]
# 11:25 INFO     OOC wrt time: [0.71057023 0.88200586 0.96189133]
# Crank-Nicolson
# 11:32 INFO     OOC wrt space: [2.12013675 1.9369714  1.46462242]
# 11:32 INFO     OOC wrt time: [2.06644115 1.6952187  1.62172681]


# todo: relationship with space-time FEM... what I the rhs is a time integral? Not easily implementable but...
