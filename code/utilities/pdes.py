from dolfin import *
from dolfin_adjoint import *
import numpy as np
import logging
from tqdm import tqdm
import pickle
from utilities.meshing import AnnulusMesh


# %% Class definitions

class HeatEquation:

    def __init__(self, efficient=False):
        self.domain_measure = None
        self.S1h = None
        self.exact_solution = None
        self.neumann_marker = None
        self.dirichlet_marker = None
        self.f_N = None
        self.f_D = None
        self.f = None
        self.initial_value = None
        self.times = None
        self.dts = None
        self.T = None
        self.ode_scheme = None
        self.facet_indicator = None
        self.mesh = None
        self.domain = None
        self.solution_list = []
        self.implemented_time_schemes = ["implicit_euler", "crank_nicolson", "crank_nicolson_midpoint",
                                         "implicit_explicit_euler"]
        self.interpolate_data = False   # if true, we use interpolated data and not the actual data with quadrature
        self.verbose = False

        self.efficient = efficient
        self.A = None  # pre_assembled system matrix, just in case
        self.pre_assembled_BCs = None  # a vector of functions, not a time dependent expression
        if self.efficient:
            logging.warning(
                "This code is suitable for shape optimization only if boundary conditions are constant/they are on fixed boundaries, and source terms are constant/defined with SpatialCoordinate")

    def set_mesh(self, mesh, mf, V=None, order=1):

        self.mesh = mesh
        self.mesh.rename("PDEMesh", "")
        self.facet_indicator = mf

        self.set_function_space(V=V, order=order)
        self.compute_domain_volume()

    def set_ODE_scheme(self, ode_scheme):
        if ode_scheme not in self.implemented_time_schemes:
            raise Exception("Unrecognized time stepping")
        self.ode_scheme = ode_scheme
        if ode_scheme=="implicit_euler":
            logging.warning("A strange error is raised if I don't include a minuscule term to the right hand side of the instant variational formulation")

    def set_time_discretization(self, final_time, N_steps=None, custom_times_array=None, relevant_mesh_size=None):
        '''
        Note, one of N_steps and custom_time_array must be provided
        :param final_time: final time of simulation
        :param N_steps: optional, if present, we will use uniform timestepping with dt = final_time/N_steps
        :param custom_times_array: we can also set a custom time discretization of the form np.array([0, ..., final_time])
        :param relevant_mesh_size: it is a spatial mesh size that is not necessarily the one of the current domain,
            which we use to automatically compute the time mesh
        :return:
        '''

        if final_time <= 0:
            raise Exception("Final time must be positive")
        self.T = final_time

        if N_steps is None and custom_times_array is None:
            logging.warning("No time discretization provided, the default will be created")
            if self.ode_scheme is None:
                raise Exception("No ODE scheme was specified, no default time stepping is available")
            if self.mesh is None:
                raise Exception("No mesh was specified, no default time stepping is available")
            ode_scheme_index = self.implemented_time_schemes.index(self.ode_scheme)
            if relevant_mesh_size is not None:
                h = relevant_mesh_size
            else:
                h = self.mesh.hmax()
            if ode_scheme_index == 0 or ode_scheme_index == 3:
                N_steps = int(np.ceil(self.T / (h ** 2)))
            elif ode_scheme_index == 1:
                N_steps = int(np.ceil(self.T / h))
            self.times = np.linspace(0, self.T, N_steps + 1)
            dt = self.T / N_steps
            if dt < DOLFIN_EPS:
                raise Exception("Negative or too small time step")
            self.dts = dt * np.ones(N_steps)  # we have N_steps + 1 points, so, N_steps intervals
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
                     exact_solution=None,
                     pre_assembled_BCs=None):
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
            if not isinstance(dirichlet_BC, list):
                raise Exception("Boundary conditions must be supplied in lists")
            if len(dirichlet_BC) != len(marker_dirichlet):
                raise Exception("The number of boundary conditions must match the number of markers")
        if marker_neumann is not None:
            if not isinstance(marker_neumann, list):
                raise Exception("The Neumann marker should be a list")
            for v in marker_neumann:
                if not isinstance(v, int):
                    raise Exception("The Neumann marker should be an integer")
            if not isinstance(neumann_BC, list):
                raise Exception("Boundary conditions must be supplied in lists")
            if len(neumann_BC) != len(marker_neumann):
                raise Exception("The number of boundary conditions must match the number of markers")

        if pre_assembled_BCs is not None:
            self.pre_assembled_BCs = pre_assembled_BCs

        self.initial_value = initial_value

        # Some values may be None
        self.f = source
        self.f_D = dirichlet_BC
        self.f_N = neumann_BC
        self.dirichlet_marker = marker_dirichlet
        self.neumann_marker = marker_neumann
        self.exact_solution = exact_solution

    def set_function_space(self, V=None, order=1):
        if not hasattr(self, "mesh"):
            raise Exception("A mesh needs to be defined at first")
        if V is None:
            logging.warning("No finite element space was provided, using default one")
            L1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), order)  # only linear FEM for now
            self.S1h = FunctionSpace(self.mesh, L1)
        else:
            self.S1h = V

    def compute_domain_volume(self):

        if not hasattr(self, "S1h"):
            raise Exception("The function space is not defined")
        if not hasattr(self, "mesh"):
            raise Exception("No mesh was defined")

        self.domain_measure = assemble(Constant(1.0) * dx(self.mesh))

    def include_neumann_conditions(self, t, dt, L, v, time_index):
        if (self.neumann_marker is not None): # note, let's add the neumann term even if we want to be efficient, as we need to attach a form to a vector, for dolfin adjoint
            for (f_N, marker) in zip(self.f_N, self.neumann_marker):

                if self.efficient and marker in self.pre_assembled_BCs.keys():
                    if self.pre_assembled_BCs[marker]["type"] == "neumann":
                        # Check the index is correct
                        assert(np.allclose(self.pre_assembled_BCs[marker]["data"].times[time_index-1],t))
                        # If so
                        f_N_real = Function(self.S1h)
                        f_N_real.vector()[:] = self.pre_assembled_BCs[marker]["data"].interpolated_BC[time_index-1].vector()[:]

                        if self.interpolate_data:
                            L += dt * interpolate(f_N_real, self.S1h) * v * ds(self.mesh)
                        else:
                            L += dt * f_N_real * v * ds(self.mesh)
                    else:
                        raise Exception("Wrong type of boundary condition")

                else:
                    if hasattr(f_N, 't'):
                        f_N.t = t

                    if self.interpolate_data:
                        L += dt * interpolate(f_N, self.S1h) * v * ds(self.mesh)
                    else:
                        L += dt * f_N * v * ds(self.mesh)

        return L

    def include_dirichlet_conditions(self, t):

        if self.dirichlet_marker is not None:
            for f_D in self.f_D:
                if hasattr(f_D, 't'):
                    f_D.t = t

    def include_source(self, t, L, dt, v, dx):

        if self.f is not None:
            if hasattr(self.f, 't'):
                self.f.t = t

            if self.interpolate_data:
                L += dt * interpolate(self.f, self.S1h) * v * dx
            else:
                L += dt * self.f * v * dx

        return L

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

        dt = Constant(self.dts[time_index - 1])
        dt.rename("dt", "")

        t_neumann = -np.inf
        t_dirichlet = -np.inf
        t_source = -np.inf

        # Choose the correct time scheme
        L = last_solution * v * dx
        a = u * v * dx
        if ode_scheme_index == 0:  # implicit euler
            t = self.times[time_index]
            t_neumann = t
            t_dirichlet = t
            t_source = t

            a += dt * inner(grad(u), grad(v)) * dx
            L += 1e-200 * inner(grad(last_solution), grad(v)) * dx

        elif ode_scheme_index == 1:  # CN (standard)
            t = self.times[time_index]
            t_prev = self.times[time_index - 1]
            t_neumann = (t + t_prev) / 2
            t_dirichlet = t
            t_source = (t + t_prev) / 2

            a += dt / 2 * inner(grad(u), grad(v)) * dx
            L += - dt / 2 * inner(grad(last_solution), grad(v)) * dx

        elif ode_scheme_index == 3:  # implicit-explicit euler
            t = self.times[time_index]
            t_last = self.times[time_index - 1]
            t_neumann = t_last
            t_dirichlet = t_last # this could be changed, depending the kind of consistency want
            t_source = t_last

            a += dt * inner(grad(u), grad(v)) * dx

        L_no_neumann = self.include_source(t_source, L, dt, v, dx)
        L = self.include_neumann_conditions(t_neumann, dt, L_no_neumann, v, time_index)
        self.include_dirichlet_conditions(t_dirichlet)

        return a, L, L_no_neumann

    def solve_single_step(self, time_index, last_solution=None, err_mode="none"):
        if not isinstance(time_index, int):
            raise Exception("Time index must be an integer")
        if time_index >= len(self.times) or time_index < 0:
            raise Exception("Time index out of range")

        if time_index == 0:
            # current_solution = project(self.initial_value, self.S1h)
            current_solution = self.initial_value  # let us assume it doesn't need projection
            if err_mode != "none":
                current_error = self.compute_instant_error(err_mode, current_solution, time_index)
                return current_solution, current_error
            else:
                return current_solution, None
        elif last_solution is None:
            raise Exception("The previous time solution was not provided")

        a, L, L_no_neumann = self.get_instant_variational_formulation(time_index, last_solution)
        current_solution = Function(self.S1h)

        if not self.efficient:

            # Dirichlet BCs
            dirichlet_BC = []
            if self.dirichlet_marker is not None:
                # self.f_D.t = self.times[time_index] # this is already taken care of in get_instant_variational_formulation
                for marker, f_D in zip(self.dirichlet_marker, self.f_D):
                    dirichlet_BC.append(DirichletBC(self.S1h, f_D, self.facet_indicator, marker))

            solve(a == L, current_solution,
                  dirichlet_BC)  # dirichlet BC might be empty. Also, they're imposed at possibily a different time than the Neumann conditions, in the case of CN
        else:
            C = self.A.copy()
            C.form = a

            b = assemble(L)

            # Dirichlet BCs
            dirichlet_BC = []
            if self.dirichlet_marker is not None:
                # self.f_D.t = self.times[time_index] # this is already taken care of in get_instant_variational_formulation
                for marker, f_D in zip(self.dirichlet_marker, self.f_D):

                    if self.efficient and marker in self.pre_assembled_BCs.keys():
                        if self.pre_assembled_BCs[marker]["type"] == "dirichlet":
                            # Check the index is correct
                            assert (np.allclose(self.pre_assembled_BCs[marker]["data"].times[time_index - 1], f_D.t))
                            dirichlet_BC.append(DirichletBC(self.S1h, self.pre_assembled_BCs[marker]["data"].interpolated_BC[time_index -1], self.facet_indicator, marker))

                        else:
                            raise Exception("Wrong boundary condition")
                    else:
                        dirichlet_BC.append(DirichletBC(self.S1h, f_D, self.facet_indicator, marker))

                    dirichlet_BC[-1].apply(C, b)


            solve(C, current_solution.vector(), b)

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
            if hasattr(self.exact_solution, 't'):
                self.exact_solution.t = self.times[time_index]
        if err_mode == "l2":
            error = errornorm(self.exact_solution, current_solution)
            return error
        elif err_mode == "h1":
            error = errornorm(self.exact_solution, current_solution, norm_type="h1")
            return error
        elif err_mode == "l2_and_h1":
            error_l2 = errornorm(self.exact_solution, current_solution, norm_type="l2")
            error_h1 = errornorm(self.exact_solution, current_solution, norm_type="h1")
            return [error_l2, error_h1]
        elif err_mode == "linf":
            u_ex_interp = project(self.exact_solution, self.S1h)
            error = np.max(np.abs(u_ex_interp.vector()[:] - current_solution.vector()[:]))
            return error
        else:
            raise Exception("Error norm not implemented")

    def solve(self, err_mode="none"):
        solution_list = []
        error_list = []
        last_solution = Function(self.S1h)  # container for the last solution

        first_solution, first_error = self.solve_single_step(0, err_mode=err_mode)
        solution_list.append(first_solution)
        error_list.append(first_error)
        last_solution.assign(first_solution)  # no need to create a copy of first solution

        logging.info("Solving the heat equation")
        if self.efficient:
            self.pre_assemblage()
        for time_step in tqdm(range(1, len(self.times))):
            current_solution, current_error = self.solve_single_step(time_step, last_solution, err_mode=err_mode)
            solution_list.append(current_solution)
            error_list.append(current_error)
            last_solution.assign(current_solution)  # no need for current_solution.copy()

        self.solution_list = solution_list

        if err_mode == "none":
            return solution_list, None
        else:
            return solution_list, error_list

    def pre_assemblage(self):
        a, _, _= self.get_instant_variational_formulation(1, Function(self.S1h))
        self.A = assemble(a)


class PDETestProblems:

    def __init__(self):
        self.marker_dirichlet = [3]
        self.marker_neumann = [2]
        self.T = 2
        self.override_pickle = True

        self.pickles_path = "/home/leonardo_mutti/PycharmProjects/masters_thesis/pde_data/pickles/test_problems/"
        self.return_exact_sol = True

    def get_data(self, problem_name):

        mesh_path = "/home/leonardo_mutti/PycharmProjects/masters_thesis/meshes/annulus/"
        resolutions = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625]  # mesh resolutions

        # We are given everything correct, the only problem is that this HEQ has 0th but not 1st compatibility
        if problem_name == "no_first_compatibility":
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
        elif problem_name == "no_second_compatibility":
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
        elif problem_name == "continuous":
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

        elif problem_name == "constant":
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

        elif problem_name == "linear_in_time":
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

        elif problem_name == "quadratic_in_time":
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

        elif problem_name == "no_source_dirichlet_neumann_smooth":  # actually this satisfies RC 0, 1, 2 but not 3

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
            heq = HeatEquation()
            annulus = AnnulusMesh(resolution=resolutions[5])
            heq.set_mesh(annulus.mesh, annulus.facet_function)

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

        elif problem_name == "no_source_dirichlet_neumann_no_compatibility":  # no compatibility at all

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
            heq = HeatEquation()
            annulus = AnnulusMesh(resolution=resolutions[5])
            heq.set_mesh(annulus.mesh, annulus.facet_function)

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

            u_ex = self.create_solution(0, heq.times, solution_list)

        elif problem_name == "no_source_dirichlet_neumann_zero_compatibility":  # 0 but not 1 compatibility

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
            heq = HeatEquation()
            annulus = AnnulusMesh(resolution=resolutions[5])
            heq.set_mesh(annulus.mesh, annulus.facet_function)

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

        elif problem_name == "no_source_dirichlet_neumann_no_compatibility_difficult":  # same as non difficult, but with spatially varying BCs

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
            heq = HeatEquation()
            annulus = AnnulusMesh(resolution=resolutions[6])
            heq.set_mesh(annulus.mesh, annulus.facet_function)

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

        elif problem_name == "no_source_dirichlet_neumann_first_compatibility_difficult":  # RC 0, 1 and not 2

            class dirichlet_data(UserExpression):
                def __init__(self, t, **kwargs):
                    super().__init__(self, **kwargs)
                    self.t = t

                def eval(self, values, x):
                    values[0] = np.sin(x[1]) * (self.t ** 2)

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
            heq = HeatEquation()
            annulus = AnnulusMesh(resolution=resolutions[5])
            heq.set_mesh(annulus.mesh, annulus.facet_function)

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

        elif problem_name == "no_source_dirichlet_neumann_zero_compatibility_neumann":  # RC 0, not 1, because of Neumann BC

            class dirichlet_data(UserExpression):
                def __init__(self, t, **kwargs):
                    super().__init__(self, **kwargs)
                    self.t = t

                def eval(self, values, x):
                    values[0] = np.sin(x[1]) * (self.t ** 3)

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
                    values[0] = x[1] * (1 - self.t)  # note, incompatible at order 1

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
            heq = HeatEquation()
            annulus = AnnulusMesh(resolution=resolutions[5])
            heq.set_mesh(annulus.mesh, annulus.facet_function)

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

        else:
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

        pickles_path = self.pickles_path

        coefficients_list = []

        for u in solution_list:
            coefficients_list.append(u.vector()[:])

        # Save to file
        with open(pickles_path + pickle_name + '.pickle', 'wb') as handle:
            pickle.dump(coefficients_list, handle)

        logging.info("Exact solution successfully pickled")

    def from_pickle(self, V, pickle_name):

        if self.return_exact_sol:

            pickles_path = self.pickles_path

            with open(pickles_path + pickle_name + '.pickle', 'rb') as handle:
                pk = pickle.load(handle)

            solution_list = []

            for nodal_values in pk:
                u = Function(V)
                u.vector()[:] = nodal_values
                solution_list.append(u)

            logging.info("Successfully loaded exact solution from pickle")

            return solution_list

        else:
            return None

    def create_solution(self, t, times, solution_list):
        return TimeExpressionFromList(t, times, solution_list)


# Now, create an expression from solution list, hopefully it works
class TimeExpressionFromList(UserExpression):
    def __init__(self, t, discrete_times, solution_list, reverse=False, **kwargs):
        super().__init__(self, **kwargs)
        self.t = t

        if not isinstance(discrete_times, np.ndarray):
            raise Exception("Times must be a numpy array")
        self.discrete_times = np.sort(discrete_times)
        self.t_min = np.min(self.discrete_times)
        self.t_max = np.max(self.discrete_times)

        self.reverse = reverse

        self.solution_list = solution_list
        self.solution_list_original = []
        for u in solution_list:
            v = Function(u.function_space())
            v.assign(u)
            self.solution_list_original.append(v)

        self.V = solution_list[-1].function_space()

    def eval(self, values, x):

        t = self.t

        if self.reverse:
            t = self.t_max - t

        if t <= self.t_min:
            values[0] = self.solution_list[0](*x)
        elif t >= self.t_max:
            values[0] = self.solution_list[-1](*x)
        else:

            i_right = np.searchsorted(self.discrete_times, t, side='right')
            i_left = i_right - 1

            t_left = self.discrete_times[i_left]
            t_right = self.discrete_times[i_right]

            v_left = self.solution_list[i_left](*x)
            v_right = self.solution_list[i_right](*x)

            dt = t_right - t_left

            # Linear interpolation
            w = (t - t_left) / dt
            values[0] = v_left + w * (v_right - v_left)

    def value_shape(self):
        return ()

    def perturb(self, noise_level):
        logging.fatal("No perturbation implemented")
        sh = self.solution_list[0].vector()[:].shape
        for (u, t) in zip(self.solution_list, self.discrete_times):
            u.vector()[:] += (np.random.rand(*sh) - .5) * noise_level
            # u.vector()[:] += .003 * t

    def unperturb(self):
        for (up, uu) in zip(self.solution_list, self.solution_list_original):
            up.vector()[:] = uu.vector().copy()


class PreAssembledBC():
    """
    Given an expression depending on time, we return a list of functions, the interpolations at a time vector of the expression
    Noise can be applied
    """

    def __init__(self, expression, time_instants, V, noise_level=0):
        if not hasattr(expression, 't'):
            raise Exception("This expression doesn't depend on time")
        self.expression = expression  # must have t as attribute
        self.times = time_instants
        self.noise_level = noise_level

        self.V = V  # the function space in which to interpolate

        self.interpolated_BC = []
        self.pre_assemble()  # actually, interpolate

    def pre_assemble(self):

        if isinstance(self.expression, TimeExpressionFromList):

            # Collect nodal vectors in an array
            fine_nodal_values = np.array([u.vector()[:] for u in self.expression.solution_list_original])
            from scipy.interpolate.interpolate import interp1d
            interp = interp1d(self.expression.discrete_times, fine_nodal_values, axis=0)

            new_vals = interp(self.times)

            for i in tqdm(range(new_vals.shape[0])):
                u = Function(self.expression.V)
                u.vector()[:] = new_vals[i]
                bc = Function(self.V)
                bc.assign(interpolate(u, self.V))
                bc.vector()[:] += self.noise_level * (np.random.rand(len(bc.vector())) * .5 - 1)
                self.interpolated_BC.append(bc)
        else:
            for t in tqdm(self.times):
                self.expression.t = t
                bc = Function(self.V)
                bc.assign(interpolate(self.expression, self.V))
                bc.vector()[:] += self.noise_level * (np.random.rand(len(bc.vector())) * .5 - 1)
                self.interpolated_BC.append(bc)
