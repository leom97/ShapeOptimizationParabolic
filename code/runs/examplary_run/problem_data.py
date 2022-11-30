"""
Configuration script to run shape optimization.
"""

import numpy as np

# %% Synthesis

# Dictionary to set up the domain to be reconstructed, the "exact" domain, from the knowledge of a function q_ex_lambda
# living on the unit sphere, "sphere", which radially generated "domain" (see section 2.3 and section 4.1)
# If W is the radial vector field corresponding to q_ex_lambda, then the optimal domain will be 2B \ (Id + W)(B), where
# B is the unit disk
# We work with an annulus for simplicity, check out other possibilities at code/utilities/meshing
exact_geometry_dict = {
    "domain": {"type": "annulus", "resolution": 0.05, "ext_refinement": 1.0, "int_refinement": 1.0, "inner_radius": 1,
               "outer_radius": 2,
               "center": np.array([0, 0]), "reload_xdmf": False},
    # set reload_xdmf to True if the mesh was already generated once and saved
    "sphere": {"dimension": 2, "resolution": 0.03},  # dimension refers to the embedding space, R^2. Don't change this
    "q_ex_lambda": 'lambda circle_coords: -.8 * circle_coords[:, 0] ** 2 +.2 * circle_coords[:, 1] ** 2'
    # circle_coords are in dof order
}

# This dictionary holds information for the simulation of the "exact" PDE for w (see section 4.1), on the "exact" domain
exact_pde_dict = {
    "ode_scheme": "crank_nicolson",  # crank_nicolson or implicit_euler
    "u_N": "Expression('pow(t,2)', t=0, a=.05, p = np.pi, degree=5)",  # Nuemann data for g (see problem 2.1.1)
    "marker_dirichlet": [3],  # internal moving boundary
    "marker_neumann": [2],  # external boundary
    "T": 2,  # time of simulation: the PDEs will be simulated on [0,T]
    "N_steps": 120  # steps number for time stepping algorithm
}

# For optimization, we can use a coarsed domain, and a coarsed sphere for the controls
simulated_geometry_dict = {
    "additional_domain_data": {"resolution": 0.1},
    "sphere_resolution": 0.15,
}

# %% Inversion/optimization

# For optimization, we can use a different time stepping schemes than in the synthesis phase
simulated_pde_dict = {
    "ode_scheme": "crank_nicolson",
    "N_steps": 60,  # the timesteps of hourglass_IE
    "noise_level_on_exact_BC": 0,
    # adds random uniform noise on boundary data before optimization starts. This number is between 0 and 1.
    # The noise level will be this number times the largest value of the boundary data.
}

# Use regularized Newton's method from section 4.2.1
# optimization_dict = {"solver": "moola_newton",
#                      "options": {'jtol': 1e-15, 'gtol': 1e-15, 'maxiter': 300},
#                      # unless machine precision results, the optimization algorithm runs for maxiter times,
#                      # or until CTRL + C is pressed
#                      "inner_product": "H1"}

# Use a BFGS method
optimization_dict = {"solver": "moola_BFGS",
                     "options": {'jtol': 1e-15, 'gtol': 1e-15, 'maxiter': 25, 'mem_lim': 30, 'rjtol': None,
                                 'rgtol': None},
                     # unless machine precision results, the optimization algorithm runs for maxiter times,
                     # or until CTRL + C is pressed
                     "inner_product": "H1"}

cost_functional_dict = {
    "final_smoothing_lambda": "lambda t: exp(-0.005/pow(t,2)) if t > DOLFIN_EPS else 0.0",
    # temporal weight from figure 4.2
    "discretization": None,  # sets the right discretization of the cost functional according to the PDE scheme
    "H1_smoothing": 0  # add Tikhonov regularization of the form + H1_smoothing ||Dq||_{L^2}^2, q the radial control
}
