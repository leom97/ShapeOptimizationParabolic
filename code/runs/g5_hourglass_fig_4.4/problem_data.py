import numpy as np

exact_geometry_dict = {
    "domain": {"type": "annulus", "resolution": 0.05, "ext_refinement": 1.0, "int_refinement": 1.0, "inner_radius": 1,
               "outer_radius": 2,
               "center": np.array([0, 0]), "reload_xdmf": False},
    "sphere": {"dimension": 2, "resolution": 0.03},
    "q_ex_lambda": 'lambda circle_coords: -.8 * circle_coords[:, 0] ** 2 +.2 * circle_coords[:, 1] ** 2'
    # circle_coords are in dof order
}

exact_pde_dict = {
    "ode_scheme": "crank_nicolson",
    "u_N": "Expression('t', t=0, a=.05, p = np.pi, degree=5)",
    "marker_dirichlet": [3],
    "marker_neumann": [2],
    "T": 2,
    "N_steps": 120  # a lot of time steps for the exact equation
}

simulated_geometry_dict = {
    "additional_domain_data": {"resolution": 0.1},
    "sphere_resolution": 0.15,
}

simulated_pde_dict = {
    "ode_scheme": "crank_nicolson",
    "N_steps": 60,  # the timesteps of hourglass_IE
    "noise_level_on_exact_BC": 0
}

optimization_dict = {"solver": "moola_BFGS",
                     "options": {'jtol': 1e-15, 'gtol': 1e-15, 'maxiter': 25, 'mem_lim': 30, 'rjtol': None,
                                 'rgtol': None},
                                 "inner_product": "H1"}

cost_functional_dict = {
    "final_smoothing_lambda": "lambda t: exp(-0.005/pow(t,2)) if t > DOLFIN_EPS else 0.0",  # it will be evaluated in T-t
    "discretization": None,
    "H1_smoothing": 0
}