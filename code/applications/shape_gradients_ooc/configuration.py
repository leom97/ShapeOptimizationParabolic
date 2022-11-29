import numpy as np
from dolfin import *

"""

Configuration file for shape_gradient_ooc_verification.py

Set:
pde_dict["ode_scheme"] = "crank_nicolson", manual = False
pde_dict["ode_scheme"] = "crank_nicolson", manual = True
to reproduce table 4.1, and:
pde_dict["ode_scheme"] = "implicit_euler", manual = False
for table 4.2.

Note, the run for the implicit Euler method is very computationally expensive, and we have therefore downscaled the 
experiment, see also section 4.2.2.

"""

experiment_dict = {
    "N_it": 7,  # how many refinements for the spatial mesh?
    "dt_multiplier": 5,  # to relate dt and h, see section 4.2.2: dt = dt_multiplier * h^(1 or 2)
}

# This dictionary describes the domain on which the shape gradient is computed (domain), and on which the spherical
# functions live (sphere)
# We only used a 2D annulus for this
geometry_dict = {
    "domain": {"type": "annulus", "resolution": None, "ext_refinement": 1.0, "int_refinement": 1.0, "inner_radius": 1,
               "outer_radius": 2,
               "center": np.array([0, 0]), "reload_xdmf": False},
    "sphere": {"dimension": 2, "resolution": 0.5},
    "q_ex_lambda": 'lambda x: 0'
}
# The resolution of the domain mesh is dictated by the following vector: every entry correspond to a mesh width,
# which should get progressively finer
h_tentative = 1 / (2 ** np.arange(0, experiment_dict["N_it"]))

# Change "ode_scheme" to "implicit_euler" or "crank_nicolson"
pde_dict = {
    "ode_scheme": "crank_nicolson",
    "marker_dirichlet": [3],  # interior of annulus
    "marker_neumann": [2],  # exterior of annulus
    "T": 1,  # final time of simulation
    "N_steps": None
}

# Describes the function \eta of figure 4.2, the smoothing parameter a = 0.005 could be tweaked
cost_functional_dict = {
    "final_smoothing_lambda": "lambda t:  exp(-0.005/pow(t,2)) if t > DOLFIN_EPS else 0.0",
    "discretization": None,
    "H1_smoothing": 0
}

# Smooth_displacements_dict["x"][i], smooth_displacements_dict["y"][i], represents a smooth vector field
# Add as many smooth vector fields as you wish, respecting the syntax rules of
# code/utilities/ooc_verification.py -> string_to_vector_field
smooth_displacements_dict = {
    "x": ["x^3", ".5*x", ".5*y^3", ".2*y", ".2*x+.2*y^2", ".2*x+.2*y^2", ".2*x*y+.2*y*2", "-.5*x-.2*x*y", ".5*x+.4*x*y",
          "x"],
    "y": [".5*y", "y^3", ".2*x", ".5*x^3", ".5*x^3-.2*x", ".5*x^3-.2*x*y", ".5*x^3-.2*x*y", "-.4*x^2*y+.4*y^2",
          "-.3*y^3-.1*x*y", "y"]
}


# Pde data: used to set up the states v, w of section 3.1
class _f(UserExpression):  # Dirichlet datum
    def __init__(self, t, **kwargs):
        super().__init__(self, **kwargs)
        self.t = t

    def eval(self, values, x):
        values[0] = self.t ** 3 * np.cos(5 * self.t) * (x[0] ** 2 + x[1])

    def value_shape(self):
        return ()


class _g(UserExpression):  # Nuemann datum
    def __init__(self, t, **kwargs):
        super().__init__(self, **kwargs)
        self.t = t

    def eval(self, values, x):
        values[0] = self.t ** 2 * np.sin(5 * self.t) * np.sin(x[0])

    def value_shape(self):
        return ()


# Set:
# pde_dict["ode_scheme"] = "crank_nicolson", manual = False
# pde_dict["ode_scheme"] = "crank_nicolson", manual = True
# to reproduce table 4.1, and:
# pde_dict["ode_scheme"] = "implicit_euler", manual = False
# for table 4.2.
manual = True  # it means that the expression of the shape gradient is computed by us (True)
# or by dolfin-adjoint (False)
