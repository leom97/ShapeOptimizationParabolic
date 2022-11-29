import numpy as np
from dolfin import *

"""
Set:
pde_dict["ode_scheme"] = "crank_nicolson", manual = False
pde_dict["ode_scheme"] = "crank_nicolson", manual = True
to reproduce table 4.1, and:
pde_dict["ode_scheme"] = "implicit_euler", manual = False
for table 4.2.

Note, the run for the implicit Euler method is very computationally expensive, and we have therefore downscaled the 
experiment, see also section 4.2.2.

"""

geometry_dict = {
    "domain": {"type": "annulus", "resolution": None, "ext_refinement": 1.0, "int_refinement": 1.0, "inner_radius": 1,
               "outer_radius": 2,
               "center": np.array([0, 0]), "reload_xdmf": False},
    "sphere": {"dimension": 2, "resolution": 0.5},
    "q_ex_lambda": 'lambda x: 0'
}

pde_dict = {
    "ode_scheme": "crank_nicolson",
    "u_N": "Expression('sin(3*t)*(pow(x[0],2)-cos(4*x[1]))', t=0, a=.05, p = np.pi, degree=5)",
    "marker_dirichlet": [3],
    "marker_neumann": [2],
    "T": 1,
    "N_steps": None
}

cost_functional_dict = {
    "final_smoothing_lambda": "lambda t:  exp(-0.005/pow(t,2)) if t > DOLFIN_EPS else 0.0",
    "discretization": None,
    "H1_smoothing": 0
}

experiment_dict = {
    "N_it": 7,  # how many refinements for the spatial mesh?
    "dt_multiplier": 5, # to relate dt and h, see section 4.2.2
}

# smooth_displacements_dict["x"][i], smooth_displacements_dict["y"][i], represents the a smooth vector field
smooth_displacements_dict = {
    "x": ["x^3", ".5*x", ".5*y^3", ".2*y", ".2*x+.2*y^2", ".2*x+.2*y^2", ".2*x*y+.2*y*2", "-.5*x-.2*x*y", ".5*x+.4*x*y",
          "x"],
    "y": [".5*y", "y^3", ".2*x", ".5*x^3", ".5*x^3-.2*x", ".5*x^3-.2*x*y", ".5*x^3-.2*x*y", "-.4*x^2*y+.4*y^2",
          "-.3*y^3-.1*x*y", "y"]
}

# Pde data
class _f(UserExpression):
    def __init__(self, t, **kwargs):
        super().__init__(self, **kwargs)
        self.t = t

    def eval(self, values, x):
        values[0] = self.t ** 3 * np.cos(5 * self.t) * (x[0] ** 2 + x[1])

    def value_shape(self):
        return ()


class _g(UserExpression):
    def __init__(self, t, **kwargs):
        super().__init__(self, **kwargs)
        self.t = t

    def eval(self, values, x):
        values[0] = self.t ** 2 * np.sin(5 * self.t) * np.sin(x[0])

    def value_shape(self):
        return ()

manual = True  # it means, manual expression of the shape gradient
