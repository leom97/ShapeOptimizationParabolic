from .optimisation_algorithm import *
from .bfgs import LimitedMemoryInverseHessian, LinearOperator
from numpy import sqrt
import signal
import sys
import logging


def dual_to_primal(x):
    return x.primal()


def primal_to_dual(x):
    return x.dual()


class RegularizedNewton(OptimisationAlgorithm):
    '''
    An inexact newton method.
    '''
    __name__ = 'RegularizedNewton'

    def __init__(self, problem, initial_point=None, precond=LinearOperator(dual_to_primal), options={}):
        '''
        Initialises the Hybrid CG method. The valid options are:
         * options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
            - tol: Not supported yet - must be None.
            - maxiter: Maximum number of iterations before the algorithm terminates. Default: 200.
            - disp: dis/enable outputs to screen during the optimisation. Default: True
            - gtol: Gradient norm stopping tolerance: ||grad j|| < gtol.
            - line_search: defines the line search algorithm to use. Default: strong_wolfe
            - line_search_options: additional options for the line search algorithm. The specific options read the help
              for the line search algorithm.
            - an optional callback method which is called after every optimisation iteration.
          '''

        # Set the default options values
        self.problem = problem
        self.set_options(options)
        self.linesearch = get_line_search_method(self.options['line_search'], self.options['line_search_options'])
        self.data = {'control': initial_point,
                     'iteration': 0,
                     'precond': precond}

    def __str__(self):
        s = "Regularized Newton method.\n"
        s += "-" * 30 + "\n"
        s += "Line search:\t\t %s\n" % self.options['line_search']
        s += "Maximum iterations:\t %i\n" % self.options['maxiter']
        return s

    # set default parameters

    @classmethod
    def default_options(cls):
        # this is defined as a function to prevent defaults from being changed at runtime.
        default = OptimisationAlgorithm.default_options()
        default.update(
            # generic parameters:
            {"jtol": None,
             "gtol": 1e-4,
             "maxiter": 200,
             "display": 2,
             "line_search": "strong_wolfe",
             "line_search_options": {"start_stp": 1},
             "callback": None,
             "record": ("grad_norm", "objective"),

             # method specific parameters:
             "ncg_reltol": .5,
             "ncg_maxiter": 200,
             "ncg_hesstol": "default",
             })
        return default

    def solve(self, callback=None):
        '''
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem
         '''
        self.display(self.__str__(), 1)

        objective = self.problem.obj
        options = self.options

        B = self.data['precond']
        Binv = LinearOperator(primal_to_dual)
        x = self.data['control']
        i = self.data['iteration']

        ################################################################################################################
        # We solve - Mi H Mi dj = ((MiH)^2+aI) d, where Mi is the Gramiam matrix of the chosen scalar product
        ################################################################################################################

        # compute initial objective and gradient
        H = objective.hessian(x)
        J = objective(x)
        r = H(B * objective.derivative(x))  # initial residual ( with dk = 0) (one application of B is missing)
        r.scale(-1.)
        self.update({'objective': J,
                     'grad_norm': r.primal_norm()})
        self.record_progress()

        if options['ncg_hesstol'] == "default":
            import numpy
            eps = numpy.finfo(numpy.float64).eps
            ncg_hesstol = eps * numpy.sqrt(len(x))
        else:
            ncg_hesstol = options['ncg_hesstol']

        global stop
        stop = False

        def signal_handler(sig, frame):
            logging.info('Graceful interruption')
            global stop
            stop = True
            self.data['status'] = 0

        signal.signal(signal.SIGINT, signal_handler)

        # Start the optimisation loop
        while self.check_convergence() == 0 and stop == False:

            a = 2**(-i)

            if callback is not None:
                callback(None)
            self.display(self.iter_status, 2)
            p = Br = (
                        B * r)  # mapping residual to primal space: note, B != Id even when the L^2 scalar product is used!

            d = p.copy().zero()
            rBr = r.apply(Br)
            H = objective.hessian(x)  # here I should add the identity regularization

            # CG iterations
            cg_tol = min(options['ncg_reltol'] ** 2, sqrt(rBr)) * rBr
            cg_iter = 0
            cg_break = 0
            while cg_iter < options['ncg_maxiter'] and rBr >= cg_tol:
                print('TEST:', rBr, cg_tol)
                if False:  # i < options['initial_bfgs_iterations']:
                    d = Br
                    break
                Hp = H(p)  # this is a dual vector, great!
                p_dual = Binv * p
                BHp = B * Hp
                reg = (p_dual).apply(p)
                pAp = Hp.apply(BHp) + a * reg

                self.display('cg_iter = {}\tcurve = {}\thesstol = {}'.format(cg_iter, pAp, ncg_hesstol), 3)
                if pAp < 0:
                    # print 'TEST: not descent direction'
                    if cg_iter == 0:
                        # Fall back to steepest descent: this is still okay!
                        d = Br
                    # otherwise use the last computed pk
                    break

                if 0 <= pAp < ncg_hesstol:
                    if cg_iter == 0:
                        # Fall back to steepest descent
                        d = Br
                    # cg_break = 2
                    # try to use what we have
                    try:
                        self.do_linesearch(objective, x, d)  # TODO: fix this hack
                        # print 'TEST: below curvature treshold'
                        break
                    except:
                        pass
                # Standard CG iterations
                alpha = rBr / pAp   # NB, this seems way too large
                d.axpy(alpha, p)  # update cg iterate
                r.axpy(-alpha, H(BHp))  # update residual
                r.axpy(-alpha * a, p_dual)  # regularize

                Br = B * r
                t = r.apply(Br)
                rBr, beta = t, t / rBr,

                p.scale(beta)
                p.axpy(1., Br)

                cg_iter += 1

            # do a line search and update
            x, a = self.do_linesearch(objective, x, d)
            d.scale(a)

            J, oldJ = objective(x), J

            # evaluate gradient at the new point
            r = objective.derivative(x)
            r.scale(-1)

            i += 1

            if options['callback'] is not None:
                options['callback'](J, r)

            # store current iteration variables
            self.update({'iteration': i,
                         'control': x,
                         'grad_norm': r.primal_norm(),
                         'delta_J': oldJ - J,
                         'objective': J,
                         'lbfgs': B})
            self.record_progress()
        self.display(self.convergence_status, 1)
        self.display(self.iter_status, 1)

        # Creation of a results data structure suitable for my purpose
        class Result():
            pass

        self.results = Result()
        self.results.gradient_infty_hist = self.history["grad_norm"]
        self.results.energy_hist = self.history["objective"]
        self.results.fun = self.data['objective']
        self.results.jac = self.data["control"].data.vector()[:]
        self.results.nit = self.data["iteration"]
        self.results.success = self.data["status"]
        self.results.message = self.convergence_status
        self.results.njev = -1  # not counted
        self.results.nfev = -1  # not counted
        return self.data["control"].data, self.results
