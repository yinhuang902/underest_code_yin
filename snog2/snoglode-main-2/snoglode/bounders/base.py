"""
Simple base class for all of the solvers.
All solvers - lower bound, upper bound, candidate generator
"""
# suppress warnings when loading infeasible models
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR) 


class BoundingProblemBase():
    """
    Initalizes all of the common elements of the bounding problems.
    Mostly solver information.
    """

    def __init__(self, 
                 solver, 
                 time_ub: float = 600) -> None:
        """
        Initializes the solver information.

        Parameters
        -----------
        solver : pyo.SolverFactory
            initialized Pyomo solver factory object
        time_ub : int
            time (seconds) to max out solve.
        """
        # init the opt object + save solver name
        self.opt = solver
        if (solver != None): self.solver = solver.name
        # if we do not require a solver for CG, might be None

        assert type(time_ub)==int or type(time_ub)==float
        assert time_ub>0
        self.time_ub = time_ub
    
# ================================================================================================ #