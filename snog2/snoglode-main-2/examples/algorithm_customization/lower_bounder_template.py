"""
Write your own lower bounder using this template.
It must be passed to the snoglode.solver.SNoGloDe initilization as the lower_bounder arguement.

NOTE: this template is specifically formatted to be plugged in seamlessly.
      in particular, inheritance, function, parameters, and returns.
      Everything else is customizable.
"""
from typing import Tuple, Optional
import snoglode as sno
import pyomo.environ as pyo

# TODO: come up with a better class name :)
class CustomLowerBounder(sno.AbstractLowerBounder):

    def __init__(self, 
                 solver: str, 
                 time_ub: float) -> None:
        
        # ultimately is passed to the BoundingProblemBase parent class init.
        # see: snoglode.bounders.base
        super().__init__(solver = solver,
                         time_ub = time_ub)
        
        # TODO: add any other inits you need.


    def solve_a_subproblem(self, 
                           subproblem_name: str, 
                           subproblem_model: pyo.ConcreteModel, 
                           subproblem_lifted_vars: dict) -> Tuple[bool, Optional[float]]:
        """
        This must be defined in the child class.
        It must always take these inputs, to maintain fluidity within the solver.
        
        Options here could be to simply solve as is, perform OBBT / FBBT, 
        generate a convex relaxation & solve, etc.

        Parameters
        -----------
        subproblem_name : str
            String corresponding to this subproblems name
        subproblem_model : pyo.ConcreteModel
            pyomo model corresponding to this subproblem
        subproblem_lifted_vars : dict
            dictionary corresponding to this current subproblems lifted variables

        Returns
        -----------
        feasible : bool
            if the solve of the subproblem model was feasible or not
        objective : float
            objective value of the subproblem model; None if infeasible.
        """

        
        # TODO: solve the subproblem in some way.


        subproblem_is_feasible = None       # TODO: bool representing if feasible or not
        subproblem_objective = None         # TODO: float representing the subproblem obj.

        raise NotImplementedError
        return subproblem_is_feasible, subproblem_objective