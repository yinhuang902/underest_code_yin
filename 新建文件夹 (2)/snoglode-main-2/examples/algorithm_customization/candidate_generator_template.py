"""
Write your own candidate generation using this template.
It must be passed to the snoglode.solver.SNoGlode initilization as the candidate_generator argument.

NOTE: this template is specifically formatted to be plugged in seamlessly.
      in particular, inheritance, function, parameters, and returns.
      Everything else is customizable.
"""
from typing import Tuple
import snoglode as sno

# TODO: come up with a better class name :)
class CustomCandidateGenerator(sno.AbstractCandidateGenerator):

    def __init__(self, 
                 solver: str, 
                 solver_options: dict, 
                 subproblems: sno.Subproblems, # used to attach a list of lifted_var_ids in the parent class
                 time_ub: float) -> None:
        
        super().__init__(solver = solver,
                         solver_options = solver_options,
                         time_ub = time_ub)
        
        # do we ALWAYS have to solve the upper bound problem?
        # NOTE: be careful - there is very few times I expect you not to have to do this.
        self.ub_required = True
        
        # TODO: add any other inits you need.


    def generate_candidate(self, 
                           node: sno.Node, 
                           subproblems: sno.Subproblems) -> Tuple[bool, dict, float]:
        """
        This is a general template - the method name & the inputs are fixed. Do not change them.

        This method should return a boolean (candidate_found) to indicate
        if the solve was successful & a dictionary, that has a key for
        each of the lifted variables and a corresponding value.

        NOTE: if we do not find a candidate, send an emtpy dict / obj back.
              They will not be checked / used if we did not find a candidate.

        Parameters
        ----------
        node : Node
            node object representing the current node we are exploring in the branch 
            and bound tree. Contains all bounding information.
        subproblems : Subproblems
            initialized subproblem manager.
            contains all subproblem names, models, probabilities, and lifted var lists/

        Returns
        ----------
        candidate_found : bool
            If this method successfully found a candidate or not
        candidate_solution : dict
            For each of the lifted variable ID's, therre should be a 
            corresponding value to fix the variable to.
        candidate_solution_obj : float
            If we do not want a global guarantee, an objective value is necessary
            to be produced.
        """
    
        # TODO: generate a candidate solution in some way.


        # --- snip -----


        # necessary information to return
        candidate_found = None                      # TODO
        candidate_solution = None                   # TODO
        candidate_solution_obj = None               # TODO

        # --- snip -----

        raise NotImplementedError
        return candidate_found, candidate_solution, candidate_solution_obj