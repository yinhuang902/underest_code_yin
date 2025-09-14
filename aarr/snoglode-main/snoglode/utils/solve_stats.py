"""
These classes are intended for keeping track of either one iteration
of the upper bounding problem or lower bounding problem.
"""
import pyomo.environ as pyo
from snoglode.utils.supported import SupportedVars
# from snoglode.components.subproblems import Subproblems

class OneLowerBoundSolve():
    """
    This manages all of the information regarding the subproblem
    solve statistics for a single LB problem solve.

    We track:
        - aggregated objective
        - subproblem specific objective
        - lifted variable solution values per subproblem
    """

    def __init__(self, 
                 subproblem_names: list) -> None:
        """
        Initializes the aggregated objective to zero and creates
        individual statistics managers for the each subproblem.

        Parameters
        ----------
        subproblem_names : list[str]
            names of all of the subproblems
        """
        # keeping track of the aggregated objective over the solve of the problem
        self.aggregated_objective = 0

        # keeping track of individual subproblem information
        self.subproblem_solutions = {}
        for subproblem_name in subproblem_names:
            self.subproblem_solutions[subproblem_name] = StatisticsOfOneLBSubproblem()


    def update(self, 
               subproblem_name: str, 
               subproblem_objective: float, 
               subproblems):
        """
        Updates these statistics to reflect the current subproblem state.

        We keep these updated 

        Parameters
        ----------
        subproblem_name : str
            name corresponding to the current subproblem
        subproblem_objective : float
            objective of the current LB
        subproblems : Subproblems
            Subproblems object holding all rank subproblem information
        """
        # add the weighted subproblem objective to the overall objective
        subproblem_probabilty = subproblems.probability[subproblem_name]
        self.aggregated_objective += subproblem_objective * subproblem_probabilty

        # update singular subproblem objective
        self.subproblem_solutions[subproblem_name].objective = subproblem_objective

        # update the variable results
        for lifted_var in subproblems.subproblem_lifted_vars[subproblem_name]:

            # get variable object associated with this ID
            lifted_var_domain, lifted_var_id, _ = subproblems.var_to_data[lifted_var]
            
            # extract the current variable value
            lifted_var_value = pyo.value(lifted_var)

            # determine the domain based on where this lifted var is stored
                    # update the lifted var value
            self.subproblem_solutions[subproblem_name].lifted_var_solution[lifted_var_domain][lifted_var_id] \
                        = lifted_var_value
                    
    
    def update_to_parent(self,  
                         subproblem_name: str, 
                         subproblem_objective: float, 
                         subproblems,
                         parent_lifted_var_solution: dict):
        """
        Updates these statistics to reflect the parent nodes solution.
        Because we inherit the solutions already from the parent,
        no need to update objective or variable solution information

        Parameters
        ----------
        subproblem_name : str
            name corresponding to the current subproblem
        subproblem_objective : float
            objective of the current LB
        subproblems : Subproblems
            Subproblems object holding all rank subproblem information
        """
        # add the weighted subproblem objective to the overall objective
        subproblem_probabilty = subproblems.probability[subproblem_name]
        self.aggregated_objective += subproblem_objective * subproblem_probabilty

        # update singular subproblem objective
        self.subproblem_solutions[subproblem_name].objective = subproblem_objective

        # update to parent solution
        self.subproblem_solutions[subproblem_name].lifted_var_solution = parent_lifted_var_solution


class StatisticsOfOneLBSubproblem(): 
    """
    Maintains the statistics of a single LB subproblem.
    """

    def __init__(self) -> None:
        """
        Init the objective, lifted_var_solution to None
        """
        self.objective = None

        # we track this at LB to scrape info for the UB
        self.lifted_var_solution = {var_type: {} for var_type in SupportedVars}


class OneUpperBoundSolve():
    """
    This manages all of the information regarding the subproblem
    solve statistics for a single LB problem solve.

    We track:
        - aggregated objective
        - subproblem specific objective
    """

    def __init__(self, 
                 subproblem_names: list) -> None:
        """
        Initializes the aggregated objective to zero and creates
        individual statistics managers for the each subproblem.

        Parameters
        ----------
        supported_var_types : list
            This is needed
        
        """
        # keeping track of the aggregated objective over the solve of the problem
        self.aggregated_objective = 0

        # keeping track of individual subproblem information
        self.subproblem_solutions = {}
        for subproblem_name in subproblem_names:
            self.subproblem_solutions[subproblem_name] = StatisticsOfOneUBSubproblem()


    def update(self, 
               subproblem_name: str, 
               subproblem_objective: float, 
               subproblem_probabilty: float):
        """
        Updates these statistics to reflect the current subproblem state.

        Parameters
        ----------
        subproblem_name : str
            name corresponding to the current subproblem
        subproblem_objective : flaot
            objective of the current UB
        subproblem_probability : float
            probability of this subproblem.
        """
        # add the weighted subproblem objective to the overall objective
        self.aggregated_objective += subproblem_objective * subproblem_probabilty

        # update singular subproblem objective
        self.subproblem_solutions[subproblem_name].objective = subproblem_objective


class StatisticsOfOneUBSubproblem(): 
    """
    Maintains the statistics of a single subproblem.
    """

    def __init__(self) -> None:
        """
        Init the objective, lifted_var_solution to none
        """
        self.objective = None



class SNoGloDeSolutionInformation():
    """
    Keeps track of solution / algorithm.
    """

    def __init__(self) -> None:
        """
        We need to keep track of the best candidate solution / objective.
        """
        self.objective = None
        self.subproblem_solutions = None
        self.iter_found = None
        self.relative_gap = None

    
    def update_best_solution(self, 
                             objective: float, 
                             full_solution: dict, 
                             iteration: int, 
                             relative_gap: float):
        """
        Updates the saved UB information with the solution
        attached to the current node.

        Parameters
        ----------
        objective: float
            objective of the ub problem.
        full_solution: dict
            dictionary representing the full solution of the UB solve.
        iteration : int
            current iteration
        relative_gap : float
            *relative optimality gap of new solution.
        """
        self.objective = objective
        self.subproblem_solutions = full_solution
        self.iter_found = iteration
        self.relative_gap = relative_gap