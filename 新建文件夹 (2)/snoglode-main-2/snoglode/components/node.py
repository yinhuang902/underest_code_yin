"""
Class for a single node in the Branch and Bound tree.
"""
from snoglode.utils.solve_stats import OneLowerBoundSolve
from snoglode.utils.supported import SupportedVars

from enum import Enum
class NodeDirection(Enum):
    upward  = 'up'
    downward = 'down'
    root = "root"


class Node():
    """
    Object representing a singular node in a tree.
    """

    def __init__(self, 
                 to_branch: dict, 
                 state: dict, 
                 id: int):
        """
        Creates the a node in the tree.
        Model must hold all relevant bounding / fixed values (i.e. state).

        Parameters
        ----------
            to_branch : dict
                each key corresponds to a string name of a type of variable
                each key has an associated list of string, with the variable names
                of all variables that HAVE NOT previously been branched on in the tree.
            state : dict
                each key corresponds to a string name of a type of variable
                each key has an associated set, with the information dictating
                the state of that variable for this node.
            id : int
                value for identifying the node in the tree
        """
        assert type(to_branch)==dict
        for var_type in list(to_branch.keys()): assert var_type in SupportedVars
        self.to_branch = to_branch

        assert type(state) == dict
        for var_type in list(state.keys()): assert var_type in SupportedVars
        self.state = state

        assert type(id)==int or type(id)==float
        self.id = id

        self.ub_problem = UpperBoundNodeMetrics()
        self.lb_problem = LowerBoundNodeMetrics()

        # determine if this node is terminal or not
        self.terminal = self._is_terminal()

    
    def _init_psuedocost_data(self,
                              dir: NodeDirection,
                              parent_id: int,
                              parent_obj: float,
                              branched_on: str,
                              branched_on_avg: float,
                              branched_var_lb: float,
                              branched_var_ub: float) -> None:
        """
        initializes optional information used for branching.
        in the case of pseudocost.
        
        Parameters
        ----------
        dir: NodeDirection
            indicates which way we branched on this variable
        parent_id : int
            value for identifying the parent node of this node.
        parent_obj: float
            the objective of the parent node.
        branched_on : str
            lifted_var_id for which we branched on
        branched_on_avg : float
            average parent solution for the variable we branched on
        branched_var_lb : float
            lb of the *parent* node's var, before branching
        branched_var_ub : float
            ub of the *parent* node's var, before branching
        """
        self.parent_id = parent_id
        self.parent_obj = parent_obj

        assert type(dir) == NodeDirection
        self.dir = dir

        self.branched_on = branched_on
        self.var_delta = branched_on_avg
        self.parent_var_lb = branched_var_lb
        self.parent_var_ub = branched_var_ub


    def __lt__(self, other_node) -> bool:
        """
        returns if the current nodes LB objective is strictly less than the 
        passed nodes LB objective.

        Parameters
        ----------
        other_node : Node
            node object representing the other node in question.
        
        Returns
        ----------
        bool
            if current LB objective is strictly less than the other LB objective.
        """
        return self.lb_problem.objective < other_node.lb_problem.objective


    def __le__(self, other_node) -> bool:
        """
        returns if the current nodes LB objective is less than
        or equal to the passed nodes LB objective.

        Parameters
        ----------
        other_node : Node
            node object representing the other node in question.

        Returns
        ----------
        bool
            if current LB objective is leq the other LB objective.
        """
        return self.lb_problem.objective <= other_node.lb_problem.objective


    def _is_terminal(self) -> bool:
        """
        Returns True if this is a terminal node (i.e. no vars left to branch on)
        Return False if this is not a terminal node (i.e. more vars can be branched)
        """
        vars_to_branch_on = 0
        for var_type in self.to_branch.keys():
            vars_to_branch_on += len(self.to_branch[var_type])
        return vars_to_branch_on == 0


    def display(self,
                lb: bool = False,
                ub: bool = False) -> None:
        """
        Prints all information related to the current node.

        optionally prints lb/ub data - this can be confusing,
        and is off by default, because the node inherits the solution
        data from the parent. be aware that *where* this data is printed,
        it might be the result of the child node OR from the parent node.
        """
        print("\n------------------  NODE DATA  -----------------")
        print(f"i.d. = {self.id}")
        print(f"terminal? {self.terminal}\n")
        print("STATE:")
        for var_type in self.state:
            print(f"  variable type = {var_type}\n")
            for varID in self.state[var_type]:
                var = self.state[var_type][varID]
                
                print(f"\tvariable: {varID}")
                print(f"\t  lb = {var.lb}")
                print(f"\t  ub = {var.ub}")
                if var_type == SupportedVars.binary or var_type == SupportedVars.integers \
                   or var_type == SupportedVars.nonnegative_integers:
                    print(f"\t  fixed? {var.is_fixed}")
                    if var.is_fixed: print(f"\t  value = {var.value}")
                print()
        
        print("\nTO BRANCH:")
        for var_type in self.state:
            print(f"  variable type = {var_type}")
            for varID in self.to_branch[var_type]:
                print(f"\t - {varID}")

        if lb:
            print()
            print("\nLOWER BOUND SOLVE:")
            print(f"  feasible? {self.lb_problem.feasible}")
            if self.lb_problem.feasible:
                print(f"  objective = {self.lb_problem.objective}")
        
        if ub:
            print()
            print("\nUPPER BOUND SOLVE:")
            print(f"  feasible? {self.ub_problem.feasible}")
            if self.ub_problem.feasible:
                print(f"  objective = {self.ub_problem.objective}")

        print("\n------------------------------------------------\n")
    

class LowerBoundNodeMetrics():
    """
    Class for holding information regarding the node metrics.
    There will be one for the UB & one for the LB.
    """
    def __init__(self) -> None:
        self.feasible = None
        self.objective = None
        self.subproblem_solutions = None
        

    def is_feasible(self, 
                    statistics: OneLowerBoundSolve) -> None:
        """
        Sets the bound metrics to reflect  the feasible solution.

        Parameters 
        ----------
        statistics : snoglode.utils.solve_stats.SubproblemStatistics
        """
        assert type(statistics)==OneLowerBoundSolve

        self.feasible = True
        self.objective = statistics.aggregated_objective
        self.subproblem_solutions = statistics.subproblem_solutions


    def is_infeasible(self) -> None:
        """
        Sets the lower bound metrics to reflect an infeasible solution.
        """
        self.feasible = False
        self.objective = float("inf")          # obj of infeasible LB node = infinity


class UpperBoundNodeMetrics():
    """
    Class for holding information regarding the node metrics.
    There will be one for the UB & one for the LB.
    """
    def __init__(self) -> None:
        self.feasible = None
        self.objective = None
    

    def is_feasible(self, 
                    objective: float) -> None:
        """
        Sets the bound metrics to reflect  the feasible solution.

        Parameters 
        ----------
        objective : UB objective value
        """
        self.feasible = True
        self.objective = objective


    def is_infeasible(self) -> None:
        """
        Sets the upper bound metrics to reflect an infeasible solution.
        """
        self.feasible = False
        self.objective = float("inf")          # obj of infeasible LB node = -infinity
