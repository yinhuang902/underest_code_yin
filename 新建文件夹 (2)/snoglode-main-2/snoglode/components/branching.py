"""
Tailored branching strategies for the tree.

This was written intentionally to be abstract such that
the user can easily define their own or plug in their choices.

There are two general architectures:
    (1) variable selection
    (2) paritioning
"""
import numpy as np
np.random.seed(17)
from typing import Tuple

from snoglode.components.node import Node, NodeDirection
from snoglode.components.subproblems import Subproblems
from snoglode.utils.supported import SupportedVars
import snoglode.utils.compute as compute

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


class SelectionStrategy():
    def __init__(self, *args, **kwargs) -> None:
        pass

    def select_variable(self, 
                        node: Node, 
                        subproblems: Subproblems) -> Tuple[SupportedVars, str]:
        """
        returns a lifted var name to branch on & var_type
        this is a derived function of a child class.
        
        Parameters
        -------------
        node : Node
            Current node in the spatial BnB tree.
        subproblems : Subproblems
            Subproblems objective containing this ranks subproblems.

        Returns
        --------------
        var_type : str
            type of variable - should be in SupportedVars
        var_name : str
            ID corresponding to the lifted variable that was selected.
        """
        raise NotImplementedError('should be implemented by derived classes')
    

class PartitionStrategy():

    def __init__(self, *args, **kwargs) -> None:
        pass

    def split_point(self, 
                    varID: str,
                    var_lb: float, 
                    var_ub: float,
                    node: Node,
                    subproblems: Subproblems) -> float:
        """
        given a variable UB / LB, determine a split point.

        NOTE: this is not called if the selected variable is binary.

        Parameters
        -------------
        var_lb : float
            current lowerbound of the variables.
        var_ub : float
            current upperbound of the variables.
        node : Node
            current node in the BnB tree

        Returns
        --------------
        split_point : float
            point at which to split the continuous domain
            var_lb <= split_point <= var_ub
        """
        raise NotImplementedError('should be implemented by derived classes')
    

    def new_bounds(self, 
                   var_name: str, 
                   var_type: str, 
                   node: Node,
                   subproblems: Subproblems) -> Tuple[float, float, float]:
        """
        Uses the abstract split_point() function and the current variable
        to return the LB / UB / split of the variable

        Parameters
        --------------
        var_type : str
            type of variable - should be in SupportedVars
        var_name : str
            ID corresponding to the lifted variable that was selected.
        node : Node
            Current node in the spatial BnB tree.

        Returns
        --------------
        var_lb : float
            current lowerbound of the variables.
        split_point : float
            point at which to split the continuous domain
            var_lb <= split_point <= var_ub
        var_ub : float
            current upperbound of the variables.
        """
        # get ub / lb from model
        var_ub = node.state[var_type][var_name].ub
        var_lb = node.state[var_type][var_name].lb

        # determine split point
        split_point = self.split_point(varID = var_name,
                                       var_lb = var_lb,
                                       var_ub = var_ub, 
                                       node = node, 
                                       subproblems=subproblems)

        left_child_lb = var_lb
        left_child_ub = split_point
        right_child_lb = left_child_ub
        right_child_ub = var_ub

        # arithmetic checks
        assert left_child_lb <= left_child_ub
        assert right_child_lb <= right_child_ub
        assert left_child_ub == right_child_lb

        # ensure branching point keeps a min. distance from var. bounds
        return left_child_lb, left_child_ub, right_child_ub


# =================== SELECTION STRATEGIES ================================= #

class RandomSelection(SelectionStrategy):
    """
    Randomly selects a variable from all those available
    No descriminations made.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    def name(self):
        return "RandomSelection"

    def select_variable(self, 
                        node: Node, 
                        subproblems: Subproblems) -> Tuple[SupportedVars, str]:
        """
        randomly select any variable
        """
        # get lists of all variables left to branch on 
        # NOTE: sorting is very important for coordination over ranks 
        binaries = sorted(list(node.to_branch[SupportedVars.binary]))
        reals = sorted(list(node.to_branch[SupportedVars.reals]))
        integers = sorted(list(node.to_branch[SupportedVars.integers]))
        nonnegative_integers = sorted(list(node.to_branch[SupportedVars.nonnegative_integers]))
        vars = binaries + reals + integers + nonnegative_integers
        
        # randomly decide
        index = MPI.COMM_WORLD.bcast(np.random.randint(0, len(vars)), root=0)
        var_name = vars[index]

        # determine which var type we selected based on index value
        if binaries and (index in range(0, len(binaries))): 
            var_type = SupportedVars.binary
        elif reals and (index in range(len(binaries), (len(binaries) + len(reals)))):
            var_type = SupportedVars.reals
        elif integers and (index in range((len(binaries) + len(reals)), (len(binaries) + len(reals) + len(integers)))):
            var_type = SupportedVars.integers
        elif nonnegative_integers and (index in range((len(binaries) + len(reals) + len(integers)), len(vars))): 
            var_type = SupportedVars.nonnegative_integers

        return var_type, var_name
    

class MostInfeasibleBinary(SelectionStrategy):
    """
    Selects the binary variable that is the most violated;
    
    Once there are no binaries left, randomly selects
    the variables with continuous domain.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def name(self):
        return "MostInfeasibleBinary"
    
    def select_variable(self, 
                        node: Node, 
                        subproblems: Subproblems) -> Tuple[SupportedVars, str]:
        """
        prioritize binaries - select most infeasible.
        """
        # get lists of all variables left to branch on 
        # NOTE: sorting is very important for coordination over ranks 
        binaries = sorted(list(node.to_branch[SupportedVars.binary]))
        
        # prioritize binary branching first
        if (binaries):

            # get the averaged solution across all of the subrpoblems
            averaged_solution = compute.average_lb_solution(node = node,
                                                            subproblems = subproblems,
                                                            round_binaries = False)
            
            # select the binary variable that is closest to 0.5 (i.e. "most infeasible")
            var_name = None
            var_distance = 1
            for var_id in binaries:

                # compute how far away it is from 0.5
                distance = abs(averaged_solution[var_id] - 0.5)

                # want the closest to 0.5 as possible (i.e. smallest distance)
                if distance < var_distance:
                    var_name = var_id
                    var_distance = distance
        
            return SupportedVars.binary, var_name
        
        # if we have run out of binaries, move on to branching the rest
        else: 
            # randomly decide on any other var
            reals = sorted(list(node.to_branch[SupportedVars.reals]))
            integers = sorted(list(node.to_branch[SupportedVars.integers]))
            nonnegative_integers = sorted(list(node.to_branch[SupportedVars.nonnegative_integers]))
            vars = reals + integers + nonnegative_integers

            index = MPI.COMM_WORLD.bcast(np.random.randint(0, len(vars)), root=0)
            var_name = reals[index]

            # determine which var type we selected based on index value
            if reals and (index in range(0, len(reals))):
                var_type = SupportedVars.reals
            elif integers and (index <= range(len(reals), (len(reals) + len(integers)))):
                var_type = SupportedVars.integers
            if nonnegative_integers and (index <= range((len(reals) + len(integers)), len(vars))): 
                var_type = SupportedVars.nonnegative_integers

            return var_type, var_name


class MaximumDisagreement(SelectionStrategy):
    """
    Selects the variable that is the most violated based on
    the highest variance across subproblem solutions. 
    
    In this case, all domains can be considered because we will normalize.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def name(self):
        return "MaximumDisagreement"
    
    def select_variable(self, 
                        node: Node, 
                        subproblems: Subproblems) -> Tuple[SupportedVars, str]:
        """
        Computes the variance of all variables (standardizing,
        when they are within a continuous/integer domain)

        Selects the variable with the maximum variance.
        """
        # compute the variance across all subproblem LB solutions
        variance = compute.variance_lb_solution(node = node,
                                                subproblems = subproblems,
                                                normalize = True)
        
        # determine which (normalized) variance is the largest
        varID = max(variance, 
                    key = variance.get)
        
        for var_type in SupportedVars:
            if varID in node.state[var_type]: break
        
        return var_type, varID


class Pseudocost(SelectionStrategy):
    """
    Compute the pseudocost of branching on this variable, based
    on the progression of the tree / behavior of the variable in previous
    solutions.
    """
    def __init__(self,
                 id_to_var: dict,
                 var_to_data, # pyo.ComponentMap
                 *args, **kwargs) -> None:
        super().__init__()

        # can update this dynamically because it depends on full tree
        # (iow, do not have to recompute everything* at each node)
        self.pseudocost = {varID: {NodeDirection.upward:   0,
                                   NodeDirection.downward: 0} 
                                        for varID in id_to_var.keys()}
        self.explored   = {varID: {NodeDirection.upward:   0,
                                   NodeDirection.downward: 0} 
                                        for varID in id_to_var.keys()}
        
        # score parameter [0,1]
        self.mu = 1/6

        # unitialized pseudocost default score
        self.average_pseudocost = 1
        self.num_vars = len(id_to_var.keys())

        # keeps track of the scores / types for each of the variables
        self.scores = {}
        self.var_types = {}
        for varID in id_to_var.keys():
            self.scores[varID] = self.average_pseudocost
            var = id_to_var[varID][0] # just grab the first one - only need var_type
            var_type, _, _ = var_to_data[var]
            self.var_types[varID] = var_type
        
        # initialized indicates if we have solved both directions at least once for this var
        self.intialized = {varID: False for varID in id_to_var.keys()}

        # scoring function also relies on most recent variable change
        self.most_recent_var_delta = {varID: {NodeDirection.upward:   0,
                                              NodeDirection.downward: 0} 
                                                for varID in id_to_var.keys()}

    def name(self):
        return "Pseudocost"

    def update_data(self, 
                    node: Node,
                    subproblems: Subproblems) -> None:
        """
        given the current node / solutions,
        update the solved_nodes data and
        the branching_on_var_at_node data.
        """
        # only possible if we have a feasible problem
        if not node.lb_problem.feasible: return

        # compute the new average at this node (across all subproblems/ranks)
        node_var_avg = compute.average_var_lb_solution(node = node,
                                                       subproblems = subproblems,
                                                       var_ID = node.branched_on)
        
        # compute the delta for this variable & normalize
        var_avg_delta = abs(node.var_delta - node_var_avg)
        normalized_var_avg_delta = var_avg_delta / (node.parent_var_ub - node.parent_var_lb)
        if var_avg_delta == 0: var_avg_delta = 1 # avoid division by zero...
        if normalized_var_avg_delta == 0: normalized_var_avg_delta = 1 # avoid division by zero...

        # update most recent var delta (needed for computing score)
        # self.most_recent_var_delta[node.branched_on] = var_avg_delta
        self.most_recent_var_delta[node.branched_on] = normalized_var_avg_delta
        
        # compute the change in objective (parent obj <= child by def); avoid numerical issues
        obj_change = max(node.lb_problem.objective - node.parent_obj, 0)
        assert obj_change >= 0, \
            f"obj_change = {obj_change}, node.lb_problem.objective = {node.lb_problem.objective}, node.parent_obj = {node.parent_obj}"
        
        # compute pseudocost of this node
        # pseudocost = obj_change / var_avg_delta
        pseudocost = obj_change / normalized_var_avg_delta

        # update the solution for this node
        self.pseudocost[node.branched_on][node.dir] += pseudocost
        self.explored[node.branched_on][node.dir]   += 1

        # update the score
        self.update_scores(node.branched_on)


    def update_scores(self, 
                      varID: str) -> None:
        """
        Computes the score of the variable, based
        on the current upward and downward direction.
        """

        # if we have initialized costs in both directions, we can compute a score
        if self.explored[varID][NodeDirection.upward] > 0.0 \
                and self.explored[varID][NodeDirection.downward] > 0.0:
            
            # reset flag (not always needed, but let's be careful anyways)
            self.intialized[varID] = True

            # compute the scores of both directions
            upward_score = self.pseudocost[varID][NodeDirection.upward] * self.most_recent_var_delta[varID] \
                                / self.explored[varID][NodeDirection.upward]
            downward_score = self.pseudocost[varID][NodeDirection.downward] * self.most_recent_var_delta[varID] \
                                / self.explored[varID][NodeDirection.downward]
            
            # store old score & change average
            prev_score = self.scores[varID]
            self.scores[varID] = \
                (1 - self.mu) * min(upward_score, downward_score) + \
                    self.mu * max(upward_score, downward_score)

            # update new average and update uninitialized scores 
            self.update_avg_pseudocosts(varID, prev_score)

    def update_avg_pseudocosts(self,
                               varID: str,
                               prev_score: float) -> None:
        """
        When we do not have an initial score,
        take the average psuedocost score and set that
        as the score for all of the variables.
        """
        # recompute the average
        self.average_pseudocost = ((self.average_pseudocost * self.num_vars) \
            - prev_score + self.scores[varID]) / self.num_vars

        # update scores for uninitialized vars, if we still have some
        for varID in self.scores:
            if self.intialized[varID] == False: 
                self.scores[varID] = self.average_pseudocost

    def best_scoring_var(self,
                         node: Node) -> str:
        """
        returns which variables is currently scored as the hightest.

        Paremeters
        -----------
        node : Node
            current node of the BnB tree
        """
        best_var = ""
        best_score = float("-inf")
        tied = []

        # check each of the scores
        for varID in self.scores:
            var_type = self.var_types[varID]

            if self.scores[varID] > best_score \
                    and varID in node.to_branch[var_type]:
                
                # update best scores / var
                best_score = self.scores[varID]
                best_var = varID
                
                # if we have updated the best score, reset tied list
                tied = [varID]

            # if it is the same, update tied variables
            if self.scores[varID] == best_score \
                    and varID in node.to_branch[var_type]:
                tied.append(varID)
        
        # if we are tied, return a random selection
        if tied: 

            # randomly decide between binary or continuous
            index = MPI.COMM_WORLD.bcast(np.random.randint(0, len(tied)), root=0)
            best_var = tied[index]
            var_type = self.var_types[best_var]

        # otw, return best variable
        return var_type, best_var

    def select_variable(self, 
                        node: Node, 
                        subproblems: Subproblems) -> Tuple[SupportedVars, str]:
        """
        computes the pseduocosts based on previous
        branching results & selects variable in this manner.
        """
        # first, update the current data (if we are not the root & feas)
        if node.dir != NodeDirection.root and node.lb_problem.feasible:
            self.update_data(node = node,
                             subproblems = subproblems)
    
        # select the variable with the highest pseudocost
        return self.best_scoring_var(node)


class HybridBranching(Pseudocost):
    """
    In this case, we explore the tree is some other manner at first
    collecting the data to eventually switch to Pseudocost branching.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method_switch_iter = 10
        self.iter = 0
        self.first_method = MaximumDisagreement()
    
    def name(self):
        return "PseudocostHybridBranching"

    def select_variable(self, 
                        node: Node, 
                        subproblems: Subproblems) -> Tuple[SupportedVars, str]:
        """
        computes the pseduocosts based on previous
        branching results & selects variable in this manner.
        """
        self.iter += 1

        # first, update the current data (if we are not the root & feas)
        if node.dir != NodeDirection.root and node.lb_problem.feasible:
            self.update_data(node = node,
                             subproblems = subproblems)
    
        # in the beggining, default to MaximumDisagreement
        if self.iter <= self.method_switch_iter:
            return self.first_method.select_variable(node = node,
                                                     subproblems = subproblems)
        else:
            return self.best_scoring_var(node)

# TODO: generalize hybrid branching

# TODO: strong branching (linear/nonlinear)

# TODO: reliability branching
    
# =================== PARTITIONING STRATEGIES ============================= #

class Midpoint(PartitionStrategy):
    """
    Based on the lower bound and the upper bound,
    simply return the midpoint as the split point.
    """
    def __init__(self) -> None:
        super().__init__()
        self.epsilon = 1e-3

    def split_point(self, 
                    varID: str,
                    var_lb: float, 
                    var_ub: float,
                    node: Node,
                    subproblems: Subproblems) -> float:
        
        # branch by splitting the space in half
        return var_lb + round(0.5 * (var_ub - var_lb), ndigits = 3)


class ExpectedValue(PartitionStrategy):
    """
    Return the average value of all solutions.

    Make sure that we do not get too close to one
    particular bound, by considering a theta tolerance.
    """
    def __init__(self) -> None:
        super().__init__()
        self.epsilon = 1e-3

        # tolerance for partition vicinity to bounds
        self.theta = 0.1

    def split_point(self,
                    varID: str,
                    var_lb: float,
                    var_ub: float,
                    node: Node,
                    subproblems: Subproblems) -> float:
        
        # compute the expected value
        ev_var = compute.average_var_lb_solution(node = node,
                                                 subproblems = subproblems,
                                                 var_ID = varID)
        
        # compute bounds that would maintain a safe distance from current branching
        safe_lb = var_lb + self.theta * (var_ub - var_lb)
        safe_ub = var_ub - self.theta * (var_ub - var_lb)
        assert safe_lb <= safe_ub

        # reset EV if we are violating safe bounds
        if ev_var < safe_lb: return safe_lb
        if ev_var > safe_ub: return safe_ub
        
        # otw, return the EV 
        return ev_var