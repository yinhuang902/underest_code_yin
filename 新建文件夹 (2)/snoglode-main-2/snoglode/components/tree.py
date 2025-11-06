"""
Class to manage the tree of nodes in the algorithm. 
"""
import numpy as np
np.random.seed(17)
import copy as cp
import math
from typing import Tuple

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

from snoglode.components.node import Node, NodeDirection
from snoglode.components.subproblems import Subproblems
from snoglode.components.branching import SelectionStrategy, PartitionStrategy
from snoglode.components.parameters import SolverParameters
from snoglode.utils.supported import SupportedVars
from .queues import NodeQueue, LIFONodeQueue, FIFONodeQueue, WorstBoundNodeQueue, QueueStrategy
import snoglode.utils.compute as compute

_queue_map = {}
_queue_map[QueueStrategy.fifo] = FIFONodeQueue
_queue_map[QueueStrategy.lifo] = LIFONodeQueue
_queue_map[QueueStrategy.bound] = WorstBoundNodeQueue


class Tree():
    """
    A functional tree for managing nodes / branching.

    Attributes
    ----------
    node_queue : NodeQueue
        que for managing nodes.
    upper_bound : dict
        holds the model (key="model") of the upper bound solution
        and the objective value (key="value")
    lower_bound : dict
        holds the model (key="model") of the lower bound solution
        and the objective value (key="value")
    variables : dict
        keeps track of variable types (binary, int, cont)
    metrics : dict
        interesting information (i.e., what nodes are pruned / how)

    """

    def __init__(
        self,
        params: SolverParameters,
        subproblems: Subproblems,
    ):
        """
        Initializes the root node in the tree.
        Initializes & organizes the info in the model.

        Parameters
        ----------
        params : SolverParameters
            instantiated solver parameters object,
            from which we will pull all necessary information.
        subproblems : Subproblems
            object containing all initialized subproblems.
        """
        self.epsilon = params._epsilon

        # queue for managing unexplored child nodes.
        self.node_queue: NodeQueue = _queue_map[QueueStrategy(params._queue_strategy)]()

        # queue for managing all terminal leaf nodes
        self.terminal_node_queue : NodeQueue = _queue_map[QueueStrategy.bound]()

        # metrics!
        self.metrics = TreeMetrics()

        # store verbose option
        self.verbose = params._verbose

        # branching strategies
        self.selection_strategy = params._selection_strategy(id_to_var = subproblems.id_to_vars,
                                                             var_to_data = subproblems.var_to_data,
                                                             solver = params._selection_strategy_solver)
        self.parition_strategy  = params._partition_strategy()
    
        # need a root node
        self.init_root_node(subproblems)

        # convergence metrics - updated at solve call
        self.max_iter = 100, 
        self.rel_tolerance = 1e-2,
        self.abs_tolerance = 1e-3,
        self.time_limit = math.inf, 
        

    def init_root_node(self, 
                       subproblems: Subproblems) -> None:
        """
        Takes the initialized scenarios and inits the root node.
        Assumed to be called after the instantiation of the scenarios.

        Parameters 
        -----------
        subproblems : Subproblems
            object containing all initialized subproblems.
        """
        state = subproblems.root_node_state
        to_branch = {
            var_type: list(state[var_type]) for var_type in SupportedVars
        }

        # create root node & add to que
        root = Node(to_branch = to_branch,
                    state = state,
                    id = 0)
        root.lb_problem.objective = -math.inf
        self.node_queue.push(root)

        # init optional data, if need
        if "Pseudocost" in self.selection_strategy.name():
            root._init_psuedocost_data(dir = NodeDirection.root,
                                       parent_id = None,
                                       parent_obj = None,
                                       branched_on = None,
                                       branched_on_avg = None,
                                       branched_var_lb = None,
                                       branched_var_ub = None)

        if self.verbose: print("Node rooted.")


    def get_node(self):
        """
        Node selection.
        """
        return self.node_queue.pop()


    def bound(self, 
              current_node: Node) -> str:
        """
        Bounds the tree based on the following:
          -  Infeasibility (if the lower bounding problem is infeasible)
          -  Bound (if the current solution will never improve beyond best known bound)
        
        NOTE: We do not account for integrality, like in classic BnB, because we do not relax
        the integers/binaries for this algorithm.

        Parameters 
        -----------
        current_node : Node
            current node in the spatial BnB tree.
        
        Returns
        -----------
        bound_result : str
            string representing what bounding changes were made.
        """
        # add iter to node count
        self.metrics.nodes.explored += 1

        # keep track of if this node changed the current UB/LB
        bound_updated = ""

        # prune by infeasibility if no feas. solution for LB
        if not current_node.lb_problem.feasible:

            # record metrics, delete node - do not need to branch
            self.metrics.nodes.pruned_by_infeasibility.append(current_node.id)
            del current_node

            return "pruned by infeasibility"

        # if feasible, check bounds & update/prune if necessary
        else:

            # if the LB > the current UB value, we can prune by bound.
            if (current_node.lb_problem.objective > self.metrics.ub):

                # record metrics, delete node - do not need to branch
                self.metrics.nodes.pruned_by_bound.append(current_node.id)
                del current_node

                return "pruned by bound"

            # if this is a better UB solution, update best UB
            if (current_node.ub_problem.objective < self.metrics.ub) \
                    or (self.metrics.ub==float("inf")):
                
                self.update_ub(current_node.ub_problem.objective)
                if self.metrics.ub != float("inf") : bound_updated += "ub"

            # return if the bound was updated
            return bound_updated
    

    def branch(self, 
               current_node: Node, 
               subproblems: Subproblems) -> str:
        """
        Branches based on the previous node solve. 
        Creates two children nodes, and adds them to the queue.
        
        NOTE: We update the LB here because we need to account for all open nodes
        This must be computed AFTER adding the child nodes.
        
        Parameters 
        -----------
        current_node : Node
            current node in the spatial BnB tree.
        subproblems : Subproblems
            object containing all initialized subproblems.
        
        Returns
        -----------
        bound_result : str
            string representing if lower bound was updated or not.
        """

        # if we are a terminal node - cannot branch.
        if (current_node.terminal): 
            self.terminal_node_queue.push(current_node)
            return self.update_lb()

        # select variable based on current strategy
        var_type, var_name = self.selection_strategy.select_variable(subproblems = subproblems,
                                                                     node = current_node)
        assert var_name in current_node.to_branch[var_type], \
            f"var_name = {var_name}, var_type = {var_type}, which is not in current_node.to_branch = {current_node.to_branch[var_type]}"

        # if this is a binary variable, no need to use a custom strategy
        if var_type == SupportedVars.binary:
            left_child_node, right_child_node = \
                self._spawn_binary_children(current_node = current_node,
                                            subproblems = subproblems,
                                            var_name = var_name)
        
        # use partitioning strategy to select how to branch on this variable
        if var_type == SupportedVars.reals: 
            left_child_node, right_child_node = \
                self._spawn_children(current_node = current_node,
                                     subproblems = subproblems,
                                     var_name = var_name)
        
        elif var_type in [SupportedVars.integers, SupportedVars.nonnegative_integers]:
            left_child_node, right_child_node = \
                self._spawn_integer_children(current_node = current_node,
                                             subproblems = subproblems,
                                             var_name = var_name,
                                             domain = var_type)
        
        # init optional data, if need
        if "Pseudocost" in self.selection_strategy.name():

            # compute the average 1st stage solution
            branched_on_avg = compute.average_var_lb_solution(node = current_node,
                                                              subproblems = subproblems,
                                                              var_ID = var_name)
            
            var = subproblems.id_to_vars[var_name][0]
            var_type, _, _ = subproblems.var_to_data[var]
            # add the left child info - dir is the important part
            left_child_node._init_psuedocost_data(dir = NodeDirection.downward,
                                                  parent_id = current_node.id,
                                                  parent_obj = current_node.lb_problem.objective,
                                                  branched_on = var_name,
                                                  branched_on_avg = branched_on_avg,
                                                  branched_var_lb = current_node.state[var_type][var_name].lb,
                                                  branched_var_ub = current_node.state[var_type][var_name].ub)
            
            # add the right child info - dir is the important part
            right_child_node._init_psuedocost_data(dir = NodeDirection.upward,
                                                   parent_id = current_node.id,
                                                   parent_obj = current_node.lb_problem.objective,
                                                   branched_on = var_name,
                                                   branched_on_avg = branched_on_avg,
                                                   branched_var_lb = current_node.state[var_type][var_name].lb,
                                                   branched_var_ub = current_node.state[var_type][var_name].ub)
            
        # add to queue
        assert (left_child_node != right_child_node)
        self.node_queue.push(left_child_node)
        self.node_queue.push(right_child_node)

        self.metrics.nodes.node_id_max += 2
        del current_node

        # lowerbound update
        # return self.update_lb()
    

    def _spawn_binary_children(self, 
                               current_node: Node,
                               subproblems: Subproblems,
                               var_name: str) -> Tuple[Node, Node]:
        """
        Takes the current node, and based on the variable name passed,
        we create two child nodes where the left is fixed to 1 
        and the right is fixed to 0.

        Parameters
        ---------------
        current_node : Node
            the current node in the sBnB tree.
        var_name : str
            the name of the current binary variable. 
        var_type : SupportedVars
            indicates if this is a relaxed or normal binary.

        Returns
        ----------------
        left_child_node  : Node
        right_child_node : Node
        """
        # LEFT child
        left_node_name = self.metrics.nodes.node_id_max + 1 # update name
        left_node_to_branch = cp.deepcopy(current_node.to_branch)
        assert var_name in left_node_to_branch[SupportedVars.binary]
        # remove this variable from the branching possibilities (cannot branch further)
        left_node_to_branch[SupportedVars.binary].remove(var_name)
        left_node_state = (cp.deepcopy(current_node.state))
        left_node_state[SupportedVars.binary][var_name].is_fixed = True
        left_node_state[SupportedVars.binary][var_name].value = 0
        left_child_node = Node(to_branch = left_node_to_branch,
                               state = left_node_state,
                               id = left_node_name)
            
        # attach the LB aggregated obj + related solutions for information
        left_child_node.lb_problem.objective = current_node.lb_problem.objective

        # init the subproblem solutions for adding cuts
        left_child_node.lb_problem.subproblem_solutions = current_node.lb_problem.subproblem_solutions

        # RIGHT child
        right_node_name = self.metrics.nodes.node_id_max + 2 # update name
        right_node_to_branch = cp.deepcopy(left_node_to_branch) # since we copy left, don't have to remove again
        right_node_state = (cp.deepcopy(current_node.state))
        right_node_state[SupportedVars.binary][var_name].is_fixed = True
        right_node_state[SupportedVars.binary][var_name].value = 1
        right_child_node = Node(to_branch = right_node_to_branch,
                                state = right_node_state,
                                id = right_node_name)
            
        # attach the LB aggregated obj + related solutions for information
        right_child_node.lb_problem.objective = current_node.lb_problem.objective
        right_child_node.lb_problem.subproblem_solutions = current_node.lb_problem.subproblem_solutions

        return left_child_node, right_child_node
    

    def _spawn_integer_children(self, 
                               current_node: Node,
                               subproblems: Subproblems,
                               var_name: str,
                               domain: SupportedVars) -> Tuple[Node, Node]:
        """
        Takes the current node, and based on the variable name passed,
        we create two child nodes where the left is fixed to 1 
        and the right is fixed to 0.

        Parameters
        ---------------
        current_node : Node
            the current node in the sBnB tree.
        subproblems: Subproblems
            object containing all initialized subproblems.
        var_name : str
            the name of the current binary variable. 
        domain : SupportedVars
            indicates if this is a constrained int (e.g., positive, neg).
            default, None (i.e., unconstrained).

        Returns
        ----------------
        left_child_node  : Node
        right_child_node : Node
        """
        assert domain in [SupportedVars.integers, SupportedVars.nonnegative_integers]

        # get the new bounds
        left_child_lb, domain_center, right_child_ub = \
                self.parition_strategy.new_bounds(var_name = var_name,
                                                  var_type = domain,
                                                  node = current_node,
                                                  subproblems = subproblems)
        
        # make sure we are branching on integers
        # most likely, domain center is continuous so we need to round
        left_child_ub  = math.floor(domain_center)
        right_child_lb = math.ceil(domain_center)
        assert left_child_lb <= left_child_ub, f"left_child_lb = {left_child_lb}, left_child_ub = {left_child_ub}"
        assert right_child_lb <= right_child_ub, f"right_child_lb = {right_child_lb}, right_child_ub = {right_child_ub}"
        assert right_child_lb >= left_child_ub, f"right_child_lb = {right_child_lb}, left_child_ub = {left_child_ub}"
        if domain == SupportedVars.nonnegative_integers:
            assert left_child_lb >= 0 # should only have to check left side LB

        # LEFT child
        left_node_name = self.metrics.nodes.node_id_max + 1 # update name
        left_node_to_branch = cp.deepcopy(current_node.to_branch)
        assert var_name in left_node_to_branch[domain]
        left_node_state = (cp.deepcopy(current_node.state))
        # if we are at a point where we can fix the integer, do so
        if (left_child_lb == left_child_ub):
            left_node_state[domain][var_name].is_fixed = True
            left_node_state[domain][var_name].value = left_child_lb
            left_node_to_branch[domain].remove(var_name)
        # otw, update the bounds
        else: 
            left_node_state[domain][var_name].ub = left_child_ub
        left_child_node = Node(to_branch = left_node_to_branch,
                               state = left_node_state,
                               id = left_node_name)
            
        # attach the LB aggregated obj + related solutions for information
        left_child_node.lb_problem.objective = current_node.lb_problem.objective

        # init the subproblem solutions for adding cuts
        left_child_node.lb_problem.subproblem_solutions = current_node.lb_problem.subproblem_solutions

        # RIGHT child
        right_node_name = self.metrics.nodes.node_id_max + 2 # update name
        right_node_to_branch = cp.deepcopy(current_node.to_branch)
        assert var_name in right_node_to_branch[domain]
        right_node_state = (cp.deepcopy(current_node.state))
        # if we are at a point where we can fix the integer, do so
        if (right_child_lb == right_child_ub):
            right_node_state[domain][var_name].is_fixed = True
            right_node_state[domain][var_name].value = right_child_lb
            right_node_to_branch[domain].remove(var_name)
        # otw, update the bounds
        else:
            right_node_state[domain][var_name].lb = right_child_lb    
        right_child_node = Node(to_branch = right_node_to_branch,
                                state = right_node_state,
                                id = right_node_name)
            
        # attach the LB aggregated obj + related solutions for information
        right_child_node.lb_problem.objective = current_node.lb_problem.objective
        right_child_node.lb_problem.subproblem_solutions = current_node.lb_problem.subproblem_solutions

        return left_child_node, right_child_node
    

    def _spawn_children(self, 
                        current_node: Node, 
                        subproblems: Subproblems,
                        var_name: str) -> Tuple[Node, Node]:
        """
        Takes the current node, and based on the variable name passed,
        determine the split point (customizable) and generate two 
        children that reflect the new domain split.

        Paramters
        ---------------
        current_node : Node
            the current node in the sBnB tree.
        var_name : str
            the name of the current binary variable. 

        Returns
        ----------------
        left_child_node  : Node
        right_child_node : Node
        """
        # get the new bounds
        left_child_lb, domain_center, right_child_ub = \
                self.parition_strategy.new_bounds(var_name = var_name,
                                                  var_type = SupportedVars.reals,
                                                  node = current_node,
                                                  subproblems = subproblems)
        
        left_child_ub  = domain_center
        right_child_lb = domain_center

        # if this variable is no longer branchable (i.e., tolerance reach) 
        #    -> make sure to remove from to_branch dict.
        stop_branching = True if abs((right_child_ub - left_child_lb) / 2)\
                            <= self.epsilon else False

        # left child
        left_node_name = self.metrics.nodes.node_id_max + 1
        left_node_state = (cp.deepcopy(current_node.state))
        left_node_to_branch = (cp.deepcopy(current_node.to_branch))
        if stop_branching: left_node_to_branch[SupportedVars.reals].remove(var_name)
        else: left_node_state[SupportedVars.reals][var_name].ub = left_child_ub
        left_child_node = Node(to_branch = left_node_to_branch,
                               state = left_node_state,
                               id = left_node_name)
        left_child_node.lb_problem.objective = current_node.lb_problem.objective
        left_child_node.lb_problem.subproblem_solutions = current_node.lb_problem.subproblem_solutions

        # right child
        right_node_name = self.metrics.nodes.node_id_max + 2
        right_node_state = (cp.deepcopy(current_node.state))
        right_node_to_branch = (cp.deepcopy(current_node.to_branch))
        if stop_branching: right_node_to_branch[SupportedVars.reals].remove(var_name)
        else: right_node_state[SupportedVars.reals][var_name].lb = right_child_lb
        right_child_node = Node(to_branch = right_node_to_branch,
                                state = right_node_state,
                                id = right_node_name)
        right_child_node.lb_problem.objective = current_node.lb_problem.objective
        right_child_node.lb_problem.subproblem_solutions = current_node.lb_problem.subproblem_solutions

        return left_child_node, right_child_node


    def converged(self,
                  current_time: float) -> bool:
        """
        Evaluate the state of the tree and determine
        if we have met the criteria for convergence.

        If we converged, prints to terminal why.

        Parameters
        ----------------
        current_time : float
            count of how long the algorithm has progressed.

        Returns
        ----------------
        converged : bool
            if we have converged or not.
        """
        # run out of nodes? gap tolerance met?
        if self.metrics.absolute_gap <= self.abs_tolerance:
            if (rank==0): print("SNoGloDe converged - absolute gap tolerance met.")
            return True
        elif self.metrics.relative_gap <= self.rel_tolerance:
            if (rank==0): print("SNoGloDe converged - relative gap tolerance met.")
            return True
        elif self.time_limit <= current_time:
            if (rank==0): print("SNoGloDe converged - timed out.")
            return True
        elif not len(self.node_queue):
            if (rank==0): print("SNoGloDe converged - no nodes left to explore.")
            return True
        # otw we have not converged
        else:
            return False
    

    def update_ub(self, 
                  ub_objective: float) -> None:
        """
        Updates the upper bound information.
        
        Paramters
        ---------------
        ub_objective : float
        """
        self.metrics.ub = ub_objective


    def update_lb(self) -> str:
        """
        Updates the lower bound information.
        Iterates over each of the open leaf nodes,
        and saves the lowest LB.
    
        Returns a string to indicate if bound has been updated.
        """
        lb = math.inf
        
        # check all open nodes
        for element in self.node_queue.__iter__():
            _, node = element
            if (node.lb_problem.objective < lb):
                lb = node.lb_problem.objective
        
        # check all nodes within the terminal leaf nodes
        for element in self.terminal_node_queue.__iter__():
            _, node = element
            if (node.lb_problem.objective < lb):
                lb = node.lb_problem.objective

        if (lb != self.metrics.lb):
            tolerance = 1e-5
            deviation = (self.metrics.lb - lb)
            if deviation > tolerance: 
                print("WARNING: LB oscillation.")
                print(f" prev LB: {self.metrics.lb}, new LB: {lb}")
            self.metrics.lb = lb
            return "lb"
        else: return ""
    

    def update_gap(self) -> None:
        """
        Takes the current values in lower_bound and upper_bound
        and recalculates relative / absolute gap.
        """
        self.metrics.absolute_gap = self.metrics.ub - self.metrics.lb
        if self.metrics.absolute_gap == 0:
            self.metrics.relative_gap = 0
        elif self.metrics.ub == 0:
            self.metrics.relative_gap = math.inf
        elif math.isfinite(self.metrics.ub) and math.isfinite(self.metrics.lb):
            self.metrics.relative_gap = self.metrics.absolute_gap / abs(self.metrics.ub)
        else:
            self.metrics.relative_gap = math.inf


    def n_nodes(self) -> int:
        """
        Returns the number of nodes in the tree
        """
        return len(self.node_queue)


class TreeNodeMetrics():
    """
    Keeps track of the nodes we have explored metrics.
    """
    def __init__(self) -> None:
        self.explored = 0
        self.pruned_by_bound = []
        self.pruned_by_infeasibility = []
        self.node_id_max = 0


class TreeMetrics():
    """
    Keeps track of all metrics.
    """
    def __init__(self) -> None:
        self.nodes = TreeNodeMetrics()
        self.lb = float("-inf")
        self.ub = float("inf")
        self.relative_gap = float("inf")
        self.absolute_gap = float("inf")