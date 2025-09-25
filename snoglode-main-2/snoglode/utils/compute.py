"""
These are methods than can help inform information
on branching / candidate generator steps. They take in,
for the most part, the node and the subproblems, and 
pass back information about the solution currently stored.
"""
from typing import Tuple, Optional
from snoglode.components.subproblems import Subproblems
from snoglode.components.node import Node
from snoglode.utils.supported import SupportedVars

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

def average_lb_solution(node: Node,
                        subproblems: Subproblems,
                        round_binaries: bool = True,
                        round_integers: bool = True,
                        normalize: bool = False,
                        return_frequencies: bool = False) -> Tuple[dict, Optional[dict]]:
    """
    Given a node, collect the LB solutions and average.

    Method: 
        (1) get lifted var solutions from each LB subproblem solution.
        (2) average 

    NOTE: this method will always return a "feasible" candidate - i.e.,
            as long as we have a solution to the LB subproblems,
            we can generate a candidate solution and return it.
            ** as long as the round_binaries flag is enabled
    
    NOTE: the candidate solution objective will be None, because we do not
            solve an optimization in this case.

    NOTE: in the context of booleans, we round:
        - if > 0.5  -> 1
        - if <= 0.5 -> 0

    Parameters
    ----------
    node : Node
        node object representing the current node we are exploring in the branch 
        and bound tree. Contains all bounding information.
    subproblems : Subproblems
        initialized subproblem manager.
        contains all subproblem names, models, probabilities, and lifted var lists/
    round_binaries : bool, optional
        this ensures a feasible solution from the point of view of satisfaction
        of integrality constraints
    normalize : bool, optional
        when computing the averagee of the continuously bounded variables,
        normalize their solutions.
    return_frequencies: bool, optional
        optional return flag to also return the frequency of each variable.

    Returns
    ----------
    average_solution : dict
        each lifted_var_id is a key, each corresponding value is the average
        of that variable across all subproblems.
    frequency_of_lifted_vars : dict, optional
        the number of times each variable was present in each subproblem
    """
    # must have a feasible LB solution (though UB problem shouldn't be called if that is the case...)
    assert node.lb_problem.feasible

    # running sum / frequency
    # these **MUST** be in the same order across ranks!!!
    lifted_vars = sorted(subproblems.lifted_var_ids)
    aggregated_lifted_vars = {varID:0 for varID in lifted_vars}
    frequency_of_lifted_vars = {varID:0 for varID in lifted_vars}

    # access the solutions for each subproblem to add to sums / freqencies
    for subproblem_name in subproblems.names:

        # go through each of the lifted var IDs
        for var in subproblems.subproblem_lifted_vars[subproblem_name]:

            # extract varID
            var_type, varID, _ = subproblems.var_to_data[var]

            # if we have this variable in the subproblem, can compute average otw pass
            if varID in node.lb_problem.subproblem_solutions[subproblem_name].lifted_var_solution[var_type]:
                
                # retrieve solution
                LB_solution = node.lb_problem.subproblem_solutions[subproblem_name].lifted_var_solution[var_type][varID]

                # normalize, if needed, and add to running sum
                if normalize and var_type != SupportedVars.binary:
                    var_lb = node.state[var_type][varID].lb
                    var_ub = node.state[var_type][varID].ub
                    normalized_LB_solution = (LB_solution - var_lb) / (var_ub - var_lb)
                    aggregated_lifted_vars[varID] += normalized_LB_solution
                else:
                    aggregated_lifted_vars[varID] += LB_solution

                # aggregate frequency
                frequency_of_lifted_vars[varID] += 1
    
    # make sure all the ranks catch up first
    MPI.COMM_WORLD.barrier()

    # aggregate all of the information across ranks
    candidate_solution_state = {varID: None for varID in lifted_vars}
    for varID in lifted_vars:
        aggregated_lifted_vars[varID] = MPI.COMM_WORLD.allreduce(aggregated_lifted_vars[varID], op=MPI.SUM)
        frequency_of_lifted_vars[varID] = MPI.COMM_WORLD.allreduce(frequency_of_lifted_vars[varID], op=MPI.SUM)
        average_lifted_var = aggregated_lifted_vars[varID] / frequency_of_lifted_vars[varID]

        # round if this is a binary variable
        if varID in node.state[SupportedVars.binary] and round_binaries:
            average_lifted_var = round(average_lifted_var, ndigits=0)
            assert (average_lifted_var >= 0 and average_lifted_var <= 1)
        
        # round if this is an integer variable
        if (varID in node.state[SupportedVars.integers] or \
            varID in node.state[SupportedVars.nonnegative_integers]) and round_integers:
            average_lifted_var = round(average_lifted_var, ndigits=0)
            if varID in node.state[SupportedVars.nonnegative_integers]: 
                assert average_lifted_var >= 0
        
        # grab value, save under var_id
        candidate_solution_state[varID] = average_lifted_var
    
    if return_frequencies: 
        return candidate_solution_state, frequency_of_lifted_vars
    else:
        return candidate_solution_state


def average_var_lb_solution(node: Node,
                            subproblems: Subproblems,
                            var_ID: str) -> float:
    """
    Given a node & specific variable, collect the LB solutions and averages
    across the solutions of *all* subproblems

    Method: 
        (1) get lifted var solution from each LB subproblem solution.
        (2) average 

    Parameters
    ----------
    node : Node
        node object representing the current node we are exploring in the branch 
        and bound tree. Contains all bounding information.
    subproblems : Subproblems
        initialized subproblem manager.
        contains all subproblem names, models, probabilities, and lifted var lists/
    var_ID : str
        string corresponding to the lifted variable ID we want the averge of.

    Returns
    ----------
    avg_var_value : float
        averge of this variable across all of the subproblems
    """

    # must have a feasible LB solution (though UB problem shouldn't be called if that is the case...)
    assert node.lb_problem.feasible

    # determine the var type
    for var_type in SupportedVars:
        if var_ID in node.state[var_type]: break

    # running sum / frequency
    aggregated_lifted_var = 0
    frequency_of_lifted_var = 0

    # access the solutions for each subproblem to add to sums / freqencies
    for subproblem_name in subproblems.names:

        # if we have this variable in the subproblem, can compute average otw pass
        if var_ID in node.lb_problem.subproblem_solutions[subproblem_name].lifted_var_solution[var_type]:
            
            # retriew solution
            LB_solution = node.lb_problem.subproblem_solutions[subproblem_name].lifted_var_solution[var_type][var_ID]

            # add to running sum / frequency
            aggregated_lifted_var   += LB_solution
            frequency_of_lifted_var += 1
    
    # make sure all the ranks catch up first
    MPI.COMM_WORLD.barrier()

    # compute averages
    # if we have this variable in the subproblem, can compute average otw pass
    if var_ID in node.lb_problem.subproblem_solutions[subproblem_name].lifted_var_solution[var_type]:

        # grab the first stage var ref
        aggregated_lifted_var = MPI.COMM_WORLD.allreduce(aggregated_lifted_var, op=MPI.SUM)
        frequency_of_lifted_var = MPI.COMM_WORLD.allreduce(frequency_of_lifted_var, op=MPI.SUM)                            

        average_lifted_var = aggregated_lifted_var / frequency_of_lifted_var
        return average_lifted_var
    
    # otw return none to indicate not present
    else: return None


def variance_lb_solution(node: Node,
                         subproblems: Subproblems,
                         normalize: bool = True) -> dict:
    """
    Given a node, collect the LB solutions and find the variance across
    subproblem solutions for each of the first stage variables.

    Method: 
        (1) get lifted var solutions from each LB subproblem solution.
        (2) normalize solution, if indicated
        (3) compute the variance across the solutions 

    NOTE: In the case of continuous domains, we normalize by default such that we can
    easily compare the variance against binary domains as well.

    We define the variance as:

        var = 1/frequency * (sum(x_i - avg(x)) ^ 2 for i in subproblems containing x)
    
    where
        var:        variance
        frequency:  number of subproblem x appears in
        x_i:        solution value of x in subproblem i containing x
        avg(x):     average solution of x across all subproblems containing x

    We first compute the sum for the subproblems containing x on this rank,
    and then we aggregated using MPI.allreduce sum operation.

    Parameters
    ----------
    node : Node
        node object representing the current node we are exploring in the branch 
        and bound tree. Contains all bounding information.
    subproblems : Subproblems
        initialized subproblem manager.
        contains all subproblem names, models, probabilities, and lifted var lists/
    normalize : bool
        when computing the variance of the continuously bounded variables,
        normalize such that they can be validly compared to the variance of binary vars.

    Returns
    ----------
    variance : dict
        each lifted_var_id is a key, each corresponding value is the variance
        of that variable across all subproblems.
    """
    # must have a feasible LB solution (though UB problem shouldn't be called if that is the case...)
    assert node.lb_problem.feasible

    # running sum / frequency
    # these **MUST** be in the same order across ranks!!!
    lifted_vars = sorted(subproblems.lifted_var_ids)
    average, frequency \
            = average_lb_solution(node = node, 
                                  subproblems = subproblems,
                                  round_binaries = False,
                                  return_frequencies = True)
    variance = {varID:0 for varID in lifted_vars}
    
    # access the solutions for each subproblem to add to sums / freqencies
    for subproblem_name in subproblems.names:

        # go through each of the lifted var IDs
        for var in subproblems.subproblem_lifted_vars[subproblem_name]:

            # extract varID
            var_type, varID, _ = subproblems.var_to_data[var]

            # if we have this variable in the subproblem, can compute average otw pass
            if varID in node.lb_problem.subproblem_solutions[subproblem_name].lifted_var_solution[var_type]:

                # retriew solution
                x_i = node.lb_problem.subproblem_solutions[subproblem_name].lifted_var_solution[var_type][varID]

                # normalize, if needed, and add to running sum
                if normalize and var_type != SupportedVars.binary:
                    var_lb = node.state[var_type][varID].lb
                    var_ub = node.state[var_type][varID].ub
                    variance[varID] += ((x_i - average[varID])/(var_ub - var_lb))**2

                # if we are normalizing, update x_i
                else:
                    variance[varID] += (x_i - average[varID])**2
    
    MPI.COMM_WORLD.barrier()

    # aggregate all of the information across ranks & compute final variance
    for varID in lifted_vars:
        variance[varID] = MPI.COMM_WORLD.allreduce(variance[varID], op=MPI.SUM) / frequency[varID]

    return variance


def median_lb_solution(node: Node,
                       subproblems: Subproblems) -> dict:
    """
    Given a node, collect the LB solutions and find the median
    for each of the lifted variables.

    Method: 
        (1) gather lifted var solutions from each LB subproblem solution to rank 0.
        (2) return the median of each lifted var.
    """
    raise NotImplementedError