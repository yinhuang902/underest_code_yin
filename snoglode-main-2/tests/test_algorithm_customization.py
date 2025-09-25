"""
Various tests to check on the Node object.
"""
import pytest as pytest
import numpy as np
import snoglode as sno
import pyomo.environ as pyo

try: from problems import pmedian_subproblem_creator, MockCandidateGenerator
except: from .problems import pmedian_subproblem_creator, MockCandidateGenerator

lb_solver = pyo.SolverFactory("gurobi")
cg_solver = pyo.SolverFactory("gurobi")
ub_solver = pyo.SolverFactory("gurobi")

@pytest.mark.mpi_skip()
def test_customizable_node_feasibility_checker():
    """
    check that the simple feasibility checker is functioning properly.

        Want to see why this customization is helpful??
        
            Try running the algorithm with / without the custom node checker.
            The log will be identical - the run times, however, will not.

            For this problem, nb_facilities directly defines the number of vars in the binary tree.
            As nb_facilities is increased, there is a clear speed up when we use
            the custom feasibility checker vs without 
            (with the same algorithmic performance log... :) )
    """
    nb_facilities = 4
    max_facilities = 1
    total_communities = 10
    nb_subproblems = 2

    subproblem_names = [f"facilities_{nb_facilities}_max_{max_facilities}_communities_{total_communities}_subproblems_{nb_subproblems}_subproblem_{subproblem}" \
                            for subproblem in np.arange(nb_subproblems)]
    
    def node_feasibility_check(node: sno.Node) -> bool:
        # if we have more than max_facilities fixed to 1, then this node 
        # and all subsequent children must be infeasible as well.
        
        nb_facilities_fixed_to_1 = 0 
        for var in node.state[sno.SupportedVars.binary]:
            if node.state[sno.SupportedVars.binary][var].is_fixed \
                        and node.state[sno.SupportedVars.binary][var].value == 1:
                nb_facilities_fixed_to_1 += 1

        if (nb_facilities_fixed_to_1 > 1): 
            return False
        
        else: return True

    # set up solver
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=pmedian_subproblem_creator,
                                  cg_solver = pyo.SolverFactory("gurobi"))
    params.set_bounds_tightening(fbbt = False,
                                 obbt = False)
    params.activate_verbose()
    params.deactivate_global_guarantee()
    params.add_node_feasibility_checker(node_feasibility_check)
    params.set_bounders(candidate_solution_finder = MockCandidateGenerator)
    params.set_queue_strategy(sno.QueueStrategy.lifo)
    
    # solve with given tolerances
    solver = sno.Solver(params)
    solver.solve(max_iter=100,
                 rel_tolerance=1e-2,
                 abs_tolerance=1e-3,
                 collect_plot_info=False)

    assert solver.iteration == 21      # should explore 21 nodes total
    assert solver.runtime <= 3         # should be around 1.8 seconds

    assert solver.tree.metrics.lb == pytest.approx(383.213)

if __name__=="__main__":
    test_customizable_node_feasibility_checker()