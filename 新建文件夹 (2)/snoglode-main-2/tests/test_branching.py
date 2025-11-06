"""
Test different aspects of the branching strategies

Key for ensuring we are not violating the important aspects of the sBnB tree
(or in some cases just a normal BnB tree...)
"""
import pytest as pytest
import numpy as np
import snoglode as sno
import snoglode.utils.compute as compute
import pyomo.environ as pyo

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

try: from problems import IntegerProgram, MockCandidateGenerator, \
        pmedian_subproblem_creator, continuous_1var_subproblem_creator, farmer_classic_subproblem_creator, integer_knapsack_subproblem_creator
except: from .problems import IntegerProgram, MockCandidateGenerator, \
        pmedian_subproblem_creator, continuous_1var_subproblem_creator, farmer_classic_subproblem_creator, integer_knapsack_subproblem_creator

gurobi = pyo.SolverFactory("gurobi")

@pytest.mark.mpi_skip()
def test_binary_branching_lifo():
    """
    If we have binary branching, check that each node is generated properly.
    
    For LIFO, SNoGloDe should generate two children where the first child
    has the first binary fixed to 0, and the second child will have the same variable
    fixed to 1.
    """
    # reset seed 
    np.random.seed(2)

    ip = IntegerProgram()
    params = sno.SolverParameters(subproblem_names = ["dummy"],
                                  subproblem_creator = ip.ip_subproblem_creator,
                                  cg_solver = gurobi)
    params.set_bounders(candidate_solution_finder = MockCandidateGenerator)
    params.set_queue_strategy(sno.QueueStrategy.lifo)
    solver = sno.Solver(params)
    
    # just solve the root note
    solver.solve(max_iter = 0)

    # check that we have generated exactly 2 nodes
    assert solver.tree.n_nodes() == 2

    # the left node should have generated a child with the first value fixed to 0
    left_child = solver.tree.get_node()
    assert left_child.state[sno.SupportedVars.binary][("y",1)].is_fixed == True
    assert left_child.state[sno.SupportedVars.binary][("y",1)].value == 1
    assert left_child.state[sno.SupportedVars.binary][("y",2)].is_fixed == False
    assert left_child.state[sno.SupportedVars.binary][("y",3)].is_fixed == False

    # the right node should have generated a child with one value fixed to 1
    right_child = solver.tree.get_node()
    assert right_child.state[sno.SupportedVars.binary][("y",1)].is_fixed == True
    assert right_child.state[sno.SupportedVars.binary][("y",1)].value == 0
    assert right_child.state[sno.SupportedVars.binary][("y",2)].is_fixed == False
    assert right_child.state[sno.SupportedVars.binary][("y",3)].is_fixed == False


@pytest.mark.mpi_skip()
def test_binary_branching_fifo():
    """
    If we have binary branching, check that each node is generated properly.

    For FIFO, SNoGloDe should generate two children where the first child
    has the first binary fixed to 1, and the second child will have the same variable
    fixed to 0.
    """
    # reset seed 
    np.random.seed(2)

    ip = IntegerProgram()
    params = sno.SolverParameters(subproblem_names = ["dummy"],
                                  subproblem_creator = ip.ip_subproblem_creator,
                                  cg_solver = gurobi)
    params.set_bounders(candidate_solution_finder = MockCandidateGenerator)
    params.set_queue_strategy(sno.QueueStrategy.fifo)
    solver = sno.Solver(params)

    # just solve the root note
    solver.solve(max_iter = 0)

    # check that we have generated exactly 2 nodes
    assert solver.tree.n_nodes() == 2

    # the left node should have generated a child with the first value fixed to 0
    left_child = solver.tree.get_node()
    assert left_child.state[sno.SupportedVars.binary][("y",1)].is_fixed == True
    assert left_child.state[sno.SupportedVars.binary][("y",1)].value == 0
    assert left_child.state[sno.SupportedVars.binary][("y",2)].is_fixed == False
    assert left_child.state[sno.SupportedVars.binary][("y",3)].is_fixed == False

    # the right node should have generated a child with one value fixed to 1
    right_child = solver.tree.get_node()
    assert right_child.state[sno.SupportedVars.binary][("y",1)].is_fixed == True
    assert right_child.state[sno.SupportedVars.binary][("y",1)].value == 1
    assert right_child.state[sno.SupportedVars.binary][("y",2)].is_fixed == False
    assert right_child.state[sno.SupportedVars.binary][("y",3)].is_fixed == False


@pytest.mark.mpi_skip()
def test_binary_branching_bound():
    """
    If we have binary branching, check that each node is generated properly.

    For FIFO, SNoGloDe should generate two children where the first child
    has the first binary fixed to 1, and the second child will have the same variable
    fixed to 0.
    """
    # reset seed 
    np.random.seed(2)

    ip = IntegerProgram()
    params = sno.SolverParameters(subproblem_names = ["dummy"],
                                  subproblem_creator = ip.ip_subproblem_creator,
                                  cg_solver = gurobi)
    params.set_bounders(candidate_solution_finder = MockCandidateGenerator)
    params.set_queue_strategy(sno.QueueStrategy.bound)
    solver = sno.Solver(params)
    
    # just solve the root note
    solver.solve(max_iter = 0)

    # check that we have generated exactly 2 nodes
    assert solver.tree.n_nodes() == 2

    # the left node should have generated a child with the first value fixed to 0
    left_child = solver.tree.get_node()
    assert left_child.state[sno.SupportedVars.binary][("y",1)].is_fixed == True
    assert left_child.state[sno.SupportedVars.binary][("y",2)].is_fixed == False
    assert left_child.state[sno.SupportedVars.binary][("y",3)].is_fixed == False
    
    # the right node should have generated a child with one value fixed to 1
    right_child = solver.tree.get_node()
    assert right_child.state[sno.SupportedVars.binary][("y",1)].is_fixed == True
    assert right_child.state[sno.SupportedVars.binary][("y",2)].is_fixed == False
    assert right_child.state[sno.SupportedVars.binary][("y",3)].is_fixed == False

    left_child_val = left_child.state[sno.SupportedVars.binary][("y",1)].value
    right_child_val = right_child.state[sno.SupportedVars.binary][("y",1)].value
    assert (left_child_val != right_child_val)


@pytest.mark.mpi_skip()
def test_binary_tree_terminal_leaf_nodes():
    """
    In the situation where we have terminal leaf nodes, 
    ensure that we considering the terminal nodes for the LB computation.
    """
    # reset seed 
    np.random.seed(2)

    nb_facilities = 3
    max_facilities = 1
    total_communities = 10
    nb_subproblems = 3

    subproblem_names = [f"facilities_{nb_facilities}_max_{max_facilities}_communities_{total_communities}_subproblems_{nb_subproblems}_subproblem_{subproblem}" \
                            for subproblem in np.arange(nb_subproblems)]

    params = sno.SolverParameters(subproblem_names = subproblem_names,
                                  subproblem_creator = pmedian_subproblem_creator,
                                  cg_solver = gurobi)
    params.set_bounds_tightening(fbbt = False,
                                 obbt = False)
    params.activate_verbose()
    params.set_queue_strategy(sno.QueueStrategy.bound)
    params.set_bounders(candidate_solution_finder = MockCandidateGenerator)
    solver = sno.Solver(params)
    solver.solve(rel_tolerance=0, 
                 abs_tolerance=0)

    # since we don't generate an upper bound - we should have no nodes left.
    assert solver.tree.n_nodes() == 0

    # we should have reached terminal nodes.
    assert solver.tree.terminal_node_queue.__len__() == 3

    # the LB of one of the terminal nodes should be the lower bound we find.
    terminal_node = solver.tree.terminal_node_queue.pop()
    assert (terminal_node.lb_problem.objective == solver.tree.metrics.lb)
    
    # if we only update based on open nodes - final LB = ~406.66189
    assert terminal_node.lb_problem.objective == pytest.approx(383.213) 


@pytest.mark.mpi_skip()
def test_continous_branching_epsilon_tolerance():
    """
    Test that once we hit the epsilon tolerance, we stop branching
    (and ensure that we define that node as a terminal node...)
    """
    # reset seed 
    np.random.seed(2)
    params = sno.SolverParameters(subproblem_names=["dummy"],
                                  subproblem_creator=continuous_1var_subproblem_creator,
                                  cg_solver = gurobi)
    params.set_bounds_tightening(fbbt = False,
                                 obbt = False)
    params.activate_verbose()
    params.set_queue_strategy(sno.QueueStrategy.bound)
    params.set_epsilon(0.25)
    params.set_bounders(candidate_solution_finder = MockCandidateGenerator)

    # set up solver
    solver = sno.Solver(params)
    solver.solve()
    
    # since we don't generate an upper bound - we should have no nodes left.
    assert solver.tree.n_nodes() == 0

    # we should have reached all possible terminal nodes.
    assert solver.tree.terminal_node_queue.__len__() == 4

    # the LB of one of the terminal nodes should be the lower bound we find.
    terminal_node = solver.tree.terminal_node_queue.pop()
    assert (terminal_node.lb_problem.objective == solver.tree.metrics.lb)
    assert terminal_node.lb_problem.objective == 0


@pytest.mark.mpi_skip()
def test_relaxed_binary_branching():
    """
    When we choose to relax binaries, check that it will progress
    as expected.

    Compare to the case where we do not relax the binaries, and make
    sure we are getting the expected behavior.
    """
    # reset seed 
    np.random.seed(2)

    ip = IntegerProgram()
    params = sno.SolverParameters(subproblem_names=["dummy"],
                                  subproblem_creator=ip.ip_subproblem_creator)
    params.set_bounds_tightening(fbbt = False,
                                 obbt = False)
    params.activate_verbose()
    params.relax_binaries(True)
    params.set_queue_strategy(sno.QueueStrategy.bound)
    params.set_bounders(candidate_solution_finder = sno.AverageLowerBoundSolution)

    # set up solver
    solver_relaxedbinaries = sno.Solver(params)
    solver_relaxedbinaries.solve()

    # based on the example from Ignacio's textbook - this should converge after 5 nodes
    assert solver_relaxedbinaries.iteration == 5

    # should find 2 infeasible nodes
    assert len(solver_relaxedbinaries.tree.metrics.nodes.pruned_by_infeasibility) == 2
    
    # optimal objective should be 5 w/ no gap.
    assert solver_relaxedbinaries.tree.metrics.ub == 5
    assert solver_relaxedbinaries.tree.metrics.relative_gap == 0

    # check that the final solution is binary
    assert solver_relaxedbinaries.solution.subproblem_solutions["dummy"]["y[1]"] == 1.0
    assert solver_relaxedbinaries.solution.subproblem_solutions["dummy"]["y[2]"] == 0.0
    assert solver_relaxedbinaries.solution.subproblem_solutions["dummy"]["y[3]"] == 1.0

    # reset seed 
    np.random.seed(2)

    ip = IntegerProgram()
    params.relax_binaries(False)

    # set up solver
    solver_binaries = sno.Solver(params)
    solver_binaries.solve()

    # without relaxation, should converge immediately
    assert solver_binaries.iteration == 1

    # optimal objective should be 5 w/ no gap.
    assert solver_binaries.tree.metrics.ub == 5
    assert solver_binaries.tree.metrics.relative_gap == 0

    # check that the final solution is binary
    assert solver_binaries.solution.subproblem_solutions["dummy"]["y[1]"] == 1.0
    assert solver_binaries.solution.subproblem_solutions["dummy"]["y[2]"] == 0.0
    assert solver_binaries.solution.subproblem_solutions["dummy"]["y[3]"] == 1.0


@pytest.mark.mpi_skip()
def test_most_infeasible_binary_branching_strategy():
    """
    When we choose to relax binaries, check that it will progress
    as expected.

    Compare to the case where we do not relax the binaries, and make
    sure we are getting the expected behavior.
    """
    # reset seed 
    np.random.seed(2)

    ip = IntegerProgram()
    params = sno.SolverParameters(subproblem_names=["dummy"],
                                  subproblem_creator=ip.ip_subproblem_creator)
    params.set_bounds_tightening(fbbt = False,
                                 obbt = False)
    params.relax_binaries()
    params.set_queue_strategy(sno.QueueStrategy.bound)
    params.set_bounders(candidate_solution_finder = sno.AverageLowerBoundSolution)
    params.set_branching(selection_strategy = sno.MostInfeasibleBinary)

    # set up solver
    solver = sno.Solver(params)
    solver.solve()

    # based on the example from Ignacio's textbook - this should converge after 5 nodes
    assert solver.iteration == 5

    # should find 2 infeasible nodes
    assert len(solver.tree.metrics.nodes.pruned_by_infeasibility) == 2
    
    # optimal objective should be 5 w/ no gap.
    assert solver.tree.metrics.ub == 5
    assert solver.tree.metrics.relative_gap == 0

    # check that the final solution is binary
    assert solver.solution.subproblem_solutions["dummy"]["y[1]"] == 1.0
    assert solver.solution.subproblem_solutions["dummy"]["y[2]"] == 0.0
    assert solver.solution.subproblem_solutions["dummy"]["y[3]"] == 1.0


@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_maximum_disagreement_branching_strategy():
    """
    This branching strategy relies on computing the variance of the solution
    across the subproblems, and selecting the variable that has the maximum variance.
    """
    subproblem_names = ["good", "fair", "bad"]
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=farmer_classic_subproblem_creator,
                                  cg_solver = gurobi)
    params.set_bounders(candidate_solution_finder = sno.AverageLowerBoundSolution)
    params.set_bounds_tightening(fbbt = False)
    params.activate_verbose()
    params.deactivate_global_guarantee()
    params.set_queue_strategy(sno.QueueStrategy.fifo)
    params.set_branching(selection_strategy = sno.MaximumDisagreement)

    # set up solver
    solver = sno.Solver(params)

    # test that the variance function computations are working correctly
    root_node = solver.tree.get_node()
    solver.dispatch_lb_solve(root_node)
    variance = compute.variance_lb_solution(node = root_node,
                                            subproblems = solver.subproblems,
                                            normalize = False)
    true_variance = {
        "wheat": ((183.33 - 134.44)**2 + (120 - 134.44)**2 + (100 - 134.44)**2)/3,
        "corn":  ((66.66 - 57.22)**2 + (80 - 57.22)**2 + (25 - 57.22)**2)/3,
        "beets": ((250 - 308.33)**2 + (300 - 308.33)**2 + (375 - 308.33)**2)/3
    }

    # the variance has a lot of of deviation in the decimal place, so only compare to ndec 1
    for varID in variance.keys():
        _, crop = varID
        assert variance[varID] == pytest.approx(true_variance[crop], rel=1e-1)

    normalized_variance = compute.variance_lb_solution(node = root_node, 
                                                       subproblems = solver.subproblems,
                                                       normalize = True)
    
    wheat_lb = root_node.state[sno.SupportedVars.reals][("devoted_acrege", "wheat")].lb
    wheat_ub = root_node.state[sno.SupportedVars.reals][("devoted_acrege", "wheat")].ub
    corn_lb  = root_node.state[sno.SupportedVars.reals][("devoted_acrege", "corn")].lb
    corn_ub  = root_node.state[sno.SupportedVars.reals][("devoted_acrege", "corn")].ub
    beets_lb = root_node.state[sno.SupportedVars.reals][("devoted_acrege", "beets")].lb
    beets_ub = root_node.state[sno.SupportedVars.reals][("devoted_acrege", "beets")].ub
    true_normalized_variance = {
        "wheat": (((183.33 - 134.44)/(wheat_ub - wheat_lb))**2 + ((120 - 134.44)/(wheat_ub - wheat_lb))**2 + ((100 - 134.44)/(wheat_ub - wheat_lb))**2)/3,
        "corn":  (((66.66 - 57.22)/(corn_ub - corn_lb))**2 + ((80 - 57.22)/(corn_ub - corn_lb))**2 + ((25 - 57.22)/(corn_ub - corn_lb))**2)/3,
        "beets": (((250 - 308.33)/(beets_ub - beets_lb))**2 + ((300 - 308.33)/(beets_ub - beets_lb))**2 + ((375 - 308.33)/(beets_ub - beets_lb))**2)/3
    }

    # the variance has a lot of of deviation in the decimal place, so only compare to ndec 4 (since these values are between 0,1)
    for varID in normalized_variance.keys():
        _, crop = varID
        assert normalized_variance[varID] == pytest.approx(true_normalized_variance[crop], rel=1e-4)

    # reset solver
    solver = sno.Solver(params)
    solver.solve(max_iter = 0)

    # check that we have generated exactly 2 nodes
    assert solver.tree.n_nodes() == 2

    # we should have selected BEETS as the branching variable
    # the left node should have generated a child where the UB of the beets var is changed.
    left_child = solver.tree.get_node()
    assert left_child.state[sno.SupportedVars.reals][("devoted_acrege", "wheat")].lb == wheat_lb
    assert left_child.state[sno.SupportedVars.reals][("devoted_acrege", "wheat")].ub == wheat_ub
    assert left_child.state[sno.SupportedVars.reals][("devoted_acrege", "corn")].lb == corn_lb
    assert left_child.state[sno.SupportedVars.reals][("devoted_acrege", "corn")].ub == corn_ub
    assert left_child.state[sno.SupportedVars.reals][("devoted_acrege", "beets")].lb == beets_lb
    assert left_child.state[sno.SupportedVars.reals][("devoted_acrege", "beets")].ub != beets_ub

    # the right node should have generated a child where the LB of the beets var is changed.
    right_child = solver.tree.get_node()
    assert right_child.state[sno.SupportedVars.reals][("devoted_acrege", "wheat")].lb == wheat_lb
    assert right_child.state[sno.SupportedVars.reals][("devoted_acrege", "wheat")].ub == wheat_ub
    assert right_child.state[sno.SupportedVars.reals][("devoted_acrege", "corn")].lb == corn_lb
    assert right_child.state[sno.SupportedVars.reals][("devoted_acrege", "corn")].ub == corn_ub
    assert right_child.state[sno.SupportedVars.reals][("devoted_acrege", "beets")].lb != beets_lb
    assert right_child.state[sno.SupportedVars.reals][("devoted_acrege", "beets")].ub == beets_ub


@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_pseudocost_branching_strategy():
    """
    The pseudocost branching strategy depends on the impact
    in the larger tree of branching on one variable over another.

    TODO: write a traceable, provable computation
    for now, satisfied with ensuring we have the expected improved performance
    """
    # reset seed 
    np.random.seed(17)

    subproblem_names = ["good", "fair", "bad"]
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=farmer_classic_subproblem_creator,
                                  cg_solver = gurobi)
    params.set_bounders(candidate_solution_finder = sno.AverageLowerBoundSolution)
    params.set_bounds_tightening(fbbt = False)
    params.activate_verbose()
    params.deactivate_global_guarantee()
    params.set_queue_strategy(sno.QueueStrategy.fifo)
    params.set_branching(selection_strategy = sno.Pseudocost)

    # set up solver
    solver = sno.Solver(params)
    solver.solve(max_iter = 100)

    # with random selection, we only reach ~6.5% gap whereas pseudocost is around 3.5%
    assert solver.tree.metrics.relative_gap <= 0.038

@pytest.mark.mpi_skip()
def test_integer_branching():
    """
    Test that integer branching works as expected.
    Just tests that integer logic can be followed, not 
    testing a specific branching strategy.

    Using the integer capacitating knapsack problem.
    """
    params = sno.SolverParameters(subproblem_names = ["dummy"],
                                subproblem_creator = integer_knapsack_subproblem_creator,
                                lb_solver = pyo.SolverFactory("gurobi"),
                                cg_solver = pyo.SolverFactory("gurobi"),
                                ub_solver = pyo.SolverFactory("gurobi"))
    params.set_bounds_tightening(fbbt = False, obbt=False)
    params.set_bounders(candidate_solution_finder = sno.AverageLowerBoundSolution,
                        lower_bounder = sno.DropNonants)
    params.set_branching(selection_strategy = sno.RandomSelection,
                         partition_strategy = sno.Midpoint)
    params.relax_integers()
    params.set_queue_strategy(sno.QueueStrategy.bound)
    solver = sno.Solver(params)
    solver.solve()

    assert solver.tree.metrics.relative_gap == 0
    assert solver.tree.metrics.ub == -10
    assert solver.runtime <= 1

    # check that the final solution is integer
    assert solver.solution.subproblem_solutions["dummy"]["x[1]"] == 2.0
    assert solver.solution.subproblem_solutions["dummy"]["x[2]"] == 0.0


if __name__=="__main__":

    # test_binary_branching_lifo()
    # test_binary_branching_fifo()
    # test_binary_branching_bound()
    # test_binary_tree_terminal_leaf_nodes()
    # test_continous_branching_epsilon_tolerance()
    # test_relaxed_binary_branching()
    # test_most_infeasible_branching_strategy()
    # test_maximum_disagreement_branching_strategy()
    # test_pseudocost_branching_strategy()
    test_integer_branching()