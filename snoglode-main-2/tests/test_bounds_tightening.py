"""
Testing bounds tightening capabilities.
"""
import pytest as pytest
import snoglode as sno
import pyomo.environ as pyo
import numpy as np

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

try: from .problems import farmer_classic_subproblem_creator, pmedian_subproblem_creator
except: from problems import farmer_classic_subproblem_creator, pmedian_subproblem_creator

gurobi = pyo.SolverFactory("gurobi")

@pytest.mark.mpi_skip()
def test_farmer_classic_ef_bounds_tightening():
    """
    Here, we solve using the EF and following specifications
    with FBBT and without, asserting everything remains the same.
    The only thing that can be variable here should be the time of the algorithm progression.

    - Test problem          : Farmer, classic (all x's are first stage, y and w are second.)
    - Lower Bounder         : default, dropping non-anticipativity
    - Upper Bounder         : solve the extensive form (to local optimality)
    - Queuing Method        : LIFO
    """

    subproblem_names = ["good", "fair", "bad"]
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=farmer_classic_subproblem_creator,
                                  cg_solver = gurobi)
    params.set_bounders(candidate_solution_finder = sno.SolveExtensiveForm)
    params.set_bounds_tightening(fbbt = False,
                                 obbt = False)
    params.activate_verbose()
    params.deactivate_global_guarantee()
    params.set_queue_strategy(sno.QueueStrategy.bound)

    # solve without FBBT
    np.random.seed(42)      # reset seed
    snoglode_no_tightening = sno.Solver(params)
    
    # solve with given tolerances
    snoglode_no_tightening.solve(max_iter=10,
                                rel_tolerance=1e-2,
                                abs_tolerance=1e-3,
                                collect_plot_info=False)
    
    # solve with FBBT
    np.random.seed(42)      # reset seed
    params.set_bounds_tightening(fbbt = True,
                                 obbt = False)
    snoglode_fbbt = sno.Solver(params)
    
    # solve with given tolerances
    snoglode_fbbt.solve(max_iter=10,
                        rel_tolerance=1e-2,
                        abs_tolerance=1e-3,
                        collect_plot_info=False)
    
    # solve with OBBT
    np.random.seed(42)      # reset seed
    params.set_bounds_tightening(fbbt = False,
                                 obbt = True)
    snoglode_obbt = sno.Solver(params)
    
    # solve with given tolerances
    snoglode_obbt.solve(max_iter=10,
                        rel_tolerance=1e-2,
                        abs_tolerance=1e-3,
                        collect_plot_info=False)
    
    # solve with FBBT and OBBT
    np.random.seed(42)      # reset seed
    params.set_bounds_tightening(fbbt = True,
                                 obbt = True)
    snoglode_fbbt_obbt = sno.Solver(params)
    
    # solve with given tolerances
    snoglode_fbbt_obbt.solve(max_iter=10,
                            rel_tolerance=1e-2,
                            abs_tolerance=1e-3,
                            collect_plot_info=False)

    # expect the exact number of iterations / final gap
    assert snoglode_no_tightening.iteration == snoglode_fbbt.iteration      
    assert snoglode_no_tightening.iteration == snoglode_fbbt.iteration      
    assert snoglode_no_tightening.iteration == snoglode_fbbt_obbt.iteration      
    assert snoglode_no_tightening.solution.relative_gap == snoglode_fbbt.solution.relative_gap
    assert snoglode_no_tightening.solution.relative_gap == snoglode_obbt.solution.relative_gap
    assert snoglode_no_tightening.solution.relative_gap == snoglode_fbbt_obbt.solution.relative_gap

    # FBBT CHECK
    # check first stage solutions - all the same + right values
    for name in snoglode_no_tightening.subproblems.names:

        assert snoglode_no_tightening.solution.subproblem_solutions[name][f"{name}.x[wheat]"] == pytest.approx(170)
        assert snoglode_no_tightening.solution.subproblem_solutions[name][f"{name}.x[corn]"] == pytest.approx(80)
        assert snoglode_fbbt.solution.subproblem_solutions[name][f"{name}.x[beets]"] == pytest.approx(250)

        assert snoglode_fbbt.solution.subproblem_solutions[name][f"{name}.x[wheat]"] == pytest.approx(170)
        assert snoglode_fbbt.solution.subproblem_solutions[name][f"{name}.x[corn]"] == pytest.approx(80)
        assert snoglode_fbbt.solution.subproblem_solutions[name][f"{name}.x[beets]"] == pytest.approx(250)
    
    # check second stage solutions
    assert snoglode_no_tightening.solution.subproblem_solutions["good"]["good.w[wheat]"] == pytest.approx(310)
    assert snoglode_no_tightening.solution.subproblem_solutions["good"]["good.w[corn]"] == pytest.approx(48)
    assert snoglode_no_tightening.solution.subproblem_solutions["good"]["good.w[beets_favorable]"] == pytest.approx(6000)
    assert snoglode_no_tightening.solution.subproblem_solutions["good"]["good.w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_no_tightening.solution.subproblem_solutions["good"]["good.y[wheat]"] == pytest.approx(0)
    assert snoglode_no_tightening.solution.subproblem_solutions["good"]["good.y[corn]"] == pytest.approx(0)

    assert snoglode_fbbt.solution.subproblem_solutions["good"]["good.w[wheat]"] == pytest.approx(310)
    assert snoglode_fbbt.solution.subproblem_solutions["good"]["good.w[corn]"] == pytest.approx(48)
    assert snoglode_fbbt.solution.subproblem_solutions["good"]["good.w[beets_favorable]"] == pytest.approx(6000)
    assert snoglode_fbbt.solution.subproblem_solutions["good"]["good.w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["good"]["good.y[wheat]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["good"]["good.y[corn]"] == pytest.approx(0)

    assert snoglode_no_tightening.solution.subproblem_solutions["fair"]["fair.w[wheat]"] == pytest.approx(225)
    assert snoglode_no_tightening.solution.subproblem_solutions["fair"]["fair.w[corn]"] == pytest.approx(0)
    assert snoglode_no_tightening.solution.subproblem_solutions["fair"]["fair.w[beets_favorable]"] == pytest.approx(5000)
    assert snoglode_no_tightening.solution.subproblem_solutions["fair"]["fair.w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_no_tightening.solution.subproblem_solutions["fair"]["fair.y[wheat]"] == pytest.approx(0)
    assert snoglode_no_tightening.solution.subproblem_solutions["fair"]["fair.y[corn]"] == pytest.approx(0)

    assert snoglode_fbbt.solution.subproblem_solutions["fair"]["fair.w[wheat]"] == pytest.approx(225)
    assert snoglode_fbbt.solution.subproblem_solutions["fair"]["fair.w[corn]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["fair"]["fair.w[beets_favorable]"] == pytest.approx(5000)
    assert snoglode_fbbt.solution.subproblem_solutions["fair"]["fair.w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["fair"]["fair.y[wheat]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["fair"]["fair.y[corn]"] == pytest.approx(0)

    assert snoglode_no_tightening.solution.subproblem_solutions["bad"]["bad.w[wheat]"] == pytest.approx(140)
    assert snoglode_no_tightening.solution.subproblem_solutions["bad"]["bad.w[corn]"] == pytest.approx(0)
    assert snoglode_no_tightening.solution.subproblem_solutions["bad"]["bad.w[beets_favorable]"] == pytest.approx(4000)
    assert snoglode_no_tightening.solution.subproblem_solutions["bad"]["bad.w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_no_tightening.solution.subproblem_solutions["bad"]["bad.y[wheat]"] == pytest.approx(0)
    assert snoglode_no_tightening.solution.subproblem_solutions["bad"]["bad.y[corn]"] == pytest.approx(48)

    assert snoglode_fbbt.solution.subproblem_solutions["bad"]["bad.w[wheat]"] == pytest.approx(140)
    assert snoglode_fbbt.solution.subproblem_solutions["bad"]["bad.w[corn]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["bad"]["bad.w[beets_favorable]"] == pytest.approx(4000)
    assert snoglode_fbbt.solution.subproblem_solutions["bad"]["bad.w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["bad"]["bad.y[wheat]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["bad"]["bad.y[corn]"] == pytest.approx(48)

    # OBBT CHECK
    # check first stage solutions - all the same + right values
    for name in snoglode_no_tightening.subproblems.names:

        assert snoglode_obbt.solution.subproblem_solutions[name][f"{name}.x[beets]"] == pytest.approx(250)

        assert snoglode_obbt.solution.subproblem_solutions[name][f"{name}.x[wheat]"] == pytest.approx(170)
        assert snoglode_obbt.solution.subproblem_solutions[name][f"{name}.x[corn]"] == pytest.approx(80)
        assert snoglode_obbt.solution.subproblem_solutions[name][f"{name}.x[beets]"] == pytest.approx(250)
    
    # check second stage solutions

    assert snoglode_obbt.solution.subproblem_solutions["good"]["good.w[wheat]"] == pytest.approx(310)
    assert snoglode_obbt.solution.subproblem_solutions["good"]["good.w[corn]"] == pytest.approx(48)
    assert snoglode_obbt.solution.subproblem_solutions["good"]["good.w[beets_favorable]"] == pytest.approx(6000)
    assert snoglode_obbt.solution.subproblem_solutions["good"]["good.w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_obbt.solution.subproblem_solutions["good"]["good.y[wheat]"] == pytest.approx(0)
    assert snoglode_obbt.solution.subproblem_solutions["good"]["good.y[corn]"] == pytest.approx(0)

    assert snoglode_obbt.solution.subproblem_solutions["fair"]["fair.w[wheat]"] == pytest.approx(225)
    assert snoglode_obbt.solution.subproblem_solutions["fair"]["fair.w[corn]"] == pytest.approx(0)
    assert snoglode_obbt.solution.subproblem_solutions["fair"]["fair.w[beets_favorable]"] == pytest.approx(5000)
    assert snoglode_obbt.solution.subproblem_solutions["fair"]["fair.w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_obbt.solution.subproblem_solutions["fair"]["fair.y[wheat]"] == pytest.approx(0)
    assert snoglode_obbt.solution.subproblem_solutions["fair"]["fair.y[corn]"] == pytest.approx(0)

    assert snoglode_obbt.solution.subproblem_solutions["bad"]["bad.w[wheat]"] == pytest.approx(140)
    assert snoglode_obbt.solution.subproblem_solutions["bad"]["bad.w[corn]"] == pytest.approx(0)
    assert snoglode_obbt.solution.subproblem_solutions["bad"]["bad.w[beets_favorable]"] == pytest.approx(4000)
    assert snoglode_obbt.solution.subproblem_solutions["bad"]["bad.w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_obbt.solution.subproblem_solutions["bad"]["bad.y[wheat]"] == pytest.approx(0)
    assert snoglode_obbt.solution.subproblem_solutions["bad"]["bad.y[corn]"] == pytest.approx(48)

    # FBBT+OBBT CHECK
    # check first stage solutions - all the same + right values
    for name in snoglode_no_tightening.subproblems.names:

        assert snoglode_obbt.solution.subproblem_solutions[name][f"{name}.x[beets]"] == pytest.approx(250)

        assert snoglode_obbt.solution.subproblem_solutions[name][f"{name}.x[wheat]"] == pytest.approx(170)
        assert snoglode_obbt.solution.subproblem_solutions[name][f"{name}.x[corn]"] == pytest.approx(80)
        assert snoglode_obbt.solution.subproblem_solutions[name][f"{name}.x[beets]"] == pytest.approx(250)
    
    # check second stage solutions

    assert snoglode_fbbt_obbt.solution.subproblem_solutions["good"]["good.w[wheat]"] == pytest.approx(310)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["good"]["good.w[corn]"] == pytest.approx(48)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["good"]["good.w[beets_favorable]"] == pytest.approx(6000)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["good"]["good.w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["good"]["good.y[wheat]"] == pytest.approx(0)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["good"]["good.y[corn]"] == pytest.approx(0)

    assert snoglode_fbbt_obbt.solution.subproblem_solutions["fair"]["fair.w[wheat]"] == pytest.approx(225)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["fair"]["fair.w[corn]"] == pytest.approx(0)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["fair"]["fair.w[beets_favorable]"] == pytest.approx(5000)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["fair"]["fair.w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["fair"]["fair.y[wheat]"] == pytest.approx(0)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["fair"]["fair.y[corn]"] == pytest.approx(0)

    assert snoglode_fbbt_obbt.solution.subproblem_solutions["bad"]["bad.w[wheat]"] == pytest.approx(140)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["bad"]["bad.w[corn]"] == pytest.approx(0)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["bad"]["bad.w[beets_favorable]"] == pytest.approx(4000)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["bad"]["bad.w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["bad"]["bad.y[wheat]"] == pytest.approx(0)
    assert snoglode_fbbt_obbt.solution.subproblem_solutions["bad"]["bad.y[corn]"] == pytest.approx(48)


@pytest.mark.mpi_skip()
def test_farmer_classic_average_solution_bounds_tightening_fbbt():
    """
    Here, we solve using the averaging of the lower bound solution and following specifications
    with FBBT and without, asserting everything remains the same.
    The only thing that can be variable here should be the time of the algorithm progression.

    - Test problem          : Farmer, classic (all x's are first stage, y and w are second.)
    - Lower Bounder         : default, dropping non-anticipativity
    - Candidate Generator   : average lower bound solutions
    - Queuing Method        : LIFO
    """

    subproblem_names = ["good", "fair", "bad"]
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=farmer_classic_subproblem_creator)
    params.set_bounders(candidate_solution_finder = sno.AverageLowerBoundSolution)
    params.set_bounds_tightening(fbbt = False,
                                 obbt = False)
    params.activate_verbose()
    params.deactivate_global_guarantee()
    params.set_queue_strategy(sno.QueueStrategy.bound)

    # solve without FBBT
    np.random.seed(42)      # reset seed
    snoglode_no_fbbt = sno.Solver(params)
    
    # solve with given tolerances
    snoglode_no_fbbt.solve(max_iter=10,
                           rel_tolerance=1e-2,
                           abs_tolerance=1e-3,
                           collect_plot_info=False)
    
    # solve with FBBT
    np.random.seed(42)      # reset seed
    params.set_bounds_tightening(fbbt = True,
                                 obbt = False)
    snoglode_fbbt = sno.Solver(params)
    
    # solve with given tolerances
    snoglode_fbbt.solve(max_iter=10,
                    rel_tolerance=1e-2,
                    abs_tolerance=1e-3,
                    collect_plot_info=False)

    # expect the exact number of iterations / final gap
    assert snoglode_no_fbbt.iteration == snoglode_fbbt.iteration      
    assert snoglode_no_fbbt.solution.relative_gap == snoglode_fbbt.solution.relative_gap
    
    # check first stage solutions - all the same + right values
    for name in snoglode_no_fbbt.subproblems.names:

        assert snoglode_no_fbbt.solution.subproblem_solutions[name]["x[wheat]"] == pytest.approx(167.7777)
        assert snoglode_no_fbbt.solution.subproblem_solutions[name]["x[corn]"] == pytest.approx(82.2222)
        assert snoglode_no_fbbt.solution.subproblem_solutions[name]["x[beets]"] == pytest.approx(250)

        assert snoglode_fbbt.solution.subproblem_solutions[name]["x[wheat]"] == pytest.approx(167.7777)
        assert snoglode_fbbt.solution.subproblem_solutions[name]["x[corn]"] == pytest.approx(82.2222)
        assert snoglode_fbbt.solution.subproblem_solutions[name]["x[beets]"] == pytest.approx(250)
    
    # check second stage solutions
    assert snoglode_no_fbbt.solution.subproblem_solutions["good"]["w[wheat]"] == pytest.approx(303.3333)
    assert snoglode_no_fbbt.solution.subproblem_solutions["good"]["w[corn]"] == pytest.approx(56)
    assert snoglode_no_fbbt.solution.subproblem_solutions["good"]["w[beets_favorable]"] == pytest.approx(6000)
    assert snoglode_no_fbbt.solution.subproblem_solutions["good"]["w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_no_fbbt.solution.subproblem_solutions["good"]["y[wheat]"] == pytest.approx(0)
    assert snoglode_no_fbbt.solution.subproblem_solutions["good"]["y[corn]"] == pytest.approx(0)

    assert snoglode_fbbt.solution.subproblem_solutions["good"]["w[wheat]"] == pytest.approx(303.3333)
    assert snoglode_fbbt.solution.subproblem_solutions["good"]["w[corn]"] == pytest.approx(56)
    assert snoglode_fbbt.solution.subproblem_solutions["good"]["w[beets_favorable]"] == pytest.approx(6000)
    assert snoglode_fbbt.solution.subproblem_solutions["good"]["w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["good"]["y[wheat]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["good"]["y[corn]"] == pytest.approx(0)

    assert snoglode_no_fbbt.solution.subproblem_solutions["fair"]["w[wheat]"] == pytest.approx(219.4444)
    assert snoglode_no_fbbt.solution.subproblem_solutions["fair"]["w[corn]"] == pytest.approx(6.66666666)
    assert snoglode_no_fbbt.solution.subproblem_solutions["fair"]["w[beets_favorable]"] == pytest.approx(5000)
    assert snoglode_no_fbbt.solution.subproblem_solutions["fair"]["w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_no_fbbt.solution.subproblem_solutions["fair"]["y[wheat]"] == pytest.approx(0)
    assert snoglode_no_fbbt.solution.subproblem_solutions["fair"]["y[corn]"] == pytest.approx(0)

    assert snoglode_fbbt.solution.subproblem_solutions["fair"]["w[wheat]"] == pytest.approx(219.4444)
    assert snoglode_fbbt.solution.subproblem_solutions["fair"]["w[corn]"] == pytest.approx(6.66666666)
    assert snoglode_fbbt.solution.subproblem_solutions["fair"]["w[beets_favorable]"] == pytest.approx(5000)
    assert snoglode_fbbt.solution.subproblem_solutions["fair"]["w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["fair"]["y[wheat]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["fair"]["y[corn]"] == pytest.approx(0)

    assert snoglode_no_fbbt.solution.subproblem_solutions["bad"]["w[wheat]"] == pytest.approx(135.5555)
    assert snoglode_no_fbbt.solution.subproblem_solutions["bad"]["w[corn]"] == pytest.approx(0)
    assert snoglode_no_fbbt.solution.subproblem_solutions["bad"]["w[beets_favorable]"] == pytest.approx(4000)
    assert snoglode_no_fbbt.solution.subproblem_solutions["bad"]["w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_no_fbbt.solution.subproblem_solutions["bad"]["y[wheat]"] == pytest.approx(0)
    assert snoglode_no_fbbt.solution.subproblem_solutions["bad"]["y[corn]"] == pytest.approx(42.66666666)

    assert snoglode_fbbt.solution.subproblem_solutions["bad"]["w[wheat]"] == pytest.approx(135.5555)
    assert snoglode_fbbt.solution.subproblem_solutions["bad"]["w[corn]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["bad"]["w[beets_favorable]"] == pytest.approx(4000)
    assert snoglode_fbbt.solution.subproblem_solutions["bad"]["w[beets_unfavorable]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["bad"]["y[wheat]"] == pytest.approx(0)
    assert snoglode_fbbt.solution.subproblem_solutions["bad"]["y[corn]"] == pytest.approx(42.66666666)


def test_pmedian_averge_solution_fbbt():
    """
    Here, we solve using the averaging of the lower bound solution and following specifications
    with FBBT and without, asserting everything remains the same.
    The only thing that can be variable here should be the time of the algorithm progression.

    - Test problem          : P-median (facility locations are first stage, community assignments are second stage.)
    - Lower Bounder         : default, dropping non-anticipativity
    - Upper Bounder         : solve the extensive form (to local optimality)
    - Queuing Method        : bound
    - Bounds Tightening     : None
    """
    nb_facilities = 5
    max_facilities = 1
    total_communities = 10
    nb_subproblems = 3

    subproblem_names = [f"facilities_{nb_facilities}_max_{max_facilities}_communities_{total_communities}_subproblems_{nb_subproblems}_subproblem_{subproblem}" \
                            for subproblem in np.arange(nb_subproblems)]
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=pmedian_subproblem_creator)
    params.set_bounders(candidate_solution_finder = sno.AverageLowerBoundSolution)
    params.set_bounds_tightening(fbbt = True,
                                 obbt = False)
    params.activate_verbose()
    params.deactivate_global_guarantee()
    params.set_queue_strategy(sno.QueueStrategy.bound)

    # solve with FBBT
    np.random.seed(42)      # reset seed
    snoglode_fbbt = sno.Solver(params)
    
    # solve with given tolerances
    snoglode_fbbt.solve()
    
    # solve with without FBBT
    np.random.seed(42)      # reset seed
    params.set_bounds_tightening(fbbt = False,
                                 obbt = False)
    snoglode_no_fbbt = sno.Solver(params)
    
    # solve with given tolerances
    snoglode_no_fbbt.solve()
    
    # expect the exact number of iterations / final gap
    assert snoglode_no_fbbt.iteration == snoglode_fbbt.iteration

    # the nodes should have progressed the same
    assert snoglode_no_fbbt.tree.n_nodes() == snoglode_fbbt.tree.n_nodes()
    assert snoglode_no_fbbt.tree.metrics.nodes.pruned_by_bound \
            == snoglode_fbbt.tree.metrics.nodes.pruned_by_bound
    assert snoglode_no_fbbt.tree.metrics.nodes.pruned_by_infeasibility \
            == snoglode_fbbt.tree.metrics.nodes.pruned_by_infeasibility
    assert snoglode_no_fbbt.tree.metrics.nodes.explored \
            == snoglode_fbbt.tree.metrics.nodes.explored

    # check objective value
    assert snoglode_no_fbbt.tree.metrics.ub == pytest.approx(303.01594)
    assert snoglode_fbbt.tree.metrics.ub == pytest.approx(303.01594)


def test_pmedian_averge_solution_obbt():
    """
    Bilinear stochastic program (nonconvex, nonlinear program with multiple scenarios)

    - Test problem          : P-median (facility locations are first stage, community assignments are second stage.)
    - Lower Bounder         : default, dropping non-anticipativity
    - Upper Bounder         : solve the extensive form (to local optimality)
    - Queuing Method        : bound
    - Bounds Tightening     : None
    """
    # nb_facilities = 2
    # max_facilities = 1
    # total_communities = 5
    # nb_subproblems = 2
    nb_facilities = 5
    max_facilities = 2
    total_communities = 25
    nb_subproblems = 3

    subproblem_names = [f"facilities_{nb_facilities}_max_{max_facilities}_communities_{total_communities}_subproblems_{nb_subproblems}_subproblem_{subproblem}" \
                            for subproblem in np.arange(nb_subproblems)]
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=pmedian_subproblem_creator)
    params.set_bounders(candidate_solution_finder = sno.AverageLowerBoundSolution)
    params.set_bounds_tightening(fbbt = False,
                                 obbt = True)
    params.activate_verbose()
    params.deactivate_global_guarantee()
    params.set_queue_strategy(sno.QueueStrategy.bound)

    # set up solver
    solver = sno.Solver(params)
    
    # solve with given tolerances
    solver.solve(max_iter=50,
                 rel_tolerance=1e-2,
                 abs_tolerance=1e-3,
                 collect_plot_info=False)

    assert solver.iteration <= 50      # should converge on relative gap tolerance after 40 nodes explored
    assert solver.runtime <= 15        # should be around 10 seconds

    # check first stage solutions
    for name in solver.subproblems.names:
        assert solver.solution.subproblem_solutions[name][f"x[0]"] == 1.0
        assert solver.solution.subproblem_solutions[name][f"x[1]"] == 1.0
        assert solver.solution.subproblem_solutions[name][f"x[2]"] == 0.0
        assert solver.solution.subproblem_solutions[name][f"x[3]"] == 0.0
        assert solver.solution.subproblem_solutions[name][f"x[4]"] == 0.0

    # check objective value
    assert solver.tree.metrics.ub == pytest.approx(718.365813)

if __name__=="__main__":
    # test_farmer_classic_ef_bounds_tightening()
    # test_farmer_classic_average_solution_bounds_tightening_fbbt()
    test_pmedian_averge_solution_fbbt()
    test_pmedian_averge_solution_obbt()