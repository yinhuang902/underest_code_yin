"""
Tests the snoglode implementation on a specific set of test problems, most of which are
available in the examples folder.

Takes only the classic farmer problem and solves using the different candidate generators.
So far, we turn off all bounds tightening to reserve that testing for another file.
"""
import pytest as pytest
import numpy as np
import snoglode as sno
import pyomo.environ as pyo

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

try: from problems import farmer_classic_subproblem_creator, \
                                bilinear_subproblem_creator, \
                                        pmedian_subproblem_creator, \
                                            farmer_skew_subproblem_creator
except: from .problems import farmer_classic_subproblem_creator, \
                                bilinear_subproblem_creator, \
                                        pmedian_subproblem_creator, \
                                            farmer_skew_subproblem_creator

gurobi = pyo.SolverFactory("gurobi")
nonconvex_gurobi = pyo.SolverFactory("gurobi")
nonconvex_gurobi.options["NonConvex"] = 2

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_farmer_classic_ef():
    """
    Tests the classic farmer problem, 3 scenarios with 3 sets of indexed vars (x,y,w)

    - Test problem          : Farmer, classic (all x's are first stage, y and w are second.)
    - Lower Bounder         : default, dropping non-anticipativity
    - Upper Bounder         : solve the extensive form (to local optimality)
    - Queuing Method        : bound
    - Bounds Tightening     : None
    """

    subproblem_names = ["good", "fair", "bad"]
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=farmer_classic_subproblem_creator,
                                  cg_solver = gurobi)
    params.set_bounders(candidate_solution_finder = sno.SolveExtensiveForm)
    params.set_bounds_tightening(fbbt = False)
    params.activate_verbose()
    params.deactivate_global_guarantee()
    params.set_queue_strategy(sno.QueueStrategy.bound)

    # set up solver
    solver = sno.Solver(params)
        
    # solve with given tolerances
    solver.solve(max_iter=10,
                 rel_tolerance=1e-2,
                 abs_tolerance=1e-3,
                 collect_plot_info=False)

    assert solver.iteration <= 15      # should converge on relative gap tolerance after 4 nodes explored
    assert solver.runtime <= 4         # should be around 1.1 seconds

    # check first stage solutions - all the same + right values
    for name in solver.subproblems.names:
        assert solver.solution.subproblem_solutions[name][f"{name}.x[wheat]"] == pytest.approx(170)
        assert solver.solution.subproblem_solutions[name][f"{name}.x[corn]"] == pytest.approx(80)
        assert solver.solution.subproblem_solutions[name][f"{name}.x[beets]"] == pytest.approx(250)
    
    # check second stage solutions
    if "good" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["good"]["good.w[wheat]"] == pytest.approx(310)
        assert solver.solution.subproblem_solutions["good"]["good.w[corn]"] == pytest.approx(48)
        assert solver.solution.subproblem_solutions["good"]["good.w[beets_favorable]"] == pytest.approx(6000)
        assert solver.solution.subproblem_solutions["good"]["good.w[beets_unfavorable]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["good"]["good.y[wheat]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["good"]["good.y[corn]"] == pytest.approx(0)

    if "fair" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["fair"]["fair.w[wheat]"] == pytest.approx(225)
        assert solver.solution.subproblem_solutions["fair"]["fair.w[corn]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["fair"]["fair.w[beets_favorable]"] == pytest.approx(5000)
        assert solver.solution.subproblem_solutions["fair"]["fair.w[beets_unfavorable]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["fair"]["fair.y[wheat]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["fair"]["fair.y[corn]"] == pytest.approx(0)

    if "bad" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["bad"]["bad.w[wheat]"] == pytest.approx(140)
        assert solver.solution.subproblem_solutions["bad"]["bad.w[corn]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["bad"]["bad.w[beets_favorable]"] == pytest.approx(4000)
        assert solver.solution.subproblem_solutions["bad"]["bad.w[beets_unfavorable]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["bad"]["bad.y[wheat]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["bad"]["bad.y[corn]"] == pytest.approx(48)


@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_farmer_skew_average_solution():
    """
    - Test problem          : Farmer, skew (some x's are first stage, y and w are second.)
    - Lower Bounder         : default, dropping non-anticipativity
    - Candidate Generator   : average lower bound solutions
    - Queuing Method        : bound
    - Bounds Tightening     : None
    """
    # reset seed 
    np.random.seed(17)

    subproblem_names = ["good", "fair", "bad"]
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=farmer_skew_subproblem_creator)
    params.inherit_solutions_from_parent(True)
    params.set_bounders(candidate_solution_finder = sno.AverageLowerBoundSolution)
    params.activate_verbose()
    params.set_queue_strategy(sno.QueueStrategy.bound)

    # set up solver
    solver = sno.Solver(params)
    solver.solve(max_iter = 1)
    
    # check second stage solutions
    if "good" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["good"]["x[wheat]"] == pytest.approx(179.1666)
        assert solver.solution.subproblem_solutions["good"]["x[corn]"] == pytest.approx(45.8333)
        assert solver.solution.subproblem_solutions["good"]["x[beets]"] == pytest.approx(275)
        assert solver.solution.subproblem_solutions["good"]["w[wheat]"] == pytest.approx(337.5)
        assert solver.solution.subproblem_solutions["good"]["w[corn]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["good"]["w[beets_favorable]"] == pytest.approx(6000)
        assert solver.solution.subproblem_solutions["good"]["w[beets_unfavorable]"] == pytest.approx(600)
        assert solver.solution.subproblem_solutions["good"]["y[wheat]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["good"]["y[corn]"] == pytest.approx(75)
    
    if "fair" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["fair"]["x[wheat]"] == pytest.approx(110)
        assert solver.solution.subproblem_solutions["fair"]["x[corn]"] == pytest.approx(115)
        assert solver.solution.subproblem_solutions["fair"]["x[beets]"] == pytest.approx(275)
        assert solver.solution.subproblem_solutions["fair"]["w[wheat]"] == pytest.approx(75)
        assert solver.solution.subproblem_solutions["fair"]["w[corn]"] == pytest.approx(105)
        assert solver.solution.subproblem_solutions["fair"]["w[beets_favorable]"] == pytest.approx(5500)
        assert solver.solution.subproblem_solutions["fair"]["w[beets_unfavorable]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["fair"]["y[wheat]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["fair"]["y[corn]"] == pytest.approx(0)

    if "bad" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["bad"]["x[wheat]"] == pytest.approx(110)
        assert solver.solution.subproblem_solutions["bad"]["x[corn]"] == pytest.approx(45.8333)
        assert solver.solution.subproblem_solutions["bad"]["x[beets]"] == pytest.approx(344.1666)
        assert solver.solution.subproblem_solutions["bad"]["w[wheat]"] == pytest.approx(20)
        assert solver.solution.subproblem_solutions["bad"]["w[corn]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["bad"]["w[beets_favorable]"] == pytest.approx(5506.6666)
        assert solver.solution.subproblem_solutions["bad"]["w[beets_unfavorable]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["bad"]["y[wheat]"] == pytest.approx(0)
        assert solver.solution.subproblem_solutions["bad"]["y[corn]"] == pytest.approx(129.9999)


@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_farmer_classic_average_solution():
    """
    - Test problem          : Farmer, classic (all x's are first stage, y and w are second.)
    - Lower Bounder         : default, dropping non-anticipativity
    - Candidate Generator   : average lower bound solutions
    - Queuing Method        : bound
    - Bounds Tightening     : None
    """
    # reset seed 
    np.random.seed(42)

    subproblem_names = ["good", "fair", "bad"]
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=farmer_classic_subproblem_creator)
    params.set_bounders(candidate_solution_finder = sno.AverageLowerBoundSolution)
    params.set_bounds_tightening(fbbt = False)
    params.activate_verbose()
    params.deactivate_global_guarantee()
    params.set_queue_strategy(sno.QueueStrategy.bound)

    # set up solver
    solver = sno.Solver(params)

    # check first stage solutions - all the same + right values
    solver.solve(max_iter = 0)
    for name in solver.subproblems.names:
        # ((good[wheat] = 183.33) + (fair[wheat] = 120) + (bad[wheat] = 100))/3
        assert solver.solution.subproblem_solutions[name]["x[wheat]"] == pytest.approx(134.4444444)
        # ((good[corn] = 66.66) + (fair[corn] = 80) + (bad[corn] = 25))/3
        assert solver.solution.subproblem_solutions[name]["x[corn]"] == pytest.approx(57.22222222222223)
        # ((good[beets] = 250) + (fair[beets] = 300) + (bad[beets] = 375))/3
        assert solver.solution.subproblem_solutions[name]["x[beets]"] == pytest.approx(308.3333333333333)


@pytest.mark.skipif(size > 5, reason="test can run with at most 5 ranks.")
def test_bilinear_ef_local():
    """
    Bilinear stochastic program (nonconvex, nonlinear program with multiple scenarios)

    - Test problem          : Farmer, classic (all x's are first stage, y and w are second.)
    - Lower Bounder         : default, dropping non-anticipativity
    - Upper Bounder         : solve the extensive form (to local optimality)
    - Queuing Method        : bound
    - Bounds Tightening     : None
    """
    num_scenarios = 5
    num_subproblems = np.arange(num_scenarios)
    subproblem_names = [f"{n}_{num_scenarios}" for n in num_subproblems]
    
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=bilinear_subproblem_creator,
                                  lb_solver = nonconvex_gurobi,
                                  cg_solver = nonconvex_gurobi,
                                  ub_solver = nonconvex_gurobi)
    params.activate_verbose()
    params.set_bounds_tightening(fbbt = False,
                                 obbt = False)
    params.deactivate_global_guarantee()
    params.set_bounders(candidate_solution_finder = sno.SolveExtensiveForm)
    params.set_queue_strategy(sno.QueueStrategy.bound)

    # set up solver
    solver = sno.Solver(params)

    # solve with given tolerances
    solver.solve(max_iter=10,
                 rel_tolerance=1e-2,
                 abs_tolerance=1e-3,
                 collect_plot_info=False)
    
    # check expected number of iters
    assert solver.iteration <= 15         # expecting about 11
    assert solver.runtime <= 7            # expecting ~4 seconds

    # check first stage solutions - all the same + right values
    for name in solver.subproblems.names:
        assert solver.solution.subproblem_solutions[name][f"{name}.x"] == pytest.approx(4.586331439187505)

    if "0_5" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["0_5"][f"0_5.y"] == pytest.approx(1.6885102226734254)
    if "1_5" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["1_5"][f"1_5.y"] == pytest.approx(1.5448316628350869)
    if "2_5" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["2_5"][f"2_5.y"] == pytest.approx(1.5655158389473034)
    if "3_5" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["3_5"][f"3_5.y"] == pytest.approx(1.6906736060589944)
    if "4_5" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["4_5"][f"4_5.y"] == pytest.approx(2.144447981023094)

    # check objective value
    assert solver.tree.metrics.ub == pytest.approx(-6.3131273)


@pytest.mark.skipif(size > 5, reason="test can run with at most 5 ranks.")
def test_bilinear_average_solution():
    """
    Bilinear stochastic program (nonconvex, nonlinear program with multiple scenarios)

    - Test problem          : Farmer, classic (all x's are first stage, y and w are second.)
    - Lower Bounder         : default, dropping non-anticipativity
    - Upper Bounder         : average 
    - Queuing Method        : bound
    - Bounds Tightening     : None
    """
    num_scenarios = 5
    num_subproblems = np.arange(num_scenarios)
    subproblem_names = [f"{n}_{num_scenarios}" for n in num_subproblems]

    # set up solver
    np.random.seed(2)      # reset seed
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=bilinear_subproblem_creator,
                                  lb_solver = nonconvex_gurobi,
                                  cg_solver = nonconvex_gurobi,
                                  ub_solver = nonconvex_gurobi)
    params.activate_verbose()
    params.set_bounds_tightening(fbbt = False,
                                 obbt = False)
    params.deactivate_global_guarantee()
    params.set_bounders(candidate_solution_finder = sno.AverageLowerBoundSolution)
    params.set_queue_strategy(sno.QueueStrategy.bound)

    # set up solver
    solver = sno.Solver(params)

    # solve with given tolerances
    solver.solve(max_iter=15,
                 rel_tolerance=1e-2,
                 abs_tolerance=1e-3,
                 collect_plot_info=False)
    
    # check expected number of iters
    assert solver.iteration <= 15         # expecting about 11
    assert solver.runtime <= 1            # expecting ~0.3 seconds

    # check first stage solutions - all the same + right values
    for name in solver.subproblems.names:
        assert solver.solution.subproblem_solutions[name][f"x"] == pytest.approx(4.565973477)

    if "0_5" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["0_5"][f"y"] == pytest.approx(1.6960386558902774)
    if "1_5" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["1_5"][f"y"] == pytest.approx(1.5517194872904927)
    if "2_5" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["2_5"][f"y"] == pytest.approx(1.5724958863791623)
    if "3_5" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["3_5"][f"y"] == pytest.approx(1.69821168499023)
    if "4_5" in solver.subproblems.names:
        assert solver.solution.subproblem_solutions["4_5"][f"y"] == pytest.approx(2.1540092695455786)

    # we should not have exceeded the EF solution
    assert solver.tree.metrics.ub >= -6.3131273
    

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_pmedian_ef_solution():
    """
    Bilinear stochastic program (nonconvex, nonlinear program with multiple scenarios)

    - Test problem          : P-median (facility locations are first stage, community assignments are second stage.)
    - Lower Bounder         : default, dropping non-anticipativity
    - Upper Bounder         : solve the extensive form (to local optimality)
    - Queuing Method        : bound
    - Bounds Tightening     : None
    """
    nb_facilities = 5
    max_facilities = 2
    total_communities = 25
    nb_subproblems = 3

    subproblem_names = [f"facilities_{nb_facilities}_max_{max_facilities}_communities_{total_communities}_subproblems_{nb_subproblems}_subproblem_{subproblem}" \
                            for subproblem in np.arange(nb_subproblems)]
    
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=pmedian_subproblem_creator,
                                  cg_solver = gurobi)
    params.set_bounders(candidate_solution_finder = sno.SolveExtensiveForm)
    params.set_bounds_tightening(fbbt = False)
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

    assert solver.iteration <= 20      # should converge on relative gap tolerance after 19 nodes explored
    assert solver.runtime <= 7         # should be around 5 seconds

    # check first stage solutions
    for name in solver.subproblems.names:
        assert solver.solution.subproblem_solutions[name][f"{name}.x[0]"] == 1.0
        assert solver.solution.subproblem_solutions[name][f"{name}.x[1]"] == 1.0
        assert solver.solution.subproblem_solutions[name][f"{name}.x[2]"] == 0.0
        assert solver.solution.subproblem_solutions[name][f"{name}.x[3]"] == 0.0
        assert solver.solution.subproblem_solutions[name][f"{name}.x[4]"] == 0.0

    # check objective value
    assert solver.tree.metrics.ub == pytest.approx(718.365813)


@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_pmedian_averge_solution():
    """
    Bilinear stochastic program (nonconvex, nonlinear program with multiple scenarios)

    - Test problem          : P-median (facility locations are first stage, community assignments are second stage.)
    - Lower Bounder         : default, dropping non-anticipativity
    - Upper Bounder         : solve the extensive form (to local optimality)
    - Queuing Method        : bound
    - Bounds Tightening     : None
    """
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
                                 obbt = False)
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
    # test_farmer_classic_ef()
    # test_farmer_classic_average_solution()
    # test_farmer_skew_average_solution()

    # test_bilinear_ef_local()
    test_bilinear_average_solution()

    # test_pmedian_ef_solution()
    # test_pmedian_averge_solution()