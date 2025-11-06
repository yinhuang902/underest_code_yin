""""
    Robertson, Dillard, Pengfei Cheng, and Joseph K. Scott. 
    "On the convergence order of value function relaxations 
    used in decomposition-based global optimization of nonconvex 
    stochastic programs." Journal of Global Optimization 91.4 (2025): 701-742.

This example implements the demonstrative problem as described in the paper.

min (1.00694 · (x_1)**3 − 4.74589 · (x_1)**2 + 5.17523 · x_1)  
        + (−0.677232 · (x_2)**3 + 3.03949 · (x_2)**2 − 3.02338 · x_2)  
    s.t. x1 = y 
         x2 = y  
         0 ≤ y ≤ 3.
"""
import snoglode as sno
import pyomo.environ as pyo
import os

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

def subproblem_creator(scenario_name):
    """
    Based on the scenario, generates 
        1) the pyomo model
        2) the dict of lifted variable IDS : pyo.Var
        3) the list of subprob lem specific variables (pyo.Vars)
        3) probability of subproblem
    and returns as a list in this order.
    """
    
    if scenario_name == "1":
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(0, 3))
        model.obj = pyo.Objective(expr=1.00694 * model.x**3 - 4.74589 * model.x**2 + 5.17523 * model.x, 
                                  sense=pyo.minimize)
    elif scenario_name == "2":
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(0, 3))
        model.obj = pyo.Objective(expr=-0.677232 * model.x**3 + 3.03949 * model.x**2 - 3.02338 * model.x, 
                                  sense=pyo.minimize)

    lifted_variable_ids = {"x": model.x}        # lifted variable IDs for this subproblem
    scenario_probability = 1                    # probability of this subproblem

    return [model,                              # pyomo model corresponding to this subproblem
            lifted_variable_ids,                # lifted varID : pyo.Var dict
            scenario_probability]               # probability of this subproblem


if __name__=="__main__":
    
    subproblem_names = ["1", "2"]
    params = sno.SolverParameters(subproblem_names = subproblem_names,
                                  subproblem_creator = subproblem_creator,
                                  lb_solver = pyo.SolverFactory("baron"),
                                  cg_solver = pyo.SolverFactory("baron"),
                                  ub_solver = pyo.SolverFactory("baron"))
    if (size==1): params.set_logging(fname = os.getcwd() + "/logs/rcs_problem_log")
    else: params.set_logging(fname = os.getcwd() + "/logs/rcs_problem_log_parallel")
    params.set_bounds_tightening(fbbt=False, obbt=False) # no constraints, so no bounds tightening
    if (rank==0): params.display()
    params.deactivate_global_guarantee()  # no global convergence guarantee needed for this problem
    params.set_bounders(lower_bounder = sno.DropNonants,
                        candidate_solution_finder = sno.AverageLowerBoundSolution)
    params.set_branching(partition_strategy = sno.ExpectedValue) # selection strategy irrelevant- only 1 branching variable

    solver = sno.Solver(params)
    solver.solve(max_iter=100)

    MPI.COMM_WORLD.barrier()
    print("\n====================================================================")
    print("SOLUTION")
    for name in solver.subproblems.names:
        print(f"subproblem = {name}")
        for var_name in solver.solution.subproblem_solutions[name]:
            var_val = solver.solution.subproblem_solutions[name][var_name]
            print(f"  var name = {var_name}, value = {var_val}")
        print()
    print("====================================================================")