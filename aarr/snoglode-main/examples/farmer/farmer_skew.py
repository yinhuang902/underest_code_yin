"""
A farmer is trying to decide what to plant for the next season.
    - He has wheat (1), corn (2), and sugar beets (3)
    - He can buy (y), sell (w), and produce (x,x_yield) any of these three products.
    - He must buy / produce a min. amount of wheat / corn for his cattle.
    - Quota on sugar beets alters the price after a certain threshold.
Goal: max. the net proficts from purchasing, selling, and planting crops for next season.

However, there is no guarantees on the weather. The actual yield of the crops changes depending on weather -> aka uncertain.
Here, let's consider 3 scenarios: good, fair, or bad weather next year.
Link the different scenario variables together to get the best possible solution.

NOTE: Here will not link everything classically, to test different linking methods.

ENFORE:
    - devoted_acres[corn, good] == devoted_acres[corn, bad]
    - devoted_acres[wheat, fair] == devoted_acres[wheet, bad]
    - devoted_acres[beats, scen] \forall scen \in [good, fair, bad]
"""
from farmer_problem import TwoStageFarmer
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
        2) the list of first stage variables
        3) probability
    and returns as a list in this order.
    """
    name_to_yield_map = {
        "good": 1.2,
        "fair": 1.0,
        "bad": 0.8
    }
    
    # create parameters / model stored in obj for this scenario
    farmer_scenario = TwoStageFarmer(name_to_yield_map[scenario_name])
    
    # for good - only need to include corn and beets
    if scenario_name=="good":
        crops = ["corn", "beets"]
    
    # for fair - only need to include wheat and beets
    if scenario_name=="fair":
        crops = ["wheat", "beets"]
    
    # for bad - only need to include corn and beets
    if scenario_name=="bad":
        crops = ["corn", "wheat"]

    # grab the list of first stage variables
    lifted_variable_ids = {("devoted_acrege", crop): farmer_scenario.model.x[crop] \
                                for crop in crops}
    
    # probability of this particular scenario occuring
    scenario_probability = 1/3

    return [farmer_scenario.model,              # pyomo model corresponding to this subproblem
            lifted_variable_ids,                # lifted varID : pyo.Var dict
            scenario_probability]               # probability of this subproblem


if __name__=="__main__":
    
    subproblem_names = ["good", "fair", "bad"]
    params = sno.SolverParameters(subproblem_names = subproblem_names,
                                  subproblem_creator = subproblem_creator,
                                  lb_solver = pyo.SolverFactory("gurobi"),
                                  cg_solver = pyo.SolverFactory("gurobi"),
                                  ub_solver = pyo.SolverFactory("gurobi"))
    params.inherit_solutions_from_parent(True)
    params.set_bounders(candidate_solution_finder=sno.AverageLowerBoundSolution)
    
    if (size==1): params.set_logging(fname = os.getcwd() + "/logs/farmer_skew_log")
    else: params.set_logging(fname = os.getcwd() + "/logs/farmer_skew_log_parallel")
    if (rank==0): params.display()

    solver = sno.Solver(params)
    solver.solve(max_iter=1)
    
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