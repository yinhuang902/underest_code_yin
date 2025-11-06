"""
A bounded knapsack example with integer variables (i.e., can only select a certain number of each item)

Note: this is only a single scenario problem, but we can still use SNoGloDe to solve;
it reverts to spatial branch and bound in this case.
"""
import pyomo.environ as pyo
import snoglode as sno
import os

def knapsack_pyomo_model():
    """
    Builds basic Integer program
    See: Ignacaio's textbook, pg. 78 for the model and depiction of the 
         branch and bound tree
    """
    items = [1,2,3,4,5]
    profits = {1: 5, 
               2: 3, 
               3: 2,
               4: 7,
               5: 4}     # profit of each item
    weights = {1: 2, 
               2: 8, 
               3: 4,
               4: 2,
               5: 5}     # weight of each item
    capacity = 11        # knapsack capacity

    # build the model & variables
    m = pyo.ConcreteModel()
    m.available_items = pyo.Set(initialize=items)
    m.x = pyo.Var(items, 
                  domain=pyo.NonNegativeIntegers,
                  bounds=(0,3))   # can select at most two of each item
    
    # min -profit = max profit
    m.obj=pyo.Objective( expr = -sum(profits[i]*m.x[i] for i in m.available_items))

    # knapsack capacity constraint
    m.capacity = pyo.Constraint(expr=sum(weights[i] * m.x[i] for i in m.available_items) <= capacity)
    return m

def subproblem_creator(subproblem_name):
    """
    Based on the scenario, generates 
        1) the pyomo model
        2) the list of first stage variables
        3) probability
    and returns as a list in this order.
    """
    # create parameters / model stored in obj for this scenario
    model = knapsack_pyomo_model()

    # grab the list of first stage variables
    lifted_variable_ids = {("item", i): model.x[i] \
                                for i in model.available_items}

    # probability of this particular scenario occuring
    scenario_probability = 1.0

    return [model,                              # pyomo model corresponding to this subproblem
            lifted_variable_ids,                # lifted varID : pyo.Var dict
            scenario_probability]               # probability of this subproblem


if __name__=="__main__":
    
    subproblem_names = ["determistic"]
    params = sno.SolverParameters(subproblem_names = subproblem_names,
                                subproblem_creator = subproblem_creator,
                                lb_solver = pyo.SolverFactory("gurobi"),
                                cg_solver = pyo.SolverFactory("gurobi"),
                                ub_solver = pyo.SolverFactory("gurobi"))
    params.set_bounds_tightening(fbbt = False, obbt=False)
    params.set_bounders(candidate_solution_finder = sno.AverageLowerBoundSolution)
    params.activate_verbose()
    params.relax_integers()
    params.set_logging(fname = os.getcwd() + "/logs/knapsack_log")
    params.display()
    solver = sno.Solver(params)
    solver.solve()

    print("\n====================================================================")
    print("SOLUTION")
    for name in solver.subproblems.names:
        print(f"subproblem = {name}")
        for var_name in solver.solution.subproblem_solutions[name]:
            var_val = solver.solution.subproblem_solutions[name][var_name]
            print(f"  var name = {var_name}, value = {var_val}")
        print()
    print("====================================================================")