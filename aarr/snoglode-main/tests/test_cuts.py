"""
Added cuts tests
"""
import pytest as pytest
import numpy as np
import pyomo.environ as pyo
import snoglode as sno

try: from problems import IntegerProgram, MockCandidateGenerator
except: from .problems import IntegerProgram, MockCandidateGenerator
    
gurobi = pyo.SolverFactory("gurobi")

@pytest.mark.mpi_skip()
def test_activation_deactivation_lb_obj_cut():
    """
    When we solve the LB problem of a node, for all child 
    nodes it follows that (in the case of minimization):

        Obj_{lb, descendent} >= Obj_{lb, predecessor}

    In other words, every LB solve should have an objective
    value that is bounded from below by all predecessor objectives.

    In this case, we simply taking the parent's objective 
    (because each is inherited, we always maintain the only cut
    that will remain active) and add a cut to the objective.
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
    solver.solve(max_iter = 1)

    # extract pyomo model
    model = solver.subproblems.model["dummy"]

    # count number of original constraints
    num_constraints = 0
    for con in model.component_data_objects(pyo.Constraint): 
        if con._active: num_constraints += 1

    # activate bounds
    solver.lower_bounder.activate_bound_cuts(node = solver.tree.get_node(),
                                             subproblem_model = model)
    
    # count number of constraints with the cut
    num_constraints_with_cut = 0
    for con in model.component_data_objects(pyo.Constraint): 
        if con._active: num_constraints_with_cut += 1

    # check that we have generated exactly one extra active constraint
    assert num_constraints + 1 == num_constraints_with_cut

    solver.lower_bounder.deactivate_bound_cuts(model)

    # count number of constraints once we have deactivated
    num_constraints_with_deactive_cut = 0
    for con in model.component_data_objects(pyo.Constraint): 
        if con._active: num_constraints_with_deactive_cut += 1

    # check that we have properly deactivated constraint
    assert num_constraints == num_constraints_with_deactive_cut


if __name__=="__main__":
    test_activation_deactivation_lb_obj_cut()