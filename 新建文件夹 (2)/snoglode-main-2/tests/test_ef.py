"""
Testing auto-build of the Extensive Form (EF) capability
"""
import pytest as pytest
import snoglode as sno
import pyomo.environ as pyo

try: from .problems import farmer_skew_subproblem_creator, farmer_classic_subproblem_creator
except: from problems import farmer_skew_subproblem_creator, farmer_classic_subproblem_creator

gurobi = pyo.SolverFactory("gurobi")

def test_ef_generation_farmer_skew():
    """
    When we generate the skewed version of the farmer problem,
    we want to ensure that we only are building the correct
    number of constraints to represent the problem
    """
    subproblem_names = ["good", "fair", "bad"]
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=farmer_skew_subproblem_creator)
    solver = sno.Solver(params)
    
    # should be three lifted vars
    assert len(solver.subproblems.ef.model.lifted_vars) == 3

    # we expected 6 constraints (because not all lifted variables are linked)
    solver.subproblems.ef.activate()
    assert len(solver.subproblems.ef.model.nonants) == 6


def test_ef_generation_farmer_classic():
    """
    When we generate the classic version of the farmer problem,
    we want to ensure that we only are building the correct
    number of constraints to represent the problem
    """
    subproblem_names = ["good", "fair", "bad"]
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=farmer_classic_subproblem_creator,
                                  cg_solver = gurobi)
    solver = sno.Solver(params)
    
    # should be three lifted vars
    assert len(solver.subproblems.ef.model.lifted_vars) == 3

    # we expected 9 constraints (because not all lifted variables are linked across all periods)
    solver.subproblems.ef.activate()
    assert len(solver.subproblems.ef.model.nonants) == 9

if __name__=="__main__":
    test_ef_generation_farmer_classic()
    test_ef_generation_farmer_skew()