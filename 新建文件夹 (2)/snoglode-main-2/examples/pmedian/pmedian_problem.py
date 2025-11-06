import pyomo.environ as pyo
import numpy as np
import random

class PMedian():
    """
    Given nb_facilities potential facilities to build, and
    nb_communities to consider, select which subset of max_facilities
    should be built.
    """
    def __init__(self, 
                 nb_facilities: int, 
                 max_facilities: int,
                 total_communities: int,
                 nb_subproblems: int) -> None:
        assert type(nb_facilities) == int
        assert type(max_facilities) == int
        assert max_facilities <= nb_facilities

        self.nb_facilities = nb_facilities
        self.facilities = np.arange(nb_facilities)
        self.max_facilities = max_facilities

        assert type(total_communities) == int
        assert type(nb_subproblems) == int
        self.total_communities = total_communities
        self.nb_subproblems = nb_subproblems
        int_subproblems = np.arange(nb_subproblems)
        self.subproblems = int_subproblems.astype(str)

        # divide communities among the subproblems
        self.nb_communities = {subproblem_nb: 0 for subproblem_nb in self.subproblems}
        num_communities_per_subproblem = np.floor(self.total_communities / self.nb_subproblems)
        num_residual_communities = self.total_communities - num_communities_per_subproblem * self.nb_subproblems

        # generate all random costs (random number generators are a pain)
        self.generate_cost_data()

        # partition all costs into subproblem specific data
        self.cost = {}
        start_index = 0
        end_index = 0
        for subproblem_nb in self.subproblems:
            if (int(subproblem_nb) <= (num_residual_communities-1)):
                self.nb_communities[subproblem_nb] = int(num_communities_per_subproblem + 1)
            else:
                self.nb_communities[subproblem_nb] = int(num_communities_per_subproblem)

            end_index += self.nb_communities[subproblem_nb]
            self.cost[subproblem_nb] = self.all_costs[:, start_index:end_index]
            start_index = end_index


    def generate_cost_data(self):
        # generates random distances (i.e., "costs") from each community to each facility
        random.seed(42)
        self.all_costs = np.ndarray(shape = (self.nb_facilities, self.total_communities))
        for facility in np.arange(self.nb_facilities):
            for community in np.arange(self.total_communities):
                self.all_costs[facility][community] = random.uniform(1, 100)


    def pmedian_pyomo_model(self, subproblem_nb: str):        
        assert type(subproblem_nb) == str
        
        communities = np.arange(self.nb_communities[subproblem_nb], dtype=int)
        facilities_and_communities = [(facility, community) \
                                        for facility in self.facilities \
                                            for community in communities]

        m = pyo.ConcreteModel()
        m.x = pyo.Var(self.facilities, 
                      within=pyo.Binary)
        m.y = pyo.Var(self.facilities, communities,
                      within=pyo.Reals,
                      bounds=(0,1))
        m.x_max = self.max_facilities
        
        m.obj = pyo.Objective( expr = sum(self.cost[subproblem_nb][facility][community]*m.y[facility, community] 
                                            for facility in self.facilities for community in communities) )
        
        @m.Constraint()
        def max_facilities_rule(m):
            return sum(m.x[facility] for facility in self.facilities) <= m.x_max
        
        @m.Constraint(communities)
        def assign_community_to_facility_rule(m, community):
            return sum(m.y[facility, community] for facility in self.facilities) == 1
        
        @m.Constraint(facilities_and_communities)
        def only_assign_to_selected_facilities_rule(m, facility, community):
            return m.y[facility, community] <= m.x[facility]
        
        return m