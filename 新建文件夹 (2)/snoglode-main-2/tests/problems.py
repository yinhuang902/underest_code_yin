""""
The classic farmer stochastic programming problem + variants for testing.

    TwoStageFarmer   -> classic, two stage stochastic program where the only RV is yield
    BilinearProblem  -> multi-scenario non-convex bilinear program
    IntegerProgram   -> 1 scenario pure IP
    PMedian          -> decomposable, scalable P-median formulation

"""
import snoglode as sno
import pyomo.environ as pyo
import numpy as np
import random 

from snoglode.bounders.upper_bounders import AbstractCandidateGenerator

# this can be used to always fail the UB - helpful when we want to test 
# elements of the algorithm
class MockCandidateGenerator(AbstractCandidateGenerator):
    def __init__(self, 
                 solver, 
                 subproblems: sno.Subproblems, 
                 time_ub: float) -> None:
        super().__init__(solver = solver,
                         subproblems = subproblems,
                         time_ub = time_ub)
        self.ub_required = True
    
    def generate_candidate(self, node, subproblems) -> None:
        return False, {}, None


class Farmer():
    def __init__(self, predicted_yield):
        """
        Base Class for the farmer decisions we want to make.
        
        Parameters
        -----------
        predicted_yield : float
            realization of the expected yeild of this scenario
        """
        assert type(predicted_yield)==float
        assert predicted_yield >= 0

        # predicted_yield = a randomly drawn expected yield
        self.crop_yield={"wheat":2.5*predicted_yield,
                        "corn":3*predicted_yield,
                        "beets":20*predicted_yield}

        # these do not change, regardless of scenario.
        self.total_acres=500
        self.planting_cost={"wheat":150,
                            "corn":230,
                            "beets":260}
        self.planting_crops=["wheat","corn","beets"]
        self.selling_price={"wheat":170,
                            "corn":150,
                            "beets_favorable":36, 
                            "beets_unfavorable":10}
        self.selling_crops=["wheat", "corn", "beets_favorable", "beets_unfavorable"]
        self.min_requirement={"wheat":200,
                              "corn":240}
        self.purchase_price={"wheat":238,
                             "corn":210}
        self.purchasing_crops=["wheat","corn"]
        self.required_crops=self.purchasing_crops
        self.beets_quota=6000

""""
A farmer is trying to decide what to plant for the next season.
    - He has wheat (1), corn (2), and sugar beets (3)
    - He can buy (y), sell (w), and produce (x,x_yield) any of these three products.
    - He must buy / produce a min. amount of wheat / corn for his cattle.
    - Quota on sugar beets alters the price after a certain threshold.
Goal: max. the net proficts from purchasing, selling, and planting crops for next season.

However, there is no guarantees on the weather. The actual yield of the crops changes depending on weather -> aka uncertain.
Consider as many weathers as is necessary to accurately represent the probability distribution of the predicted yields.
"""

class TwoStageFarmer(Farmer):
    """
    Akin to the classic farmer problem.
    Each instances is a realization of the random variable of
    the predicted yield, for only the subsequent year.
    """

    def __init__(self, predicted_yield):
        """
        Initializes all of the parameters based on the 
        realization of the prediced yield.
        Builds the appropriate scenario model.

        Parameters
        -----------
        predicted_yield : float
            realization of the expected yeild of this scenario
        """
        super().__init__(predicted_yield)
        self._build_model()

    def _build_model(self):
        """
        Builds the LP to represent the random realization of the predicted
        yield paramter for the next year.
        """
                        
        # create pyomo model
        model = pyo.ConcreteModel()

        """ VARIABLES """

        # land variables [=] acres of land devoted to each crop
        model.x=pyo.Var(self.planting_crops, 
                        within=pyo.Reals,
                        bounds=(0,500))

        # selling decision variables [=] tons of crop sold
        model.w=pyo.Var(self.selling_crops, 
                        within=pyo.Reals,
                        bounds=(0,10000))

        # purchasing decision variables [=] tons of crop purchased
        model.y=pyo.Var(self.purchasing_crops, 
                        within=pyo.Reals,
                        bounds=(0,10000))

        """ CONSTRAINTS """

        # now, we split objective into first / second stage varialbes for mpi-sppy
        model.planting_cost=sum(model.x[planted_crop]*self.planting_cost[planted_crop] for planted_crop in self.planting_crops)
        model.selling_cost=sum(model.w[sold_crop]*self.selling_price[sold_crop] for sold_crop in self.selling_crops)
        model.puchasing_cost=sum(model.y[purchased_crop]*self.purchase_price[purchased_crop] for purchased_crop in self.purchasing_crops)

        model.obj=pyo.Objective( expr= model.planting_cost - model.selling_cost + model.puchasing_cost )

        # total acres allocated cannot exceed total available acreas
        @model.Constraint()
        def total_acreage_allowed(model):
            return ( sum(model.x[planted_crop] for planted_crop in self.planting_crops) <= self.total_acres )

        # must have at least x of wheat,corn
        @model.Constraint(self.required_crops)
        def minimum_requirement(model, required_crop):
            return ( model.x[required_crop]*self.crop_yield[required_crop] + model.y[required_crop] - model.w[required_crop] \
                       >= self.min_requirement[required_crop])
        
        @model.Constraint()
        def sugar_beet_mass_balance(model):
            return ( model.w["beets_favorable"] + model.w["beets_unfavorable"] \
                    <= self.crop_yield["beets"]*model.x["beets"] )

        # the favorably priced beets cannot exceed 6000 (T)
        @model.Constraint()
        def sugar_beet_quota(model):
            return ( model.w["beets_favorable"] <= self.beets_quota )
        
        self.model = model


def farmer_classic_subproblem_creator(subproblem_name):
    """
    Based on the scenario, generates 
        1) the pyomo model
        2) the dict of lifted variable IDS : pyo.Var
        3) the list of subproblem specific variables (pyo.Vars)
        3) probability of subproblem
    and returns as a list in this order.
    """
    name_to_yield_map = {
        "good": 1.2,
        "fair": 1.0,
        "bad": 0.8
    }
    
    # create parameters / model stored in obj for this scenario
    farmer_scenario = TwoStageFarmer(name_to_yield_map[subproblem_name])

    # grab the list of first stage variables
    lifted_variable_ids = {("devoted_acrege", crop): farmer_scenario.model.x[crop] \
                                for crop in farmer_scenario.planting_crops}
    
    # probability of this particular scenario occuring
    scenario_probability = 1/3

    return [farmer_scenario.model,              # pyomo model corresponding to this subproblem
            lifted_variable_ids,                # lifted varID : pyo.Var dict
            scenario_probability]               # probability of this subproblem


def farmer_skew_subproblem_creator(scenario_name):
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
        subproblem_specific = "wheat"
    
    # for fair - only need to include wheat and beets
    if scenario_name=="fair":
        crops = ["wheat", "beets"]
        subproblem_specific = "corn"
    
    # for bad - only need to include corn and beets
    if scenario_name=="bad":
        crops = ["corn", "wheat"]
        subproblem_specific = "beets"

    # grab the list of first stage variables
    lifted_variable_ids = {("devoted_acrege", crop): farmer_scenario.model.x[crop] \
                                for crop in crops}
    
    # probability of this particular scenario occuring
    scenario_probability = 1/3

    return [farmer_scenario.model,              # pyomo model corresponding to this subproblem
            lifted_variable_ids,                # lifted varID : pyo.Var dict
            scenario_probability]               # probability of this subproblem

# ===============================================================================================

class BilinearProblem():
    def __init__(self, c, y_ub, y_lb, x_ub, x_lb) -> None:
        assert y_lb <= y_ub
        assert x_lb <= x_ub
        self.c = c
        self.y_ub = y_ub
        self.y_lb = y_lb
        self.x_ub = x_ub
        self.x_lb = x_lb
        self.build_bilinear_model()

    def build_bilinear_model(self) -> None:
        m = pyo.ConcreteModel()
        m.x = pyo.Var(within=pyo.Reals,
                      bounds=(max(0,self.x_lb), self.x_ub))      
        m.y = pyo.Var(within=pyo.Reals,
                      bounds=(max(0,self.y_lb), self.y_ub))
        m.obj = pyo.Objective( expr = ( - m.x - m.y ) )
        @m.Constraint()
        def c1_rule(m):
            return (m.x * m.y <= self.c)
        self.m=m

def bilinear_subproblem_creator(subproblem_name):

    # seed numpy with the problem name for reproducibility
    seed, num_scenarios = subproblem_name.split("_")
    seed = int(seed)
    np.random.seed(seed)

    c = np.random.uniform(5,10)
    x_lb = np.random.uniform(0,3)
    x_ub = np.random.uniform(9,12)
    y_lb = np.random.uniform(0,3)
    y_ub = np.random.uniform(9,12)
    optmodel = BilinearProblem(c=c,
                               y_lb=y_lb,
                               y_ub=y_ub,
                               x_lb=x_lb,
                               x_ub=x_ub)
    model = optmodel.m
    
    lifted_variable_ids = {"x": model.x}
    subproblem_probability = 1 / int(num_scenarios)

    return [model,                              # pyomo model corresponding to this subproblem
            lifted_variable_ids,                # lifted varID : pyo.Var dict
            subproblem_probability]             # probability of this subproblem

# ===============================================================================================

class IntegerProgram():
    
    def __init__(self) -> None:
        pass

    def ip_pyomo_model(self):
        """
        Builds basic Integer program
        See: Ignacaio's textbook, pg. 78 for the model and depiction of the 
            branch and bound tree
        """
        m = pyo.ConcreteModel()
        i = [1,2,3]
        m.y = pyo.Var(i,
                    within=pyo.Binary)

        m.obj=pyo.Objective( expr = (m.y[1] + 2*m.y[2] + 4*m.y[3]), 
                            sense=pyo.minimize )
        
        def c1_rule(m):
            return ( sum(m.y[ind] for ind in i) >= 1 )
        m.c1=pyo.Constraint( expr = c1_rule )

        def c2_rule(m):
            return ( m.y[1] - m.y[2] - m.y[3] <= 0 )
        m.c2=pyo.Constraint( expr = c2_rule )

        def c3_rule(m):
            return ( m.y[1] >= m.y[2] + 0.2 )
        m.c3=pyo.Constraint( expr = c3_rule )
        return m


    def ip_subproblem_creator(self, subproblem_name):

        """
        Based on the scenario, generates 
            1) the pyomo model
            2) the list of first stage variables
            3) probability
        and returns as a list in this order.
        """
        # create parameters / model stored in obj for this scenario
        model = self.ip_pyomo_model()

        # grab the list of first stage variables
        lifted_variable_ids = {("y", i): model.y[i] \
                                    for i in [1,2,3]}
        self.lifted_variable_ids = lifted_variable_ids

        # probability of this particular scenario occuring
        scenario_probability = 1.0

        return [model,                              # pyomo model corresponding to this subproblem
                lifted_variable_ids,                # lifted varID : pyo.Var dict
                scenario_probability]               # probability of this subproblem
    
# ===============================================================================================

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


def pmedian_subproblem_creator(subproblem_name):

    """
    Based on the scenario, generates 
        1) the pyomo model
        2) the list of first stage variables
        3) probability
    and returns as a list in this order.
    """
    # unpack
    _,nb_facilities,_,max_facilities,_,total_communities,_,nb_subproblems,_,subproblem_nb = subproblem_name.split("_")
    
    # convert from str -> proper dtype
    nb_facilities = int(nb_facilities)
    max_facilities = int(max_facilities)
    total_communities = int(total_communities)
    nb_subproblems = int(nb_subproblems)

    # create parameters / model stored in obj for this scenario
    pmedian = PMedian(nb_facilities=nb_facilities,
                      max_facilities=max_facilities,
                      total_communities=total_communities,
                      nb_subproblems=nb_subproblems)
    model = pmedian.pmedian_pyomo_model(subproblem_nb)

    # grab the list of first stage variables
    lifted_variable_ids = {("facility", i): model.x[i] \
                                for i in pmedian.facilities}

    # probability of this particular scenario occuring
    scenario_probability = 1.0

    return [model,                              # pyomo model corresponding to this subproblem
            lifted_variable_ids,                # lifted varID : pyo.Var dict
            scenario_probability]               # probability of this subproblem

# ===============================================================================================

def continuous_1var_subproblem_creator(_):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(domain=pyo.Reals,
                  bounds=(0,1))
    m.obj = pyo.Objective(expr=m.x)
    return [m, {"x": m.x}, 1]


def integer_knapsack_subproblem_creator(_):
    # create parameters / model stored in obj for this scenario
    items = [1,2]
    profits = {1: 5, 
               2: 3}
    weights = {1: 2, 
               2: 8}
    capacity = 5        # knapsack capacity

    # build the model & variables
    m = pyo.ConcreteModel()
    m.available_items = pyo.Set(initialize=items)
    m.x = pyo.Var(items, 
                  domain=pyo.NonNegativeIntegers,
                  bounds=(0,2))   # can select at most two of each item
    
    # min -profit = max profit
    m.obj=pyo.Objective( expr = -sum(profits[i]*m.x[i] for i in m.available_items))

    # knapsack capacity constraint
    m.capacity = pyo.Constraint(expr=sum(weights[i] * m.x[i] for i in m.available_items) <= capacity)

    # grab the list of first stage variables
    lifted_variable_ids = {("item", i): m.x[i] \
                                for i in m.available_items}

    # probability of this particular scenario occuring
    scenario_probability = 1.0

    return [m,                                  # pyomo model corresponding to this subproblem
            lifted_variable_ids,                # lifted varID : pyo.Var dict
            scenario_probability]               # probability of this subproblem