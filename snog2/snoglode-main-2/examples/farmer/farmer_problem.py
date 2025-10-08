""""
The classic farmer stochastic programming problem + variants for testing.

    TwoStageFarmer   -> classic, two stage stochastic program where the only RV is yield

"""
import pyomo.environ as pyo

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


""""
A farmer is trying to decide what to plant for the next season.
    - He has wheat (1), corn (2), and sugar beets (3)
    - He can buy (y), sell (w), and produce (x,x_yield) any of these three products.
    - He must buy / produce a min. amount of wheat / corn for his cattle.
    - Quota on sugar beets alters the price after a certain threshold.
Goal: max. the net proficts from purchasing, selling, and planting crops for next season.

However, there is no guarantees on the weather. The actual yield of the crops changes depending on weather -> aka uncertain.
Consider as many weathers as is necessary to accurately represent the probability distribution of the predicted yields.

Case (1) - Integer First Stage
--------------------
Consider the case where the farmer posses four fields of sizes 185, 145, 106, and 65 acres.
Observe - the total of 500 acres is unchanged.
The fields are unfortunately located in different parts of the village. 
For reasons of efficiency, the farmer wants to raise only one type of crop on each field.


Case (2) - Integer Second Stage
--------------------
Consider the case where sales and purhcases of corn and wheat can only be obtained through constracts
involving multiples of hundred tons. 

Case (3) - Both, because why not :)
"""


class MILPFarmer(Farmer):
    """
    Akin to the classic farmer problem.
    Each instances is a realization of the random variable of
    the predicted yield, for only the subsequent year.
    """

    def __init__(self, predicted_yield, integer_first_stage: bool, integer_second_stage: bool):
        """
        Initializes all of the parameters based on the 
        realization of the prediced yield.
        Builds the appropriate scenario model.

        Parameters
        -----------
        predicted_yield : float
            realization of the expected yeild of this scenario
        integer_first_stage : bool
            consider the case of discrete choices of plots.
        integer_second_stage : bool
            consider the case of contract sizes for corn / wheat
        """
        super().__init__(predicted_yield)

        # discrete field sizes
        self.field_sizes = [185, 145, 105, 65]

        # modeling decisions
        self.integer_first_stage = integer_first_stage
        self.integer_second_stage = integer_second_stage

        self._build_model()

    def _build_model(self):
        """
        Builds the LP to represent the random realization of the predicted
        yield paramter for the next year.
        """

        # TODO: two cases!!
                                
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


""""
A farmer is trying to decide what to plant for the next season.
    - He has wheat (1), corn (2), and sugar beets (3)
    - He can buy (y), sell (w), and produce (x,x_yield) any of these three products.
    - He must buy / produce a min. amount of wheat / corn for his cattle.
    - Quota on sugar beets alters the price after a certain threshold.
Goal: max. the net proficts from purchasing, selling, and planting crops for next season.

However, there is no guarantees on the weather. The actual yield of the crops changes depending on weather -> aka uncertain.
Consider as many weathers as is necessary to accurately represent the probability distribution of the predicted yields.

Additionally, the farmer now has access to his barn to store wheat / corn for the following year. 
Assume we start with zero inventory; maximum inventory is 50 bushels of wheat and/or corn.
"""

class TemporalFarmer(Farmer):
    """
    Creates a multi-year planning horizon for the farmer LP.
    Introduces an inventory - a barn is now allowed to store excess corn / wheat
    from the previous year.
    """

    def __init__(self, predicted_yield, start_year, end_year):
        """
        Initializes all of the parameters based on the 
        realization of the prediced yield.
        Assumes a constant storage of 

        Parameters
        -----------
        predicted_yield : float
            realization of the expected yeild of this scenario
        start_year : int
            what year this scenario starts at
        end_year : int
            what year this scenario ends at.
        """
        super().__init__(predicted_yield)

        # add the start/end year
        assert type(start_year)==int
        assert type(end_year)==int
        self.start_year = start_year
        self.end_year = end_year

        # create the indices for the corn/wheat + start/end
        self.storage_index = []
        for year in [self.start_year, self.end_year]:
            for crop in self.purchasing_crops:
                self.storage_index.append( (year, crop) )

        # add the constant storage
        # we can hold a max of 50 units of corn and/or wheat
        self.barn_storage = 50

# TODO: write the model!!
    
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
        
        # barn storage levels for the start / end of this time period
        model.stored_crop_level=pyo.Var(self.storage_index,
                                        within=pyo.Reals,
                                        bounds=(0,self.barn_storage))

        # var for how much of the stored crop was used from storage levels
        model.stored_crop_used=pyo.Var(self.purchasing_crops,
                                       ["sold", "min_requirement"], # did we use it for selling or for meeting min requirement?
                                       within=pyo.Reals,
                                       bounds=(0,self.barn_storage))
        
        # var for how much of the crop was added to storage
        model.stored_crop_added=pyo.Var(self.purchasing_crops, 
                                        ["grown", "bought"], # did we use add it from growth or bought it?
                                        within=pyo.Reals,
                                        bounds=(0,self.barn_storage))

        """ CONSTRAINTS """

        # now, we split objective into first / second stage varialbes for mpi-sppy
        model.planting_cost=sum(model.x[planted_crop]*self.planting_cost[planted_crop] for planted_crop in self.planting_crops)
        model.selling_cost=sum(model.w[sold_crop]*self.selling_price[sold_crop] for sold_crop in self.selling_crops) + \
                                sum(model.stored_crop_used[ (required_crop, "sold") ] for required_crop in self.required_crops)
        model.puchasing_cost=sum(model.y[purchased_crop]*self.purchase_price[purchased_crop] for purchased_crop in self.purchasing_crops) + \
                                    sum(model.stored_crop_add[ (purchase_crop, "bought")] for purchase_crop in self.purchasing_crops)

        model.obj=pyo.Objective( expr= model.planting_cost - model.selling_cost + model.puchasing_cost )

        # total acres allocated cannot exceed total available acreas
        @model.Constraint()
        def total_acreage_allowed(model):
            return ( sum(model.x[planted_crop] for planted_crop in self.planting_crops) <= self.total_acres )

        # must have at least x of wheat,corn
        @model.Constraint(self.required_crops)
        def minimum_requirement(model, required_crop):
            return ( model.x[required_crop]*self.crop_yield[required_crop] + model.y[required_crop] - model.w[required_crop] \
                        + model.stored_crop_used[required_crop] - model.stored_crop_add[required_crop] \
                            >= self.min_requirement[required_crop])
        
        @model.Constraint()
        def sugar_beet_mass_balance(model):
            return ( model.w["beets_favorable"] + model.w["beets_unfavorable"] \
                    <= self.crop_yield["beets"]*model.x["beets"] )

        # the favorably priced beets cannot exceed 6000 (T)
        @model.Constraint()
        def sugar_beet_quota(model):
            return ( model.w["beets_favorable"] <= self.beets_quota )
        
        # ensure that the storage at the start of the time period + whatever is used / added balances at the end of the period!
        @model.Constraint(self.required_crops)
        def crop_storage_balance(model, required_crop):
            return (model.stored_crop_level[self.start_year, required_crop] - model.stored_crop_used[required_crop] \
                        + model.stored_crop_added[required_crop] == model.stored_crop_level[self.end_year, required_crop])
        
        self.model = model