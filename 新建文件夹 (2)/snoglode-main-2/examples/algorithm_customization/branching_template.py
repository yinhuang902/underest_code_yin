"""
Write your own variable selection method & variable partition method.
for selection: must be passed to the snoglode.solver.SNoGloDe initilization as the selection_strategy arguement.
for partition: must be passed to the snoglode.solver.SNoGloDe initilization as the partition_strategy arguement.

NOTE: this template is specifically formatted to be plugged in seamlessly.
      in particular, inheritance, function, parameters, and returns.
      Everything else is customizable.
"""
import math
import snoglode as sno

class SomeSelectionStrategy(sno.SelectionStrategy):
    def __init__(self) -> None:
        super().__init__()

    def select_variable(self, 
                        node: sno.Node, 
                        subproblems: sno.Subproblems) -> list:

        # ---- do something to select a variable ------

        var_type = "some var type"
        var_name = "some lifted var name"

        return var_type, var_name

class SomeParititionStrategy(sno.PartitionStrategy):
    def __init__(self) -> None:
        super().__init__()

    def split_point(self, 
                    var_lb: float, 
                    var_ub: float) -> float:
        
        # ---- do something to select a split point within the domain ------
        return math.inf