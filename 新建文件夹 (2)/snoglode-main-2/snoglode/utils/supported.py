"""
Keeps track of things that are supported for easy access throughout
the scripts.
"""
from enum import Enum, EnumMeta

# frequently want to check if the var type is contained
# use a meta class so we can override the customary contains method
# on an enum s.t. it can check for the presence of a string
class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True    

class SupportedVars(Enum, 
                    metaclass = MetaEnum):
    binary = 'Binary'
    reals = 'Reals'
    integers = 'Integers'
    nonnegative_integers = 'NonNegativeIntegers'