# SNoGloDe: A Structured Nonlinear Global Decomposition Solver

The goal of this package is to provide the framework for a decomposition, global optimization algorithm. 
The backbone of the algorithm relies on a spatial branch and bound tree, with a customizable lower and upper bound problem.
This algorithm is designed for a block-angular decomposable structure (i.e., typically identifiable by complicating variables across subproblems).

For the lower bound problem, a natural relaxation is to take the Extensive Form (EF) and drop the imposed equality constraints on the complicating variables in each subproblems, in essence creating completely individual problems at the lower bound. This is ideal for parallelization.

For the upper bound problem, we first need to generate a candidate solution, $\bar{x}$, for each of the complicating variables. These are then fixed within each of the subproblems (i.e., the equality constraints are already satisified so no need to include them) and can then be solved again in parallel for each of the subproblems.

There are many different methods that can be used to generate candidate solutions. There are also a number of ways to define a lower bounding problem. For this purpose, there are customizable functions that can be defined by the user to construct lower bounds or define candidates given the current spatial branch and bound node. 

If all of the problems (upper and lower disaggregated) are solved to global optimality, then there are convergance guarantees for this algorithm to the globally optimal solution. Without global optimality, we can still search for feasible solutions and attempt to compute gaps on that solution, but there is no guarantee that the gap will close (though, it still might). 

<!-- ## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install snoglode.

```bash
pip install snoglode
```

## Usage -->


## License

[MIT](https://choosealicense.com/licenses/mit/)