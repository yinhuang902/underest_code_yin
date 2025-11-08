# PID Simplex Optimization Framework

This project implements a **3D Simplex-based Interpolation algorithm** for PID controller tuning across multiple scenarios.  
It adaptively explores the PID stochastic programming first stage variables (Kp, Ki, Kd) using **simplex(tetrahedral) decomposition** and model-based subproblems solved efficiently via **Pyomo + Gurobi**.

---

## Project Structure

| File | Description |
|------|--------------|
| `app.ipynb` | Example Jupyter Notebook to run experiments and visualize iteration progress. |
| `modeling.py` | Load scenarios from CSV and define Pyomo PID models. |
| `bundles.py` | Wraps Pyomo models in persistent Gurobi solvers (`BaseBundle`, `MSBundle`) to enable efficient re-solving in main loop. |
| `simplex.py` | Core 3D simplex search algorithm (`run_pid_simplex_3d`) using Delaunay triangulation and ms subproblems. |
| `utils.py` | Printing, and visualization utilities (e.g. `SimplexTracker`, Plotly 3D plots). |
| `data.csv` | Generated input data to create PID model for different scenarios. |
| `environment-mac.yml` | Conda environment file for mac. |
| `environment-win.yml` | Conda environment file for win. |


##  Important Functions

### simplex.py
- `run_pid_simplex_3d(...)`  
  Main loop — performs adaptive simplex subdivision. Including the policy to rank and choose candidate points.

- `evaluate_all_tetra(...)`  
  Evaluate all simplices formed by the current node set, for all scenarios, and compute their ms, bounds, and candidate points.

