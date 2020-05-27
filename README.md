# A Hybrid Optimization Strategy of Meta-Heuristic and Mathematical Programming Methods for Meshed Power Systems
Grid Planning Benchmark Dataset and Scripts for the Paper "A Hybrid Optimization Strategy Combining Network Expansion Planning and Switching State Optimization"

# Instructions
1. Clone the repository
2. Install pandapower and PowerModels.jl (see 'Installation Instructions')
3. Run the optimization, e.g., with `python run_greedy.py`

# Data
This repository contains these subfolders with the following data:
* "power_system_data" containing the power system benchmark cases included additional line and replacement measures
* "scaled_loadcases" contains the data to reproduce the mathematical programming results

# Main script to run the hybrid optimization method
The script "run_greedy.py" runs the greedy heuristic combined with the PowerModels.jl optimization framework. 
An example call starting a combined optimization of switching measures and line measures for the 'brigande' test case is:

```
python run_greedy.py -grid 'brigande' -kind 'combo' -max_iterations 3 -res_dir './results'
```

See more options with:
```
python run_greedy.py -h
```

If you have trouble using the PowerModels.jl interface, you can just run the script without PowerModels.jl. This gives
you a greedy optimization of switching and line measures. Try:

```
python run_greedy.py -pm ''
```

# Additional script to run the PowerModels.jl Optimization
The repository also contains a script "run_powermodels.py" which runs the PowerModels.jl based results with the pandapower-python to PowerModels.jl-julia interface

The script has several command line options:
 * model (str) - The PowerModels.jl power model e.g. "DCPPowerModel"
 * solver (str) - The solver to use, e.g. "juniper", "gurobi"
 * grid (str) - optional if you only want to calculate one grid, e.g. "brigande"
 * kind (str) - the optimizations to run, e.g. "tnep,ots,repl". Can only be a part of these like "tnep,repl"

Example to run the model "DCPPowerModel" with "gurobi" as a solver for the grid "brigande" with the time series loadcases "ts" for the REPl "repl" problem:

```
python-jl run_powermodels.py --model="DCPPowerModel" --solver="gurobi" --grid="brigande" --kind="repl"
```

# Installation instructions
You need the following software to run the script:
* pandapower: https://github.com/e2nIEE/pandapower
* PowerModels.jl: https://github.com/lanl-ansi/PowerModels.jl/
* tqdm: https://github.com/tqdm/tqdm
* numpy
* pandas

The python-julia interface might require some additional installations. See the following link for details:
https://pandapower.readthedocs.io/en/latest/opf/powermodels.html
