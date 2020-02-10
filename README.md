# A Hybrid Optimization Strategy of Meta-Heuristic and Mathematical Programming Methods for Meshed Power Systems
Grid Planning Benchmark Dataset for the Paper "A Hybrid Optimization Strategy of Meta-Heuristic and Mathematical Programming Methods for Meshed Power Systems"

# Data
This repository contains a subfolder with the following data:
* "scaled_loadcases" which contains the data for Chapter VI "Optimization Results"

# Script
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
* PowerModels.jl: https://lanl-ansi.github.io/PowerModels.jl/
* tqdm: https://github.com/tqdm/tqdm
* numpy
* pandas

Also the python-julia interface might require some additional installations. See the following link for details:
https://pandapower.readthedocs.io/en/latest/opf/powermodels.html
