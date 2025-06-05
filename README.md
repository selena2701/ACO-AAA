# ACO-AAA: Ant Colony Optimization for Resource-Constrained Project Scheduling

## Overview
ACO-AAA is a Python implementation of an Ant Colony Optimization (ACO) algorithm for solving the Resource-Constrained Project Scheduling Problem (RCPSP). The RCPSP is a classic combinatorial optimization problem where the goal is to schedule project activities subject to precedence and resource constraints, minimizing the overall project duration (makespan).

This project supports reading standard .sm project files, running parameterized ACO experiments, and exporting results for analysis.

## Features
- Parses standard RCPSP .sm files for project and resource data
- Implements a flexible ACO metaheuristic with customizable parameters
- Supports parameter grid search and experiment automation
- Outputs results as CSV for further analysis
- Includes visualization utilities for schedules

## Directory Structure
```
ACO-AAA/
├── ACO_RCPSP.py         # Main ACO algorithm and experiment logic
├── RunExperiment.py     # Script to run parameter grid search experiments
├── ReadSMFiles.py       # Parser for .sm input files
├── Resource.py          # Resource class definition
├── Schedule.py          # Schedule representation and utilities
├── Task.py              # Task/job class definition
├── j60.sm/              # Example input files in .sm format
├── results/             # Output CSVs from experiments
└── README.md            # This file
```

## Getting Started

### Prerequisites
- Python 3.7+
- Required packages: `numpy`, `pandas`, `matplotlib`, `networkx`, `seaborn`

Install dependencies with:
```bash
pip install numpy pandas matplotlib networkx seaborn
```

### Input Format
The project uses standard RCPSP .sm files as input. Example files are provided in the `j60.sm/` directory. These files contain sections for resources, project information, precedence relations, task durations, and resource availabilities.

#### Example `.sm` File Structure
```
RESOURCES
  - renewable                 :  4   R
  - nonrenewable              :  0   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     60      0       77       50       77
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   ...
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  R 3  R 4
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  ...
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  R 3  R 4
   13   11   12   13
************************************************************************
```

### Running Experiments

To run a parameter grid search and save results:
```bash
python RunExperiment.py
```
This will:
- Parse the default input file (e.g., `j60.sm/j6025_1.sm`)
- Run the ACO algorithm with various parameter combinations
- Save results to CSV files in the `results/` directory

You can modify the input file or parameter ranges in `RunExperiment.py`.

### Main Classes
- `ACO_RCPSP.py`: Contains the `AntColonyRCPSP` class implementing the ACO metaheuristic.
- `ReadSMFiles.py`: Contains `SMFileParser` for parsing .sm files.
- `Task.py`, `Resource.py`, `Schedule.py`: Core data structures for jobs, resources, and schedules.

### Output
Results are saved as CSV files in the `results/` directory, e.g.:
```
results/aco_parameter_search_results_j6025_1.csv
```
Each row contains the parameters and makespan statistics for a run.

## Customization
- To use different input files, change the file path in `ACO_RCPSP.py` or `RunExperiment.py`.
- To adjust ACO parameters, modify the ranges in `RunExperiment.py`.

## Authors
- Nguyễn Đình Phú - 2470891
- Nguyễn Tường Phúc - 2470886
- Phan Văn Nguyên Khánh - 2470883
- Huỳnh Công Minh - 2470890
- Lê Thị Hồng Cúc - 2470882
