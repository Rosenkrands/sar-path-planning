# Introduction to the repository
The purpose of this repo is to share code and results from simulation stuides done for the project. 
The code will not be runnable as is, some paths are hard coded among other potential issues. 
The meaning behind including the code is to show how the implementation was done, and can maybe help in understanding the algorithm implementation.

# Code
The actual implementation of the map generator and solution methods are lies in the `sar_moe8` directory (which is structured as a Python package).
We took an Object-Oriented approach to the implementation, the structure of the package is as follows:
- `map.py` Includes two classes related to generating and instantiating map instances
  - `class MapGenerator` Generate a score map and save the scores, target locations and difficulty, as `txt` files, in the `maps` directory.
  - `class Map` Loads a map instance from the `maps` directory and includes methods to generate the network structure, plotting a heatmap etc.
- `solution.py` Includes classes related to generating solutions using different methods.
  - `class Solution` is a base class from which the different solution approaches will inherit, it includes various plotting methods for visualizing the solutions found.
  - `Class Cplex` the exact solution methods based on the IBM ILOG CPLEX Python API.
  - `Class Greedy` the greedy heuristic.
  - `Class HillClimbing` improves an existing solution using different operators in a hillclimbing fashion.
  - `Function GRASP` performs the Greedy Randomized Adaptive Search Procedure for a fixed number of iterations.
  - `Function GRASP_single_iter` performs a single GRASP iteration.
- `utils.py` Includes various functions used printing a command-line progress bar etc.

# Results
There are two files in the results directory:
- `factorial-design-best-after-5.csv`
  - This file includes the best cumulative score obtained after running GRASP for 5 minutes per map_id and parameter combination.
- `parameter_tuning.csv`
  - This file includes a row for each time the cumulative score improved per problem feature combination (map_id - min_score - num_uavs - range) and parameter combination (rcl - nghbr_lvl - use_centroids - tsp_operator). 
