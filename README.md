# Introduction to the repository
The purpose of this repo is to share code and results from simulation studies done for the project. 
The code will not be runnable as is, some paths are hard coded among other potential issues.
The meaning behind including the code is to show how the implementation was done, and can maybe help in understanding the algorithm implementation.

However if you want to try and run some of the code, we included requirements in `./requirements.txt`. 
To use the exact solution approach, it is a prerequisite that IBM ILOG CPLEX is installed (this is paid software, although there is a free academic tier).

# Code
The actual implementation of the map generator and solution methods lies in the `sar_moe8` directory (which is structured as a Python package).
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

# Demonstration
In this section a small example will demonstrate how to use map generation and solution methods.

First of all we need to import the functions and classes from the sar_moe8 package.
```python
from sar_moe8.map import MapGenerator, Map
from sar_moe8.solution import Greedy, HillClimbing, GRASP
```
Having done this we can now generate a map instance to work on in this demonstration, we choose to generate a quadratic map of size 20. This generated map object has methods to plot both the heatmap as well as a network representation. Lastly it has a method to save it in the map directory.
```python
map_inst = MapGenerator(map_dim=(20,20))
map_inst.plot()
map_inst.plot_flows(nghbr_lvl=1, show_nodes=True, plot_base=False)
map_inst.save()
```
Heat map             | Network Representation
:-------------------:|:------------------:
![map_plot](/demonstration/map_plot.png) | ![flow_plot](/demonstration/flow_plot.png)

After the generated map instance have been saved we can load it in as a Map object with the Map class. Each generated map will have an associated id, so if you are redoing this demonstration you will have to change the id to your specific one. Having loaded the map instance we can then proceed to solve it using the greedy approach. The greedy solution object has a method to plot the solution. The color of the line correspond to each specific UAV.
```python
map_inst = Map('maps/1c91fb') # change to the generated map instance
greedy = Greedy(map_inst, min_score=1, nghbr_lvl=1)
greedy.solve(L=75, num_vehicles = 2, use_centroids=True, rcl=.9)
greedy.plot(single_ax=True)
```
![greedy_solution](/demonstration/greedy_solution.png)

From the greedy solution we can go on and use the hillclimbing approach to improve on this, which yield the following paths.
```python
hillclimb = HillClimbing(map_inst, greedy, min_score=1, nghbr_lvl=1)
hillclimb.solve(L=75)
hillclimb.plot(single_ax=True)
```
![hillclimbing_solution](/demonstration/hillclimbing_solution.png)

Lastly we can use the GRASP algorithm with 10 iterations and plot the paths generated for each UAV.
```python
grasp = GRASP(
    map_inst,
    n_iter = 10, 
    rcl=.9, 
    nghbr_lvl=1, 
    num_vehicles=2, 
    L=75,
    use_centroids=True
)
grasp.plot(single_ax=True)
```
![grasp_solution](/demonstration/grasp_solution.png)

The full demonstration script is also available at `./demonstration.py`.

# Results
There are two files in the results directory:
- `factorial-design-best-after-5.csv`
  - This file includes the best cumulative score obtained after running GRASP for 5 minutes per map_id and parameter combination.
- `parameter_tuning.csv`
  - This file includes a row for each time the cumulative score improved per problem feature combination (map_id - min_score - num_uavs - range) and parameter combination (rcl - nghbr_lvl - use_centroids - tsp_operator). 
