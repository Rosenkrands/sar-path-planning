from numpy import single
from sar_moe8.map import MapGenerator, Map
from sar_moe8.solution import Greedy, HillClimbing, GRASP

map_inst = MapGenerator(map_dim=(20,20))
map_inst.plot()
map_inst.plot_flows(nghbr_lvl=1, show_nodes=True, plot_base=False)
map_inst.save()

map_inst = Map('maps/1c91fb') # change to the generated map instance
greedy = Greedy(map_inst, min_score=1, nghbr_lvl=1)
greedy.solve(L=75, num_vehicles = 2, use_centroids=True, rcl=.9)
greedy.plot(single_ax=True)

hillclimb = HillClimbing(map_inst, greedy, min_score=1, nghbr_lvl=1)
hillclimb.solve(L=75)
hillclimb.plot(single_ax=True)

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