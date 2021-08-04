from sar_moe8.map import Map
from sar_moe8.solution import Greedy, HillClimbing, GRASP_single_iter

import os
import pandas as pd

maps = os.listdir(os.path.join('.','maps'))
results = []

for map in maps:
    print(f'### current map is {map} ###')
    map_inst = Map(os.path.join('.','maps',map))

    reps = 10
    min_score=1
    nghbr_lvl=2
    use_centroids=False
    L=84
    num_vehicles=2

    # Finding solutions

    # Greedy algorithm & Hill Climbing
    for i in range(reps):
        # Greedy
        greedy = Greedy(map_inst, min_score=min_score, nghbr_lvl=nghbr_lvl)
        greedy.solve(L=L, num_vehicles=num_vehicles,use_centroids=use_centroids,rcl=1)
        results.append(['Greedy', map, greedy.score, greedy.time_spent])

        # Hill Climbing
        hillclimbing = HillClimbing(map_inst, initial_solution=greedy, min_score=min_score, nghbr_lvl=nghbr_lvl)
        hillclimbing.solve(L=L)
        results.append(['Hillclimbing', map, hillclimbing.score_improvement[1], 'Missing time spent'])
    
    # GRASP


df = pd.DataFrame(results, columns=['algorithm', 'instance', 'objective', 'runtime'])
df.groupby(['algorithm', 'instance'])['objective', 'runtime'].mean()
df.groupby(['algorithm', 'instance'])['objective', 'runtime'].std()

summary = df.groupby(['algorithm', 'instance']).agg(['mean', 'std'])
summary.columns = [' '.join(col).strip() for col in summary.columns.values]
summary.reset_index()
