from sar_moe8.map import Map
from sar_moe8.solution import Greedy, HillClimbing, GRASP_single_iter

import os
import sys
import pandas as pd
from math import floor

# Disable Printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore Printing
def enablePrint():
    sys.stdout = sys.__stdout__

maps = os.listdir(os.path.join('.','maps'))
results = []

travel_length = {
    '132167': [3, 300],
    '3e559b': [4, [168, 168, 300, 300]],
    '4fceab': [4, [168, 168, 300, 300]],
    '506fa3': [1, 168],
    '802616': [2, [168]*2],
    '88182c': [3, [168, 168, 300]],
    'cd97cf': [2, 300],
    'f79242': [3, [168]*3],
    '8ea3cb': [1, 300]
}

for map in maps:
    print(f'### current map is {map} ###')
    map_inst = Map(os.path.join('.','maps',map))

    reps = 30
    min_score=1
    nghbr_lvl=2
    use_centroids=True
    num_vehicles=travel_length[map_inst.id][0] #floor(len(map_inst.map.columns.values)**(1/2)) - 3
    L=travel_length[map_inst.id][1]

    # Finding solutions

    # Greedy algorithm & Hill Climbing
    for i in range(reps):
        print(i+1)
        # Greedy
        print('Started on the greedy solution')
        greedy = Greedy(map_inst, min_score=min_score, nghbr_lvl=nghbr_lvl)
        blockPrint()
        greedy.solve(L=L, num_vehicles=num_vehicles,use_centroids=use_centroids,rcl=1)
        enablePrint()
        results.append(['Greedy', map, greedy.score, greedy.time_spent])

        # Hill Climbing
        print('Started on the hillclimb solution')
        hillclimbing = HillClimbing(map_inst, initial_solution=greedy, min_score=min_score, nghbr_lvl=nghbr_lvl)
        blockPrint()
        hillclimbing.solve(L=L)
        enablePrint()
        results.append(['Hillclimbing', map, hillclimbing.score_improvement[1], hillclimbing.time_spent])

#df1 is greedy and hc, df2 is grasp.
df1 = pd.DataFrame(results, columns=['algorithm', 'instance', 'objective', 'runtime'])
df2 = pd.read_csv('grasp_results.csv')
#df2

df = df1.append(df2, ignore_index=True)

df.groupby(['algorithm', 'instance'])['objective', 'runtime'].mean()
df.groupby(['algorithm', 'instance'])['objective', 'runtime'].std()

summary = df.groupby(['algorithm', 'instance']).agg(['mean', 'std'])
summary.columns = [' '.join(col).strip() for col in summary.columns.values]
summary.reset_index()
summary.to_csv(os.path.join('.','benchmark_results.csv'))
