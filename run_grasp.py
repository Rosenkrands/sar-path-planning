import itertools
import multiprocessing as mp
from pandas import DataFrame
from math import floor
import os
import sys
import time
from sar_moe8.map import Map
from sar_moe8.solution import GRASP_single_iter

#path names
maps = os.listdir(os.path.join('.','maps'))
root_dir = os.path.dirname(__file__)
map_dir = os.path.join(root_dir, 'maps')
map_inst = [Map(os.path.join(root_dir, map_dir, path), base_node = 0) for path in maps]

def params_product(dict_of_possible):
    """
    Generates a list of parameters dictionaries to use for solution functions.

    Parameters
    ----------
    dict_of_possible: dict
        Should hold a key for each parameter and the value should be a list of possible
        parameter values.
    """
    comb = list(itertools.product(*dict_of_possible.values()))
    list_of_params = []
    for _ in comb:
        params = {}
        for i, key in enumerate(dict_of_possible.keys()):
            params[key] = _[i]
        list_of_params.append(params)
    return list_of_params

def main(arg):
    start_main = time.time()
    blockPrint()
    try:
        sol = GRASP_single_iter(
            arg['map_inst'], 
            arg['rcl'],
            arg['nghbr_lvl'],
            arg['num_vehicles'],
            arg['L'],
            arg['min_score'],
            arg['use_centroids'],
            arg['initial_tsp']
        )
    except:
        enablePrint()
        print('There was an issue here')
        return 0, None, start_main    
    enablePrint()
    return sol.score_improvement[1], sol, start_main

## main part of the script
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

list_of_params = [{
                'map_inst': inst,
                'rcl': .8,
                'nghbr_lvl': 2,
                'num_vehicles': travel_length[inst.id][0],
                'L': travel_length[inst.id][1],
                'min_score': 1,
                'use_centroids': True,
                'initial_tsp': False
                }
                for inst in map_inst if inst.id in ['8ea3cb', 'cd97cf']
                ]

current_time = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

# Disable Printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore Printing
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':
    results = []
    total_iter = len(list_of_params)*30
    for i, params in enumerate(list_of_params):
##### Number of repetitions for each map ##############
        for j in range(30):
            parameters = list(params.values())
            parameters[0] = parameters[0].id 
            full_greedy = {
                'map_inst': params['map_inst'],
                'rcl': 1,
                'nghbr_lvl': params['nghbr_lvl'],
                'num_vehicles': params['num_vehicles'],
                'L': params['L'],
                'min_score': params['min_score'],
                'use_centroids': params['use_centroids'],
                'initial_tsp': params['initial_tsp']
            }
            params_with_greedy = [full_greedy] 
            params_with_greedy.extend([params]*10000000)
            complete_estimate = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 5*(total_iter-i*30)*60))
            print(j+i*30, 'of', total_iter, 'time is', current_time(),'eta', complete_estimate)
###### number of cores to be used ########################
            pool = mp.Pool(processes=7) 
            try:
                start = time.time()
                best_score = 0
                main_time = None
                for _ in pool.imap_unordered(main, params_with_greedy, chunksize=1):
                    if main_time is None:
                        main_time = _[2] - start
                    if _[0] > best_score:
                        best_score = _[0]
######## 5 minute runtime ##############
                    
                    if time.time() - start + main_time > 5*60:
                        print(time.time() - start)
                        raise RuntimeError
                    # note: here we use runtime from last saved iteration. If we have a solution at 4.45 and at 5.15, the runtime will be 4.45 with the best score at that time.
                    runtime = time.time() - start - main_time
            except RuntimeError:
                print('Time limit reached, terminating the pool')
                pool.terminate()
            pool.close()
            pool.join()

            try:
                results.append(['GRASP', parameters[0], best_score, runtime])
            except:
                results.append(['GRASP', parameters[0], best_score, -1])
            df = DataFrame(
                results,
                columns=['algorithm', 'instance', 'objective', 'runtime']
            )
            df.to_csv('grasp_results_reduced_number.csv', index=False)
            # clean up so the runtime variable do not get reused
            try:
                del runtime
            except:
                pass

    print('All done')
    print('Elapsed time:', time.time() - start)
