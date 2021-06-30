from math import ceil
from dijkstar.algorithm import NoPathError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from matplotlib.collections import LineCollection
from matplotlib import colors
import time

if os.path.basename(os.getcwd()) == 'sar_moe8':
    os.chdir('../')

from sar_moe8.map import Map
from sar_moe8.utils import PrintProgressBar

from functools import reduce
from docplex.mp.model import Model
from dijkstar import Graph, find_path

# Timing
from pytictoc import TicToc

WORK_DIR = '//math.aau.dk/ProjectGroups/f21moe8or'
MAP_DIR = os.path.join(WORK_DIR, 'maps')

######################## helper functions ########################

split = lambda x: x.name.split('_') if hasattr(x, 'name') else x.split('_')
int_str = lambda string: string if (string == 'end') | (string == 'start') | (string  == 'base') else int(string)

def _dijkstra(A, lengths, from_node, to_node, blacklist = [], graph_object = None):
    ''' 
    Calculates the shortest path between two nodes using Dijkstra.
    TODO KGN: make a custom memoize function for a function that makes the graph. can't use the current as A and lengths are dicts and memoize cannot take dicts as arguments.
                idea can be to read only the values from A and lengths in the memoization.
    '''
    if graph_object is None:
        # create graph if none provided
        graph = Graph() #directed by default
        for arc_id in A:
            tail = A[arc_id][0]
            head = A[arc_id][1]
            graph.add_edge(tail, head, {'arc_id': arc_id, 'cost': lengths[int_str(arc_id)]})
    else:
        graph = copy.deepcopy(graph_object)

    if len(blacklist) == 0:
        pass
        # print('Recived no blacklist nodes')
    else:
        # print(f'Recieved the following blacklist {blacklist}')
        for node in list(set(blacklist)):
            if (node == from_node) or (node == to_node):
                continue
            graph.remove_node(node)

    my_costs = lambda u, v, edge, prev_edge: edge['cost']
    path = find_path(graph, from_node, to_node, cost_func = my_costs)
    # path object returns: nodes, edges, costs and total_cost
    nodes = path[0]
    arcs = path[1]
    length = path[3]
    return nodes, arcs, length

######################## solution classes ########################
_colors = ['#00008B','#008B8B','#B8860B','#bb30c2','#009600','#A9A9A9','#BDB76B','#8B008B','#556B2F','#FF8C00','#9932CC','#8B0000','#E9967A','#8FBC8F','#483D8B']

class Solution():
    """
    A base class for all solution types considered.
    This class should never be used explicitly, only inherited from.
    """
    def __init__(self, map_inst, min_score, nghbr_lvl=0):
        """
        An object of this class must be based on an object of the Map class.

        parameters
        ----------
        map_inst : Map object

        min_score : int
            a lower bound for the score of included vertices.
        
        nghbr_lvl : int 
            The number of neighbors that should be included from the Delaunay triangulation.
            If 0, no neighbors are included, if 1 only the first neighbors are included etc.
        
        start_node : {int, None}
            index of the node that should be connected with start_node.
            Set to None if no start_node.
        
        end_node : {int, None}
            index of the node that should be connected with end_node.
            Set to None if no end_node.
            If end_node = -1, it will connect to the last node of the network.
        """
        self.base_node = map_inst.base_node

        self.start_node = 'base'
        self.end_node = 'base'
        self.add_start = True
        self.add_end = True

        # attributes that comes from the map instance, only for internal use
        # if needed they should be accessed through the map instance
        self._map_inst = map_inst
        self._map = map_inst.map
        self._scores = map_inst.scores(min_score=min_score, add_start = self.add_start, add_end = self.add_end)
        self._targets = map_inst.targets
        self._target_dict = map_inst.nodes(type='targets', add_start=False, add_end=False)
        self._nodes = map_inst.nodes(min_score=min_score, add_start=self.add_start, add_end=self.add_end)
        self._A = map_inst.A(min_score=min_score, nghbr_lvl=nghbr_lvl)
        self._lengths = map_inst.lengths(min_score=min_score, nghbr_lvl=nghbr_lvl)
        self._delta_minus, self._delta_plus, self._delta_minus_nodes, self._delta_plus_nodes = map_inst.delta(min_score=min_score, nghbr_lvl=nghbr_lvl)
        self._min_score = min_score
        self.nghbr_lvl = nghbr_lvl
        # if self.end_node == -1:
        #     self.end_node = len(map_inst.nodes(min_score=min_score, add_start = False, add_end = False)) - 1

        # attributes specific to the solution
        # vertices is mainly used to calculate the objective of the route
        self.vertices = None
        # path is mainly used to plot the active arcs for each vehicle
        self.path = None
        # used for plotting
        self.num_vehicles = None
        # stores the calculated score
        self.score = None
    
    def plot(self, nrows=1, size=5, nodes=False, fig_out=False, single_ax=False, plot_base=False):
        """
        Plots the entire graph in light gray with the solution overlayed in dark color.
        If there are multiple vehicles there will be added extra columns for each vehicle.
        UPDATE: You are now able to specify the number of rows you want in the plot.
        """
        print('Constructing the visualization...')
        # remove 'x_start' and 'x_end' as we cannot plot these
        path = {vehicle_id: [x for x in self.path[vehicle_id] if split(x)[1] not in ['start','end']] for vehicle_id in range(self.num_vehicles)}
        if nrows == 1:
            ncols = self.num_vehicles
        else:
            ncols = ceil(self.num_vehicles/nrows)

        if single_ax == True:
            nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize = (ncols*size, nrows*size))
        
        # If there is only one vehicle ax must be a list
        if not isinstance(ax, np.ndarray):
            ax = [ax]

        def plot_arcs_background(ax):
            # plots all arcs in light gray color
            points = list(self._nodes.values())
            if self.add_end:
                points.pop(-2)
            if self.add_start:
                points.pop(-2)
            index_max = len(points[:-1]) - 1
            
            A = self._A.copy()
            A.pop('end')
            A.pop('start')
            
            blacklist = []
            for arc in A:
                if type(A[arc][1]) == str:
                    if A[arc][1] == 'base':
                        blacklist.append(arc)
                        continue
            
            A_base = {}
            for arc in blacklist:
                if plot_base:
                    A_base[arc] = A[arc]
                del A[arc]

            if plot_base:
                for arc in A_base:
                    # print(arc)
                    # self.A_base = A_base
                    if type(A_base[arc][0]) == str:
                        first = index_max + 1 # int(A_base[arc][0][-1])
                        second = A_base[arc][1]
                        A_base[arc] = (first, second)
                    if type(A_base[arc][1]) == str:
                        first = A_base[arc][0]
                        second = index_max + 1 #int(A_base[arc][1][-1]) +  
                        A_base[arc] = (first, second)

            for arc in A: 
                if type(A[arc][0]) == str:
                    first = index_max + 1 # int(A[arc][0][-1]) + 
                    second = A[arc][1]
                    A[arc] = (first, second)
                if type(A[arc][1]) == str:
                    first = A[arc][0]
                    second = index_max + 1 # int(A[arc][1][-1]) + 
                    A[arc] = (first, second)

            arcs = list(A.values())
            
            #remove start and end points
            # arcs = [item for item in arcs if (item[0] != 'start') & (item[1] != 'end')]
            points = np.array(points)

            points_swapxy = np.array([point[::-1] for point in points])
            arc_collection = LineCollection(points_swapxy[arcs], color='0.9', zorder=0)
            ax.add_collection(arc_collection)

        def plot_arcs_path(ax, path, vehicle_id):
            # plot the arcs used by vehicle
            for _ in path:
                arc = split(_)[1]
                arc = int(arc)
                temp_y = [self._nodes[self._A[arc][0]][0], self._nodes[self._A[arc][1]][0]]
                temp_x = [self._nodes[self._A[arc][0]][1], self._nodes[self._A[arc][1]][1]]
                ax.plot(temp_x, temp_y, color=_colors[vehicle_id], zorder=5)

        def plot_points(ax, nodes = False):
            # plot each vertex as a black dot 
            points = list(self._nodes.values())
            # removing the last two entries in nodes if these are 'start' and 'stop'
            if self.add_end:
                points.pop(-2)
            if self.add_start:
                points.pop(-2)
            points = np.array(points)

            if nodes:
                ax.scatter(points.T[1], points.T[0],marker='.', linewidths=1, color='black', zorder=10)
            
            # find vertices that include targets and plot these in red color
            target_marker = ['.' if self._targets.iloc[point[0], point[1]] == 0 else 'x' for point in points]
            target_ids = [i for i in range(len(points)) if target_marker[i] == 'x']
            ax.scatter(points[target_ids].T[1], points[target_ids].T[0],color='red',marker='.', linewidths=1, zorder=15)
        
        if (nrows == 1) & (ncols == 1):
            plot_arcs_background(ax[0])
            plot_points(ax[0], nodes = nodes)
            for vehicle_id in range(self.num_vehicles):
                plot_arcs_path(ax[0], path[vehicle_id], vehicle_id)
            ax[0].invert_yaxis()
            # ax[0].axis('off')

        elif nrows == 1 or ncols == 1:
            for vehicle_id in range(self.num_vehicles):
                plot_arcs_background(ax[vehicle_id])
                plot_arcs_path(ax[vehicle_id], path[vehicle_id], vehicle_id)
                plot_points(ax[vehicle_id], nodes = nodes)
    
                # invert the y_axis to match the plot method for the Map class
                ax[vehicle_id].invert_yaxis()
                # ax[vehicle_id].axis('off')
        else:
            vehicle_id = 0
            for i in range(nrows):
                for j in range(ncols):
                    if vehicle_id < len(path.keys()):
                        plot_arcs_background(ax[i][j])
                        plot_arcs_path(ax[i][j], path[vehicle_id], vehicle_id)
                        plot_points(ax[i][j], nodes = nodes)
                        ax[i][j].invert_yaxis()
                    # ax[i][j].axis('off')
                    vehicle_id += 1
        if fig_out:
            return fig

    def get_located_targets(self):
        '''
        Method makes a dictionary of located targets with the given solution.
        Requires a solution method has been run first.

        returns
        -------
        Does not return an object but adds the attribute self.located_targets.
        '''
        if self.vertices != None:
            # heuristic solution attribute
            self.visited_nodes = [node for _ in self.vertices.values() for node in _]
        elif self.nodes_all != None:
            # cplex solution attribute
            self.visited_nodes = list(self.nodes_all.keys())
        else: print('no object of solution nodes found.')

        visited_points = [list(self._nodes[int(node.split('_')[1])]) for node in self.visited_nodes if node.split('_')[1] not in ['start', 'end', 'base']]
        self.located_targets = {}
        for target_id in self._target_dict:
            if list(self._target_dict[target_id]) in visited_points:
                self.located_targets[target_id] = self._target_dict[target_id]

class Cplex(Solution):
    """
    Use the docplex library to find an exact solution.
    
    TODO: KGN: It seems like the algorithm is a lot slower when setting other start- or end-nodes, 
    but I cannot figure out why.
    Are these because of the new subtour constraint or something else?
    """
    def __init__(self, map_inst, min_score, nghbr_lvl = 0):
        super().__init__(map_inst, min_score, nghbr_lvl)

    def solve(self, L, num_vehicles, timelimit = None, scale_obj = 1):
        """
        Populates self.path and self.num_vehicles with the path obtained from cplex.
        NOTE: It is also possible to return the Model object if more information is needed. 

        parameters
        ----------
        L : int or list
            Travel capacity of vehicles. If list should have len = num_vehicles. and the values can differ
            If only an int is supplied, it will be transformed to the list [L]*num_vehicles
        """

        assert (self.start_node != None) & (self.end_node != None), "start- or end-node not supplied. Algorithm not implemented without these."
        if type(L) == int:
            L = [L]*num_vehicles

        scores = self._map
        targets = self._targets
        nodes = self._nodes
        A = self._A
        lengths = self._lengths
        delta_minus = self._delta_minus
        delta_plus = self._delta_plus
        if (len(delta_plus[self.start_node]) <= num_vehicles) and (len(delta_minus[self.end_node]) <= num_vehicles):
            num_vehicles = min(delta_plus[self.start_node], len(delta_minus[self.end_node]))
            print(f'The number of vehicles exceeds the arc constraint, num_vehicles is: {num_vehicles}')
        vehicles = range(num_vehicles)

        # def powerset(iterable):
        #     s = list(iterable)
        #     return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        
        mdl = Model(name='cplex-solution')

        for k in vehicles:
            mdl.binary_var_dict(A, name = f'x_{k}')
            mdl.continuous_var_dict(nodes, name = f'u_{k}')
        mdl.binary_var_dict(nodes, name = 'y')

        x_vars = [var for var in mdl.iter_binary_vars() if 'x' in var.name]
        y_vars = [var for var in mdl.iter_binary_vars() if 'y' in var.name]
        u_vars = [var for var in mdl.iter_continuous_vars() if 'u' in var.name]

        # Small functions
        vehicle = lambda x: x.name.split('_')[1]
        arc = lambda x: x.name.split('_')[2]
        node = lambda y: y.name.split('_')[1]
        # int_str = lambda string: string if (string == 'end') | (string == 'start') else int(string)

        print('Adding node constraints:')
        total_iterations = len(nodes)
        iteration = 0
        PrintProgressBar(0, total_iterations, prefix = 'Progress:', suffix = 'Complete', length = 25)
        for node_id in nodes:
            if (node_id != 'start') & (node_id != 'end'):
            # if node_id != 'start':
                # store arcs in delta^- for vertex_id
                a = [str(arc_id) for arc_id in delta_minus[node_id]]
                # find x variables that correspond to the arcs in delta^- for vertex_id
                x_vars_a = [x_var for x_var in x_vars if arc(x_var) in a]
                if (node_id == self.start_node) | (node_id == self.end_node):
                    pass
                    # mdl.add_constraint(mdl.sum(x_var for x_var in x_vars_a) == len(vehicles)*y_vars[node_id])
                else:
                    mdl.add_constraint(mdl.sum(x_var for x_var in x_vars_a) == y_vars[node_id])
            # flow conservation constraint
                for k in vehicles:
                    x_vars_k = [x_var for x_var in x_vars if vehicle(x_var) == str(k)]
                    mdl.add_constraint(mdl.sum(x_in for x_in in x_vars_k if int_str(arc(x_in)) in delta_minus[node_id]) == mdl.sum(x_out for x_out in x_vars_k if int_str(arc(x_out)) in delta_plus[node_id]))
                ### KGN: not sure of this eqn. But i removed it and doesn't seem to affect the result.
                # if node_id == 'end':
                #     for k in vehicles:
                #         x_vars_a_k = [x_var for x_var in x_vars_a if vehicle(x_var) == str(k)]
                #         mdl.add_constraint(mdl.sum(x_var for x_var in x_vars_a_k) == y_vars[list(nodes.keys())[self.end_node]])
                
            iteration += 1
            PrintProgressBar(iteration, total_iterations, prefix = 'Progress:', suffix = 'Complete', length = 25)
        print()

        print('Adding subtour constraints:')
        total_iterations = len(vehicles)*len(self._A)
        iteration = 0
        PrintProgressBar(0, total_iterations, prefix = 'Progress:', suffix = 'Complete', length = 25)
        for k in vehicles:
            u_vars_k = [u_var for u_var in u_vars if (u_var.name.split('_')[1] == str(k)) & (u_var.name.split('_')[2] not in [self.start_node, 'start', 'end'])]
            # u_vars_k = [u_var for u_var in u_vars if (u_var.name.split('_')[1] == str(k)) & (u_var.name.split('_')[2] not in ['0', 'start', 'end'])]
            mdl.add_constraints([u_var >= 2 for u_var in u_vars_k])
            mdl.add_constraints([u_var <= len(nodes)-2 for u_var in u_vars_k])
            for arc_id in self._A:
                iteration += 1
                PrintProgressBar(iteration, total_iterations, prefix = 'Progress:', suffix = 'Complete', length = 25)
                
                _arc = self._A[arc_id]
                tail = _arc[0]
                head = _arc[1]
                if tail in [self.start_node, 'start', 'end'] or head in [self.start_node, 'start', 'end']:
                # if tail in [0, 'start', 'end'] or head in [0, 'start', 'end']:
                    continue
                x_var_k = [x_var for x_var in x_vars if (x_var.name.split('_')[1] == str(k)) & (x_var.name.split('_')[2] == str(arc_id))]
                u_var_i = [u_var for u_var in u_vars_k if (u_var.name.split('_')[2] == str(tail))]
                u_var_j = [u_var for u_var in u_vars_k if (u_var.name.split('_')[2] == str(head))]
                mdl.add_constraint(u_var_i[0] - u_var_j[0] + 1 <= (len(nodes)-3)*(1 - x_var_k[0]))
        print()

        for k in vehicles:
            x_vars_k = [x_var for x_var in x_vars if vehicle(x_var) == str(k)]
            mdl.add_constraint(mdl.sum(lengths[int_str(arc(x_var))]*x_var for x_var in x_vars_k) <= L[k])

            a_v_0 = [var for var in x_vars_k if any(var for arc_id in delta_plus['start'] if (str(arc_id) == arc(var)))]
            mdl.add_constraint(mdl.sum(x_var for x_var in a_v_0) == 1)

            a_v_T = [var for var in x_vars_k if any(var for arc_id in delta_minus['end'] if (str(arc_id) == arc(var)))]
            mdl.add_constraint(mdl.sum(x_var for x_var in a_v_T) == 1)

        y_vars_obj = y_vars[0:-3]
        mdl.maximize(mdl.sum(scores.iloc[nodes[int(node(y_var))][0], nodes[int(node(y_var))][1]] * scale_obj * y_var for y_var in y_vars_obj))
        
        if timelimit is not None:
            mdl.set_time_limit(timelimit)
        mdl.solve(log_output=True)
        mdl.print_solution()
        self.mdl = mdl

        solution = mdl.solution.as_name_dict()
        path_all = dict(filter(lambda elem: split(elem[0])[0] == 'x', solution.items()))
        self.nodes_all = dict(filter(lambda elem: split(elem[0])[0] == 'y', solution.items()))
        self.path_all = path_all
        self.vehicles = vehicles
        path = {vehicle_id: ['_'.join([split(elem[0])[i] for i in range(len(split(elem[0]))) if i != 1]) for elem in path_all.items() if split(elem[0])[1] == str(vehicle_id)] for vehicle_id in vehicles}
        y_scores = 0
        for i in range(len(self.nodes_all)):
            y_scores += self._scores[int(list(self.nodes_all.keys())[i].split('_')[1])]
        self.final_score = y_scores

        self.path = path
        self.num_vehicles = num_vehicles
        # make dict of located targets in self.located_targets
        super().get_located_targets()

class Greedy(Solution):
    def __init__(self, map_inst, min_score, nghbr_lvl = 0):
        super().__init__(map_inst, min_score, nghbr_lvl)

    def increase_connectivity(self, tail, nodes_all, max_depth = 5):
        '''
        Used if algorithm has cornered itself. A single arc is added to the network s.t. the path can continue.

        parameters
        ----------
        tail : int
            id of the tail-node
        nodes_all : list
            a list of all visited nodes.
        max_depth : int
            The highest additional depth to look in the neighbors of Delaunay triangulation
        '''
        
        # checks if all nodes of network have been visited already.
        _ = [node for node in nodes_all if node not in ('y_start', 'y_end')]
        if len(_)+self.add_start+self.add_end != len(self._nodes):
            extra_lvl = 0
            temp_new_nodes = []
            while (temp_new_nodes == []):
                if extra_lvl >= max_depth:
                    print('No new nodes with additional depth of', max_depth, '. Not looking deeper in the network')
                    return []
                extra_lvl += 1
                _, temp_delta_plus, __, temp_delta_plus_nodes = self._map_inst.delta(min_score=self._min_score, nghbr_lvl = self.nghbr_lvl+extra_lvl)
                temp_new_nodes = list(set(temp_delta_plus_nodes[tail]) - set([int_str(split(y_var)[1]) for y_var in nodes_all]) - {'start'})
            # Pick nearest possible node.
            temp_lengths = [np.linalg.norm(self._nodes[tail] - self._nodes[new_node]) for new_node in temp_new_nodes]
            self.temp_lengths = temp_lengths
            choice = temp_lengths.index(min(temp_lengths))
            new_nodes = [temp_new_nodes[choice]]
            # Update the network
            new_id = len(self._A) - self.add_start - self.add_end
            #from tail to head
            self._A[new_id] = (tail, new_nodes[0])
            self._lengths[new_id] = temp_lengths[choice]
            self._delta_minus[new_nodes[0]] += [new_id]
            self._delta_plus[tail] += [new_id]
            self._delta_minus_nodes[new_nodes[0]] += [tail]
            self._delta_plus_nodes[tail] += [new_nodes[0]]
            #from head to tail
            self._A[new_id+1] = (new_nodes[0], tail)
            self._lengths[new_id+1] = temp_lengths[choice]
            self._delta_minus[tail] += [new_id+1]
            self._delta_plus[new_nodes[0]] += [new_id+1]
            self._delta_minus_nodes[tail] += [new_nodes[0]]
            self._delta_plus_nodes[new_nodes[0]] += [tail]

            # print(f'Algorithm cornered. Network extended with arc_id {new_id} from node {tail} to node {new_nodes[0]}.')
            return new_nodes
        else: 
            # print(f'All nodes are visited. No reason to increase connectivity of network.')
            return []

    def solve(self, L, num_vehicles, std = 0, use_centroids = False, random_choice=0, rcl=1):
        t = TicToc()
        """
        Generates a myopic greedy path by searching for the nearest 
        node with the highest score.
        Here the objective is score/length instead of only score. Paths are made sequentially.
        A condition is that vehicle should always have battery capacity to go
        to directly to sink node. If a node is not connected to sink node, we will expand the network. 
        If vehicle has no more capacity to visit new nodes, it will go to sink
        node in a direct path. 
        
        If the algorithm corners itself, it will expand the network to look for deeper neighbors.
        A single arc will be added to the network between the current node and the nearest feasible neighbor.

        It is possible to have the greedy algorithm fly directly to centroids with kmeans as the first step.

        Parameters
        ----------
        L : int or list
            Travel capacity of vehicles. If list should have len = num_vehicles. and the values can differ
            If only an int is supplied, it will be transformed to the list [L]*num_vehicles
        
        num_vehicles : int
            Number of vehicles to generate a path for.
        
        use_centroids : bool
            If True, calculate kmeans with num_vehicles as clusters. Find the nodes nearest to the centroids
            and makes each vehicle go to a centroid as the first objective.

        random_choice : float between 0 and 1
            Probability of making a random choice when selecting the next node.
        
        rcl : float between 0 and 1
            Restricted Candidate List, parameter from the GRASP heuristic
            At each step rcl determines the amount of candidates to choose randomly from.
            If rcl is 0.2, the semi-greedy algorithm will choose randomly between the top 80% candidates.
            If rcl is 1.0, the algorithm will always choose the candidate with the largest score/dist ratio.
        """
        start = time.time()
        if type(L) == int:
            L = [L]*num_vehicles
        else: L = sorted(L)
        self.L, self.std, self.use_centroids, self.random_choice, self.rcl = L, std, use_centroids, random_choice, rcl
        

        assert self.start_node != None, 'No start-node given. This is necessary for the greedy algorithm.'
        t.tic()
        # defining variables

        vehicles = range(num_vehicles)
        remaining_length = {vehicle_id: L[vehicle_id] for vehicle_id in vehicles}
        x_vars = [f'x_{arc_id}' for arc_id in self._A] 
        y_vars = [f'y_{node_id}' for node_id in self._nodes]
        greedy_path = {vehicle_id: ['x_start'] for vehicle_id in vehicles}
        greedy_vertices = {vehicle_id: ['y_start', 'y_' + str(self.start_node)] for vehicle_id in vehicles}
        nodes_all = list(set(node_id for _ in greedy_vertices.values() for node_id in _))
        scores = self._scores

        #Add randomness
        for key in scores.keys():
            noise = np.random.normal(0,std)
            scores[key] = scores[key] + noise
        scores['start'] = 0
        scores['end'] = 0

        print('Running the Greedy algorithm...')
        if use_centroids: # make all vehicles go directly to clusters.
            self.center_nodes = self._map_inst.get_centroids(type='score', min_score = self._min_score, num_vehicles = num_vehicles)
            cluster_dists = [np.linalg.norm(self._nodes[self.center_nodes[k]] - self._nodes[self.start_node]) for k in range(num_vehicles)]
            self.center_nodes = [x for _, x in sorted(zip(cluster_dists, self.center_nodes))]

            for vehicle_id in vehicles:
                print(f'finding path to centroid for vehicle {vehicle_id}')
                bl = [int_str(split(item)[1]) for item in nodes_all if item not in [f'y_{self.start_node}', 'y_start', 'y_end']]
                bl += [centroid for centroid in self.center_nodes if (centroid != self.center_nodes[vehicle_id]) & (centroid not in bl)]
                dijk_nodes, dijk_arcs, dijk_len = _dijkstra(self._A, self._lengths, self.start_node, self.center_nodes[vehicle_id], blacklist = bl)
                greedy_path[vehicle_id] += ['x_' + str(arc['arc_id']) for arc in dijk_arcs]
                greedy_vertices[vehicle_id] += ['y_' + str(node) for node in dijk_nodes[1:]] #if 'y_' + str(node) not in greedy_vertices]
                nodes_all += ['y_' + str(node) for node in dijk_nodes[1:] if 'y_' + str(node) not in nodes_all]
                remaining_length[vehicle_id] -= dijk_len
        # the actual algorithm with an end point
        if self.add_end:
            for vehicle_id in vehicles:
                while greedy_path[vehicle_id][-1] != 'x_end':
                    tail = self._A[int_str(split(greedy_path[vehicle_id][-1])[1])][1]
                    # finding feasible nodes
                    # Find non-visited nearby nodes, if no non-visited nodes, then network is extended by one arc.
                    new_nodes = list(set(self._delta_plus_nodes[tail]) - set([int_str(split(y_var)[1]) for y_var in nodes_all]) - {'start'})
                    if new_nodes == []:
                        new_nodes = self.increase_connectivity(tail, nodes_all)

                    next_arc = None
                    dists = [self._lengths[arc] for arc in self._delta_plus[tail]]
                    feasible = [i for i in range(len(dists)) if
                                                            (remaining_length[vehicle_id] >= dists[i] + (np.linalg.norm(self._nodes[self._delta_plus_nodes[tail][i]] - self._nodes[self.end_node])) if self._delta_plus_nodes[tail][i] != 'end' else 0)
                                                            & (self._delta_plus_nodes[tail][i] != 'end')
                                                            & (self._delta_plus_nodes[tail][i] in new_nodes)]
                    if feasible:
                        feasible_nodes = [self._delta_plus_nodes[tail][i] for i in feasible]
                        
                        dists = [dists[i] for i in feasible]
                        score_list = [scores[self._delta_plus_nodes[tail][i]] for i in feasible]
                        
                        #list of objectives
                        obj_list = [score_list[i] / dists[i] for i in range(len(dists))]
                        
                        if rcl == 1:
                            max_obj = max(obj_list)
                        else:
                            entry = ceil(rcl*len(obj_list)) - 1
                            max_obj = sorted(obj_list)[entry]

                        max_obj_nodes = [feasible_nodes[i] for i in range(len(dists)) if obj_list[i] >= max_obj]
                        # choose randomly between the nodes with equal obj_value
                        if np.random.rand() < random_choice:
                            choice = np.random.choice(range(len(feasible_nodes)), size=1)
                            indx = self._delta_plus_nodes[tail].index(feasible_nodes[int(choice)])
                            
                        else:
                            choice = np.random.choice(range(len(max_obj_nodes)), size=1)
                            indx = self._delta_plus_nodes[tail].index(max_obj_nodes[int(choice)])

                        next_arc = self._delta_plus[tail][indx]
                        head = self._delta_plus_nodes[tail][indx]
                        dist = self._lengths[next_arc]

                        # Set choice as next node in path
                        greedy_path[vehicle_id].append(f'x_{next_arc}')
                        greedy_vertices[vehicle_id].append(f'y_{head}')
                        nodes_all.append(f'y_{head}')
                        remaining_length[vehicle_id] -= dist

                    else: #no feasible nodes, algorithm goes directly to end node and terminates
                        if self.end_node not in self._delta_plus_nodes[tail]:
                            print(f'the node {tail} is not connected to end_node. This should not be possible. Adding the arc to the network.')
                            # Update the network
                            new_id = len(self._A) - self.add_start - self.add_end
                            self._A[new_id] = (tail, self.end_node)
                            self._lengths[new_id] = np.linalg.norm(self._nodes[tail] - self._nodes[self.end_node])
                            self._delta_minus[self.end_node] += [new_id]
                            self._delta_plus[tail] += [new_id]
                            self._delta_minus_nodes[self.end_node] += [tail]
                            self._delta_plus_nodes[tail] += [self.end_node]
                        last_arc_id = self._delta_plus[tail][self._delta_plus_nodes[tail].index(self.end_node)]
                        greedy_path[vehicle_id] += [f'x_{last_arc_id}', 'x_end']
                        greedy_vertices[vehicle_id] += [f'y_{self.end_node}', 'y_end']
                        nodes_all.extend([item for item in [f'y_{self.end_node}', 'y_end'] if item not in nodes_all])
                        remaining_length[vehicle_id] -= self._lengths[last_arc_id]

                        print(f'Vehicle_id {vehicle_id}: algorithm terminates with remaining length {remaining_length[vehicle_id]}', 
                        f'last visited node before termination is {tail}',
                        sep = '\n')
                        break

        #Algorithm without fixed endpoint.
        else: # self.add_end==False
            for vehicle_id in vehicles:
                while remaining_length[vehicle_id] > 0:
                # while greedy_path[vehicle_id][-1] != 'x_end':
                    tail = self._A[int_str(split(greedy_path[vehicle_id][-1])[1])][1]
                    new_nodes = list(set(self._delta_plus_nodes[tail]) - set([int_str(split(y_var)[1]) for y_var in nodes_all]) - {'start'})
                    if new_nodes == []:
                        new_nodes = self.increase_connectivity(tail, nodes_all)

                    #find feasible nodes
                    next_arc = None
                    dists = [self._lengths[arc] for arc in self._delta_plus[tail]]
                    # feasible = [i for i in range(len(dists)) if (remaining_length[vehicle_id] >= dists[i] + _dijkstra(self._A, self._lengths, self._delta_plus_nodes[tail][i], 'end')[2]) & (self._delta_plus_nodes[tail][i] != 'end')]
                    feasible = [i for i in range(len(dists)) if 
                                                        (remaining_length[vehicle_id] >= dists[i])
                                                        & (self._delta_plus_nodes[tail][i] in new_nodes)]
                    if feasible:
                        feasible_nodes = [self._delta_plus_nodes[tail][i] for i in feasible]
                        dists = [dists[i] for i in feasible]
                        score_list = [scores[self._delta_plus_nodes[tail][i]] for i in feasible]

                        #list of objectives
                        obj_list = [score_list[i] / dists[i] for i in range(len(dists))]
                        max_obj = max(obj_list)
                        max_obj_nodes = [feasible_nodes[i] for i in range(len(dists)) if obj_list[i] == max_obj]
                        # choose randomly between the nodes with equal obj_value
                        choice = np.random.choice(range(len(max_obj_nodes)), size=1)
                        indx = self._delta_plus_nodes[tail].index(max_obj_nodes[int(choice)])

                        next_arc = self._delta_plus[tail][indx]
                        head = self._delta_plus_nodes[tail][indx]
                        dist = self._lengths[next_arc]

                        # Set choice as next node in path
                        greedy_path[vehicle_id].append(f'x_{next_arc}')
                        greedy_vertices[vehicle_id].append(f'y_{head}')
                        nodes_all.append(f'y_{head}')
                        remaining_length[vehicle_id] -= dist

                    else: #no feasible nodes, algorithm terminates (no end node)
                        print(f'Vehicle_id {vehicle_id}: algorithm terminates with remaining length {remaining_length[vehicle_id]}', 
                        f'last visited node before termination is {tail}',
                        sep = '\n')
                        break
        
        visited_nodes = [split(node_id)[1] for node_id in nodes_all if split(node_id)[1] not in ['start', 'end']]

        total_score = 0
        for node_id in visited_nodes:
            if node_id == 'base':
                continue
            total_score += self._map.iloc[self._nodes[int(node_id)][0],self._nodes[int(node_id)][1]]
        print(f'Total score: {total_score}')

        self.vertices = greedy_vertices
        self.path = greedy_path
        self.num_vehicles = num_vehicles
        self.score = total_score
        # make dict of located targets in self.located_targets
        super().get_located_targets()
        end = time.time()
        self.time_spent = round(end - start, 2) 

        t.toc()

class HillClimbing(Solution):
    def __init__(self, map_inst, initial_solution, min_score, nghbr_lvl = 0, start_node = 0, end_node = 0):
        #super().__init__(map_inst, min_score, nghbr_lvl, start_node=start_node, end_node=end_node)

        self.initial_solution = initial_solution
        self._A = self.initial_solution._A
        self._nodes = self.initial_solution._nodes
        self.num_vehicles = self.initial_solution.num_vehicles
        self.add_end = self.initial_solution.add_end
        self.add_start = self.initial_solution.add_start
        self._targets = self.initial_solution._targets
        self._lengths = self.initial_solution._lengths
        self._scores = self.initial_solution._scores
        self._delta_plus_nodes = self.initial_solution._delta_plus_nodes
        self._delta_minus_nodes = self.initial_solution._delta_minus_nodes
        self._targets = self.initial_solution._targets
        self._target_dict = self.initial_solution._target_dict


    # TODO: Take the length of the route into consideration
    def solve(self, L, num_rand=0, rand_prob = 0.8, initial_tsp = False):
        """
        parameters
        ----------
        initial_path : dict of paths
            A dictionary with a key for each vehicle, the item should be a list of node id's.
        
        L : int or list
            list of travel capacity for each vehicle. int can be supplied if all vehicles have 
            same capacity 

        num_rand : int
            number of times nodes are randomly bitflipped

        rand_prob : float between 0 and 1
            probability of selection a random node.
            Only used wiith num_rand > 0.

        initial_tsp : bool
            whether or not the supplied solution should be improved initially with a tsp 2-opt
        """
        # TODO: access the first element because single vehicle
        initial_solution = self.initial_solution.vertices
        if type(L) == int:
            L = [L]*self.num_vehicles
        else: L = sorted(L)


        # defining the neighbors of a given solution
        def neighbors(node_ids):
            """
            finds the neighbors of a given path.

            parameters
            ----------
            vertices : list of int's
                list of vertices in the order visited, should not include 'start' and 'end'.
            """
            neighbor_list = []

            for i in range(1, len(node_ids) - 1):
                before = node_ids[i-1]
                after = node_ids[i+1]

                intersection = list(set(self.initial_solution._delta_plus_nodes[node_ids[i-1]]) & set(self.initial_solution._delta_minus_nodes[node_ids[i+1]]))
                candidates = [node for node in intersection if node != node_ids[i]]

                if len(candidates) > 0:
                    # print(f'For node {i} there where nodes in the intersection:')
                    # print(candidates)
                    for j in candidates:
                        # print(f'We try to replace {i} with {j}')
                        temp = node_ids.copy()
                        temp[i] = j
                        neighbor = temp
                        # print(node_ids)
                        # print(neighbor)
                        neighbor_list.append(temp)
            
            return neighbor_list
        
        # defining the objective for a given list of node_ids
        def get_score(node_ids):
            """
            calculate the objective for a given set of node id's

            parameters
            ----------
            node_ids : list of int's
                The list of node id's in the visited order (but the order does not matter)
            """
            visited_nodes = list(set(node_ids))
            total_score = 0
            for node_id in visited_nodes:
                if node_id == 'base':
                    continue
                total_score += self.initial_solution._map.iloc[self.initial_solution._nodes[int(node_id)][0],self.initial_solution._nodes[int(node_id)][1]]
            
            return total_score
        
        last_route = {'node_ids': None, 'arc_ids': None, 'length': None}
        def get_length(current_solution, delta=False):
            # check whether a cached route is available
            if all(i == None for i in last_route.values()) or (delta == False):
                # we are in the first itertation and need to find the used arcs
                pairs = [current_solution[i:i+2] for i in range(0, len(current_solution)-1)]
                path = []
                for arc_id in self.initial_solution._A:
                    for pair in pairs:
                        if self.initial_solution._A[arc_id][0] == pair[0] and self.initial_solution._A[arc_id][1] == pair[1]:
                            path.append(arc_id)
                
                # then we add the information to last_route
                last_route['node_ids'] = current_solution
                last_route['arc_ids'] = path
                arc_lengths = [
                    length for (arc_id, length) in self.initial_solution._lengths.items() 
                    if arc_id in path # [i for i in path]
                ]
                last_route['length'] = reduce(lambda x,y: x+y, arc_lengths)
            return last_route['length']
            # else:
                # we can use the last route to calculate the length of the new route

                # find the nodes in the new node_ids that are not in the old node_ids'

        def Opt(current_solution, blacklist):
            # Compute distance matrix
            # M = [[None]*len(current_solution) for i in range(len(current_solution))]
            # for i in range(len(current_solution)):
            #     for j in range(len(current_solution)):
            #         if i != j:
            #             temp = _dijkstra(self.initial_solution._A, self.initial_solution._lengths, i, j)
            #             M[i][j] = temp[2]
            #         else:
            #             M[i][j] = 0
            # Takes dobble the work, so maybe optimize later

            def Swap(route, i, k):
                # print(route)
                new_route = route[0:i]
                new_route += (route[i:k+1])[::-1]
                new_route += route[k+1:len(route)]
                # print(new_route)
                # print(f'first dijkstra call {new_route[i-1]} and {new_route[i]}')
                test = _dijkstra(self.initial_solution._A, self.initial_solution._lengths, new_route[i-1], new_route[i],blacklist=new_route + blacklist)
                # print(f'Dijkstra from {new_route[i-1]} to {new_route[i]} is:')
                # print(test[0])
                for node in test[0][1:-1][::-1]:
                    new_route.insert(i, node)
                # print(new_route)
                # print(f'first dijkstra call {new_route[k+len(test)]} and {new_route[k+len(test)+1]}')
                test2 = _dijkstra(self.initial_solution._A, self.initial_solution._lengths, new_route[k+len(test[0])-2], new_route[k+len(test[0])-2+1],blacklist=new_route + blacklist)
                # print(f'Dijkstra from {new_route[k+len(test[0])-2]} to {new_route[k+len(test[0])-2+1]} is:')
                # print(test2[0])
                for node in test2[0][1:-1][::-1]:
                    new_route.insert(k+len(test[0])-2+1, node)
                # print(new_route)
                length_after = test[2] + test2[2]
                # print(f'length after is {length_after}')

                length_before= 0
                counter = 0
                for arc_id in self.initial_solution._A:
                    if (self.initial_solution._A[arc_id] == (route[i-1], route[i])) or (self.initial_solution._A[arc_id] == (route[k], route[k+1])):
                        length_before += self.initial_solution._lengths[arc_id]
                        counter += 1
                        if counter == 2:
                            break
                # print(f'length before is {length_before}')
                return new_route, length_before, length_after
            
            #### Make graph object for the dijkstra-fct in swap
            graph_obj = Graph() #directed by default
            for arc_id in self.initial_solution._A:
                tail = self.initial_solution._A[arc_id][0]
                head = self.initial_solution._A[arc_id][1]
                graph_obj.add_edge(tail, head, {'arc_id': arc_id, 'cost': self.initial_solution._lengths[int_str(arc_id)]})


            # Swap(temp,1,4)
            # print(Swap(temp, 2, 4))
            existing_route = current_solution
            best_distance = get_length(existing_route)
            last_distance = 0
            # while last_distance < best_distance:
            best_distance = get_length(existing_route)
            last_distance = best_distance
            #number of nodes eligible to be swapped
            n = len(existing_route)
            for i in np.arange(1, n-1):
                for k in np.arange(i+1, n-1):
                    # print(f'existing route is {existing_route}')
                    # print(f'trying to swap index {i}: {existing_route[i]} and {k}: {existing_route[k]}')
                    try:
                        new_route, length_before, length_after = Swap(existing_route, i, k)
                    except NoPathError:
                        # print(f'Could not swap {i} and {k}')
                        continue
                    # print(f'new route is: {new_route}')
                    # new_distance = get_length(new_route)
                    if length_before > length_after:
                        existing_route = new_route
                        best_distance = best_distance - length_before + length_after
            return existing_route

        # defining the add-able nodes of a given solution
        def add_node(node_ids, remaining_length = np.inf, blacklist = [], rand_choice=False):
            """
            Adds a node to the current path.
            Finds first the nodes that can be added to the path. 
            Does not consider nodes before the first node or after the last node.
            Finds best nodes in terms of score/dist or alternatively a random node can be chosen.

            The dictionary nodes_add is on the form: {node_in_path: [addable_nodes]}
            nodes_obj is on the form: {node_in_path: [obj_of_addable_nodes]}
            nodes_dist is on the form: {node_in_path: [dist_of_addable_nodes]}

            now saves the dictionaries, so if add_node was run in the last iteration, the last dict is only mutated on the nodes around the lastly added.
            ----------
            node_ids : list of int's
                list of vertices in the order visited, should not include 'start' and 'end'.
            
            remaining_length : float
                remaining length of the current path. 
                Used to calculate if the best node is feasible to add. So only has an impact if rand_choice==False.
            
            blacklist : list of node-names
                Nodes that are not allowed to be added to the path
                This can fx be paths from other uavs or if using a tabu-search
            
            rand_choice : bool
                If true: will add a random node to  path
                If false: will add the best node to path. 

            """
            ### objects for adding node. If function was run in last iteration, picks up the dictionary from last and updates with the node that has been added.
            global new_enabled
            global nodes_add; global nodes_obj; global nodes_dist

            best_node = None
            best_before = None
            best_obj = 0
            best_idx = None

            if new_enabled == []:
                nodes_add = dict()
                nodes_obj = dict()
                # nodes_score = dict()
                nodes_dist = dict()
                loop_range = range(1, len(node_ids))  #go through all nodes

            else:
                newest_idx = node_ids.index(new_enabled[-1])
                nodes_add.pop(node_ids[newest_idx-1], None) #remove node before from dict to redo them.
                nodes_obj.pop(node_ids[newest_idx-1], None)
                nodes_dist.pop(node_ids[newest_idx-1], None)
                for key in nodes_add:
                    if new_enabled[-1] in nodes_add[key]:
                        pop_idx = nodes_add[key].index(new_enabled[-1])
                        nodes_add[key].pop(pop_idx)
                        nodes_obj[key].pop(pop_idx)
                        nodes_dist[key].pop(pop_idx)
                loop_range = range(newest_idx, newest_idx+2) #only go through the node before and the actual added (see beginning of loop to understand index choice)
                # nodes_add.pop(node_ids[newest_idx-1], None); nodes_add.pop(node_ids[newest_idx+1], None) #remove node before and after from dict to redo them.
                # nodes_obj.pop(node_ids[newest_idx-1], None); nodes_obj.pop(node_ids[newest_idx+1], None)
                # nodes_dist.pop(node_ids[newest_idx-1], None); nodes_dist.pop(node_ids[newest_idx+1], None)
                # loop_range = range(newest_idx-1, newest_idx+2) #only go through the node before, the actual added, and the node after
                
                # find best feasible node of existing dict.
                # print(nodes_obj)
                list_of_dists = [item for _ in nodes_dist.values() for item in _] 
                list_of_feasible = [1 if item < remaining_length else 0 for item in list_of_dists] 
                list_of_objs = [item for _ in nodes_obj.values() for item in _]
                list_of_vals = [feasible*obj for feasible,obj in zip(list_of_feasible,list_of_objs)]
                if len(list_of_vals) > 0:
                    highest_val = max(list_of_vals)
                    # print(highest_val)
                    if highest_val > 0:
                        for key in nodes_obj:
                            if highest_val in nodes_obj[key]:
                                best_idx = nodes_obj[key].index(highest_val)
                                best_node = nodes_add[key][best_idx]
                                best_before = key
                                best_obj = highest_val
                                break

            arc_keys, arc_values = zip(*self._A.items())
            arc_keys, arc_values = list(arc_keys), list(arc_values)

            for i in loop_range:
            # for i in range(1, len(node_ids)):
                before = node_ids[i-1]
                after = node_ids[i]

                if i != len(node_ids) -1:
                    candidates = list((set(self._delta_plus_nodes[before]) & set(self._delta_minus_nodes[after])) - set(node_ids) - set(blacklist))
                else:
                    candidates = list(set(self._delta_plus_nodes[before]) - set(node_ids) - set(blacklist))
                    
                # if len(candidates) == 0: print(f'candidate {before} does not have any feasible nodes to add')
                if len(candidates) > 0:
                    candidates_obj = []
                    candidates_score = []
                    candidates_dist = []

                    # find score/dist of node if added to path.
                    for new_node in candidates:
                        # find dist, maybe this part can be optimized in terms of running speed. 
                        before_arc_key = arc_keys[arc_values.index((before, new_node))]
                        after_arc_key = arc_keys[arc_values.index((new_node, after))]                        
                        previous_arc_key = arc_keys[arc_values.index((before, after))]

                        before_arc_dist = self._lengths[before_arc_key]
                        after_arc_dist = self._lengths[after_arc_key]
                        previous_arc_dist = self._lengths[previous_arc_key]
                        extra_len = before_arc_dist + after_arc_dist - previous_arc_dist

                        candidates_dist.append(extra_len)

                        if extra_len == 0:
                            new_node_objective = np.inf
                        elif extra_len < remaining_length:
                            new_node_objective = (self._scores[new_node] / extra_len)
                        else:
                            new_node_objective = 0
                        # new_node_objective = (self._scores[new_node] / extra_len) if extra_len < remaining_length else 0

                        candidates_obj.append(new_node_objective)

                        if new_node_objective > best_obj:
                            best_node = new_node
                            best_obj = new_node_objective
                            best_before = before
                            best_idx = candidates.index(new_node)
                    
                    nodes_add[before] = candidates
                    nodes_obj[before] = candidates_obj
                    nodes_dist[before] = candidates_dist
                    # best_adds[before] = best_obj

            if best_node == None: 
                print('no feasible nodes to enable')
                return node_ids, 0, 0
            
            # If: choose best option.
            if rand_choice == False:
                print(f'best node is: {best_node}')
                # global new_enabled
                new_enabled.append(best_node)
            # elif: make a random choice possible, for non-local search
            elif rand_choice == True:
                best_before = int_str(np.random.choice(list(nodes_add.keys())))
                best_node = np.random.choice(nodes_add[best_before])
                best_idx = nodes_add[best_before].index(best_node)
                # best_idx = np.random.choice(len(nodes_add[best_before][0]))
                # best_node = nodes_add[best_before][0][best_idx]

                print(f'randomly enabling node: {best_node}')

            before_loc = node_ids.index(best_before, 0)
            new_path = node_ids[0:before_loc+1] + [best_node] + node_ids[before_loc+1:]
            delta_score = self._scores[best_node]
            delta_len = nodes_dist[best_before][best_idx]
            
            return new_path, delta_score, delta_len

        # defining the removable nodes of a given solution
        def del_node(node_ids, rand_choice = False):
            """
            This method disables a node in the given path. 
            The node can be chosen randomly or by the worst objective value (score/dist)

            The dict nodes_remove is on the form: {node_in_path: obj_val}
            The dict nodes_remove_dist is on the form: {node_in_path: shorter-potential-distance}

            TODO: in a loop we will be making a lot of similar dictionaries, so we should probably keep the objects.
            ----------
            node_ids : list of int's
                list of vertices in the order visited, should not include 'start' and 'end'.
            
            rand_choice : bool
                If true: will remove a random node from the path
                If false: will remove the worst node from path. 
            """
            ### objects for removing node
            nodes_remove = dict() #{node_id: obj_val}
            nodes_remove_dist = dict() #{node_id: dist_saved}

            worst_node = None
            worst_obj = np.inf

            arc_keys, arc_values = zip(*self._A.items())
            arc_keys, arc_values = list(arc_keys), list(arc_values)

            for i in range(1, len(node_ids)-1):
                before = node_ids[i-1]
                this_node = node_ids[i]
                after = node_ids[i+1]

                # check if it is possible to remove this_node
                if after in self._delta_plus_nodes[before]:
                    before_arc_key1 = arc_keys[arc_values.index((before, this_node))]
                    before_arc_key2 = arc_keys[arc_values.index((this_node, after))]

                    after_arc_key = arc_keys[arc_values.index((before, after))]

                    before_arc_dist = self._lengths[before_arc_key1] + self._lengths[before_arc_key2]
                    after_arc_dist = self._lengths[after_arc_key]
                    remove_len = before_arc_dist - after_arc_dist
                    
                    before_node_score = self._scores[this_node]

                    this_obj = (before_node_score / remove_len) if remove_len != 0 else np.inf

                    nodes_remove_dist[this_node] = remove_len
                    nodes_remove[this_node] = this_obj

                    if this_obj < worst_obj:
                        worst_node = this_node
                        worst_obj = this_obj

            if worst_node == None:
                print('no feasible nodes to disable')
                return node_ids, 0, 0

            # If remove worst node
            if rand_choice == False:
                print(f'worst node is: {worst_node}')
                global new_disabled
                new_disabled.append(worst_node)
            # elif randomly remove node
            elif rand_choice == True:
                worst_node = np.random.choice(list(nodes_remove.keys()))
                print(f'randomly disabling node: {worst_node}')

            # node_ids.remove(worst_node)
            # new_path = node_ids
            new_path = node_ids.copy()
            new_path.remove(worst_node)
            delta_score = - self._scores[worst_node]
            delta_len = - nodes_remove_dist[worst_node]
            return new_path, delta_score, delta_len

        # Start of the hill climbing algorithm
        previous_solution = {}
        initial_score = {}
        initial_length = {}
        
        current_solution = {i: None for i in range(self.num_vehicles)}
        for vehicle_id in range(self.num_vehicles):
            current_solution[vehicle_id] = [int_str(i.split('_')[1]) for i in initial_solution[vehicle_id][1:len(initial_solution[vehicle_id]) - 1]]
        
        for vehicle_id in range(self.num_vehicles):
            blacklist = []
            for i in range(self.num_vehicles):
                if i != vehicle_id:
                    blacklist.extend(current_solution[i])

            previous_solution[vehicle_id] = []
            initial_score[vehicle_id] = get_score(current_solution[vehicle_id])
            initial_length[vehicle_id] = get_length(current_solution[vehicle_id])
            iteration = 0
            print(f"#### Starting the Hill Climbing algorithm for vehicle {vehicle_id}####")
            current_score = initial_score[vehicle_id]
            current_len = initial_length[vehicle_id]
            tsp_operator = initial_tsp
            rand_search = True if num_rand > 0 else False
            del_operator = True 
            global new_enabled
            global new_disabled
            new_enabled = [] #list of enabled nodes since last change
            new_disabled = []
            last_enabled = [None] #becomes list of enabled nodes in the last change
            last_disabled = [None]
            flipped_runs = 0
            while True:
                if tsp_operator:
                    while current_solution[vehicle_id] != previous_solution[vehicle_id]:
                        iteration += 1
                        print(f'Iteration {iteration:02} | ')
                        next_eval = -np.inf
                        next_solution = None

                        previous_solution[vehicle_id] = current_solution[vehicle_id]
                        current_solution[vehicle_id] = Opt(current_solution[vehicle_id], blacklist)
                        
                    tsp_operator = False
                    current_len = get_length(current_solution[vehicle_id])
                    current_score = get_score(current_solution[vehicle_id])
                    continue
                # random search
                elif rand_search: 
                    disable_node = np.random.choice([True, False])
                    rand_choice = True if rand_prob > np.random.random() else False
                    if disable_node:
                        pot_solution, delta_score, delta_len = del_node(current_solution[vehicle_id], 
                                                                        rand_choice = rand_choice)
                    else:
                        pot_solution, delta_score, delta_len = add_node(current_solution[vehicle_id], 
                                                                    # remaining_length = L[vehicle_id] - current_len, 
                                                                    blacklist = blacklist,
                                                                    rand_choice = rand_choice)
                    if iteration > num_rand:
                        rand_search = False
                        #checking if solution is feasible
                        while current_len > L[vehicle_id]:
                            iteration += 1
                            pot_solution, delta_score, delta_len = del_node(current_solution[vehicle_id], 
                                                rand_choice = False)
                            current_solution[vehicle_id] = pot_solution.copy()
                            current_score += delta_score
                            current_len += delta_len



                #local search
                elif del_operator:
                    pot_solution, delta_score, delta_len = del_node(current_solution[vehicle_id], 
                                                                    rand_choice = False)
                    flipped_runs += 1
                #local search
                elif not del_operator:
                    pot_solution, delta_score, delta_len = add_node(current_solution[vehicle_id], 
                                                                    remaining_length = L[vehicle_id] - current_len, 
                                                                    blacklist = blacklist,
                                                                    rand_choice = False)
                    flipped_runs += 1


                # using local search for efficiency increase:
                if (rand_search == False) & (((delta_score + current_score) / (delta_len + current_len)) <= (current_score/current_len)):
                    # checking if recently flipped
                    # if (delta_score == 0) & (flipped_runs < 2): 
                    if flipped_runs < 2: 
                        print('Path has reached local limit of efficiency. Will add nodes for the remaining length of route.')
                        flipped_runs = 0
                        new_enabled = []
                        while True:
                            pot_solution, delta_score, delta_len = add_node(current_solution[vehicle_id], 
                                                                            remaining_length = L[vehicle_id] - current_len, 
                                                                            blacklist = blacklist,
                                                                            rand_choice = False)
                            flipped_runs += 1
                            iteration += 1
                            if delta_score == 0:
                                print('Algorithm terminates as score cannot be increased.')
                                print(f'number of iterations: {iteration}')
                                break
                            else:
                                current_solution[vehicle_id] = pot_solution.copy()
                                current_score += delta_score
                                current_len += delta_len
                        break

                    print(f'flipping the algorithm')
                    del_operator = not del_operator
                    flipped_runs = 0
                    if del_operator:
                        last_disabled = new_disabled.copy()
                        new_disabled = []
                    else: 
                        last_enabled = new_enabled.copy()
                        new_enabled = []
                else:
                    current_solution[vehicle_id] = pot_solution.copy()
                    current_score += delta_score
                    current_len += delta_len
                iteration += 1

                # print('current_score', current_score)
                # print('calculating current_score', get_score(current_solution[vehicle_id]))
                # print('current_len', current_len)
                # print('calculating current_len', get_length(current_solution[vehicle_id]))
                # print('delta_len', delta_len)
                # print('delta_score', delta_score)
                # print('')

        for vehicle_id in range(self.num_vehicles):        
            print(f'Initial score for vehicle {vehicle_id} was {initial_score[vehicle_id]}, final score is {get_score(current_solution[vehicle_id])}')
            print(f'Initial length for vehicle {vehicle_id} was {round(initial_length[vehicle_id], 2)}, final length is {round(get_length(current_solution[vehicle_id]), 2)}')
            print(f'Initial efficiency for vehicle {vehicle_id} was {round(initial_score[vehicle_id]/initial_length[vehicle_id], 2)}, final efficiency is {round(get_score(current_solution[vehicle_id])/get_length(current_solution[vehicle_id]), 2)}')
            # print(current_score)
            # print(current_len)
        
        self.score_improvement = [sum(initial_score.values()), sum([get_score(sol) for sol in current_solution.values()])]
        self.length_change = [sum(initial_length.values()), sum([get_length(sol) for sol in current_solution.values()])] 

        print(f'Total initial score for all vehicles was {self.score_improvement[0]}, final score is {self.score_improvement[1]}')
        print(f'Total efficiency for all vehicles was {round(self.score_improvement[0]/self.length_change[0],2)}, final efficiency is {round(self.score_improvement[1]/self.length_change[1],2)}')

        print('Finding the arcs used in the solution... ', end = '')
        
        self.path = {i: None for i in range(self.num_vehicles)}
        self.vertices = {i: None for i in range(self.num_vehicles)}

        # The arcs used are found 
        for vehicle_id in range(self.num_vehicles):
            if (split(initial_solution[vehicle_id][0])[1] != 'start') or (split(initial_solution[vehicle_id][-1])[1] != 'end'):
                raise Exception("The initial solution given to HillClimbing does not include 'start' and 'end'")
            current_solution[vehicle_id].insert(0, 'start')
            current_solution[vehicle_id].insert(len(current_solution[vehicle_id]), 'end')
            pairs = [current_solution[vehicle_id][x:x+2] for x in range(0, len(current_solution[vehicle_id])-1)]
            path = []
            for pair in pairs:
                for arc_id in self.initial_solution._A:
                    found = False
                    if self.initial_solution._A[arc_id][0] == pair[0] and self.initial_solution._A[arc_id][1] == pair[1]:
                        path.append(arc_id)
                        found = True
                        break
                if found == False:
                    print(f'There is a missing arc between {pair[0]} and {pair[1]}')
            self.path[vehicle_id] = ['x_' + str(i) for i in path]
            self.vertices[vehicle_id] = ['y_' + str(i) for i in current_solution[vehicle_id]]
        print('Done!')

def GRASP(map_inst, n_iter, rcl, nghbr_lvl, num_vehicles, L, min_score=1, use_centroids=True, initial_tsp=False):
    '''
    Greedy randomized adaptive search procedure
    in a loop of size n_iter, a new path is constructed semi-greedily and afterwards it is locally optimized
    In each iteration the accumulated score of the path is evaluated, and if it is the largest yet it will be saved, otherwise it will be discarded.

    In the end the best of the n_iter paths are returned

    parameters
    ----------
    map_inst: the map instance

    n_iter: int
        number of paths to generate

    rcl: float between 0 and 1
        restricted candidate list.
        The amount of candidates that are removed. If rcl=0.1, the worst 10% candidates are removed from the candidate list to choose randomly from.

    returns
    --------
        the best path object.
    '''
    start=time.time()
    gr = Greedy(
                map_inst, 
                min_score = min_score,
                nghbr_lvl=nghbr_lvl)
    best_sol_val = 0
    best_sol = None
    for _ in range(n_iter):
        gr.solve(
                num_vehicles=num_vehicles, 
                L=L, 
                use_centroids=use_centroids, 
                rcl=rcl)

        hc = HillClimbing(
                        map_inst, 
                        initial_solution=gr, 
                        min_score=min_score, 
                        nghbr_lvl=nghbr_lvl)
        hc.solve(
                L=L, 
                initial_tsp = initial_tsp)
        
        if hc.score_improvement[1] > best_sol_val:
            best_sol_val = hc.score_improvement[1]
            #TODO: check if best_sol keeps the best solution or if it will be updated in the next step, when hc is changed.
            best_sol = hc
    end = time.time()
    best_sol.time_spent = round(end - start, 2) 
    print('GRASP has terminated with a score of ', best_sol_val)
    print('time spend is', best_sol.time_spent)
    return best_sol

def GRASP_single_iter(map_inst, rcl, nghbr_lvl, num_vehicles, L, min_score=1, use_centroids=True, initial_tsp=False):
    '''
    Greedy randomized adaptive search procedure
    Performs a single grasp iteration: constructing a greedy solution and locally optimizing it.
    Does not evaluate previous solutions

    parameters
    ----------
    map_inst: the map instance

    rcl: float between 0 and 1
        restricted candidate list.
        The amount of candidates that are removed. If rcl=0.1, the worst 10% candidates are removed from the candidate list to choose randomly from.

    returns
    --------
        the solution object.
    '''
    start=time.time()
    gr = Greedy(
                map_inst, 
                min_score = min_score,
                nghbr_lvl=nghbr_lvl)
    gr.solve(
            num_vehicles=num_vehicles, 
            L=L, 
            use_centroids=use_centroids, 
            rcl=rcl)

    hc = HillClimbing(
                    map_inst, 
                    initial_solution=gr, 
                    min_score=min_score, 
                    nghbr_lvl=nghbr_lvl)
    hc.solve(
            L=L, 
            initial_tsp = initial_tsp)
    
    sol = hc
    end = time.time()
    sol.time_spent = round(end - start, 2) 
    return sol

### TESTING ####
# # load the sample map
# path_to_sample = '//math.aau.dk/ProjectGroups/f21moe8or/maps/sample_map'
# path_to_sample = '//math.aau.dk/ProjectGroups/f21moe8or/maps/facility_allocation/b672aa'
# sample = Map(path_to_sample, base_node = 0)
# sample.plot()
# # sample.plot_flows(nghbr_lvl=2, show_nodes=True)
# gr = Greedy(
#     sample, 
#     min_score=1,
#     nghbr_lvl=2
#     )
# gr.solve(
#     num_vehicles = 2, 
#     L=300, 
#     use_centroids=True, 
#     rcl=0.8
#     )

# hc = HillClimbing(
#     sample,
#     initial_solution = gr,
#     min_score=1,
#     nghbr_lvl=2)
# hc.solve(
#     L=300,
#     initial_tsp=False
#     )
# gr.plot(single_ax=True)
# hc.plot(single_ax=True)
# grasp_cent = GRASP(sample, n_iter = 5, rcl=0.8, nghbr_lvl = 2, num_vehicles = 3,L = 200, use_centroids=True)
# # grasp_no_cent = GRASP(sample, n_iter = 5, rcl=0.8, nghbr_lvl = 2, num_vehicles = 2,L = 200, use_centroids=False)
# # grasp_cent.plot(single_ax=True)
# # grasp_no_cent.plot(single_ax=True)
# # path_to_sample = '//math.aau.dk/ProjectGroups/f21moe8or/e09dbf'
# # sample = Map(path_to_sample, base_node = 0)
# # grasp_smallmap = GRASP(sample, n_iter = 50, rcl=0.2, nghbr_lvl = 0, num_vehicles = 1,L = 40, use_centroids=False)
# # grasp_smallmap.score_improvement
# # grasp_smallmap.plot(nodes=True)

# grasp_nghbr = [None] * 4
# for nghbrs in [0, 1, 2, 3]:
#     grasp_nghbr[nghbrs] = GRASP(sample, n_iter = 5, rcl=0.8, nghbr_lvl = nghbrs, num_vehicles = 3,L = 200, use_centroids=False)

# grasp_nghbr += [None, None, None]
# for nghbrs in [4,5,6]:
#     grasp_nghbr[nghbrs] = GRASP(sample, n_iter = 5, rcl=0.8, nghbr_lvl = nghbrs, num_vehicles = 3,L = 200, use_centroids=False)

# for sol in grasp_nghbr:
#     print(f'time spent is: {sol.time_spent}, for a solution value of {sol.score_improvement[1]}')

# rcl=0.5
# min_score = 1
# nghbr_lvl = 2
# L = [300, 350]
# num_vehicles=2
# gr = Greedy(
# sample, 
# min_score = min_score,
# nghbr_lvl=nghbr_lvl)

# gr.solve(num_vehicles=num_vehicles, L=L_list, use_centroids=True, rcl=rcl)
# hc = HillClimbing(sample, initial_solution=gr, min_score=min_score, nghbr_lvl=nghbr_lvl)
# hc.solve(L=L_list, initial_tsp = False)

# best_sol_val = [0]*10
# best_sol = [None]*10
# for i, rcl in enumerate(np.arange(0, 1, 0.1)):
#     print('Starting GRASP with rcl', rcl)
#     for rep in range(0,100):
#         gr.solve(num_vehicles=num_vehicles, L=L_list, use_centroids=True, rcl=rcl)

#         hc = HillClimbing(sample, initial_solution=gr, min_score=min_score, nghbr_lvl=nghbr_lvl)
#         hc.solve(L=L_list, initial_tsp = False)
        
#         if hc.score_improvement[1] > best_sol_val[i]:
#             print('Found a better solution at iteration', rep, '. The score is now', hc.score_improvement[1])
#             best_sol_val[i] = hc.score_improvement[1]
#             best_sol[i] = hc

# gr2 = Greedy(
# sample, 
# min_score = min_score,
# nghbr_lvl=nghbr_lvl)

# gr2.solve(num_vehicles=num_vehicles, L=L_list, use_centroids=True)
# hc2 = HillClimbing(sample, initial_solution=gr, min_score=min_score, nghbr_lvl=nghbr_lvl)
# hc2.solve(L=L_list, initial_tsp = False)

# hc3 = HillClimbing(sample, initial_solution=gr, min_score=min_score, nghbr_lvl=nghbr_lvl)
# hc3.solve(L=L_list, initial_tsp = True)

# gr2.plot(single_ax=True, nodes=False)
# hc2.plot(single_ax=True, nodes=False)
# hc3.plot(single_ax=True, nodes=False)
