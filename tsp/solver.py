#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import random
from collections import namedtuple
import time

Point = namedtuple("Point", ['x', 'y'])
nn = 0
def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def object_value(solution):
    obj = length(solution[-1], solution[0])
    for index in range(len(solution)-1):
        obj += length(solution[index], solution[index+1])
    return obj

def trivial(node):
    '''
    just connection the node one by one
    '''
    solution = []
    n = len(node)
    for i in range(n):
        solution.append(node[i])
        print('Total %s nodes, there are %s node(s) not connected...' % (n, n-i-1))
    return solution


def greedy_algorithm(node):
    '''
    greedy algorithm to initialize a feasible solution for later optimization
    '''
    #print('Applying greedy algorithm...')
    
    free_node = node[:]
    solution = []
    cur = free_node[0]
    free_node.remove(cur)
    solution.append(cur)
    while free_node:
        print('There are %s nodes left...'%str(len(free_node)))
        min_length = None
        min_node = None
        for i in free_node:
            l = length(i, cur)
            if min_length is None:
                min_length = l
                min_node = i
            elif l <= min_length:
                min_length = l
                min_node = i
        solution.append(min_node)
        free_node.remove(min_node)
        cur = min_node
    return solution


def two_opt(solution):
    '''
    optimize the solution of greedy algorithm by using 2-opt
    '''
    step = 0
    solution.append(solution[0])
    best = solution
    improved = True
    while improved:
        improved = False
        for i in range(1, len(solution)-2):
            for j in range(i+1, len(solution)):
                if j-i == 1: continue # changes nothing, skip then
                new_solution = solution[:]
                new_solution[i:j] = solution[j-1:i-1:-1] # this is the 2woptSwap
                if object_value(new_solution) < object_value(best):
                    best = new_solution
                    improved = True
        #old_soution = solution
        solution = best
        step += 1
        print("obj val:%s, step:%s..." % (object_value(solution), step) )
    return best[0:-1]

def optimize2opt(nodes, solution, number_of_nodes):
    best = 0
    best_move = None
    # For all combinations of the nodes
    for ci in range(0, number_of_nodes):
        for xi in range(0, number_of_nodes):
            yi = (ci + 1) % number_of_nodes  # C is the node before Y
            zi = (xi + 1) % number_of_nodes  # Z is the node after X

            c = solution[ ci ]
            y = solution[ yi ]
            x = solution[ xi ]
            z = solution[ zi ]
            # Compute the lengths of the four edges.
            cy = length( c, y )
            xz = length( x, z )
            cx = length( c, x )
            yz = length( y, z )

            # Only makes sense if all nodes are distinct
            if xi != ci and xi != yi:
                # What will be the reduction in length.
                gain = (cy + xz) - (cx + yz)
                # Is is any better then best one sofar?
                if gain > best:
                    # Yup, remember the nodes involved
                    best_move = (ci,yi,xi,zi)
                    best = gain

    print(best_move, best)
    if best_move is not None:
        (ci,yi,xi,zi) = best_move
        # This four are needed for the animation later on.
        c = solution[ ci ]
        y = solution[ yi ]
        x = solution[ xi ]
        z = solution[ zi ]

        # Create an empty solution
        new_solution = list(range(0,number_of_nodes))
        # In the new solution C is the first node.
        # This we we only need two copy loops instead of three.
        new_solution[0] = solution[ci]

        n = 1
        # Copy all nodes between X and Y including X and Y
        # in reverse direction to the new solution
        while xi != yi:
            new_solution[n] = solution[xi]
            n = n + 1
            xi = (xi-1)%number_of_nodes
        new_solution[n] = solution[yi]

        n = n + 1
        # Copy all the nodes between Z and C in normal direction.
        while zi != ci:
            new_solution[n] = solution[zi]
            n = n + 1
            zi = (zi+1)%number_of_nodes
        # Create a new animation frame
        #frame4(nodes, new_solution, number_of_nodes, c, y, x, z, gain)
        return (True,new_solution)
    else:
        return (False,solution)

def two_opt_algorithm(nodes, number_of_nodes, solution=None):
    # Create an initial solution
    if not solution:
        solution = [n for n in nodes]
    go = True
    # Try to optimize the solution with 2opt until
    # no further optimization is possible.
    while go:
        (go,solution) = optimize2opt(nodes, solution, number_of_nodes)
    return solution

def sa_optimize_step(nodes, solution, number_of_nodes, t):
    global nn
    # Pick X and Y at random.
    ci = random.randint(0, number_of_nodes-1)
    yi = (ci + 1) % number_of_nodes
    xi = random.randint(0, number_of_nodes-1)
    zi = (xi + 1) % number_of_nodes

    if xi != ci and xi != yi:
        c = solution[ci]
        y = solution[yi]
        x = solution[xi]
        z = solution[zi]
        cy = length(c, y)
        xz = length(x, z)
        cx = length(c, x)
        yz = length(y, z)

        gain = (cy + xz) - (cx + yz)
        if gain < 0:
            # We only accept a negative gain conditionally
            # The probability is based on the magnitude of the gain
            # and the temperature.
            u = math.exp( gain / t )
        elif gain > 0.05:
            u = 1 # always except a good gain.
        else:
            u = 0 # No idea why I did this....

        # random chance, picks a number in [0,1)
        if (random.random() < u):
            nn = nn + 1
            #print "      ", gain
            # Make a new solution with both edges swapped.
            new_solution = list(range(0,number_of_nodes))
            new_solution[0] = solution[ci]
            n = 1
            while xi != yi:
                new_solution[n] = solution[xi]
                n = n + 1
                xi = (xi-1)%number_of_nodes
            new_solution[n] = solution[yi]
            n = n + 1
            while zi != ci:
                new_solution[n] = solution[zi]
                n = n + 1
                zi = (zi+1)%number_of_nodes
                
            return new_solution
        else:
            return solution
    else:
        return solution
    
def sa_algorithm(nodes, number_of_nodes, solution=None):
    # Create an initial solution that we can improve upon.
    if not solution:
        solution = [n for n in nodes]

    # The temperature t. This is the most important parameter of the SA
    # algorithm. It starts at a high temperature and is then slowly decreased.
    # Both rate of decrease and initial values are parameters that need to be
    # tuned to get a good solution.

    # The initial temperature.  This should be high enough to allow the
    # algorithm to explore many sections of the search space.  Set too high it
    # will waste a lot of computation time randomly bouncing around the search
    # space.
    t = 100

    # Length of the best solution so far.
    l_min = object_value(solution)
    best_solution = []
    i = 0
    while t > 0.1:
        i = i + 1
        # Given a solution we create a new solution
        solution = sa_optimize_step(nodes, solution, number_of_nodes, t)
        # every ~200 steps
        if i >= 200:
            i = 0
            # Compute the length of the solution
            l = object_value(solution)
            print("    ", l, t, nn)
            # Lower the temperature.
            # The slower we do this, the better then final solution
            # but also the more times it takes.
            t = t*0.99999

            # See if current solution is a better solution then the previous
            # best one.
            if l_min is None: # TODO: This can be removed, as l_min is set above.
                l_min = l
            elif l < l_min:
                # Yup it is, remember it.
                l_min = l
                #print ("++", l, t)
                best_solution = solution[:]
            else:
                pass

    return best_solution

def find_index(solution, points):
    '''
    find the point index in the solution
    '''
    return [points.index(i) for i in solution]

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    #solution = range(0, nodeCount)

    # calculate the length of the tour

    #solution_greedy = greedy_algorithm(points)
    if nodeCount < 2000:
        tri_solution = trivial(points)
        solution_two_opt = two_opt_algorithm(points,nodeCount, tri_solution)
        solution_sa = sa_algorithm(points, nodeCount, solution_two_opt)
    else:
        tri_solution = trivial(points)
        solution_sa = sa_algorithm(points, nodeCount, tri_solution)

    obj = object_value(solution_sa)
    solution_idx = find_index(solution_sa, points)
    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution_idx))
    
    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        #start_time = time.time()
        print(solve_it(input_data))
        #print('total time is %s min' % ((time.time()-start_time)/60))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

