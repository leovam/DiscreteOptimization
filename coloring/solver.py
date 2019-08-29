#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import gurobipy as gp

def trivial(node_count):
    # build a trivial solution
    # every node has its own color
    return range(0, node_count)

def greedy(node_count, edges):
    edge_mat = np.zeros((node_count, node_count))

    for item in edges:
        edge_mat[item[0], item[1]] = 1
        edge_mat[item[1], item[0]] = 1

    # d = []
    # for i in range(node_count):
    #     d.append(sum(edge_mat[i,:]))
    # d_idx = np.array(d).argsort()[::-1][:len(d)]
    # edge_mat = edge_mat[d_idx,:]
    # edge_mat = edge_mat[:, d_idx]

    colors = {0:0}
    for i in range(node_count-1):
        used_color = []
        for j in range(node_count):
            if edge_mat[i+1, j] == 1 and j in colors:
                used_color.append(colors[j])
        
        x = list(range(min(colors.values()), max(colors.values())+1))
        flag = True
        for k in x:
            if k in used_color:
                continue
            else:
                colors[i+1] = k
                flag = False
        
        if flag:
            colors[i+1] = max(used_color) + 1
        
    return colors.values() 

def cp_gurobi(node_count, edges):
    '''
    solve using CP by Gurobi
    '''
    G = nx.Graph()
    G.add_edges_from(edges)
    G.add_nodes_from([node for node in range(node_count)])

    m = gp.Model('coloring')
    m.setParam('TimeLimit', 17940)
    k = len(set(greedy(node_count, edges)))
    n = node_count

    x = []
    for l in range(n):
        x.append([])
        for j in range(k):
            x[-1].append(m.addVar(vtype='B', name='x_%d_%d' % (l, j), obj=0))

    y = m.addVars(k, vtype='B', name='y_%d' % j, obj=1)

    m.setObjective(y.sum('*'), gp.GRB.MINIMIZE)
    m.update

    for u in range(n):
        m.addConstr(gp.quicksum(x[u]) == 1)

    for u in range(n):
        for j in range(k):
            m.addConstr(x[u][j] <= y[j])

    for u in range(n):
        for v in G[u]:
            if v > u:
                for j in range(k):
                    m.addConstr(x[u][j] + x[v][j] <= 1)
    for j in range(k-1):
        m.addConstr(y[j] >= y[j+1])

    m.update()
    m.optimize()


    result = [0] * k
    for j in range(k):
        result[j] = int(y[j].getAttr(gp.GRB.Attr.X))

    x_result = np.zeros((n, k))
    for l in range(n):
        for j in range(k):
            x_result[l, j] = int(x[l][j].getAttr(gp.GRB.Attr.X))

    color = []
    for l in range(n):
        color.append(list(x_result[l, :]).index(1))

    return color

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))


    solution = cp_gurobi(node_count, edges)

    # prepare the solution in the specified output format
    output_data = str(len(set(solution))) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

