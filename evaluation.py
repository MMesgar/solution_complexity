#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:15:11 2017

@author: mesgarmn
"""
import numpy as np
from scipy.stats import kendalltau
from collections import defaultdict
from process_data import load_problem_solutions
rng = np.random.RandomState(seed = 2398)
#%%
def compute_kendalltau(gold_order, sys_order):
    tau, p_value = kendalltau(gold_order, sys_order)
    return tau, p_value
#%%
def fold_output_evaluation(fold_path, prob_solutions):
    '''
    fold_path: is the path to the error file of a fold
    prob_solutions: is a dict with problem as key and dict of solutions as value
    '''
    # load information in fold_output. output is a dictt= {prob: [(solution, solution, label)]}
    nn_output = load_fold(fold_path)
    print('num_of_problems_in_testset_of_fold : %d'%len(nn_output))
    
    #we need a dict to keep track of  taus for  problems in the testset
    results = {}
    taus = []
    
    
    rank_1_correct = 0
    rank_n_correct = 0
    rank_both_correct = 0
    total = 0
    
    
    # compute kendall for each problem
    for prob_id in nn_output.keys():
        
        # get gold order from the prob_solutions
        gold_order = get_gold_order(prob_id, prob_solutions)
        
        #build order graph
        pairwise_orders = nn_output[prob_id]
        graph = get_solution_graph(pairwise_orders, prob_solutions[prob_id])
    
        
        #apply topological sort on graph
        #sys_order = topological_order(graph, nodes=gold_order)
        # or do greedy ordering
        sys_order = edited_topological_sort(graph, nodes= gold_order)
        
        print sys_order
        #compute kendal's tau
        tau, p_value = compute_kendalltau(gold_order=gold_order, sys_order=sys_order)
        
        #update results
        results[prob_id]= {'sys_order':sys_order, 'gold_order':gold_order, 'tau':tau, 'p_value':p_value}
        
        #update taus
        taus.append(tau)
        
        
        # update accuracy rakn1,rankn, both
        total += 1
        if sys_order[0]== gold_order[0]:
            rank_1_correct += 1
            
        if sys_order[-1] == gold_order[-1]:
            rank_n_correct += 1
        
        if (sys_order[0]==gold_order[0]) and (sys_order[-1] == gold_order[-1]):
            rank_both_correct += 1
            
    # fold's tau is mean of all taus    
    fold_tau= np.mean(taus)
    
    rank_1_acc = rank_1_correct / float(total)
    rank_n_acc = rank_n_correct / float(total)
    rank_both_acc = rank_both_correct / float(total)
    
    return fold_tau,results, [rank_1_acc, rank_n_acc, rank_both_acc]

#%%
def load_fold(fold_path):
    '''
    output: {prob_txt: [(sol_txt_1,sol_txt_2,label), (sol_txt_3, sol_txt_4,label),...]}
    '''
    with open(fold_path, 'r') as f:
        lines = f.readlines()
    output =   defaultdict(lambda:[])
    for line in lines:
        line_segments = line.split('@@')
        prob_0 = float(line_segments[0])
        prob_1 = float(line_segments[1])
        predicted_label = int(line_segments[2])
        gold_label = int(line_segments[3])
        problem_id = int(line_segments[4])
        problem_txt = line_segments[5].strip().lower()
        solution1_id = int(line_segments[6])
        solution1_txt = line_segments[7].strip().lower()
        solution2_id = int(line_segments[8])
        solution2_txt = line_segments[9].strip().lower()
        

        output[problem_id].append((problem_txt, solution1_id, solution1_txt, solution2_id, solution2_txt, predicted_label, prob_0, prob_1))
    return output
#%%
def get_gold_order(prob_id, prob_solutions):
    '''
    prob: is the text of the problem
    problem_solution: is a dict of solutions {sol_txt_1:3, sol_txt_2:2,sol_txt_3:1,....}
    output: [1,2,3,4]
    '''
    solutions = prob_solutions[prob_id]
    output=solutions.keys()
    output.remove('prob_txt')
    output.sort()
    return output
#%%
def get_solution_graph(pairwise_orders, solutions):
    '''
    pairwise_orders : is list of tuples [(prob_txt,sol1_id, sol1_txt, sol2_id, sol2_txt, label),...]
    output = adjacency list of the matrix as dict
    '''
    edges = []
    for order in pairwise_orders:
        problem_txt = order[0].strip().lower()
        solution1_id= int(order[1])
        solution1_text = order[2].strip().lower()
        solution2_id= int(order[3])
        solution2_text = order[4].strip().lower()
        label = int(order[5])
        prob_0 = float(order[6])
        prob_1 = float(order[7])
        
        edges.append((solution1_id,solution2_id,prob_1))
        edges.append((solution2_id, solution1_id,prob_0))

    print 'classified paires:\n',pairwise_orders
    
    # the new version does not need any pruning
    pruned_edges = edges
#    pruned_edges = []
#    #pruned edges
#    for pair in edges:
#        if (pair[1], pair[0]) not in edges:
#            pruned_edges.append(pair)
#        else:
#            print('prune...',pair)
#    

    return pruned_edges
    
    
#%%
def edited_topological_sort(graph, nodes):
    """
    I inspired the idea from the "Learn to Order Things"  paper
    https://papers.nips.cc/paper/1431-learning-to-order-things.pdf
    graph: is a dictionary of edges {1:[(2,0.5),(3,0.2)],2:[(1,0.5),(3,0.6)],3:[(1,0.8),(2,0.4)]}
    {source:[(target_1, prob_1),(target2_prob_2),...],....,target_1:[(source, 1-prob_1)]}
    output: is a list of nodes
    """
    #if we cannot predict any pair order, then return a random order of nodes
    if len(graph)==0 :
        nodes_copy = nodes[:]
        rng.shuffle(nodes_copy)
        return nodes_copy
        
    # edge_set is consumed, need a copy
    edge_set = set([tuple(i) for i in graph])
	
    # node_list will contain the ordered nodes

    
    print 
    # build the adjacency matrix
    adj_matrix = np.zeros((len(nodes),len(nodes)))
    for edge in graph:
        source = edge[0]
        target = edge[1]
        weight = edge[2]
        adj_matrix[source,target]= weight
        

    #print 'adj_matrix=\n',adj_matrix
    
    # compute the sum of outgoing edges and sum of ingoing edges
    sum_outgoing_edges = np.sum(adj_matrix,axis=1)
    sum_ingoing_edges = np.sum(adj_matrix,axis=0)
    #print 'sum_outgoing_edges=\n',sum_outgoing_edges
    #print 'sum_ingoing_edges=\n',sum_ingoing_edges
    
    # comute p matrix
    p= list(sum_outgoing_edges - sum_ingoing_edges)
    #print 'p=\n',p
    
    V= nodes[:]
    node_list = list()
    while(len(node_list) < len(V)):
        t_index = np.argmax(p)
        #get t
        t = V[t_index]
        # add t to the output stack
        node_list.append(t)
        
        #update values of p
        for v in V:
            p[v] = p[v] +adj_matrix[t,v] - adj_matrix[v,t]
        
        
        #make t unreachable by setting it to -inf
        p[t_index] = -float('inf')
        
        
    return node_list
#%%
def topological_order(graph, nodes):
    '''
    graph: is a dictionary of edges {1:[2,3],2:[3]}
    output: is a list of nodes
    code mostly from:
        http://code.activestate.com/recipes/578406-topological-sorting-again/ 
    '''
    #if we cannot predict any pair order, then return a random order of nodes
    if len(graph)==0 :
        nodes_copy = nodes[:]
        rng.shuffle(nodes_copy)
        return nodes_copy
        
    # edge_set is consummed, need a copy
    edge_set = set([tuple(i) for i in graph])
	
    # node_list will contain the ordered nodes
    node_list = list()
	
    # source_set is the set of nodes with no incomming edges
    node_from_list, node_to_list = zip(* edge_set)
    #source_set = set(node_from_list) - set(node_to_list)
    source_set = set(nodes) -set(node_to_list)
    source_set = list(source_set)
    
    while len(source_set) != 0:
        if len(source_set)>1:
            print 'random selection from ', source_set
        # pop node_from off source_set and insert it in node_list
        rand_index = rng.randint(0,high=len(source_set))
        node_from = source_set.pop(rand_index)
        print 'process node %d'%node_from
        node_list.append(node_from)
        
        # find nodes which have a common edge with node_from
        from_selection = [e for e in edge_set if e[0] == node_from]
        for edge in from_selection :
            # remove the edge from the graph
            node_to = edge[1]
            edge_set.discard(edge)
            
            # if node_to don't have any remaining incomming edge :
            to_selection = [e for e in edge_set if e[1] == node_to]
            if len(to_selection) == 0 :
                # add node_to to source_set
                source_set.append(node_to)
    if len(edge_set) != 0 :
        print('not a direct acyclic graph')
        raise IndexError # not a direct acyclic graph
    else:
        return node_list

#%%
def code_check():
    corpus_path='data/corpus_all/corpus.txt'
    for problem_id in [21]:
    #,22, 24,32, 36, 37, 42, 46, 88, 104, 115,117, 118, 121, 130, 132, 155, 158, 181, 197,                    206, 209, 238, 240, 241, 243, 281, 288, 292, 299,]:
        print '************ problem_id = %d'%problem_id
        fold_path = 'evalutions/nonstatic_wrd2vec_2_3/fold2.error'
        fold_path = 'fold0.error'
        
        problem_solutions = load_problem_solutions(corpus_path, clean_string= True)
        gold_order = get_gold_order(problem_id, problem_solutions)
        if len(gold_order)==3:
            continue
        print('gold_order: %s'%gold_order)
        nn_output = load_fold(fold_path)
        pairwise_order = nn_output[problem_id] 
        print('num_pairwise_order= %d'%len(pairwise_order))
        graph= get_solution_graph(pairwise_order, problem_solutions[problem_id])
        print('graph:',graph)
        predicted_order = edited_topological_sort(graph, nodes = gold_order)
        print('predicted_order : %s'%predicted_order) 
    #    print('kendall tau : %s'%str(compute_kendalltau(gold_order,predicted_order)))
    #    
    #    fold_perf, results = fold_output_evaluation(fold_path=fold_path, prob_solutions=problem_solutions)
    #   #print('results=%s'%results)
    #    print('fold_kendall = %f'%fold_perf)
    return problem_solutions,nn_output
#%%
if __name__=="__main__":
    ps,out= code_check()
