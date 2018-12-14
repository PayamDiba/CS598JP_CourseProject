from itertools import permutations
from itertools import combinations
import numpy as np
import networkx as nx
from data_tools import scParser
#import rbfopt
import cma
import csv
from Simulator import simulator
import random
from absl import flags
from absl import logging
from absl import app
from Optimizer import CMAES_GG


FLAGS = flags.FLAGS

flags.DEFINE_integer('graph_index', None, 'Index of the the graph to be read from the graph file')

def main(argv):

    '''Read current graph'''
    with open('all_graphs.csv','r') as f:
        reader = csv.reader(f, delimiter=',')
        curr_row = [row for idx, row in enumerate(reader) if idx == FLAGS.graph_index]

    Graph = []
    for i in curr_row[0]:
        Graph.append(int(np.float(i)))

    '''real real data'''
    data=scParser('dge_normalized.csv')
    data.set_point_coords('geometry.txt')
    data.set_cell_mapping('mapping.txt',np.exp(0.55))
    binCenters, binExp, binCellLabels =data.get_expression(['eve','hb','gt','Kr'],[0.33,0.5,0.2],[0.44,1,0.8],[4,1,1])

    '''make synthetic data as GT'''
    sim=simulator(number_genes=4,number_bins=4,number_sc=500,noise_params=0.5,noise_type='sp',decays=0.7,sampling_state=10)
    sim.build_graph(input_file_taregts='target_synthetic.csv',input_file_regs='reg_synthetic.csv',shared_coop_state=2)
    sim.simulate()
    A = sim.getExpressions()
    synExpGT = []
    for i in range(A.shape[0]):
        synExpGT.append(A[i,:,:])

    '''Run pipeline'''
    opt = CMAES_GG(number_bins = 4, number_genes = 4, number_sc = 500, real_data = binExp, linearized_graph_as_list = Graph)
    result = opt.optimize(optimize_decays = True, optimize_noises = True, obj_type = 'w1')

    filename = "Output_" + str(FLAGS.graph_index) + ".csv"
    np.savetxt(filename, result, delimiter=',')

if __name__ == "__main__":
    app.run(main)
