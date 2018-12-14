# CS598JP_CourseProject
This repository contains our object-oriented code for GRN inference pipeline written for CS598JP course project. This consists of the following objects: data_tools, simulator, perturber, gene, and optimizer. 

main.py is the one to run for GRN inference. This requires a flag --graph_index to be passed. This shows the index of the graph (zero-based indexing) to be read from all_graphs.csv file. Therefore, the greedy graph search is done by passing all graphs in separate runs to main.py. The current all_graphs.csv file contains all DAGs for 4 vertices (4 genes). It also excludes DAGs that contain master regulators with no outgoing edge. To test the pipeline for other graph structures, one should prepare an appropriate all_graphs.csv file. Each line represents a graph. Particularly, each line shows the linearized version of adjacency matrix for a graph when diagonal elements are excluded. 

simulator: contains code for synthetic data generation. It uses gene object at its backend.

optimizer: Contains two class of optimizers (simulated annealing, CMAES-GG). Due to the poor performance of RBFopt, it was not included here.

Perturbur: proposes a new state of graph and parameters based on user-defined probabilities. 

data_tools: contain single-cell parser used for parsing the real single-cell RNAseq data.

