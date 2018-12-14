import numpy as np
import csv
import random
import copy
import csv

class perturber (object):

    def __init__(self, number_genes, input_file_taregts, shared_coop_state = 0):
        """
        1- shared_coop_state: if >0 then all interactions are modeled with that
        coop state, and coop_states in input_file_taregts are ignored. Otherwise,
        coop states are read from input file. Reasonbale values ~ 1-3
        2- input_file_taregts: a csv file, one row per targets. Columns: Target Idx, #regulators,
        regIdx1,...,regIdx(#regs), K1,...,K(#regs), coop_state1,...,
            coop_state(#regs)
        """

        self.nGenes_ = number_genes
        self.maxEdges_ = 0.5 * number_genes * (number_genes - 1)
        self.graph_ = {}
        self.edges_ = {}

        for i in range(self.nGenes_):
            self.graph_[i] = {}
            self.graph_[i]['targets'] = []


        with open(input_file_taregts,'r') as f:
            reader = csv.reader(f, delimiter=',')
            if (shared_coop_state <= 0):
                edgeID = 0
                for row in reader:
                    nRegs = np.int(row[1])

                    ##################### Raise Error ##########################
                    if nRegs == 0:
                        print "Error: a master regulator (#Regs = 0) appeared in input"
                        sys.exit()
                    ############################################################

                    currInteraction = []
                    currParents = []
                    for regId, K, C_state in zip(row[2: 2 + nRegs], row[2+nRegs : 2+2*nRegs], row[2+2*nRegs : 2+3*nRegs]):
                        currInteraction.append((np.int(regId), np.float(K), np.float(C_state), 0)) # last zero shows half-response, it is modified in another method
                        currParents.append(np.int(regId))
                        self.graph_[np.int(regId)]['targets'].append(np.int(row[0]))
                        self.edges_[edgeID] = (np.int(regId), np.int(row[0])) # edgeID --> (start,end)
                        edgeID += 1

                    self.graph_[np.int(row[0])]['params'] = currInteraction
                    self.graph_[np.int(row[0])]['regs'] = currParents
            else:
                edgeID = 0
                for row in reader:
                    nRegs = np.int(row[1])
                    ##################### Raise Error ##########################
                    if nRegs == 0:
                        print "Error: a master regulator (#Regs = 0) appeared in input"
                        sys.exit()
                    ############################################################

                    currInteraction = []
                    currParents = []
                    for regId, K, in zip(row[2: 2 + nRegs], row[2+nRegs : 2+2*nRegs]):
                        currInteraction.append((np.int(regId), np.float(K), shared_coop_state, 0)) # last zero shows half-response, it is modified in another method
                        currParents.append(np.int(regId))
                        self.graph_[np.int(regId)]['targets'].append(np.int(row[0]))
                        self.edges_[edgeID] = (np.int(regId), np.int(row[0])) # edgeID --> (start,end)
                        edgeID += 1

                    self.graph_[np.int(row[0])]['params'] = currInteraction
                    self.graph_[np.int(row[0])]['regs'] = currParents

        # find master regulators and make sure that graph contains all genes
        self.master_regulators_idx_ = set()
        for i in range(self.nGenes_):
            if i not in self.graph_.keys():
                self.master_regulators_idx_.add(i)
                self.graph_[i]['params'] = []
                self.graph_[i]['regs'] = []
                self.graph_[i]['targets'] = []
            elif 'regs' not in self.graph_[i].keys():
                self.master_regulators_idx_.add(i)
                self.graph_[i]['regs'] = []
                self.graph_[i]['params'] = []


        self.nEdges_ = len(self.edges_.keys())

    def select_perturbation (self, list_of_free_params, prob_list):
        """
        list_of_free_params: conatins the parameters that could be perturbed. It
        should contain one or more of the followings:
        'g': graph
        'k': interactions' strength and sing
        'n': hill coefficient or coop_state parameters

        prob_list: the probability by which we select any of the perturbations.
        The two inputs should have the same size and a 1-to-1 correspondence between them is assumed.
        prob_list should sum to 1.

        Return: the type of perturbation, either 'g', 'k' or 'n'
        """

        random = np.random.uniform(0,1)

        i = 0
        Pi = prob_list[i]

        while (random > Pi):
            i += 1
            Pi += prob_list[i]

        return list_of_free_params[i]


    def perturb (self, selected_perturbation):
        """
        select_perturbation: the output of select_perturbation

        This function applies the perturbation on the graph or parameters.
        For now it assumes pre-defined probabilities, consider generalizing this
        to user defined probabilities.

        For the case of graph perturbation, it makes sure that the new graph is
        a DAG, otherwise it perturbs again until a DAG is achieved
        """

        if selected_perturbation == 'g':

            P = []
            P_dir_flip = 0.3
            P_del_edge = 0.7 * np.true_divide(self.nEdges_ - 1, self.maxEdges_ - 1)
            P_add_edge = 1 - P_dir_flip - P_del_edge

            P.append(P_dir_flip)
            P.append(P_add_edge)
            P.append(P_del_edge)

            randomNum = np.random.uniform(0,1)

            i = 0 # i=0: P_dir_flip,  i=1: P_add_edge,  i=2: P_del_edge
            Pi = P[i]

            while (randomNum > Pi):
                i += 1
                Pi += P[i]

            # Dir_flip: randomly select an edge and flip its direction
            if i == 0:
                graph_copy = copy.deepcopy(self.graph_)
                edges_copy = copy.deepcopy(self.edges_)
                masterRegs_copy = copy.deepcopy(self.master_regulators_idx_)
                repeat = True
                while (repeat):
                    self.graph_ = copy.deepcopy(graph_copy)
                    self.edges_ = copy.deepcopy(edges_copy)
                    self.master_regulators_idx_ = copy.deepcopy(masterRegs_copy)

                    random_edgeID = random.choice(self.edges_.keys())
                    previousStart = self.edges_[random_edgeID][0]
                    previousEnd = self.edges_[random_edgeID][1]

                    self.update_graph_('flip', random_edgeID)
                    self.edges_[random_edgeID] = (previousEnd, previousStart)

                    #TODO: implement this
                    repeat = not self.IsDAG_()

            # Add Edge
            if i == 1:
                graph_copy = copy.deepcopy(self.graph_)
                edges_copy = copy.deepcopy(self.edges_)
                masterRegs_copy = copy.deepcopy(self.master_regulators_idx_)
                repeat = True

                while (repeat):
                    self.graph_ = copy.deepcopy(graph_copy)
                    self.edges_ = copy.deepcopy(edges_copy)
                    self.master_regulators_idx_ = copy.deepcopy(masterRegs_copy)

                    random_start = np.random.randint(0,self.nGenes_)
                    random_end = np.random.randint(0,self.nGenes_)

                    # check if no edge is selected
                    if random_start == random_end:
                        continue

                    # check if the edges already exists
                    if random_start in self.graph_[random_end]['regs']:
                        continue

                    # check if an edge in reverse direction exists
                    if random_start in self.graph_[random_end]['targets']:
                        continue

                    edgeID = random.choice(np.setdiff1d(range(int(self.maxEdges_)), self.edges_.keys()))
                    self.edges_[edgeID] = (random_start, random_end)

                    self.update_graph_('add', edgeID)

                    #TODO: implement this
                    repeat = not self.IsDAG_()

                self.nEdges_ += 1

            # Delete Edge
            if i == 2:
                # In this case there is no need to repeat, since DAG remains DAG

                random_edgeID = random.choice(self.edges_.keys())


                self.update_graph_('delete', random_edgeID)
                self.edges_.pop(random_edgeID)
                self.nEdges_ -= 1

        if selected_perturbation == 'k':

            positive_scaling = random.choice([True, False])

            if positive_scaling:
                scale = np.random.uniform(0.5,2)
            else:
                scale = np.random.uniform(-2,-0.5)

            random_edgeID = random.choice(self.edges_.keys())
            start = self.edges_[random_edgeID][0]
            end = self.edges_[random_edgeID][1]

            interactionIDX = self.graph_[end]['regs'].index(start)
            oldTuple = self.graph_[end]['params'][interactionIDX]
            self.graph_[end]['params'][interactionIDX] = (oldTuple[0], scale * oldTuple[1], oldTuple[2], 0)

        if selected_perturbation == 'n':

            new_sate = np.random.uniform(1,3)
            random_edgeID = random.choice(self.edges_.keys())
            start = self.edges_[random_edgeID][0]
            end = self.edges_[random_edgeID][1]

            interactionIDX = self.graph_[end]['regs'].index(start)
            oldTuple = self.graph_[end]['params'][interactionIDX]
            self.graph_[end]['params'][interactionIDX] = (oldTuple[0], oldTuple[1], new_sate, 0)

    def update_graph_(self, edit_type, edge_ID):
        """
        edit_type: 'flip', 'add', 'delete'

        This function updates the graph (self.graph_) by editting the given edge
        based on the given edit_type. It also updates master_regulators_idx_.
        """
        previousStart = self.edges_[edge_ID][0]
        previousEnd = self.edges_[edge_ID][1]

        if edit_type == 'flip':
            self.graph_[previousStart]['targets'] = np.setdiff1d(self.graph_[previousStart]['targets'], previousEnd).tolist()

            idx = self.graph_[previousEnd]['regs'].index(previousStart)
            tupleInter = self.graph_[previousEnd]['params'].pop(idx)
            self.graph_[previousEnd]['regs'].pop(idx)

            newStart = previousEnd
            newEnd = previousStart
            tupleInter = (newStart, tupleInter[1], tupleInter[2], 0)
            self.graph_[newEnd]['regs'].append(newStart)
            self.graph_[newEnd]['params'].append(tupleInter)
            self.graph_[newStart]['targets'].append(newEnd)

            if previousStart in self.master_regulators_idx_:
                self.master_regulators_idx_.remove(previousStart)

            if self.graph_[newStart]['regs'] == []:
                self.master_regulators_idx_.add(newStart)

        if edit_type == 'add':

            # First generate a random K (either activator or repressor)
            # For now it uses pre-defined ranges, can take from user in later versions
            # For now it always sets coop_state = 2 rather than sampling for this
            positive = random.choice([True, False])

            if positive:
                randomK = np.random.uniform(30,70)
            else:
                randomK = np.random.uniform(-70,-30)

            self.graph_[previousStart]['targets'].append(previousEnd)
            self.graph_[previousEnd]['regs'].append(previousStart)
            self.graph_[previousEnd]['params'].append((previousStart, randomK, 2, 0))

            if previousEnd in self.master_regulators_idx_:
                self.master_regulators_idx_.remove(previousEnd)

        if edit_type == 'delete':
            self.graph_[previousStart]['targets'] = np.setdiff1d(self.graph_[previousStart]['targets'], previousEnd).tolist()

            idx = self.graph_[previousEnd]['regs'].index(previousStart)
            self.graph_[previousEnd]['params'].pop(idx)
            self.graph_[previousEnd]['regs'].pop(idx)

            if self.graph_[previousEnd]['regs'] == []:
                self.master_regulators_idx_.add(previousEnd)

    def IsDAG_(self):
        """
        This function finds cycles in graph. If it finds any cycle, then the graph
        is not DAG anymore an returns False, otherwise returns True.

        This functions is based on topological sorting and uses Kahn's algorithm.
        """
        G = copy.deepcopy(self.graph_)
        in_degree = np.zeros(self.nGenes_).tolist()
        queue = []
        for v in G.keys():
            in_degree[v] = len(G[v]['regs'])
            if in_degree[v] == 0:
                queue.append(v)

        numVisitedVerts = 0

        while queue:
            u = queue.pop(0)
            adjVerts = G[u]['targets']

            for i in adjVerts:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)
            numVisitedVerts += 1

        if numVisitedVerts != self.nGenes_:
            return False
        else:
            return True

    def write_interactions(self, output_file):

        out = []
        allTargets = np.setdiff1d(range(self.nGenes_), list(self.master_regulators_idx_)).tolist()

        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            for t in allTargets:
                allRegs = self.graph_[t]['regs']
                allK = [self.graph_[t]['params'][i][1] for i in range(len(allRegs))]
                allC = [self.graph_[t]['params'][i][2] for i in range(len(allRegs))]
                currLine = [t] + [len(allRegs)] + allRegs + allK + allC
                writer.writerow(currLine)

        f.close()
