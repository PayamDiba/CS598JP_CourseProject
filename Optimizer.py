from Perturber import perturber
from Simulator import simulator
from sklearn.preprocessing import minmax_scale
from scipy.stats import wasserstein_distance
import numpy as np
import copy
import os
import csv
import shutil
import cma
import sys

class simulated_annealing (object):
    def __init__(self, number_bins, number_genes, number_sc, real_data):
        """
        genes' and bins' and numbers should be consistent with real_data
        in real data, different bins may have different number sc

        real_data: list of np.array(#genes * #cells)

        name_initial_targets_file: the initial state of graphs, especially the targets'
        file that is fed to simulator. Initial regulators' file is automatically inferred

        number_sc: number of single cells for simulations, this might not be the same as
        the number of single cells in the real data
        """

        self.nBins_ = number_bins
        self.nGenes_ = number_genes
        self.nSC_ = number_sc
        self.real_data_ = real_data
        self.normalizedRealData_ = self.Normalize_(self.real_data_)
        self.meanRealData_ = self.calculate_mean_exp_(self.real_data_)

    def Normalize_(self, data):
        """
        data is list of np.arrays(#genes * #cells) of size #bins

        This function normalizes the data by min-max normalization. Expression of each gene
        in all bins and cells is min-max normalized to range [0,1]. Returns the normalized data
        """
        nCells = [] # a list containing number of cells in each bin
        for binExp in data:
            nCells.append(np.shape(binExp)[1])

        ret = copy.deepcopy(data)
        for i in range(self.nGenes_):
            currGeneExp = np.array([])
            for binExp in ret:
                currGeneExp = np.concatenate((currGeneExp, binExp[i,:]))
            normalized_allBins = minmax_scale(currGeneExp, feature_range=(0.0016, 1))

            normalized_data = []
            j = 0
            for nc in nCells:
                normalized_data.append(np.array(normalized_allBins[j:j+nc]))
                j += nc

            for bIdx, bExp in enumerate(normalized_data):
                ret[bIdx][i, :] = bExp

        return ret

    def calculate_mean_exp_(self, data):
        """
        data is a list of np.arraya(#genes * #cells) of size #bins

        returns a np.array (#bins * # genes)
        """
        ret = np.zeros((self.nBins_, self.nGenes_))
        for g in range(self.nGenes_):
            for b in range(self.nBins_):
                ret[b,g] = np.mean(data[b][g,:])

        return ret



    def write_regs_file_ (self, idx_master_regs, decay, output_file):
        """
        idx_master_regs: list of master regulators' indices

        decay: decay parameter. If scalar, uses the same value for all genes. Otherwise,
        it is a list with size self.nGenes_. Then relevnt decays are obtained by idx_master_regs.
        """
        if np.isscalar(decay):
            decay = np.repeat(decay, self.nGenes_)

        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            for i in idx_master_regs:
                meanExp = self.meanRealData_[:,i]
                prod_rate = np.multiply(decay[i], meanExp)
                curr_line = np.concatenate(([i],prod_rate)).tolist()
                writer.writerow(curr_line)

    def perturb_ (self, input_file_taregts, list_desired_perturbations, list_prob_perturbations, output_file_targets):
        """
        input_file_taregts: current state of targets and their interactions

        list_prob_perturbations: a list containing all desired perturbations to be considered
        possible perturbations: 'g': graph, 'k': interaction strength and type, 'n': hill coefficient or coop_state parameters

        list_prob_perturbations: list of probabilites associated to desired perturbations

        output_file_targets: the output file name for wirting the perturbed graph (only targets' file)

        returns a list of the master regulator indices in the perturbed graph
        """

        p = perturber(self.nGenes_, input_file_taregts)
        pert = p.select_perturbation(list_desired_perturbations, list_prob_perturbations)
        p.perturb(pert)
        p.write_interactions(output_file_targets)

        return list(p.master_regulators_idx_)

    def get_synthetic_ (self, noise_params, noise_type, decays, sampling_state, tol , window_length , dt , name_targets_file, name_regs_file, shared_coop_state, optimize_sampling = False):
        """
        This function runs the simulator to generate synthetic gene expression data

        It returns list of np.array(#genes * #cells) of size #bins
        """
        sim = simulator(number_genes = self.nGenes_, number_bins = self.nBins_, number_sc = self.nSC_, noise_params = noise_params, decays = decays, sampling_state = sampling_state, noise_type = noise_type, tol = tol, dt = dt, optimize_sampling = optimize_sampling)
        sim.build_graph(input_file_taregts = name_targets_file, input_file_regs = name_regs_file, shared_coop_state = shared_coop_state)
        sim.simulate()

        return sim.getExpressions()


    def calculate_objective_ (self, list_synthetic_expression, obj_type = 'w1'):
        """
        Evaluates objective function.

        obj_type: determines the type of objective function. Currently it supports:

        w1: Wasserstein 1 distance between distrbution of real and synthetic gene expressions
        cov: element-wise MSE between covariance matrix of real and synthetic data.
        cov-w1: sum of w1 and cov objectives
        """

        if obj_type == 'w1':
            return self.w1_distance_(list_synthetic_expression, self.normalizedRealData_)
        elif obj_type == 'cov':
            return self.elemWise_MSE_cov_(list_synthetic_expression, self.normalizedRealData_)
        elif obj_type == 'cov-w1':
            return self.w1_distance_(list_synthetic_expression, self.normalizedRealData_) + self.elemWise_MSE_cov_(list_synthetic_expression, self.normalizedRealData_)

    def w1_distance_ (self, list_synthetic_expression, list_real_data):
        """
        This function uses normalized real data by default
        """

        ret = []
        for b in range(self.nBins_):
            for g in range(self.nGenes_):
                ret.append(wasserstein_distance(list_synthetic_expression[b][g], list_real_data[b][g]))

        return np.mean(ret)

    def elemWise_MSE_cov_ (self, list_synthetic_expression, list_real_data):
        """
        This function uses normalized real data by default
        """

        ret = 0
        for b in range(self.nBins_):
            currCovReal = np.cov(np.log(list_real_data[b]))
            currCovSynth = np.cov(np.log(list_synthetic_expression[b]))

            ret += np.sum(np.power(currCovReal - currCovSynth, 2))

        return np.true_divide(ret, self.nBins_ * self.nGenes_ * self.nGenes_)

    def set_anneal(self, start_temperature, final_temperature, fraction):
        """
        This should be called by user, before calling optimize, to set annealing settings.

        fraction: a scaler between 0 to 1. Denotes the fraction of optimization steps during
        which temperature should reach to its final value
        """
        self.startT_ = start_temperature
        self.finalT_ = final_temperature
        self.currT_ = start_temperature
        self.annealFraction_ = fraction


    def acc_prob_(self, oldObj, newObj):
        """
        returns the acceptance probability for perturbed state.

        fraction_optimized: fraction of running steps = #currStep/#totalSteps
        """
        ret = np.power(np.true_divide(oldObj, newObj), self.currT_)
        return ret

    def optimize (self, name_initial_targets_file, number_steps, list_desired_perturbations, list_prob_perturbations, obj_type, decays, noise_params, noise_type = 'sp', sampling_state = 10, tol = 1e-3, window_length = 100, dt = 0.01, shared_coop_state = 0):
        """
        This is the main function to run simulated annealing. For now, we excluded decay and noise_params from optimization parameters.

        number_steps: number of simulated annealing steps to run
        list_desired_perturbations: list containing perturbations. possible perturbations: 'g', 'k', 'n'
        list_prob_perturbations: list containing perturbation probabilites. Should sum to 1.
        obj_type: Objective use for simulated annealing: 'w1' or 'cov' or 'cov-w1'
        decays: a scalar or list of decay params per genes. Conversion from scalar to list is taken care in all downstream functions
        noise_params: a scalar or list of noise params per genes. Conversion from scalar to list is taken care in all downstream functions
        noise_type: 'sp' or 'sd' or 'dpd'. SEE Simulator for more information
        sampling_state: number of steady state steps from which single cells are sampled is sampling_state * nSC_
        tol: convergence tolerance for simulator
        window_length: See simulator
        dt: see simulator
        shared_coop_state: see simulator
        """
        # Find initial regulator indices
        allGeneIndices = range(self.nGenes_)
        initTargets = []
        with open(name_initial_targets_file,'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                initTargets.append(int(row[0]))

        initRegs = np.setdiff1d(allGeneIndices, initTargets)

        # Write intial regulator files and set input files for simultor
        self.write_regs_file_(initRegs, decays, "regs.txt")


        # initilize sim_ann settings and do optimization
        shutil.copyfile(name_initial_targets_file, 'coppied_init_file.txt')
        currTargetFile = 'coppied_init_file.txt'
        currRegFile = "regs.txt"
        oldObj = np.Inf
        optimalObj = np.Inf

        for s in range(number_steps):
            synthExpression = self.get_synthetic_(noise_params, noise_type, decays, sampling_state, tol , window_length , dt , currTargetFile, currRegFile, shared_coop_state)
            normalizedSynthExp = self.Normalize_(synthExpression)
            currObj = self.calculate_objective_(normalizedSynthExp, obj_type = obj_type)
            print s
            print optimalObj
            # set current state and relevnt parameters
            if currObj < optimalObj:
                optimalObj = currObj
                shutil.copyfile(currTargetFile, 'optimal.txt')

            if currObj < oldObj:
                oldObj = currObj

            else:
                accP = self.acc_prob_(oldObj, currObj)
                print accP
                if np.random.uniform() > accP:
                    currTargetFile = "old_targets.txt"
                else:
                    oldObj = currObj


            os.rename(currTargetFile, "old_targets.txt")

            regsIdx = self.perturb_("old_targets.txt", list_desired_perturbations, list_prob_perturbations, "new_targets.txt")
            self.write_regs_file_(regsIdx, decays, "regs.txt")
            currTargetFile = "new_targets.txt"
            dT = np.true_divide(self.finalT_ - self.startT_, self.annealFraction_*number_steps)
            if np.true_divide(s, number_steps) < self.annealFraction_:
                self.currT_ += dT

class CMAES_GG (object):
    def __init__(self, number_bins, number_genes, number_sc, real_data, linearized_graph_as_list):
        """
        This is CMA-ES optimizer for optimizing interaction parameters for a
        fixed given graph. So it is useful ifmain greedy graph (GG) seaches

        Inputs:
        linearized_graph_as_list: a list that is equivalent to linearlized version
        of adj matrix with no diagonal elements. #TODO: Make it better!

        obj_type: 'w1', 'cov', or 'cov-w1'
        optimize_decays: boolean, whether optimize decay parameters
        optimize_noises: boolean, whether optimize noise parameters
        """

        self.nBins_ = number_bins
        self.nGenes_ = number_genes
        self.nSC_ = number_sc # This denotes number cell used for optimization not the number of cells in real data
        self.nKs_ = sum(linearized_graph_as_list)
        self.K_ = {} # mapps target idx to a list of all of its Ks. The list is ordered by regIdx
        self.real_data_ = real_data
        self.normalizedRealData_ = self.Normalize_(self.real_data_)
        self.meanRealData_ = self.calculate_mean_exp_(self.real_data_)
        self.adjMatrix_ = self.convert2adjMat_(linearized_graph_as_list)
        self.master_regulators_idx_ = self.find_Master_regs_(self.adjMatrix_)
        self.targets_idx_ = np.setdiff1d(range(self.nGenes_), self.master_regulators_idx_).tolist()


    def Normalize_(self, data):
        """
        data is list of np.arrays(#genes * #cells) of size #bins

        This function normalizes the data by min-max normalization. Expression of each gene
        in all bins and cells is min-max normalized to range [0,1]. Returns the normalized data
        """
        nCells = [] # a list containing number of cells in each bin
        for binExp in data:
            nCells.append(np.shape(binExp)[1])

        ret = copy.deepcopy(data)
        for i in range(self.nGenes_):
            currGeneExp = np.array([])
            for binExp in ret:
                currGeneExp = np.concatenate((currGeneExp, binExp[i,:]))
            normalized_allBins = minmax_scale(currGeneExp, feature_range=(0.0016, 1))

            normalized_data = []
            j = 0
            for nc in nCells:
                normalized_data.append(np.array(normalized_allBins[j:j+nc]))
                j += nc

            for bIdx, bExp in enumerate(normalized_data):
                ret[bIdx][i, :] = bExp

        return ret

    def calculate_mean_exp_(self, data):
        """
        data is a list of np.arraya(#genes * #cells) of size #bins

        returns a np.array (#bins * # genes)
        """
        ret = np.zeros((self.nBins_, self.nGenes_))
        for g in range(self.nGenes_):
            for b in range(self.nBins_):
                ret[b,g] = np.mean(data[b][g,:])

        return ret

    def convert2adjMat_(self, linearized_graph_as_list):
        number_genes = self.nGenes_
	expanded = [0]
        for l in range(number_genes - 1):
            for i in range(number_genes):
                expanded.append(linearized_graph_as_list.pop(0))
            expanded.append(0)

        return np.reshape(expanded, (number_genes, number_genes))

    def find_Master_regs_(self, adjMatrix):
        ret = []
        for i, row in enumerate(np.transpose(adjMatrix)):
            if sum(row) == 0:
                ret.append(i)

        return ret

    def write_interactions_(self, output_file):

        with open(output_file, 'w') as f:
            writer = csv.writer(f)
	    #print self.adjMatrix_
            for t in self.targets_idx_:
                allRegs = self.adjMatrix_[:,t].tolist()
                allRegs = np.nonzero(allRegs)[0].tolist()

                allK = self.K_[t]
                currLine = [t] + [len(allRegs)] + allRegs + allK
                writer.writerow(currLine)

        f.close()

    def write_regs_file_ (self, output_file):
        """
        idx_master_regs: list of master regulators' indices

        decay: decay parameter. If scalar, uses the same value for all genes. Otherwise,
        it is a list with size self.nGenes_. Then relevnt decays are obtained by idx_master_regs.
        """
        if np.isscalar(self.decays_):
            decay = np.repeat(self.decays_, self.nGenes_)
	else:
	    decay = self.decays_

        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            for i in self.master_regulators_idx_:
                meanExp = self.meanRealData_[:,i]
                prod_rate = np.multiply(decay[i], meanExp)
                curr_line = np.concatenate(([i],prod_rate)).tolist()
                writer.writerow(curr_line)

    def set_Ks_(self, params):
        '''
        parms: list of all optimization parameters

        Assigns Ks to their interaction. This is very confusing, it could be improved by
        using a better graph data structure.
        '''
        K = params[-self.nKs_:].tolist()
        for t in self.targets_idx_:
            curr_nRegs = len(np.nonzero(self.adjMatrix_[:,t])[0].tolist())

            if not t in self.K_.keys():
                self.K_[t] = np.zeros((curr_nRegs,1)).tolist()

            for j in range(curr_nRegs):
                self.K_[t][j] = K.pop(0)
	if K != []:
	    sys.exit()
    def w1_distance_ (self, list_synthetic_expression):
        """
        This function uses normalized real data by default
        """

        ret = []
        for b in range(self.nBins_):
            for g in range(self.nGenes_):
                ret.append(wasserstein_distance(list_synthetic_expression[b][g], self.real_data_[b][g]))

        return np.mean(ret)

    def elemWise_MSE_cov_ (self, list_synthetic_expression):
        """
        This function uses normalized real data by default
        """

        ret = 0
        for b in range(self.nBins_):
            currCovReal = np.cov(np.log(self.normalizedRealData_[b]))
            currCovSynth = np.cov(np.log(list_synthetic_expression[b]))

            ret += np.sum(np.power(currCovReal - currCovSynth, 2))

        return np.true_divide(ret, self.nBins_ * self.nGenes_ * self.nGenes_)

    def eval_objective_(self, params):
        """
        Note: following order is assumed for params: (decays), (noises), Ks
        paranthesis denote optional optimization parameters (might not exist)
        """
        if self.optDecays_:
            self.decays_ = params[0:self.nGenes_]

            if self.optNoises_:
                self.noises_ = params[self.nGenes_: 2*self.nGenes_]

        elif self.optNoises_:
            self.noises_ = params[0:self.nGenes_]

        self.set_Ks_(params)
        self.write_interactions_('target.csv')
        self.write_regs_file_('reg.csv')
        sim = simulator(number_genes = self.nGenes_, number_bins = self.nBins_, number_sc = self.nSC_, noise_params = self.noises_, tol=1e-1, noise_type='sp', decays=self.decays_, sampling_state=10)
        sim.build_graph(input_file_taregts = 'target.csv', input_file_regs = 'reg.csv', shared_coop_state=2)
        sim.simulate()
        currSynthData = sim.getExpressions()

        if self.objType_ == 'w1':
            return self.w1_distance_(currSynthData)
        elif self.objType_ == 'cov':
            normalizedSynthExp = self.Normalize_(currSynthData)
            return self.elemWise_MSE_cov_(normalizedSynthExp)
        elif self.objType_ == 'cov-w1':
            normalizedSynthExp = self.Normalize_(currSynthData)
            return self.w1_distance_(currSynthData) + self.elemWise_MSE_cov_(normalizedSynthExp)

    def optimize(self, optimize_decays, optimize_noises, obj_type, decays = None, noises = None):
        """
        obj_type: 'w1', 'cov', or 'cov-w1'
        optimize_decays: boolean, whether optimize decay parameters
        optimize_noises: boolean, whether optimize noise parameters

        Note that if optimize_decays and/or optimize_noises are false, decays and/or noises have to be specified
        """

        self.optDecays_ = optimize_decays
        self.optNoises_ = optimize_noises
        self.decays_ = decays
        self.noises_ = noises
        self.objType_ = obj_type

        # initilize optimization parameters:
        nOptPars = self.nKs_ + self.nGenes_ * (int(self.optDecays_) + int(self.optNoises_))
        x0 = np.ones((1,nOptPars))

        decay_low = 0
        decay_up = 10

        noise_low = 0.1
        noise_up = 10

        K_low = -300
        K_up = 300

	decay_low = np.repeat(decay_low, self.nGenes_ * int(self.optDecays_))
	noise_low = np.repeat(noise_low, self.nGenes_ * int(self.optNoises_))
	K_low = np.repeat(K_low, self.nKs_)

	decay_up = np.repeat(decay_up, self.nGenes_ * int(self.optDecays_))
	noise_up = np.repeat(noise_up, self.nGenes_ * int(self.optNoises_))
	K_up = np.repeat(K_up, self.nKs_)

	low_bounds = np.concatenate((decay_low, noise_low, K_low))
	up_bounds = np.concatenate((decay_up, noise_up, K_up))	
	
	print low_bounds
	print up_bounds

        es = cma.CMAEvolutionStrategy(x0, 1, {'bounds':[low_bounds, up_bounds],'tolx':0.1})
        es.optimize(self.eval_objective_)
        res = es.result
        Output = np.append(res[1], np.array([res[0]]))
        Output = np.reshape(Output, (1, Output.shape[0]))
        #filename = "Output_" + str(FLAGS.graph_index) + ".csv"
        #np.savetxt(filename, Output, delimiter=',')
        return Output
