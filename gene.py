import numpy as np


class gene(object):

    def __init__(self, geneID, geneType, binID = -1):

        """
        geneType: 'MR' master regulator or 'T' target
        bindID is optional
        """

        self.ID = geneID
        self.Type = geneType
        self.Conc = []
        self.dConc = []
        self.binID = binID
        self.simulatedSteps_ = 0
        self.converged_ = False

    def append_Conc (self, currConc):
        if currConc < 0:
            currConc = 0
        self.Conc.append(currConc)

    def append_dConc (self, currdConc):
        self.dConc.append(currdConc)

    def clear_Conc (self):
        """
        This method clears all the concentrations except the last one that may
        serve as intial condition for rest of the simulations
        """
        self.Conc = self.Conc[-1:]

    def clear_dConc (self):
        self.dConc = []

    def incrementStep (self):
        self.simulatedSteps_ += 1

    def setConverged (self):
        self.converged_ = True

    def set_scExpression(self, list_indices):
        """
        selects input indices from self.Conc and form sc Expression
        """
        self.scExpression = np.array(self.Conc)[list_indices]
