import numpy as np
import pandas as pd
from scipy.spatial import KDTree

class scParser (object):
    def __init__(self, sc_data_file):

        sc_DataFrame = pd.read_csv(sc_data_file, index_col = 0)
        self.data_ = sc_DataFrame.values
        self.geneNames_ = np.array(sc_DataFrame.axes[0])
        self.cellLabels_ = np.array(sc_DataFrame.axes[1])
        self.nGenes_ = len(self.geneNames_)
        self.nCells_ = len(self.cellLabels_)


    def set_point_coords (self, geometry_file):
        """
        This function expects a white-space delimited file. Each line shows a point
        Columns are x, y, z coordinates on embryo.
        """
        point_DataFrame = pd.read_csv(geometry_file, sep = ' ')
        self.pointCoords_ = point_DataFrame.values

        self.minX = min(self.pointCoords_[:,0])
        self.minY = min(self.pointCoords_[:,1])
        self.minZ = min(self.pointCoords_[:,2])

        self.maxX = max(self.pointCoords_[:,0])
        self.maxY = max(self.pointCoords_[:,1])
        self.maxZ = max(self.pointCoords_[:,2])

    def set_cell_mapping (self, mapping_file, threshold = np.exp(0.5)):
        """
        Input is a mapping file contating exp(MCC) between all cell-point pairs

        Set a mapping between all high score cell and points. Note that the whole class
        assumes points' indices start from 0.
        """

        mapping_DataFrame = pd.read_csv(mapping_file, index_col = 0)
        mapping_array = mapping_DataFrame.values
        self.mapping_ = {}

        for point in mapping_DataFrame.axes[0].tolist():
            pointIdx = point - 1
            cellScores = mapping_array[pointIdx]
            self.mapping_[pointIdx] = [] # pointIdx starts from 0

            for cIdx, cScore in enumerate(cellScores):
                if cScore >= threshold:
                    self.mapping_[pointIdx].append(cIdx)

    def get_expression (self, list_genes, window_low = [0,0,0], window_high = [1,1,1], num_bin_list = None):
        """
        returns a list containing center coordinates of all bins
        returns a list containing np.arrays (1 per each bin) for expression of given genes in all cells lie within a bin
        binIndices match between the two returning lists
        returns list of numpy 1d arrays of cell labels in each bin

        window_low: a list of lower boundaries (x,y,z) of the scanning window. Entries between 0 and 1
        that specifiy distance ratios on the corresponding axis.

        window_high: a list of upper boundaries (x,y,z) of the scanning window. Entries between 0 and 1
        that specifiy distance ratios on the corresponding axis.

        num_bin_list: a list of size 3, one per each direction (x,y,z), containing number of bin in corresponding direction
        1 bin denotes no binning.
        """
        # find window's start and end coordinates
        xStart = window_low[0] * (self.maxX - self. minX) + self.minX
        xEnd = window_high[0] * (self.maxX - self. minX) + self.minX

        yStart = window_low[1] * (self.maxY - self. minY) + self.minY
        yEnd = window_high[1] * (self.maxY - self. minY) + self.minY

        zStart = window_low[2] * (self.maxZ - self. minZ) + self.minZ
        zEnd = window_high[2] * (self.maxZ - self. minZ) + self.minZ

        points = []

        # find and set bin properties
        # binBoundaries will be a numpy aray of size nBins * 6 containing lower and
        # boundaries of each bin (xl,yl,zl,xh,yh,zh)
        # binCenters will be a numpy array containing center coordinates of all bins
        nBins = np.prod(num_bin_list)
        dx = np.true_divide(xEnd - xStart, num_bin_list[0])
        dy = np.true_divide(yEnd - yStart, num_bin_list[1])
        dz = np.true_divide(zEnd - zStart, num_bin_list[2])

        binBoundaries = []
        binCenters = []
        for bx in range(num_bin_list[0]):
            currMinX = xStart + dx * bx
            currMaxX = xStart + dx * (1+bx)
            for by in range(num_bin_list[1]):
                currMinY = yStart + dy * by
                currMaxY = yStart + dy * (by+1)
                for bz in range(num_bin_list[2]):
                    currMinZ = zStart + dz * bz
                    currMaxZ = zStart + dz * (bz+1)

                    currBounds = [currMinX, currMinY, currMinZ, currMaxX, currMaxY, currMaxZ]
                    currCenters = [np.true_divide(currMinX+currMaxX,2), np.true_divide(currMinY+currMaxY,2), np.true_divide(currMinZ+currMaxZ,2)]
                    binBoundaries.append(currBounds)
                    binCenters.append(currCenters)

        # This is KDTree containing bin centers, we will query this tree to associate
        # point to bins
        kdTree = KDTree(binCenters)


        # build two dictionaries, one for mapping bins to all the points lie in that bin
        # one for mapping bins to all the cells that lie in that bin
        bin2points = {}
        bin2cells = {}
        for i in range(nBins):
            bin2points[i] = []
            bin2cells[i] = []

        for pIdx, pCoord in enumerate(self.pointCoords_):
            if xStart <= pCoord[0] <= xEnd and yStart <= pCoord[1] <= yEnd and zStart<= pCoord[2] <= zEnd:
                binIdx = kdTree.query(pCoord)[1]
                bin2points[binIdx].append(pIdx)
                bin2cells[binIdx] += self.mapping_[pIdx]

        # binExpression is a list containing 2d expression array (gene-cells) for each bin
        binExpression = []
        gIndices = [self.geneNames_.tolist().index(g) for g in list_genes]
        gExpression = np.take(self.data_, gIndices, axis = 0)
        # a list containing array of all cells' labels in each bin
        binCellLabels = []
        for b in bin2cells.keys():
            # first get unique cells
            bin2cells[b] = np.unique(bin2cells[b]).tolist()
            binCellLabels.append(np.take(self.cellLabels_, bin2cells[b]))

            # following line is a fancy indexing, all it does is to take rows and Columns
            # corresponding to given genes and cells in current bin from expression data
            binExpression.append(np.take(gExpression, bin2cells[b], axis = 1))

        return binCenters, binExpression, binCellLabels
