import numpy
from numpy import linalg

def psa(trainingSet, k):
	covarMatrix = getCovarianceMatrix(trainingSet)
	if covarMatrix == None:
		return None
	ev = linalg.eigh(covarMatrix)
	dim = len(trainingSet[0])
	if k > dim:
		k = dim
	return ev[1][len(trainingSet[0])-k:]

def getCovarianceMatrix(trainingSet):
	if len(trainingSet) == 0:
		return None
	dim = len(trainingSet[0])
	if dim == 0:
		return None
	covarMatrix = [[0 for i in range(dim)] for j in range(dim)]
	for trainingExample in trainingSet:
		for columnNum in range(dim):
			for rowNum in range(dim):
				covarMatrix[rowNum][columnNum] += (trainingExample[rowNum] * trainingExample[columnNum])
	for columnNum in range(dim):
		for rowNum in range(dim):
			covarMatrix[rowNum][columnNum] /= float(len(trainingSet))
	return covarMatrix

trainingSet = [[1,2],[3,4],[6,9],[3,5],[1,4],[5,7],[4,8],[3,6],[2,6],[4,5],[3,7]]

print psa(trainingSet, 3)
