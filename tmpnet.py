import numpy as np
import pprint as pp

def neuralnet(trainingSet, netStructure):
	#TODO(Implement me!)
	return None

def initializeTransitionMatrices(netStructure):
	numTransitionMatrices = len(netStructure)-1
	thetaArray = []
	for i in range(numTransitionMatrices):
		theta = np.ones((netStructure[i+1], netStructure[i]+1))
		thetaArray.append(theta)
	return thetaArray

def forwardProp(trainingSet, thetaArray):
	activationMatrix = []
	for example in trainingSet:
		v = example[0]
		label = example[1]
		activationArray = []
		for transitionMatrix in thetaArray:
			v = np.insert(v, 0, 1)  #add bias component
			v = np.dot(transitionMatrix, v)
			activationArray.append(v)
		activationMatrix.append(activationArray)
	return activationMatrix

def backwardProp(trainingSet, activationMatrix, thetaArray):
	numTransitionMatrices = len(thetaArray)
	errorMatrix = []
	for i in range(len(trainingSet)):
		errorMatrix.append([])
		outputError = np.subtract(activationMatrix[i][numTransitionMatrices-1],trainingSet[i][1])
		errorMatrix[i].append(outputError)
		for j in reversed(range(numTransitionMatrices)):
			error = np.dot(np.matrix.transpose(thetaArray[j]), errorMatrix[i][len(errorMatrix[i])-1])
			aLayer = np.insert(activationMatrix[i][j-1], 0, 1)
			activationDerivative = np.multiply(aLayer, np.subtract(np.ones(len(aLayer)), aLayer))
			error = np.multiply(error, activationDerivative)
			errorMatrix[i].append(error)
			pp.pprint(error)
		pp.pprint(errorMatrix)

trainingSet = [
([5000,5,3], [700]),
([1600,3,2],[330]),
([2000,3,3],[350]),
([4000,4,3],[500]),
([700,1,0],[110]),
([1200,1,1],[270]),
([3300,3,3],[400]),
([4000,4,2],[450]),
([7000,6,5],[800]),
([10000,8,8],[1200]),
([500,1,0],[100]),
([1200,2,1],[170]),
([3000,2,2],[400]),
([2500,3,2],[290]),
]

#Update this array to configure the number of activation layers and then number of nodes per layer.
netStructure = [len(trainingSet[0][0]), 5, len(trainingSet[0][1])]

thetaArray = initializeTransitionMatrices(netStructure)
activationMatrix = forwardProp(trainingSet, thetaArray)
backwardProp(trainingSet, activationMatrix, thetaArray)
	
