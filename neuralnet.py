import numpy as np
import pprint as pp
import random

def matrixFactory(netStructure, matrixType):
	numTransitionMatrices = len(netStructure)-1
	mArray = []
	for i in range(numTransitionMatrices):
		if matrixType == "transition":
			m = np.ones((netStructure[i+1], netStructure[i]+1))
		elif matrixType == "delta" or matrixType == "derivative":
			m = np.zeros((netStructure[i+1], netStructure[i]+1))
		else:
			return None
		mArray.append(m)
	return mArray

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

def getErrorMatrix(trainingSet, activationMatrix, thetaArray):
	numTransitionMatrices = len(thetaArray)
	errorMatrix = []
	for i in range(len(trainingSet)):
		errorMatrix.append([])
		outputError = np.subtract(activationMatrix[i][numTransitionMatrices-1],trainingSet[i][1])
		errorMatrix[i].append(outputError)
		for j in reversed(range(numTransitionMatrices-1)):
			error = np.dot(np.matrix.transpose(thetaArray[j+1]), errorMatrix[i][len(errorMatrix[i])-1])
			aLayer = activationMatrix[i][j]
			activationDerivative = np.multiply(aLayer, np.subtract(np.ones(len(aLayer)), aLayer))
			error = np.multiply(error, activationDerivative)
			errorMatrix[i].insert(0, error)
	return errorMatrix

def calculateDelta(activationMatrix, errorMatrix, deltaArray):
	for i in range(len(activationMatrix)):
		for layerNum in range(len(errorMatrix[i])):
			tmp = np.outer(errorMatrix[i][layerNum], activationMatrix[i][layerNum])
			deltaArray[layerNum] = np.add(deltaArray[layerNum], np.outer(errorMatrix[i][layerNum], activationMatrix[i][layerNum]))
	for i in range(len(deltaArray)):
		deltaArray[i] /= len(activationMatrix)
	return deltaArray

def removeBiasColumn(thetaArray):
	thetaArrayUnbiased = []
	for i in range(len(thetaArray)):
		thetaArrayUnbiased.append(np.matrix.copy(thetaArray[i]))
		thetaArrayUnbiased[i] = np.delete(thetaArrayUnbiased[i], 1, 1)
	return thetaArrayUnbiased

def adjustActivationMatrix(activationMatrix, trainingSet):
	for i in range(len(trainingSet)):
		activationMatrix[i].insert(0, np.asarray(trainingSet[i][0]))
		for j in range(len(activationMatrix[i])-1):
			activationMatrix[i][j] = np.insert(activationMatrix[i][j], 0, 100)
	return activationMatrix

def getDerivativeArray(deltaArray, thetaArrayUnbiased, regCoefficient):
	for l in range(len(deltaArray)):
		for i in range(len(deltaArray[l])):
			for j in range(len(deltaArray[l][i])):
				if j==0:
					continue
				else:
					deltaArray[l][i][j] = deltaArray[l][i][j] + regCoefficient * thetaArrayUnbiased[l][i][j-1]
	return deltaArray

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

#Initialize guesses at weights
thetaArray = matrixFactory(netStructure, "transition")

for l in range(len(thetaArray)):
	for i in range(len(thetaArray[l])):
		for j in range(len(thetaArray[l][i])):
			if l == 0:
				if j == 0:
					thetaArray[l][i][j] = float(random.randint(5,100))
				if j == 1:
					thetaArray[l][i][j] = float(random.randint(1,10)) / 50
				if j == 2:
					thetaArray[l][i][j] = float(random.randint(10,100))
				if j == 3:
					thetaArray[l][i][j] = float(random.randint(10,100))
			else:
				thetaArray[l][i][j] = float(random.randint(1,100)) / 50

def gradientDescent(derivativeArray, thetaArray, alpha):
	for l in range(len(derivativeArray)):
		derivativeArray[l] *= alpha
		thetaArray[l] = np.subtract(thetaArray[l], derivativeArray[l])
	return thetaArray
	

def neuralnet(trainingSet, netStructure, thetaArray):
	for i in range(9):
		activationMatrix = forwardProp(trainingSet, thetaArray)
		thetaArrayUnbiased = removeBiasColumn(thetaArray)
		errorMatrix = getErrorMatrix(trainingSet, activationMatrix, thetaArrayUnbiased)
		activationMatrix = adjustActivationMatrix(activationMatrix, trainingSet)
		deltaArray = matrixFactory(netStructure, "delta")
		deltaArray = calculateDelta(activationMatrix, errorMatrix, deltaArray)
		derivativeArray = getDerivativeArray(deltaArray, thetaArrayUnbiased, 1)
		thetaArray = gradientDescent(derivativeArray, thetaArray, .0000000000000005)
	return thetaArray

thetaArray = neuralnet(trainingSet, netStructure, thetaArray)

final = forwardProp(trainingSet, thetaArray)

pp.pprint(final)
print "\n"
pp.pprint(trainingSet)
