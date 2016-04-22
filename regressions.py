import copy
import math
import sys
from enum import Enum
from math import exp
from abc import ABCMeta, abstractmethod

class ErrorCalculationAlgorithm:
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculateError(self, trainingExample, weights): pass

class LinearAlgorithm(ErrorCalculationAlgorithm):
	def calculateError(self, trainingExample, weights):
		total = 0
		for i in range(len(weights)):
			total = total + weights[i] * trainingExample[i]
		#TODO depends on format of trainingexample...
		error = total - trainingExample[len(trainingExample)-1]
		return error

class RegressionTypes(Enum):
	linear = 1
	local = 2
	logistic = 3

class Regression:
	def __init__(self, algorithm, trainingSet, learningRate):
		self.setAlgorithm(algorithm)
		self.setTrainingSet(trainingSet)
		self.setlearningRate(learningRate)
		self.m = len(trainingSet)
		self.n = len(trainingSet[0])

	def setAlgorithm(self, algorithm):
		self.algorithm = algorithm

	def setTrainingSet(self, trainingSet):
		self.trainingSet = trainingSet

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate

	def performRegression(self):
		weights = self.initializeWeights()
		#while(TODO: convergence condition):
		for c in range(100):
			weightsForThisIteration = copy.deepcopy(weights)
			for featureIndex in range(self.n):
				error = 0
				for i in range(self.m):
					#TODO Is it possible to remove parameters from method signature of calculateError?
					error = error + self.algorithm.calculateError(self.trainingSet[i], weightsForThisIteration, self.n) * trainingSet[i][j]
				weightUpdate = self.trainingRate * error
				weights[featureIndex] = weights[featureIndex] - weightUpdate
		return weights

	def initializeWeights(self):
		#TODO Should this be defined in scope of performRegression()?
		return [1] * self.n

trainingSet = [
[1,2,3,250],
[1,4,10,700],
[1,3,2,130],
[1,6,12,900],
[1,1,1,100],
[1,4,4,300],
[1,12,15,1400],
[1,4,4,350],
[1,5,5,410],
]

x = Regression(LinearAlgorithm(), trainingSet, .05)

'''
def guess(trainingExample, theta):
	total = 0
	for i in range(len(theta)):
		total = total + theta[i] * trainingExample[i]
	error = total - trainingExample[len(trainingExample)-1]
	return error

def guessLocal(trainingExample, theta, x):
	def getWeight(trainingExample, theta, x, bandwidth):
		distance = 0 
		for i in range(len(theta)):
			distance = distance + math.pow((x[i] - trainingExample[i]), 2)
		w = math.exp(-distance/math.pow(bandwidth,2))
		return w
	total = 0
	for i in range(len(theta)):
		total = total + theta[i] * trainingExample[i]
	error = total - trainingExample[len(trainingExample)-1]
	return getWeight(trainingExample, theta, x, 1) * error

def guessLogistic(trainingExample, theta):
	total = 0
	for i in range(len(theta)):
		total = total + theta[i] * trainingExample[i]
	z = 1 / (1 + math.exp(-total))
	error = z - trainingExample[len(trainingExample)-1]
	return error
			
	
trainingSet = [
[1,2,3,250],
[1,4,10,700],
[1,3,2,130],
[1,6,12,900],
[1,1,1,100],
[1,4,4,300],
[1,12,15,1400],
[1,4,4,350],
[1,5,5,410],
]

logisticTrainingSet = [
[1,72,180,1],
[1,69,165,1],
[1,74,215,1],
[1,65,150,1],
[1,67,145,0],
[1,60,100,0],
[1,65,122,0],
[1,63,110,0],
[1,66,125,0],
]

coefficients = [-2.552570613007586, -4.237065780247473, 1.913292406642348]
#linearRegression(trainingSet, .01, "linear")
linearRegression(logisticTrainingSet, .005, "logistic")


#Test logistic results
"""
for example in logisticTrainingSet:
	guess = 0
	for i in range(len(example)):
		if i == len(example)-1:
			continue
		guess = guess + example[i]*coefficients[i]*10
	print guess
	print example[len(example)-1]
	print "\n"
"""	

'''

