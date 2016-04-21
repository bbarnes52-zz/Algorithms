import math
from math import exp

def linearRegression(trainingSet, alpha, algorithm, x=None):
	numFeatures = len(trainingSet[0])-1
	theta = [1]*numFeatures
	thetaCopy = theta
	for c in range(1000000):
		for j in range(len(theta)):
			error = 0
			for i in range(len(trainingSet)):
				if algorithm == "local":
					error = error + guessLocal(trainingSet[i], thetaCopy, x)*trainingSet[i][j]
				elif algorithm == "logistic":
					error = error + guessLogistic(trainingSet[i], thetaCopy) * trainingSet[i][j]
				else:
					error = error + guess(trainingSet[i], thetaCopy)*trainingSet[i][j]
			update = alpha * error
			theta[j] = theta[j]-update
		thetaCopy = theta
		print theta
		

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
