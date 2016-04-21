import numpy
from numpy import random

import math

import scipy.stats
from scipy.stats import multivariate_normal

def gda(mean, cov, size):
	girlSample = random.multivariate_normal(mean[0], cov, size)
	guySample = random.multivariate_normal(mean[1], cov, size)
	empiricalGuyMean = [0, 0]; empiricalGirlMean = [0, 0]
	for i in range(size):
		empiricalGuyMean[0] += guySample[i][0]
		empiricalGuyMean[1] += guySample[i][1]
		empiricalGirlMean[0] += girlSample[i][0]
		empiricalGirlMean[1] += girlSample[i][1]
	empiricalGuyMean[0] = empiricalGuyMean[0]/size
	empiricalGuyMean[1] = empiricalGuyMean[1]/size
	empiricalGirlMean[0] = empiricalGirlMean[0]/size
	empiricalGirlMean[1] = empiricalGirlMean[1]/size
	heightVariance = 0
	weightVariance = 0
	covariance = 0
	for i in range(size):
		heightVariance += math.pow((guySample[i][0]-empiricalGuyMean[0]), 2)
		heightVariance += math.pow((girlSample[i][0]-empiricalGirlMean[0]), 2)
		weightVariance += math.pow((guySample[i][1]-empiricalGuyMean[1]), 2)
		weightVariance += math.pow((girlSample[i][1]-empiricalGirlMean[1]), 2)
		covariance += (guySample[i][0]-empiricalGuyMean[0]) * (guySample[i][1]-empiricalGuyMean[1])
		covariance += (girlSample[i][0]-empiricalGirlMean[0]) * (girlSample[i][1]-empiricalGirlMean[1])
	heightVariance = heightVariance / (size * 2)
	weightVariance = weightVariance / (size * 2)
	covariance = covariance / (size * 2)
	classify(empiricalGuyMean, empiricalGirlMean, [[heightVariance, covariance],[covariance,weightVariance]], girlSample)
	#print empiricalGuyMean
	#print empiricalGirlMean
	#print [[heightVariance, covariance], [covariance, weightVariance]]

def classify(guyMeans, girlMeans, covarianceMatrix, sample):
	for item in sample:
		pDataGivenGuy = multivariate_normal.pdf(sample, guyMeans, covarianceMatrix)
		pDataGivenGirl = multivariate_normal.pdf(sample, girlMeans, covarianceMatrix)
		pGuyGivenData = pDataGivenGuy / (pDataGivenGuy + pDataGivenGirl)
	print pGuyGivenData

covar = [[100,150],[150,400]]
gda([[156.1,156],[169.6,196]], covar, 20)
