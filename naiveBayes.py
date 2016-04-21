dictionary = ["free", "try", "offer", "trial", "credit", "single", "clearance", "earnings", "cheap", "dollars", "debt", "traffic"]

spam = [
[0,1,0,1,0,1,0,1,0,1,0,1],
[1,1,1,1,1,0,0,0,0,1,0,0],
[1,0,0,0,0,1,0,0,0,0,0,0],
[1,0,0,0,0,1,0,0,0,0,0,0],
[1,0,0,0,0,1,0,0,0,0,0,0],
[1,0,0,0,0,1,0,0,0,0,0,0]
]

nonspam = [
[0,1,0,0,0,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0,0,0,0],
[0,0,1,0,0,0,0,0,0,1,0,0],
[0,0,0,0,1,0,0,0,0,1,1,0]
]

def spamClassifier(plusSample, negativeSample, dictionaryLength, testExample):
	noPlusExamples = len(plusSample)
	noNegativeExamples = len(negativeSample)
	plusProbabilityDistribution = laplaceSmooth(getProbabilityDistribution(plusSample), noPlusExamples)
	negativeProbabilityDistribution = laplaceSmooth(getProbabilityDistribution(negativeSample), noNegativeExamples)
	probabilityPositiveExample = float(noPlusExamples) / (noPlusExamples + noNegativeExamples)
	probabilityDataGivenPlusLabel = getProbabilityDataGivenLabel(testExample, plusProbabilityDistribution)
	numerator = probabilityDataGivenPlusLabel * probabilityPositiveExample
	probabilityDataGivenNegativeLabel = getProbabilityDataGivenLabel(testExample, negativeProbabilityDistribution)
	denominator = numerator + (probabilityDataGivenNegativeLabel * (1-probabilityPositiveExample))
	return numerator / denominator
	

def getProbabilityDistribution(sample):
	noTrainingExamples = len(sample)
	noFeatures = len(sample[0])
	probabilityDistribution = [0] * noFeatures
	for i in range(noTrainingExamples):
		for j in range(noFeatures):
			probabilityDistribution[j] += float(sample[i][j]) / noTrainingExamples
	return probabilityDistribution

def getProbabilityDataGivenLabel(trainingExample, probabilityDistribution):
	p = 1
	for i in range(len(trainingExample)):
		if trainingExample[i] == 0:
			p = p * (1-probabilityDistribution[i])
		else:
			p = p * probabilityDistribution[i]
	return p

def laplaceSmooth(probabilityDistribution, sampleSize):
	for i in range(len(probabilityDistribution)):
		p = probabilityDistribution[i]
		p = ((p * sampleSize)+1)/(sampleSize + 2)
		probabilityDistribution[i] = p
	return probabilityDistribution

for i in range(len(spam)):
	print spamClassifier(spam, nonspam, 12, spam[i])
for i in range(len(nonspam)):
	print spamClassifier(spam, nonspam, 12, nonspam[i])
		
	
