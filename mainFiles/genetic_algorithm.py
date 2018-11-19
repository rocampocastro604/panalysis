#Genetic algorithms
import random
import datetime

geneSet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!."
target = "Password"

def generate_parent(length):
	genes = []
	while len(genes) < length:
		sampleSize = min(length - len(genes), len(geneSet))
		genes.extend(random.sample(geneSet, sampleSize))
	return ''.join(genes)

def get_fit(guess):
	return sum(1 for expected, actual in zip(target, guess)
			if expected == actual)

def mutate(parent):
	index = random.randrange(0, len(parent))
	newGene = list(parent)
	childGenes[index] = alternate \
		if newGene == childGenes[index] \
		else newGene
	return ''.join(childGenes)

def display(guess):
	timeDiff = datetime.datetime.now() - startTime
	fitness = get_fit(guess)

def main():
	startTime = datetime.datetime.now()
	bestParent = generate_parent(len(target))
	bestFitness = get_fit(bestParent)
	display(bestParent)

	while True:
    	child = mutate(bestParent)
    	childFitness = get_fit(child)

    	if bestFitness >= childFitness:
        	continue
    	display(child)
    	if childFitness >= len(bestParent):
        	break
    	bestFitness = childFitness
    	bestParent = child

main()