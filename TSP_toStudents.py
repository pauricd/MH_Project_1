

"""
Author:
file:
"""

import random

from Individual import *
import sys
from timeit import default_timer as timer
import math

class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations):
        """
        Parameters and general variables
        """

        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.genSize        = None
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}

        self.readInstance()
        self.initPopulation()

    def generateIndividualFromKeys(self,keys,indv):
        indv.genes = keys
        return indv





    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data)
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print ("Best initial sol: ",self.best.getFitness())

    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            print ("iteration: ",self.iteration, "best: ",self.best.getFitness())

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        return [indA, indB]

    def rouletteWheel(self):
        """
        Your Roulette Wheel Selection Implementation
        """
        pass

    def uniformCrossover(self, indA, indB):
        child = [0] * self.genSize
        tmpA = {}
        tmpB = {}

        for i in range(0, self.genSize):
            if random.random() < 0.5:
                tmpA[indA.genes[i]] = False
                tmpB[indB.genes[i]] = False
            else:
                tmpA[indA.genes[i]] = True
                tmpB[indB.genes[i]] = True
                child.insert(i, indA.genes[i])
                del child[-1]

        lastAdded = 0
        for i in range(0, self.genSize):
            if not (tmpA[indA.genes[i]]):

                for b_counter in range(lastAdded, self.genSize):

                    if not indB.genes[b_counter] in child:
                        del child[i]
                        child.insert(i,indB.genes[b_counter])
                        lastAdded = b_counter
                        break

        return child

    def cycleCrossover(self, indA, indB):

        child = [0] * self.genSize
        #First try identify cycles

        tmpA = {}
        tmpB = {}
        cycles =  []

        for i in range(0, self.genSize):
            tmpA[indA.genes[i]] = False
            tmpB[indB.genes[i]] = False
            child.insert(i, indA.genes[i])
            del child[-1]
        cycle = []
        cyclecomplete  = False
        position_control = 0
        while False in tmpA.values():
            cyclecomplete = False
            if position_control == -1:
                position_control = 0
                for  key, value in tmpA.items():
                    if value == False:
                        break
                    else:
                        position_control += 1
            while not cyclecomplete :
                if not (tmpA[indA.genes[position_control]]):
                    cycle.append(indA.genes[position_control])
                   # cycle.append(indB.genes[position_control])
                    tmpA[indA.genes[position_control]] = True
                    tmpB[indB.genes[position_control]] = True
                    position_control = indA.genes.index(indB.genes[position_control])
                else:
                    cyclecomplete = True
                    position_control = -1
                    cycles.append(cycle.copy())
                    cycle = []

                    print(" ")
        print(" ")












        return child



    def reciprocalExchangeMutation(self, ind):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)

    def scrambleMutation(self, ind):
        """
        Your Scramble Mutation implementation
        """
        pass

    def crossover(self, indA, indB):
        """
        Executes a 1 order crossover and returns a new individual
        """
        child = []
        tmp = {}

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        for i in range(0, self.genSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                tmp[indA.genes[i]] = False
            else:
                tmp[indA.genes[i]] = True
        aux = []
        for i in range(0, self.genSize):
            if not tmp[indB.genes[i]]:
                child.append(indB.genes[i])
            else:
                aux.append(indB.genes[i])
        child += aux
        return child

    def mutation(self, ind):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        for ind_i in self.population:
            self.matingPool.append( ind_i.copy() )

    def newGeneration(self,config_to_run):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """




            if config_to_run['selection'] == 'random':
                indvselection = self.randomSelection()
            elif config_to_run['selection'] == 'random':
                self.rouletteWheel()
            elif config_to_run['selection'] == 'bestandsecond':
                print("BEST AND SECOND")
                #indvselection = self.rouletteWheel()
                # do best and second


            if config_to_run['crossover'] == 'cycle':
                childreturn = self.cycleCrossover(indvselection[0], indvselection[1])
                newGeneration = self.generateIndividualFromKeys(childreturn, indvselection[0])
            elif config_to_run['crossover'] == 'uniform':
                childreturn = self.uniformCrossover(indvselection[0], indvselection[1])
                newGeneration = self.generateIndividualFromKeys(childreturn, indvselection[0])

            if config_to_run['mutation'] == 'reciprocal':
                self.reciprocalExchangeMutation(newGeneration)
            elif config_to_run['mutation'] == 'scramble':
                self.mutation(newGeneration)



            #indvselection = self.randomSelection()
           # childreturn = self.crossover(indvselection[0], indvselection[1])
            #newGeneration = self.generateIndividualFromKeys(childreturn,indvselection[0])
           # self.mutation(newGeneration)


    def GAStep(self,config_to_run):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        self.newGeneration(config_to_run)

    def search(self,config_to_run):
        """
        General search template.
        Iterates for a given number of steps
        """
        self.iteration = 0
        while self.iteration < self.maxIterations:
            self.GAStep(config_to_run)
            self.iteration += 1

        print ("Total iterations: ", self.iteration)
        print ("Best Solution: ", self.best.getFitness())

#instances = ["dataset/inst-0.tsp","dataset/inst-13.tsp","dataset/inst-16.tsp"]
instances = ["dataset/inst-0.tsp"]

problem_file = sys.argv[1]
#configurations = {1: {'crossover': 'uniform', 'mutation': 'reciprocal', 'selection': 'random'},
#                  2: {'crossover': 'cycle', 'mutation': 'scramble', 'selection': 'random'},
#                  3: {'crossover': 'uniform', 'mutation': 'reciprocal', 'selection': 'roulette'},
#                  4: {'crossover': 'cycle', 'mutation': 'reciprocal', 'selection': 'roulette'},
#                  5: {'crossover': 'cycle', 'mutation': 'scramble', 'selection': 'roulette'},
#                  6: {'crossover': 'uniform', 'mutation': 'scramble', 'selection': 'bestandsecond'}}
configurations = {1: {'crossover': 'cycle', 'mutation': 'reciprocal', 'selection': 'random'}}

resultsofconfigs = {1: {'time': 0.0, 'iteration': 0.0, 'fitness': 0.0},
                    2: {'time': 0.0, 'iteration': 0.0, 'fitness': 0.0},
                    3: {'time': 0.0, 'iteration': 0.0, 'fitness': 0.0},
                    4: {'time': 0.0, 'iteration': 0.0, 'fitness': 0.0},
                    5: {'time': 0.0, 'iteration': 0.0, 'fitness': 0.0},
                    6: {'time': 0.0, 'iteration': 0.0, 'fitness': 0.0}}

resultsofinstances = {}


def run(currentinstance):
    print("======================== Running Instance {0} ===============".format(currentinstance))
    print(" ")
    for key, value in configurations.items():
        print("-----------------Running Configuration where Crossover is: {0[crossover]}, Mutation is: {0[mutation]}  and  Selection is: {0[selection]} -------------".format(value))
        start = timer()
        ga = BasicTSP(currentinstance, 300, 0.1, 300)
        ga.search(value)
        print("Total iterations: ", ga.iteration)
        print("Best Solution: ", ga.best.getFitness())
        end = timer()
        print(end - start)
        resultsofconfigs[key]['time'] = end - start
        resultsofconfigs[key]['iteration'] = ga.iteration
        resultsofconfigs[key]['fitness'] = ga.best.getFitness()
        resultsofinstances[currentinstance] = resultsofinstances.get(currentinstance, {})
        resultsofinstances[currentinstance][key] = resultsofconfigs[key]
        print("-------------------------------End of configuration run -----------------------------")
        print(" ")
    print("===================================== End of Instance run ===============================")


def print_results():
    print("======================== Results for all configurations ====================================================================")
    print()
    for instancekey, instancevalue in resultsofinstances.items():
        print("----------------------------------------Results for Instance  {0}-------------------------------------------------------------------".format(instancekey))

        for resultkey, resultvalue in configurations.items():
            print("Result for Configuration {0} where Crossover is: {1[crossover]}, Mutation is: {1[mutation]}  and  Selection is: {1[selection]}".format(resultkey, resultvalue))
            print("Execution time  {0[time]} and the Iteration is {0[iteration]} and the Best Solution is {0[fitness]}".format(instancevalue[resultkey]))
            print(            "-------------------------------------------------------------------------------------------------------------------")
    print("======================== End of Results ================================================================================")


for instance_number in range(0, len(instances)):
    print(instances[instance_number])
    run(instances[instance_number])
print_results()








