

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

    def bestandsecondbest(self):
        tmp = self.matingPool.copy()

        tmp.sort(key=lambda x: x.fitness)

        return [tmp[0],tmp[1]]

    def rouletteWheel(self):

        tmp = self.matingPool.copy()
        totalifitness = 0

        #i will be using the 1/ifitness caculation
        for i in range(0, self.popSize):
            currentIndividual = tmp[i]
            ifitness = 1 / currentIndividual.fitness
            totalifitness += ifitness
            #going to use probability on the individual for  ifitness as temp storage
            #as i will need to calculate the probability using a sum of all ifitness
            currentIndividual.probability = ifitness

        for i in range(0, self.popSize):
            currentIndividual = tmp[i]
            currentIndividual.probability = currentIndividual.probability/totalifitness


        tmp.sort(key=lambda x: x.probability)

        range_range = random.random()
        partialsum = 0
        for i in range(0, self.popSize):
            ind = tmp[i]
            partialsum += ind.probability
            if partialsum >= range_range:
                break;
        indA = ind

        range_range = random.random()
        partialsum = 0
        for i in range(0, self.popSize):
            ind = tmp[i]
            partialsum += ind.probability
            if partialsum >= range_range:
                break;
        indB = ind

        return [indA, indB]

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
        #last add is jus tto save me  extra loops on the B parent
        #other wise i would be stating from position 0 and looping through
        #instead i pick up where was left off the last time
        lastAdded = 0
        for i in range(0, self.genSize):
            if not (tmpA[indA.genes[i]]):

                for b_counter in range(lastAdded, self.genSize):

                    if not indB.genes[b_counter] in child:
                        del child[i]
                        child.insert(i, indB.genes[b_counter])
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
        cycle = []
        cyclecomplete  = False
        position_control = 0
        #two while loops, inner look clear a cycle, then outter loops controls a new cycle loop,
        #End result is that we should have a list of lists containg cycles
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

        #How to cross over , to do this we will loop on out cycles and
        #alternating between A to A B to B and A to B and B to A copies.
        a_to_a_crossover = True
        for cycle_to_process in cycles[:]:
            if a_to_a_crossover:
                for key in cycle_to_process[:]:
                    insert_position = indA.genes.index(key)
                    del child[insert_position]
                    child.insert(insert_position, key)
                    a_to_a_crossover = False
            else:
                for key in cycle_to_process[:]:
                    insert_position = indB.genes.index(key)
                    del child[insert_position]
                    child.insert(insert_position, key)
                    a_to_a_crossover = True
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
        return ind

    def scrambleMutation(self, ind):
        """
        Your Scramble Mutation implementation
        """

        tmp = ind.genes.copy()

        if(len(tmp) <3):
            #List too small to do scramble mutations on
            ind.computeFitness()
            self.updateBest(ind)
            return
        startpoint = 0
        endpoint = 0
        bad_start_point = True
        if (len(tmp) == 3):
            # its small list , just gerrymander the values start is pos 0 and end is start +1
            endpoint = startpoint +1
            bad_start_point = False

        #Check the start poinbt id good, we dont want to too close to the end of the list
        #if it is try again and get a better start point
        while bad_start_point:
            startpoint = tmp.index(random.choice(tmp))
            #make sure the end poit is at least 3 position away from end of list
            if ( (len(tmp)-1) - startpoint) >= 2:
                endpoint = random.randrange(startpoint + 1, len(tmp) - 1)
                if endpoint != startpoint :
                    #if the start point and end point are the same try again
                    bad_start_point = False

        # we should have good start and end points now. Scrame the data between the points
        new_list_to_scramble = tmp[startpoint:endpoint + 1]
        scramble_values = random.sample(new_list_to_scramble, len(new_list_to_scramble))

        # Now loop though out list and recplace the scrabmled values between the start and end point
        for i in range(0, len(tmp)):
            if i == startpoint:
                for x in range(0, len(scramble_values)):
                    tmp.insert(i, scramble_values[x])
                    del tmp[i+1]
                    i +=1
                #break from main loop as no point in going through the the rest of the list as scrambled items have been replaced now
                break
        #Scramble has now been completed
        ind.genes = tmp
        ind.computeFitness()
        self.updateBest(ind)
        return ind


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
        return ind

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        for ind_i in self.population:
            self.matingPool.append( ind_i.copy())

    def newGeneration(self,config_to_run):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        newIndividual = []
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """




            if config_to_run['selection'] == 'random':
                indvselection = self.randomSelection()
            elif config_to_run['selection'] == 'roulette':
                indvselection = self.rouletteWheel()
            elif config_to_run['selection'] == 'bestandsecond':
                indvselection = self.bestandsecondbest()



            if config_to_run['crossover'] == 'cycle':
                childreturn = self.cycleCrossover(indvselection[0], indvselection[1])
                newGeneration = self.generateIndividualFromKeys(childreturn, indvselection[0])
            elif config_to_run['crossover'] == 'uniform':
                childreturn = self.uniformCrossover(indvselection[0], indvselection[1])
                newGeneration = self.generateIndividualFromKeys(childreturn, indvselection[0])

            if config_to_run['mutation'] == 'reciprocal':
                newIndividual = self.reciprocalExchangeMutation(newGeneration)
            elif config_to_run['mutation'] == 'scramble':
                newIndividual = self.scrambleMutation(newGeneration)


            #add our new individual to the population only if it is better than its parents
            if (newIndividual is not None ):
                if (newIndividual.fitness > indvselection[0].fitness):
                    matingpool_index = self.matingPool.index(indvselection[0])
                    self.matingPool.insert(matingpool_index, newIndividual)
                    del self.matingPool[matingpool_index + 1]
                elif (newIndividual.fitness > indvselection[1].fitness):
                    matingpool_index = self.matingPool.index(indvselection[0])
                    self.matingPool.insert(matingpool_index, newIndividual)
                    del self.matingPool[matingpool_index + 1]






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

instances = ["dataset/inst-0.tsp","dataset/inst-13.tsp","dataset/inst-16.tsp"]
#instances = ["dataset/inst-0.tsp"]

problem_file = sys.argv[1]
configurations = {1: {'crossover': 'uniform', 'mutation': 'reciprocal', 'selection': 'random'},
                  2: {'crossover': 'cycle', 'mutation': 'scramble', 'selection': 'random'},
                  3: {'crossover': 'uniform', 'mutation': 'reciprocal', 'selection': 'roulette'},
                  4: {'crossover': 'cycle', 'mutation': 'reciprocal', 'selection': 'roulette'},
                  5: {'crossover': 'cycle', 'mutation': 'scramble', 'selection': 'roulette'},
                  6: {'crossover': 'uniform', 'mutation': 'scramble', 'selection': 'bestandsecond'}}
#onfigurations = {1: {'crossover': 'cycle', 'mutation': 'scramble', 'selection': 'bestandsecond'}}

resultsofconfigs = {1: {'time': 0.0, 'iteration': 0.0, 'fitness': 0.0},
                    2: {'time': 0.0, 'iteration': 0.0, 'fitness': 0.0},
                    3: {'time': 0.0, 'iteration': 0.0, 'fitness': 0.0},
                    4: {'time': 0.0, 'iteration': 0.0, 'fitness': 0.0},
                    5: {'time': 0.0, 'iteration': 0.0, 'fitness': 0.0},
                    6: {'time': 0.0, 'iteration': 0.0, 'fitness': 0.0}}

resultsofinstances = {}
resultsofrepetations = {}


def run(currentinstance,numberofrepeatingrun,popSize,mutationRate,numberofiterations ):
    print("======================== Running Instance {0} ===============".format(currentinstance))
    print(" ")
    #loop on configurations
    for key, value in configurations.items():
        #loop on reperations of configuration
        for i in range(1, numberofrepeatingrun+1):
            print("--------------------------------------------Run:, {0} , For Config,   {1} , -----------------------------".format(i,key))
            print("-----------------Crossover is, {0[crossover]}, Mutation is, {0[mutation]}  , Selection is, {0[selection]} ,-------------".format(value))
            start = timer()
            ga = BasicTSP(currentinstance, popSize, mutationRate, numberofiterations)
            ga.search(value)

            end = timer()
            print(end - start)
            resultsofconfigs[key]['time'] = end - start
            resultsofconfigs[key]['iteration'] = ga.iteration
            resultsofconfigs[key]['fitness'] = ga.best.getFitness()
            resultsofinstances[currentinstance] = resultsofinstances.get(currentinstance, {})
            resultsofinstances[currentinstance][key] = resultsofinstances[currentinstance].get(key , {})
            #resultsofinstances[currentinstance][key] = resultsofconfigs[key]
            resultsofinstances[currentinstance][key][i] = resultsofinstances[currentinstance][key].get(i , {})

            resultsofinstances[currentinstance][key][i] = resultsofconfigs[key]
            print("-------------------------------End of configuration run -----------------------------")
            print(" ")
    print("===================================== End of Instance run ===============================")


def print_and_save_results():
    fName = 'output.csv'

    file = open(fName, 'w')
    print("======================== Results for all configurations ====================================================================")
    file.write("======================== Results for all configurations and instance and runs ====================================================================\n")
    print()
    for instancekey, instancevalue in resultsofinstances.items():
        print("----------------------------------------Results for Instance,  {0} ,-------------------------------------------------------------------".format(instancekey))

        for configkey, configvalue in configurations.items():
            for runresultskey, runresultsvalue in instancevalue[configkey].items():
                print("Run number:, {2} Configuration, {0} , Crossover , {1[crossover]} , Mutation , {1[mutation]} , Selection , {1[selection]}".format(configkey, configvalue, runresultskey))

                print("Execution time,  {0[time]}  ,Iteration , {0[iteration]} , Best Solution , {0[fitness]}".format(runresultsvalue))
                file.write("Instance: , {3} , Run number:, {2} ,  Configuration, {0} ,  Crossover , {1[crossover]} , Mutation , {1[mutation]} , Selection , {1[selection]} , Execution time,  {4[time]}  ,Iteration , {4[iteration]} , Best Solution , {4[fitness]} \n".format(configkey, configvalue, runresultskey, instancekey, runresultsvalue))

                print("-------------------------------------------------------------------------------------------------------------------")
    print("======================== End of Results ================================================================================")

numberofrepeatingrun = 3
popSize = 100
mutationRate =0.1
numberofiterations = 300
for instance_number in range(0, len(instances)):
    print(instances[instance_number])
    run(instances[instance_number], numberofrepeatingrun, popSize, mutationRate, numberofiterations )
print_and_save_results()








