#!/usr/bin/python
import os
import sys
sys.path.append("/home/peter")
sys.path.append("/home/peter/Desktop")
sys.path.append("/home/peter/Desktop/projects")
import time
import random as rnd
import commands as comm
import cv2
import numpy as np
import cPickle as pickle
import MultiNEAT as NEAT
import multiprocessing as mpc


params = NEAT.loadParameters('params.txt')


# the simple 2D substrate with 3 input points, 2 hidden and 1 output for XOR 
substrate = NEAT.EvolvableSubstrate(params, [(-1, -1), (-1, 0), (-1, 1)],
                           [(1, 0)])

# let's set the activation functions
substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.TANH
substrate.m_outputs_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID

# code 
cv2.namedWindow('CPPN', 0)
cv2.namedWindow('NN', 0)


def evaluate(genome):
    net = NEAT.NeuralNetwork()
    try:
        genome.BuildHyperNEATESPhenotype(net, substrate)

        error = 0
        #depth = net.CalculateDepth();
        depth = 2;

        # do stuff and return the fitness
        net.Flush()

        net.Input([1, 0, 1])
        [net.Activate() for _ in range(depth)]
        o = net.Output()
        error += abs(o[0] - 1)

        net.Flush()
        net.Input([0, 1, 1])
        [net.Activate() for _ in range(depth)]
        o = net.Output()
        error += abs(o[0] - 1)

        net.Flush()
        net.Input([1, 1, 1])
        [net.Activate() for _ in range(depth)]
        o = net.Output()
        error += abs(o[0] - 0)

        net.Flush()
        net.Input([0, 0, 1])
        [net.Activate() for _ in range(depth)]
        o = net.Output()
        error += abs(o[0] - 0)

        result = ajustCoeff(net) * (4 - error)**2
        #result = (4 - error)**2
        return result

    except Exception as ex:
        print 'Exception:', ex
        return 1.0



rng = NEAT.RNG()
rng.TimeSeed()

def ajustCoeff(nn):
    neuron_count = len(nn.neurons)

    result = 1.0
    if 10 < neuron_count <= 15:
      result = 0.90
    elif 15 < neuron_count < 20:
      result = 0.80
    elif neuron_count >= 20:
      result = 0.65
    return result

def getbest():
    g = NEAT.Genome(5, 
                    substrate.GetMinCPPNInputs(), 
                    0, 
                    substrate.GetMinCPPNOutputs(), 
                    False, 
                    NEAT.ActivationFunction.SIGNED_GAUSS, 
                    NEAT.ActivationFunction.SIGNED_GAUSS, 
                    0, 
                    params)
    pop = NEAT.Population(g, params, True, 1.0)

    for generation in range(1000):
        genome_list = NEAT.GetGenomeList(pop)
        #fitnesses = NEAT.EvaluateGenomeList_Parallel(genome_list, evaluate, cores=3, display=False)

        fitnesses = NEAT.EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
        [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]

        best = max([x.GetLeader().GetFitness() for x in pop.Species])
#        print 'Best fitness:', best

        # test
        net = NEAT.NeuralNetwork()
        pop.Species[0].GetLeader().BuildPhenotype(net)
        img = np.zeros((250, 250, 3), dtype=np.uint8)
        img += 10
        NEAT.DrawPhenotype(img, (0, 0, 250, 250), net )
        cv2.imshow("CPPN", img)
    
        net = NEAT.NeuralNetwork()
        pop.Species[0].GetLeader().BuildHyperNEATESPhenotype(net, substrate)
        img = np.zeros((250, 250, 3), dtype=np.uint8)
        img += 10
        NEAT.DrawPhenotype(img, (0, 0, 250, 250), net, substrate=True )
        cv2.imshow("NN", img)

        cv2.waitKey(1)

        pop.Epoch()
        print "Generation:", generation
        print "best:", best
        print "depth:", net.CalculateDepth()
        generations = generation
        if best > 15.5:
            break

    return generations

gens = []
for run in range(100):
    gen = getbest()
    print 'Run:', run, 'Generations to solve XOR:', gen
    gens += [gen]

avg_gens = sum(gens) / len(gens)

print 'All:', gens
print 'Average:', avg_gens


