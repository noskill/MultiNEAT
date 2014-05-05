from __future__ import division
import multiprocessing as mpc
import re
import ConfigParser
import time
from _MultiNEAT import *  


try:
    from progressbar import ProgressBar, Counter, ETA, AnimatedMarker
    prbar_installed = True
except:
    print ('Tip: install the progressbar Python package through pip or '
           'easy_install')
    print ('     to get good looking evolution progress bar with ETA')
    prbar_installed = False


try:
    import cv2
    import numpy as np
    cvnumpy_installed = True
except:
    print ('Tip: install the OpenCV computer vision library (2.0+) with '
           'Python bindings')
    print ('     to get convenient neural network visualization to NumPy '
           'arrays')
    cvnumpy_installed = False



# Get all genomes from the population
def GetGenomeList(pop):
    genome_list = []
    for s in pop.Species:
        for i in s.Individuals:
            genome_list.append(i)
    return genome_list


RetrieveGenomeList = GetGenomeList
FetchGenomeList = GetGenomeList


# Evaluates all genomes in sequential manner (using only 1 process) and
# returns a list of corresponding fitness values and the time it took
# evaluator is a callable that is supposed to take Genome as argument and
# return a double
def EvaluateGenomeList_Serial(genome_list, evaluator, display=True):
    fitnesses = []
    count = 0

    if prbar_installed and display:
        widg = ['Individuals: ', Counter(), ' of ' + str(len(genome_list)),
                ' ', ETA(), ' ', AnimatedMarker()]
        progress = ProgressBar(maxval=len(genome_list), widgets=widg).start()

    for g in genome_list:
        f = evaluator(g)
        fitnesses.append(f)

        if display:
            if prbar_installed:
                progress.update(count+1)
            else:
                print 'Individuals: (%s/%s)' % (count, len(genome_list))
        
                count += 1

    if prbar_installed and display:
        progress.finish()

    return fitnesses
    
# Evaluates all genomes in parallel manner (many processes) and returns a
# list of corresponding fitness values and the time it took  evaluator is
# a callable that is supposed to take Genome as argument and return a double
def EvaluateGenomeList_Parallel(genome_list, evaluator, cores=4, display=True):
    fitnesses = []
    pool = mpc.Pool(processes=cores)
    curtime = time.time()

    if prbar_installed and display:
        widg = ['Individuals: ', Counter(),
                ' of ' + str(len(genome_list)), ' ', ETA(), ' ',
                AnimatedMarker()]
        progress = ProgressBar(maxval=len(genome_list), widgets=widg).start()

    for i, fitness in enumerate(pool.imap(evaluator, genome_list)):
        if prbar_installed and display:
            progress.update(i)
        else:
            if display:
                print 'Individuals: (%s/%s)' % (i, len(genome_list))

        if cvnumpy_installed:
            cv2.waitKey(1)
        fitnesses.append(fitness)

    if prbar_installed and display:
        progress.finish()

    elapsed = time.time() - curtime

    if display:
        print 'seconds elapsed: %s' % elapsed

    pool.close()
    pool.join()

    return fitnesses


# Just set the fitness values to the genomes
def ZipFitness(genome_list, fitness_list):
    for g, f in zip(genome_list, fitness_list):
        g.SetFitness(f)


def Scale(a, a_min, a_max, a_tr_min, a_tr_max):
    t_a_r = a_max - a_min
    if t_a_r == 0:
        return a_max

    t_r = a_tr_max - a_tr_min
    rel_a = (a - a_min) / t_a_r
    return a_tr_min + t_r * rel_a


def Clamp(a, min, max):
    if a < min:
        return min
    elif a > max:
        return max
    else:
        return a


def AlmostEqual(a, b, margin):
    if abs(a-b) > margin:
        return False
    else:
        return True


# Neural Network display code
# rect is a tuple in the form (x, y, size_x, size_y)
if not cvnumpy_installed:
    def DrawPhenotype(image, rect, nn, neuron_radius=10,
                      max_line_thickness=3, substrate=False):
        print "OpenCV/NumPy don't appear to be installed"
        raise NotImplementedError
else:
    MAX_DEPTH = 64
    def draw_arrow(image, p, q, color, thickness=1, arrow_magnitude=9, line_type=8, shift=0):
        # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html

        # draw arrow tail
        cv2.line(image, p, q, color, thickness, line_type, shift)
        # calc angle of the arrow
        angle = np.arctan2(p[1]-q[1], p[0]-q[0])
        # starting point of first line of arrow head
        p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
        int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
        # draw first half of arrow head
        cv2.line(image, p, q, color, thickness, line_type, shift)
        # starting point of second line of arrow head
        p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
        int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
        # draw second half of arrow head
        cv2.line(image, p, q, color, thickness, line_type, shift)

    def DrawPhenotype(image, rect, nn, neuron_radius=10,
                      max_line_thickness=3, substrate=False):
        for i, n in enumerate(nn.neurons):
            nn.neurons[i].x = 0
            nn.neurons[i].y = 0

        rect_x = rect[0]
        rect_y = rect[1]
        rect_x_size = rect[2]
        rect_y_size = rect[3]

        if not substrate:
            depth = 0
            # for every depth, count how many nodes are on this depth
            all_depths = np.linspace(0.0, 1.0, MAX_DEPTH)

            for depth in all_depths:
                neuron_count = 0
                for neuron in nn.neurons:
                    if AlmostEqual(neuron.split_y, depth, 1.0 / (MAX_DEPTH+1)):
                        neuron_count += 1
                if neuron_count == 0:
                    continue

                # calculate x positions of neurons
                xxpos = rect_x_size / (1 + neuron_count)
                j = 0
                for neuron in nn.neurons:
                    if AlmostEqual(neuron.split_y, depth, 1.0 / (MAX_DEPTH+1)):
                        neuron.x = rect_x + xxpos + j * (rect_x_size / (2 + neuron_count))
                        j = j + 1

            # calculate y positions of nodes
            for neuron in nn.neurons:
                base_y = rect_y + neuron.split_y
                size_y = rect_y_size - neuron_radius

                if neuron.split_y == 0.0:
                    neuron.y = base_y * size_y + neuron_radius
                else:
                    neuron.y = base_y * size_y

        else:
            # HyperNEAT substrate
            # only the first 2 dimensions are used for drawing
            # if a layer is 1D,  y values will be supplied to make 3 rows

            # determine min/max coords in NN
            xs = [(neuron.substrate_coords[0]) for neuron in nn.neurons]
            ys = [(neuron.substrate_coords[1]) for neuron in nn.neurons]
            min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)

            #dims = [len(neuron.substrate_coords) for neuron in nn.neurons]

            for neuron in nn.neurons:
                # TODO(jkoelker) Make the rect_x_size / 15 a variable
                neuron.x = Scale(neuron.substrate_coords[0], min_x, max_x,
                                 rect_x_size / 15,
                                 rect_x_size - rect_x_size / 15)
                neuron.y = Scale(neuron.substrate_coords[1], min_y, max_y,
                                 rect_x_size / 15,
                                 rect_y_size - rect_x_size / 15)

        # the positions of neurons is computed, now we draw
        # connections first
        if nn.connections:
            max_weight = max([abs(x.weight) for x in nn.connections])
        else:
            max_weight = 1.0

        for conn in nn.connections:
            thickness = conn.weight
            thickness = Scale(thickness, 0, max_weight, 1, max_line_thickness)
            thickness = Clamp(thickness, 1, max_line_thickness)

            w = Scale(abs(conn.weight), 0.0, max_weight, 0.0, 1.0)
            w = Clamp(w, 0.75, 1.0)

            if conn.recur_flag:
                if conn.weight < 0:
                    # green weight
                    color = (0, int(255.0 * w), 0)

                else:
                    # white weight
                    color = (int(255.0 * w), int(255.0 * w), int(255.0 * w))

            else:
                if conn.weight < 0:
                    # blue weight
                    color = (int(255.0 * w), 0, 0)

                else:
                    # red weight
                    color = (0, 0, int(255.0 * w))

            # if the link is looping back on the same neuron, draw it with
            # ellipse
            if conn.source_neuron_idx == conn.target_neuron_idx:
                pass  # todo: later

            else:
                # Draw a line
                pt1 = (int(nn.neurons[conn.source_neuron_idx].x),
                       int(nn.neurons[conn.source_neuron_idx].y))
                pt2 = (int(nn.neurons[conn.target_neuron_idx].x),
                       int(nn.neurons[conn.target_neuron_idx].y))
                #cv2.line(image, pt1, pt2, color, int(thickness))
                draw_arrow(image, pt1, pt2, color, int(thickness))

         # draw all neurons
        #for neuron in nn.neurons:
        #    pt = (int(neuron.x), int(neuron.y))
         #   cv2.circle(image, pt, neuron_radius, (255, 255, 255), -1)

class ParametersLoader(Parameters):
    _param_list = """
    InitialDepth
    MaximumDepth
    DivisionThreshold
    VarianceThreshold
    BandingThreshold
    ESIterations

    PopulationSize

    DynamicCompatibility
    CompatTreshold
    YoungAgeTreshold
    SpeciesMaxStagnation
    OldAgeTreshold
    MinSpecies
    MaxSpecies
    RouletteWheelSelection

    MutateRemLinkProb
    RecurrentProb
    OverallMutationRate
    MutateAddLinkProb
    MutateAddNeuronProb
    MutateWeightsProb
    MaxWeight
    WeightMutationMaxPower
    WeightReplacementMaxPower

    MutateActivationAProb
    ActivationAMutationMaxPower
    MinActivationA
    MaxActivationA

    MutateNeuronActivationTypeProb

    ActivationFunction_SignedSigmoid_Prob
    ActivationFunction_UnsignedSigmoid_Prob
    ActivationFunction_Tanh_Prob
    ActivationFunction_TanhCubic_Prob
    ActivationFunction_SignedStep_Prob
    ActivationFunction_UnsignedStep_Prob
    ActivationFunction_SignedGauss_Prob
    ActivationFunction_UnsignedGauss_Prob
    ActivationFunction_Abs_Prob
    ActivationFunction_SignedSine_Prob 
    ActivationFunction_UnsignedSine_Prob 
    ActivationFunction_Linear_Prob
    OldAgePenalty
    """
    _param_set = {}
    def __init__(self):
        Parameters.__init__(self)
        for item in self._param_list.split():
            self._param_set[item.strip()] = False
            
    def __setattr__(self, name, value):
        if name in self._param_set:
            self._param_set[name] = True
        assert(name in dir(self))
        Parameters.__setattr__(self, name, value)
        
    def assertInitialized(self):
        for item in self._param_set:
            assert(self._param_set[item])
    
def loadParameters(path):
    res = ParametersLoader()
    with open(path) as f:
      for line in f.readlines():
          if '=' in line:
             line_list = line.split('=')
             line_list = [x.strip() for x in line_list]
             
             if 'true' in line_list[1].lower():
                value = True
             elif 'false' in line_list[1].lower():
                value = False
             elif '.' in line_list[1]:
                value = float(line_list[1])
             else:
                 value = int(line_list[1])
             setattr(res, line_list[0], value)
    res.assertInitialized()
    return res

      