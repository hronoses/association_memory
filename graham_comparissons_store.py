import numpy as np
import pickle
import senses
import sys
import matplotlib.pyplot as plt
from itertools import permutations, repeat
import itertools


from graham_clusteron_smart_multi import Field
# from graham_corr_mat import Field
# from graham_original_noise import Field
# from graham_explicit_patterns import Field

N = [200]
a = [0.05]
N_syn = [10]
N_clst = [2]

conn = [0.6]
# noise = [0.2, 0.3, 0.4, 0.5]
R = 30000
s = [N, a, N_syn, N_clst, conn]
parameters = list(itertools.product(*s))
# print parameters
# print len(parameters)
data = []
for p in range(len(parameters)):
    print p, parameters[p]
    N, activation, num_synapses, cluster_size, connections = parameters[p]
    # connections = 1
    noise = 0

    color = Field(size=N)
    label = Field(size=N, activation=activation, num_synapses=num_synapses, connections=connections, cluster_size=cluster_size)
    R = 40000 # number of patterns
    c = senses.ColorSense(total_number=N, sparsity=activation, color_size=R)
    num = senses.NumberSense(total_number=N, sparsity=activation, number_size=R)


    label.init_weights(color)

    patterns = []
    accuracies = []
    fullness = []

    import time
    t = time.time()
    # print 'started iteration',
    # print i

    step = 300

    start_step = 5
    if start_step > 1:
        for i in xrange(0, (start_step-1)*step):
            color.cells = c.sense(i)
            label.cells = num.sense(i)
            label.store()
            if not i%step:
                print i,

    for n in xrange(start_step, R//step + 1):
        print n*step,
        #storing
        for i in xrange((n-1)*step, n*step):
            color.cells = c.sense(i)
            label.cells = num.sense(i)
            label.store()


        # print 'stored ' + str(n*step),
        # print 'fullness=' + str(label.fullness)
        accuracy = 0
        stored_patterns = 0
        patterns_to_compare = num.get_part(n*step)

        # retrieve
        for i in xrange(n*step):
            color.cells = c.sense(i, noise=0)
            label.cells = 0
            label.retrieve()
            results = np.dot(patterns_to_compare, label.cells)
            if i == np.argmax(results):
                accuracy += 1
            # if not i%100:
            #     print i,
        # print str(n*step) + ' retrieved'
        # print float(accuracy)/(n*step)
        accuracies.append(float(accuracy)/(n*step))
        patterns.append(n*step)

        fullness.append(label.fullness)
        if float(accuracy)/(n*step) < 0.1:
            print 'accuracy too low'
            break
        stored_patterns = np.array(patterns) * np.array(accuracies)
        # print stored_patterns[-1]/N
        if stored_patterns.size > 1:
            if np.abs(stored_patterns[-1] - stored_patterns[-2]) < 10 or stored_patterns[-1] < stored_patterns[-2]:
                print stored_patterns[-1]/N
                break
    try:
        data.append([N, activation, num_synapses, cluster_size, connections, noise, stored_patterns[-1]/N, float(np.count_nonzero(label.distances))/label.distances.size, patterns, accuracies, fullness])
    except:
        data.append([N, activation, num_synapses, cluster_size, connections, noise, 0, float(np.count_nonzero(label.distances))/label.distances.size, patterns, accuracies, fullness])




datafile = 'data/graham_comparisons_connection_clustN200.p'
with open(datafile, 'w') as file:
    pickle.dump(data, file)

import sys
sys.exit()

#
# from graham_original_noise import Field
# N = [200]
# a = [0.05]
#
# conn = [0.1]
# # noise = [0.1, 0.2, 0.3, 0.4, 0.5]
# R = 30000
# s = [N, a, conn]
# parameters = list(itertools.product(*s))
# # print parameters
# # print len(parameters)
# data = []
# for p in range(len(parameters)):
#     print p, parameters[p]
#     N, activation, connections = parameters[p]
#     # connections = 1
#     noise = 0
#
#     color = Field(size=N)
#     label = Field(size=N, activation=activation, connections=connections)
#     R = 5000 # number of patterns
#     c = senses.ColorSense(total_number=N, sparsity=activation, color_size=R)
#     num = senses.NumberSense(total_number=N, sparsity=activation, number_size=R)
#
#
#     label.init_weights(color)
#
#     patterns = []
#     accuracies = []
#     fullness = []
#
#     import time
#     t = time.time()
#     # print 'started iteration',
#     # print i
#
#     step = 50
#
#     start_step = 1
#     if start_step > 1:
#         for i in xrange(0, (start_step-1)*step):
#             color.cells = c.sense(i)
#             label.cells = num.sense(i)
#             label.store()
#             if not i%step:
#                 print i,
#
#     for n in xrange(start_step, R//step + 1):
#         print n*step,
#         #storing
#         for i in xrange((n-1)*step, n*step):
#             color.cells = c.sense(i)
#             label.cells = num.sense(i)
#             label.store()
#
#
#         # print 'stored ' + str(n*step),
#         # print 'fullness=' + str(label.fullness)
#         accuracy = 0
#         stored_patterns = 0
#         patterns_to_compare = num.get_part(n*step)
#
#         # retrieve
#         for i in xrange(n*step):
#             color.cells = c.sense(i, noise=0)
#             label.cells = 0
#             label.retrieve()
#             results = np.dot(patterns_to_compare, label.cells)
#             if i == np.argmax(results):
#                 accuracy += 1
#             # if not i%100:
#             #     print i,
#         # print str(n*step) + ' retrieved'
#         # print float(accuracy)/(n*step)
#         accuracies.append(float(accuracy)/(n*step))
#         patterns.append(n*step)
#
#         fullness.append(label.fullness)
#         if float(accuracy)/(n*step) < 0.1:
#             print 'accuracy too low'
#             break
#         stored_patterns = np.array(patterns) * np.array(accuracies)
#         # print stored_patterns[-1]/N
#         if stored_patterns.size > 1:
#             if np.abs(stored_patterns[-1] - stored_patterns[-2]) < 10 or stored_patterns[-1] < stored_patterns[-2]:
#                 print stored_patterns[-1]/N
#                 break
#     try:
#         data.append([N, activation, connections, 0, stored_patterns[-1]/N, patterns, accuracies])
#     except:
#         data.append([N, activation, connections, 0, 0, patterns, accuracies])
#
#
#
#
# datafile = 'data/graham_comparisons_conn_grahamN200.p'
# with open(datafile, 'w') as file:
#     pickle.dump(data, file)






# data = {}
# from graham_corr_mat import Field as Fl
#
#
# N = [100, 200, 300]
# a = [0.05, 0.1, 0.15, 0.2, 0.3]
#
#
# conn = [1, 0.8, 0.6, 0.4, 0.2]
# noise = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
# R = 25000
# s = [N, a]
# parameters = list(itertools.product(*s))
# # print parameters
# # print len(parameters)
# data = []
# for p in range(len(parameters)):
#     print p, parameters[p]
#     N, activation = parameters[p]
#     connections = 1
#     noise = 0
#
#     color = Fl(size=N)
#     label = Fl(size=N, activation=activation)
#     R = 25000 # number of patterns
#     c = senses.ColorSense(total_number=N, sparsity=activation, color_size=R)
#     num = senses.NumberSense(total_number=N, sparsity=activation, number_size=R)
#
#
#     label.init_weights(color)
#
#     patterns = []
#     accuracies = []
#     fullness = []
#
#     import time
#     t = time.time()
#     # print 'started iteration',
#     # print i
#
#     step = 400
#     for n in xrange(1, R//step + 1):
#         print n*step,
#         #storing
#         for i in xrange((n-1)*step, n*step):
#             color.cells = c.sense(i)
#             label.cells = num.sense(i)
#             label.store()
#
#
#         # print 'stored ' + str(n*step),
#         # print 'fullness=' + str(label.fullness)
#         accuracy = 0
#         stored_patterns = 0
#         patterns_to_compare = num.get_part(n*step)
#
#         # retrieve
#         for i in xrange(n*step):
#             color.cells = c.sense(i, noise=noise)
#             label.cells = 0
#             label.retrieve()
#             results = np.dot(patterns_to_compare, label.cells)
#             if i == np.argmax(results):
#                 accuracy += 1
#         accuracies.append(float(accuracy)/(n*step))
#         patterns.append(n*step)
#
#         fullness.append(label.fullness)
#         if float(accuracy)/(n*step) < 0.1:
#             break
#         stored_patterns = np.array(patterns) * np.array(accuracies)
#         if stored_patterns.size > 1:
#             if np.abs(stored_patterns[-1] - stored_patterns[-2]) < 10 or stored_patterns[-1] < stored_patterns[-2]:
#                 print stored_patterns[-1]/N
#                 break
#     try:
#         data.append([N, activation, connections, noise, stored_patterns[-1]/N, patterns, accuracies])
#     except:
#         data.append([N, activation, connections, noise, 0, patterns, accuracies])
#
#
# datafile = 'data/graham_comparisons_corrmat.p'
# with open(datafile, 'w') as file:
#     pickle.dump(data, file)
#
#
#
#
