import numpy as np
import pickle
import senses
import sys
import matplotlib.pyplot as plt


from graham_clusteron_smart_multi import Field
# from graham_corr_mat import Field
# from graham_original_noise import Field
# from graham_explicit_patterns import Field

N = 100
activation = 0.05

num_synapses = 5
cluster_size = 3

connections = 1
noise = 0.0

color = Field(size=N)
try:
    label = Field(size=N, activation=activation, num_synapses=num_synapses, connections=connections, cluster_size=cluster_size)
except:
    label = Field(size=N, activation=activation, connectivity=connections)
R = 10000 # number of patterns
c = senses.ColorSense(total_number=N, sparsity=activation, color_size=R)
num = senses.NumberSense(total_number=N, sparsity=activation, number_size=R)


label.init_weights(color)

patterns = []
accuracies = []
fullness = []

import time
t = time.time()
print 'started'


step = 200
start_step = 7
if start_step > 1:
    for i in xrange(0, (start_step-1)*step):
        color.cells = c.sense(i)
        label.cells = num.sense(i)
        label.store()
        if not i%100:
            print i,

for n in xrange(start_step, R//step + 1):
    #storing
    for i in xrange((n-1)*step, n*step):
        color.cells = c.sense(i)
        label.cells = num.sense(i)
        label.store()
        if not i%100:
            print i,


    print 'stored ' + str(n*step)
    # print 'fullness=' + str(label.fullness)
    accuracy = 0
    patterns_to_compare = num.get_part(n*step)

    # retrieve
    for i in xrange(n*step):
        color.cells = c.sense(i, noise=noise)
        label.cells = 0
        label.retrieve()
        results = np.dot(patterns_to_compare, label.cells)
        if i == np.argmax(results):
            accuracy += 1
        if not i%100:
            print i,
    # print str(n*step) + ' retrieved'
    print float(accuracy)/(n*step)
    accuracies.append(float(accuracy)/(n*step))
    patterns.append(n*step)

    fullness.append(label.fullness)
    if float(accuracy)/(n*step) < 0.1:
        break
    stored_patterns = np.array(patterns) * np.array(accuracies)
    # print stored_patterns
    if stored_patterns.size > 1:
        if np.abs(stored_patterns[-1] - stored_patterns[-2]) < 5 or stored_patterns[-1] < stored_patterns[-2]:
            print stored_patterns
            break




print time.time() - t

data = {}

# data['clusteron'] = {(N, activation, num_synapses, cluster_size, connections, noise):(patterns, accuracies)}



# datafile = 'data/graham_comparisons.p'
# with open(datafile) as file:
#     try:
#         data = pickle.load(file)
#     except:
#         raise 'empty data'
#
# with open(datafile, 'w') as file:
#     # data['clusteron'] = []
#     # data['graham'] = []
#     # data['corr_mat'] = []
#     data['clusteron'].append((N, activation, R, num_synapses, cluster_size, connections, noise, patterns, accuracies, label.fullness))
#     pickle.dump(data, file)



print data

print list(patterns)
print accuracies
# print np.count_nonzero(label.corr)
print np.array(patterns) * np.array(accuracies)/N
# print list(np.array(patterns) * np.array(accuracies))

import matplotlib.pyplot as plt
plt.plot(patterns, accuracies)
plt.ylim([0, 1.5])
# plt.show()

plt.plot(patterns,  np.array(patterns) * np.array(accuracies))
# plt.show()

