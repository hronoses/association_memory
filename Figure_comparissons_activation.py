import pickle
import numpy as np
import matplotlib.pyplot as plt

datafile1 = 'data/graham_comparisons_activation_clustN100.p'
# datafile2 = 'data/graham_comparisons_activation_grahamN100.p'

# datafile3 = 'graham_comparisons_corrmat.p'


with open(datafile1) as file:
    data1 = pickle.load(file)
#
# with open(datafile2) as file:
#     data2 = pickle.load(file)

# with open(datafile3) as file:
#     data3 = pickle.load(file)



clust2_1 = {}
clust2_5 = {}
clust3_1 = {}
clust3_5 = {}
clust5_20 = {}
clust5_20[0.03] = 0
clust5_20[0.05] = 50.37
clust5_20[0.07] = 46.81
clust5_20[0.10] = 37.28
clust5_20[0.15] = 24.29
clust5_20[0.20] = 19.19
clust5_20[0.25] = 15.78
clust5_20[0.30] = 12.81

graham = {}
graham[0.03] = 4.06
graham[0.05] = 3.64
graham[0.07] = 2.38
graham[0.1] = 0.52
graham[0.15] = 0
graham[0.2] = 0
graham[0.25] = 0
graham[0.3] = 0
corrmat = {}
# corrmat[0.03] = 0
corrmat[0.05] = 124.55
corrmat[0.07] = 74.83
corrmat[0.1] = 33.63
corrmat[0.15] = 9.66
corrmat[0.2] = 3.56
corrmat[0.25] = 0
corrmat[0.3] = 0

for i in data1:
    N, activation, num_synapses, cluster_size, connections, noise, capacity, dist_full, patterns, accuracies, fullness = i
    # print i
    if cluster_size == 2 and num_synapses == 1:
        clust2_1[activation] = capacity
    if cluster_size == 2 and num_synapses == 5:
        clust2_5[activation] = capacity
    if cluster_size == 3 and num_synapses == 1:
        clust3_1[activation] = capacity
    if cluster_size == 3 and num_synapses == 5:
        clust3_5[activation] = capacity



fig, ax = plt.subplots()
import matplotlib
font_size = 14
matplotlib.rcParams.update({'font.size': font_size})

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(color='k', linestyle=':', linewidth=0.1)

import collections
clust2_5 = collections.OrderedDict(sorted(clust2_5.items()))
clust3_5 = collections.OrderedDict(sorted(clust3_5.items()))
graham = collections.OrderedDict(sorted(graham.items()))
corrmat = collections.OrderedDict(sorted(corrmat.items()))

marker_size = 6
plt.plot(clust2_5.keys(), clust2_5.values(),  marker='x', color='k', markersize=marker_size, label='clusteron 2_5')
plt.plot(clust3_5.keys(), clust3_5.values(),  marker='+', color='r', markersize=marker_size, label='clusteron 3_5')
plt.plot(graham.keys(), graham.values(), marker='d', color='b', markersize=marker_size, label='Wilshaw')
plt.plot(corrmat.keys(), corrmat.values(), marker='o', color='g', markersize=marker_size, label='triple correlation')

plt.xlabel("Sparsity", size=font_size)
plt.ylabel("Capacity per neuron", size=font_size)
plt.xticks([0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3])
plt.legend(loc='upper right', fontsize=15)

plt.show()

# width = 0.1
# # ind = np.array([0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3])
# N = len([0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3])
# ind = np.arange(N)
# fig, ax = plt.subplots()
# rects1 = ax.bar(ind,  clust2_1.values(), width, color='r')
# rects2 = ax.bar(ind + width, clust2_5.values(), width, color='y')
# rects2 = ax.bar(ind + 2*width, clust3_1.values(), width, color='k')
# rects2 = ax.bar(ind + 3*width, clust3_5.values(), width, color='b')
# # rects2 = ax.bar(ind + 4*width, clust5_20.values(), width, color='m')
# rects2 = ax.bar(ind + 5*width, corrmat.values(), width, color='g')
# # rects3 = ax.bar(ind + 6*width, graham.values(), width, color='r')
# print  clust2_1.values()
# print graham.values()
# ax.set_xticklabels([0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3])
# plt.show()
