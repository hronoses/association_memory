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
clust2_10 = {}
clust3_1 = {}
clust3_10 = {}
clust5_20 = {}
clust2_10[0.1] = 0
clust2_10[0.2] = 5.655
clust2_10[0.4] = 14.915
clust2_10[0.6] = 17.15
clust2_10[0.8] = 16.975
clust2_10[1] = 16.83

clust3_10[0.1] = 0
clust3_10[0.2] = 0.775
clust3_10[0.4] = 23.725
clust3_10[0.6] = 38.12
clust3_10[0.8] = 51.05
clust3_10[1] = 69.915

graham = {}
graham[0.1] = 0.6
graham[0.2] = 1.28
graham[0.4] = 2.62
graham[0.6] = 3.13
graham[0.8] = 3.775
graham[1] = 4.035

corrmat = {}
corrmat[1] = 124.55
corrmat[0.8] = 120.82
corrmat[0.6] = 107.96
corrmat[0.4] = 84.24
corrmat[0.2] = 46.54
corrmat[0.1] = 20.65

fig, ax = plt.subplots()
import matplotlib
font_size = 14
matplotlib.rcParams.update({'font.size': font_size})

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(color='k', linestyle=':', linewidth=0.1)

import collections
clust2_10 = collections.OrderedDict(sorted(clust2_10.items()))
clust3_10 = collections.OrderedDict(sorted(clust3_10.items()))
graham = collections.OrderedDict(sorted(graham.items()))
corrmat = collections.OrderedDict(sorted(corrmat.items()))


marker_size = 6
plt.plot(clust2_10.keys(), clust2_10.values(),  marker='x', color='k', markersize=marker_size, label='clusteron 2_5')
plt.plot(clust3_10.keys(), clust3_10.values(),  marker='+', color='r', markersize=marker_size, label='clusteron 3_5')
plt.plot(graham.keys(), graham.values(),  marker='d', color='b', markersize=marker_size, label='Wilshaw')
plt.plot(corrmat.keys(), corrmat.values(),  marker='o', color='g', markersize=marker_size, label='triple correlation')

plt.xlabel("Connectivity", size=font_size)
plt.ylabel("Capacity per neuron", size=font_size)
plt.xticks([0.1, 0.2, 0.4, 0.6, 0.8, 1])
plt.legend(loc='upper left', fontsize=13)

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
