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
clust2_1[0.1] = 6.255
clust2_1[0.2] = 4.15
clust2_1[0.3] = 2.485
clust2_1[0.4] = 1.54
clust2_1[0.5] = 0.84

clust2_5[0.0] = 13.54
clust2_5[0.1] = 8.085
clust2_5[0.2] = 4.855
clust2_5[0.3] = 2.5
clust2_5[0.4] = 1.585
clust2_5[0.5] = 1.045

clust3_1[0.1] = 4.315
clust3_1[0.2] = 3.9
clust3_1[0.3] = 3.065
clust3_1[0.4] = 2.345
clust3_1[0.5] = 1.585

clust3_5[0] = 24.26
clust3_5[0.1] = 6.775
clust3_5[0.2] = 5.11
clust3_5[0.3] = 3.71
clust3_5[0.4] = 2.45
clust3_5[0.5] = 1.815

clust5_20[0.1] = 71.665
clust5_20[0.2] = 48.935
clust5_20[0.3] = 28.385
clust5_20[0.4] = 15.625
clust5_20[0.5] = 8.685



graham = {}
graham[0.0] = 4.22
graham[0.1] = 2.98
graham[0.2] = 2.0
graham[0.3] = 1.33
graham[0.4] = 0.79
graham[0.5] = 0.44

corrmat = {}
corrmat[0] = 124.55
corrmat[0.2] = 31.56
corrmat[0.4] = 8.66


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
plt.plot(clust2_5.keys(), clust2_5.values(), marker='x', color='k', markersize=marker_size, label='clusteron 2_5')
plt.plot(clust3_5.keys(), clust3_5.values(), marker='+', color='r', markersize=marker_size, label='clusteron 3_5')
plt.plot(graham.keys(), graham.values(),  marker='d', color='b', markersize=marker_size, label='Wilshaw')
plt.plot(corrmat.keys(), corrmat.values(),  marker='o', color='g', markersize=marker_size, label='triple correlation')

plt.xlabel("Noise", size=font_size)
plt.ylabel("Capacity per neuron", size=font_size)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
plt.legend(loc='upper right', fontsize=15)

plt.show()