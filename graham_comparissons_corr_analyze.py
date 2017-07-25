import pickle
import numpy as np
import matplotlib.pyplot as plt

datafile = 'graham_comparisons_corrmat.p'
# datafile1 = 'data/graham_comparisons_synapsesN150.p'
# datafile2 = 'data/graham_comparisons_synapses.p'
# datafile3 = 'data/graham_comparisons_synapsesN200.p'

with open(datafile) as file:
    data = pickle.load(file)

num_s = []
cap1 = []
cap2 = []
cap3 = []

for i in data:
    N, activation, connections, noise, capacity, patterns, accuracies = i
    print i
    if N == 100 and activation == 0.05:
        print np.array(patterns)*np.array(accuracies)
        cap1.append(capacity)

# for i in data:
#     N, activation, num_synapses, cluster_size, connections, noise, capacity, dist_full, patterns, accuracies, fullness = i
#     if N == 200 and activation == 0.07 and cluster_size == 2:
#         num_s.append(num_synapses)
#         cap.append(capacity)
#         print capacity, num_synapses, cluster_size, np.array(patterns)*np.array(accuracies)

dt = np.zeros((3, len(num_s)))
dt[0] = cap2
dt[1] = cap1
dt[2] = cap3
print dt
print np.mean(dt, axis=0)
# plt.plot(num_s, cap1)
# plt.plot(num_s, cap3)
plt.plot(num_s, np.mean(dt, axis=0))
plt.xlabel("Number of synapses")
plt.ylabel("Capacity per neuron")
plt.xticks(range(11))
# plt.ylim([0,10])
plt.show()