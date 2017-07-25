import pickle
import numpy as np
import matplotlib.pyplot as plt

datafile = 'graham_comparisons.p'
datafile2 = 'graham_comparisons_corrmat.p'

with open(datafile) as file:
    data = pickle.load(file)

with open(datafile2) as file:
    data2 = pickle.load(file)

# for i in data:
#     print i
    # N, activation, num_synapses, cluster_size, connections, noise, capacity, dist_full, patterns, accuracies, fullness = i
    # cap1.append(capacity)


for i in data2:
    print i
    # N, activation, num_synapses, cluster_size, connections, noise, capacity, dist_full, patterns, accuracies, fullness = i
    # cap1.append(capacity)



# for i in data:
#     N, activation, num_synapses, cluster_size, connections, noise, capacity, dist_full, patterns, accuracies, fullness = i
#     if N == 200 and activation == 0.07 and cluster_size == 2:
#         num_s.append(num_synapses)
#         cap.append(capacity)
#         print capacity, num_synapses, cluster_size, np.array(patterns)*np.array(accuracies)
#
# dt = np.zeros((3, len(num_s)))
# dt[0] = cap2
# dt[1] = cap1
# dt[2] = cap3
# print dt
# print np.mean(dt, axis=0)
# # plt.plot(num_s, cap1)
# # plt.plot(num_s, cap3)
# print cap_vs_syn
# plt.plot(num_s, np.mean(dt, axis=0))
# plt.plot(cap_vs_syn.keys(), cap_vs_syn.values())
# plt.xlabel("Number of synapses")
# plt.ylabel("Capacity per neuron")
# plt.xticks(range(0,60,5))
# plt.show()