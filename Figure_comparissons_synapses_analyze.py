import pickle
import numpy as np
import matplotlib.pyplot as plt

datafile = 'graham_comparisons.p'
datafile1 = 'data/graham_comparisons_synapsesN150.p'
datafile2 = 'data/graham_comparisons_synapses.p'
datafile3 = 'data/graham_comparisons_synapsesN200.p'

datafile4 = 'data/graham_comparisons_synapses3N100.p'
datafile5 = 'data/graham_comparisons_synapses3_1N100.p'
datafile6 = 'data/graham_comparisons_synapses3_2N100.p'
datafile9 = 'data/graham_comparisons_synapses3_5N100.p'
datafile10 = 'data/graham_comparisons_synapses3_6N100.p'

with open(datafile1) as file:
    data1 = pickle.load(file)

with open(datafile2) as file:
    data2 = pickle.load(file)

with open(datafile3) as file:
    data3 = pickle.load(file)
with open(datafile4) as file:
    data4 = pickle.load(file)
with open(datafile5) as file:
    data5 = pickle.load(file)
with open(datafile6) as file:
    data6 = pickle.load(file)

with open(datafile9) as file:
    data9 = pickle.load(file)

with open(datafile10) as file:
    data10 = pickle.load(file)


num_s = []
cap1 = []
cap2 = []
cap3 = []
# for 3 clusters
cap4 = []
cap5 = []
cap6 = []

cap_vs_syn = {}
cap_vs_syn_4 = {}
cap_vs_syn_4[1] = 1.78
cap_vs_syn_4[6] = 17.06
cap_vs_syn_4[11] = 31.49
cap_vs_syn_4[16] = 46.91
cap_vs_syn_4[21] = 62.99
cap_vs_syn_4[26] = 80.05
cap_vs_syn_4[31] = 97.11
cap_vs_syn_4[36] = 114.45
cap_vs_syn_4[41] = 131.45
cap_vs_syn_4[46] = 148.63
cap_vs_syn_4[51] = 166.4
cap_vs_syn_4[56] = 183.46
cap_vs_syn_4[61] = 201.03
cap_vs_syn_4[66] = 218.51
cap_vs_syn_4[71] = 236.09
cap_vs_syn_4[76] = 253.48
cap_vs_syn_4[81] = 271.28
cap_vs_syn_4[86] = 288.84
cap_vs_syn_4[91] = 305.88
cap_vs_syn_4[96] = 323.46
cap_vs_syn_4[101] = 341.26
cap_vs_syn_4[106] = 358.46

for i in data1:
    N, activation, num_synapses, cluster_size, connections, noise, capacity, dist_full, patterns, accuracies, fullness = i
    cap1.append(capacity)

for i in data2:
    N, activation, num_synapses, cluster_size, connections, noise, capacity, dist_full, patterns, accuracies, fullness = i
    num_s.append(num_synapses)
    cap2.append(capacity)

for i in data3:
    N, activation, num_synapses, cluster_size, connections, noise, capacity, dist_full, patterns, accuracies, fullness = i
    print i
    cap3.append(capacity)


for i in data4:
    N, activation, num_synapses, cluster_size, connections, noise, capacity, dist_full, patterns, accuracies, fullness = i

    print i
    cap_vs_syn[num_synapses] = capacity
    cap4.append(capacity)


for i in data5:
    N, activation, num_synapses, cluster_size, connections, noise, capacity, dist_full, patterns, accuracies, fullness = i
    print i
    cap5.append(capacity)
    cap_vs_syn[num_synapses] = capacity

for i in data6:
    N, activation, num_synapses, cluster_size, connections, noise, capacity, dist_full, patterns, accuracies, fullness = i
    print i
    cap6.append(capacity)
    cap_vs_syn[num_synapses] = capacity


for i in data9:
    N, activation, num_synapses, cluster_size, connections, noise, capacity, dist_full, patterns, accuracies, fullness = i
    print i
    cap_vs_syn[num_synapses] = capacity
    # cap6.append(capacity)


for i in data10:
    N, activation, num_synapses, cluster_size, connections, noise, capacity, dist_full, patterns, accuracies, fullness = i
    print i
    cap_vs_syn[num_synapses] = capacity
    # cap6.append(capacity)


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
print cap_vs_syn
plt.plot(num_s, np.mean(dt, axis=0))
plt.plot(cap_vs_syn.keys(), cap_vs_syn.values())
plt.xlabel("Number of synapses")
plt.ylabel("Capacity per neuron")
plt.xticks(range(0,60,5))
plt.show()