import numpy as np
import pickle

class Field:
    def __init__(self, size, activation=0.05, connections=1, num_synapses=1, cluster_size=2):
        self.size = size
        self.activation = activation
        self.connections = connections
        self.num_synapses = num_synapses
        self.cluster_size = cluster_size
        self.cells = np.zeros(self.size)
        self.from_field = 0
        self.fullness = 0

    def init_weights(self, from_field):
        self.from_field = from_field
        self.weights = np.zeros((self.size, from_field.size))
        self.distances = np.zeros((self.size, from_field.size, self.num_synapses))

        sparsity = self.connections # show percentage of connected input neurons to target
        self.connections = np.random.choice([0, 1], size=(self.size, from_field.size), p=[1-sparsity, sparsity])


    def store_old(self):
        for i in np.nonzero(self.cells)[0]:
            # self.distances[i] is a matrix, where return array of indices [(rows), (cols)]
            overlap = np.intersect1d(np.where(self.distances[i] == 0)[0], np.nonzero(self.connections[i]*self.from_field.cells)[0])
            #  select two active pre neurons
            if overlap.size > 1 and self.cluster_size == 2:
                pre1, pre2 = np.random.choice(overlap, 2, replace=False)
                a = np.random.choice(np.where(self.distances[i, pre1] == 0)[0])
                b = np.random.choice(np.where(self.distances[i, pre2] == 0)[0])
                self.distances[i, [pre1, pre2], [a, b]] = np.max(self.distances[i]) + 1  # max +1 is number of cluster
            elif overlap.size > 2 and self.cluster_size == 3:
                pre1, pre2, pre3 = np.random.choice(overlap, 3, replace=False)
                a = np.random.choice(np.where(self.distances[i, pre1] == 0)[0])
                b = np.random.choice(np.where(self.distances[i, pre2] == 0)[0])
                c = np.random.choice(np.where(self.distances[i, pre3] == 0)[0])
                self.distances[i, [pre1, pre2, pre3], [a, b, c]] = np.max(self.distances[i]) + 1  # max +1 is number of cluster
            else:
                self.fullness += 1

    def store(self):
        for i in np.nonzero(self.cells)[0]:
            # self.distances[i] is a matrix, where return array of indices [(rows), (cols)]
            overlap = np.intersect1d(np.where(self.distances[i] == 0)[0], np.nonzero(self.connections[i]*self.from_field.cells)[0])
            #  select two active pre neurons
            if overlap.size >= self.cluster_size:
                pre = np.random.choice(overlap, self.cluster_size, replace=False)
                pre_tree = []
                for k in range(self.cluster_size):
                    pre_tree.append(np.random.choice(np.where(self.distances[i, pre[k]] == 0)[0]))
                    # b = np.random.choice(np.where(self.distances[i, pre2] == 0)[0])
                self.distances[i, pre, pre_tree] = np.max(self.distances[i]) + 1  # max +1 is number of cluster
            else:
                self.fullness += 1


    def retrieve_slower(self):
        # print np.nonzero(self.from_field.cells)[0]
        dst = self.distances[:, np.nonzero(self.from_field.cells)[0]]
        counts = np.zeros(200)
        for i, dt in enumerate(dst):
            _, count = np.unique(dt[np.nonzero(dt)[0],np.nonzero(dt)[1]], return_counts=True)
            if len(count):
                counts[i] = np.max(count)
            else:
                counts[i] = 0
        self.cells = counts >= np.sort(counts)[-int(self.size*0.05)]

    def retrieve(self):
        dst = self.distances[:, np.nonzero(self.from_field.cells)[0]]
        # find counts of input*distance to see if there is a cluster
        data = [np.unique(dt[np.nonzero(dt)[0], np.nonzero(dt)[1]], return_counts=True) for dt in dst]
        # find max in each count to see if there is cluster (2 if there is a cluster)
        data2 = [np.max(counts) if counts.size else 0 for counts in np.array(data)[:, 1]]
        self.cells = data2 >= np.sort(data2)[-int(self.size*self.activation)]


if __name__ == '__main__':
    import senses
    import sys
    import matplotlib.pyplot as plt

    N = 100
    activation = 0.05
    size_color = N
    size_label = N
    color = Field(size=size_color)
    label = Field(size=size_label, activation=activation, connections=1, num_synapses=30, cluster_size=3)
    R = 10000 # number of patterns
    c = senses.ColorSense(total_number=N, sparsity=activation, color_size=R)
    num = senses.NumberSense(total_number=N, sparsity=activation, number_size=R)


    label.init_weights(color)

    patterns = []
    accuracies = []

    import time
    t = time.time()
    print 'started'

    #storing
    for i in xrange(R):
        color.cells = c.sense(i)
        label.cells = num.sense(i)
        # label.store_old()
        label.store()
        # label.store2()
    accuracy = 0

    print 'stored, fullness=',
    print label.fullness
    print 'Ratio of nonzeros in distances ',
    print float(np.count_nonzero(label.distances))/label.distances.size


    # retrieve
    for i in xrange(R):
        color.cells = c.sense(i)
        label.cells = 0
        label.retrieve()
        # label.retrieve_slower()
        results = np.dot(num.patterns, label.cells)
        if i == np.argmax(results):
            accuracy += 1
        if not i%100:
            print i
            print float(accuracy)/(i+1)
            print float(accuracy)/N



    # print np.count_nonzero(label.weights)/float(N **2)
    print float(accuracy)/R
    print float(accuracy)/N
    print time.time() - t


    import matplotlib.pyplot as plt
    plt.plot(patterns, accuracies)
    plt.ylim([0, 1.5])
    # plt.show()


