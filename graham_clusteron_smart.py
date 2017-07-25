import numpy as np
import pickle

class Field:
    def __init__(self, size):
        self.size = size
        self.cells = np.zeros(self.size)
        self.from_field = 0
        self.spikes = 0
        self.spikes_next = 0

    def init_weights(self, from_field):
        self.from_field = from_field
        self.weights = np.zeros((self.size, from_field.size))
        self.distances = np.zeros((self.size, from_field.size))

        sparsity = 1 # show percentage of connected input neurons to target
        self.connections = np.random.choice([0, 1], size=(self.size, from_field.size), p=[1-sparsity, sparsity])


    def store(self):
        for i in np.nonzero(self.cells)[0]:
            overlap = np.intersect1d(np.where(self.distances[i] == 0), np.nonzero(self.connections[i]*self.from_field.cells)[0])
            #  select two active pre neurons
            if overlap.size > 1:
                pre1, pre2 = np.random.choice(overlap, 2, replace=False)
                self.distances[i, [pre1, pre2]] = np.max(self.distances[i]) + 1  # max +1 is number of cluster
            else:
                print 'full'


    def retrieve_slower(self):
        # print np.nonzero(self.from_field.cells)[0]
        dst = self.distances[:, np.nonzero(self.from_field.cells)[0]]
        counts = np.zeros(200)
        for i, dt in enumerate(dst):
            _, count = np.unique(dt[np.nonzero(dt)[0]], return_counts=True)
            if len(count):
                counts[i] = np.max(count)
            else:
                counts[i] = 0
            # print i.shape
        # print np.sort(counts)
        # data = [np.unique(i, return_counts=True) for i in dst.T]

    def retrieve(self):
        dst = self.distances[:, np.nonzero(self.from_field.cells)[0]]
        # find counts of input*distance to see if there is a cluster
        data = [np.unique(dt[np.nonzero(dt)[0]], return_counts=True) for dt in dst]
        # find max in each count to see if there is cluster (2 if there is a cluster)
        data2 = [np.max(counts) if counts.size else 0 for counts in np.array(data)[:, 1]]
        self.cells = data2 >= np.sort(data2)[-int(self.size*0.05)]


if __name__ == '__main__':
    import senses
    import sys
    import matplotlib.pyplot as plt

    N = 200
    size_color = N
    size_label = N
    color = Field(size=size_color)
    label = Field(size=size_label)
    R = 2000 # number of patterns
    c = senses.ColorSense(total_number=N, sparsity=0.05, color_size=R)
    # c2 = senses.ColorSense(total_number=N, sparsity=0.03, color_size=int(R+1))
    # txt = senses.TextSense(total_number=N, sparsity=0.05, vocabulary_size=int(R))
    num = senses.NumberSense(total_number=N, sparsity=0.05, number_size=R)


    label.init_weights(color)

    patterns = []
    accuracies = []

    import time
    t = time.time()
    print 'started'

    #storing
    for i in xrange(R):
        color.cells = c.sense(i)
        # label.cells = c2.sense(i)
        label.cells = num.sense(i)
        label.store()
        # label.store2()
    accuracy = 0

    print 'stored'
    print np.count_nonzero(label.distances)
    print label.distances.shape

    # retrieve
    for i in xrange(R):
        color.cells = c.sense(i)
        label.cells = 0
        label.retrieve()
        results = np.dot(num.patterns, label.cells)
        if i == np.argmax(results):
            accuracy += 1
        if not i%100:
            print i
            try:
                accuracies.append(float(accuracy)/(i+1))
                patterns.append(i)
            except:
                pass



    print np.count_nonzero(label.weights)/float(N **2)
    print float(accuracy)/R
    print time.time() - t


    import matplotlib.pyplot as plt
    plt.plot(patterns, accuracies)
    plt.ylim([0, 1.5])
    plt.show()


