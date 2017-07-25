import numpy as np
import text_sense as tx
import pickle

class InputField:
    def __init__(self):
        # data should be numpy array
        self.cells = 0
        self.size = 0

class Field:
    def __init__(self, size, activation=0.05, connectivity=1, noise=0):
        self.size = size
        self.activity = activation
        self.connectivity = connectivity
        self.cells = np.zeros(self.size)
        self.from_field = 0
        self.fullness = 0

    def init_weights(self, from_field):
        self.from_field = from_field
        self.weights = np.zeros((self.size, from_field.size))
        self.connections = np.random.choice([0, 1], size=(self.size, from_field.size), p=[1-self.connectivity, self.connectivity ])

    def store(self):
        self.weights = (self.weights + np.outer(self.cells, self.from_field.cells)) >= 1

    def retrieve(self):
        self.cells2 = np.dot(self.weights*self.connections, self.from_field.cells)
        # add noramlization retrieval based on activity
        # it works because with partial connectivity cell that receive less active inputs but have larger activation
        #  (due to larger w=1 amount), is more probable stored pattern, and not cell that just due to connectivity
        # connected to more active neuron
        self.activations = np.dot(self.connections, self.from_field.cells)
        self.cells2 = self.cells2/self.activations

        self.cells = self.cells2 >= sorted(self.cells2)[-int(self.size*self.activity)]




if __name__ == '__main__':
    import senses

    N = 200
    size_color = N
    size_label = N
    color = Field(size=size_color)
    label = Field(size=size_label)
    R = 1000 # number of patterns
    c = senses.ColorSense(total_number=N, sparsity=0.05, color_size=R)
    # c2 = senses.ColorSense(total_number=N, sparsity=0.03, color_size=int(R+1))
    num = senses.NumberSense(total_number=N, sparsity=0.05, number_size=R)

    label.init_weights(color)
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
    patterns = []
    accuracies = []

    # retrieve
    for i in xrange(R):
        color.cells = c.sense(i, noise=0.2)
        label.cells = 0
        label.retrieve()
        results = np.dot(num.patterns, label.cells)
        if i == np.argmax(results):
            accuracy += 1
        if not i%100 and i > 0:
            print i
            try:
                accuracies.append(float(accuracy)/(i+1))
                patterns.append(i)
            except:
                pass


    # print np.count_nonzero(label.weights)/float(N **2)
    print float(accuracy)/R
    print time.time() - t
    print accuracies
    print patterns
    print np.array(accuracies)*np.array(patterns)
    import matplotlib.pyplot as plt
    plt.plot(patterns, accuracies)
    plt.ylim([0, 1.5])
    plt.show()

