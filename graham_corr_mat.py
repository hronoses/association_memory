import numpy as np

class Field:
    def __init__(self, size,  activation=0.05, connectivity=1):
        self.size = size
        self.activation = activation
        self.connectivity = connectivity
        self.cells = np.zeros(self.size, dtype='bool')
        self.from_field = 0
        self.fullness = 0

    def init_weights(self, from_field):
        self.from_field = from_field
        self.corr = np.zeros((self.size, from_field.size, from_field.size), dtype='bool')
        self.connections = np.random.choice([0, 1], size=(self.size, from_field.size), p=[1-self.connectivity, self.connectivity])

    def store(self):
        # self.corr += np.multiply.outer(self.cells, np.outer(self.from_field.cells, self.from_field.cells))
        self.corr = (self.corr + np.multiply.outer(self.cells, np.outer(self.from_field.cells, self.from_field.cells))) >= 1

    def retrieve(self):
        self.cells2 = np.sum(np.sum(self.connections*self.corr*np.outer(self.from_field.cells, self.from_field.cells), axis=1), axis=1)
        self.cells = self.cells2 >= sorted(self.cells2)[-int(self.size*self.activation)]


if __name__ == '__main__':
    import senses
    import sys
    N = 100
    activation = 0.07
    size_color = N
    size_label = N
    color = Field(size=size_color)
    label = Field(size=size_label)
    R = 8000 # number of patterns
    c = senses.ColorSense(total_number=N, sparsity=activation, color_size=R)
    num = senses.NumberSense(total_number=N, sparsity=activation, number_size=R)

    label.init_weights(color)
    import time
    t = time.time()
    print 'started'

    #storing
    for i in xrange(R):
        color.cells = c.sense(i)
        label.cells = num.sense(i)
        label.store()
    accuracy = 0
    print 'stored. Nonzeros in corr mat: ',
    print np.count_nonzero(label.corr)/float(N ** 3)

    patterns = []
    accuracies = []
    # retrieve
    for i in xrange(R):
        color.cells = c.sense(i, noise=0.)
        label.cells = 0
        label.retrieve()
        results = np.dot(num.patterns, label.cells)
        if i == np.argmax(results):
            accuracy += 1
        if not i%100:
            print i
            accuracies.append(float(accuracy)/(i+1))
            patterns.append(i)
            print float(accuracy)/(i+1)
        # print word, txt.vocabulary[np.argmax(results)]

    print float(accuracy)/R
    print time.time() - t

    print accuracies

    import matplotlib.pyplot as plt
    plt.plot(patterns, accuracies)
    plt.ylim([0, 1.5])
    plt.show()


