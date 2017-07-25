import numpy as np
import text_sense as tx
import pickle


class Field:
    def __init__(self, size, activation=0.05):
        self.size = size
        self.activation = activation
        self.cells = np.zeros(self.size)
        self.from_field = 0
        self.fullness = 0

    def init_weights(self, from_field):
        self.from_field = from_field
        self.weights = np.zeros((self.size, from_field.size))

    def store(self):
        self.weights = (self.weights + np.outer(self.cells, self.from_field.cells)) >= 1

    def retrieve(self):
        self.cells2 = np.dot(self.weights, self.from_field.cells)
        self.cells = self.cells2 >= sorted(self.cells2)[-int(self.size*self.activation)]



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


    # print np.count_nonzero(label.weights)/float(N **2)
    print float(accuracy)/R
    print time.time() - t

    import matplotlib.pyplot as plt
    plt.plot(patterns, accuracies)
    plt.ylim([0, 1.5])
    plt.show()




