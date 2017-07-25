import numpy as np
import text_sense as tx
import pickle


class Cell:
    def __init__(self, size=200):
        self.state = 0
        self.size = size
        self.patterns = []
        self.data_patterns = []

    def make_binary(self, num_pattern):
        a = np.zeros(self.size)
        a[self.patterns[num_pattern]] = 1
        return a

    def get_pattern_matrix(self):
        patt = np.zeros((len(self.patterns)+1, self.size))
        # print len(self.patterns)
        for i in range(len(self.patterns)):
            patt[i, self.patterns[i]] = 1
            # print patt[i,:]

        return patt

    def get_pattern_matrix_custom(self):
        patt = np.array(self.patterns).flatten()
        if len(patt):
            data = np.zeros(len(self.patterns)*self.size)
            data[patt] = 1
            return data.reshape((-1, self.size))
        else:
            return np.zeros(self.size)



class Field:
    def __init__(self, size, activation=0.05):
        self.size = size
        self.activation = activation
        self.cells_explicit = [Cell(size=self.size) for _ in xrange(self.size)]
        self.cells = np.zeros(self.size)
        self.from_field = 0
        self.fullness = 0

        self.stored = 0

    def init_weights(self, from_field):
        self.from_field = from_field
        self.weights = np.zeros((self.size, from_field.size, from_field.size))
        sparsity = 1  # show percentage of connected input neurons to target
        self.connections = np.random.choice([0, 1], size=(self.size, from_field.size), p=[1-sparsity, sparsity])


    # def store2(self):
    #     for i in np.nonzero(self.cells)[0]:
    #         self.cells_explicit[i].patterns.append(np.random.choice(np.nonzero(self.from_field.cells)[0], 2, replace=False))  # here i store explicitly what pattern elicit excitation
        # self.weights = (self.weights + np.outer(self.cells, self.from_field.cells)) >= 1

    def store(self):
        # here i store not just indises but indises in future faltten array
        for i in np.nonzero(self.cells)[0]:
            self.cells_explicit[i].patterns.append(np.random.choice(np.nonzero(self.from_field.cells)[0], 3, replace=False)+self.size*len(self.cells_explicit[i].patterns))  # here i store explicitly what pattern elicit excitation
        self.stored = 1
        # self.weights = (self.weights + np.outer(self.cells, self.from_field.cells)) >= 1

    def retrieve(self):
        if self.stored:
            for cell in self.cells_explicit:
                cell.data_patterns = cell.get_pattern_matrix_custom()
            self.stored = 0
        # second experimental method

        self.cells2 = np.zeros(self.size)
        for i, cell in enumerate(self.cells_explicit):
            self.cells2[i] = np.max(np.dot(cell.data_patterns, self.connections[i]*self.from_field.cells))
            # self.cells2[i] = np.max(np.dot(cell.get_pattern_matrix_custom(), self.from_field.cells))
        self.cells = self.cells2 >= sorted(self.cells2)[-int(self.size*self.activation)]



if __name__ == '__main__':
    import senses
    import sys
    import scipy.spatial.distance as hm
    N = 100
    activation = 0.05
    size_color = N
    size_label = N
    color = Field(size=size_color)
    label = Field(size=size_label, activation=activation)
    R = 2500 # number of patterns
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


    # for cell in range(100):
        # cell = 101
        # num = len(label.cells_explicit[cell].patterns)
        # overlap = np.array([[np.sum(label.cells_explicit[cell].make_binary(i) * label.cells_explicit[cell].make_binary(j)) for i in range(num)] for j in range(num)], dtype=int)
        # overlap -= np.eye(num, dtype=int)*10
        # print overlap
        # print np.max(overlap)
    # print len( label.cells_explicit[10].patterns)
    # print label.cells_explicit[10].patterns
    # print label.cells_explicit[10].get_pattern_matrix_custom()
    # print label.cells_explicit[10].get_pattern_matrix_custom().shape
    # print np.nonzero(label.cells_explicit[10].get_pattern_matrix_custom())[1].reshape((-1, 2))
    means = []
    num_syn = np.zeros(50)
    for i in range(N):
        unique, counts = np.unique(np.nonzero(label.cells_explicit[i].get_pattern_matrix_custom())[1], return_counts=True)
        un1, coun1 = np.unique(counts, return_counts=True)
        for i, val in enumerate(coun1):
            num_syn[i] += val

        # num_syn.append([coun1])
        # print counts
        # print coun1
        # print counts
        means.append(np.mean(counts))
    print np.mean(means)
    print list(num_syn/N)
    import matplotlib.pyplot as plt
    plt.plot(range(1,11), num_syn[:10]/N)
    # plt.show()

    import sys
    sys.exit()
    # a = np.zeros(200)
    # a[1] = 1
    # all_inputs = np.sort(np.array(label.cells_explicit[10].patterns).flatten())
    # print all_inputs
    # print all_inputs.shape
    # print len(set(all_inputs))
    # unique, counts = np.unique(all_inputs, return_counts=True)
    # counts - is how many synapse
    # print counts
    # print np.sum(counts <= 2)
    # print np.sum(counts) # should be equal approximately R
    # for i in range(200):

    # print np.dot(label.cells_explicit[10].get_pattern_matrix(), a)
    # import matplotlib.pyplot as plt
    # plt.imshow(overlap >= 3, interpolation='none')
    # plt.colorbar()
    # plt.show()
    # print np.sum(label.cells_explicit[10].make_binary(1) * label.cells_explicit[10].make_binary(0))
    # print len(label.cells_explicit[10].patterns)
    # print hm.hamming(label.cells_explicit[10].patterns[0], label.cells_explicit[10].patterns[1])
    # print label.cells_explicit[10].patterns

    # sys.exit()





    patterns = []
    accuracies = []
    # retrieve
    for i in xrange(R):
        color.cells = c.sense(i, noise=0.)
        label.cells = 0
        label.retrieve()
        results = np.dot(num.patterns, label.cells)
        # inds = np.argpartition(results, -3)[-3:]
        # print inds, results[inds]
        # print sorted(zip([list(txt.vocabulary)[i] for i in inds], results[inds]/float(np.max(results))), key=lambda x: x[1])[::-1]
        if i == np.argmax(results):
            accuracy += 1
        if not i%100:
            print i
            accuracies.append(float(accuracy)/(i+1))
            patterns.append(i)
            print float(accuracy)/(i+1)
        # print word, txt.vocabulary[np.argmax(results)]
    # print label.weights[:3,:3,:3]
    print np.count_nonzero(label.weights)/float(N ** 3)
    print accuracy/R
    print time.time() - t

    import matplotlib.pyplot as plt
    plt.plot(patterns, accuracies)
    plt.ylim([0, 1.5])
    plt.show()



