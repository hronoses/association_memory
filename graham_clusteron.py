import numpy as np
import text_sense as tx
import pickle

class InputField:
    def __init__(self):
        # data should be numpy array
        self.cells = 0
        self.size = 0

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
        self.distances = np.sort(np.random.rand(self.size, from_field.size))

    def connect(self, from_field, sparsity=0.1):
        self.from_field = from_field
        filename = 'data/weights_' + str(from_field.size) + '_' + str(sparsity)+ '.pickle'
        try:
            with open(filename, 'rb') as handle:
                data = pickle.load(handle)
            self.weights = data['weight']
            print 'success connect'
        except:
            for x in range(self.size):
                self.weights.append({'num': np.random.randint(from_field.size, size=int(from_field.size*sparsity))})
                self.weights[-1]['str'] = 1
            filename = 'data/weights_' + str(from_field.size) + '_' + str(sparsity)+ '.pickle'
            with open(filename, 'wb') as handle:
                pickle.dump(({'weight': self.weights}), handle)

    def connect_topo(self, from_field, radius=1):
        self.from_field = from_field
        self.weights = np.zeros(self.size + from_field.size)
        for (x, y), cell in np.ndenumerate(self.cells):
            self.weights[x, y] = self.get_convol(x, y, from_field.size, self.size, radius=radius)

    def store(self):
        xy = np.outer(self.cells, self.from_field.cells)
        self.weights = (self.weights + xy) >= 1
        # self.weights = self.weights + np.outer(self.cells, self.from_field.cells)
        for i in range(self.size):
            dst = self.distances[i, np.nonzero(xy[i])][0]
            # print np.nonzero(xy[i])
            if len(dst):
                self.distances[i] += 0.1*(np.mean(dst) - self.distances[i])  # one cluster
                # print self.distances[i]
            # print self.distances[i,:][np.nonzero(self.weights[i])]

            # num_neighbor = 2
            # dst_mean = []
            #
            # dst_sort = np.sort(dst)
            # dst_ind = np.argsort(dst)
            # dst_mean = np.zeros(len(dst))
            # dst_copy = np.copy(self.distances[i])
            # if len(dst):
            #     print self.distances[i, :][np.nonzero(self.weights[i])]
            #     for index in range(len(dst)):
                    # print dst_sort[max(0, index-num_neighbor):num_neighbor+index+1]
                    # right= dst_sort[index+1:num_neighbor+index+1]
                    # print dst_sort[i-neighbor:i+neighbor]
                    # dst_mean[index] = np.mean(dst_sort[max(0, index-num_neighbor):num_neighbor+index+1])
                # self.distances[i, :][np.nonzero(self.weights[i])][dst_ind] = dst_mean
                # print self.distances[i, :][np.nonzero(self.weights[i])]
                # print self.distances[i]
                # print dst
                # print dst_sort
                # print dst_mean
                # print self.distances[i] == dst_copy
            # print i

    def gaussian(self, x, mu, sig=1):
        return 1 + np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


    def sum_gauss(self, m=[]):
        l = np.linspace(0, 1, 100)
        field = self.gaussian(l, m[0])
        for i in m[1:]:
            field *= self.gaussian(l, i)
        return np.sum(field)

    def get_overlap(self, distances):
        result = [self.sum_gauss(distances[i]) for i in range(distances.shape[0])]
        return np.array(result)

    def retrieve(self):
        overlap = self.get_overlap(self.distances[:, np.nonzero(self.from_field.cells)[0]])
        self.cells2 = np.dot(self.weights, self.from_field.cells) * overlap/np.max(overlap)
        # print np.nonzero(self.from_field.cells)
        # print self.distances[:, np.nonzero(self.from_field.cells)[0]]
        # print overlap/np.max(overlap)

        # for (x,y), cell in np.ndenumerate(self.cells):
        #     self.cells2[x,y] = np.sum(self.weights[x, y] * self.from_field.cells)
        # print self.cells2
        # self.cells = np.nonzero(self.cells2[self.cells2 < sorted(self.cells2)[-5]])[0]
        self.cells = self.cells2 >= sorted(self.cells2)[-10]
        # print self.cells2
        # print sorted(self.cells2)
        # print self.cells2[self.cells2 < sorted(self.cells2)[-5]]
        # print self.cells




if __name__ == '__main__':
    import senses
    import sys
    import matplotlib.pyplot as plt

    N = 200
    size_color = N
    size_label = N
    color = Field(size=size_color)
    label = Field(size=size_label)
    R = 500. # number of patterns
    c = senses.ColorSense(total_number=N, sparsity=0.05, color_size=int(R))
    # c2 = senses.ColorSense(total_number=N, sparsity=0.03, color_size=int(R+1))
    txt = senses.TextSense(total_number=N, sparsity=0.05, vocabulary_size=int(R))

    # print label.sum_gauss([0.4, 0.5])
    # for mu, sig in [(0.65, 1), (0, 2), (0.8, 3)]:
    #     plt.plot(label.gaussian(np.linspace(0, 1, 100), mu, sig))
    #
    # plt.show()
    # sys.exit()
    label.init_weights(color)
    import time
    t = time.time()
    print 'started'

    #storing
    for i, word in enumerate(txt.vocabulary):
        color.cells = c.sense(i)
        # label.cells = c2.sense(i)
        label.cells = txt.sense_full(word)
        label.store()
    accuracy = 0
    # label.weights = label.weights/np.max(label.weights)
    print 'stored'
    # retrieve
    for i, word in enumerate(txt.vocabulary):
        color.cells = c.sense(i)
        # label.cells = txt.sense_full(word)
        label.cells = 0
        label.retrieve()
        results = np.dot(txt.patterns, label.cells)
        inds = np.argpartition(results, -3)[-3:]
        # print inds, results[inds]
        # print sorted(zip([list(txt.vocabulary)[i] for i in inds], results[inds]/float(np.max(results))), key=lambda x: x[1])[::-1]
        if word == txt.vocabulary[np.argmax(results)]:
            accuracy += 1
        print i
        # print word, txt.vocabulary[np.argmax(results)]

    print np.count_nonzero(label.weights)/float(N **2)
    print accuracy/R
    print time.time() - t

    # import time
    # t = time.time()
    # print 'started'
    # A = [T1.evolve() for _ in range(100)]
    # for i in range(1000):
        # tx_input.cells = np.random.poisson(0.05, size=size_input)
        # T1.evolve()
        # print i
    # print time.time() - t

