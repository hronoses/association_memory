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


    def store(self):
        # self.weights = (self.weights + np.outer(self.cells, self.from_field.cells)) >= 1
        self.weights = self.weights + np.outer(self.cells, self.from_field.cells)


    def retrieve(self):
        self.cells2 = np.dot(self.weights, self.from_field.cells)

        # for (x,y), cell in np.ndenumerate(self.cells):
        #     self.cells2[x,y] = np.sum(self.weights[x, y] * self.from_field.cells)
        # print self.cells2
        # self.cells = np.nonzero(self.cells2[self.cells2 < sorted(self.cells2)[-5]])[0]
        self.cells = self.cells2 >= sorted(self.cells2)[-int(self.size*0.05)]
        # print self.cells2
        # print sorted(self.cells2)
        # print self.cells2[self.cells2 < sorted(self.cells2)[-5]]
        # print self.cells



if __name__ == '__main__':
    import senses

    N = 200
    size_color = N
    size_label = N
    color = Field(size=size_color)
    label = Field(size=size_label)
    R = 1000. # number of patterns
    c = senses.ColorSense(total_number=N, sparsity=0.05, color_size=int(R))
    # c2 = senses.ColorSense(total_number=N, sparsity=0.03, color_size=int(R+1))
    txt = senses.TextSense(total_number=N, sparsity=0.05, vocabulary_size=int(R))

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

    # retrieve
    for i, word in enumerate(txt.vocabulary):
        color.cells = c.sense(i)
        # label.cells = txt.sense_full(word)
        label.cells = 0
        label.retrieve()
        results = np.dot(txt.patterns, label.cells)
        # inds = np.argpartition(results, -3)[-3:]
        # print inds, results[inds]
        # print sorted(zip([list(txt.vocabulary)[i] for i in inds], results[inds]/float(np.max(results))), key=lambda x: x[1])[::-1]
        if word == txt.vocabulary[np.argmax(results)]:
            accuracy += 1
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

