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
        self.kernel = np.ones((self.size, self.size))*0.01
        self.kernel[np.diag_indices(self.size)] = 1

        self.kernel2 = np.random.randint(0, 2, size=(self.size, self.size))
        self.kernel2[np.diag_indices(self.size)] = 1

    def init_weights(self, from_field):
        self.from_field = from_field
        self.weights = np.zeros((self.size, from_field.size, from_field.size))

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
        for k in np.nonzero(self.cells)[0]:
            for i in np.nonzero(self.from_field.cells)[0]:
                self.weights[k, i] += self.from_field.cells   # here i store an input vector to weights
        self.weights = np.array(self.weights > 0, dtype='uint8')  # binarization
            # print self.weights[k,:]


    def retrieve(self):
        self.cells2 = np.sum(np.dot(self.weights, self.from_field.cells), axis=1)
        # print self.cells2
        # for (x,y), cell in np.ndenumerate(self.cells):
        #     self.cells2[x,y] = np.sum(self.weights[x, y] * self.from_field.cells)
        # print self.cells2
        self.cells = self.cells2 >= sorted(self.cells2)[-int(self.size*0.05)]



    def get_convol(self, x, y, image, field, radius=1):
        weight = np.zeros(image)
        im_x, im_y = image
        f_x,f_y = field
        r, t = float(im_x)/f_x, float(im_y)/f_y
        # print r,t
        # print int(x*r), int(y*t)
        xl,xh,yl,yh = int(x*r)-radius, int(x*r)+radius, int(y*t)-radius, int(y*t)+radius
        yh = im_y if yh > im_y else yh
        xh = im_x if xh > im_x else xh
        yl = 0 if yl < 0 else yl
        xl = 0 if xl < 0 else xl
        receptive_f = [(i, j) for i in range(xl-1, xh) for j in range(yl-1,yh)]
        weight[zip(*receptive_f)] = 1
        return weight


if __name__ == '__main__':
    import senses

    N = 200
    size_color = N
    size_label = N
    color = Field(size=size_color)
    label = Field(size=size_label)
    R = 500. # number of patterns
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
        print i
    accuracy = 0
    print 'stored'
    # label.weights = label.weights/np.max(label.weights)
    # label.weights[np.diag_indices(200, ndim=3)] = 1
    # label.weights *= label.kernel2
    # retrieve
    for i, word in enumerate(txt.vocabulary):
        color.cells = c.sense(i, noise=0.)
        # color.cells = c.sense(i, noise=0.3)
        # label.cells = txt.sense_full(word)
        label.cells = 0
        label.retrieve()
        results = np.dot(txt.patterns, label.cells)
        inds = np.argpartition(results, -3)[-3:]
        # print inds, results[inds]
        # print sorted(zip([list(txt.vocabulary)[i] for i in inds], results[inds]/float(np.max(results))), key=lambda x: x[1])[::-1]
        if word == txt.vocabulary[np.argmax(results)]:
            accuracy += 1
        if not i%100:
            print i
        # print word, txt.vocabulary[np.argmax(results)]
    print label.weights[:3,:3,:3]
    print np.count_nonzero(label.weights)/float(N ** 3)
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

