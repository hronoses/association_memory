import string
import numpy as np
import os
import itertools
import pickle


class ColorSense2:
    def __init__(self, total_number=20, sparsity=0.1):
        self.color_range = 180
        self.N = total_number
        self.n = int(total_number*sparsity)
        self.datafile = 'data/color_data'+str(total_number)+'_'+str(sparsity)+'.p'

        try:
            with open(self.datafile) as file:
                data = pickle.load(file)
                self.patterns = data['patterns']
        except:
            self.patterns = np.zeros((self.color_range, self.N), dtype='uint8')
            for i in range(self.color_range):
                self.patterns[i] = np.random.choice([0, 1], size=(self.N,), p=[1-sparsity, sparsity])
                # self.L1[i] = np.random.choice(self.L1_len, self.L1_pat, replace=False)
            print 'new color data'
            with open(self.datafile, 'wb') as file:
                pickle.dump({'patterns': self.patterns}, file)

    def sense(self, data=0, noise=0.):
        if data == -1:
            return np.zeros(self.N, dtype='uint8')
        if noise > 0:
            patt = np.copy(self.patterns[data])
            a = np.nonzero(patt)[0]
            patt[np.random.choice(a, int(len(a)*noise))] = 0   # get several ones in array and make it zeros
            patt[np.random.randint(0,self.N, size=int(len(a)*noise))] = 1   # put back ones at different locations
            return patt
        return self.patterns[data]

class ColorSense:
    def __init__(self, total_number=20, sparsity=0.1, color_size=180):
        self.color_size = color_size
        self.N = total_number
        self.n = int(total_number*sparsity) # number of active cells
        self.datafile = 'data/color_data_col_size_'+str(color_size) + '_' +str(total_number)+'_'+str(sparsity)+'.p'
        if not os.path.exists('data/'):
            os.makedirs('data/')
        try:
            with open(self.datafile) as file:
                data = pickle.load(file)
                self.pattern = data['pattern']
                self.n = data['n']
                self.N = data['N']
        except:
            self.pattern = np.zeros((self.color_size, self.n), dtype=int)      # shows active cells for every letter in vocabulary
            for i in range(self.color_size):
                self.pattern[i] = np.random.choice(self.N, self.n, replace=False) #randomly store number of cell that should be active
            with open(self.datafile, 'wb') as file:
                pickle.dump({'N': self.N, 'n': self.n, 'pattern': self.pattern}, file)
            print 'new color data'
        #  interesting another form to make binary vector.
        #  however you can set only probability of number of ones, not exact number
        # self.patterns[i] = np.random.choice([0, 1], size=(self.N,), p=[1-sparsity, sparsity])

    def sense(self, data=0, noise=0.):
        bin_patterns = np.zeros(self.N, dtype='uint8')
        bin_patterns[self.pattern[data]] = 1
        if data == -1:
            return np.zeros(self.N, dtype='uint8')
        if noise > 0:
            patt = np.copy(bin_patterns)
            a = np.nonzero(patt)[0]
            patt[np.random.choice(a, int(len(a)*noise))] = 0   # get several ones in array and make it zeros
            patt[np.random.randint(0,self.N, size=int(len(a)*noise))] = 1   # put back ones at different locations
            return patt
        return bin_patterns


class TextSense:
    def __init__(self, total_number=20, sparsity=0.1, vocabulary_size=-1):
        self.vocabulary = self.get_vocabulary(vocabulary_size)
        # if vocabulary:
        #     self.vocabulary = vocabulary
        if vocabulary_size == -1:  # pass -1 to load ascii letters
            self.vocabulary = string.ascii_letters
            # self.vocabulary = list(string.ascii_lowercase)
            self.vocabulary += ' '
            self.vocabulary += '-'
        self.voc_size = len(self.vocabulary)
        self.vocNum = dict(zip(range(len(self.vocabulary)), self.vocabulary))
        self.vocLet = dict(zip(self.vocabulary, range(len(self.vocabulary))))
        self.datafile = 'data/text_data_vocsize_' + str(self.voc_size) + '_tot_number_' + str(total_number)+'_sp_'+str(sparsity)+'.p'
        if not os.path.exists('data/'):
            os.makedirs('data/')
        try:
            with open(self.datafile) as file:
                data = pickle.load(file)
                self.L1_len = data['L1_len']
                self.L1_pat = data['L1_pat']
                self.L1 = data['L1']
                # print self.L1
        except:
            self.L1_len = total_number
            self.L1_pat = int(total_number*sparsity)   # features per latter
            self.L1 = np.zeros((len(self.vocabulary), self.L1_pat), dtype=int)      # shows active cells for every letter in vocabulary
            for i in range(len(self.vocabulary)):
                self.L1[i] = np.random.choice(self.L1_len, self.L1_pat, replace=False)
            with open(self.datafile, 'wb') as file:
                pickle.dump({'L1_len': self.L1_len, 'L1_pat': self.L1_pat, 'L1': self.L1}, file)
            print 'new text data'
        self.patterns = self.get_all()

    def sense(self, data=''):
        return self.L1[self.vocLet[data]]

    def get_all(self):
        # return matrix of all responses to all vocabulary
        data = np.zeros((len(self.vocabulary), self.L1_len), dtype='uint8')
        for i, w in enumerate(self.vocabulary):
            data[i, :] = self.sense_full(w)
        return data

    # @staticmethod
    def get_vocabulary(self, size):
        import pandas as pd
        file = 'common_words5000.csv'
        try:
            data = pd.read_csv(file)
            if size < 2500:
                return data[data['Part of speech'] == 'n']['Word'][:size].tolist()
            else:
                return data['Word'][:size].tolist()
        except:
            print 'Cannot load file'
            return ''

    def sense_full(self, data=' '):
        if data == -1:
            return np.zeros(self.L1_len, dtype='uint8')
        a = np.zeros(self.L1_len, dtype='uint8')
        if type(data) == str:
            a[self.L1[self.vocLet[data]]] = 1
        if type(data) == int:
            a[self.L1[data]] = 1
        return a

class NumberSense:
    def __init__(self, total_number=20, sparsity=0.1, number_size=180):
        self.numbers_size = number_size
        self.N = total_number
        self.n = int(total_number*sparsity) # number of active cells
        self.datafile = 'data/number_data_num_size_'+str(number_size) + '_' +str(total_number)+'_'+str(sparsity)+'.p'
        if not os.path.exists('data/'):
            os.makedirs('data/')
        try:
            with open(self.datafile) as file:
                data = pickle.load(file)
                self.pattern = data['pattern']
                self.n = data['n']
                self.N = data['N']
        except:
            self.pattern = np.zeros((self.numbers_size, self.n), dtype=int)      # shows active cells for every letter in vocabulary
            for i in range(self.numbers_size):
                self.pattern[i] = np.random.choice(self.N, self.n, replace=False) #randomly store number of cell that should be active
            with open(self.datafile, 'wb') as file:
                pickle.dump({'N': self.N, 'n': self.n, 'pattern': self.pattern}, file)
            print 'new number data'
        self.patterns = self.get_all()
        #  interesting another form to make binary vector.
        #  however you can set only probability of number of ones, not exact number
        # self.patterns[i] = np.random.choice([0, 1], size=(self.N,), p=[1-sparsity, sparsity])

    def sense(self, data=0, noise=0.):
        bin_patterns = np.zeros(self.N, dtype='uint8')
        bin_patterns[self.pattern[data]] = 1
        if data == -1:
            return np.zeros(self.N, dtype='uint8')
        if noise > 0:
            patt = np.copy(bin_patterns)
            a = np.nonzero(patt)[0]
            patt[np.random.choice(a, int(len(a)*noise))] = 0   # get several ones in array and make it zeros
            patt[np.random.randint(0,self.N, size=int(len(a)*noise))] = 1   # put back ones at different locations
            return patt
        return bin_patterns

    def get_all(self):
        # return matrix of all responses to all vocabulary
        data = np.zeros((self.numbers_size, self.N), dtype='uint8')
        for i in xrange(self.numbers_size):
            data[i, :] = self.sense(i)
        return data

    def get_part(self, part):
        # return matrix of part responses to all vocabulary
        data = np.zeros((part, self.N), dtype='uint8')
        for i in xrange(part):
            data[i, :] = self.sense(i)
        return data

class Schedule:
    def __init__(self):
        self.schedule = [[-2, -2]]*2000

    def reset_schedule(self):
        self.schedule = [[-2, -2]]*2000

    def set_schedule(self, (color, label), t_start, t_end):
        for i in range(t_start, t_end):
            self.schedule[i] = [color, label]

    def make_schedule(self, labels, num_patterns, step=5):
        data = labels[:num_patterns]
        self.reset_schedule()
        for k, j in enumerate(data):
            self.set_schedule((k, j), k*step, (k+1)*step)  # five step present pattern
        ## recall
        l = len(data)
        for k in range(l):
            # if for label there is no response than ativity in color through association area should define state
            self.set_schedule((k, -2), (k+l)*step, int((k+1+l)*step))  # five steps present only color, and simultaneously look response


if __name__ == '__main__':


    c = ColorSense(total_number=100, sparsity=0.1)
    for i in range(10):
        we = c.sense(i)


    print c.sense(1)
    print c.sense(1, noise=0.1)
    print c.sense(1, noise=0.2) == c.sense(1)
        # print we
        # print len(we)
    # print c.pattern[4]
    # print c.sense(4)
    # print c.sense(4, noise=0.2)
    # we2 = c.sense(10)
    # print we, np.count_nonzero(we)
    # print we2, np.count_nonzero(we2)
    # print np.count_nonzero(we == we2)
    #
    # # print np.nonzero(c.sense(2))
    # # for i in range(40):
    # #     print len(np.nonzero(c.sense(i))[0])
    #
    # sch = Schedule()
    # sch.set_schedule((10, 'r'), 10,15)
    # a, b = sch.schedule[14]
    # print a, b

    tot_num = 200
    sp = 0.05
    n = NumberSense(total_number=tot_num, sparsity=sp, number_size=1000)

    def count_overlap():
        print 'started'
        overlap = np.array([[np.count_nonzero(n.patterns[j] * n.patterns[i]) for i in range(n.numbers_size)] for j in range(n.numbers_size)], dtype=int)
        overlap -= np.eye(n.numbers_size, dtype=int)*int(sp*tot_num)
        print 'ended'
        print 'Max overlap',
        print int(sp*tot_num)
        for i in range(15):
            num_overlap = np.count_nonzero(overlap == i)/2   # count how many patterns has overlap with i+1 neurons. Divide by 2 = symetric matrix
            print np.mean(np.sum(overlap == i, axis=1))

            print i, num_overlap  # it means that num_overlap representation pairs has >= i+1  neurons in common to represent pattern
            # if num_overlap == 0:
            #     break
        # import matplotlib.pyplot as plt
        # plt.imshow(overlap >= 6)
        # plt.show()
    # print t.vocabulary
    # count_overlap()
