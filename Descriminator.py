import theano
import theano.tensor as T
import numpy as np
import random
from lasagne.layers import DenseLayer, LSTMLayer
from data_iterator import TextIterator
from pad_list import pad_list
import lasagne
import numpy
import cPickle as pkl

'''
-Build a discriminator.
-Each time we train, use 1 for "real" and 0 for "sample".
-In later uses, we'll need to define a different transformation for sampling from the generator-RNN which is differentiable.
-Takes input matrix of integers.
-For each time step, index into word matrix using saved indices.
'''


class discriminator:

    def __init__(self, number_words, num_hidden, seq_length, mb_size):
        self.mb_size = mb_size
        x = T.imatrix()
        target = T.ivector()
        word_embeddings = theano.shared(np.random.normal(size = ((number_words, 1, num_hidden))).astype('float32'))
        feature_lst = []
        for i in range(0, seq_length):
            feature = word_embeddings[x[:,i]]
            feature_lst.append(feature)
        features = T.concatenate(feature_lst, 1)

        #example x sequence_position x feature
        #inp = InputLayer(shape = (seq_length, mb_size, num_hidden), input_var = features)
        l_lstm_1 = LSTMLayer((seq_length, mb_size, num_hidden), num_units = num_hidden, nonlinearity = lasagne.nonlinearities.tanh)
        l_lstm_2 = LSTMLayer((seq_length, mb_size, num_hidden), num_units = num_hidden, nonlinearity = lasagne.nonlinearities.tanh)

        #minibatch x sequence x feature
        final_out = T.mean(l_lstm_2.get_output_for([l_lstm_1.get_output_for([features])]), axis = 1)

        #final_out = T.mean(features, axis = 1)
        h_out = DenseLayer((mb_size, num_hidden), num_units = 1, nonlinearity=None)
        h_out_value = h_out.get_output_for(final_out)
        classification = T.nnet.sigmoid(h_out_value)
        self.loss = T.mean(T.nnet.binary_crossentropy(output = classification.flatten(), target = target))
        self.params = lasagne.layers.get_all_params(h_out,trainable=True) + [word_embeddings] + lasagne.layers.get_all_params(l_lstm_1, trainable = True) + lasagne.layers.get_all_params(l_lstm_2, trainable = True)
        updates = lasagne.updates.adam(self.loss, self.params)
        self.train_func = theano.function(inputs = [x, target], outputs = {'l' : self.loss, 'c' : classification}, updates = updates)
        self.evaluate_func = theano.function(inputs = [x], outputs = {'c' : classification})

    def train_real(self, x):
        return self.train_func(x, [1] * self.mb_size)

    def train_fake(self, x):
        return self.train_func(x, [0] * self.mb_size)

    def evaluate(self, x):
        return self.evaluate_func(x)


if __name__ == "__main__":

    seq_length = 30
    dictionary='/data/lisatmp4/anirudhg/wiki.tok.txt.gz.pkl'
    valid_dataset='/data/lisatmp3/chokyun/wikipedia/extracted/wiki.tok.txt.gz'

    gen_dataset = 'generated_text.txt'
    batch_size  = 1
    n_words = 30000
    maxlen = 30

    # load dictionary
    with open(dictionary, 'rb') as f:
        worddicts = pkl.load(f)

    # invert dictionary
    worddicts_r = dict()
    for kk, vv in worddicts.iteritems():
        worddicts_r[vv] = kk

    print 'Loading data'
    actual_sentences = TextIterator(valid_dataset,
                                    dictionary,
                                    n_words_source = n_words,
                                    batch_size = batch_size,
                                    maxlen=maxlen)

    gen_sentences = TextIterator(gen_dataset,
                                 dictionary,
                                 n_words_source = n_words,
                                 batch_size = batch_size,
                                 maxlen=maxlen)

    orig_sen = []
    count = 0
    for x in actual_sentences:
        if len(x[0]) > 5 and count <=100000:
            count = count +1;
            orig_sen.append(x[0])


    orig_sen.append(numpy.zeros(30).tolist())

    print 'Done 1'
    orig_s = pad_list(orig_sen)
    print orig_s.shape

    gen_sen =  []
    for x in gen_sentences:
        if len(x[0]) > 5:
            gen_sen.append(x[0])

    print 'Done 2'
    gen_sen.append(numpy.zeros(30).tolist())

    print len(gen_sen)
    print len(orig_sen)

    gen_s = pad_list(gen_sen)
    print gen_s.shape

    gen_s = gen_s[0:gen_s.shape[0] - 1]
    orig_s = orig_s[0:orig_s.shape[0] - 1]

    print 'saving'

    np.savez('orig_s.npz', orig_s)
    np.savez('gen_s.npz', gen_s)

    print 'saved'

    #5000 gen
    #2144 orig

    orig_s = np.load('orig_s.npz')
    gen_s = np.load('gen_s.npz')

    orig_s = orig_s['arr_0']
    gen_s = gen_s['arr_0']

    print orig_s

    print "compiling"

    d = discriminator(number_words = 30000, num_hidden = 400, seq_length = seq_length, mb_size = 64)

    print "training started"

    for i in range(0,40):
        u = random.uniform(0,1)
        indexGen = random.randint(0, 200 / 64)
        indexOrig = random.randint(0, 200 / 64)
        if u < 0.5:
            d.train_real(orig_s[0 : 64].astype('int32'))
        else:
            d.train_fake(gen_s[0 : 64].astype('int32'))

    print "real", d.evaluate(orig_s[0:64].astype('int32'))['c'].tolist()
    print "fake", d.evaluate(gen_s[0:64].astype('int32'))['c'].tolist()

