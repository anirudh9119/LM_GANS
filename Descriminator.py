import theano
import theano.tensor as T
import numpy as np
import random
from lasagne.layers import DenseLayer, LSTMLayer

from pad_list import pad_list
import lasagne
from data_iterator import TextIterator
import cPickle as pkl
import time

from to_one_hot import to_one_hot

from theano.ifelse import ifelse



'''
-Build a discriminator.
-Each time we train, use 1 for "real" and 0 for "sample".
-In later uses, we'll need to define a different transformation for sampling from the generator-RNN which is differentiable.
-Takes input matrix of integers.
-For each time step, index into word matrix using saved indices.
'''


class discriminator:

    def __init__(self, number_words, num_hidden, seq_length, mb_size, one_hot_input):
        self.mb_size = mb_size

        x = T.imatrix('descriminator_indices')

        #sequence x minibatch x index
        #one_hot_input = T.ftensor3()
        self.one_hot_input = one_hot_input
        use_one_hot_input_flag = T.scalar()

        self.indices = x
        self.use_one_hot_input_flag = use_one_hot_input_flag
        self.one_hot_input = one_hot_input

        '''
        flag for input: one-hot or index.
        If index, compute one-hot and use that.

        If one-hot, just use one-hot input.
        '''

        #Time seq x examples x words

        target = T.ivector('target_vector')

        self.target = target

        #word_embeddings = theano.shared(np.random.normal(size = ((number_words, 1, num_hidden))).astype('float32'))

        word_embeddings = theano.shared(np.random.normal(size = ((number_words, num_hidden))).astype('float32'))

        feature_lst = []

        for i in range(0, seq_length):
            #feature = word_embeddings[x[:,i]]
            #instead of this, multiply by one-hot matrix

            one_hot = T.extra_ops.to_one_hot(x[:,i], number_words)

            #W : 30k x 1 x 400
            #one_hot: 128 x 30k
            #one_hot * W
            #128 x 1 x 400


            one_hot_use = ifelse(use_one_hot_input_flag, one_hot_input[i], T.extra_ops.to_one_hot(x[:,i], number_words))

            feature = T.reshape(T.dot(one_hot_use, word_embeddings), (1,mb_size,num_hidden)).transpose(1,0,2)

            feature_lst.append(feature)

        features = T.concatenate(feature_lst, 1)

        #example x sequence_position x feature
        l_lstm_1 = LSTMLayer((seq_length, mb_size, num_hidden), num_units = num_hidden, nonlinearity = lasagne.nonlinearities.tanh, grad_clipping=100.0)
        l_lstm_2 = LSTMLayer((seq_length, mb_size, num_hidden * 2), num_units = num_hidden, nonlinearity = lasagne.nonlinearities.tanh, grad_clipping=100.0, backwards = True)
        l_lstm_3 = LSTMLayer((seq_length, mb_size, num_hidden * 2), num_units = num_hidden, nonlinearity = lasagne.nonlinearities.tanh, grad_clipping=100.0)

        lstm_1_out = l_lstm_1.get_output_for([features])
        lstm_2_out = l_lstm_2.get_output_for([T.concatenate([lstm_1_out, features], axis = 2)])
        lstm_3_out = l_lstm_3.get_output_for([T.concatenate([lstm_2_out, features], axis = 2)])

        final_out = T.mean(lstm_3_out, axis = 1)

        #final_out = T.mean(features, axis = 1)
        h_out_1 = DenseLayer((mb_size, num_hidden), num_units = 2048, nonlinearity=lasagne.nonlinearities.rectify)

        h_out_2 = DenseLayer((mb_size, 2048), num_units = 2048, nonlinearity=lasagne.nonlinearities.rectify)

        h_out_3 = DenseLayer((mb_size, 2048), num_units = 1, nonlinearity=None)

        h_out_1_value = h_out_1.get_output_for(final_out)
        h_out_2_value = h_out_2.get_output_for(h_out_1_value)
        h_out_3_value = h_out_3.get_output_for(h_out_2_value)
        raw_y = h_out_3_value
        classification = T.nnet.sigmoid(raw_y)

        #self.loss = T.mean(T.nnet.binary_crossentropy(output = classification.flatten(), target = target))


        self.loss = T.mean(-1.0 * (target * -1.0 * T.log(1 + T.exp(-1.0*raw_y.flatten())) + (1 - target) * (-raw_y.flatten() - T.log(1 + T.exp(-raw_y.flatten())))))

        self.params = lasagne.layers.get_all_params(h_out_1,trainable=True) + lasagne.layers.get_all_params(h_out_3,trainable=True) + [word_embeddings] + lasagne.layers.get_all_params(l_lstm_1, trainable = True) + lasagne.layers.get_all_params(l_lstm_2, trainable = True)

        self.params += lasagne.layers.get_all_params(h_out_2,trainable=True)
        self.params += lasagne.layers.get_all_params(l_lstm_3,trainable=True)

        all_grads = T.grad(self.loss, self.params)

        for j in range(0, len(all_grads)):
            all_grads[j] = T.switch(T.isnan(all_grads[j]), T.zeros_like(all_grads[j]), all_grads[j])

        scaled_grads = lasagne.updates.total_norm_constraint(all_grads, 5.0)

        updates = lasagne.updates.adam(scaled_grads, self.params)
        self.train_func = theano.function(inputs = [x, target, use_one_hot_input_flag, one_hot_input], outputs = {'l' : self.loss, 'c' : classification, 'g_w' : T.sum(T.sqr(T.grad(self.loss, word_embeddings)))}, updates = updates)
        self.evaluate_func = theano.function(inputs = [x, use_one_hot_input_flag, one_hot_input], outputs = {'c' : classification})

    def train_real_indices(self, x):
        return self.train_func(x, [1] * self.mb_size, 0, np.asarray([[[1]]]).astype('float32'))

    def train_fake_indices(self, x):
        return self.train_func(x, [0] * self.mb_size, 0, np.asarray([[[1]]]).astype('float32'))

    def evaluate_indices(self, x):
        return self.evaluate_func(x, 0, np.asarray([[[1]]]).astype('float32'))

    '''
        Provide a one-hot 3-tensor to specify the inputs.
        Usually to train the discriminator, we want to pass in indices.
        When we use it with the generator, we want to give one hots.
    '''

if __name__ == "__main__":

    seq_length = 30
    dictionary='/data/lisatmp4/anirudhg/wiki.tok.txt.gz.pkl'
    valid_dataset='/data/lisatmp4/anirudhg/temp.en.tok'

    gen_dataset = 'temp_workfile'
    batch_size  = 1
    n_words = 30000
    maxlen = 30


    orig_s = np.load('/u/goyalani/LM_GANS/orig_s.npz')['arr_0']
    gen_s = np.load('/u/goyalani/LM_GANS/gen_s.npz')['arr_0']

    print "compiling"

    d = discriminator(number_words = 30000, num_hidden = 1024, seq_length = seq_length, mb_size = 64)

    print "training started"

    t0 = time.time()

    for i in range(0,200000):
        u = random.uniform(0,1)
        #should use 1250
        indexGen = random.randint(15, 1250)
        indexOrig = random.randint(15, 1500)
        if u < 0.5:
            d.train_real(orig_s[indexOrig * 64 : (indexOrig + 1) * 64].astype('int32'))
        else:
            d.train_fake(gen_s[indexGen * 64 : (indexGen + 1) * 64].astype('int32'))

        if i % 200 == 1:

            print "time", time.time() - t0
            t0 = time.time()

            print "Epoch", i

            print "============Train=============="

            print "acc", 0.5 * ((d.evaluate(orig_s[2000:2064].astype('int32'))['c'] > 0.5).sum() / 64.0 + (d.evaluate(gen_s[2000:2064].astype('int32'))['c'] < 0.5).sum() / 64.0)

            print "===========Validation============="
            realSum = 0.0
            fakeSum = 0.0

            for index in range(0,9):

                realSum += (d.evaluate(orig_s[index * 64:(index + 1)*64].astype('int32'))['c'] > 0.5).sum()
                fakeSum += (d.evaluate(gen_s[index*64:(index + 1)*64].astype('int32'))['c'] < 0.5).sum()

            print "acc", 0.5 * (realSum / (64.0 * 9.0) + fakeSum / (64.0 * 9.0))

