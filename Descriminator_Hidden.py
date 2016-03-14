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

from layers import param_init_gru, gru_layer
from utils import init_tparams

from ConvolutionalLayer import ConvPoolLayer

'''
-Build a discriminator.
-Each time we train, use 1 for "real" and 0 for "sample".
-In later uses, we'll need to define a different transformation for sampling from the generator-RNN which is differentiable.
-Takes input matrix of integers.
-For each time step, index into word matrix using saved indices.
'''


class discriminator:

    '''
    target and features will contain BOTH real and generated.  

    Input hidden state features are:
     minibatch index x sequence x feature

    '''

    def __init__(self, num_hidden, num_features, seq_length, mb_size, hidden_state_features, target):
        self.mb_size = mb_size
        self.seq_length = seq_length

        #when training generator, use hidden state features

        features_seq_first = hidden_state_features#.transpose(1,0,2)
        #from example,seq,feature ==> seq,example,feature

        features = features_seq_first

        #features: example x sequence_position x feature

        gru_params_1 = init_tparams(param_init_gru(None, {}, prefix = "gru1", dim = num_hidden, nin = num_features))
        gru_params_2 = init_tparams(param_init_gru(None, {}, prefix = "gru2", dim = num_hidden, nin = num_hidden + num_features))
        gru_params_3 = init_tparams(param_init_gru(None, {}, prefix = "gru3", dim = num_hidden, nin = num_hidden + num_features))

        gru_1_out = T.maximum(0.0, gru_layer(gru_params_1,features,None, prefix = 'gru1')[0])
        gru_2_out = T.maximum(0.0, gru_layer(gru_params_2,T.concatenate([gru_1_out, features], axis = 2),None, prefix = 'gru2', backwards = True)[0])
        gru_3_out = T.maximum(0.0, gru_layer(gru_params_3,T.concatenate([gru_2_out, features], axis = 2), None, prefix = 'gru3')[0])

        final_out = T.mean(gru_3_out, axis = 0)

        h_out_1 = DenseLayer((mb_size, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_2 = DenseLayer((mb_size, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_3 = DenseLayer((mb_size, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_4 = DenseLayer((mb_size, num_hidden), num_units = 1, nonlinearity=None)

        h_out_1_value = h_out_1.get_output_for(final_out)
        h_out_2_value = h_out_2.get_output_for(h_out_1_value)
        h_out_3_value = h_out_3.get_output_for(h_out_2_value)
        h_out_4_value = h_out_4.get_output_for(h_out_3_value)
        raw_y = h_out_4_value
        classification = T.nnet.sigmoid(raw_y)

        self.loss = -1.0 * (target * -1.0 * T.log(1 + T.exp(-1.0*raw_y.flatten())) + (1 - target) * (-raw_y.flatten() - T.log(1 + T.exp(-raw_y.flatten()))))

        self.classification = classification

        self.params = []
        self.params += lasagne.layers.get_all_params(h_out_4,trainable=True)
        self.params += lasagne.layers.get_all_params(h_out_3,trainable=True)
        self.params += lasagne.layers.get_all_params(h_out_2,trainable=True)
        self.params += lasagne.layers.get_all_params(h_out_1,trainable=True)

        self.params += gru_params_1.values()
        self.params += gru_params_2.values()
        self.params += gru_params_3.values()


        #all_grads = T.grad(self.loss, self.params)

        #for j in range(0, len(all_grads)):
        #    all_grads[j] = T.switch(T.isnan(all_grads[j]), T.zeros_like(all_grads[j]), all_grads[j])

        #self.updates = lasagne.updates.adam(all_grads, self.params, learning_rate = 0.0001, beta1 = 0.5)

        self.accuracy = T.mean(T.eq(target, T.gt(classification, 0.5).flatten()))

        #self.train_func = theano.function(inputs = [x, target, use_one_hot_input_flag, one_hot_input, hidden_state_features_discriminator, self.hidden_state_features_generator], outputs = {'l' : self.loss, 'c' : classification, 'accuracy' : T.mean(T.eq(target, T.gt(classification, 0.5).flatten()))}, updates = updates)


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

    one_hot_input = T.ftensor3()

    hidden_state_features_discriminator = T.ftensor3()

    d = discriminator(number_words = 30000, num_hidden = 1024, seq_length = seq_length, mb_size = 64, one_hot_input = one_hot_input, hidden_state_features_discriminator = hidden_state_features_discriminator)

    print "training started"

    t0 = time.time()

    for i in range(0,200000):
        u = random.uniform(0,1)
        #should use 1250
        indexGen = random.randint(15, 1250)
        indexOrig = random.randint(15, 1500)
        #if u < 0.5:
        #    d.train_real_indices(orig_s[indexOrig * 64 : (indexOrig + 1) * 64].astype('int32'))
        #else:
        #    d.train_fake_indices(gen_s[indexGen * 64 : (indexGen + 1) * 64].astype('int32'))

        res = d.train_real_fake_indices(orig_s[indexOrig * 32 : (indexOrig + 1) * 32].astype('int32'), gen_s[indexOrig * 32 : (indexOrig + 1) * 32].astype('int32'))


        if i % 200 == 1:

            print "time", time.time() - t0
            t0 = time.time()

            print "Epoch", i

            print "============Train=============="

            print "acc", 0.5 * ((d.evaluate_indices(orig_s[2000:2064].astype('int32'))['c'] > 0.5).sum() / 64.0 + (d.evaluate_indices(gen_s[2000:2064].astype('int32'))['c'] < 0.5).sum() / 64.0)

            print "===========Validation============="
            realSum = 0.0
            fakeSum = 0.0

            for index in range(0,9):

                realSum += (d.evaluate_indices(orig_s[index * 64:(index + 1)*64].astype('int32'))['c'] > 0.5).sum()
                fakeSum += (d.evaluate_indices(gen_s[index*64:(index + 1)*64].astype('int32'))['c'] < 0.5).sum()

            print "acc", 0.5 * (realSum / (64.0 * 9.0) + fakeSum / (64.0 * 9.0))

