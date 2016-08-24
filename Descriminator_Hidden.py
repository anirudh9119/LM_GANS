import theano
import theano.tensor as T
import numpy as np
#import random
from lasagne.layers import DenseLayer #LSTMLayer
import lasagne
#import time

#from data_iterator import TextIterator
#from pad_list import pad_list
#import cPickle as pkl
#from to_one_hot import to_one_hot
#from theano.ifelse import ifelse

from layers import param_init_gru, gru_layer
from utils import init_tparams
from ConvolutionalLayer2D import ConvPoolLayer2D

from HiddenLayer import HiddenLayer

import random as rng
srng = theano.tensor.shared_randomstreams.RandomStreams(420)

bce = T.nnet.binary_crossentropy


'''
-Build a discriminator.
-Each time we train, use 1 for "real" and 0 for "sample".
-In later uses, we'll need to define a different transformation for sampling from the generator-RNN which is differentiable.
-Takes input matrix of integers.
-For each time step, index into word matrix using saved indices.
'''

def dropout(in_layer, p = 0.5):
    return in_layer * T.cast(srng.binomial(n=1,p=p,size=in_layer.shape),'float32')

class discriminator:

    '''
    target and features will contain BOTH real and generated.

    Input hidden state features are:
     sequence x example x feature

    '''
    #2048, 1024 * 3 + 620, 30, 64,   , 64
    def __init__(self, num_hidden, num_features, seq_length, mb_size, hidden_state_features, target):
        self.mb_size = mb_size
        self.seq_length = seq_length

        #(1920, 3692), (2048, 4096)
        gru_params_1 = init_tparams(param_init_gru(None, {}, prefix = "gru1", dim = num_hidden, nin = num_features))
        gru_params_2 = init_tparams(param_init_gru(None, {}, prefix = "gru2", dim = num_hidden, nin = num_hidden))
        gru_params_3 = init_tparams(param_init_gru(None, {}, prefix = "gru3", dim = num_hidden, nin = num_hidden))

        gru_1_out = gru_layer(gru_params_1, hidden_state_features, None, prefix = 'gru1')[0]
        gru_2_out = gru_layer(gru_params_2, T.concatenate([gru_1_out], axis = 2), None, prefix = 'gru2', backwards = True)[0]
        gru_3_out = gru_layer(gru_params_3, T.concatenate([gru_2_out], axis = 2), None, prefix = 'gru3')[0]

        final_out_recc = T.mean(gru_3_out, axis = 0)

        '''
        h_out_1 = DenseLayer((mb_size, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_2 = DenseLayer((mb_size, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_3 = DenseLayer((mb_size, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_4 = DenseLayer((mb_size, num_hidden), num_units = 1, nonlinearity=None)
        # 3692, 1844,920, 458
        # (30, 64, 3692)
        # 1644, 820, 408, 202
        c1 = ConvPoolLayer(in_channels = 64, out_channels = 64,
                           batch_size = 30, in_length = 1644,
                           kernel_len = 6, stride = 2)

        # (30, 96, 1022)
        c2 = ConvPoolLayer(in_channels = 64, out_channels = 64,
                           batch_size = 30, in_length = 820,
                           kernel_len = 6, stride = 2)

        # (30, 96, 509)
        c3 = ConvPoolLayer(in_channels = 64, out_channels = 64,
                           batch_size = 30, in_length = 408,
                           kernel_len = 6, stride = 2)
        #(30, 96, 458)

        c1o = c1.output(hidden_state_features)
        c2o = c2.output(c1o)
        c3o = c3.output(c2o)

        final_out_conv = T.mean(c3o, axis = 0)
        '''

        '''
        h_out_1 = DenseLayer((64, 458), num_units = 458, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_2 = DenseLayer((64, 458), num_units = 458, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_3 = DenseLayer((64, 458), num_units = 458, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_4 = DenseLayer((64, 458), num_units = 1, nonlinearity=None)
        # h_out_1_value = h_out_1.get_output_for(T.concatenate([final_out_recc, final_out_conv], axis = 1))
        '''

        #Shape is 784 x 64 x 2048

        #batch_size x filters x pos1 x pos2

        #c1_in = hidden_state_features.transpose(1,2,0).reshape((64, 2048, 28, 28))[:,:10,:,:]

        #c1 = ConvPoolLayer2D(in_channels = 10, out_channels = 10, kernel_len = 3, stride = 2)

        #c1_out = T.mean(c1.output(c1_in), axis = (2,3))

        #batch_size x filters

        #Goes to 14 x 14 x 64 x 1024

        #Mean pool to 64 x 1024.  Concatenate with h_out_1 input.  

        h_out_1 = DenseLayer((64, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_2 = DenseLayer((64, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_3 = DenseLayer((64, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        
        #h_out_1 = HiddenLayer(num_in = 2048, num_out = 2048, activation = 'relu', batch_norm = False)
        #h_out_2 = HiddenLayer(num_in = 2048, num_out = 2048, activation = 'relu', batch_norm = False)
        #h_out_3 = HiddenLayer(num_in = 2048, num_out = 2048, activation = 'relu', batch_norm = False)

        h_out_4 = DenseLayer((64, num_hidden), num_units = 1, nonlinearity=None)

        h_out_1_value = h_out_1.get_output_for(T.concatenate([final_out_recc], axis = 1))
        h_out_2_value = h_out_2.get_output_for(h_out_1_value)
        h_out_3_value = h_out_3.get_output_for(h_out_2_value)
        h_out_4_value = h_out_4.get_output_for(h_out_3_value)

        raw_y = T.clip(h_out_4_value, -10.0, 10.0)

        classification = T.nnet.sigmoid(raw_y)

        self.get_matrix = theano.function(inputs=[hidden_state_features],
                                          outputs=[classification])

        self.loss = -1.0 * (target * -1.0 * T.log(1 + T.exp(-1.0*raw_y.flatten())) +
                                                 (1 - target) * (-raw_y.flatten() - T.log(1 + T.exp(-raw_y.flatten()))))
        p_real =  classification[0:mb_size]
        p_gen  = classification[mb_size:mb_size*2]

        self.d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
        self.d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
        self.g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()

        self.d_cost = self.d_cost_real + self.d_cost_gen
        self.g_cost = self.g_cost_d

        self.classification = classification

        self.params = []
        self.params += lasagne.layers.get_all_params(h_out_4,trainable=True)
        self.params += lasagne.layers.get_all_params(h_out_3,trainable=True)
        self.params += lasagne.layers.get_all_params(h_out_2,trainable=True)
        self.params += lasagne.layers.get_all_params(h_out_1,trainable=True)

        #self.params += c1.getParams().values()

        self.params += gru_params_1.values()
        self.params += gru_params_2.values()
        self.params += gru_params_3.values()


        self.accuracy = T.mean(T.eq(target, T.gt(classification, 0.5).flatten()))



