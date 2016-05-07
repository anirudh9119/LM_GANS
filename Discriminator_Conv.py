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
    def __init__(self, num_hidden, num_features, seq_length, mb_size, hidden_state_features, target):
        self.mb_size = mb_size
        self.seq_length = seq_length

        print "USING CONVOLUTIONAL DISCRIMINATOR"


        c1_in = hidden_state_features[:784].transpose(1,2,0).reshape((mb_size * 2, num_features, 28, 28))

        c1 = ConvPoolLayer2D(in_channels = num_features, out_channels = 1024, kernel_len = 5, stride = 2)
        c2 = ConvPoolLayer2D(in_channels = 1024, out_channels = 512, kernel_len = 5, stride = 2)
        c3 = ConvPoolLayer2D(in_channels = 512, out_channels = 256, kernel_len = 5, stride = 2)

        c1_out = c1.output(c1_in)
        c2_out = c2.output(c1_out)
        c3_out = c3.output(c2_out)

        h_out_4 = DenseLayer((mb_size * 2, 4096), num_units = 1, nonlinearity=None)

        h_out_4_value = h_out_4.get_output_for(T.concatenate([c3_out.flatten(2)], axis = 1))

        raw_y = h_out_4_value

        classification = T.nnet.sigmoid(T.clip(raw_y, -10.0, 10.0))

        #self.get_matrix = theano.function(inputs=[hidden_state_features],
        #                                  outputs=[classification])

        self.loss = -1.0 * (target * -1.0 * T.log(1 + T.exp(-1.0*raw_y.flatten())) +
                                                 (1 - target) * (-raw_y.flatten() - T.log(1 + T.exp(-raw_y.flatten()))))
        p_real =  classification[0:mb_size]
        p_gen  = classification[mb_size:]

        self.d_cost_real = bce(p_real, 0.99 * T.ones(p_real.shape)).mean()
        self.d_cost_gen = bce(p_gen, 0.01 + T.zeros(p_gen.shape)).mean()
        self.g_cost_d = bce(p_gen, 0.99 * T.ones(p_gen.shape)).mean()
        self.d_cost = self.d_cost_real + self.d_cost_gen
        self.g_cost = self.g_cost_d


        self.classification = classification

        self.params = []
        self.params += lasagne.layers.get_all_params(h_out_4,trainable=True)

        self.params += c1.getParams().values()
        self.params += c2.getParams().values()
        self.params += c3.getParams().values()


        self.accuracy = T.mean(T.eq(target, T.gt(classification, 0.5).flatten()))

