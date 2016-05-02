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
from ConvolutionalLayer import ConvPoolLayer

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
        gru_params_2 = init_tparams(param_init_gru(None, {}, prefix = "gru2", dim = num_hidden, nin = num_hidden + num_features))
        gru_params_3 = init_tparams(param_init_gru(None, {}, prefix = "gru3", dim = num_hidden, nin = num_hidden + num_features))

        gru_1_out = gru_layer(gru_params_1, hidden_state_features, None, prefix = 'gru1')[0]
        gru_2_out = gru_layer(gru_params_2, T.concatenate([gru_1_out, hidden_state_features], axis = 2), None, prefix = 'gru2', backwards = True)[0]
        gru_3_out = gru_layer(gru_params_3, T.concatenate([gru_2_out, hidden_state_features], axis = 2), None, prefix = 'gru3')[0]

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
        h_out_1 = DenseLayer((64, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_2 = DenseLayer((64, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_3 = DenseLayer((64, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        
        #h_out_1 = HiddenLayer(num_in = 2048, num_out = 2048, activation = 'relu', batch_norm = False)
        #h_out_2 = HiddenLayer(num_in = 2048, num_out = 2048, activation = 'relu', batch_norm = False)
        #h_out_3 = HiddenLayer(num_in = 2048, num_out = 2048, activation = 'relu', batch_norm = False)

        h_out_4 = DenseLayer((64, num_hidden), num_units = 1, nonlinearity=None)

        h_out_1_value = dropout(h_out_1.get_output_for(final_out_recc))
        h_out_2_value = dropout(h_out_2.get_output_for(h_out_1_value))
        h_out_3_value = dropout(h_out_3.get_output_for(h_out_2_value))
        h_out_4_value = h_out_4.get_output_for(h_out_3_value)

        raw_y = h_out_4_value

        classification = T.nnet.sigmoid(raw_y)

        self.get_matrix = theano.function(inputs=[hidden_state_features],
                                          outputs=[classification])

        self.loss = -1.0 * (target * -1.0 * T.log(1 + T.exp(-1.0*raw_y.flatten())) +
                                                 (1 - target) * (-raw_y.flatten() - T.log(1 + T.exp(-raw_y.flatten()))))
        p_real =  classification[0:32]
        p_gen  = classification[32:64]

        self.d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
        self.d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
        self.g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()
        self.d_cost = self.d_cost_real + self.d_cost_gen
        self.g_cost = self.g_cost_d




        '''
        gX = gen(Z, *gen_params)

        p_real = discrim(X, *discrim_params)
        p_gen = discrim(gX, *discrim_params)

        d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
        d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
        g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()

        d_cost = d_cost_real + d_cost_gen
        g_cost = g_cost_d

        cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]
        d_updates = d_updater(discrim_params, d_cost)
        g_updates = g_updater(gen_params, g_cost)

        '''



        self.classification = classification

        self.params = []
        self.params += lasagne.layers.get_all_params(h_out_4,trainable=True)
        self.params += lasagne.layers.get_all_params(h_out_3,trainable=True)
        self.params += lasagne.layers.get_all_params(h_out_2,trainable=True)
        self.params += lasagne.layers.get_all_params(h_out_1,trainable=True)

        #self.params += h_out_1.getParams() + h_out_2.getParams() + h_out_3.getParams()

#        self.params += lasagne.layers.get_all_params(h_initial_1,trainable=True)
#        self.params += lasagne.layers.get_all_params(h_initial_2,trainable=True)

        self.params += gru_params_1.values()
        self.params += gru_params_2.values()
        self.params += gru_params_3.values()

        '''
        layerParams = c1.getParams()
        for paramKey in layerParams:
            self.params += [layerParams[paramKey]]
        layerParams = c2.getParams()
        for paramKey in layerParams:
            self.params += [layerParams[paramKey]]
        layerParams = c3.getParams()
        for paramKey in layerParams:
            self.params += [layerParams[paramKey]]

        '''

        #all_grads = T.grad(self.loss, self.params)
        #for j in range(0, len(all_grads)):
        #    all_grads[j] = T.switch(T.isnan(all_grads[j]), T.zeros_like(all_grads[j]), all_grads[j])
        #self.updates = lasagne.updates.adam(all_grads, self.params, learning_rate = 0.0001, beta1 = 0.5)

        self.accuracy = T.mean(T.eq(target, T.gt(classification, 0.5).flatten()))

        '''
        self.train_func = theano.function(inputs = [x, target, use_one_hot_input_flag,
                                                     one_hot_input, hidden_state_features_discriminator,
                                                     self.hidden_state_features_generator],
                                           outputs = {'l' : self.loss,
                                                      'c' : classification,
                                                      'accuracy' : T.mean(T.eq(target, T.gt(classification, 0.5).flatten()))},
                                           updates = updates)
        '''

