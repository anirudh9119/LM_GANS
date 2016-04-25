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

bce = T.nnet.binary_crossentropy


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
        h_out_1 = DenseLayer((64, 2048), num_units = 2048, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_2 = DenseLayer((64, 2048), num_units = 2048, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_3 = DenseLayer((64, 2048), num_units = 2048, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_4 = DenseLayer((64, 2048), num_units = 1, nonlinearity=None)


        h_out_1_value = h_out_1.get_output_for(final_out_recc)
        h_out_2_value = h_out_2.get_output_for(h_out_1_value)
        h_out_3_value = h_out_3.get_output_for(h_out_2_value)
        h_out_4_value = h_out_4.get_output_for(h_out_3_value)


        raw_y = T.clip(h_out_4_value, -10.0, 10.0)

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

    '''
        Provide a one-hot 3-tensor to specify the inputs.
        Usually to train the discriminator, we want to pass in indices.
        When we use it with the generator, we want to give one hots.
    '''

if __name__ == "__main__":


#   def __init__(self, in_channels, out_channels, in_length, batch_size, kernel_len, stride = 1, activation = "relu", batch_norm = False, unflatten_input = None):
#   randData = np.random.normal(size = (1,3,64)).astype('float32')

    x = T.tensor3()
    randData = np.random.normal(size = (30,64,2048)).astype('float32')

    # (30, 64, 2048)
    c1 = ConvPoolLayer(in_channels = 64, out_channels = 96,
                       batch_size = 30, in_length = 2048,
                       kernel_len = 6, stride = 2)

    # (30, 96, 1022)
    c2 = ConvPoolLayer(in_channels = 96, out_channels = 96,
                       batch_size = 30, in_length = 1022,
                       kernel_len = 6, stride = 2)

    # (30, 96, 509)
    c3 = ConvPoolLayer(in_channels = 96, out_channels = 96,
                       batch_size = 30, in_length = 509,
                       kernel_len = 6, stride = 2)
    c1o = c1.output(x)
    c2o = c2.output(c1o)
    c3o = c3.output(c2o)
    final_out = T.mean(c3o, axis = 0)

    f = theano.function(inputs = [x], outputs = {'c1' : c3o, 'c2':final_out})
    out = f(randData)
    print out['c1'].shape
    print out['c2'].shape

    '''
    c1 = ConvPoolLayer(input=x.dimshuffle(0,3,1,2),
                       in_channels = 3,
                       out_channels = 96,
                       kernel_len = 5,
                       in_rows = 256,
                       in_columns = 256,
                       batch_size = 100,
                       convstride=2,
                       padsize=2,
                       poolsize=1,
                       poolstride=0,
                       bias_init=0.1, name = "h2", paramMap = None)


    c2 = ConvPoolLayer(input=c1.output, in_channels = 96, out_channels = 256, kernel_len = 5, in_rows = 17, in_columns = 17, batch_size = 100,
                                        convstride=2, padsize=2,
                                        poolsize=1, poolstride=0,
                                        bias_init=0.0, name = "c2", paramMap = None
                                        )


    c3 = ConvPoolLayer(input=c2.output, in_channels = 256, out_channels = 384, kernel_len = 5, in_rows = 33, in_columns = 33, batch_size = 100,
                                        convstride=2, padsize=2,
                                        poolsize=1, poolstride=0,
                                        bias_init=0.0, name = "h3", paramMap = None
                                        )

    c4 = ConvPoolLayer(input=c3.output, in_channels = 384, out_channels = 384, kernel_len = 5, in_rows = 15, in_columns = 15, batch_size = 100,
                                        convstride=2, padsize=2,
                                        poolsize=1, poolstride=0,
                                        bias_init=0.1, name = "h3", paramMap = None
                                        )

    c5 = ConvPoolLayer(input=c4.output, in_channels = 384, out_channels = 256, kernel_len = 5, in_rows = 6, in_columns = 6, batch_size = 100,
                                        convstride=2, padsize=2,
                                        poolsize=1, poolstride=0,
                                        bias_init=0.0, name = "h3", paramMap = None
                                        )

    y = c5.output


    f = theano.function(inputs = [x], outputs = {'y' : y,
                                                 'c1' : c1.output.transpose(0,2,3,1),
                                                 'c2' : c2.output.transpose(0,2,3,1),
                                                 'c3' : c3.output.transpose(0,2,3,1),
                                                 'c4' : c4.output.transpose(0,2,3,1),
                                                 'c5' : c5.output.transpose(0,2,3,1)}
                       )

    '''
    '''
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
    d = discriminator(number_words = 30000,
                      num_hidden = 1024,
                      seq_length = seq_length,
                      mb_size = 64,
                      one_hot_input = one_hot_input,
                      hidden_state_features_discriminator = hidden_state_features_discriminator)

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

        res = d.train_real_fake_indices(orig_s[indexOrig * 32 : (indexOrig + 1) * 32].astype('int32'),
                                        gen_s[indexOrig * 32 : (indexOrig + 1) * 32].astype('int32'))


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
    '''
