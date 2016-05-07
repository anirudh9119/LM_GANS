import numpy as np
import theano
import theano.tensor as T

from lasagne.theano_extensions.conv import conv1d_md

import warnings
warnings.filterwarnings("ignore")

rng = np.random.RandomState(23455)
# Set a fixed number for 2 purpose:
# Repeatable experiments; 2. for multiple-GPU, the same initial weights


class Weight(object):

    def __init__(self, w_shape, mean=0, std=1.0):

        super(Weight, self).__init__()

        print "conv layer using std of", std, "and mean of", mean, "with shape", w_shape

        if std != 0:

            self.np_values = np.asarray(
               1.0 * rng.normal(mean, std, w_shape), dtype=theano.config.floatX)

        else:
            self.np_values = np.cast[theano.config.floatX](
                mean * np.ones(w_shape, dtype=theano.config.floatX))

        self.val = theano.shared(value=self.np_values)


class ConvPoolLayer(object):

    def __init__(self, in_channels, out_channels,
                 in_length, batch_size, kernel_len, stride = 1,
                 activation = "relu", batch_norm = False, unflatten_input = None):

        self.stride = stride
        self.batch_norm = batch_norm
        bias_init = 0.01
        self.activation = activation
        self.unflatten_input = unflatten_input
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_length = kernel_len
        self.in_length = in_length
        self.batch_size = batch_size

        std = 0.01

        self.filter_shape = np.asarray((out_channels, in_channels, kernel_len))

        self.W = Weight(self.filter_shape, std = std)
        self.b = Weight(self.filter_shape[0], bias_init, std=0)

        if batch_norm:
            self.bn_mean = theano.shared(np.zeros(shape = (1,out_channels,1,1)).astype('float32'))
            self.bn_std = theano.shared(np.random.normal(1.0, 0.001, size = (1,out_channels,1,1)).astype('float32'))


    def output(self, input):

        if self.unflatten_input != None:
            input = T.reshape(input, self.unflatten_input)

        W_shuffled = self.W.val

        conv_out = conv1d_md(input, W_shuffled, image_shape = (self.batch_size, self.in_channels, self.in_length),
                                                filter_shape = (self.out_channels, self.in_channels, self.filter_length),
                                                subsample = (self.stride,))

        #conv_out = T.nnet.conv2d(input, W_shuffled, subsample=(1, 1), border_mode='valid')

        conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x')

        if self.batch_norm:
            conv_out = (conv_out - T.mean(conv_out, axis = (0,2,3), keepdims = True)) / (1.0 + T.std(conv_out, axis=(0,2,3), keepdims = True))
            conv_out = conv_out * T.addbroadcast(self.bn_std,0,2,3) + T.addbroadcast(self.bn_mean, 0,2,3)

        self.out_store = conv_out

        if self.activation == "relu":
            self.out = T.maximum(0.0, conv_out)
        elif self.activation == "tanh":
            self.out = T.tanh(conv_out)
        elif self.activation == None:
            self.out = conv_out

        #if self.residual:
        #    print "USING RESIDUAL"
        #    self.out += input

        self.params = {'W' : self.W.val, 'b' : self.b.val}

        if self.batch_norm:
            self.params["mu"] = self.bn_mean
            self.params["sigma"] = self.bn_std

        return self.out

    def getParams(self):
        return self.params


if __name__ == "__main__":

    x = T.tensor3()
    #30 X 64 X 2048!
    randData = np.random.normal(size = (1,3,64)).astype('float32')

    c1 = ConvPoolLayer(in_channels = 3, out_channels = 96, batch_size = 1, in_length = 64, kernel_len = 6, stride = 2)

    c1o = c1.output(x)

    f = theano.function(inputs = [x], outputs = {'c1' : c1o})

    #print f(randData)['g']
    out = f(randData)


    print (randData**2).sum()
    print out['c1'].shape



