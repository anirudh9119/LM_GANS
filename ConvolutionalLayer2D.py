import sys

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn
from theano.sandbox.cuda.basic_ops import gpu_contiguous

import warnings
warnings.filterwarnings("ignore")

rng = np.random.RandomState(23455)
# set a fixed number for 2 purpose:
#  1. repeatable experiments; 2. for multiple-GPU, the same initial weights


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


class ConvPoolLayer2D(object):

    def __init__(self, in_channels, out_channels, kernel_len, stride = 1, activation = "relu", batch_norm = False):

        self.convstride = stride
        if kernel_len == 1:
            self.padsize = 0
        elif kernel_len == 3:
            self.padsize = 1
        elif kernel_len == 5:
            self.padsize = 2
        elif kernel_len == 7:
            self.padsize = 3
        elif kernel_len == 11:
            self.padsize = 5
        else:
            raise Exception()
        self.batch_norm = batch_norm
        bias_init = 0.0
        self.activation = activation

        std = 0.02

        self.filter_shape = np.asarray((in_channels, kernel_len, kernel_len, out_channels))

        #If this layer doesn't change the shape of the input, add a residual-layer style skip connection.  
        if in_channels == out_channels and self.convstride == 1:
            self.residual = True
        else:
            self.residual = False

        self.W = Weight(self.filter_shape, std = std)
        self.b = Weight(self.filter_shape[3], bias_init, std=0)

        if batch_norm:
            self.bn_mean = theano.shared(np.zeros(shape = (1,out_channels,1,1)).astype('float32'))
            self.bn_std = theano.shared(np.random.normal(1.0, 0.001, size = (1,out_channels,1,1)).astype('float32'))


    def output(self, input):


        W_shuffled = self.W.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01

        conv_out = dnn.dnn_conv(img=input,
                                        kerns=W_shuffled,
                                        subsample=(self.convstride, self.convstride),
                                        border_mode=self.padsize)

        conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')

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

    x = T.tensor4()

    randData = np.random.normal(size = (1,256,256,3)).astype('float32')

    c1 = ConvPoolLayer(input=x.dimshuffle(0,3,1,2), in_channels = 3, out_channels = 96, kernel_len = 5, in_rows = 256, in_columns = 256, batch_size = 100,
                                        convstride=2, padsize=2, 
                                        poolsize=1, poolstride=0,
                                        bias_init=0.1, name = "h2", paramMap = None
                                        )

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


    f = theano.function(inputs = [x], outputs = {'y' : y, 'c1' : c1.output.transpose(0,2,3,1), 'c2' : c2.output.transpose(0,2,3,1), 'c3' : c3.output.transpose(0,2,3,1), 'c4' : c4.output.transpose(0,2,3,1), 'c5' : c5.output.transpose(0,2,3,1)})

    #print f(randData)['g']
    out = f(randData)


    print (randData**2).sum()
    print (out['c1']**2).sum()
    print (out['c2']**2).sum()
    print (out['c3']**2).sum()
    print (out['c4']**2).sum()
    print (out['c5']**2).sum()

    for element in sorted(out.keys()):
        print element, out[element].shape

