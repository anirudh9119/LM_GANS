import numpy as np
import theano
import theano.tensor as T

from toolz import interleave
from blocks.bricks import Rectifier, Tanh, Logistic
from conv import (BNSequence,
                  Convolutional,
                  MaxPooling)
from blocks.initialization import Constant, Uniform
from blocks.bricks import Initializable
from bricks.mlp import BNMLP
from bricks.batch_norm import MaskedBatchNorm
from blocks.bricks.base import application, lazy, Brick

from blocks.utils import shared_floatx_zeros, shared_floatx

class BNConv(Initializable):
    use_bias = False
    @lazy()
    def __init__(self, layers, activations, filter_size,
                 num_filters, pooling_size, num_channels,
                 full_layer_dim, full_layer_activations,
                 pooling_step, num_pieces=1,
                 image_size=(None, None),
                 batch_size=None,
                 tied_biases=True, features='fbank', **kwargs):
        super(BNConv, self).__init__(**kwargs)
        self.layers = layers
        self.filter_size = filter_size
        self.pooling_size = pooling_size
        self.pooling_step = pooling_step
        self.num_filters = num_filters
        self.features = features

        convs = [Convolutional(filter_size[i],
                               num_filters[i],
                               name='conv_{}'.format(i))
                 for i in range(layers)]
        pools = [MaxPooling(pooling_size[i],
                            step=pooling_step[i],
                            name='pool_{}'.format(i))
                 if np.prod(pooling_size[i]) != 1
                 else None for i in range(layers)]
        bn = [MaskedBatchNorm(input_dim=num_filters[i],
                              name='bnconv_{}'.format(i))
                 for i in range(layers)]

        activs = [activations[i] for i in range(layers)]

        convs_pools = []
        for layer in interleave([convs, pools, bn, activs]):
            if layer is None:
                continue
            else:
                convs_pools.append(layer)
        self.convseq = BNSequence(convs_pools,
                                  num_channels,
                                  image_size=image_size,
                                  batch_size=batch_size,
                                  tied_biases=tied_biases)

        full_dim_pre = self.get_conv_pool_dim()
        self.mlp = BNMLP(activations=full_layer_activations,
                         dims=[full_dim_pre] + full_layer_dim,
                         name='conv_mlp')
        self.children = [self.convseq, self.mlp]

    def get_conv_pool_dim(self):
        if self.features == 'fbank':
            input_dim = 41
        elif self.features == 'swbd':
            input_dim = 37
        else:
            input_dim = 40
        for i in xrange(self.layers):
            input_dim = np.ceil((input_dim - self.filter_size[i][0] - \
                    self.pooling_size[i][0] + 2.) /\
                         self.pooling_step[i][0])
        return input_dim.astype('int') * self.num_filters[-1]

    @application(inputs=['input_', 'input_mask'], outputs=['output'])
    def apply(self, input_, input_mask):
        conv_out = self.children[0].apply(input_, input_mask)
        shape = conv_out.shape
        re_conv_out = conv_out.reshape((shape[0],
                                        shape[1]*shape[2],
                                        shape[3]))
        return self.children[1].apply(re_conv_out.dimshuffle(0, 2, 1),
                                      input_mask)

    def get_dim(self, name):
        if name == 'output':
            return (self.full_layer_dim[-1],)
        return super(BNConv, self).get_dim(name)
