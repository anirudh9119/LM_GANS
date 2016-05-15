import argparse
import time
import os
import sys
import logging

import numpy as np

import theano
import theano.tensor as T

from blocks.bricks import Logistic, Tanh, Softmax, Rectifier
from blocks.graph import ComputationGraph, apply_noise
from blocks.initialization import IsotropicGaussian, Constant, Orthogonal, Uniform
from blocks.main_loop import MainLoop
from blocks.monitoring import aggregation
from blocks.model import Model
from blocks.roles import WEIGHT
from bricks.generator import GeneratorTest
from bricks.recurrent import GenLSTM

from initialization import NormalizedInitialization
from datasets.timit_hmm import TIMIT
from utils import (construct_hmm_stream,
                   get_encoding,
                   beam_search)


floatX = theano.config.floatX
logger = logging.getLogger(__name__)
logger.setLevel('INFO')


def parse_args():
    parser = argparse.ArgumentParser(description='TIMIT experiment')
    parser.add_argument('--save_path', type=str,
                        default='./',
                        help='Location for writing results')
    parser.add_argument('--input_dim', type=int,
                        default=120,
                        help='Input dimension')
    parser.add_argument('--state_dim', type=int,
                        default=250,
                        help='States dimensions')
    parser.add_argument('--label_dim', type=int,
                        default=181,
                        help='Label dimension')
    parser.add_argument('--seed', type=int,
                        default=123,
                        help='Random generator seed')
    parser.add_argument('--load_path',
                        default=argparse.SUPPRESS,
                        help='File with parameter to be loaded)')
    parser.add_argument('--pool_size', type=int,
                        default=200,
                        help='Pool size for dataset')
    parser.add_argument('--maximum_frames', type=int,
                        default=10000,
                        help='Pool size for dataset')
    parser.add_argument('--window_features', type=int,
                        default=1,
                        help='Use neighbour frames (window_features / 2'
                             'before and window_features / 2 after).'
                             'Should be odd.')
    parser.add_argument('--initialization', choices=['glorot', 'uniform'],
                        default='glorot')

    return parser.parse_args()

def eval(save_path, input_dim, state_dim, label_dim,
         seed, window_features, pool_size,
         maximum_frames, initialization, **kwargs):
    print '.. TIMIT experiment'
    print '.. arguments:', ' '.join(sys.argv)
    t0 = time.time()

    rng = np.random.RandomState(seed)
    stream_args = dict(rng=rng, pool_size=pool_size,
                       maximum_frames=maximum_frames,
                       window_features=window_features)

    print '.. initializing iterators'
    test_dataset = TIMIT('test')
    test_stream = construct_hmm_stream(test_dataset, **stream_args)
    print '.. building model'
    x = T.matrix('features')
    y = T.lvector('labels')
    input_mask = T.vector('features_mask')
    output_mask = T.vector('labels_mask')
    y0 = T.lmatrix('y0')
    init_states, init_cells = T.matrices('init_states', 'init_cells')

    if initialization == 'glorot':
        weights_init = NormalizedInitialization()
    elif initialization == 'uniform':
        weights_init = Uniform(width=.2)
    else:
        raise ValueError('No such initialization')

    gen_unidir1 = GenLSTM(name='unidir1',
                  dim=state_dim, activation=Tanh())

    gen_unidir2 = GenLSTM(name='unidir2',
                  dim=state_dim, activation=Tanh())

    generator = GeneratorTest(weights_init=weights_init,
                              biases_init=Constant(.0),
                              networks=[gen_unidir1, gen_unidir2],
                              dims=[(input_dim + label_dim) * window_features,
                                     state_dim,
                                     state_dim,
                                     label_dim])

    generator.initialize()
    y_hat_o, states_o, cells_o = generator.apply(x, input_mask, targets=y0,
                            states=init_states, cells=init_cells)

    y_hat = Softmax().apply(y_hat_o)

    model = Model(y_hat)
    parameters = model.get_parameter_dict()

    if 'load_path' not in kwargs:
        raise KeyError('params should be provided!')
    params = np.load(kwargs['load_path']).item()
    params_names = params.keys()

    for par in parameters.keys():
        dash_name = par.split('/')
        dash_name[1] = 'generator'
        dash_name_ = '/'.join(dash_name)
        parameters[par].set_value(params[dash_name_])

    init_cells_val = np.hstack((params['/generator/unidir1.initial_cells'][None, ...],
                                params['/generator/unidir2.initial_cells'][None, ...]))

    func = theano.function([x, input_mask, y0, init_states, init_cells],
                           [y_hat, states_o, cells_o])

    neq = 0
    valid_length = .0

    for data in test_stream.get_epoch_iterator():
        length = data[0].shape[0]
        batch = data[0].shape[1]
        y_val = np.zeros((batch, 181))
        y_val[:, 0] = 1
        y_val = y_val.astype('int32')
        state_out = np.zeros((batch, state_dim * 2)).astype('float32')
        cell_out = np.tile(init_cells_val, (batch, 1))

        for seq in xrange(length):
            x_val = data[0][seq]
            x_mask = data[1][seq]
            if seq == 0:
                soft_out, state_out, cell_out = func(x_val, x_mask,
                                                     y_val, state_out,
                                                     cell_out)
                (top_traces, top_labels, top_probs,
                        top_states, top_cells) = get_top(soft_out, 10,
                                                         state_out, cell_out)
            else:
                (top_traces, top_labels, top_probs,
                    top_states, top_cells) = beam_search(func, x_val, x_mask,
                                                         top_labels, top_states,
                                                         top_cells, top_probs,
                                                         last_traces=top_traces)
        #select top
        top_traces = np.asarray(top_traces)[:, 0, :] * data[1].T
        neq += np.not_equal(top_traces, data[2][1:].T).sum()
        valid_length += data[1].sum()

        print neq / valid_length

    mis_rate = neq / valid_length
    print mis_rate

if __name__ == '__main__':
    args = parse_args()
    eval(**args.__dict__)
