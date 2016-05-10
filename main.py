import argparse
import time
import os
import sys
import logging
import lasagne
import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T

from blocks.algorithms import (GradientDescent, StepClipping, CompositeRule,
                               Momentum, Adam, RMSProp)
from blocks.bricks import Logistic, Tanh, Softmax, Rectifier
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.extensions.saveload import Load, Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise
from blocks.initialization import IsotropicGaussian, Constant, Orthogonal, Uniform
from blocks.monitoring import aggregation
from blocks.model import Model
from blocks.roles import WEIGHT
from bricks.generator import Generator, MultiLayerEncoder
from bricks.recurrent import GenLSTM, BidirectionalGraves
from blocks.bricks.recurrent import LSTM
from datasets.timit_hmm import TIMIT
from extensions import EarlyStopping
from initialization import NormalizedInitialization
from utils import (sequence_categorical_crossentropy,
                   sequence_binary_crossentropy,
                   sequence_misclass_rate,
                   sequence_binary_misrate,
                   construct_hmm_stream,
                   get_encoding)

from sampler import gen_sample
import pickle

floatX = theano.config.floatX
logger = logging.getLogger(__name__)
logger.setLevel('INFO')


def learning_algorithm(args):
    name = args.algorithm
    learning_rate = float(args.learning_rate)
    momentum = args.momentum
    clipping_threshold = args.clipping
    if name == 'adam':
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        adam = Adam(learning_rate=learning_rate)
        step_rule = CompositeRule([adam, clipping])
    elif name == 'rms_prop':
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        rms_prop = RMSProp(learning_rate=learning_rate)
        step_rule = CompositeRule([clipping, rms_prop])
    else:
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        sgd_momentum = Momentum(learning_rate=learning_rate, momentum=momentum)
        step_rule = CompositeRule([clipping, sgd_momentum])
    return step_rule


def parse_args():
    parser = argparse.ArgumentParser(description='TIMIT experiment')
    parser.add_argument('--experiment_path', type=str,
                        default='./',
                        help='Location for writing results')
    parser.add_argument('--input_dim', type=int,
                        default=301,
                        help='Input dimension')
    parser.add_argument('--gen_dim', type=int,
                        default=250,
                        help='States dimensions')
    parser.add_argument('--disc_dim', type=int,
                        default=250,
                        help='States dimensions')
    parser.add_argument('--label_dim', type=int,
                        default=181,
                        help='Label dimension')
    parser.add_argument('--epochs', type=int,
                        default=200,
                        help='Number of epochs')
    parser.add_argument('--seed', type=int,
                        default=123,
                        help='Random generator seed')
    parser.add_argument('--load_path',
                        default=argparse.SUPPRESS,
                        help='File with parameter to be loaded)')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_noise', type=float, default=0.,
                        help='Learning rate')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        help='Momentum for SGD')
    parser.add_argument('--clipping',
                        default=200,
                        type=float,
                        help='Gradient clipping norm')
    parser.add_argument('--beam_search', action='store_true',
                        default=False,
                        help='Perform beam search and report WER')
    parser.add_argument('--l2regularization', type=float,
                        default=argparse.SUPPRESS,
                        help='Apply L2 regularization')
    parser.add_argument('--algorithm', choices=['rms_prop', 'adam',
                                                'sgd_momentum'],
                        default='sgd_momentum',
                        help='Learning algorithm to use')
    parser.add_argument('--dropout', type=float,
                        default=0,
                        help='Use dropout in middle layers')
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
    parser.add_argument('--patience', type=int,
                        default=25,
                        help='How many epochs to do before early stopping.')
    parser.add_argument('--to_watch', type=str,
                        default='dev_fer',
                        help='Variable to watch for early stopping'
                             '(the smaller the better).')
    parser.add_argument('--initialization', choices=['glorot', 'uniform'],
                        default='glorot')
    return parser.parse_args()


def train(step_rule, input_dim, gen_dim, disc_dim, label_dim, epochs,
          seed, dropout, beam_search, experiment_path, window_features,
          pool_size, maximum_frames, initialization, weight_noise,
          to_watch, learning_rate, patience, **kwargs):
    print '.. TIMIT experiment'
    print '.. arguments:', ' '.join(sys.argv)
    t0 = time.time()

    if initialization == 'glorot':
        weights_init = NormalizedInitialization()
    elif initialization == 'uniform':
        weights_init = Uniform(width=.2)
    else:
        raise ValueError('No such initialization')

    rng = np.random.RandomState(seed)
    stream_args = dict(rng=rng, pool_size=pool_size,
                       maximum_frames=maximum_frames,
                       window_features=window_features)

    print '.. initializing iterators'
    train_dataset = TIMIT('train')
    train_stream = construct_hmm_stream(train_dataset, **stream_args)
    dev_dataset = TIMIT('dev')
    dev_stream = construct_hmm_stream(dev_dataset, **stream_args)
    test_dataset = TIMIT('test')
    test_stream = construct_hmm_stream(test_dataset, **stream_args)

    print '.. building model'
    x = T.tensor3('features')
    y = T.imatrix('labels')
    input_mask = T.matrix('features_mask')
    output_mask = T.matrix('labels_mask')
    y0_gen = T.imatrix('y0')

    #################
    #BUILD GENERATOR#
    #################
    gen_bidir1 = GenLSTM(name='bidir1',
                  dim=gen_dim, activation=Tanh())

    gen_bidir2 = GenLSTM(name='bidir2',
                  dim=gen_dim, activation=Tanh())
    generator = Generator(weights_init=weights_init,
                          biases_init=Constant(.0),
                          networks=[gen_bidir1, gen_bidir2],
                          dims=[input_dim * window_features,
                                gen_dim,
                                gen_dim,
                                label_dim])
    generator.initialize()

    #################
    #TEACHER FORCING#
    #################
    encoding_y = get_encoding(y, label_dim)
    y_hat_o, y_states = generator.apply(x, input_mask, targets=encoding_y[:-1])
    shape = y_hat_o.shape
    y_hat = Softmax().apply(y_hat_o.reshape((-1, shape[-1]))).reshape(shape)
    disc_i_tf = T.concatenate([y_hat, y_states], axis=2)

    ###############
    #SAMPLING MODE#
    ###############
    (y_hat_gen_o, gen_states_o,
            gen_cells_o, gen_soft_o) = generator.apply(x, input_mask,
                                                       targets=y0_gen,
                                                       tf=False)
    disc_i_gen = T.concatenate([gen_soft_o,
                                gen_states_o[:, :, -gen_dim:]],
                                axis=2)

    ############################
    #LAOD PRETRAINED TF NETWORK#
    ############################
    if 'load_path' not in kwargs:
        raise KeyError('The pretrained TF network should be provided!')

    print '.. load parameters of the TF network'
    model = Model(y_hat)
    parameters = model.get_parameter_dict()
    params = np.load(kwargs['load_path'])
    params_names = params.keys()
    for par in parameters.keys():
        split_name = par[1:].split('/')[1:]
        if len(split_name) > 1:
            split_name = '-'.join(split_name)
        else:
            split_name = split_name[0]
        dash_name = 'multilayerencoder_alex-' + \
                split_name
        if parameters[par].get_value().shape != \
                params[dash_name].shape:
                    raise ValueError('dimension wrong!')
        parameters[par].set_value(params[dash_name])

    #disfunc = theano.function([x, input_mask, y0_gen, init_gen_states, init_gen_cells], [y_hat_gen_o, gen_states_o])
    #data = train_stream.get_epoch_iterator()
    #kk = next(data)
    #y0_gen = np.zeros((kk[0].shape[1],181)).astype('int32')
    #init_cells_val = np.ones((kk[0].shape[1],500)).astype('int32')
    #init_states_val = np.ones((kk[0].shape[1],500)).astype('int32')
    #pp = disfunc(kk[0], kk[1], y0_gen, init_states_val, init_cells_val)
    #import pdb; pdb.set_trace()

    #####################
    #BUILD DISCRIMINATOR#
    #####################
    disc_bidir1 = BidirectionalGraves(name='disc_bidir1',
                                      prototype=LSTM(
                                      dim=disc_dim, activation=Tanh()))

    disc_bidir2 = BidirectionalGraves(name='disc_bidir2',
                                      prototype=LSTM(
                                      dim=disc_dim, activation=Tanh()))

    discriminator = MultiLayerEncoder(weights_init=weights_init,
                                      biases_init=Constant(.0),
                                      networks=[disc_bidir1, disc_bidir2],
                                      dims=[gen_dim + label_dim,
                                            disc_dim,
                                            disc_dim,
                                            1])

    discriminator.initialize()
    disc_i = T.concatenate([disc_i_tf, disc_i_gen], axis=1).astype('float32')
    input_mask_ = T.repeat(input_mask, 2, axis=1)

    raw_disc_o = discriminator.apply(disc_i, input_mask_)
    shape = raw_disc_o.shape
    disc_o = Logistic().apply(raw_disc_o.reshape((-1,
        shape[-1]))).reshape(shape)

    batch_size = input_mask.shape[1]
    disc_o_tf = disc_o[:, :batch_size]
    disc_o_gen = disc_o[:, batch_size:]

    ####################
    #DISCRIMINATOR COST#
    ####################
    disc_cost = (sequence_binary_crossentropy(disc_o_tf,
                                              T.ones(disc_o_tf.shape[:2]),
                                              input_mask) +
                 sequence_binary_crossentropy(disc_o_gen,
                                              T.zeros(disc_o_gen.shape[:2]),
                                              input_mask))

    disc_cost_train = aggregation.mean(disc_cost,
                                       batch_size).copy("disc_sequence_cost")

    disc_tf_misrate = sequence_binary_misrate(disc_o_tf > .5,
                                              T.ones(disc_o_tf.shape[:2]),
                                              input_mask)
    disc_gen_misrate = sequence_binary_misrate(disc_o_gen > .5,
                                               T.zeros(disc_o_tf.shape[:2]),
                                               input_mask)
    disc_misrate = (disc_tf_misrate + disc_gen_misrate) / 2.

    ################
    #GENERATOR COST#
    ################

    #sampling cost
    gen_cost = sequence_binary_crossentropy(disc_o_gen,
                                            T.ones(disc_o_gen.shape[:2]),
                                            input_mask)
    gen_cost_train = aggregation.mean(gen_cost,
                                      batch_size).copy("gen_sequence_cost")


    gen_sample_misrate = sequence_misclass_rate(y_hat_gen_o,
                                                y[1:],
                                                input_mask)

    #teacher forcing cost
    gen_tf_cost = sequence_categorical_crossentropy(y_hat,
                                                    y[1:],
                                                    input_mask)

    gen_tf_cost_train = aggregation.mean(gen_tf_cost,
                                         batch_size).copy("gen_tf_sequence_cost")

    gen_tf_misrate = sequence_misclass_rate(y_hat,
                                            y[1:],
                                            input_mask)


    ################
    #WN REGULARIZER#
    ################
    if weight_noise > 0:
        weights = VariableFilter(roles=[WEIGHT])(cg_train.variables)
        cg_train = apply_noise(cg_train, weights, weight_noise)
        cost_train = cg_train.outputs[0].copy('cost_train')

    #######################
    #FILTER OUT PARAMETERS#
    #######################
    disc_train_model = Model(disc_cost_train)
    disc_params = []
    for param_name, param in disc_train_model.get_parameter_dict().iteritems():
        if param_name[:4] == '/mul':
            disc_params.append(param)

    gen_train_model = Model([gen_cost_train, gen_tf_cost_train])
    gen_params = []
    for param_name, param in gen_train_model.get_parameter_dict().iteritems():
        if param_name[:4] == '/gen':
            gen_params.append(param)

    ################
    #ADAM OPTIMIZER#
    ################
    print '.. compile discriminator'
    disc_updates = lasagne.updates.adam(disc_cost_train.copy('disc_cost'),
                                        disc_params,
                                        learning_rate,
                                        beta1=.9,
                                        beta2=.999)

    disc_func = theano.function(inputs=[x, input_mask, y, y0_gen],
                                outputs=[disc_cost_train, disc_misrate],
                                updates=disc_updates)


    print '.. compile generator with TF mode'
    gen_tf_updates = lasagne.updates.adam(gen_tf_cost_train.copy('gen_tf_cost'),
                                          gen_params,
                                          learning_rate,
                                          beta1=.9,
                                          beta2=.999)

    gen_tf_func = theano.function(inputs=[x, input_mask, y],
                                  outputs=[gen_tf_cost_train, gen_tf_misrate],
                                  updates=gen_tf_updates)

    print '.. compile generator with sampling mode'
    gen_sample_updates = lasagne.updates.adam(gen_cost_train.copy('gen_sample_cost'),
                                              gen_params,
                                              learning_rate,
                                              beta1=.9,
                                              beta2=.999)

    gen_sample_func = theano.function(inputs=[x, input_mask, y, y0_gen],
                                      outputs=[gen_cost_train,
                                               gen_sample_misrate],
                                      updates=gen_sample_updates)

    #for name, param in parameters.iteritems():
    #    observed_vars.append(param.norm(2).copy(name + "_norm"))
    t1 = time.time()
    print "Building time: %f" % (t1 - t0)

    ####################
    #BUILD ACTUAL MODEL#
    ####################
    loop_log = OrderedDict()
    print '.. training the model'
    for ep in xrange(epochs):
        loop_log[(ep, 'disc_cost')] = 0.
        loop_log[(ep, 'disc_misrate')] = 0.
        loop_log[(ep, 'gen_tf_cost')] = 0.
        loop_log[(ep, 'gen_tf_misrate')] = 0.
        loop_log[(ep, 'gen_sample_cost')] = 0.
        loop_log[(ep, 'gen_sample_misrate')] = 0.

        print '..Epoch {}'.format(ep)

        for idx in xrange(3):
            num_batches = 0
            for data in train_stream.get_epoch_iterator():
                num_batches += 1
                if idx == 0:
                    batch_size = data[1].shape[1]
                    y0_gen_val = np.zeros((batch_size,
                                           label_dim)).astype('int32')
                    y0_gen_val[:, 0] = 1

                    cost_val, misrate_val = disc_func(data[0], data[1],
                                                      data[2], y0_gen_val)
                    loop_log[(ep, 'disc_cost')] += cost_val
                    loop_log[(ep, 'disc_misrate')] += misrate_val

                elif idx == 1:
                    cost_val, misrate_val = gen_tf_func(data[0], data[1],
                                                        data[2])
                    loop_log[(ep, 'gen_tf_cost')] += cost_val
                    loop_log[(ep, 'gen_tf_misrate')] += misrate_val
                else:
                    batch_size = data[1].shape[1]
                    y0_gen_val = np.zeros((batch_size,
                                           label_dim)).astype('int32')
                    y0_gen_val[:, 0] = 1

                    cost_val, misrate_val = gen_sample_func(data[0], data[1],
                                                            data[2], y0_gen_val)
                    loop_log[(ep, 'gen_sample_cost')] += cost_val
                    loop_log[(ep, 'gen_sample_misrate')] += cost_val

            if idx == 0:
                loop_log[(ep, 'disc_cost')] = \
                        loop_log[(ep, 'disc_cost')] / num_batches
                loop_log[(ep, 'disc_misrate')] = \
                        loop_log[(ep, 'disc_misrate')] / num_batches
                print 'cost of discriminator {0},', \
                       'misclassifcation rate {1}'.format(loop_log[(ep, 'disc_cost')],
                                                         loop_log[(ep, 'disc_misrate')])
            elif idx == 1:
                loop_log[(ep, 'gen_tf_cost')] = \
                        loop_log[(ep, 'gen_tf_cost')] / num_batches
                loop_log[(ep, 'gen_tf_misrate')] = \
                        loop_log[(ep, 'gen_tf_misrate')]  / num_batches
                print 'cost of generator with tf {0},', \
                       'misclassification rate {1}'.format(loop_log[(ep, 'gen_tf_cost')],
                                                          loop_log[(ep, 'gen_tf_misrate')])
            else:
                loop_log[(ep, 'gen_sample_cost')] = \
                        loop_log[(ep, 'gen_sample_cost')] / num_batches
                loop_log[(ep, 'gen_sample_misrate')] = \
                        loop_log[(ep, 'gen_sample_misrate')] / num_batches
                print 'cost of generator with sampling {0},', \
                       'misclassification rate {1}'.format(loop_log[(ep, 'gen_sample_cost')],
                                                          loop_log[(ep, 'gen_sample_misrate')])
    print '.. save parameters'
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    saved_path = os.path.join(experiment_path, 'model.zip')
    with open(saved_path, 'wb') as outfile:
        pickle.dump(loop_log, outfile, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = parse_args()
    step_rule = learning_algorithm(args)
    train(step_rule, **args.__dict__)
