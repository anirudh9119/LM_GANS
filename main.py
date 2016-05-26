import argparse
import time
import os
import sys
import logging
import lasagne
from lasagne.updates import adam, apply_momentum, sgd, total_norm_constraint
import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T

from blocks.bricks import Logistic, Tanh, Softmax, Rectifier
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.extensions.saveload import Load, Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
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

from blocks.roles import WEIGHT, INPUT
from sampler import gen_sample
import pickle

floatX = theano.config.floatX
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

def parse_args():
    parser = argparse.ArgumentParser(description='TIMIT experiment')
    parser.add_argument('--experiment_path', type=str,
                        default='./',
                        help='Location for writing results')
    parser.add_argument('--input_dim', type=int,
                        default=120,
                        help='Input dimension')
    parser.add_argument('--gen_dim', type=int,
                        default=250,
                        help='States dimensions')
    parser.add_argument('--disc_dim', type=int,
                        default=1000,
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
    parser.add_argument('--best_epoch', type=int,
                        default=0,
                        help='epoch to load parameters')
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


def train(input_dim, gen_dim, disc_dim, label_dim, epochs,
          seed, dropout, beam_search, experiment_path, window_features,
          pool_size, maximum_frames, initialization,
          to_watch, learning_rate, patience, clipping, **kwargs):
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
    gen_unidir1 = GenLSTM(name='unidir1',
                  dim=gen_dim, activation=Tanh())

    gen_unidir2 = GenLSTM(name='unidir2',
                  dim=gen_dim, activation=Tanh())
    generator = Generator(weights_init=weights_init,
                          biases_init=Constant(.0),
                          networks=[gen_unidir1, gen_unidir2],
                          dims=[(input_dim + label_dim) * window_features,
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

    ###############
    #SAMPLING MODE#
    ###############
    (y_hat_gen_o, gen_states_o,
            gen_cells_o, gen_soft_o, gen_scan_up) = generator.apply(x, input_mask,
                                                                    targets=y0_gen,
                                                                    tf=False)
    #####################
    #BUILD DISCRIMINATOR#
    #####################
    disc_bidir1 = BidirectionalGraves(name='disc_bidir1',
                                      prototype=LSTM(
                                      dim=disc_dim, activation=Tanh()))

    #disc_bidir2 = BidirectionalGraves(name='disc_bidir2',
    #                                  prototype=LSTM(
    #                                  dim=disc_dim, activation=Tanh()))

    #disc_bidir3 = BidirectionalGraves(name='disc_bidir3',
    #                                  prototype=LSTM(
    #                                  dim=disc_dim, activation=Tanh()))

    discriminator = MultiLayerEncoder(weights_init=weights_init,
                                       biases_init=Constant(.0),
                                       networks=[disc_bidir1],
                                       dims=[gen_dim * 2 + label_dim + input_dim,
                                             disc_dim,
                                             1])

    discriminator.initialize()

    disc_tf = T.tensor3('disc_tf')
    disc_gen = T.tensor3('disc_gen')

    disc_i = T.concatenate([disc_tf, disc_gen], axis=1).astype('float32')

    input_mask_ = T.repeat(input_mask, 2, axis=1)

    raw_disc_o = discriminator.apply(disc_i, input_mask_)
    shape = raw_disc_o.shape
    disc_o = Logistic().apply(raw_disc_o.reshape((-1,
        shape[-1]))).reshape(shape)

    batch_size = input_mask.shape[1]
    disc_o_tf = disc_o[:, :batch_size]
    disc_o_gen = disc_o[:, batch_size:]

    ############################
    #LAOD TRAINED NETWORK#
    ############################
    if 'load_path' in kwargs:
        print '.. load parameters of the trained network'
        model = Model(disc_o)
        parameters = model.get_parameter_dict()
        loaded = open(kwargs['load_path'], 'r')
        for _ in xrange(kwargs['best_epoch']):
            params = pickle.load(loaded)
        params_names = params.keys()
        for par in parameters.keys():
            if par not in params_names:
                raise KeyError('not exist!')
            if parameters[par].get_value().shape != \
                    params[par].shape:
                        raise ValueError('dimension wrong!')
            parameters[par].set_value(params[par])


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

    ################
    #GENERATOR COST#
    ################

    #sampling cost
    gen_cost = (sequence_binary_crossentropy(disc_o_gen,
                                             T.ones(disc_o_gen.shape[:2]),
                                             input_mask) +
                sequence_binary_crossentropy(disc_o_tf,
                                             T.zeros(disc_o_tf.shape[:2]),
                                             input_mask))
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

    ##########################
    #Dropout on Discriminator#
    ##########################
    mlp_layers = discriminator.mlp.linear_transformations
    if dropout:
        disc_cg_train = ComputationGraph(disc_cost_train)
        variables_mlp = VariableFilter(roles=[INPUT],
                                       bricks=mlp_layers)(disc_cg_train.variables)
        cg_dropout = apply_dropout(disc_cg_train,
                                   variables_mlp, 0.1)
        disc_cost_train = cg_dropout.outputs[0].copy('disc_cost_train')

    ################
    #ADAM OPTIMIZER#
    ################
    print '.. compile discriminator'
   # all_disc_grads = T.grad(disc_cost_train.copy('disc_cost'), disc_params)
   # scaled_disc_grads, disc_norm = total_norm_constraint(all_disc_grads,
   #                                                      clipping,
   #                                                      return_norm=True)

    train_disc_outputs = [disc_cost_train, disc_tf_misrate, disc_gen_misrate]
    disc_cost_ = disc_cost_train.copy('disc_cost')

    disc_updates = adam(disc_cost_,
                        disc_params,
                        learning_rate)

    disc_func = theano.function(inputs=[disc_tf, disc_gen, input_mask],
                                outputs=train_disc_outputs,
                                updates=disc_updates)

    disc_eval = theano.function(inputs=[disc_tf, disc_gen, input_mask],
                                outputs=[disc_cost_train, disc_tf_misrate,
                                         disc_gen_misrate])

    print '.. compile generator with TF mode'

    train_tf_outputs = [gen_tf_cost_train, gen_tf_misrate,
                        y_hat_o, y_states]
    gen_tf_cost_ = gen_tf_cost_train.copy('gen_tf_cost')

    #all_gen_tf_grads = T.grad(gen_tf_cost_, gen_params)
    #scaled_gen_tf_grads = total_norm_constraint(all_gen_tf_grads,
    #                                            clipping)

    #train_tf_outputs.append(gen_tf_norm)

    gen_tf_updates = adam(gen_tf_cost_,
                          gen_params,
                          learning_rate)
    gen_tf_func = theano.function(inputs=[x, input_mask, y],
                                  outputs=train_tf_outputs,
                                  updates=gen_tf_updates)

    gen_tf_eval = theano.function(inputs=[x, input_mask, y],
                                  outputs=[gen_tf_cost_train,
                                           gen_tf_misrate,
                                           y_hat_o, y_states])


    print '.. compile generator with sampling mode'
    train_gen_outputs = [gen_cost_train, gen_sample_misrate,
                         gen_soft_o, gen_states_o]
    gen_sample_cost_ = gen_cost_train.copy('gen_sample_cost')

    #all_gen_sample_grads = T.grad(gen_sample_cost_, gen_params)
    #scaled_gen_sample_grads, gen_sample_norm = total_norm_constraint(all_gen_sample_grads,
    #                                                                 clipping,
    #                                                                 return_norm=True)

    #train_gen_outputs.append(gen_sample_norm)

    gen_sample_updates = adam(gen_sample_cost_,
                              gen_params,
                              learning_rate)
    gen_sample_updates.update(gen_scan_up.items())
    gen_sample_func = theano.function(inputs=[x, input_mask, y, y0_gen],
                                      outputs=train_gen_outputs,
                                      updates=gen_sample_updates)

    gen_sample_eval = theano.function(inputs=[x, input_mask, y, y0_gen],
                                      outputs=[gen_cost_train,
                                               gen_sample_misrate,
                                               gen_soft_o, gen_states_o],
                                      updates=gen_scan_up)

    t1 = time.time()
    print "Building time: %f" % (t1 - t0)

    ####################
    #BUILD ACTUAL MODEL#
    ####################
    loop_log = OrderedDict()
    print '.. training the model'

    print '.. save parameters and log to'
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    param_path = os.path.join(experiment_path, 'params.pkl')
    param_file = open(param_path, 'wb')
    log_path = os.path.join(experiment_path, 'log.zip')

    print '.. pretrain teacher forcing generator'
    for pre_ep in xrange(epochs):
        if 'load_path' in kwargs:
            break
        loop_log[(pre_ep, 'pretrain_tf_cost')] = 0.
        loop_log[(pre_ep, 'pretrain_tf_misrate')] = 0.
        num_batches = 0
        #all_norms = 0.
        for data in train_stream.get_epoch_iterator():
            num_batches += 1

            cost_val, misrate_val, _, _ = gen_tf_func(data[0], data[1],
                                                   data[2])
            #all_norms += norm
            loop_log[(pre_ep, 'pretrain_tf_cost')] += cost_val
            loop_log[(pre_ep, 'pretrain_tf_misrate')] += misrate_val

        loop_log[(pre_ep, 'pretrain_tf_cost')] = \
                loop_log[(pre_ep, 'pretrain_tf_cost')] / num_batches
        loop_log[(pre_ep, 'pretrain_tf_misrate')] = \
                loop_log[(pre_ep, 'pretrain_tf_misrate')]  / num_batches

        print '################## Pretrained Epoch {} ###################'.format(pre_ep)
        print 'Pretrained Teacher Forcing Cost {}'.format(loop_log[(pre_ep,
                                                                    'pretrain_tf_cost')])
        print 'Pretrained Teacher Forcing Misrate {}'.format(loop_log[(pre_ep,
                                                                      'pretrain_tf_misrate')])

        num_batches = 0
        cost_val_eval = 0.
        misrate_val_eval = 0.
        for data in dev_stream.get_epoch_iterator():
            num_batches += 1
            cost_val, misrate_val, _, _ = gen_tf_eval(data[0], data[1],
                                                      data[2])
            cost_val_eval += cost_val
            misrate_val_eval += misrate_val

        print 'Pretrained Teacher Forcing Cost on Dev Set {}'.format(cost_val_eval / num_batches)
        print 'Pretrained Teacher Forcing Misrate on Dev Set {}'.format(misrate_val_eval / num_batches)
        #print 'gradient norm {}'.format(all_norms / num_batches)
        print ("\n")

        if misrate_val_eval / num_batches < .25:
            break

    for ep in xrange(epochs):
        loop_log[(ep, 'disc_cost')] = 0.
        loop_log[(ep, 'disc_tf_misrate')] = 0.
        loop_log[(ep, 'disc_gen_misrate')] = 0.
        loop_log[(ep, 'gen_tf_cost')] = 0.
        loop_log[(ep, 'gen_tf_misrate')] = 0.
        loop_log[(ep, 'gen_sample_cost')] = 0.
        loop_log[(ep, 'gen_sample_misrate')] = 0.

        loop_log[(ep, 'disc_cost_eval')] = 0.
        loop_log[(ep, 'disc_tf_misrate_eval')] = 0.
        loop_log[(ep, 'disc_gen_misrate_eval')] = 0.
        loop_log[(ep, 'gen_tf_cost_eval')] = 0.
        loop_log[(ep, 'gen_tf_misrate_eval')] = 0.
        loop_log[(ep, 'gen_sample_cost_eval')] = 0.
        loop_log[(ep, 'gen_sample_misrate_eval')] = 0.
        loop_log[(-1, 'disc_tf_misrate_eval')] = 1
        loop_log[(-1, 'disc_gen_misrate_eval')] = 1
        loop_log['iterations'] = 0

        print '################## Epoch {} ###################'.format(ep)

        num_batches = 0
        t0 = time.time()

        for data in train_stream.get_epoch_iterator():
            num_batches += 1
            batch_size = data[1].shape[1]
            y0_gen_val = np.zeros((batch_size,
                                   label_dim)).astype('int32')
            y0_gen_val[:, 0] = 1

            #if loop_log[(ep-1, 'disc_tf_misrate_eval')] < 0.05 and \
            #    loop_log[(ep-1, 'disc_gen_misrate_eval')] < 0.15:
            #        pass
            #else:
            _, _, gen_tf_sample, gen_tf_states = gen_tf_eval(data[0],
                                                             data[1],
                                                             data[2])
            (cost_val,
             misrate_val,
             gen_free_sample,
             gen_free_states) = gen_sample_func(data[0], data[1],
                                                data[2], y0_gen_val)

            #gen_sample_all_norms += norm
            loop_log[(ep, 'gen_sample_cost')] += cost_val
            loop_log[(ep, 'gen_sample_misrate')] += misrate_val


            gen_tf = np.concatenate(gen_tf_sample, gen_tf_states,
                                    data[0], axis=2)
            gen_free = np.concatenate(gen_free_sample, gen_free_states,
                                      data[0], axis=2)
            cost_val, misrate_tf_val, misrate_gen_val = disc_func(gen_tf,
                                                                  gen_free,
                                                                  data[2])

           # #disc_all_norms += norm
            loop_log[(ep, 'disc_cost')] += cost_val
            loop_log[(ep, 'disc_tf_misrate')] += misrate_tf_val
            loop_log[(ep, 'disc_gen_misrate')] += misrate_gen_val

            #gen_tf_all_norms += norm
            cost_val, misrate_val, _, _ = gen_tf_func(data[0], data[1],
                                                      data[2])

            loop_log[(ep, 'gen_tf_cost')] += cost_val
            loop_log[(ep, 'gen_tf_misrate')] += misrate_val

        t1 = time.time()

        loop_log['iterations'] += num_batches

        print '..performance on training set'
        loop_log[(ep, 'gen_tf_cost')] = \
                loop_log[(ep, 'gen_tf_cost')] / num_batches
        loop_log[(ep, 'gen_tf_misrate')] = \
                loop_log[(ep, 'gen_tf_misrate')]  / num_batches
        print 'Teacher Forcing Cost {}'.format(loop_log[(ep, 'gen_tf_cost')])
        print 'Teacher Forcing Misrate {}'.format(loop_log[(ep, 'gen_tf_misrate')])

        loop_log[(ep, 'disc_cost')] = \
                loop_log[(ep, 'disc_cost')] / num_batches
        loop_log[(ep, 'disc_tf_misrate')] = \
                loop_log[(ep, 'disc_tf_misrate')] / num_batches
        loop_log[(ep, 'disc_gen_misrate')] = \
                loop_log[(ep, 'disc_gen_misrate')] / num_batches
        print 'Discriminator Cost {}'.format(loop_log[(ep, 'disc_cost')])
        print 'Discriminator TF Misrate {}'.format(loop_log[(ep, 'disc_tf_misrate')])
        print 'Discriminator Gen Misrate {}'.format(loop_log[(ep, 'disc_gen_misrate')])

        loop_log[(ep, 'gen_sample_cost')] = \
                loop_log[(ep, 'gen_sample_cost')] / num_batches
        loop_log[(ep, 'gen_sample_misrate')] = \
                loop_log[(ep, 'gen_sample_misrate')] / num_batches
        print 'Generator Cost {}'.format(loop_log[(ep, 'gen_sample_cost')])
        print 'Generator Misrate {}'.format(loop_log[(ep, 'gen_sample_misrate')])

       # print 'gradient norm of discriminator {}'.format(disc_all_norms / num_batches)
       # print 'gradient norm of generator with sampling mode {}'.format(gen_sample_all_norms / num_batches)
       # print 'gradient norm of generator with tf mode {}'.format(gen_tf_all_norms / num_batches)

        print 'training time {}s'.format(t1 - t0)
        print ("\n")

        print '.. evaluate on dev set'
        num_batches = 0
        for data in dev_stream.get_epoch_iterator():
            num_batches += 1
            batch_size = data[1].shape[1]

            y0_gen_val = np.zeros((batch_size,
                                   label_dim)).astype('int32')
            y0_gen_val[:, 0] = 1

            (cost_val,
             misrate_val,
             gen_free_sample,
             gen_free_states) = gen_sample_eval(data[0], data[1],
                                                data[2], y0_gen_val)
            loop_log[(ep, 'gen_sample_cost_eval')] += cost_val
            loop_log[(ep, 'gen_sample_misrate_eval')] += misrate_val

            (cost_val,
             misrate_val,
             gen_tf_sample,
             gen_tf_states) = gen_tf_eval(data[0], data[1],
                                          data[2])
            loop_log[(ep, 'gen_tf_cost_eval')] += cost_val
            loop_log[(ep, 'gen_tf_misrate_eval')] += misrate_val

            gen_tf = np.concatenate(gen_tf_sample, gen_tf_states,
                                    data[0], axis=2)
            gen_free = np.concatenate(gen_free_sample, gen_free_states,
                                      data[0], axis=2)

            cost_val, misrate_tf_val, misrate_gen_val = disc_eval(gen_tf,
                                                                  gen_free,
                                                                  data[2])
            loop_log[(ep, 'disc_cost_eval')] += cost_val
            loop_log[(ep, 'disc_tf_misrate_eval')] += misrate_tf_val
            loop_log[(ep, 'disc_gen_misrate_eval')] += misrate_gen_val




        loop_log[(ep, 'gen_tf_cost_eval')] = \
                loop_log[(ep, 'gen_tf_cost_eval')] / num_batches
        loop_log[(ep, 'gen_tf_misrate_eval')] = \
                loop_log[(ep, 'gen_tf_misrate_eval')]  / num_batches
        print 'Teacher Forcing Cost {}'.format(loop_log[(ep, 'gen_tf_cost_eval')])
        print 'Teacher Forcing Misrate {}'.format(loop_log[(ep, 'gen_tf_misrate_eval')])

        loop_log[(ep, 'disc_cost_eval')] = \
                loop_log[(ep, 'disc_cost_eval')] / num_batches
        loop_log[(ep, 'disc_tf_misrate_eval')] = \
                loop_log[(ep, 'disc_tf_misrate_eval')] / num_batches
        loop_log[(ep, 'disc_gen_misrate_eval')] = \
                loop_log[(ep, 'disc_gen_misrate_eval')] / num_batches
        print 'Discriminator Cost {}'.format(loop_log[(ep, 'disc_cost_eval')])
        print 'Discriminator TF Misrate {}'.format(loop_log[(ep, 'disc_tf_misrate_eval')])
        print 'Discriminator Gen Misrate {}'.format(loop_log[(ep, 'disc_gen_misrate_eval')])

        loop_log[(ep, 'gen_sample_cost_eval')] = \
                loop_log[(ep, 'gen_sample_cost_eval')] / num_batches
        loop_log[(ep, 'gen_sample_misrate_eval')] = \
                loop_log[(ep, 'gen_sample_misrate_eval')] / num_batches
        print 'Generator Cost {}'.format(loop_log[(ep, 'gen_sample_cost_eval')])
        print 'Generator Misrate {}'.format(loop_log[(ep, 'gen_sample_misrate_eval')])
        print ("\n")

        params_dict = OrderedDict()
        for param_name, param in disc_train_model.get_parameter_dict().iteritems():
            params_dict[param_name] = param.get_value()
        pickle.dump(params_dict, param_file)

        with open(log_path, 'wb') as fin:
            pickle.dump(loop_log, fin, protocol=pickle.HIGHEST_PROTOCOL)

    param_file.close()

if __name__ == '__main__':
    args = parse_args()
    train(**args.__dict__)
